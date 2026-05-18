import argparse
import math
import sys
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn.functional as F

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.backbone.common import *
from models.backbone.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import check_yaml, make_divisible, print_args, set_logging
from models.loss.loss import *
from models.head.yolov5_head import Detect

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)


class AnonymousAdapterExpert(nn.Module):
    """Small residual expert used to upcycle a dense YOLO backbone into MoE."""

    def __init__(self, channels, hidden_channels, kernel_size, activation):
        super().__init__()
        padding = kernel_size // 2
        self.reduce = nn.Conv2d(channels, hidden_channels, 1, 1, 0, bias=False)
        self.reduce_bn = nn.BatchNorm2d(hidden_channels)
        self.dw = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size,
            1,
            padding,
            groups=hidden_channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(hidden_channels)
        self.expand = nn.Conv2d(hidden_channels, channels, 1, 1, 0, bias=True)
        self.act, _ = get_activation(activation)

        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, x):
        y = self.act(self.reduce_bn(self.reduce(x)))
        y = self.act(self.dw_bn(self.dw(y)))
        return self.expand(y)


def _balanced_expert_choice_probs(dense_probs, top_k, capacity_factor):
    """Return sparse probabilities that give every expert direct training signal.

    The forward path is an Expert-Choice/BASE-style assignment: each expert
    claims its top scoring samples, then every sample also keeps its own top-k
    experts.  This avoids the average-everything failure mode while keeping the
    selected probabilities differentiable before the straight-through wrapper.
    """

    if dense_probs.ndim != 2 or dense_probs.shape[1] <= 1:
        return dense_probs

    num_tokens, num_experts = dense_probs.shape
    k = max(1, min(int(top_k), num_experts))
    capacity = max(1, int(math.ceil(float(capacity_factor) * num_tokens * k / num_experts)))
    take = min(capacity, num_tokens)

    with torch.no_grad():
        scores = dense_probs.detach()
        mask = torch.zeros_like(dense_probs)
        for expert_idx in range(num_experts):
            _, token_idx = scores[:, expert_idx].topk(take, dim=0, largest=True, sorted=False)
            mask[token_idx, expert_idx] = 1.0

        _, token_top_idx = scores.topk(k, dim=1, largest=True, sorted=False)
        token_top_mask = torch.zeros_like(dense_probs).scatter_(1, token_top_idx, 1.0)
        mask = torch.maximum(mask, token_top_mask)

    sparse_probs = dense_probs * mask
    return sparse_probs / sparse_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)


class DqaBackboneMoELevel(nn.Module):
    """Anonymous routed micro-expert adapter for one backbone feature level."""

    def __init__(self, channels, moe_cfg, activation):
        super().__init__()
        self.num_experts = max(1, int(getattr(moe_cfg, "num_experts", 8)))
        self.top_k = int(getattr(moe_cfg, "top_k", 2))
        self.temperature = float(getattr(moe_cfg, "temperature", 1.0))
        self.scale = float(getattr(moe_cfg, "scale", 0.25))
        self.shared_scale = float(getattr(moe_cfg, "shared_scale", 1.0))
        self.context_dim = max(0, int(getattr(moe_cfg, "context_dim", 0)))
        self.quality_dim = max(0, int(getattr(moe_cfg, "quality_dim", 4)))
        self.router_noise_std = max(0.0, float(getattr(moe_cfg, "router_noise_std", 0.0)))
        self.threshold_routing = bool(getattr(moe_cfg, "threshold_routing", False))
        self.threshold = float(getattr(moe_cfg, "threshold", 0.0))
        self.routing_mode = str(getattr(moe_cfg, "routing_mode", "soft")).lower()
        self.straight_through = bool(getattr(moe_cfg, "straight_through", False))
        self.expert_choice_capacity_factor = max(
            1e-6,
            float(getattr(moe_cfg, "expert_choice_capacity_factor", 1.0)),
        )
        self.dqa_prior_strength = float(getattr(moe_cfg, "dqa_prior_strength", 0.0))
        self.router_init_std = max(0.0, float(getattr(moe_cfg, "router_init_std", 0.0)))
        self.force_router_cache = False
        self.last_router_probs = []
        self.last_router_logits = []
        self.last_router_hard_probs = []
        self.act, _ = get_activation(activation)

        ratio = float(getattr(moe_cfg, "adapter_ratio", 0.125))
        min_channels = int(getattr(moe_cfg, "min_channels", 16))
        hidden = make_divisible(max(min_channels, int(channels * ratio)), 8)
        kernels = list(getattr(moe_cfg, "kernels", [3, 5, 7])) or [3]

        self.shared = AnonymousAdapterExpert(channels, hidden, 3, activation)
        self.experts = nn.ModuleList(
            AnonymousAdapterExpert(channels, hidden, int(kernels[i % len(kernels)]), activation)
            for i in range(self.num_experts)
        )
        router_hidden = make_divisible(max(16, channels // 8), 8)
        self.router_fc1 = nn.Linear(channels + self.context_dim + self.quality_dim, router_hidden)
        self.router_fc2 = nn.Linear(router_hidden, self.num_experts)
        if self.router_init_std > 0:
            nn.init.normal_(self.router_fc2.weight, mean=0.0, std=self.router_init_std)
            nn.init.zeros_(self.router_fc2.bias)

        context = list(getattr(moe_cfg, "context", []))
        if self.context_dim > 0:
            values = torch.zeros(self.context_dim, dtype=torch.float32)
            for idx, value in enumerate(context[: self.context_dim]):
                values[idx] = float(value)
        else:
            values = torch.zeros(0, dtype=torch.float32)
        self.register_buffer("context", values, persistent=False)

    def _context_batch(self, batch_size, device, dtype):
        if self.context_dim <= 0:
            return None
        return self.context.to(device=device, dtype=dtype).view(1, -1).expand(batch_size, -1)

    def _quality_batch(self, x):
        if self.quality_dim <= 0:
            return None
        flat = x.float().flatten(1)
        quality = torch.stack(
            (
                flat.mean(dim=1),
                flat.std(dim=1, unbiased=False),
                flat.square().mean(dim=1).sqrt(),
                flat.abs().amax(dim=1),
            ),
            dim=1,
        ).to(dtype=x.dtype)
        if self.quality_dim < quality.shape[1]:
            return quality[:, : self.quality_dim]
        if self.quality_dim > quality.shape[1]:
            pad = quality.new_zeros(quality.shape[0], self.quality_dim - quality.shape[1])
            quality = torch.cat([quality, pad], dim=1)
        return quality

    def _dqa_prior_logits(self, context, quality, logits):
        if self.dqa_prior_strength == 0:
            return None
        parts = []
        if context is not None and context.numel() > 0:
            parts.append(context.float())
        if quality is not None and quality.numel() > 0:
            parts.append(quality.float())
        if not parts:
            return None

        signature = torch.cat(parts, dim=1)
        dims = torch.arange(
            1,
            signature.shape[1] + 1,
            device=signature.device,
            dtype=signature.dtype,
        )
        centers = []
        for expert_idx in range(self.num_experts):
            phase = float(expert_idx + 1)
            centers.append(torch.sin(dims * phase * 1.61803398875) + torch.cos(dims * phase * 0.754877666))
        centers = torch.stack(centers, dim=0)
        signature = F.normalize(signature, dim=1, eps=1e-6)
        centers = F.normalize(centers, dim=1, eps=1e-6)
        return (signature @ centers.transpose(0, 1)).to(device=logits.device, dtype=logits.dtype)

    def _sparse_router_probs(self, dense_probs):
        if self.num_experts <= 1:
            return dense_probs

        if self.threshold_routing and self.threshold > 0 and self.routing_mode not in ("expert_choice", "balanced", "base"):
            mask = (dense_probs >= self.threshold).to(dense_probs.dtype)
            empty = mask.sum(dim=1, keepdim=True) <= 0
            if empty.any():
                top = dense_probs.argmax(dim=1, keepdim=True)
                mask = mask.scatter(1, top, 1.0)
            probs = dense_probs * mask
            return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)

        if self.routing_mode in ("expert_choice", "balanced", "base"):
            return _balanced_expert_choice_probs(
                dense_probs,
                self.top_k,
                self.expert_choice_capacity_factor,
            )

        if 0 < self.top_k < self.num_experts:
            _, indices = dense_probs.topk(self.top_k, dim=1)
            mask = torch.zeros_like(dense_probs).scatter_(1, indices, 1.0)
            probs = dense_probs * mask
            return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)

        return dense_probs

    def _router_probs(self, x):
        feature = F.adaptive_avg_pool2d(x, 1).flatten(1)
        context = self._context_batch(feature.shape[0], feature.device, feature.dtype)
        quality = self._quality_batch(x)
        pooled_parts = [feature]
        if context is not None:
            pooled_parts.append(context)
        if quality is not None:
            pooled_parts.append(quality)
        pooled = torch.cat(pooled_parts, dim=1)
        logits = self.router_fc2(self.act(self.router_fc1(pooled)))
        dqa_prior_logits = self._dqa_prior_logits(context, quality, logits)
        if dqa_prior_logits is not None:
            logits = logits + float(self.dqa_prior_strength) * dqa_prior_logits
        if self.training and self.router_noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.router_noise_std
        dense_probs = torch.softmax(logits / max(self.temperature, 1e-6), dim=1)
        hard_probs = self._sparse_router_probs(dense_probs)
        if self.straight_through or self.routing_mode in ("expert_choice", "balanced", "base"):
            probs = hard_probs.detach() + dense_probs - dense_probs.detach()
        else:
            probs = hard_probs
        return probs, dense_probs, logits, hard_probs

    def forward(self, x):
        x = x.clone()
        probs, dense_probs, logits, hard_probs = self._router_probs(x)
        force_router_cache = bool(getattr(self, "force_router_cache", False))
        cache_router = self.training and (torch.is_grad_enabled() or force_router_cache)
        if cache_router:
            if force_router_cache and not torch.is_grad_enabled():
                self.last_router_probs.append(dense_probs.detach())
                self.last_router_logits.append(logits.detach())
                self.last_router_hard_probs.append(hard_probs.detach())
            else:
                self.last_router_probs.append(dense_probs)
                self.last_router_logits.append(logits)
                self.last_router_hard_probs.append(hard_probs)

        routed = None
        for idx, expert in enumerate(self.experts):
            expert_out = expert(x) * probs[:, idx].view(-1, 1, 1, 1)
            routed = expert_out if routed is None else routed + expert_out

        shared = self.shared(x) * self.shared_scale
        return x + self.scale * (shared + routed)


class DqaBackboneAdapterMoE(nn.Module):
    """Applies anonymous MoE adapters to selected YOLOv5 backbone outputs."""

    LEVEL_NAMES = ("c3", "c4", "c5")

    def __init__(self, channels_by_level, moe_cfg, activation):
        super().__init__()
        enabled_levels = set(getattr(moe_cfg, "levels", ["c3", "c4", "c5"]))
        self.levels = nn.ModuleDict()
        for name in self.LEVEL_NAMES:
            if name in enabled_levels:
                self.levels[name] = DqaBackboneMoELevel(channels_by_level[name], moe_cfg, activation)

    @property
    def last_router_probs(self):
        probs = []
        for level in self.levels.values():
            probs.extend(getattr(level, "last_router_probs", []))
        return probs

    @last_router_probs.setter
    def last_router_probs(self, value):
        for level in self.levels.values():
            level.last_router_probs = []

    @property
    def last_router_logits(self):
        logits = []
        for level in self.levels.values():
            logits.extend(getattr(level, "last_router_logits", []))
        return logits

    @last_router_logits.setter
    def last_router_logits(self, value):
        for level in self.levels.values():
            level.last_router_logits = []

    def forward(self, features):
        outputs = []
        for name, feature in zip(self.LEVEL_NAMES, features):
            adapter = self.levels[name] if name in self.levels else None
            outputs.append(adapter(feature) if adapter is not None else feature)
        return tuple(outputs)

class YoloV5BackBone(nn.Module):
    def __init__(self, cfg):
        super(YoloV5BackBone, self).__init__()
        self.gd = cfg.Model.depth_multiple
        self.gw = cfg.Model.width_multiple

        self.channels_out = {
            'stage1': 64,
            'stage2_1': 128,
            'stage2_2': 128,
            'stage3_1': 256,
            'stage3_2': 256,
            'stage4_1': 512,
            'stage4_2': 512,
            'stage5': 1024,
            'spp': 1024,
            'csp1': 1024,
            'conv1': 1024
        }
        self.re_channels_out()

        if cfg.Model.Backbone.activation == 'SiLU': 
            CONV_ACT = 'silu'
            C_ACT = 'silu'
        elif cfg.Model.Backbone.activation == 'ReLU': 
            CONV_ACT = 'relu'
            C_ACT = 'relu'
        else:
            CONV_ACT = 'hard_swish'
            C_ACT = 'relu_hswish'
        self.stage1 = Conv(3, self.channels_out['stage1'], 6, 2, 2, 1, CONV_ACT)

        # for latest yolov5, you can change BottleneckCSP to C3
        self.stage2_1 = Conv(self.channels_out['stage1'], self.channels_out['stage2_1'], 3, 2, None, 1, CONV_ACT)
        self.stage2_2 = C3(self.channels_out['stage2_1'], self.channels_out['stage2_2'], self.get_depth(3), True, 1, 0.5, C_ACT)
        self.stage3_1 = Conv(self.channels_out['stage2_2'], self.channels_out['stage3_1'], 3, 2, None, 1, CONV_ACT)
        self.stage3_2 = C3(self.channels_out['stage3_1'], self.channels_out['stage3_2'], self.get_depth(6), True, 1, 0.5, C_ACT)
        self.stage4_1 = Conv(self.channels_out['stage3_2'], self.channels_out['stage4_1'], 3, 2, None, 1, CONV_ACT)
        self.stage4_2 = C3(self.channels_out['stage4_1'], self.channels_out['stage4_2'], self.get_depth(9), True, 1, 0.5, C_ACT)
        self.stage5_1 = Conv(self.channels_out['stage4_2'], self.channels_out['stage5'], 3, 2, None, 1, CONV_ACT)
        self.stage5_2 = C3(self.channels_out['stage5'], self.channels_out['csp1'], self.get_depth(3), True, 1, 0.5, C_ACT)
        self.sppf = SPPF(self.channels_out['csp1'], self.channels_out['spp'], 5, CONV_ACT)
        moe_cfg = getattr(cfg, "BackboneMoE", None)
        self.adapter_moe = None
        if moe_cfg is not None and bool(getattr(moe_cfg, "enabled", False)):
            self.adapter_moe = DqaBackboneAdapterMoE(
                {
                    "c3": self.channels_out['stage3_2'],
                    "c4": self.channels_out['stage4_2'],
                    "c5": self.channels_out['spp'],
                },
                moe_cfg,
                CONV_ACT,
            )
        # self.conv1 = Conv(self.channels_out['csp1'], self.channels_out['conv1'], 1, 1)
        self.out_shape = {'C3_size': self.channels_out['stage3_2'],
                          'C4_size': self.channels_out['stage4_2'],
                          'C5_size': self.channels_out['conv1']}
        # print("backbone output channel: C3 {}, C4 {}, C5 {}".format(self.channels_out['stage3_2'],
                                                                    # self.channels_out['stage4_2'],
                                                                    # self.channels_out['spp']))

    def forward(self, x):
        x1 = self.stage1(x) #0-P1/2
        x21 = self.stage2_1(x1) #1-P2/4
        x22 = self.stage2_2(x21)
        x31 = self.stage3_1(x22) #3-P3/8
        c3 = self.stage3_2(x31)
        x41 = self.stage4_1(c3) #5-P4/16
        c4 = self.stage4_2(x41)
        x51 = self.stage5_1(c4) #7-P5/32
        x5 = self.stage5_2(x51)

        sppf = self.sppf(x5)
        if self.adapter_moe is not None:
            c3, c4, sppf = self.adapter_moe((c3, c4, sppf))
        return c3, c4, sppf

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for k, v in self.channels_out.items():
            self.channels_out[k] = self.get_width(v)
