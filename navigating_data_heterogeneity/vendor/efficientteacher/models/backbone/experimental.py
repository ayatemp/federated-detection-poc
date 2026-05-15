# EfficientTeacher by Alibaba Cloud 
"""
Experimental modules
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.backbone.common import Conv
from utils.downloads import attempt_download


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1., n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super().__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            #y.append(module(x, augment, profile, visualize)[0])
            y.append(module(x, augment, profile)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


class ScaledEnsemble(nn.ModuleList):
    # Ensemble that calibrates each member by scaling objectness before NMS.
    def __init__(self, scales):
        super().__init__()
        self.scales = [float(scale) for scale in scales]

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module, scale in zip(self, self.scales):
            try:
                pred = module(x, augment, profile, visualize)[0]
            except TypeError:
                pred = module(x, augment, profile)[0]
            if scale != 1.0 and pred.shape[-1] > 4:
                pred = pred.clone()
                pred[..., 4] = (pred[..., 4] * scale).clamp(0, 1)
            y.append(pred)
        return torch.cat(y, 1), None


class BrightnessRoutedEnsemble(nn.Module):
    # Route each image to a day or night model-level ensemble from a JSON spec.
    def __init__(self, spec, device=None, inplace=True, fuse=True):
        super().__init__()
        threshold = float(spec.get('threshold', 0.34))
        groups = spec.get('groups') or {}
        if not groups or 'day' not in groups or 'night' not in groups:
            raise ValueError('BrightnessRoutedEnsemble requires day and night groups')
        self.threshold = threshold
        self.mode = spec.get('mode', 'hard')
        self.primary_scale = float(spec.get('primary_scale', 1.0))
        self.leak = float(spec.get('leak', 0.0))
        self.night_gain = float(spec.get('night_gain', 1.0))
        self.night_gamma = float(spec.get('night_gamma', 1.0))
        group_scales = spec.get('group_scales') or {}
        self.group_names = ['day', 'night']
        self.groups = nn.ModuleDict(
            {
                name: load_scaled_ensemble(
                    [str(path) for path in groups[name]],
                    group_scales[name],
                    device=device,
                    inplace=inplace,
                    fuse=fuse,
                )
                if name in group_scales
                else attempt_load([str(path) for path in groups[name]], device=device, inplace=inplace, fuse=fuse)
                for name in self.group_names
            }
        )
        first = self.groups[self.group_names[0]]
        for key in 'names', 'nc':
            setattr(self, key, getattr(first, key))
        self.yaml = getattr(first, 'yaml', {})
        self.stride = first.stride

    @staticmethod
    def _prediction(output):
        while isinstance(output, (tuple, list)):
            output = output[0]
        return output

    @staticmethod
    def _pad_prediction(pred, max_boxes):
        if pred.shape[0] >= max_boxes:
            return pred
        pad = torch.zeros(
            (max_boxes - pred.shape[0], pred.shape[1]),
            device=pred.device,
            dtype=pred.dtype,
        )
        return torch.cat((pred, pad), dim=0)

    @staticmethod
    def _scale_objectness(pred, scale):
        if scale == 1.0 or pred.shape[-1] <= 4:
            return pred
        pred = pred.clone()
        pred[..., 4] = (pred[..., 4] * scale).clamp(0, 1)
        return pred

    def _input_for_group(self, x, name):
        if name != 'night' or (self.night_gain == 1.0 and self.night_gamma == 1.0):
            return x
        enhanced = x.clamp(0, 1)
        if self.night_gamma != 1.0:
            enhanced = enhanced.pow(self.night_gamma)
        if self.night_gain != 1.0:
            enhanced = (enhanced * self.night_gain).clamp(0, 1)
        return enhanced

    def forward(self, x, augment=False, profile=False, visualize=False):
        brightness = x.float().mean(dim=(1, 2, 3))
        route_night = brightness < self.threshold
        predictions = {}
        max_boxes = {}
        for name, model in self.groups.items():
            group_input = self._input_for_group(x, name)
            try:
                output = model(group_input, augment, profile, visualize)
            except TypeError:
                output = model(group_input, augment, profile)
            pred = self._prediction(output)
            predictions[name] = pred
            max_boxes[name] = pred.shape[1]

        selected = []
        for index in range(x.shape[0]):
            primary_name = 'night' if bool(route_night[index]) else 'day'
            if self.mode == 'soft_leak' and self.leak > 0:
                parts = []
                for name in self.group_names:
                    scale = self.primary_scale if name == primary_name else self.leak
                    pred = self._scale_objectness(predictions[name][index], scale)
                    parts.append(self._pad_prediction(pred, max_boxes[name]))
                pred = torch.cat(parts, dim=0)
            else:
                pred = self._pad_prediction(predictions[primary_name][index], max(max_boxes.values()))
            selected.append(pred)
        return torch.stack(selected, dim=0), None


def load_scaled_ensemble(paths, scales, device=None, inplace=True, fuse=True):
    if len(paths) != len(scales):
        raise ValueError(f'Scale count must match paths: {len(scales)} != {len(paths)}')
    model = ScaledEnsemble(scales)
    for path in paths:
        model.append(attempt_load(path, device=device, inplace=inplace, fuse=fuse))
    first = model[0]
    for key in 'names', 'nc':
        setattr(model, key, getattr(first, key))
    model.yaml = getattr(first, 'yaml', {})
    model.stride = first.stride
    assert all(first.nc == module.nc for module in model), f'Models have different class counts: {[module.nc for module in model]}'
    return model


def attempt_load(weights, device=None, inplace=True, fuse=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    from models.detector.yolo import Detect, Model

    if isinstance(weights, (str, Path)) and str(weights).endswith('.routed.json'):
        with Path(weights).open(encoding='utf-8') as f:
            spec = json.load(f)
        model = BrightnessRoutedEnsemble(spec, device=device, inplace=inplace, fuse=fuse)
        print(f'Brightness routed ensemble created with {weights}\n')
        return model
    if isinstance(weights, list) and len(weights) == 1 and str(weights[0]).endswith('.routed.json'):
        return attempt_load(weights[0], device=device, inplace=inplace, fuse=fuse)

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location='cpu', weights_only=False)  # load
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, 'stride'):
            ckpt.stride = torch.tensor([32.])
        if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())  # model in eval mode

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc':
        setattr(model, k, getattr(model[0], k))
    model.yaml = getattr(model[0], 'yaml', {})
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    return model
