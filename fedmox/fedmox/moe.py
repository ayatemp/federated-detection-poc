"""Sparse Mixture-of-Experts modules matching the FedMox paper design."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


def hard_top1_mask(
    logits: Tensor,
    expert_dim: int = 1,
    *,
    straight_through: bool = False,
    temperature: float = 1.0,
) -> Tensor:
    """Return a one-hot hard-max mask along the expert dimension.

    FedMox describes the router as a hard-max top-1 operator. A literal argmax
    has no useful gradient for learning the router, so the MoE modules default
    to a straight-through hard-max: the forward value is one-hot top-1, while
    the backward pass follows a softmax relaxation.
    """

    indices = logits.argmax(dim=expert_dim, keepdim=True)
    hard = torch.zeros_like(logits).scatter_(expert_dim, indices, 1.0)
    if not straight_through:
        return hard
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    soft = torch.softmax(logits / temperature, dim=expert_dim)
    return hard - soft.detach() + soft


class SpatialTop1MoE(nn.Module):
    """Spatial sparse MoE for feature maps x in [B, C, H, W].

    FedMox uses a 1x1 convolution router to produce K x H x W routing maps,
    then activates only the top-1 expert at each spatial location.
    """

    def __init__(
        self,
        in_channels: int,
        expert_factory: Callable[[], nn.Module],
        num_experts: int,
        *,
        straight_through_router: bool = True,
        router_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        self.router = nn.Conv2d(in_channels, num_experts, kernel_size=1)
        self.experts = nn.ModuleList(expert_factory() for _ in range(num_experts))
        self.num_experts = num_experts
        self.straight_through_router = straight_through_router
        self.router_temperature = router_temperature

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("SpatialTop1MoE expects [B, C, H, W] input")
        logits = self.router(x)
        mask = hard_top1_mask(
            logits,
            expert_dim=1,
            straight_through=self.straight_through_router,
            temperature=self.router_temperature,
        )
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        return (expert_outputs * mask.unsqueeze(2)).sum(dim=1)

    @torch.no_grad()
    def routing_map(self, x: Tensor) -> Tensor:
        """Return selected expert ids in [B, H, W] for inspection."""

        return self.router(x).argmax(dim=1)


class Top1MoE(nn.Module):
    """Fixed-dimensional top-1 MoE for ROI features or vectors.

    The supplementary material specifies a traditional router for the ROI head,
    where ROI features have a fixed dimension.
    """

    def __init__(
        self,
        in_features: int,
        expert_factory: Callable[[], nn.Module],
        num_experts: int,
        *,
        straight_through_router: bool = True,
        router_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        self.router = nn.Linear(in_features, num_experts)
        self.experts = nn.ModuleList(expert_factory() for _ in range(num_experts))
        self.num_experts = num_experts
        self.straight_through_router = straight_through_router
        self.router_temperature = router_temperature

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError("Top1MoE expects a tensor with feature dimension last")
        flat = x.reshape(-1, x.shape[-1])
        logits = self.router(flat)
        mask = hard_top1_mask(
            logits,
            expert_dim=1,
            straight_through=self.straight_through_router,
            temperature=self.router_temperature,
        )
        expert_outputs = torch.stack([expert(flat) for expert in self.experts], dim=1)
        mixed = (expert_outputs * mask.unsqueeze(-1)).sum(dim=1)
        return mixed.reshape(*x.shape[:-1], mixed.shape[-1])

    @torch.no_grad()
    def routing_ids(self, x: Tensor) -> Tensor:
        flat = x.reshape(-1, x.shape[-1])
        return self.router(flat).argmax(dim=1).reshape(*x.shape[:-1])
