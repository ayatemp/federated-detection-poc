#!/usr/bin/env python3
"""Dynamic Quality-Aware Selective Backbone Adaptation aggregation.

DQA-SBA is a Phase-1 aggregation helper.  FedSTO Phase 1 updates only the
backbone; this module keeps that split, but replaces plain client FedAvg with a
server-anchored residual update whose client weights come from pseudo-label
quality statistics.
"""

from __future__ import annotations

import copy
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import dqa_cwa_aggregation as v1
import dqa_cwa_aggregation_v2 as v2
from dqa_cwa_aggregation import ClientClassStats, EPS


ROUND_RE = re.compile(r"phase(?P<phase>\d+)_round(?P<round>\d+)")


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return int(raw)


def round_from_path(path: Path) -> int | None:
    match = ROUND_RE.search(str(path))
    return int(match.group("round")) if match else None


def scheduled_phase1_residual(round_idx: int | None) -> float:
    start = _float_env("DQA084_PHASE1_RESIDUAL_START", 0.45)
    end = _float_env("DQA084_PHASE1_RESIDUAL_END", 0.18)
    total = max(_int_env("DQA084_PHASE1_ROUNDS", 12), 1)
    if round_idx is None or total <= 1:
        return start
    progress = min(max((round_idx - 1) / (total - 1), 0.0), 1.0)
    return start + (end - start) * progress


def _is_backbone_key(key: str) -> bool:
    return "backbone" in key.lower()


def _class_mass(stats: Sequence[ClientClassStats], num_classes: int) -> torch.Tensor:
    mass = torch.zeros(num_classes, dtype=torch.float64)
    for item in stats:
        mass += torch.tensor([max(float(x), 0.0) for x in item.counts], dtype=torch.float64)
    # log1p prevents car/traffic-light dominated scenes from completely
    # drowning out rare classes while still respecting actual pseudo support.
    mass = torch.log1p(mass)
    if float(mass.sum()) <= EPS:
        return torch.full((num_classes,), 1.0 / max(num_classes, 1), dtype=torch.float64)
    return mass / torch.clamp(mass.sum(), min=EPS)


def client_scalar_weights(
    stats: Sequence[ClientClassStats],
    state: dict[str, Any],
    config: v2.AggregationConfig,
) -> tuple[dict[str, Any], list[float], dict[str, Any]]:
    """Collapse class-wise DQA reliability into one scalar per client.

    The server anchor produced by v2 reliability is intentionally not
    renormalized away.  If the pseudo evidence is weak or unstable, client
    scalar weights sum to less than one and the residual update becomes smaller.
    """

    state, alpha, source_ids, active = v2.compute_reliability(stats, state, config)
    class_mass = _class_mass(stats, config.num_classes)
    if active.numel() == config.num_classes and bool(active.any()):
        active_mass = class_mass * active.double()
        if float(active_mass.sum()) > EPS:
            class_mass = active_mass / torch.clamp(active_mass.sum(), min=EPS)

    client_weights: list[float] = []
    client_ids = [f"client:{item.client_id}" for item in stats]
    for client_id in client_ids:
        if client_id not in source_ids:
            client_weights.append(0.0)
            continue
        row = alpha[source_ids.index(client_id)].double()
        client_weights.append(float(torch.sum(row * class_mass)))

    server_weight = 0.0
    if "server" in source_ids:
        server_weight = float(torch.sum(alpha[source_ids.index("server")].double() * class_mass))

    metadata = {
        "source_ids": source_ids,
        "class_mass": [float(x) for x in class_mass.tolist()],
        "active_classes": [bool(x) for x in active.tolist()],
        "client_scalar_weights": client_weights,
        "server_scalar_weight": server_weight,
        "client_weight_sum": float(sum(client_weights)),
    }
    return state, client_weights, metadata


def _phase1_reliability_config(config: v2.AggregationConfig) -> v2.AggregationConfig:
    phase1_config = copy.copy(config)
    phase1_config.server_anchor = _float_env("DQA084_PHASE1_SERVER_ANCHOR", min(float(config.server_anchor), 1.5))
    phase1_config.temperature = _float_env("DQA084_PHASE1_TEMPERATURE", float(config.temperature))
    phase1_config.stability_lambda = _float_env("DQA084_PHASE1_STABILITY_LAMBDA", float(config.stability_lambda))
    phase1_config.uniform_mix = _float_env("DQA084_PHASE1_UNIFORM_MIX", float(config.uniform_mix))
    return phase1_config


def _clip_update(
    proposed: torch.Tensor,
    server_value: torch.Tensor,
    *,
    max_relative_update: float,
) -> tuple[torch.Tensor, float]:
    if max_relative_update <= 0:
        return proposed, 1.0
    update = proposed.float() - server_value.float()
    update_norm = float(torch.linalg.vector_norm(update))
    server_norm = float(torch.linalg.vector_norm(server_value.float()))
    limit = max_relative_update * max(server_norm, 1.0)
    if update_norm <= limit or update_norm <= EPS:
        return proposed, 1.0
    scale = limit / max(update_norm, EPS)
    clipped = server_value.float() + update * scale
    return clipped.to(server_value.dtype), float(scale)


def _server_anchored_backbone_state(
    client_state_dicts: Sequence[Mapping[str, torch.Tensor]],
    server_state_dict: Mapping[str, torch.Tensor],
    *,
    client_weights: Sequence[float],
    residual_blend: float,
    localize_bn: bool,
    max_relative_update: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    result: dict[str, torch.Tensor] = {}
    clip_scales: list[float] = []
    touched = 0
    client_weights_t = [float(max(w, 0.0)) for w in client_weights]

    for key, server_value in server_state_dict.items():
        if (
            localize_bn
            and v1._is_batchnorm_key(key)
            or not torch.is_tensor(server_value)
            or not server_value.dtype.is_floating_point
            or not _is_backbone_key(key)
        ):
            result[key] = server_value
            continue

        server_float = server_value.float()
        residual = torch.zeros_like(server_float)
        for weight, client_state in zip(client_weights_t, client_state_dicts):
            if weight <= 0:
                continue
            residual = residual + weight * (client_state[key].float() - server_float)

        proposed = (server_float + residual_blend * residual).to(server_value.dtype)
        proposed, clip_scale = _clip_update(
            proposed,
            server_value,
            max_relative_update=max_relative_update,
        )
        result[key] = proposed
        touched += 1
        clip_scales.append(clip_scale)

    metadata = {
        "backbone_tensors_updated": touched,
        "clip_min_scale": min(clip_scales) if clip_scales else 1.0,
        "clip_mean_scale": sum(clip_scales) / max(len(clip_scales), 1),
        "max_relative_update": max_relative_update,
        "residual_blend": residual_blend,
    }
    return result, metadata


def aggregate_phase1_backbone_checkpoints(
    client_checkpoints: Sequence[Path],
    server_checkpoint: Path,
    output_checkpoint: Path,
    stats: Sequence[ClientClassStats],
    state_path: Path | None,
    config: v2.AggregationConfig,
    repo_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Aggregate Phase-1 client checkpoints with DQA-SBA."""

    if len(client_checkpoints) != len(stats):
        raise ValueError(
            f"client checkpoint count ({len(client_checkpoints)}) must match stats count ({len(stats)})"
        )

    state = v2.load_state(state_path)
    phase1_config = _phase1_reliability_config(config)
    state, client_weights, quality_meta = client_scalar_weights(stats, state, phase1_config)

    client_ckpts = [v1._load_checkpoint(path, repo_root) for path in client_checkpoints]
    server_ckpt = v1._load_checkpoint(server_checkpoint, repo_root)
    base = copy.deepcopy(server_ckpt)

    residual_blend = scheduled_phase1_residual(round_from_path(output_checkpoint))
    max_relative_update = _float_env("DQA084_PHASE1_MAX_RELATIVE_UPDATE", 0.04)

    client_state_dicts = [v1._model_state_dict(ckpt, "model") for ckpt in client_ckpts]
    server_state_dict = v1._model_state_dict(server_ckpt, "model")
    model_state, update_meta = _server_anchored_backbone_state(
        client_state_dicts,
        server_state_dict,
        client_weights=client_weights,
        residual_blend=residual_blend,
        localize_bn=config.localize_bn,
        max_relative_update=max_relative_update,
    )
    v1._replace_model_state(base, model_state, "model")

    if base.get("ema") is not None:
        ema_client_dicts = [v1._model_state_dict(ckpt, "ema") for ckpt in client_ckpts if ckpt.get("ema") is not None]
        server_ema = v1._model_state_dict(server_ckpt, "ema") if server_ckpt.get("ema") is not None else None
        if len(ema_client_dicts) == len(client_ckpts) and server_ema is not None:
            ema_state, _ = _server_anchored_backbone_state(
                ema_client_dicts,
                server_ema,
                client_weights=client_weights,
                residual_blend=residual_blend,
                localize_bn=config.localize_bn,
                max_relative_update=max_relative_update,
            )
            v1._replace_model_state(base, ema_state, "ema")

    base["epoch"] = -1
    base["optimizer"] = None
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, output_checkpoint)

    record = {
        "round": round_from_path(output_checkpoint),
        "phase1_reliability_config": {
            "server_anchor": phase1_config.server_anchor,
            "temperature": phase1_config.temperature,
            "stability_lambda": phase1_config.stability_lambda,
            "uniform_mix": phase1_config.uniform_mix,
        },
        **quality_meta,
        **update_meta,
    }
    state.setdefault("phase1_dqa_sba", []).append(record)
    state["last_phase1_dqa_sba"] = record
    state["last_sources"] = quality_meta["source_ids"]
    state["last_active_classes"] = quality_meta["active_classes"]
    v2.save_state(state_path, state)
    return output_checkpoint, state


def aggregate_phase1_fallback(
    client_checkpoints: Sequence[Path],
    server_checkpoint: Path,
    output_checkpoint: Path,
    *,
    repo_root: Path,
    localize_bn: bool = True,
) -> Path:
    """Stats-free conservative Phase-1 fallback."""

    client_ckpts = [v1._load_checkpoint(path, repo_root) for path in client_checkpoints]
    server_ckpt = v1._load_checkpoint(server_checkpoint, repo_root)
    base = copy.deepcopy(server_ckpt)

    client_state_dicts = [v1._model_state_dict(ckpt, "model") for ckpt in client_ckpts]
    server_state_dict = v1._model_state_dict(server_ckpt, "model")
    uniform = [1.0 / max(len(client_state_dicts), 1)] * len(client_state_dicts)
    model_state, _ = _server_anchored_backbone_state(
        client_state_dicts,
        server_state_dict,
        client_weights=uniform,
        residual_blend=scheduled_phase1_residual(round_from_path(output_checkpoint)),
        localize_bn=localize_bn,
        max_relative_update=_float_env("DQA084_PHASE1_MAX_RELATIVE_UPDATE", 0.04),
    )
    v1._replace_model_state(base, model_state, "model")

    if base.get("ema") is not None:
        ema_client_dicts = [v1._model_state_dict(ckpt, "ema") for ckpt in client_ckpts if ckpt.get("ema") is not None]
        server_ema = v1._model_state_dict(server_ckpt, "ema") if server_ckpt.get("ema") is not None else None
        if len(ema_client_dicts) == len(client_ckpts) and server_ema is not None:
            ema_state, _ = _server_anchored_backbone_state(
                ema_client_dicts,
                server_ema,
                client_weights=uniform,
                residual_blend=scheduled_phase1_residual(round_from_path(output_checkpoint)),
                localize_bn=localize_bn,
                max_relative_update=_float_env("DQA084_PHASE1_MAX_RELATIVE_UPDATE", 0.04),
            )
            v1._replace_model_state(base, ema_state, "ema")

    base["epoch"] = -1
    base["optimizer"] = None
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, output_checkpoint)
    return output_checkpoint
