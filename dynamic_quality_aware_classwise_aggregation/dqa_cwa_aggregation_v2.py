#!/usr/bin/env python3
"""DQA-CWA v2 aggregation.

Version 1 averaged client checkpoints first, then blended class-wise head rows.
The 03 run showed that this can erase too much of the labeled server model before
the server update recovers it.  Version 2 keeps the server checkpoint as the
anchor and applies quality-weighted client residuals on top of it.
"""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import dqa_cwa_aggregation as v1
from dqa_cwa_aggregation import (  # re-exported for the existing runner API
    ClientClassStats,
    EPS,
    load_round_stats,
    load_state,
    save_state,
)


@dataclass
class AggregationConfig(v1.AggregationConfig):
    min_server_alpha: float = 0.45
    residual_blend: float | None = None
    moe_expert_blend: float = 0.0
    moe_router_blend: float = 0.0
    bn_blend: float = 0.0

    def validate(self) -> None:
        super().validate()
        if not 0.0 <= self.min_server_alpha <= 1.0:
            raise ValueError(f"min_server_alpha must be in [0, 1], got {self.min_server_alpha}")
        if self.residual_blend is not None and not 0.0 <= self.residual_blend <= 1.0:
            raise ValueError(f"residual_blend must be in [0, 1], got {self.residual_blend}")
        if not 0.0 <= self.moe_expert_blend <= 1.0:
            raise ValueError(f"moe_expert_blend must be in [0, 1], got {self.moe_expert_blend}")
        if not 0.0 <= self.moe_router_blend <= 1.0:
            raise ValueError(f"moe_router_blend must be in [0, 1], got {self.moe_router_blend}")
        if not 0.0 <= self.bn_blend <= 1.0:
            raise ValueError(f"bn_blend must be in [0, 1], got {self.bn_blend}")


MOE_EXPERT_RE = re.compile(r"(^|\.)head\.expert_m\.\d+\.(\d+)\.(weight|bias)$")
MOE_ROUTER_RE = re.compile(r"(^|\.)head\.router\.\d+\.(weight|bias)$")


def _client_residual_blend(config: AggregationConfig) -> float:
    if config.residual_blend is not None:
        return float(config.residual_blend)
    return min(float(config.classwise_blend), 0.35)


def _normalise_expert_assignments(assignments: Sequence[Mapping[str, Any]] | None) -> list[dict[str, float]]:
    normalised: list[dict[str, float]] = []
    for item in assignments or []:
        try:
            target = int(item.get("target_expert", item.get("target", -1)))
            weight = float(item.get("weight", item.get("specialization_weight", 0.0)))
        except (TypeError, ValueError):
            target, weight = -1, 0.0
        if target < 0 or weight <= 0:
            normalised.append({"target_expert": -1.0, "weight": 0.0})
        else:
            normalised.append({"target_expert": float(target), "weight": weight})
    return normalised


def _client_quality_weight(stat: ClientClassStats) -> float:
    counts = torch.tensor(stat.counts, dtype=torch.float32)
    qualities = torch.tensor(stat.mean_quality_scores, dtype=torch.float32).clamp_min(0.0)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    quality = float((counts * qualities).sum() / torch.clamp(counts.sum(), min=EPS))
    # Count is useful as a confidence gate, but saturates quickly so large
    # pseudoGT clients do not erase smaller client/domain specialists.
    count_gate = min(1.0, total / 700.0)
    return max(0.0, quality * count_gate)


def _normalise_bn_weights(
    stats: Sequence[ClientClassStats],
    expert_assignments: Sequence[Mapping[str, Any]] | None,
) -> list[float]:
    assignment_weights = _normalise_expert_assignments(expert_assignments)
    weights: list[float] = []
    for idx, stat in enumerate(stats):
        quality_weight = _client_quality_weight(stat)
        assignment_weight = 1.0
        if idx < len(assignment_weights):
            assignment_weight = max(float(assignment_weights[idx].get("weight", 0.0)), 0.0)
        weights.append(quality_weight * assignment_weight)

    total = sum(weights)
    if total <= EPS:
        weights = [_client_quality_weight(stat) for stat in stats]
        total = sum(weights)
    if total <= EPS:
        return []
    return [weight / total for weight in weights]


def apply_dynamic_batchnorm_residuals(
    anchored: dict[str, torch.Tensor],
    client_state_dicts: Sequence[Mapping[str, torch.Tensor]],
    server_state_dict: Mapping[str, torch.Tensor],
    stats: Sequence[ClientClassStats],
    expert_assignments: Sequence[Mapping[str, Any]] | None,
    config: AggregationConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Blend client/domain BN statistics into the server anchor with DQA weights.

    FedBN says BN statistics carry feature-shift information and should not be
    blindly averaged.  Here we keep the server as the default detector and only
    inject a small DQA-gated residual from the clients whose pseudoGT evidence
    is stable enough to specialize router/expert state for the round.
    """

    if config.bn_blend <= 0:
        return anchored, {}
    weights = _normalise_bn_weights(stats, expert_assignments)
    if not weights or len(weights) != len(client_state_dicts):
        return anchored, {"skipped": "no_valid_bn_weights"}

    result = dict(anchored)
    updated_keys = 0
    for key, server_value in server_state_dict.items():
        if not v1._is_batchnorm_key(key):
            continue
        if not torch.is_tensor(server_value) or not server_value.dtype.is_floating_point:
            continue
        server_float = server_value.float()
        client_average = torch.zeros_like(server_float)
        for state, weight in zip(client_state_dicts, weights):
            client_average = client_average + state[key].float() * float(weight)
        blended = (1.0 - float(config.bn_blend)) * server_float + float(config.bn_blend) * client_average
        result[key] = blended.to(server_value.dtype)
        updated_keys += 1

    diagnostics = {
        "updated_keys": updated_keys,
        "blend": float(config.bn_blend),
        "weights": weights,
    }
    return result, diagnostics


def apply_dynamic_moe_expert_residuals(
    anchored: dict[str, torch.Tensor],
    client_state_dicts: Sequence[Mapping[str, torch.Tensor]],
    server_state_dict: Mapping[str, torch.Tensor],
    expert_assignments: Sequence[Mapping[str, Any]] | None,
    config: AggregationConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Aggregate latent-MoE expert/router residuals with DQA-gated expert ownership.

    The shared detector can remain server-anchored while each explicit MoE expert
    receives residuals only from clients routed to that expert. This is the
    FedMoX-like part: specialists are learned locally, then mixed according to
    the DQA quality of the pseudoGT that produced them.
    """

    if config.moe_expert_blend <= 0 and config.moe_router_blend <= 0:
        return anchored, {}
    assignments = _normalise_expert_assignments(expert_assignments)
    if len(assignments) != len(client_state_dicts):
        return anchored, {"skipped": "assignment_count_mismatch"}

    result = dict(anchored)
    expert_counts: dict[str, int] = {}
    expert_weight_sums: dict[str, float] = {}
    router_weight_sum = 0.0

    for key, server_value in server_state_dict.items():
        if not torch.is_tensor(server_value) or not server_value.dtype.is_floating_point:
            continue

        expert_match = MOE_EXPERT_RE.search(key)
        router_match = MOE_ROUTER_RE.search(key)
        if expert_match and config.moe_expert_blend > 0:
            expert_idx = int(expert_match.group(2))
            selected: list[tuple[Mapping[str, torch.Tensor], float]] = [
                (state, float(item["weight"]))
                for state, item in zip(client_state_dicts, assignments)
                if int(item["target_expert"]) == expert_idx and float(item["weight"]) > 0
            ]
            if not selected:
                continue
            total_weight = sum(weight for _, weight in selected)
            if total_weight <= 0:
                continue
            server_float = server_value.float()
            residual = torch.zeros_like(server_float)
            for state, weight in selected:
                residual = residual + (state[key].float() - server_float) * (weight / total_weight)
            result[key] = (server_float + float(config.moe_expert_blend) * residual).to(server_value.dtype)
            expert_key = str(expert_idx)
            expert_counts[expert_key] = expert_counts.get(expert_key, 0) + 1
            expert_weight_sums[expert_key] = expert_weight_sums.get(expert_key, 0.0) + total_weight
        elif router_match and config.moe_router_blend > 0:
            weighted = [
                (state, float(item["weight"]))
                for state, item in zip(client_state_dicts, assignments)
                if int(item["target_expert"]) >= 0 and float(item["weight"]) > 0
            ]
            total_weight = sum(weight for _, weight in weighted)
            if total_weight <= 0:
                continue
            server_float = server_value.float()
            residual = torch.zeros_like(server_float)
            for state, weight in weighted:
                residual = residual + (state[key].float() - server_float) * (weight / total_weight)
            result[key] = (server_float + float(config.moe_router_blend) * residual).to(server_value.dtype)
            router_weight_sum += total_weight

    diagnostics = {
        "expert_counts": expert_counts,
        "expert_weight_sums": expert_weight_sums,
        "router_weight_sum": router_weight_sum,
    }
    return result, diagnostics


def _enforce_server_floor(
    alpha: torch.Tensor,
    source_ids: Sequence[str],
    active: torch.Tensor,
    config: AggregationConfig,
) -> torch.Tensor:
    if "server" not in source_ids or config.min_server_alpha <= 0:
        return alpha

    adjusted = alpha.clone()
    server_idx = source_ids.index("server")
    for class_idx in range(adjusted.shape[1]):
        if not bool(active[class_idx]):
            continue
        server_weight = float(adjusted[server_idx, class_idx])
        floor = float(config.min_server_alpha)
        if server_weight >= floor:
            continue
        client_mass = max(1.0 - server_weight, EPS)
        scale = (1.0 - floor) / client_mass
        adjusted[:, class_idx] *= scale
        adjusted[server_idx, class_idx] = floor
        adjusted[:, class_idx] /= torch.clamp(adjusted[:, class_idx].sum(), min=EPS)
    return adjusted


def compute_reliability(
    stats: Sequence[ClientClassStats],
    state: dict[str, Any],
    config: AggregationConfig,
) -> tuple[dict[str, Any], torch.Tensor, list[str], torch.Tensor]:
    """Compute v1 reliability, then enforce a minimum server anchor per class."""

    config.validate()
    state, alpha, source_ids, active = v1.compute_reliability(stats, state, config)
    alpha = _enforce_server_floor(alpha.float(), source_ids, active, config)

    alpha_key = "|".join(source_ids)
    state["alpha"] = {alpha_key: alpha.tolist()}
    state["config"] = asdict(config) | {"implementation": "dqa_ver2_server_residual_anchor"}
    return state, alpha, source_ids, active


def _server_anchored_state_dict(
    client_state_dicts: Sequence[Mapping[str, torch.Tensor]],
    server_state_dict: Mapping[str, torch.Tensor],
    *,
    residual_blend: float,
    localize_bn: bool,
) -> dict[str, torch.Tensor]:
    anchored: dict[str, torch.Tensor] = {}
    for key, server_value in server_state_dict.items():
        if localize_bn and v1._is_batchnorm_key(key):
            anchored[key] = server_value
        elif torch.is_tensor(server_value) and server_value.dtype.is_floating_point:
            server_float = server_value.float()
            residuals = torch.stack(
                [state[key].float() - server_float for state in client_state_dicts],
                dim=0,
            ).mean(dim=0)
            anchored[key] = (server_float + residual_blend * residuals).to(server_value.dtype)
        else:
            anchored[key] = server_value
    return anchored


def apply_dynamic_classwise_head(
    anchored: dict[str, torch.Tensor],
    client_state_dicts: Sequence[Mapping[str, torch.Tensor]],
    server_state_dict: Mapping[str, torch.Tensor],
    alpha: torch.Tensor,
    active: torch.Tensor,
    config: AggregationConfig,
) -> dict[str, torch.Tensor]:
    """Apply class-wise quality-weighted client residuals to server head rows."""

    result = dict(anchored)
    source_state_dicts = list(client_state_dicts) + [server_state_dict]
    for key, anchored_value in anchored.items():
        if not torch.is_tensor(anchored_value) or not anchored_value.dtype.is_floating_point:
            continue
        rows_by_class = v1._classification_rows(key, anchored_value, config.num_classes)
        if rows_by_class is None:
            continue

        updated = anchored_value.float().clone()
        server_value = server_state_dict[key].float()
        source_values = [state[key].float() for state in source_state_dicts]
        for class_idx, rows in enumerate(rows_by_class):
            if not bool(active[class_idx]):
                continue
            weights = alpha[:, class_idx].to(updated.dtype)
            for row in rows:
                stacked_residuals = torch.stack(
                    [value[row] - server_value[row] for value in source_values],
                    dim=0,
                )
                dynamic_row = server_value[row] + torch.sum(
                    stacked_residuals * weights.view(-1, *([1] * (stacked_residuals.ndim - 1))),
                    dim=0,
                )
                updated[row] = (
                    (1.0 - config.classwise_blend) * updated[row]
                    + config.classwise_blend * dynamic_row
                )
        result[key] = updated.to(anchored_value.dtype)
    return result


def aggregate_checkpoints(
    client_checkpoints: Sequence[Path],
    server_checkpoint: Path,
    output_checkpoint: Path,
    stats: Sequence[ClientClassStats],
    state_path: Path | None,
    config: AggregationConfig,
    repo_root: Path,
    expert_assignments: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Aggregate checkpoints with server-anchored DQA-CWA v2."""

    if len(client_checkpoints) != len(stats):
        raise ValueError(
            f"client checkpoint count ({len(client_checkpoints)}) must match stats count ({len(stats)})"
        )

    state = load_state(state_path)
    state, alpha, source_ids, active = compute_reliability(stats, state, config)
    if "server" not in source_ids:
        raise ValueError("DQA v2 requires server_anchor > 0 so server residual anchoring is active")

    client_ckpts = [v1._load_checkpoint(path, repo_root) for path in client_checkpoints]
    server_ckpt = v1._load_checkpoint(server_checkpoint, repo_root)
    base = copy.deepcopy(server_ckpt)

    client_state_dicts = [v1._model_state_dict(ckpt, "model") for ckpt in client_ckpts]
    server_state_dict = v1._model_state_dict(server_ckpt, "model")
    base_state = _server_anchored_state_dict(
        client_state_dicts,
        server_state_dict,
        residual_blend=_client_residual_blend(config),
        localize_bn=config.localize_bn,
    )
    base_state, bn_diagnostics = apply_dynamic_batchnorm_residuals(
        base_state,
        client_state_dicts,
        server_state_dict,
        stats,
        expert_assignments,
        config,
    )
    base_state, moe_diagnostics = apply_dynamic_moe_expert_residuals(
        base_state,
        client_state_dicts,
        server_state_dict,
        expert_assignments,
        config,
    )
    dynamic = apply_dynamic_classwise_head(base_state, client_state_dicts, server_state_dict, alpha, active, config)
    v1._replace_model_state(base, dynamic, "model")

    if base.get("ema") is not None:
        ema_client_dicts = [v1._model_state_dict(ckpt, "ema") for ckpt in client_ckpts if ckpt.get("ema") is not None]
        server_ema = v1._model_state_dict(server_ckpt, "ema") if server_ckpt.get("ema") is not None else None
        if len(ema_client_dicts) == len(client_ckpts) and server_ema is not None:
            ema_base = _server_anchored_state_dict(
                ema_client_dicts,
                server_ema,
                residual_blend=_client_residual_blend(config),
                localize_bn=config.localize_bn,
            )
            ema_base, _ = apply_dynamic_batchnorm_residuals(
                ema_base,
                ema_client_dicts,
                server_ema,
                stats,
                expert_assignments,
                config,
            )
            ema_base, _ = apply_dynamic_moe_expert_residuals(
                ema_base,
                ema_client_dicts,
                server_ema,
                expert_assignments,
                config,
            )
            ema_dynamic = apply_dynamic_classwise_head(ema_base, ema_client_dicts, server_ema, alpha, active, config)
            v1._replace_model_state(base, ema_dynamic, "ema")

    base["epoch"] = -1
    base["optimizer"] = None
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, output_checkpoint)

    state["last_sources"] = source_ids
    state["last_alpha"] = alpha.tolist()
    state["last_active_classes"] = [bool(x) for x in active.tolist()]
    state["last_moe_expert_assignments"] = _normalise_expert_assignments(expert_assignments)
    state["last_moe_diagnostics"] = moe_diagnostics
    state["last_bn_diagnostics"] = bn_diagnostics
    save_state(state_path, state)
    return output_checkpoint, state


def aggregate_fedavg_checkpoints(
    client_checkpoints: Sequence[Path],
    server_checkpoint: Path,
    output_checkpoint: Path,
    *,
    repo_root: Path,
    localize_bn: bool = True,
) -> Path:
    """Guard fallback for v2: server-anchored client residual averaging."""

    client_ckpts = [v1._load_checkpoint(path, repo_root) for path in client_checkpoints]
    server_ckpt = v1._load_checkpoint(server_checkpoint, repo_root)
    base = copy.deepcopy(server_ckpt)

    client_state_dicts = [v1._model_state_dict(ckpt, "model") for ckpt in client_ckpts]
    server_state_dict = v1._model_state_dict(server_ckpt, "model")
    anchored = _server_anchored_state_dict(
        client_state_dicts,
        server_state_dict,
        residual_blend=0.35,
        localize_bn=localize_bn,
    )
    v1._replace_model_state(base, anchored, "model")

    if base.get("ema") is not None:
        ema_client_dicts = [v1._model_state_dict(ckpt, "ema") for ckpt in client_ckpts if ckpt.get("ema") is not None]
        server_ema = v1._model_state_dict(server_ckpt, "ema") if server_ckpt.get("ema") is not None else None
        if len(ema_client_dicts) == len(client_ckpts) and server_ema is not None:
            ema_anchored = _server_anchored_state_dict(
                ema_client_dicts,
                server_ema,
                residual_blend=0.35,
                localize_bn=localize_bn,
            )
            v1._replace_model_state(base, ema_anchored, "ema")

    base["epoch"] = -1
    base["optimizer"] = None
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, output_checkpoint)
    return output_checkpoint
