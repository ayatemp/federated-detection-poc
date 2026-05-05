#!/usr/bin/env python3
"""DQA-CWA v2 aggregation.

Version 1 averaged client checkpoints first, then blended class-wise head rows.
The 03 run showed that this can erase too much of the labeled server model before
the server update recovers it.  Version 2 keeps the server checkpoint as the
anchor and applies quality-weighted client residuals on top of it.
"""

from __future__ import annotations

import copy
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

    def validate(self) -> None:
        super().validate()
        if not 0.0 <= self.min_server_alpha <= 1.0:
            raise ValueError(f"min_server_alpha must be in [0, 1], got {self.min_server_alpha}")
        if self.residual_blend is not None and not 0.0 <= self.residual_blend <= 1.0:
            raise ValueError(f"residual_blend must be in [0, 1], got {self.residual_blend}")


def _client_residual_blend(config: AggregationConfig) -> float:
    if config.residual_blend is not None:
        return float(config.residual_blend)
    return min(float(config.classwise_blend), 0.35)


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
            ema_dynamic = apply_dynamic_classwise_head(ema_base, ema_client_dicts, server_ema, alpha, active, config)
            v1._replace_model_state(base, ema_dynamic, "ema")

    base["epoch"] = -1
    base["optimizer"] = None
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, output_checkpoint)

    state["last_sources"] = source_ids
    state["last_alpha"] = alpha.tolist()
    state["last_active_classes"] = [bool(x) for x in active.tolist()]
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
