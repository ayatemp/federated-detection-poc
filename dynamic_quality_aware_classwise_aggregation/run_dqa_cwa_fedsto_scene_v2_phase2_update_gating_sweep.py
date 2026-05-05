#!/usr/bin/env python3
"""Run DQA08_3 phase2-only update-gating sweeps.

This runner reuses the 08_2 phase2-only setup, then adds a small DQA-style
trust region around target client updates.  The key difference from 08/08_2 is
that client residual strength is configurable and can decay across phase2
rounds.  DQA still decides class/client reliability, but the global model is
only allowed to move a small distance toward target pseudoGT updates.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import replace
from pathlib import Path

import dqa_cwa_aggregation_v2 as v2
import run_dqa_cwa_fedsto as base
import run_dqa_cwa_fedsto_scene_v2_phase2_head_protected_policy as phase2_seeded
import run_dqa_cwa_fedsto_scene_v2_tri_stage_policy as tri_stage


DQA_PROTOCOL_VERSION = "dqa08_3_scene_phase2_update_gating_sweep_v1"
DEFAULT_WORK_ROOT = base.RESEARCH_ROOT / "efficientteacher_dqa08_3_phase2_update_gating_sweep"
ROUND_RE = re.compile(r"phase2_round(?P<round>\d+)")


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


def _round_from_path(path: Path) -> int | None:
    match = ROUND_RE.search(str(path))
    return int(match.group("round")) if match else None


def _scheduled_residual(round_idx: int | None) -> float:
    start = _float_env("DQA08_3_RESIDUAL_START", 0.10)
    end = _float_env("DQA08_3_RESIDUAL_END", start)
    total = max(_int_env("DQA08_3_PHASE2_ROUNDS", 10), 1)
    if round_idx is None or total <= 1:
        return start
    progress = min(max((round_idx - 1) / (total - 1), 0.0), 1.0)
    return start + (end - start) * progress


def _dqa_config(args) -> v2.AggregationConfig:
    return v2.AggregationConfig(
        num_classes=args.num_classes,
        count_ema=args.count_ema,
        quality_ema=args.quality_ema,
        alpha_ema=args.alpha_ema,
        temperature=args.temperature,
        uniform_mix=args.uniform_mix,
        classwise_blend=args.classwise_blend,
        stability_lambda=args.stability_lambda,
        min_effective_count=args.min_effective_count,
        server_anchor=args.server_anchor,
        localize_bn=args.localize_bn,
        min_server_alpha=_float_env("DQA08_3_MIN_SERVER_ALPHA", 0.75),
        residual_blend=_float_env("DQA08_3_RESIDUAL_START", 0.10),
    )


def _aggregate_update_gated(
    client_checkpoints,
    server_checkpoint,
    output_checkpoint,
    stats,
    state_path,
    config,
    repo_root,
):
    round_idx = _round_from_path(output_checkpoint)
    residual = _scheduled_residual(round_idx)
    cfg = replace(
        copy.copy(config),
        residual_blend=residual,
        min_server_alpha=_float_env("DQA08_3_MIN_SERVER_ALPHA", getattr(config, "min_server_alpha", 0.75)),
    )
    output, state = v2.aggregate_checkpoints(
        client_checkpoints=client_checkpoints,
        server_checkpoint=server_checkpoint,
        output_checkpoint=output_checkpoint,
        stats=stats,
        state_path=state_path,
        config=cfg,
        repo_root=repo_root,
    )
    state.setdefault("update_gating", []).append(
        {
            "round": round_idx,
            "residual_blend": residual,
            "min_server_alpha": cfg.min_server_alpha,
            "classwise_blend": cfg.classwise_blend,
            "server_anchor": cfg.server_anchor,
            "variant": os.environ.get("DQA08_3_VARIANT", "unnamed"),
        }
    )
    v2.save_state(state_path, state)
    print(
        "DQA08_3 update gate:",
        f"round={round_idx}",
        f"residual_blend={residual:.4f}",
        f"min_server_alpha={cfg.min_server_alpha:.3f}",
        f"classwise_blend={cfg.classwise_blend:.3f}",
        f"server_anchor={cfg.server_anchor:.3f}",
    )
    return output, state


phase2_seeded.DQA_PROTOCOL_VERSION = DQA_PROTOCOL_VERSION
base.DQA_PROTOCOL_VERSION = DQA_PROTOCOL_VERSION
base.DEFAULT_DQA_WORK_ROOT = DEFAULT_WORK_ROOT
base._prepare_fedsto_modules = phase2_seeded._prepare_head_protected_scene_modules
base.run_train = tri_stage._tri_stage_run_train
base.AggregationConfig = v2.AggregationConfig
base.aggregate_checkpoints = _aggregate_update_gated
base._dqa_config = _dqa_config


def main() -> None:
    parsed = base.parse_args()
    if parsed.protocol_version in {tri_stage.DQA_PROTOCOL_VERSION, phase2_seeded.DQA_PROTOCOL_VERSION}:
        parsed.protocol_version = DQA_PROTOCOL_VERSION
    if parsed.workspace_root == tri_stage.base.RESEARCH_ROOT / "efficientteacher_dqa08_scene_tri_stage_policy_8h":
        parsed.workspace_root = DEFAULT_WORK_ROOT

    os.environ["DQA08_STATS_ROOT"] = str(parsed.stats_root.resolve())
    os.environ["DQA08_PHASE2_ROUNDS"] = str(parsed.phase2_rounds)
    os.environ["DQA08_3_PHASE2_ROUNDS"] = str(parsed.phase2_rounds)
    os.environ["DQA08_NUM_CLASSES"] = str(parsed.num_classes)
    os.environ.setdefault("DQA08_POLICY_HORIZON_ROUNDS", str(parsed.phase2_rounds))
    os.environ.setdefault(
        "DQA08_THRESHOLD_LOG",
        str((parsed.stats_root / "update_gating_policy_schedule.jsonl").resolve()),
    )

    if parsed.setup_only:
        base._prepare_fedsto_modules(parsed.workspace_root)[0].build_base_configs()
        return

    if parsed.phase1_rounds != 0:
        raise ValueError("08_3 is phase2-only. Run with --phase1-rounds 0.")

    if not parsed.dry_run:
        phase2_seeded._seed_phase1_checkpoint(parsed.workspace_root)
        if not tri_stage._policy_model_path().exists():
            raise FileNotFoundError(
                f"Missing learned threshold policy model: {tri_stage._policy_model_path()}. "
                "Run threshold_policy_model/01_train_dqa05_threshold_policy.ipynb first."
            )

    print("DQA08_3 profile: phase2-only dynamic quality-aware update gating")
    print(f"DQA08_3 variant: {os.environ.get('DQA08_3_VARIANT', 'unnamed')}")
    print(f"DQA08_3 seed: {phase2_seeded._source_workspace()} phase1 round {phase2_seeded._source_phase1_round()}")
    print(
        "DQA08_3 residual schedule:",
        _float_env("DQA08_3_RESIDUAL_START", 0.10),
        "->",
        _float_env("DQA08_3_RESIDUAL_END", 0.10),
    )
    print(f"DQA08_3 min_server_alpha: {_float_env('DQA08_3_MIN_SERVER_ALPHA', 0.75)}")
    print(f"DQA08_3 client train_scope: {os.environ.get('DQA08_2_CLIENT_TRAIN_SCOPE', 'backbone')}")
    base.run_protocol(parsed)


if __name__ == "__main__":
    main()
