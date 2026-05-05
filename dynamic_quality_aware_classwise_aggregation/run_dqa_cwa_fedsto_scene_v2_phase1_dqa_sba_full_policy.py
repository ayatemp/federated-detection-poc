#!/usr/bin/env python3
"""Run warm-up -> Phase 1 -> Phase 2 with DQA-SBA from Phase 1.

This runner keeps the FedSTO phase split:

* warm-up: supervised server training
* Phase 1: backbone-only client/server training
* Phase 2: full-parameter regularized refinement

The difference from the earlier 08 runners is that DQA starts at Phase 1.
During Phase 1, pseudo-label statistics do not directly update the detector
head.  They control how strongly each client's backbone residual is mixed into
the server model.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import dqa_cwa_aggregation_v2 as v2
import dqa_sba_aggregation as sba
import run_dqa_cwa_fedsto as base
import run_dqa_cwa_fedsto_scene_v2_tri_stage_policy as tri_stage


DQA_PROTOCOL_VERSION = "dqa08_4_scene_phase1_dqa_sba_full_policy_v1"
DEFAULT_WORK_ROOT = base.RESEARCH_ROOT / "efficientteacher_dqa08_4_scene_phase1_dqa_sba_full_policy"
PHASE_RE = re.compile(r"phase(?P<phase>\d+)_round(?P<round>\d+)")


def _phase_from_path(path: Path) -> int | None:
    match = PHASE_RE.search(str(path))
    return int(match.group("phase")) if match else None


def _aggregate_checkpoints(
    client_checkpoints,
    server_checkpoint,
    output_checkpoint,
    stats,
    state_path,
    config,
    repo_root,
):
    phase = _phase_from_path(output_checkpoint)
    if phase == 1:
        print(
            "DQA08_4 Phase 1 aggregation: DQA-SBA "
            f"residual={sba.scheduled_phase1_residual(sba.round_from_path(output_checkpoint)):.4f}"
        )
        return sba.aggregate_phase1_backbone_checkpoints(
            client_checkpoints=client_checkpoints,
            server_checkpoint=server_checkpoint,
            output_checkpoint=output_checkpoint,
            stats=stats,
            state_path=state_path,
            config=config,
            repo_root=repo_root,
        )
    return v2.aggregate_checkpoints(
        client_checkpoints=client_checkpoints,
        server_checkpoint=server_checkpoint,
        output_checkpoint=output_checkpoint,
        stats=stats,
        state_path=state_path,
        config=config,
        repo_root=repo_root,
    )


def _aggregate_fedavg_checkpoints(
    client_checkpoints,
    server_checkpoint,
    output_checkpoint,
    *,
    repo_root,
    localize_bn=True,
):
    phase = _phase_from_path(output_checkpoint)
    if phase == 1:
        print("DQA08_4 Phase 1 fallback: conservative backbone residual average")
        return sba.aggregate_phase1_fallback(
            client_checkpoints=client_checkpoints,
            server_checkpoint=server_checkpoint,
            output_checkpoint=output_checkpoint,
            repo_root=repo_root,
            localize_bn=localize_bn,
        )
    return v2.aggregate_fedavg_checkpoints(
        client_checkpoints=client_checkpoints,
        server_checkpoint=server_checkpoint,
        output_checkpoint=output_checkpoint,
        repo_root=repo_root,
        localize_bn=localize_bn,
    )


base.AggregationConfig = v2.AggregationConfig
base.aggregate_fedavg_checkpoints = _aggregate_fedavg_checkpoints
base.aggregate_checkpoints = _aggregate_checkpoints
base.compute_reliability = v2.compute_reliability
base.load_round_stats = v2.load_round_stats
base.load_state = v2.load_state
base.save_state = v2.save_state
base.DQA_PROTOCOL_VERSION = DQA_PROTOCOL_VERSION
base.DEFAULT_DQA_WORK_ROOT = DEFAULT_WORK_ROOT
base._prepare_fedsto_modules = tri_stage._prepare_tri_stage_scene_modules
base.run_train = tri_stage._tri_stage_run_train


def main() -> None:
    parsed = base.parse_args()
    if parsed.protocol_version != DQA_PROTOCOL_VERSION:
        parsed.protocol_version = DQA_PROTOCOL_VERSION

    if parsed.dqa_start_phase != 1:
        print(f"DQA08_4 forces --dqa-start-phase 1; got {parsed.dqa_start_phase}, overriding.")
        parsed.dqa_start_phase = 1

    os.environ["DQA08_STATS_ROOT"] = str(parsed.stats_root.resolve())
    os.environ["DQA08_PHASE2_ROUNDS"] = str(parsed.phase2_rounds)
    os.environ["DQA08_NUM_CLASSES"] = str(parsed.num_classes)
    os.environ["DQA084_PHASE1_ROUNDS"] = str(parsed.phase1_rounds)
    os.environ.setdefault("DQA08_POLICY_HORIZON_ROUNDS", str(max(parsed.phase2_rounds, 1)))
    os.environ.setdefault(
        "DQA08_THRESHOLD_LOG",
        str((parsed.stats_root / "phase1_dqa_sba_full_policy_schedule.jsonl").resolve()),
    )

    if not parsed.setup_only and not parsed.dry_run and not tri_stage._policy_model_path().exists():
        raise FileNotFoundError(
            f"Missing learned threshold policy model: {tri_stage._policy_model_path()}. "
            "Run threshold_policy_model/01_train_dqa05_threshold_policy.ipynb first."
        )

    if parsed.setup_only:
        base._prepare_fedsto_modules(parsed.workspace_root)[0].build_base_configs()
    else:
        print("DQA08_4 profile: phase1 DQA-SBA full warmup-to-phase2")
        print(f"DQA08_4 protocol: {DQA_PROTOCOL_VERSION}")
        print(
            "DQA08_4 phase1 residual schedule:",
            os.environ.get("DQA084_PHASE1_RESIDUAL_START", "0.45"),
            "->",
            os.environ.get("DQA084_PHASE1_RESIDUAL_END", "0.18"),
        )
        print(f"DQA08_4 phase1 max relative update: {os.environ.get('DQA084_PHASE1_MAX_RELATIVE_UPDATE', '0.04')}")
        print(f"DQA08_4 policy model: {tri_stage._policy_model_path()}")
        base.run_protocol(parsed)


if __name__ == "__main__":
    main()
