#!/usr/bin/env python3
"""Run DQA-CWA v2 with BDD100K scene-based clients."""

from __future__ import annotations

import dqa_cwa_aggregation_v2 as v2
import run_dqa_cwa_fedsto as base
from run_dqa_cwa_fedsto_scene import _prepare_scene_modules


DQA_PROTOCOL_VERSION = "dqa_ver2_scene_server_residual_anchor_v1"

base.AggregationConfig = v2.AggregationConfig
base.aggregate_fedavg_checkpoints = v2.aggregate_fedavg_checkpoints
base.aggregate_checkpoints = v2.aggregate_checkpoints
base.compute_reliability = v2.compute_reliability
base.load_round_stats = v2.load_round_stats
base.load_state = v2.load_state
base.save_state = v2.save_state
base.DQA_PROTOCOL_VERSION = DQA_PROTOCOL_VERSION
base.DEFAULT_DQA_WORK_ROOT = base.RESEARCH_ROOT / "efficientteacher_dqa_ver2_scene_12h"
base._prepare_fedsto_modules = _prepare_scene_modules


def main() -> None:
    parsed = base.parse_args()
    if parsed.protocol_version == DQA_PROTOCOL_VERSION:
        parsed.protocol_version = DQA_PROTOCOL_VERSION
    if parsed.setup_only:
        base._prepare_fedsto_modules(parsed.workspace_root)[0].build_base_configs()
    else:
        base.run_protocol(parsed)


if __name__ == "__main__":
    main()
