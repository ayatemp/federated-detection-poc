#!/usr/bin/env python3
"""Run DQA-targeted latent-MoE expert aggregation loops.

36 keeps the FedMoX-like client sampling and router specialization from 35, but
changes the important part: aggregation now explicitly carries each client-owned
expert residual into the matching global expert.  The shared detector can remain
server-anchored while the MoE branch learns specialists.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_35_dqa_router_specialized_fedmox_loop as base  # noqa: E402


base.DEFAULT_OUTPUT = base.AGG_ROOT / "output" / "36_dqa_targeted_expert_aggregation_loop"
base.SUMMARY_CSV = base.REPORTS_ROOT / "36_dqa_targeted_expert_aggregation_loop_summary.csv"
base.RUN_LABEL = "36"
base.RUN_DESCRIPTION = "DQA-targeted latent-MoE expert aggregation loop"

base.STAGES = [
    base.Stage("probe2", phase1_rounds=2, phase2_rounds=0, min_best_map50=0.464, min_gain_vs_warmup=0.003),
    base.Stage("probe5", phase1_rounds=5, phase2_rounds=0, min_best_map50=0.475, min_gain_vs_warmup=0.010),
    base.Stage("probe10", phase1_rounds=10, phase2_rounds=0, min_best_map50=0.490, min_gain_vs_warmup=0.022),
    base.Stage("phase1_20", phase1_rounds=20, phase2_rounds=0, min_best_map50=0.510, min_gain_vs_warmup=0.040),
    base.Stage("full20_30", phase1_rounds=20, phase2_rounds=30, min_best_map50=0.550, min_gain_vs_warmup=0.080),
]

base.TRIALS = [
    base.Trial(
        name="36a_domain4_moehead_targeted_expert_agg",
        hypothesis=(
            "Domain-specialist MoE with actual expert aggregation.  Clients train only the latent MoE head, "
            "router targets day/night/scene experts, and DQA-gated aggregation copies each client residual "
            "only into its assigned global expert while leaving the shared detector anchored."
        ),
        args=[
            "--num-experts", "4",
            "--top-k", "1",
            "--router-temperature", "0.90",
            "--router-balance-weight", "0.003",
            "--router-entropy-weight", "0.0002",
            "--router-specialization-map", "domain4",
            "--router-specialization-weight", "0.120",
            "--router-specialization-max-weight", "0.090",
            "--router-specialization-min-quality", "0.58",
            "--router-specialization-min-boxes", "650",
            "--dqa-moe-expert-blend", "0.90",
            "--dqa-moe-router-blend", "0.35",
            "--phase1-train-scope", "moe_head",
            "--phase1-repair-train-scope", "moe_head",
            "--phase1-client-lr", "0.00085",
            "--phase1-source-repeat", "1",
            "--phase1-pseudo-repeat", "2",
            "--phase1-loss-box", "0.00012",
            "--phase2-train-scope", "all",
            "--phase2-repair-train-scope", "all",
            "--phase2-client-lr", "0.000030",
            "--phase2-source-repeat", "1",
            "--phase2-pseudo-repeat", "1",
            "--phase2-loss-box", "0.00007",
            "--server-repair-lr", "0.00012",
            "--server-repair-loss-box", "0.008",
            "--dqa-server-anchor", "0.70",
            "--dqa-min-server-alpha", "0.65",
            "--dqa-residual-blend", "0.00",
            "--late-dqa-server-anchor", "0.42",
            "--late-dqa-min-server-alpha", "0.36",
            "--late-dqa-residual-blend", "0.00",
            "--curriculum-start-round", "21",
            "--expert-keep-fraction", "0.60",
            "--expert-max-class-fraction", "0.25",
            "--actual-max-class-fraction", "0.35",
            "--late-expert-keep-fraction", "0.78",
            "--late-expert-max-class-fraction", "0.30",
            "--late-actual-max-class-fraction", "0.42",
            "--min-views", "2",
            "--min-models", "0",
            "--min-score", "0.30",
            "--min-stability", "0.70",
            "--late-min-score", "0.24",
            "--late-min-stability", "0.64",
            "--max-boxes-per-image", "6",
        ],
    ),
    base.Trial(
        name="36b_hybrid_class_domain_targeted_expert_agg",
        hypothesis=(
            "Hybrid class/domain specialist MoE with target-aware expert aggregation.  VRU-heavy and traffic-heavy "
            "pseudoGT rounds own class experts; otherwise domain-time experts are used.  This tests whether class "
            "specialists can improve rare classes without letting noisy shared parameters drift."
        ),
        args=[
            "--num-experts", "4",
            "--top-k", "1",
            "--router-temperature", "0.95",
            "--router-balance-weight", "0.004",
            "--router-entropy-weight", "0.0002",
            "--router-specialization-map", "hybrid_dqa4",
            "--router-specialization-weight", "0.130",
            "--router-specialization-max-weight", "0.095",
            "--router-specialization-min-quality", "0.60",
            "--router-specialization-min-boxes", "750",
            "--router-specialization-class-threshold", "0.27",
            "--dqa-moe-expert-blend", "0.85",
            "--dqa-moe-router-blend", "0.30",
            "--phase1-train-scope", "moe_head",
            "--phase1-repair-train-scope", "moe_head",
            "--phase1-client-lr", "0.00075",
            "--phase1-source-repeat", "1",
            "--phase1-pseudo-repeat", "2",
            "--phase1-loss-box", "0.00011",
            "--phase2-train-scope", "all",
            "--phase2-repair-train-scope", "all",
            "--phase2-client-lr", "0.000030",
            "--phase2-source-repeat", "1",
            "--phase2-pseudo-repeat", "1",
            "--phase2-loss-box", "0.00007",
            "--server-repair-lr", "0.00012",
            "--server-repair-loss-box", "0.008",
            "--dqa-server-anchor", "0.70",
            "--dqa-min-server-alpha", "0.65",
            "--dqa-residual-blend", "0.00",
            "--late-dqa-server-anchor", "0.42",
            "--late-dqa-min-server-alpha", "0.36",
            "--late-dqa-residual-blend", "0.00",
            "--curriculum-start-round", "21",
            "--expert-keep-fraction", "0.62",
            "--expert-max-class-fraction", "0.25",
            "--actual-max-class-fraction", "0.35",
            "--late-expert-keep-fraction", "0.78",
            "--late-expert-max-class-fraction", "0.30",
            "--late-actual-max-class-fraction", "0.42",
            "--min-views", "2",
            "--min-models", "0",
            "--min-score", "0.30",
            "--min-stability", "0.70",
            "--late-min-score", "0.24",
            "--late-min-stability", "0.64",
            "--max-boxes-per-image", "6",
        ],
    ),
]


def main(argv: list[str] | None = None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
