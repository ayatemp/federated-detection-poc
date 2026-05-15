#!/usr/bin/env python3
"""Run selective backbone + MoE DQA-MoX learning loops.

41 showed that pseudoGT corrupts the detection head even when box loss is nearly
off or only MoE head/router is trainable.  42 follows the FedSTO/FedMoX clue:
adapt visual features and sparse experts, while keeping the ordinary detector
head mostly frozen.  DQA pseudoGT quality still controls routing and aggregation.
"""

from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve()
AGG_ROOT = SCRIPT_PATH.parents[1]
BASE_SCRIPT = AGG_ROOT / "scripts" / "run_35_dqa_router_specialized_fedmox_loop.py"
if str(BASE_SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(BASE_SCRIPT.parent))

import run_35_dqa_router_specialized_fedmox_loop as base  # noqa: E402


base.DEFAULT_OUTPUT = AGG_ROOT / "output" / "42_backbone_moe_selective_dqamox_loop"
base.SUMMARY_CSV = AGG_ROOT / "reports" / "42_backbone_moe_selective_dqamox_loop_summary.csv"
base.RUN_LABEL = "42"
base.RUN_DESCRIPTION = "selective backbone plus MoE DQA-MoX learning loop"
base.RUN_PAPER_NOTES = (
    "FedMoX spatial router and Soft-Mixture for stable semi-supervised FL; "
    "FedSTO selective backbone refinement and local EMA pseudo labels; "
    "FedMix/FedMoE client-specific expert sharing; HI-MoE scene-first routing for detection."
)

base.STAGES = [
    base.Stage("probe3", phase1_rounds=3, phase2_rounds=0, min_best_map50=0.466, min_gain_vs_warmup=0.004),
    base.Stage("probe6", phase1_rounds=6, phase2_rounds=0, min_best_map50=0.478, min_gain_vs_warmup=0.012),
    base.Stage("probe12", phase1_rounds=12, phase2_rounds=0, min_best_map50=0.495, min_gain_vs_warmup=0.026),
    base.Stage("phase1_24", phase1_rounds=24, phase2_rounds=0, min_best_map50=0.515, min_gain_vs_warmup=0.045),
    base.Stage("full24_26", phase1_rounds=24, phase2_rounds=26, min_best_map50=0.550, min_gain_vs_warmup=0.080),
]

base.TRIALS = [
    base.Trial(
        name="42a_backbone_moe_softmix_domain6",
        hypothesis=(
            "PseudoGT should adapt appearance and routing, not the ordinary detection head.  This trial trains only "
            "backbone plus MoE router/expert slots with soft top-2 routing, strict pseudo boxes, high DQA anchoring, "
            "and six domain experts."
        ),
        args=[
            "--client-sampling-seed", "420069",
            "--imgsz", "640",
            "--pseudo-imgsz", "1152",
            "--num-experts", "6",
            "--top-k", "2",
            "--router-temperature", "1.60",
            "--router-balance-weight", "0.020",
            "--router-entropy-weight", "0.0030",
            "--router-specialization-map", "domain6",
            "--router-specialization-weight", "0.060",
            "--router-specialization-max-weight", "0.040",
            "--router-specialization-min-quality", "0.64",
            "--router-specialization-min-boxes", "260",
            "--dqa-moe-expert-blend", "0.18",
            "--dqa-moe-router-blend", "0.24",
            "--dqa-classwise-blend", "0.25",
            "--dqa-temperature", "0.82",
            "--dqa-client-balance-target", "max",
            "--dqa-client-balance-max-scale", "2.4",
            "--phase1-train-scope", "backbone_moe_head",
            "--phase1-repair-train-scope", "backbone_moe_head",
            "--phase1-client-lr", "0.000055",
            "--phase1-source-repeat", "2",
            "--phase1-pseudo-repeat", "1",
            "--phase1-loss-box", "0.000035",
            "--client-loss-cls", "0.55",
            "--client-loss-obj", "0.95",
            "--phase2-train-scope", "backbone_moe_head",
            "--phase2-repair-train-scope", "backbone_moe_head",
            "--phase2-client-lr", "0.000014",
            "--phase2-source-repeat", "2",
            "--phase2-pseudo-repeat", "1",
            "--phase2-loss-box", "0.000020",
            "--server-repair-lr", "0.000035",
            "--server-repair-loss-box", "0.003",
            "--server-repair-loss-cls", "0.45",
            "--server-repair-loss-obj", "0.85",
            "--dqa-server-anchor", "0.90",
            "--dqa-min-server-alpha", "0.84",
            "--dqa-residual-blend", "0.000",
            "--late-dqa-server-anchor", "0.76",
            "--late-dqa-min-server-alpha", "0.68",
            "--late-dqa-residual-blend", "0.000",
            "--curriculum-start-round", "25",
            "--expert-keep-fraction", "0.52",
            "--expert-max-class-fraction", "0.18",
            "--actual-max-class-fraction", "0.25",
            "--late-expert-keep-fraction", "0.64",
            "--late-expert-max-class-fraction", "0.22",
            "--late-actual-max-class-fraction", "0.30",
            "--min-views", "2",
            "--min-models", "0",
            "--min-score", "0.38",
            "--min-stability", "0.76",
            "--late-min-score", "0.32",
            "--late-min-stability", "0.70",
            "--max-boxes-per-image", "5",
            "--max-class-fraction", "0.26",
            "--min-class-keep", "50",
            "--client-mixup", "0.00",
            "--client-mosaic", "0.45",
            "--client-scale", "0.16",
            "--client-hsv-s", "0.18",
            "--client-hsv-v", "0.14",
        ],
    ),
    base.Trial(
        name="42b_backbone_moe_stats_only",
        hypothesis=(
            "Control the dangerous part even harder: generate self pseudoGT only to score DQA quality and choose "
            "domain-router targets, but do not include pseudo images in the supervised client dataloader.  Experts "
            "learn source-grounded domain slots without pseudo-box noise."
        ),
        args=[
            "--client-sampling-seed", "420069",
            "--imgsz", "640",
            "--pseudo-imgsz", "1152",
            "--num-experts", "6",
            "--top-k", "2",
            "--router-temperature", "1.45",
            "--router-balance-weight", "0.016",
            "--router-entropy-weight", "0.0025",
            "--router-specialization-map", "domain6",
            "--router-specialization-weight", "0.090",
            "--router-specialization-max-weight", "0.060",
            "--router-specialization-min-quality", "0.64",
            "--router-specialization-min-boxes", "240",
            "--dqa-moe-expert-blend", "0.16",
            "--dqa-moe-router-blend", "0.30",
            "--dqa-classwise-blend", "0.18",
            "--dqa-temperature", "0.90",
            "--dqa-client-balance-target", "max",
            "--dqa-client-balance-max-scale", "2.0",
            "--phase1-train-scope", "backbone_moe_head",
            "--phase1-repair-train-scope", "backbone_moe_head",
            "--phase1-client-lr", "0.000028",
            "--phase1-source-repeat", "1",
            "--phase1-pseudo-repeat", "0",
            "--phase1-loss-box", "0.0020",
            "--client-loss-cls", "0.50",
            "--client-loss-obj", "1.00",
            "--phase2-train-scope", "backbone_moe_head",
            "--phase2-repair-train-scope", "backbone_moe_head",
            "--phase2-client-lr", "0.000010",
            "--phase2-source-repeat", "1",
            "--phase2-pseudo-repeat", "0",
            "--phase2-loss-box", "0.0010",
            "--server-repair-lr", "0.000020",
            "--server-repair-loss-box", "0.0020",
            "--server-repair-loss-cls", "0.45",
            "--server-repair-loss-obj", "0.90",
            "--dqa-server-anchor", "0.94",
            "--dqa-min-server-alpha", "0.88",
            "--dqa-residual-blend", "0.000",
            "--late-dqa-server-anchor", "0.82",
            "--late-dqa-min-server-alpha", "0.74",
            "--late-dqa-residual-blend", "0.000",
            "--curriculum-start-round", "25",
            "--expert-keep-fraction", "0.54",
            "--expert-max-class-fraction", "0.18",
            "--actual-max-class-fraction", "0.25",
            "--late-expert-keep-fraction", "0.64",
            "--late-expert-max-class-fraction", "0.22",
            "--late-actual-max-class-fraction", "0.30",
            "--min-views", "2",
            "--min-models", "0",
            "--min-score", "0.38",
            "--min-stability", "0.76",
            "--late-min-score", "0.32",
            "--late-min-stability", "0.70",
            "--max-boxes-per-image", "5",
            "--max-class-fraction", "0.26",
            "--min-class-keep", "50",
            "--client-mixup", "0.00",
            "--client-mosaic", "0.35",
            "--client-scale", "0.12",
            "--client-hsv-s", "0.14",
            "--client-hsv-v", "0.12",
        ],
    ),
]


def main(argv: list[str] | None = None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
