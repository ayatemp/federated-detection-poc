#!/usr/bin/env python3
"""Run high-resolution self-pseudoGT FedMoX/DQA-MoE learning loops.

36 showed that targeted expert aggregation moved the latent MoE slots
mechanically, but paper-protocol mAP stayed at the warmup.  This loop keeps the
FedMoX-like learning shape and improves the signal before learning: pseudoGT is
generated from the current self model with high-resolution identity+hflip
stability, while training/evaluation stay at the standard 640 setting.
"""

from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve()
AGG_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = AGG_ROOT.parent
REPO_ROOT = PROJECT_ROOT.parents[1]

BASE_SCRIPT = AGG_ROOT / "scripts" / "run_35_dqa_router_specialized_fedmox_loop.py"
if str(BASE_SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(BASE_SCRIPT.parent))

import run_35_dqa_router_specialized_fedmox_loop as base  # noqa: E402


base.DEFAULT_OUTPUT = AGG_ROOT / "output" / "37_highres_self_pseudogt_soft_moe_loop"
base.SUMMARY_CSV = AGG_ROOT / "reports" / "37_highres_self_pseudogt_soft_moe_loop_summary.csv"
base.RUN_LABEL = "37"
base.RUN_DESCRIPTION = "high-resolution self-pseudoGT soft-MoE learning loop"
base.RUN_PAPER_NOTES = (
    "FedMoX spatial routing/soft mixture; FedMoE-DA fine-grained domain-aware aggregation; "
    "MixPL warning about pseudo-label missed detections; SAHI high-resolution/sliced small-object recall."
)

base.STAGES = [
    base.Stage("probe2", phase1_rounds=2, phase2_rounds=0, min_best_map50=0.464, min_gain_vs_warmup=0.003),
    base.Stage("probe5", phase1_rounds=5, phase2_rounds=0, min_best_map50=0.474, min_gain_vs_warmup=0.010),
    base.Stage("probe10", phase1_rounds=10, phase2_rounds=0, min_best_map50=0.490, min_gain_vs_warmup=0.022),
    base.Stage("phase1_20", phase1_rounds=20, phase2_rounds=0, min_best_map50=0.510, min_gain_vs_warmup=0.040),
    base.Stage("full20_30", phase1_rounds=20, phase2_rounds=30, min_best_map50=0.550, min_gain_vs_warmup=0.080),
]

base.TRIALS = [
    base.Trial(
        name="37a_pseudo960_neckhead_soft_hybrid_router",
        hypothesis=(
            "High-resolution self pseudoGT should fix the weak signal seen in 36.  The detector still trains in the "
            "FedMoX-like loop at 640, but pseudo boxes are generated at 960 with identity+hflip stability.  Router "
            "specialization is soft/top-2 and DQA aggregation is allowed to move shared neck/head a little."
        ),
        args=[
            "--imgsz", "640",
            "--pseudo-imgsz", "960",
            "--num-experts", "4",
            "--top-k", "2",
            "--router-temperature", "1.25",
            "--router-balance-weight", "0.012",
            "--router-entropy-weight", "0.0010",
            "--router-specialization-map", "hybrid_dqa4",
            "--router-specialization-weight", "0.060",
            "--router-specialization-max-weight", "0.045",
            "--router-specialization-min-quality", "0.60",
            "--router-specialization-min-boxes", "900",
            "--router-specialization-class-threshold", "0.29",
            "--dqa-moe-expert-blend", "0.20",
            "--dqa-moe-router-blend", "0.10",
            "--phase1-train-scope", "neck_head",
            "--phase1-repair-train-scope", "neck_head",
            "--phase1-client-lr", "0.00028",
            "--phase1-source-repeat", "2",
            "--phase1-pseudo-repeat", "2",
            "--phase1-loss-box", "0.0015",
            "--phase2-train-scope", "all",
            "--phase2-repair-train-scope", "all",
            "--phase2-client-lr", "0.000035",
            "--phase2-source-repeat", "1",
            "--phase2-pseudo-repeat", "1",
            "--phase2-loss-box", "0.00015",
            "--server-repair-lr", "0.00022",
            "--server-repair-loss-box", "0.020",
            "--dqa-server-anchor", "0.62",
            "--dqa-min-server-alpha", "0.56",
            "--dqa-residual-blend", "0.06",
            "--late-dqa-server-anchor", "0.40",
            "--late-dqa-min-server-alpha", "0.34",
            "--late-dqa-residual-blend", "0.05",
            "--curriculum-start-round", "21",
            "--expert-keep-fraction", "0.68",
            "--expert-max-class-fraction", "0.27",
            "--actual-max-class-fraction", "0.38",
            "--late-expert-keep-fraction", "0.82",
            "--late-expert-max-class-fraction", "0.32",
            "--late-actual-max-class-fraction", "0.45",
            "--min-views", "2",
            "--min-models", "0",
            "--min-score", "0.29",
            "--min-stability", "0.72",
            "--late-min-score", "0.23",
            "--late-min-stability", "0.65",
            "--max-boxes-per-image", "8",
            "--max-class-fraction", "0.50",
            "--min-class-keep", "300",
        ],
    ),
    base.Trial(
        name="37b_pseudo1152_strict_domain_soft_consolidated",
        hypothesis=(
            "A stricter 1152 pseudoGT pass tests whether small-object recall is the missing ingredient.  Updates are "
            "lower-LR top-2 domain experts with one source consolidation round so pseudoGT can teach recall without "
            "erasing the source-localized detector."
        ),
        args=[
            "--imgsz", "640",
            "--pseudo-imgsz", "1152",
            "--post-dqa-repair-rounds", "1",
            "--post-dqa-repair-train-scope", "neck_head",
            "--num-experts", "4",
            "--top-k", "2",
            "--router-temperature", "1.35",
            "--router-balance-weight", "0.010",
            "--router-entropy-weight", "0.0012",
            "--router-specialization-map", "domain4",
            "--router-specialization-weight", "0.055",
            "--router-specialization-max-weight", "0.040",
            "--router-specialization-min-quality", "0.62",
            "--router-specialization-min-boxes", "850",
            "--dqa-moe-expert-blend", "0.15",
            "--dqa-moe-router-blend", "0.08",
            "--phase1-train-scope", "neck_head",
            "--phase1-repair-train-scope", "neck_head",
            "--phase1-client-lr", "0.00020",
            "--phase1-source-repeat", "3",
            "--phase1-pseudo-repeat", "1",
            "--phase1-loss-box", "0.0010",
            "--phase2-train-scope", "all",
            "--phase2-repair-train-scope", "all",
            "--phase2-client-lr", "0.000030",
            "--phase2-source-repeat", "2",
            "--phase2-pseudo-repeat", "1",
            "--phase2-loss-box", "0.00012",
            "--server-repair-lr", "0.00018",
            "--server-repair-loss-box", "0.018",
            "--dqa-server-anchor", "0.68",
            "--dqa-min-server-alpha", "0.62",
            "--dqa-residual-blend", "0.04",
            "--late-dqa-server-anchor", "0.45",
            "--late-dqa-min-server-alpha", "0.38",
            "--late-dqa-residual-blend", "0.04",
            "--curriculum-start-round", "21",
            "--expert-keep-fraction", "0.64",
            "--expert-max-class-fraction", "0.26",
            "--actual-max-class-fraction", "0.36",
            "--late-expert-keep-fraction", "0.80",
            "--late-expert-max-class-fraction", "0.30",
            "--late-actual-max-class-fraction", "0.42",
            "--min-views", "2",
            "--min-models", "0",
            "--min-score", "0.32",
            "--min-stability", "0.76",
            "--late-min-score", "0.25",
            "--late-min-stability", "0.68",
            "--max-boxes-per-image", "7",
            "--max-class-fraction", "0.48",
            "--min-class-keep", "300",
        ],
    ),
]


def main(argv: list[str] | None = None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
