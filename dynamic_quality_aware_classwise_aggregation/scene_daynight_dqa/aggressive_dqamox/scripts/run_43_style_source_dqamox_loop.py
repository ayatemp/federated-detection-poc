#!/usr/bin/env python3
"""Run source-GT target-style DQA-MoX learning loops.

42 showed that even selective backbone/MoE training is pulled down when target
pseudo boxes are treated as supervised labels.  43 keeps the self-only and
FedMoX-like client loop, but moves the target signal into appearance: each
selected client supplies Fourier style statistics, source images keep their true
source GT boxes, and pseudoGT is used only as DQA/router evidence.
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


base.DEFAULT_OUTPUT = AGG_ROOT / "output" / "43_style_source_dqamox_loop"
base.SUMMARY_CSV = AGG_ROOT / "reports" / "43_style_source_dqamox_loop_summary.csv"
base.RUN_LABEL = "43"
base.RUN_DESCRIPTION = "source-GT target-style DQA-MoX learning loop"
base.RUN_PAPER_NOTES = (
    "FedMoX soft domain/expert routing; FedSTO source-anchored selective refinement; "
    "Fourier Domain Adaptation for label-preserving target appearance transfer; "
    "FedBN/DSBN-style idea that client appearance statistics should not be blindly averaged."
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
        name="43a_fda_sourcegt_domain6_all",
        hypothesis=(
            "Use client target images only as style donors.  Source images are FDA-stylized per selected client and "
            "trained with source GT, while pseudoGT only gates DQA/router specialization.  This tests whether the "
            "missing FedMoX ingredient is target appearance learning rather than pseudo-label supervision."
        ),
        args=[
            "--client-sampling-seed", "430069",
            "--imgsz", "640",
            "--pseudo-imgsz", "1152",
            "--num-experts", "6",
            "--top-k", "2",
            "--router-temperature", "1.28",
            "--router-balance-weight", "0.018",
            "--router-entropy-weight", "0.0020",
            "--router-specialization-map", "domain6",
            "--router-specialization-weight", "0.125",
            "--router-specialization-max-weight", "0.085",
            "--router-specialization-min-quality", "0.62",
            "--router-specialization-min-boxes", "220",
            "--dqa-moe-expert-blend", "0.18",
            "--dqa-moe-router-blend", "0.30",
            "--dqa-classwise-blend", "0.22",
            "--dqa-temperature", "0.86",
            "--dqa-client-balance-target", "max",
            "--dqa-client-balance-max-scale", "2.2",
            "--phase1-train-scope", "all",
            "--phase1-repair-train-scope", "all",
            "--phase1-client-lr", "0.000050",
            "--phase1-source-repeat", "1",
            "--phase1-pseudo-repeat", "0",
            "--phase1-loss-box", "0.035",
            "--client-loss-cls", "0.50",
            "--client-loss-obj", "1.00",
            "--phase2-train-scope", "all",
            "--phase2-repair-train-scope", "all",
            "--phase2-client-lr", "0.000018",
            "--phase2-source-repeat", "1",
            "--phase2-pseudo-repeat", "0",
            "--phase2-loss-box", "0.020",
            "--server-repair-lr", "0.000028",
            "--server-repair-loss-box", "0.010",
            "--server-repair-loss-cls", "0.45",
            "--server-repair-loss-obj", "0.90",
            "--dqa-server-anchor", "0.88",
            "--dqa-min-server-alpha", "0.82",
            "--dqa-residual-blend", "0.000",
            "--late-dqa-server-anchor", "0.72",
            "--late-dqa-min-server-alpha", "0.64",
            "--late-dqa-residual-blend", "0.000",
            "--curriculum-start-round", "25",
            "--expert-keep-fraction", "0.56",
            "--expert-max-class-fraction", "0.18",
            "--actual-max-class-fraction", "0.25",
            "--late-expert-keep-fraction", "0.66",
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
            "--client-mosaic", "0.25",
            "--client-scale", "0.12",
            "--client-hsv-s", "0.08",
            "--client-hsv-v", "0.08",
            "--style-source-repeat", "1",
            "--style-source-limit", "1800",
            "--style-beta", "0.012",
            "--style-imgsz", "640",
            "--style-seed", "430001",
        ],
    ),
    base.Trial(
        name="43b_fda_sourcegt_backbone_moe_guarded",
        hypothesis=(
            "Same label-preserving target-style signal, but with the ordinary detector head guarded.  The backbone "
            "and MoE router/expert slots learn client appearance; the head remains mostly source calibrated."
        ),
        args=[
            "--client-sampling-seed", "430069",
            "--imgsz", "640",
            "--pseudo-imgsz", "1152",
            "--num-experts", "6",
            "--top-k", "2",
            "--router-temperature", "1.42",
            "--router-balance-weight", "0.018",
            "--router-entropy-weight", "0.0025",
            "--router-specialization-map", "domain6",
            "--router-specialization-weight", "0.105",
            "--router-specialization-max-weight", "0.075",
            "--router-specialization-min-quality", "0.62",
            "--router-specialization-min-boxes", "220",
            "--dqa-moe-expert-blend", "0.16",
            "--dqa-moe-router-blend", "0.32",
            "--dqa-classwise-blend", "0.18",
            "--dqa-temperature", "0.92",
            "--dqa-client-balance-target", "max",
            "--dqa-client-balance-max-scale", "2.0",
            "--phase1-train-scope", "backbone_moe_head",
            "--phase1-repair-train-scope", "backbone_moe_head",
            "--phase1-client-lr", "0.000040",
            "--phase1-source-repeat", "1",
            "--phase1-pseudo-repeat", "0",
            "--phase1-loss-box", "0.018",
            "--client-loss-cls", "0.50",
            "--client-loss-obj", "1.00",
            "--phase2-train-scope", "backbone_moe_head",
            "--phase2-repair-train-scope", "backbone_moe_head",
            "--phase2-client-lr", "0.000014",
            "--phase2-source-repeat", "1",
            "--phase2-pseudo-repeat", "0",
            "--phase2-loss-box", "0.010",
            "--server-repair-lr", "0.000022",
            "--server-repair-loss-box", "0.006",
            "--server-repair-loss-cls", "0.45",
            "--server-repair-loss-obj", "0.90",
            "--dqa-server-anchor", "0.93",
            "--dqa-min-server-alpha", "0.86",
            "--dqa-residual-blend", "0.000",
            "--late-dqa-server-anchor", "0.78",
            "--late-dqa-min-server-alpha", "0.70",
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
            "--client-mosaic", "0.20",
            "--client-scale", "0.10",
            "--client-hsv-s", "0.06",
            "--client-hsv-v", "0.06",
            "--style-source-repeat", "1",
            "--style-source-limit", "2200",
            "--style-beta", "0.009",
            "--style-imgsz", "640",
            "--style-seed", "430101",
        ],
    ),
]


def main(argv: list[str] | None = None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
