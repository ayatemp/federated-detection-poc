#!/usr/bin/env python3
"""Run class-coverage DQA-MoX learning loops.

38/39 indicate that explicit domain experts alone are not enough: the learned
experts still inherit incomplete, class-skewed pseudoGT.  40 keeps the
FedMoX-like client-sampled MoE training shape, but makes the DQA signal care
more about pseudo-label coverage: class caps, actual selected-pool balancing,
high-resolution self pseudoGT, MixPL-style client augmentation, and client-count
balanced DQA aggregation.
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


base.DEFAULT_OUTPUT = AGG_ROOT / "output" / "40_class_coverage_dqamox_loop"
base.SUMMARY_CSV = AGG_ROOT / "reports" / "40_class_coverage_dqamox_loop_summary.csv"
base.RUN_LABEL = "40"
base.RUN_DESCRIPTION = "class-coverage DQA-MoX learning loop"
base.RUN_PAPER_NOTES = (
    "FedMoX sparse spatial routing and Soft-Mixture; FedSTO local EMA pseudo labels; "
    "Rethinking Pseudo Labels for classwise threshold/reweighting; MixPL for missed/tail pseudo-label imbalance; "
    "FedMoE personalized sub-MoE aggregation."
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
        name="40a_domain6_coverage_mixpl",
        hypothesis=(
            "The failure mode looks like incomplete pseudoGT being averaged into every domain expert.  This trial "
            "keeps six domain/client experts, but lowers the pseudoGT gate just enough to recover missed objects, "
            "then uses strict selected-pool class caps, MixPL-style mixup/mosaic, and max-client DQA balancing so "
            "night/rare-class clients are not drowned by easy vehicle-heavy clients."
        ),
        args=[
            "--client-sampling-seed", "350069",
            "--imgsz", "640",
            "--pseudo-imgsz", "1152",
            "--post-dqa-repair-rounds", "1",
            "--post-dqa-repair-train-scope", "neck_head",
            "--num-experts", "6",
            "--top-k", "2",
            "--router-temperature", "1.10",
            "--router-balance-weight", "0.010",
            "--router-entropy-weight", "0.0008",
            "--router-specialization-map", "domain6",
            "--router-specialization-weight", "0.130",
            "--router-specialization-max-weight", "0.105",
            "--router-specialization-min-quality", "0.50",
            "--router-specialization-min-boxes", "240",
            "--dqa-moe-expert-blend", "0.48",
            "--dqa-moe-router-blend", "0.22",
            "--dqa-classwise-blend", "0.48",
            "--dqa-temperature", "0.55",
            "--dqa-client-balance-target", "max",
            "--dqa-client-balance-max-scale", "5.0",
            "--phase1-train-scope", "neck_head",
            "--phase1-repair-train-scope", "neck_head",
            "--phase1-client-lr", "0.00020",
            "--phase1-source-repeat", "3",
            "--phase1-pseudo-repeat", "2",
            "--phase1-loss-box", "0.00055",
            "--phase2-train-scope", "all",
            "--phase2-repair-train-scope", "all",
            "--phase2-client-lr", "0.000028",
            "--phase2-source-repeat", "2",
            "--phase2-pseudo-repeat", "1",
            "--phase2-loss-box", "0.00006",
            "--server-repair-lr", "0.00015",
            "--server-repair-loss-box", "0.011",
            "--dqa-server-anchor", "0.64",
            "--dqa-min-server-alpha", "0.56",
            "--dqa-residual-blend", "0.025",
            "--late-dqa-server-anchor", "0.42",
            "--late-dqa-min-server-alpha", "0.36",
            "--late-dqa-residual-blend", "0.025",
            "--curriculum-start-round", "25",
            "--expert-keep-fraction", "0.86",
            "--expert-max-class-fraction", "0.20",
            "--actual-max-class-fraction", "0.25",
            "--late-expert-keep-fraction", "0.88",
            "--late-expert-max-class-fraction", "0.24",
            "--late-actual-max-class-fraction", "0.31",
            "--min-views", "2",
            "--min-models", "0",
            "--min-score", "0.25",
            "--min-stability", "0.66",
            "--late-min-score", "0.20",
            "--late-min-stability", "0.58",
            "--max-boxes-per-image", "10",
            "--max-class-fraction", "0.34",
            "--min-class-keep", "120",
            "--client-mixup", "0.08",
            "--client-mosaic", "1.0",
            "--client-scale", "0.42",
            "--client-hsv-s", "0.42",
            "--client-hsv-v", "0.28",
        ],
    ),
    base.Trial(
        name="40b_domain6_night_recall_guarded",
        hypothesis=(
            "If 40a is still too noisy, push recall mostly through night/domain coverage while guarding source localization.  "
            "The router specialization is softer, source repeat is higher, and pseudo box loss is almost off; pseudoGT teaches "
            "objectness/class/router coverage more than geometry."
        ),
        args=[
            "--client-sampling-seed", "350069",
            "--imgsz", "640",
            "--pseudo-imgsz", "1280",
            "--post-dqa-repair-rounds", "1",
            "--post-dqa-repair-train-scope", "neck_head",
            "--num-experts", "6",
            "--top-k", "2",
            "--router-temperature", "1.25",
            "--router-balance-weight", "0.012",
            "--router-entropy-weight", "0.0012",
            "--router-specialization-map", "domain6",
            "--router-specialization-weight", "0.105",
            "--router-specialization-max-weight", "0.075",
            "--router-specialization-min-quality", "0.46",
            "--router-specialization-min-boxes", "200",
            "--dqa-moe-expert-blend", "0.38",
            "--dqa-moe-router-blend", "0.16",
            "--dqa-classwise-blend", "0.42",
            "--dqa-temperature", "0.60",
            "--dqa-client-balance-target", "max",
            "--dqa-client-balance-max-scale", "4.5",
            "--phase1-train-scope", "neck_head",
            "--phase1-repair-train-scope", "neck_head",
            "--phase1-client-lr", "0.00016",
            "--phase1-source-repeat", "4",
            "--phase1-pseudo-repeat", "1",
            "--phase1-loss-box", "0.00012",
            "--phase2-train-scope", "all",
            "--phase2-repair-train-scope", "all",
            "--phase2-client-lr", "0.000024",
            "--phase2-source-repeat", "2",
            "--phase2-pseudo-repeat", "1",
            "--phase2-loss-box", "0.00004",
            "--server-repair-lr", "0.00014",
            "--server-repair-loss-box", "0.010",
            "--dqa-server-anchor", "0.70",
            "--dqa-min-server-alpha", "0.62",
            "--dqa-residual-blend", "0.020",
            "--late-dqa-server-anchor", "0.48",
            "--late-dqa-min-server-alpha", "0.42",
            "--late-dqa-residual-blend", "0.020",
            "--curriculum-start-round", "25",
            "--expert-keep-fraction", "0.80",
            "--expert-max-class-fraction", "0.22",
            "--actual-max-class-fraction", "0.28",
            "--late-expert-keep-fraction", "0.86",
            "--late-expert-max-class-fraction", "0.26",
            "--late-actual-max-class-fraction", "0.34",
            "--min-views", "2",
            "--min-models", "0",
            "--min-score", "0.22",
            "--min-stability", "0.60",
            "--late-min-score", "0.18",
            "--late-min-stability", "0.54",
            "--max-boxes-per-image", "12",
            "--max-class-fraction", "0.38",
            "--min-class-keep", "160",
            "--client-mixup", "0.04",
            "--client-mosaic", "1.0",
            "--client-scale", "0.36",
            "--client-hsv-s", "0.38",
            "--client-hsv-v", "0.32",
        ],
    ),
]


def main(argv: list[str] | None = None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
