#!/usr/bin/env python3
"""Run loss-factorized DQA-MoX learning loops.

40 showed a useful but damaging pattern: self pseudoGT raised recall a little,
while paper-protocol localization/AP quality dropped.  41 keeps the FedMoX-like
client-sampled MoE loop, but changes the learning role of pseudoGT.  PseudoGT is
treated mainly as an objectness/class/router/domain signal; source GT and strong
DQA anchoring protect geometry.
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


base.DEFAULT_OUTPUT = AGG_ROOT / "output" / "41_loss_factorized_dqamox_loop"
base.SUMMARY_CSV = AGG_ROOT / "reports" / "41_loss_factorized_dqamox_loop_summary.csv"
base.RUN_LABEL = "41"
base.RUN_DESCRIPTION = "loss-factorized DQA-MoX learning loop"
base.RUN_PAPER_NOTES = (
    "FedMoX sparse spatial routing and Soft-Mixture; FedSTO local EMA pseudo labels; "
    "certainty-aware pseudo labels for classification/localization quality; "
    "MixPL warning that missed pseudo boxes bias SSOD; FedMoE personalized expert recommendation."
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
        name="41a_domain6_clsobj_router_signal",
        hypothesis=(
            "40 increased recall but damaged geometry.  This trial makes pseudoGT a class/objectness/router signal: "
            "box loss is nearly off during client pseudo training, source repeats are high, DQA stays close to the "
            "server model, and six domain experts are still explicitly specialized."
        ),
        args=[
            "--client-sampling-seed", "410069",
            "--imgsz", "640",
            "--pseudo-imgsz", "1152",
            "--num-experts", "6",
            "--top-k", "2",
            "--router-temperature", "1.35",
            "--router-balance-weight", "0.014",
            "--router-entropy-weight", "0.0015",
            "--router-specialization-map", "domain6",
            "--router-specialization-weight", "0.095",
            "--router-specialization-max-weight", "0.065",
            "--router-specialization-min-quality", "0.54",
            "--router-specialization-min-boxes", "260",
            "--dqa-moe-expert-blend", "0.26",
            "--dqa-moe-router-blend", "0.18",
            "--dqa-classwise-blend", "0.34",
            "--dqa-temperature", "0.72",
            "--dqa-client-balance-target", "max",
            "--dqa-client-balance-max-scale", "3.2",
            "--phase1-train-scope", "neck_head",
            "--phase1-repair-train-scope", "neck_head",
            "--phase1-client-lr", "0.00013",
            "--phase1-source-repeat", "5",
            "--phase1-pseudo-repeat", "1",
            "--phase1-loss-box", "0.000025",
            "--client-loss-cls", "0.85",
            "--client-loss-obj", "1.35",
            "--phase2-train-scope", "all",
            "--phase2-repair-train-scope", "neck_head",
            "--phase2-client-lr", "0.000020",
            "--phase2-source-repeat", "3",
            "--phase2-pseudo-repeat", "1",
            "--phase2-loss-box", "0.000015",
            "--server-repair-lr", "0.00008",
            "--server-repair-loss-box", "0.006",
            "--server-repair-loss-cls", "0.45",
            "--server-repair-loss-obj", "0.90",
            "--dqa-server-anchor", "0.84",
            "--dqa-min-server-alpha", "0.78",
            "--dqa-residual-blend", "0.000",
            "--late-dqa-server-anchor", "0.66",
            "--late-dqa-min-server-alpha", "0.58",
            "--late-dqa-residual-blend", "0.000",
            "--curriculum-start-round", "25",
            "--expert-keep-fraction", "0.62",
            "--expert-max-class-fraction", "0.20",
            "--actual-max-class-fraction", "0.28",
            "--late-expert-keep-fraction", "0.72",
            "--late-expert-max-class-fraction", "0.24",
            "--late-actual-max-class-fraction", "0.34",
            "--min-views", "2",
            "--min-models", "0",
            "--min-score", "0.34",
            "--min-stability", "0.72",
            "--late-min-score", "0.28",
            "--late-min-stability", "0.66",
            "--max-boxes-per-image", "7",
            "--max-class-fraction", "0.30",
            "--min-class-keep", "80",
            "--client-mixup", "0.00",
            "--client-mosaic", "0.70",
            "--client-scale", "0.22",
            "--client-hsv-s", "0.24",
            "--client-hsv-v", "0.18",
        ],
    ),
    base.Trial(
        name="41b_moehead_clsobj_specialists",
        hypothesis=(
            "If neck/head updates are still too destructive, specialize only the MoE head/router.  PseudoGT gets a "
            "strong class/objectness role, but shared representation movement is constrained so the warmup geometry "
            "survives while experts learn domain/client preferences."
        ),
        args=[
            "--client-sampling-seed", "410069",
            "--imgsz", "640",
            "--pseudo-imgsz", "1280",
            "--num-experts", "6",
            "--top-k", "1",
            "--router-temperature", "1.05",
            "--router-balance-weight", "0.018",
            "--router-entropy-weight", "0.0006",
            "--router-specialization-map", "domain6",
            "--router-specialization-weight", "0.140",
            "--router-specialization-max-weight", "0.095",
            "--router-specialization-min-quality", "0.58",
            "--router-specialization-min-boxes", "220",
            "--dqa-moe-expert-blend", "0.48",
            "--dqa-moe-router-blend", "0.32",
            "--dqa-classwise-blend", "0.28",
            "--dqa-temperature", "0.64",
            "--dqa-client-balance-target", "max",
            "--dqa-client-balance-max-scale", "3.8",
            "--phase1-train-scope", "moe_head",
            "--phase1-repair-train-scope", "moe_head",
            "--phase1-client-lr", "0.00042",
            "--phase1-source-repeat", "4",
            "--phase1-pseudo-repeat", "2",
            "--phase1-loss-box", "0.000020",
            "--client-loss-cls", "1.00",
            "--client-loss-obj", "1.55",
            "--phase2-train-scope", "neck_head",
            "--phase2-repair-train-scope", "moe_head",
            "--phase2-client-lr", "0.000040",
            "--phase2-source-repeat", "4",
            "--phase2-pseudo-repeat", "1",
            "--phase2-loss-box", "0.000015",
            "--server-repair-lr", "0.00006",
            "--server-repair-loss-box", "0.004",
            "--server-repair-loss-cls", "0.45",
            "--server-repair-loss-obj", "0.90",
            "--dqa-server-anchor", "0.88",
            "--dqa-min-server-alpha", "0.82",
            "--dqa-residual-blend", "0.000",
            "--late-dqa-server-anchor", "0.72",
            "--late-dqa-min-server-alpha", "0.64",
            "--late-dqa-residual-blend", "0.000",
            "--curriculum-start-round", "25",
            "--expert-keep-fraction", "0.58",
            "--expert-max-class-fraction", "0.18",
            "--actual-max-class-fraction", "0.25",
            "--late-expert-keep-fraction", "0.70",
            "--late-expert-max-class-fraction", "0.23",
            "--late-actual-max-class-fraction", "0.32",
            "--min-views", "2",
            "--min-models", "0",
            "--min-score", "0.36",
            "--min-stability", "0.74",
            "--late-min-score", "0.30",
            "--late-min-stability", "0.68",
            "--max-boxes-per-image", "6",
            "--max-class-fraction", "0.28",
            "--min-class-keep", "70",
            "--client-mixup", "0.00",
            "--client-mosaic", "0.55",
            "--client-scale", "0.18",
            "--client-hsv-s", "0.20",
            "--client-hsv-v", "0.16",
        ],
    ),
]


def main(argv: list[str] | None = None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
