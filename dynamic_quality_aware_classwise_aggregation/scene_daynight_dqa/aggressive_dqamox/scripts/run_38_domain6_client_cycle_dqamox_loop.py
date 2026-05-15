#!/usr/bin/env python3
"""Run domain-6/client-cycle DQA-MoX learning loops.

37 showed that higher-resolution self pseudoGT can improve the source-style
server validation, but the paper-protocol target mAP stays near warmup.  This
loop changes the mixing axis: instead of folding city/residential domains into
four experts, each scene/day-night client owns one explicit expert.  The first
probe covers all six clients in three FedMoX-style rounds with two online
clients per round.
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


base.DEFAULT_OUTPUT = AGG_ROOT / "output" / "38_domain6_client_cycle_dqamox_loop"
base.SUMMARY_CSV = AGG_ROOT / "reports" / "38_domain6_client_cycle_dqamox_loop_summary.csv"
base.RUN_LABEL = "38"
base.RUN_DESCRIPTION = "domain-6 client-cycle DQA-MoX learning loop"
base.RUN_PAPER_NOTES = (
    "FedMoX sparse/soft expert routing; FedMoE-DA domain-aware aggregation; "
    "FedJETs specialist subsets; MixPL pseudo-label incompleteness; SAHI high-resolution recall."
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
        name="38a_domain6_cycle_highres_balanced_experts",
        hypothesis=(
            "Six explicit domain/client experts should stop citystreet and residential clients from being folded into "
            "the same expert.  Probe3 uses seed 350069 so all six clients are seen once under the same 33.3% online "
            "client ratio, then DQA moves each client update mainly into its own expert."
        ),
        args=[
            "--client-sampling-seed", "350069",
            "--imgsz", "640",
            "--pseudo-imgsz", "960",
            "--num-experts", "6",
            "--top-k", "1",
            "--router-temperature", "0.95",
            "--router-balance-weight", "0.009",
            "--router-entropy-weight", "0.0006",
            "--router-specialization-map", "domain6",
            "--router-specialization-weight", "0.110",
            "--router-specialization-max-weight", "0.080",
            "--router-specialization-min-quality", "0.60",
            "--router-specialization-min-boxes", "700",
            "--dqa-moe-expert-blend", "0.50",
            "--dqa-moe-router-blend", "0.20",
            "--phase1-train-scope", "neck_head",
            "--phase1-repair-train-scope", "neck_head",
            "--phase1-client-lr", "0.00024",
            "--phase1-source-repeat", "2",
            "--phase1-pseudo-repeat", "2",
            "--phase1-loss-box", "0.0012",
            "--phase2-train-scope", "all",
            "--phase2-repair-train-scope", "all",
            "--phase2-client-lr", "0.000030",
            "--phase2-source-repeat", "1",
            "--phase2-pseudo-repeat", "1",
            "--phase2-loss-box", "0.00010",
            "--server-repair-lr", "0.00018",
            "--server-repair-loss-box", "0.014",
            "--dqa-server-anchor", "0.62",
            "--dqa-min-server-alpha", "0.56",
            "--dqa-residual-blend", "0.04",
            "--late-dqa-server-anchor", "0.40",
            "--late-dqa-min-server-alpha", "0.34",
            "--late-dqa-residual-blend", "0.04",
            "--curriculum-start-round", "25",
            "--expert-keep-fraction", "0.66",
            "--expert-max-class-fraction", "0.26",
            "--actual-max-class-fraction", "0.37",
            "--late-expert-keep-fraction", "0.80",
            "--late-expert-max-class-fraction", "0.31",
            "--late-actual-max-class-fraction", "0.43",
            "--min-views", "2",
            "--min-models", "0",
            "--min-score", "0.30",
            "--min-stability", "0.72",
            "--late-min-score", "0.24",
            "--late-min-stability", "0.65",
            "--max-boxes-per-image", "7",
            "--max-class-fraction", "0.48",
            "--min-class-keep", "300",
        ],
    ),
    base.Trial(
        name="38b_domain6_cycle_strict_source_guarded",
        hypothesis=(
            "If 38a drifts, use the same six-expert client cycle but make pseudoGT stricter and keep the source anchor "
            "heavier.  This asks whether domain specialization helps only when localization remains source-guarded."
        ),
        args=[
            "--client-sampling-seed", "350069",
            "--imgsz", "640",
            "--pseudo-imgsz", "1152",
            "--post-dqa-repair-rounds", "1",
            "--post-dqa-repair-train-scope", "neck_head",
            "--num-experts", "6",
            "--top-k", "1",
            "--router-temperature", "1.05",
            "--router-balance-weight", "0.007",
            "--router-entropy-weight", "0.0008",
            "--router-specialization-map", "domain6",
            "--router-specialization-weight", "0.095",
            "--router-specialization-max-weight", "0.065",
            "--router-specialization-min-quality", "0.64",
            "--router-specialization-min-boxes", "800",
            "--dqa-moe-expert-blend", "0.34",
            "--dqa-moe-router-blend", "0.14",
            "--phase1-train-scope", "neck_head",
            "--phase1-repair-train-scope", "neck_head",
            "--phase1-client-lr", "0.00018",
            "--phase1-source-repeat", "4",
            "--phase1-pseudo-repeat", "1",
            "--phase1-loss-box", "0.00075",
            "--phase2-train-scope", "all",
            "--phase2-repair-train-scope", "all",
            "--phase2-client-lr", "0.000026",
            "--phase2-source-repeat", "2",
            "--phase2-pseudo-repeat", "1",
            "--phase2-loss-box", "0.00008",
            "--server-repair-lr", "0.00015",
            "--server-repair-loss-box", "0.012",
            "--dqa-server-anchor", "0.72",
            "--dqa-min-server-alpha", "0.66",
            "--dqa-residual-blend", "0.03",
            "--late-dqa-server-anchor", "0.48",
            "--late-dqa-min-server-alpha", "0.42",
            "--late-dqa-residual-blend", "0.03",
            "--curriculum-start-round", "25",
            "--expert-keep-fraction", "0.60",
            "--expert-max-class-fraction", "0.25",
            "--actual-max-class-fraction", "0.35",
            "--late-expert-keep-fraction", "0.78",
            "--late-expert-max-class-fraction", "0.30",
            "--late-actual-max-class-fraction", "0.42",
            "--min-views", "2",
            "--min-models", "0",
            "--min-score", "0.34",
            "--min-stability", "0.78",
            "--late-min-score", "0.26",
            "--late-min-stability", "0.70",
            "--max-boxes-per-image", "6",
            "--max-class-fraction", "0.46",
            "--min-class-keep", "260",
        ],
    ),
]


def main(argv: list[str] | None = None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
