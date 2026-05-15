#!/usr/bin/env python3
"""Run DQA-gated BN-stat MoE learning loops.

43 showed that simply teaching the model target appearance through FDA-stylized
source GT still hurts the target paper protocol.  44 keeps the learning signal
self-only and label-safe, but treats feature statistics as the dynamic part:
clients learn BN statistics/affine parameters plus explicit MoE router/expert
slots, then DQA chooses how much of those client/domain statistics enter the
server-anchored aggregate.
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


base.DEFAULT_OUTPUT = AGG_ROOT / "output" / "44_dqa_bn_stat_moe_loop"
base.SUMMARY_CSV = AGG_ROOT / "reports" / "44_dqa_bn_stat_moe_loop_summary.csv"
base.RUN_LABEL = "44"
base.RUN_DESCRIPTION = "DQA-gated BN-stat specialist MoE learning loop"
base.RUN_PAPER_NOTES = (
    "FedMoX/FedMix explicit expert routing for heterogeneous clients; "
    "FedBN and Domain-Specific BN showing feature statistics should be specialized under feature shift; "
    "FDA used only as label-preserving target-appearance exposure."
)

base.STAGES = [
    base.Stage("probe3", phase1_rounds=3, phase2_rounds=0, min_best_map50=0.462, min_gain_vs_warmup=0.001),
    base.Stage("probe6", phase1_rounds=6, phase2_rounds=0, min_best_map50=0.472, min_gain_vs_warmup=0.008),
    base.Stage("probe12", phase1_rounds=12, phase2_rounds=0, min_best_map50=0.490, min_gain_vs_warmup=0.024),
    base.Stage("phase1_24", phase1_rounds=24, phase2_rounds=0, min_best_map50=0.515, min_gain_vs_warmup=0.050),
    base.Stage("full24_26", phase1_rounds=24, phase2_rounds=26, min_best_map50=0.550, min_gain_vs_warmup=0.085),
]

base.TRIALS = [
    base.Trial(
        name="44a_dqa_bn_bank_domain6_style",
        hypothesis=(
            "FedBN-style dynamic statistic specialists: selected clients train only BN parameters/statistics and "
            "MoE router/expert slots on source GT plus target-styled source images.  Aggregation keeps the detector "
            "server anchored but injects DQA-weighted BN statistics, so target appearance is learned without using "
            "pseudo boxes as labels or moving the ordinary head."
        ),
        args=[
            "--client-sampling-seed", "440069",
            "--imgsz", "640",
            "--pseudo-imgsz", "1152",
            "--num-experts", "6",
            "--top-k", "2",
            "--router-temperature", "1.34",
            "--router-balance-weight", "0.020",
            "--router-entropy-weight", "0.0025",
            "--router-specialization-map", "domain6",
            "--router-specialization-weight", "0.115",
            "--router-specialization-max-weight", "0.080",
            "--router-specialization-min-quality", "0.62",
            "--router-specialization-min-boxes", "220",
            "--dqa-bn-blend", "0.32",
            "--dqa-moe-expert-blend", "0.13",
            "--dqa-moe-router-blend", "0.26",
            "--dqa-classwise-blend", "0.04",
            "--dqa-temperature", "0.90",
            "--dqa-client-balance-target", "max",
            "--dqa-client-balance-max-scale", "2.0",
            "--phase1-train-scope", "bn_moe_head",
            "--phase1-repair-train-scope", "bn_moe_head",
            "--phase1-client-lr", "0.00012",
            "--phase1-source-repeat", "1",
            "--phase1-pseudo-repeat", "0",
            "--phase1-loss-box", "0.006",
            "--client-loss-cls", "0.45",
            "--client-loss-obj", "0.90",
            "--phase2-train-scope", "bn_moe_head",
            "--phase2-repair-train-scope", "bn_moe_head",
            "--phase2-client-lr", "0.000045",
            "--phase2-source-repeat", "1",
            "--phase2-pseudo-repeat", "0",
            "--phase2-loss-box", "0.004",
            "--server-repair-epochs", "0",
            "--dqa-server-anchor", "0.96",
            "--dqa-min-server-alpha", "0.90",
            "--dqa-residual-blend", "0.000",
            "--late-dqa-server-anchor", "0.88",
            "--late-dqa-min-server-alpha", "0.82",
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
            "--client-mosaic", "0.10",
            "--client-scale", "0.06",
            "--client-hsv-s", "0.04",
            "--client-hsv-v", "0.04",
            "--style-source-repeat", "1",
            "--style-source-limit", "2200",
            "--style-beta", "0.006",
            "--style-imgsz", "640",
            "--style-seed", "440101",
        ],
    ),
    base.Trial(
        name="44b_dqa_bn_bank_tiny_style_strong_anchor",
        hypothesis=(
            "Same BN-stat specialist idea, but more conservative: weaker style transfer and smaller BN injection.  "
            "This tests whether 43 failed because the target-style perturbation was too strong, while still allowing "
            "DQA to specialize client/domain statistics and MoE routing."
        ),
        args=[
            "--client-sampling-seed", "440069",
            "--imgsz", "640",
            "--pseudo-imgsz", "1152",
            "--num-experts", "6",
            "--top-k", "2",
            "--router-temperature", "1.48",
            "--router-balance-weight", "0.018",
            "--router-entropy-weight", "0.0030",
            "--router-specialization-map", "domain6",
            "--router-specialization-weight", "0.095",
            "--router-specialization-max-weight", "0.065",
            "--router-specialization-min-quality", "0.64",
            "--router-specialization-min-boxes", "260",
            "--dqa-bn-blend", "0.18",
            "--dqa-moe-expert-blend", "0.10",
            "--dqa-moe-router-blend", "0.22",
            "--dqa-classwise-blend", "0.00",
            "--dqa-temperature", "1.00",
            "--dqa-client-balance-target", "max",
            "--dqa-client-balance-max-scale", "1.8",
            "--phase1-train-scope", "bn_moe_head",
            "--phase1-repair-train-scope", "bn_moe_head",
            "--phase1-client-lr", "0.00009",
            "--phase1-source-repeat", "2",
            "--phase1-pseudo-repeat", "0",
            "--phase1-loss-box", "0.004",
            "--client-loss-cls", "0.42",
            "--client-loss-obj", "0.85",
            "--phase2-train-scope", "bn_moe_head",
            "--phase2-repair-train-scope", "bn_moe_head",
            "--phase2-client-lr", "0.000035",
            "--phase2-source-repeat", "1",
            "--phase2-pseudo-repeat", "0",
            "--phase2-loss-box", "0.003",
            "--server-repair-epochs", "0",
            "--dqa-server-anchor", "1.05",
            "--dqa-min-server-alpha", "0.94",
            "--dqa-residual-blend", "0.000",
            "--late-dqa-server-anchor", "0.95",
            "--late-dqa-min-server-alpha", "0.88",
            "--late-dqa-residual-blend", "0.000",
            "--curriculum-start-round", "25",
            "--expert-keep-fraction", "0.50",
            "--expert-max-class-fraction", "0.16",
            "--actual-max-class-fraction", "0.24",
            "--late-expert-keep-fraction", "0.60",
            "--late-expert-max-class-fraction", "0.20",
            "--late-actual-max-class-fraction", "0.28",
            "--min-views", "2",
            "--min-models", "0",
            "--min-score", "0.40",
            "--min-stability", "0.78",
            "--late-min-score", "0.34",
            "--late-min-stability", "0.72",
            "--max-boxes-per-image", "4",
            "--max-class-fraction", "0.24",
            "--min-class-keep", "45",
            "--client-mixup", "0.00",
            "--client-mosaic", "0.05",
            "--client-scale", "0.04",
            "--client-hsv-s", "0.03",
            "--client-hsv-v", "0.03",
            "--style-source-repeat", "1",
            "--style-source-limit", "1800",
            "--style-beta", "0.0035",
            "--style-imgsz", "640",
            "--style-seed", "440201",
        ],
    ),
]


def main(argv: list[str] | None = None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
