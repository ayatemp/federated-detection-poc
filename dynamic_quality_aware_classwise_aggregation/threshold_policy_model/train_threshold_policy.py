#!/usr/bin/env python3
"""Train the DQA pseudoGT threshold policy from DQA05 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from threshold_policy import PolicyPaths, run_training


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats-root",
        type=Path,
        default=root / "stats_dqa05_scene_class_profile_5h",
        help="DQA05 pseudo-label stats directory.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "efficientteacher_dqa05_scene_class_profile_5h",
        help="DQA05 workspace containing runs/*/results.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
        help="Directory for the trained model and reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_training(
        PolicyPaths(
            stats_root=args.stats_root.resolve(),
            run_root=args.run_root.resolve(),
            output_dir=args.output_dir.resolve(),
        )
    )
    printable = {key: value for key, value in report.items() if key != "latest_decision"}
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
