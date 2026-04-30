#!/usr/bin/env python3
"""Infer class-wise pseudoGT thresholds from a trained DQA policy model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from threshold_policy import load_bundle, predict_policy, read_client_stats, summarize_latest_decision


def parse_args() -> argparse.Namespace:
    policy_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=policy_dir / "artifacts" / "dqa05_threshold_policy.joblib",
        help="Trained threshold policy joblib bundle.",
    )
    parser.add_argument(
        "--stats-root",
        type=Path,
        default=policy_dir.parent / "stats_dqa05_scene_class_profile_5h",
        help="Directory containing phase2_round*_client*.json stats.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=policy_dir / "artifacts" / "latest_policy_decision.json",
        help="Where to write the JSON decision.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.model.resolve())
    class_names = bundle.get("class_names")
    data = read_client_stats(args.stats_root.resolve(), class_names=class_names)
    predictions = predict_policy(bundle, data)
    decision = summarize_latest_decision(predictions, class_names=class_names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
