"""Report the current FedMox full-paper reproduction score."""

from __future__ import annotations

import argparse


AREAS = [
    ("PSSFL protocol and aggregation equations", 23, 25),
    ("MoE mechanism", 14, 20),
    ("Paper constants, splits, and reported targets", 18, 20),
    ("Detector and SSL training integration", 3, 25),
    ("Empirical result reproduction", 0, 10),
]

BLOCKERS = [
    "MMDetection Faster R-CNN integration",
    "ViT-Adapter-Small with DINOv2 and MS-COCO adapter pretraining",
    "Soft Teacher training loop",
    "COALA-style federated simulation",
    "Exact dataset materialization and split seeds",
    "Executed 50-round runs matching the reported mAP tables",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-100", action="store_true", help="exit nonzero unless the score is 100")
    args = parser.parse_args()

    score = sum(item[1] for item in AREAS)
    total = sum(item[2] for item in AREAS)
    print(f"FedMox full-paper reproduction score: {score}/{total}")
    for name, points, max_points in AREAS:
        print(f"- {name}: {points}/{max_points}")

    if score < total:
        print("\nBlocking gaps:")
        for blocker in BLOCKERS:
            print(f"- {blocker}")

    if args.require_100 and score < total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
