#!/usr/bin/env python3
"""Collect per-client class-wise pseudo-label statistics for DQA-CWA.

The collector accepts pseudo-label dumps in common YOLO/EfficientTeacher-like
text formats and emits the JSON consumed by run_dqa_cwa_fedsto.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def _to_float(token: object) -> float | None:
    try:
        return float(token)
    except (TypeError, ValueError):
        return None


def _parse_text_line(line: str, default_confidence: float) -> tuple[int, float] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.replace(",", " ").split()
    numbers = [_to_float(part) for part in parts]

    if len(parts) >= 7 and numbers[0] is not None and numbers[1] is not None:
        return int(numbers[1]), float(numbers[6])
    if len(parts) >= 7 and numbers[0] is None and numbers[1] is not None:
        return int(numbers[1]), float(numbers[6])
    if len(parts) >= 6 and numbers[0] is not None:
        confidence = numbers[5] if numbers[5] is not None else default_confidence
        return int(numbers[0]), float(confidence)
    if len(parts) >= 5 and numbers[0] is not None:
        return int(numbers[0]), default_confidence
    return None


def _iter_text_labels(path: Path, default_confidence: float) -> Iterable[tuple[int, float]]:
    for raw in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_text_line(raw, default_confidence)
        if parsed is not None:
            yield parsed


def _iter_json_labels(path: Path, default_confidence: float) -> Iterable[tuple[int, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if "labels" in raw:
            items = raw["labels"]
        elif "pseudo_labels" in raw:
            items = raw["pseudo_labels"]
        else:
            items = raw.values()
    elif isinstance(raw, list):
        items = raw
    else:
        return

    for item in items:
        if isinstance(item, dict):
            cls = item.get("class", item.get("cls", item.get("category_id")))
            conf = item.get("confidence", item.get("conf", item.get("score", default_confidence)))
            if cls is not None:
                yield int(cls), float(conf)
        elif isinstance(item, (list, tuple)):
            if len(item) >= 7:
                yield int(item[1]), float(item[6])
            elif len(item) >= 6:
                yield int(item[0]), float(item[5])


def _label_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = []
    for suffix in ("*.txt", "*.csv", "*.json"):
        files.extend(path.rglob(suffix))
    return sorted(files)


def collect_client(path: Path, client_id: str, num_classes: int, default_confidence: float) -> dict:
    counts = [0.0] * num_classes
    confidence_sums = [0.0] * num_classes

    for file_path in _label_files(path):
        if file_path.suffix.lower() == ".json":
            labels = _iter_json_labels(file_path, default_confidence)
        else:
            labels = _iter_text_labels(file_path, default_confidence)
        for cls, confidence in labels:
            if 0 <= cls < num_classes:
                clipped = min(max(float(confidence), 0.0), 1.0)
                counts[cls] += 1.0
                confidence_sums[cls] += clipped

    mean_confidences = [
        confidence_sums[idx] / counts[idx] if counts[idx] > 0 else 0.0
        for idx in range(num_classes)
    ]
    return {
        "id": client_id,
        "source": str(path),
        "counts": counts,
        "confidence_sums": confidence_sums,
        "mean_confidences": mean_confidences,
    }


def _parse_client_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    client_id, raw_path = value.split("=", 1)
    return client_id, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", action="append", required=True, help="client_id=/path/to/pseudo_labels")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--default-confidence", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clients = []
    for raw in args.client:
        client_id, path = _parse_client_arg(raw)
        clients.append(collect_client(path, client_id, args.num_classes, args.default_confidence))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"clients": clients}, indent=2), encoding="utf-8")
    print(f"Wrote DQA-CWA stats for {len(clients)} clients to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

