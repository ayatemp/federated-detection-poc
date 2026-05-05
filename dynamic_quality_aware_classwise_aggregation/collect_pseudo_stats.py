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


def _box_localization_quality(x: float, y: float, w: float, h: float) -> float:
    if w <= 0.0 or h <= 0.0:
        return 0.0
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    overflow = max(0.0, -x1) + max(0.0, -y1) + max(0.0, x2 - 1.0) + max(0.0, y2 - 1.0)
    return min(max(1.0 - overflow, 0.0), 1.0)


def _quality_score(confidence: float, objectness: float, class_confidence: float, localization: float) -> float:
    return min(
        max(
            0.50 * confidence
            + 0.20 * objectness
            + 0.20 * class_confidence
            + 0.10 * localization,
            0.0,
        ),
        1.0,
    )


def _parse_text_line(line: str, default_confidence: float) -> tuple[int, float, float, float, float, float] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.replace(",", " ").split()
    numbers = [_to_float(part) for part in parts]

    if len(parts) >= 9 and numbers[0] is not None and numbers[1] is not None:
        confidence = float(numbers[6])
        objectness = float(numbers[7])
        class_confidence = float(numbers[8])
        localization = _box_localization_quality(float(numbers[2]), float(numbers[3]), float(numbers[4]), float(numbers[5]))
        return int(numbers[1]), confidence, objectness, class_confidence, localization, _quality_score(confidence, objectness, class_confidence, localization)
    if len(parts) >= 7 and numbers[0] is not None and numbers[1] is not None:
        confidence = float(numbers[6])
        localization = _box_localization_quality(float(numbers[2]), float(numbers[3]), float(numbers[4]), float(numbers[5]))
        return int(numbers[1]), confidence, confidence, confidence, localization, _quality_score(confidence, confidence, confidence, localization)
    if len(parts) >= 7 and numbers[0] is None and numbers[1] is not None:
        confidence = float(numbers[6])
        localization = _box_localization_quality(float(numbers[2]), float(numbers[3]), float(numbers[4]), float(numbers[5]))
        return int(numbers[1]), confidence, confidence, confidence, localization, _quality_score(confidence, confidence, confidence, localization)
    if len(parts) >= 8 and numbers[0] is not None:
        confidence = numbers[5] if numbers[5] is not None else default_confidence
        objectness = numbers[6] if numbers[6] is not None else confidence
        class_confidence = numbers[7] if numbers[7] is not None else confidence
        localization = _box_localization_quality(float(numbers[1]), float(numbers[2]), float(numbers[3]), float(numbers[4]))
        return int(numbers[0]), float(confidence), float(objectness), float(class_confidence), localization, _quality_score(float(confidence), float(objectness), float(class_confidence), localization)
    if len(parts) >= 6 and numbers[0] is not None:
        confidence = numbers[5] if numbers[5] is not None else default_confidence
        localization = _box_localization_quality(float(numbers[1]), float(numbers[2]), float(numbers[3]), float(numbers[4]))
        return int(numbers[0]), float(confidence), float(confidence), float(confidence), localization, _quality_score(float(confidence), float(confidence), float(confidence), localization)
    if len(parts) >= 5 and numbers[0] is not None:
        localization = _box_localization_quality(float(numbers[1]), float(numbers[2]), float(numbers[3]), float(numbers[4]))
        return int(numbers[0]), default_confidence, default_confidence, default_confidence, localization, _quality_score(default_confidence, default_confidence, default_confidence, localization)
    return None


def _iter_text_labels(path: Path, default_confidence: float) -> Iterable[tuple[int, float, float, float, float, float]]:
    for raw in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_text_line(raw, default_confidence)
        if parsed is not None:
            yield parsed


def _iter_json_labels(path: Path, default_confidence: float) -> Iterable[tuple[int, float, float, float, float, float]]:
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
            obj = item.get("objectness", item.get("obj_conf", item.get("objectness_confidence", conf)))
            cls_conf = item.get("class_confidence", item.get("cls_conf", item.get("class_score", conf)))
            localization = item.get("localization_quality", item.get("box_quality"))
            if localization is None and all(key in item for key in ("x", "y", "w", "h")):
                localization = _box_localization_quality(float(item["x"]), float(item["y"]), float(item["w"]), float(item["h"]))
            if localization is None:
                localization = 1.0
            if cls is not None:
                quality = item.get("quality", item.get("quality_score"))
                if quality is None:
                    quality = _quality_score(float(conf), float(obj), float(cls_conf), float(localization))
                yield int(cls), float(conf), float(obj), float(cls_conf), float(localization), float(quality)
        elif isinstance(item, (list, tuple)):
            if len(item) >= 9:
                localization = _box_localization_quality(float(item[2]), float(item[3]), float(item[4]), float(item[5]))
                yield int(item[1]), float(item[6]), float(item[7]), float(item[8]), localization, _quality_score(float(item[6]), float(item[7]), float(item[8]), localization)
            elif len(item) >= 7:
                localization = _box_localization_quality(float(item[2]), float(item[3]), float(item[4]), float(item[5]))
                yield int(item[1]), float(item[6]), float(item[6]), float(item[6]), localization, _quality_score(float(item[6]), float(item[6]), float(item[6]), localization)
            elif len(item) >= 6:
                localization = _box_localization_quality(float(item[1]), float(item[2]), float(item[3]), float(item[4]))
                yield int(item[0]), float(item[5]), float(item[5]), float(item[5]), localization, _quality_score(float(item[5]), float(item[5]), float(item[5]), localization)


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
    objectness_sums = [0.0] * num_classes
    class_confidence_sums = [0.0] * num_classes
    localization_sums = [0.0] * num_classes
    quality_sums = [0.0] * num_classes

    for file_path in _label_files(path):
        if file_path.suffix.lower() == ".json":
            labels = _iter_json_labels(file_path, default_confidence)
        else:
            labels = _iter_text_labels(file_path, default_confidence)
        for cls, confidence, objectness, class_confidence, localization, quality in labels:
            if 0 <= cls < num_classes:
                clipped = min(max(float(confidence), 0.0), 1.0)
                clipped_obj = min(max(float(objectness), 0.0), 1.0)
                clipped_cls = min(max(float(class_confidence), 0.0), 1.0)
                clipped_loc = min(max(float(localization), 0.0), 1.0)
                clipped_quality = min(max(float(quality), 0.0), 1.0)
                counts[cls] += 1.0
                confidence_sums[cls] += clipped
                objectness_sums[cls] += clipped_obj
                class_confidence_sums[cls] += clipped_cls
                localization_sums[cls] += clipped_loc
                quality_sums[cls] += clipped_quality

    mean_confidences = [
        confidence_sums[idx] / counts[idx] if counts[idx] > 0 else 0.0
        for idx in range(num_classes)
    ]
    mean_objectness = [
        objectness_sums[idx] / counts[idx] if counts[idx] > 0 else 0.0
        for idx in range(num_classes)
    ]
    mean_class_confidences = [
        class_confidence_sums[idx] / counts[idx] if counts[idx] > 0 else 0.0
        for idx in range(num_classes)
    ]
    mean_localization_qualities = [
        localization_sums[idx] / counts[idx] if counts[idx] > 0 else 0.0
        for idx in range(num_classes)
    ]
    mean_quality_scores = [
        quality_sums[idx] / counts[idx] if counts[idx] > 0 else 0.0
        for idx in range(num_classes)
    ]
    return {
        "id": client_id,
        "source": str(path),
        "counts": counts,
        "confidence_sums": confidence_sums,
        "objectness_sums": objectness_sums,
        "class_confidence_sums": class_confidence_sums,
        "localization_sums": localization_sums,
        "quality_sums": quality_sums,
        "mean_confidences": mean_confidences,
        "mean_objectness": mean_objectness,
        "mean_class_confidences": mean_class_confidences,
        "mean_localization_qualities": mean_localization_qualities,
        "mean_quality_scores": mean_quality_scores,
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
