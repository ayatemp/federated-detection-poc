from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_ROOT = PROJECT_ROOT / "raw" / "bdd100k"
REPORT_ROOT = PROJECT_ROOT / "reports" / "bdd100k_inspection"
DATA_ROOT = PROJECT_ROOT / "data"

BDD_TO_YOLO_CLASS = {
    "pedestrian": "person",
    "rider": "rider",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "bike": "bike",
    "motor": "motor",
    "bicycle": "bike",
    "motorcycle": "motor",
    "traffic light": "traffic light",
    "traffic sign": "traffic sign",
    "train": "train",
}
YOLO_NAMES = [
    "person",
    "rider",
    "car",
    "bus",
    "truck",
    "bike",
    "motor",
    "traffic light",
    "traffic sign",
    "train",
]
YOLO_CLASS_TO_ID = {name: idx for idx, name in enumerate(YOLO_NAMES)}


@dataclass(frozen=True)
class PrepConfig:
    server_weather: str = "partly cloudy"
    client_weathers: tuple[str, ...] = ("overcast", "rainy", "snowy")
    symlink_images: bool = True
    overwrite: bool = False
    preview_scene: str = "highway"
    num_preview_images: int = 3
    max_server_train: int | None = None
    max_server_val: int | None = None
    max_client_images: int | None = None


def dataset_root() -> Path:
    path_file = RAW_ROOT / "kagglehub_path.txt"
    if path_file.exists():
        path = Path(path_file.read_text(encoding="utf-8").strip())
        if path.exists():
            return path

    symlink = RAW_ROOT / "downloaded"
    if symlink.exists():
        return symlink.resolve()

    fallback = Path(
        "/Users/kakuayato/.cache/kagglehub/datasets/awsaf49/bdd100k-dataset/versions/1"
    )
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "BDD100K dataset was not found. Run 00_download_and_inspect_bdd100k.ipynb first."
    )


def label_json(root: Path, split: str) -> Path:
    path = root / "labels" / f"det_v2_{split}_release.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def image_root(root: Path, split: str) -> Path:
    path = root / "bdd100k" / "bdd100k" / "images" / "100k" / split
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_records(root: Path, split: str) -> list[dict]:
    with label_json(root, split).open("r", encoding="utf-8") as f:
        return json.load(f)


def record_attr(record: dict, key: str) -> str:
    return (record.get("attributes") or {}).get(key) or "undefined"


def iter_box_labels(record: dict) -> Iterable[tuple[str, dict]]:
    for label in record.get("labels") or []:
        box = label.get("box2d")
        category = label.get("category")
        if box and category:
            yield category, box


def mapped_yolo_lines(record: dict, image_size: tuple[int, int]) -> list[str]:
    width, height = image_size
    lines: list[str] = []

    for bdd_category, box in iter_box_labels(record):
        yolo_class = BDD_TO_YOLO_CLASS.get(bdd_category)
        if yolo_class is None:
            continue

        x1 = max(0.0, min(float(box["x1"]), float(width)))
        y1 = max(0.0, min(float(box["y1"]), float(height)))
        x2 = max(0.0, min(float(box["x2"]), float(width)))
        y2 = max(0.0, min(float(box["y2"]), float(height)))
        if x2 <= x1 or y2 <= y1:
            continue

        xc = ((x1 + x2) / 2.0) / width
        yc = ((y1 + y2) / 2.0) / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        cls_id = YOLO_CLASS_TO_ID[yolo_class]
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    return lines


def reset_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def analyze_dataset(root: Path, cfg: PrepConfig) -> dict:
    rows_attr: list[dict] = []
    rows_class: list[dict] = []
    rows_class_scene: list[dict] = []
    rows_scene_weather: list[dict] = []

    preview_records: list[tuple[str, dict]] = []
    summary: dict[str, dict] = {}

    for split in ("train", "val"):
        records = load_records(root, split)
        attr_counts: dict[str, Counter] = {
            "weather": Counter(),
            "scene": Counter(),
            "timeofday": Counter(),
        }
        class_box_counts: Counter = Counter()
        class_record_counts: Counter = Counter()
        class_by_scene: dict[str, Counter] = defaultdict(Counter)
        scene_weather: dict[str, Counter] = defaultdict(Counter)

        for record in records:
            attrs = record.get("attributes") or {}
            weather = attrs.get("weather") or "undefined"
            scene = attrs.get("scene") or "undefined"
            timeofday = attrs.get("timeofday") or "undefined"

            attr_counts["weather"][weather] += 1
            attr_counts["scene"][scene] += 1
            attr_counts["timeofday"][timeofday] += 1
            scene_weather[scene][weather] += 1

            seen_classes: set[str] = set()
            has_box = False
            for category, _box in iter_box_labels(record):
                class_box_counts[category] += 1
                class_by_scene[scene][category] += 1
                seen_classes.add(category)
                has_box = True

            for category in seen_classes:
                class_record_counts[category] += 1

            if (
                split == "train"
                and scene == cfg.preview_scene
                and has_box
                and len(preview_records) < cfg.num_preview_images
            ):
                preview_records.append((split, record))

        for attr_name, counter in attr_counts.items():
            for value, count in counter.most_common():
                rows_attr.append(
                    {"split": split, "attribute": attr_name, "value": value, "num_images": count}
                )

        for category, count in class_box_counts.most_common():
            rows_class.append(
                {
                    "split": split,
                    "category": category,
                    "num_boxes": count,
                    "num_images": class_record_counts[category],
                    "mapped_yolo_class": BDD_TO_YOLO_CLASS.get(category, ""),
                }
            )

        for scene, counter in sorted(class_by_scene.items()):
            for category, count in counter.most_common():
                rows_class_scene.append(
                    {"split": split, "scene": scene, "category": category, "num_boxes": count}
                )

        for scene, counter in sorted(scene_weather.items()):
            for weather, count in counter.most_common():
                rows_scene_weather.append(
                    {"split": split, "scene": scene, "weather": weather, "num_images": count}
                )

        summary[split] = {
            "num_records": len(records),
            "weather": dict(attr_counts["weather"].most_common()),
            "scene": dict(attr_counts["scene"].most_common()),
            "timeofday": dict(attr_counts["timeofday"].most_common()),
            "classes": dict(class_box_counts.most_common()),
        }

    write_csv(
        REPORT_ROOT / "attribute_counts.csv",
        ["split", "attribute", "value", "num_images"],
        rows_attr,
    )
    write_csv(
        REPORT_ROOT / "category_counts.csv",
        ["split", "category", "num_boxes", "num_images", "mapped_yolo_class"],
        rows_class,
    )
    write_csv(
        REPORT_ROOT / "category_counts_by_scene.csv",
        ["split", "scene", "category", "num_boxes"],
        rows_class_scene,
    )
    write_csv(
        REPORT_ROOT / "scene_weather_counts.csv",
        ["split", "scene", "weather", "num_images"],
        rows_scene_weather,
    )
    write_preview_images(root, preview_records)
    write_text(REPORT_ROOT / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def write_preview_images(root: Path, preview_records: list[tuple[str, dict]]) -> None:
    preview_dir = REPORT_ROOT / "previews"
    reset_dir(preview_dir, overwrite=True)

    rows: list[dict] = []
    colors = {
        "person": "red",
        "car": "lime",
        "bus": "cyan",
        "truck": "orange",
        "traffic sign": "yellow",
    }

    for idx, (split, record) in enumerate(preview_records, start=1):
        src = image_root(root, split) / record["name"]
        with Image.open(src) as img:
            img = img.convert("RGB")
            draw = ImageDraw.Draw(img)
            for bdd_category, box in iter_box_labels(record):
                yolo_class = BDD_TO_YOLO_CLASS.get(bdd_category)
                if yolo_class is None:
                    continue
                x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
                color = colors.get(yolo_class, "white")
                draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
                draw.text((x1 + 3, max(0, y1 - 12)), yolo_class, fill=color)

            out = preview_dir / f"{idx:02d}_{record_attr(record, 'scene')}_{record['name']}"
            img.save(out, quality=92)

        rows.append(
            {
                "preview_path": str(out),
                "split": split,
                "name": record["name"],
                "weather": record_attr(record, "weather"),
                "scene": record_attr(record, "scene"),
                "timeofday": record_attr(record, "timeofday"),
            }
        )

    write_csv(
        REPORT_ROOT / "preview_images.csv",
        ["preview_path", "split", "name", "weather", "scene", "timeofday"],
        rows,
    )


def select_records(
    records: list[dict],
    attr_name: str,
    attr_value: str,
    limit: int | None,
    require_yolo_labels: bool,
    image_dir: Path,
) -> list[dict]:
    selected: list[dict] = []
    for record in records:
        if record_attr(record, attr_name) != attr_value:
            continue

        if require_yolo_labels:
            src = image_dir / record["name"]
            if not src.exists():
                continue
            with Image.open(src) as img:
                lines = mapped_yolo_lines(record, img.size)
            if not lines:
                continue

        selected.append(record)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def write_server_split(root: Path, split: str, records: list[dict], cfg: PrepConfig) -> dict:
    img_dir = image_root(root, split)
    out_img_dir = DATA_ROOT / "server" / "images" / split
    out_label_dir = DATA_ROOT / "server" / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    num_boxes = 0
    num_images = 0
    class_counts: Counter = Counter()

    for record in records:
        src = img_dir / record["name"]
        with Image.open(src) as img:
            lines = mapped_yolo_lines(record, img.size)
        if not lines:
            continue

        link_or_copy(src, out_img_dir / record["name"], cfg.symlink_images)
        label_path = out_label_dir / f"{Path(record['name']).stem}.txt"
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        num_images += 1
        num_boxes += len(lines)
        for line in lines:
            class_counts[YOLO_NAMES[int(line.split()[0])]] += 1

    return {
        "split": split,
        "num_images": num_images,
        "num_boxes": num_boxes,
        "class_counts": dict(class_counts),
    }


def write_client_images(
    root: Path,
    records: list[dict],
    client_id: int,
    weather: str,
    cfg: PrepConfig,
) -> dict:
    src_root = image_root(root, "train")
    out_dir = DATA_ROOT / "clients" / f"client_{client_id}" / "images_unlabeled"
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for record in records:
        src = src_root / record["name"]
        if not src.exists():
            continue
        link_or_copy(src, out_dir / record["name"], cfg.symlink_images)
        count += 1

    return {"client_id": client_id, "weather": weather, "num_images": count}


def write_data_yaml() -> None:
    payload = {
        "path": str((DATA_ROOT / "server").resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(YOLO_NAMES),
        "names": YOLO_NAMES,
    }
    with (DATA_ROOT / "server" / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def prepare_dataset(root: Path, cfg: PrepConfig) -> dict:
    if DATA_ROOT.exists() and not cfg.overwrite:
        raise FileExistsError(
            f"{DATA_ROOT} already exists. Re-run with --overwrite to rebuild it."
        )
    reset_dir(DATA_ROOT, overwrite=cfg.overwrite)

    train_records = load_records(root, "train")
    val_records = load_records(root, "val")

    server_train = select_records(
        train_records,
        "weather",
        cfg.server_weather,
        cfg.max_server_train,
        require_yolo_labels=True,
        image_dir=image_root(root, "train"),
    )
    server_val = select_records(
        val_records,
        "weather",
        cfg.server_weather,
        cfg.max_server_val,
        require_yolo_labels=True,
        image_dir=image_root(root, "val"),
    )

    if not server_train:
        raise ValueError(f"No server train records found for weather={cfg.server_weather!r}.")
    if not server_val:
        raise ValueError(f"No server val records found for weather={cfg.server_weather!r}.")

    server_summary = [
        write_server_split(root, "train", server_train, cfg),
        write_server_split(root, "val", server_val, cfg),
    ]
    write_data_yaml()

    client_summary: list[dict] = []
    for client_id, weather in enumerate(cfg.client_weathers):
        client_records = select_records(
            train_records,
            "weather",
            weather,
            cfg.max_client_images,
            require_yolo_labels=False,
            image_dir=image_root(root, "train"),
        )
        client_summary.append(write_client_images(root, client_records, client_id, weather, cfg))

    summary = {
        "server_weather": cfg.server_weather,
        "client_weathers": list(cfg.client_weathers),
        "server": server_summary,
        "clients": client_summary,
        "data_yaml": str((DATA_ROOT / "server" / "data.yaml").resolve()),
        "class_mapping": BDD_TO_YOLO_CLASS,
    }
    write_text(DATA_ROOT / "dataset_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BDD100K as a YOLO dataset for FedSTO.")
    parser.add_argument("--server-weather", default="partly cloudy")
    parser.add_argument("--client-weather", action="append", dest="client_weathers")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preview-scene", default="highway")
    parser.add_argument("--num-preview-images", type=int, default=3)
    parser.add_argument("--max-server-train", type=int)
    parser.add_argument("--max-server-val", type=int)
    parser.add_argument("--max-client-images", type=int)
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PrepConfig(
        server_weather=args.server_weather,
        client_weathers=tuple(args.client_weathers or ("overcast", "rainy", "snowy")),
        symlink_images=not args.copy_images,
        overwrite=args.overwrite,
        preview_scene=args.preview_scene,
        num_preview_images=args.num_preview_images,
        max_server_train=args.max_server_train,
        max_server_val=args.max_server_val,
        max_client_images=args.max_client_images,
    )

    root = dataset_root()
    print(f"BDD100K root: {root}")

    print("Analyzing attributes, classes, scene/class counts, and previews...")
    analysis_summary = analyze_dataset(root, cfg)
    print(json.dumps(analysis_summary, ensure_ascii=False, indent=2)[:4000])

    if args.analyze_only:
        print(f"Analysis artifacts saved under: {REPORT_ROOT}")
        return

    print("Preparing FedSTO YOLO dataset...")
    dataset_summary = prepare_dataset(root, cfg)
    print(json.dumps(dataset_summary, ensure_ascii=False, indent=2))
    print(f"Dataset ready: {DATA_ROOT}")


if __name__ == "__main__":
    main()
