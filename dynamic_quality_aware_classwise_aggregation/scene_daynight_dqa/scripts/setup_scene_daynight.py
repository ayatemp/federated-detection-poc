#!/usr/bin/env python3
"""Build BDD100K scene x day/night client lists for DQA."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Iterable

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
NAV_ROOT = REPO_ROOT / "navigating_data_heterogeneity"
if str(NAV_ROOT) not in sys.path:
    sys.path.insert(0, str(NAV_ROOT))

import prepare_bdd100k_for_fedsto as prep
import setup_fedsto_exact_reproduction as base


DATA_ROOT = base.DATA_ROOT
WORK_ROOT = PROJECT_ROOT / "output" / "01_repair_oriented_scene_daynight_dqa"
LIST_ROOT = WORK_ROOT / "data_lists"
CONFIG_ROOT = WORK_ROOT / "configs"
RUN_ROOT = WORK_ROOT / "runs"
ET_ROOT = base.ET_ROOT
SSFOD_ROOT = base.SSFOD_ROOT
BDD_NAMES = base.BDD_NAMES
CLIENT_LIMIT = 1500

CLIENTS = [
    {"id": 0, "name": "highway_day", "weather": "highway_day", "scene": "highway", "timeofday": "daytime"},
    {"id": 1, "name": "highway_night", "weather": "highway_night", "scene": "highway", "timeofday": "night"},
    {"id": 2, "name": "citystreet_day", "weather": "citystreet_day", "scene": "city street", "timeofday": "daytime"},
    {"id": 3, "name": "citystreet_night", "weather": "citystreet_night", "scene": "city street", "timeofday": "night"},
    {"id": 4, "name": "residential_day", "weather": "residential_day", "scene": "residential", "timeofday": "daytime"},
    {"id": 5, "name": "residential_night", "weather": "residential_night", "scene": "residential", "timeofday": "night"},
]

EVAL_SPLITS = [
    {"name": client["name"], "raw_scene": client["scene"], "timeofday": client["timeofday"]}
    for client in CLIENTS
]


def _sync_base_paths() -> None:
    base.WORK_ROOT = WORK_ROOT
    base.LIST_ROOT = LIST_ROOT
    base.CONFIG_ROOT = CONFIG_ROOT
    base.RUN_ROOT = RUN_ROOT


def _load_raw_records(split: str) -> list[dict]:
    raw_root = prep.dataset_root()
    label_path = prep.label_json(raw_root, split)
    if label_path.stat().st_size == 0:
        raise RuntimeError(f"BDD100K annotation file is empty: {label_path}")
    return prep.load_records(raw_root, split)


def _record_attr(record: dict, key: str) -> str:
    return (record.get("attributes") or {}).get(key) or "undefined"


def _select_scene_time_records(
    records: Iterable[dict],
    *,
    scene: str,
    timeofday: str,
    limit: int | None,
    require_yolo_labels: bool,
    image_dir: Path,
) -> list[dict]:
    selected: list[dict] = []
    for record in records:
        if _record_attr(record, "scene") != scene:
            continue
        if _record_attr(record, "timeofday") != timeofday:
            continue
        src = image_dir / record["name"]
        if not src.exists():
            continue
        if require_yolo_labels:
            with Image.open(src) as img:
                lines = prep.mapped_yolo_lines(record, img.size)
            if not lines:
                continue
        selected.append(record)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _write_image_list(path: Path, records: list[dict], image_root: Path) -> dict:
    images = sorted((image_root / record["name"]).resolve() for record in records if (image_root / record["name"]).exists())
    base._write_list(path, images)
    return {"list": str(path.resolve()), "images": len(images)}


def _write_eval_split(split: dict, val_records: list[dict]) -> dict:
    raw_root = prep.dataset_root()
    val_root = prep.image_root(raw_root, "val")
    selected = _select_scene_time_records(
        val_records,
        scene=split["raw_scene"],
        timeofday=split["timeofday"],
        limit=None,
        require_yolo_labels=True,
        image_dir=val_root,
    )

    eval_root = WORK_ROOT / "paper_eval_scene_daynight" / split["name"]
    image_dir = eval_root / "images" / "val"
    label_dir = eval_root / "labels" / "val"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    keep_images: set[str] = set()
    keep_labels: set[str] = set()
    image_paths: list[Path] = []
    class_counts: Counter[str] = Counter()
    num_boxes = 0

    for record in selected:
        src = val_root / record["name"]
        with Image.open(src) as img:
            lines = prep.mapped_yolo_lines(record, img.size)
        if not lines:
            continue
        dst_image = image_dir / record["name"]
        prep.link_or_copy(src, dst_image, symlink=True)
        dst_label = label_dir / f"{Path(record['name']).stem}.txt"
        dst_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        image_paths.append(dst_image)
        keep_images.add(dst_image.name)
        keep_labels.add(dst_label.name)
        num_boxes += len(lines)
        for line in lines:
            class_counts[BDD_NAMES[int(line.split()[0])]] += 1

    base._prune_stale_entries(image_dir, keep_images)
    base._prune_stale_entries(label_dir, keep_labels)

    list_path = LIST_ROOT / f"paper_eval_scene_daynight_{split['name']}_val.txt"
    base._write_list(list_path, image_paths)
    return {
        "name": split["name"],
        "raw_scene": split["raw_scene"],
        "timeofday": split["timeofday"],
        "list": str(list_path.resolve()),
        "images": len(image_paths),
        "boxes": num_boxes,
        "class_counts": dict(class_counts),
    }


def build_eval_lists(val_records: list[dict]) -> dict:
    splits = [_write_eval_split(split, val_records) for split in EVAL_SPLITS]
    total_images: list[Path] = []
    for split in splits:
        total_images.extend(Path(line.strip()) for line in Path(split["list"]).read_text().splitlines() if line.strip())
    total_images = sorted(total_images)
    total_list = LIST_ROOT / "paper_eval_scene_daynight_total_val.txt"
    base._write_list(total_list, total_images)
    return {
        "splits": splits,
        "total": {
            "name": "scene_daynight_total",
            "list": str(total_list.resolve()),
            "images": len(total_images),
            "source_splits": [split["name"] for split in splits],
        },
    }


def build_data_lists() -> dict:
    _sync_base_paths()
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Missing paper-scale dataset: {DATA_ROOT}")

    train_records = _load_raw_records("train")
    val_records = _load_raw_records("val")
    raw_root = prep.dataset_root()
    train_image_root = prep.image_root(raw_root, "train")

    server_train = base._images(DATA_ROOT / "server" / "images" / "train")
    server_val = base._images(DATA_ROOT / "server" / "images" / "val")
    base._repair_server_labels_from_bdd100k("train", server_train)
    base._repair_server_labels_from_bdd100k("val", server_val)
    base._assert_labels_exist("train", server_train)
    base._assert_labels_exist("val", server_val)
    base._write_list(LIST_ROOT / "server_cloudy_train.txt", server_train)
    base._write_list(LIST_ROOT / "server_cloudy_val.txt", server_val)

    clients = []
    for client in CLIENTS:
        selected = _select_scene_time_records(
            train_records,
            scene=client["scene"],
            timeofday=client["timeofday"],
            limit=CLIENT_LIMIT,
            require_yolo_labels=False,
            image_dir=train_image_root,
        )
        list_path = LIST_ROOT / f"client_{client['id']}_{client['weather']}_target.txt"
        list_info = _write_image_list(list_path, selected, train_image_root)
        clients.append({**client, **list_info})

    eval_info = build_eval_lists(val_records)
    manifest = {
        "paper": "Navigating Data Heterogeneity in Federated Learning: A Semi-Supervised Federated Object Detection",
        "variant": "scene x day/night clients for DQA",
        "official_ssfod_repo": "https://github.com/Kthyeon/ssfod",
        "official_ssfod_sha": base._git_sha(SSFOD_ROOT),
        "efficientteacher_repo": "https://github.com/AlibabaResearch/efficientteacher",
        "efficientteacher_sha": base._git_sha(ET_ROOT),
        "server": {
            "weather": "cloudy represented by BDD100K Kaggle weather='partly cloudy'",
            "train_list": str((LIST_ROOT / "server_cloudy_train.txt").resolve()),
            "val_list": str((LIST_ROOT / "paper_eval_scene_daynight_total_val.txt").resolve()),
            "source_val_list": str((LIST_ROOT / "server_cloudy_val.txt").resolve()),
            "train_images": len(server_train),
            "val_images": eval_info["total"]["images"],
            "source_val_images": len(server_val),
            "validation_target": "scene_daynight_total",
        },
        "clients": clients,
        "paper_evaluation": eval_info,
        "classes": BDD_NAMES,
        "client_limit": CLIENT_LIMIT,
    }
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    (WORK_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def efficientteacher_config(**kwargs) -> dict:
    _sync_base_paths()
    return base.efficientteacher_config(**kwargs)


def write_config(filename: str, config: dict) -> Path:
    _sync_base_paths()
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    return base.write_config(filename, config)


def build_base_configs() -> dict:
    _sync_base_paths()
    manifest = build_data_lists()
    train = LIST_ROOT / "server_cloudy_train.txt"
    val = LIST_ROOT / "server_cloudy_val.txt"
    configs = {
        "server_warmup": str(
            write_config(
                "server_warmup_yolov5l_bdd100k.yaml",
                efficientteacher_config(name="server_warmup", train=train, val=val, target=None, epochs=50, train_scope="all"),
            )
        )
    }
    for client in CLIENTS:
        target = LIST_ROOT / f"client_{client['id']}_{client['weather']}_target.txt"
        configs[f"client_{client['id']}_phase1"] = str(
            write_config(
                f"client_{client['id']}_{client['weather']}_phase1.yaml",
                efficientteacher_config(
                    name=f"client_{client['id']}_{client['weather']}_phase1",
                    train=train,
                    val=val,
                    target=target,
                    epochs=1,
                    train_scope="backbone",
                ),
            )
        )
        configs[f"client_{client['id']}_phase2"] = str(
            write_config(
                f"client_{client['id']}_{client['weather']}_phase2.yaml",
                efficientteacher_config(
                    name=f"client_{client['id']}_{client['weather']}_phase2",
                    train=train,
                    val=val,
                    target=target,
                    epochs=1,
                    train_scope="all",
                    orthogonal_weight=1e-4,
                ),
            )
        )
    configs["server_phase1"] = str(
        write_config(
            "server_phase1_backbone.yaml",
            efficientteacher_config(name="server_phase1_backbone", train=train, val=val, target=None, epochs=1, train_scope="backbone"),
        )
    )
    configs["server_phase2"] = str(
        write_config(
            "server_phase2_orthogonal.yaml",
            efficientteacher_config(
                name="server_phase2_orthogonal",
                train=train,
                val=val,
                target=None,
                epochs=1,
                train_scope="all",
                orthogonal_weight=1e-4,
            ),
        )
    )
    (WORK_ROOT / "config_index.json").write_text(json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"manifest": manifest, "configs": configs}


if __name__ == "__main__":
    print(json.dumps(build_base_configs(), indent=2, ensure_ascii=False))
