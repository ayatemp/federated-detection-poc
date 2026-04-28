from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from PIL import Image

import prepare_bdd100k_for_fedsto as prep
import setup_fedsto_exact_reproduction as base


PROJECT_ROOT = base.PROJECT_ROOT
DATA_ROOT = base.DATA_ROOT
WORK_ROOT = PROJECT_ROOT / "efficientteacher_fedsto_scene"
LIST_ROOT = WORK_ROOT / "data_lists"
CONFIG_ROOT = WORK_ROOT / "configs"
RUN_ROOT = WORK_ROOT / "runs"
ET_ROOT = base.ET_ROOT
SSFOD_ROOT = base.SSFOD_ROOT
BDD_NAMES = base.BDD_NAMES

CLIENTS = [
    {"id": 0, "weather": "highway", "scene": "highway"},
    {"id": 1, "weather": "citystreet", "scene": "city street"},
    {"id": 2, "weather": "residential", "scene": "residential"},
]

SCENE_EVAL_SPLITS = [
    {"name": "highway", "raw_scene": "highway"},
    {"name": "citystreet", "raw_scene": "city street"},
    {"name": "residential", "raw_scene": "residential"},
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
        raise RuntimeError(
            f"BDD100K annotation file is empty: {label_path}. "
            "Scene-based clients require the raw det_v2 train/val JSON annotations. "
            "Re-download or restore BDD100K labels before running this notebook."
        )
    return prep.load_records(raw_root, split)


def _write_scene_image_list(split_name: str, records: list[dict], image_root: Path) -> dict:
    list_path = LIST_ROOT / f"client_{split_name}_target.txt"
    images = sorted((image_root / record["name"]).resolve() for record in records if (image_root / record["name"]).exists())
    base._write_list(list_path, images)
    return {"list": str(list_path.resolve()), "images": len(images)}


def build_scene_eval_lists(val_records: list[dict]) -> dict:
    raw_root = prep.dataset_root()
    val_root = prep.image_root(raw_root, "val")
    eval_root = WORK_ROOT / "paper_eval_scene"

    splits = []
    total_images: list[Path] = []

    for spec in SCENE_EVAL_SPLITS:
        split_name = spec["name"]
        raw_scene = spec["raw_scene"]
        selected = prep.select_records(
            val_records,
            "scene",
            raw_scene,
            None,
            require_yolo_labels=True,
            image_dir=val_root,
        )

        image_dir = eval_root / split_name / "images" / "val"
        label_dir = eval_root / split_name / "labels" / "val"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        keep_images: set[str] = set()
        keep_labels: set[str] = set()
        image_paths: list[Path] = []
        class_counts: Counter[str] = Counter()
        num_boxes = 0

        for record in selected:
            src = val_root / record["name"]
            if not src.exists():
                continue
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

        list_path = LIST_ROOT / f"paper_eval_scene_{split_name}_val.txt"
        base._write_list(list_path, image_paths)
        splits.append(
            {
                "name": split_name,
                "raw_scene": raw_scene,
                "list": str(list_path.resolve()),
                "images": len(image_paths),
                "boxes": num_boxes,
                "class_counts": dict(class_counts),
            }
        )
        total_images.extend(image_paths)

    total_list = LIST_ROOT / "paper_eval_scene_total_val.txt"
    total_images = sorted(total_images)
    base._write_list(total_list, total_images)
    return {
        "splits": splits,
        "total": {
            "name": "scene_total",
            "list": str(total_list.resolve()),
            "images": len(total_images),
            "source_splits": [spec["name"] for spec in SCENE_EVAL_SPLITS],
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
        records = prep.select_records(
            train_records,
            "scene",
            client["scene"],
            5000,
            require_yolo_labels=False,
            image_dir=train_image_root,
        )
        list_info = _write_scene_image_list(f"{client['id']}_{client['weather']}", records, train_image_root)
        clients.append({**client, **list_info})

    scene_evaluation = build_scene_eval_lists(val_records)

    manifest = {
        "paper": "Navigating Data Heterogeneity in Federated Learning: A Semi-Supervised Federated Object Detection",
        "variant": "scene clients: highway, city street, residential",
        "official_ssfod_repo": "https://github.com/Kthyeon/ssfod",
        "official_ssfod_sha": base._git_sha(SSFOD_ROOT),
        "efficientteacher_repo": "https://github.com/AlibabaResearch/efficientteacher",
        "efficientteacher_sha": base._git_sha(ET_ROOT),
        "server": {
            "weather": "cloudy represented by BDD100K Kaggle weather='partly cloudy'",
            "train_list": str((LIST_ROOT / "server_cloudy_train.txt").resolve()),
            "val_list": str((LIST_ROOT / "paper_eval_scene_total_val.txt").resolve()),
            "source_val_list": str((LIST_ROOT / "server_cloudy_val.txt").resolve()),
            "train_images": len(server_train),
            "val_images": scene_evaluation["total"]["images"],
            "source_val_images": len(server_val),
            "validation_target": "scene_total",
        },
        "clients": clients,
        "paper_evaluation": scene_evaluation,
        "classes": BDD_NAMES,
        "paper_schedule": {"warmup_epochs": 50, "phase1_rounds": 100, "phase2_rounds": 150, "local_epochs": 1},
    }
    (WORK_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def efficientteacher_config(**kwargs) -> dict:
    _sync_base_paths()
    return base.efficientteacher_config(**kwargs)


def write_config(filename: str, config: dict) -> Path:
    _sync_base_paths()
    return base.write_config(filename, config)


def build_base_configs() -> dict:
    _sync_base_paths()
    manifest = build_data_lists()
    train = LIST_ROOT / "server_cloudy_train.txt"
    val = LIST_ROOT / "paper_eval_scene_total_val.txt"

    configs = {
        "server_warmup": base.write_config(
            "server_warmup_yolov5l_bdd100k.yaml",
            base.efficientteacher_config(name="server_warmup", train=train, val=val, target=None, epochs=50, train_scope="all"),
        )
    }
    for client in CLIENTS:
        target = LIST_ROOT / f"client_{client['id']}_{client['weather']}_target.txt"
        configs[f"client_{client['id']}_phase1"] = base.write_config(
            f"client_{client['id']}_{client['weather']}_phase1.yaml",
            base.efficientteacher_config(
                name=f"client_{client['id']}_{client['weather']}_phase1",
                train=train,
                val=val,
                target=target,
                epochs=1,
                train_scope="backbone",
            ),
        )
        configs[f"client_{client['id']}_phase2"] = base.write_config(
            f"client_{client['id']}_{client['weather']}_phase2.yaml",
            base.efficientteacher_config(
                name=f"client_{client['id']}_{client['weather']}_phase2",
                train=train,
                val=val,
                target=target,
                epochs=1,
                train_scope="all",
                orthogonal_weight=1e-4,
            ),
        )

    configs["server_phase1"] = base.write_config(
        "server_phase1_backbone.yaml",
        base.efficientteacher_config(name="server_phase1_backbone", train=train, val=val, target=None, epochs=1, train_scope="backbone"),
    )
    configs["server_phase2"] = base.write_config(
        "server_phase2_orthogonal.yaml",
        base.efficientteacher_config(name="server_phase2_orthogonal", train=train, val=val, target=None, epochs=1, train_scope="all", orthogonal_weight=1e-4),
    )

    out = {key: str(path.resolve()) for key, path in configs.items()}
    (WORK_ROOT / "config_index.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": manifest, "configs": out}, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    build_base_configs()
