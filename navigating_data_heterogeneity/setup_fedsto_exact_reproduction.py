from __future__ import annotations

from collections import Counter
import json
import subprocess
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data_paper20k"
WORK_ROOT = PROJECT_ROOT / "efficientteacher_fedsto"
LIST_ROOT = WORK_ROOT / "data_lists"
CONFIG_ROOT = WORK_ROOT / "configs"
RUN_ROOT = WORK_ROOT / "runs"
ET_ROOT = PROJECT_ROOT / "vendor" / "efficientteacher"
SSFOD_ROOT = PROJECT_ROOT / "vendor" / "ssfod"

BDD_NAMES = [
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

CLIENTS = [
    {"id": 0, "weather": "overcast"},
    {"id": 1, "weather": "rainy"},
    {"id": 2, "weather": "snowy"},
]

PAPER_EVAL_SPLITS = [
    {"name": "cloudy", "raw_weather": "partly cloudy"},
    {"name": "overcast", "raw_weather": "overcast"},
    {"name": "rainy", "raw_weather": "rainy"},
    {"name": "snowy", "raw_weather": "snowy"},
]


def _images(path: Path) -> list[Path]:
    return sorted(p.resolve() for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def _label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        image_index = parts.index("images")
    except ValueError:
        raise ValueError(f"Image path does not contain an 'images' directory: {image_path}") from None
    parts[image_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _repair_server_labels_from_bdd100k(split: str, images: list[Path]) -> None:
    missing = [image for image in images if not _label_path_for_image(image).exists()]
    if not missing:
        return

    import prepare_bdd100k_for_fedsto as prep

    raw_root = prep.dataset_root()
    label_json = prep.label_json(raw_root, split)
    if not label_json.exists():
        examples = "\n".join(str(_label_path_for_image(image)) for image in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} server {split} label files are missing and raw BDD100K annotations were not found: {label_json}\n"
            f"Missing examples:\n{examples}"
        )

    with label_json.open(encoding="utf-8") as f:
        records_by_name = {record["name"]: record for record in json.load(f)}

    repaired = 0
    still_missing = []
    for image in missing:
        record = records_by_name.get(image.name)
        if record is None:
            still_missing.append(image)
            continue
        from PIL import Image

        with Image.open(image) as img:
            lines = prep.mapped_yolo_lines(record, img.size)
        if not lines:
            still_missing.append(image)
            continue
        label_path = _label_path_for_image(image)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        repaired += 1

    if repaired:
        print(f"Repaired {repaired} missing server {split} label files from BDD100K annotations.")
    if still_missing:
        examples = "\n".join(str(image) for image in still_missing[:5])
        raise RuntimeError(
            f"Could not repair {len(still_missing)} server {split} labels. Examples:\n{examples}"
        )


def _assert_labels_exist(split: str, images: list[Path]) -> None:
    missing = [image for image in images if not _label_path_for_image(image).exists()]
    if missing:
        examples = "\n".join(
            f"image: {image}\nlabel: {_label_path_for_image(image)}" for image in missing[:5]
        )
        raise RuntimeError(f"{len(missing)} server {split} labels are missing after repair.\n{examples}")


def _write_list(path: Path, images: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(p) for p in images) + "\n", encoding="utf-8")
    cache_path = path.with_suffix(".cache")
    if cache_path.exists():
        cache_path.unlink()


def _prune_stale_entries(path: Path, keep_names: set[str]) -> None:
    if not path.exists():
        return
    for entry in path.iterdir():
        if entry.name in keep_names:
            continue
        if entry.is_dir() and not entry.is_symlink():
            raise RuntimeError(f"Unexpected directory under paper-eval assets: {entry}")
        entry.unlink()


def _git_sha(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def build_paper_eval_lists() -> dict:
    import prepare_bdd100k_for_fedsto as prep
    from PIL import Image

    raw_root = prep.dataset_root()
    val_root = prep.image_root(raw_root, "val")
    val_records = prep.load_records(raw_root, "val")
    eval_root = WORK_ROOT / "paper_eval"

    splits = []
    total_images: list[Path] = []

    for spec in PAPER_EVAL_SPLITS:
        split_name = spec["name"]
        raw_weather = spec["raw_weather"]
        selected = prep.select_records(
            val_records,
            "weather",
            raw_weather,
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

        _prune_stale_entries(image_dir, keep_images)
        _prune_stale_entries(label_dir, keep_labels)

        list_path = LIST_ROOT / f"paper_eval_{split_name}_val.txt"
        _write_list(list_path, image_paths)
        splits.append(
            {
                "name": split_name,
                "raw_weather": raw_weather,
                "list": str(list_path.resolve()),
                "images": len(image_paths),
                "boxes": num_boxes,
                "class_counts": dict(class_counts),
            }
        )
        total_images.extend(image_paths)

    total_list = LIST_ROOT / "paper_eval_total_val.txt"
    total_images = sorted(total_images)
    _write_list(total_list, total_images)
    return {
        "splits": splits,
        "total": {
            "name": "total",
            "list": str(total_list.resolve()),
            "images": len(total_images),
            "source_splits": [spec["name"] for spec in PAPER_EVAL_SPLITS],
        },
    }


def build_data_lists() -> dict:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Missing paper-scale dataset: {DATA_ROOT}")

    server_train = _images(DATA_ROOT / "server" / "images" / "train")
    server_val = _images(DATA_ROOT / "server" / "images" / "val")
    _repair_server_labels_from_bdd100k("train", server_train)
    _repair_server_labels_from_bdd100k("val", server_val)
    _assert_labels_exist("train", server_train)
    _assert_labels_exist("val", server_val)
    _write_list(LIST_ROOT / "server_cloudy_train.txt", server_train)
    _write_list(LIST_ROOT / "server_cloudy_val.txt", server_val)

    clients = []
    for client in CLIENTS:
        imgs = _images(DATA_ROOT / "clients" / f"client_{client['id']}" / "images_unlabeled")
        list_path = LIST_ROOT / f"client_{client['id']}_{client['weather']}_target.txt"
        _write_list(list_path, imgs)
        clients.append({**client, "list": str(list_path.resolve()), "images": len(imgs)})

    paper_evaluation = build_paper_eval_lists()

    manifest = {
        "paper": "Navigating Data Heterogeneity in Federated Learning: A Semi-Supervised Federated Object Detection",
        "official_ssfod_repo": "https://github.com/Kthyeon/ssfod",
        "official_ssfod_sha": _git_sha(SSFOD_ROOT),
        "efficientteacher_repo": "https://github.com/AlibabaResearch/efficientteacher",
        "efficientteacher_sha": _git_sha(ET_ROOT),
        "server": {
            "weather": "cloudy represented by BDD100K Kaggle weather='partly cloudy'",
            "train_list": str((LIST_ROOT / "server_cloudy_train.txt").resolve()),
            "val_list": str((LIST_ROOT / "server_cloudy_val.txt").resolve()),
            "train_images": len(server_train),
            "val_images": len(server_val),
        },
        "clients": clients,
        "paper_evaluation": paper_evaluation,
        "classes": BDD_NAMES,
        "paper_schedule": {"warmup_epochs": 50, "phase1_rounds": 100, "phase2_rounds": 150, "local_epochs": 1},
    }
    (WORK_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def efficientteacher_config(
    *,
    name: str,
    train: Path,
    val: Path,
    target: Path | None,
    weights: str = "",
    epochs: int = 1,
    train_scope: str = "all",
    orthogonal_weight: float = 0.0,
    batch_size: int = 32,
    workers: int = 0,
    device: str = "",
) -> dict:
    cfg = {
        "project": str(RUN_ROOT.resolve()),
        "name": name,
        "exist_ok": True,
        "device": device,
        "adam": False,
        "epochs": epochs,
        "weights": weights,
        "prune_finetune": False,
        # EfficientTeacher's linear scheduler divides by (epochs - 1).
        # FedSTO local rounds are 1 epoch, so use the cosine helper there,
        # which is constant because lrf is 1.0 below.
        "linear_lr": epochs > 1,
        "find_unused_parameters": True,
        "hyp": {
            "lr0": 0.01,
            "lrf": 1.0,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "scale": 0.9,
            "burn_epochs": 0,
            "no_aug_epochs": 0,
            "mixup": 0.1,
        },
        "Model": {
            "depth_multiple": 1.00,
            "width_multiple": 1.00,
            "Backbone": {"name": "YoloV5", "activation": "SiLU"},
            "Neck": {"name": "YoloV5", "in_channels": [256, 512, 1024], "out_channels": [256, 512, 1024], "activation": "SiLU"},
            "Head": {"name": "YoloV5", "activation": "SiLU"},
            "anchors": [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]],
        },
        "Loss": {"type": "ComputeLoss", "cls": 0.3, "obj": 0.7, "anchor_t": 4.0},
        "Dataset": {
            "data_name": "bdd100k_fedsto",
            "train": str(train.resolve()),
            "val": str(val.resolve()),
            "test": str(val.resolve()),
            "nc": len(BDD_NAMES),
            "np": 0,
            "names": BDD_NAMES,
            "img_size": 640,
            "batch_size": batch_size,
            "workers": workers,
        },
        "FedSTO": {
            "train_scope": train_scope,
            "orthogonal_weight": orthogonal_weight,
            "orthogonal_scope": "non_backbone",
            "unlabeled_only_client": target is not None,
        },
    }

    if target is not None:
        cfg["Dataset"]["target"] = str(target.resolve())
        cfg["SSOD"] = {
            "train_domain": True,
            "nms_conf_thres": 0.1,
            "nms_iou_thres": 0.65,
            "teacher_loss_weight": 1.0,
            "cls_loss_weight": 0.3,
            "box_loss_weight": 0.05,
            "obj_loss_weight": 0.7,
            "loss_type": "ComputeStudentMatchLoss",
            "ignore_thres_low": 0.1,
            "ignore_thres_high": 0.6,
            "uncertain_aug": True,
            "use_ota": False,
            "multi_label": False,
            "ignore_obj": False,
            "pseudo_label_with_obj": True,
            "pseudo_label_with_bbox": True,
            # The paper objective includes Lu_cls, so keep pseudo-label cls loss on.
            "pseudo_label_with_cls": True,
            "with_da_loss": False,
            "da_loss_weights": 0.01,
            "epoch_adaptor": True,
            "resample_high_percent": 0.25,
            "resample_low_percent": 0.99,
            "ema_rate": 0.999,
            "cosine_ema": True,
            "imitate_teacher": False,
            "ssod_hyp": {
                "with_gt": False,
                "mosaic": 1.0,
                "cutout": 0.5,
                "autoaugment": 0.5,
                "scale": 0.8,
                "degrees": 0.0,
                "shear": 0.0,
            },
        }
    else:
        cfg["SSOD"] = {"train_domain": False}

    return cfg


def write_config(filename: str, config: dict) -> Path:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    path = CONFIG_ROOT / filename
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def build_base_configs() -> dict:
    manifest = build_data_lists()
    train = LIST_ROOT / "server_cloudy_train.txt"
    val = LIST_ROOT / "server_cloudy_val.txt"

    configs = {
        "server_warmup": write_config(
            "server_warmup_yolov5l_bdd100k.yaml",
            efficientteacher_config(name="server_warmup", train=train, val=val, target=None, epochs=50, train_scope="all"),
        )
    }
    for client in CLIENTS:
        configs[f"client_{client['id']}_phase1"] = write_config(
            f"client_{client['id']}_{client['weather']}_phase1.yaml",
            efficientteacher_config(
                name=f"client_{client['id']}_{client['weather']}_phase1",
                train=train,
                val=val,
                target=LIST_ROOT / f"client_{client['id']}_{client['weather']}_target.txt",
                epochs=1,
                train_scope="backbone",
            ),
        )
        configs[f"client_{client['id']}_phase2"] = write_config(
            f"client_{client['id']}_{client['weather']}_phase2.yaml",
            efficientteacher_config(
                name=f"client_{client['id']}_{client['weather']}_phase2",
                train=train,
                val=val,
                target=LIST_ROOT / f"client_{client['id']}_{client['weather']}_target.txt",
                epochs=1,
                train_scope="all",
                orthogonal_weight=1e-4,
            ),
        )

    configs["server_phase1"] = write_config(
        "server_phase1_backbone.yaml",
        efficientteacher_config(name="server_phase1_backbone", train=train, val=val, target=None, epochs=1, train_scope="backbone"),
    )
    configs["server_phase2"] = write_config(
        "server_phase2_orthogonal.yaml",
        efficientteacher_config(name="server_phase2_orthogonal", train=train, val=val, target=None, epochs=1, train_scope="all", orthogonal_weight=1e-4),
    )

    out = {key: str(path.resolve()) for key, path in configs.items()}
    (WORK_ROOT / "config_index.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": manifest, "configs": out}, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    build_base_configs()
