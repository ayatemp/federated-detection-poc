#!/usr/bin/env python3
"""Run PseudoGT Learnability 02.

02 tests whether pseudo labels become learnable when bbox supervision is gated
by augmentation stability instead of raw detector confidence.  A single warmup
teacher predicts each target image twice, on the original image and on a
horizontally flipped view.  Only class-consistent boxes whose de-augmented
coordinates agree are written as offline YOLO labels.  Client training then
uses the source labeled set plus these stable target pseudo labels.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
NAV_ROOT = REPO_ROOT / "navigating_data_heterogeneity"
ET_ROOT = NAV_ROOT / "vendor" / "efficientteacher"
PROTOCOL_VERSION = "pseudogt_learnability_02_stable_aug_v1"


@dataclass(frozen=True)
class Variant:
    name: str
    train_scope: str
    aggregate_scope: str
    epochs: int
    lr0: float
    pseudo_repeat: int
    source_repeat: int = 1
    orthogonal_weight: float = 0.0
    note: str = ""


VARIANTS: dict[str, Variant] = {
    "stable_mix_backbone": Variant(
        name="stable_mix_backbone",
        train_scope="backbone",
        aggregate_scope="backbone",
        epochs=3,
        lr0=0.0025,
        pseudo_repeat=1,
        note="Source-anchored, Phase-1-like backbone adaptation with stable pseudo target boxes.",
    ),
    "stable_mix_neck_head": Variant(
        name="stable_mix_neck_head",
        train_scope="neck_head",
        aggregate_scope="all",
        epochs=2,
        lr0=0.0009,
        pseudo_repeat=1,
        note="High-precision stable boxes used for conservative neck/head adaptation.",
    ),
    "stable_mix_all_lowlr": Variant(
        name="stable_mix_all_lowlr",
        train_scope="all",
        aggregate_scope="all",
        epochs=2,
        lr0=0.0007,
        pseudo_repeat=1,
        orthogonal_weight=1e-4,
        note="Low-LR full-model adaptation with source labels anchoring the detector.",
    ),
}


@dataclass
class BoxPrediction:
    cls: int
    conf: float
    xyxy: tuple[float, float, float, float]
    view: str


@dataclass
class StableBox:
    cls: int
    conf: float
    stability: float
    score: float
    views: str
    xyxy: tuple[float, float, float, float]


def add_nav_path() -> None:
    for path in (str(NAV_ROOT), str(ET_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def configure_modules(workspace: Path):
    add_nav_path()
    setup = importlib.import_module("setup_fedsto_scene_reproduction")
    setup.WORK_ROOT = workspace
    setup.LIST_ROOT = workspace / "data_lists"
    setup.CONFIG_ROOT = workspace / "configs"
    setup.RUN_ROOT = workspace / "runs"

    fedsto = importlib.import_module("run_fedsto_efficientteacher_exact")
    fedsto.setup = setup
    fedsto.PRETRAINED_PATH = workspace / "weights" / "efficient-yolov5l.pt"
    fedsto.GLOBAL_DIR = workspace / "global_checkpoints"
    fedsto.CLIENT_STATE_DIR = workspace / "client_states"
    fedsto.HISTORY_PATH = workspace / "history.json"
    return setup, fedsto


def ensure_dirs(workspace: Path) -> None:
    for relative in (
        "runs",
        "configs",
        "stats",
        "validation_reports",
        "logs",
        "checkpoints",
        "client_states",
        "global_checkpoints",
        "weights",
        "pseudo_dataset",
    ):
        (workspace / relative).mkdir(parents=True, exist_ok=True)


def copy_warmup_to_workspace(source: Path, workspace: Path, force: bool) -> Path:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Missing warmup checkpoint: {source}")
    dst = workspace / "global_checkpoints" / "round000_warmup.pt"
    if force or not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dst)
    return dst.resolve()


def resolve_variants(raw: str, epochs_override: int | None) -> list[Variant]:
    names = list(VARIANTS) if raw.strip().lower() == "all" else [item.strip() for item in raw.split(",") if item.strip()]
    variants: list[Variant] = []
    for name in names:
        if name not in VARIANTS:
            raise ValueError(f"Unknown variant {name!r}. Available: {', '.join(VARIANTS)}")
        variant = VARIANTS[name]
        if epochs_override is not None:
            variant = replace(variant, epochs=epochs_override)
        variants.append(variant)
    return variants


def resolve_clients(raw: str, setup) -> list[dict[str, Any]]:
    if raw.strip().lower() == "all":
        return list(setup.CLIENTS)
    wanted = {int(item.strip()) for item in raw.split(",") if item.strip()}
    clients = [client for client in setup.CLIENTS if int(client["id"]) in wanted]
    if not clients:
        raise ValueError(f"No clients selected from {raw!r}")
    return clients


def config_device(args: argparse.Namespace) -> str:
    return "" if args.gpus > 1 else args.device


def read_image_list(path: Path, max_images: int) -> list[Path]:
    images = [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_images > 0:
        images = images[:max_images]
    missing = [image for image in images if not image.exists()]
    if missing:
        raise FileNotFoundError(f"{path} contains {len(missing)} missing images. Example: {missing[0]}")
    return images


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            if dst.resolve() == src.resolve():
                return
        except FileNotFoundError:
            pass
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def box_iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area1 = max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-9)


def mean_iou_to_reference(reference: np.ndarray, boxes: np.ndarray) -> float:
    if len(boxes) == 0:
        return 0.0
    return float(box_iou_one(reference, boxes).mean())


def clipped_yolo_line(stable: StableBox, width: int, height: int) -> str | None:
    x1, y1, x2, y2 = stable.xyxy
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1.0 or bh <= 1.0:
        return None
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return f"{stable.cls} {xc / width:.6f} {yc / height:.6f} {bw / width:.6f} {bh / height:.6f}"


class StableAugPseudoLabeler:
    def __init__(self, weights: Path, device: str, imgsz: int, conf_thres: float, iou_thres: float, max_det: int):
        add_nav_path()
        from models.backbone.experimental import attempt_load
        from utils.general import check_img_size
        from utils.torch_utils import select_device

        self.device = select_device(device)
        self.half = self.device.type != "cpu"
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.model = attempt_load(str(weights), device=self.device, fuse=True)
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(imgsz, s=self.stride)
        self.model.eval()
        if self.half:
            self.model.half()
        if self.device.type != "cpu":
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))

    def _predict_view(self, image_bgr: np.ndarray, view: str, original_width: int) -> list[BoxPrediction]:
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_coords

        img = letterbox(image_bgr, self.imgsz, stride=self.stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor /= 255.0
        tensor = tensor.unsqueeze(0)

        pred = self.model(tensor, augment=False)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        detections = non_max_suppression(
            pred,
            self.conf_thres,
            self.iou_thres,
            classes=None,
            agnostic=False,
            multi_label=False,
            max_det=self.max_det,
        )[0]
        if detections is None or not len(detections):
            return []
        detections[:, :4] = scale_coords(tensor.shape[2:], detections[:, :4], image_bgr.shape).round()
        det_np = detections.detach().float().cpu().numpy()
        if view == "hflip":
            x1 = det_np[:, 0].copy()
            x2 = det_np[:, 2].copy()
            det_np[:, 0] = original_width - x2
            det_np[:, 2] = original_width - x1
        outputs: list[BoxPrediction] = []
        for x1, y1, x2, y2, conf, cls in det_np[:, :6]:
            if x2 <= x1 or y2 <= y1:
                continue
            outputs.append(
                BoxPrediction(
                    cls=int(cls),
                    conf=float(conf),
                    xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    view=view,
                )
            )
        return outputs

    @torch.no_grad()
    def predict_views(self, image_path: Path) -> tuple[list[BoxPrediction], tuple[int, int]]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]
        predictions = self._predict_view(image, "identity", width)
        flipped = cv2.flip(image, 1)
        predictions.extend(self._predict_view(flipped, "hflip", width))
        return predictions, (width, height)


def cluster_stable_boxes(
    predictions: list[BoxPrediction],
    *,
    match_iou: float,
    min_views: int,
    min_stability: float,
    min_score: float,
    max_boxes_per_image: int,
) -> list[StableBox]:
    by_class: dict[int, list[BoxPrediction]] = defaultdict(list)
    for pred in predictions:
        by_class[pred.cls].append(pred)

    stable: list[StableBox] = []
    for cls, cls_preds in by_class.items():
        cls_preds = sorted(cls_preds, key=lambda item: item.conf, reverse=True)
        used = [False] * len(cls_preds)
        for index, seed in enumerate(cls_preds):
            if used[index]:
                continue
            seed_box = np.array(seed.xyxy, dtype=np.float32)
            boxes = np.array([pred.xyxy for pred in cls_preds], dtype=np.float32)
            ious = box_iou_one(seed_box, boxes)
            group_indices = [i for i, iou in enumerate(ious) if not used[i] and iou >= match_iou]
            for group_index in group_indices:
                used[group_index] = True
            group = [cls_preds[i] for i in group_indices]
            views = sorted({item.view for item in group})
            if len(views) < min_views:
                continue
            group_boxes = np.array([item.xyxy for item in group], dtype=np.float32)
            group_confs = np.array([item.conf for item in group], dtype=np.float32)
            weights = group_confs / max(float(group_confs.sum()), 1e-9)
            weighted_box = (group_boxes * weights[:, None]).sum(axis=0)
            stability = mean_iou_to_reference(weighted_box, group_boxes)
            conf = float(group_confs.mean())
            score = conf * stability
            if stability < min_stability or score < min_score:
                continue
            stable.append(
                StableBox(
                    cls=cls,
                    conf=conf,
                    stability=stability,
                    score=score,
                    views=",".join(views),
                    xyxy=tuple(float(v) for v in weighted_box.tolist()),
                )
            )

    stable.sort(key=lambda item: item.score, reverse=True)
    return stable[:max_boxes_per_image]


def apply_class_cap(image_boxes: dict[Path, list[StableBox]], max_class_fraction: float, min_class_keep: int) -> dict[Path, list[StableBox]]:
    if max_class_fraction <= 0:
        return image_boxes
    entries: list[tuple[Path, int, StableBox]] = []
    for image_path, boxes in image_boxes.items():
        for index, box in enumerate(boxes):
            entries.append((image_path, index, box))
    total = len(entries)
    if total == 0:
        return image_boxes
    cap = max(min_class_keep, int(math.ceil(total * max_class_fraction)))
    by_class: dict[int, list[tuple[Path, int, StableBox]]] = defaultdict(list)
    for entry in entries:
        by_class[entry[2].cls].append(entry)
    keep: set[tuple[Path, int]] = set()
    for class_entries in by_class.values():
        class_entries = sorted(class_entries, key=lambda item: item[2].score, reverse=True)
        for image_path, index, _box in class_entries[:cap]:
            keep.add((image_path, index))
    capped: dict[Path, list[StableBox]] = {}
    for image_path, boxes in image_boxes.items():
        selected = [box for index, box in enumerate(boxes) if (image_path, index) in keep]
        if selected:
            capped[image_path] = selected
    return capped


def pseudo_dataset_ready(workspace: Path, clients: list[dict[str, Any]]) -> bool:
    stats_path = workspace / "stats" / "02_pseudo_label_stats.json"
    if not stats_path.exists():
        return False
    for client in clients:
        list_path = workspace / "data_lists" / f"pl02_client{client['id']}_{client['weather']}_stable_train.txt"
        if not list_path.exists() or list_path.stat().st_size == 0:
            return False
    return True


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_pseudo_labels(setup, warmup: Path, args: argparse.Namespace, clients: list[dict[str, Any]]) -> dict[str, Any]:
    if args.skip_pseudo_generation and pseudo_dataset_ready(args.workspace_root, clients):
        stats_path = args.workspace_root / "stats" / "02_pseudo_label_stats.json"
        print(f"Reusing stable pseudo labels: {stats_path}")
        return json.loads(stats_path.read_text(encoding="utf-8"))

    labeler = StableAugPseudoLabeler(
        weights=warmup,
        device=args.device,
        imgsz=args.imgsz,
        conf_thres=args.conf_thres,
        iou_thres=args.nms_iou_thres,
        max_det=args.max_det,
    )
    pseudo_root = args.workspace_root / "pseudo_dataset" / "02_stable_aug"
    stats_rows: list[dict[str, Any]] = []
    all_client_stats: dict[str, Any] = {}

    for client in clients:
        client_tag = f"client{client['id']}_{client['weather']}"
        source_list = setup.LIST_ROOT / f"client_{client['id']}_{client['weather']}_target.txt"
        images = read_image_list(source_list, args.max_images_per_client)
        image_boxes: dict[Path, list[StableBox]] = {}
        dimensions: dict[Path, tuple[int, int]] = {}

        for idx, image_path in enumerate(images, start=1):
            predictions, (width, height) = labeler.predict_views(image_path)
            stable_boxes = cluster_stable_boxes(
                predictions,
                match_iou=args.match_iou,
                min_views=args.min_views,
                min_stability=args.min_stability,
                min_score=args.min_score,
                max_boxes_per_image=args.max_boxes_per_image,
            )
            if stable_boxes:
                image_boxes[image_path] = stable_boxes
                dimensions[image_path] = (width, height)
            if idx == 1 or idx % args.progress_every == 0 or idx == len(images):
                print(f"{client_tag}: pseudo scan {idx}/{len(images)} images, kept {sum(len(v) for v in image_boxes.values())} boxes")

        image_boxes = apply_class_cap(image_boxes, args.max_class_fraction, args.min_class_keep)

        image_dir = pseudo_root / client_tag / "images" / "train"
        label_dir = pseudo_root / client_tag / "labels" / "train"
        if args.force_pseudo and (pseudo_root / client_tag).exists():
            shutil.rmtree(pseudo_root / client_tag)
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        list_images: list[Path] = []
        box_rows: list[dict[str, Any]] = []
        class_counts: Counter[int] = Counter()
        confs: list[float] = []
        stabilities: list[float] = []
        scores: list[float] = []
        keep_image_names: set[str] = set()
        keep_label_names: set[str] = set()

        for image_path, boxes in sorted(image_boxes.items(), key=lambda item: str(item[0])):
            width, height = dimensions[image_path]
            dst_image = image_dir / image_path.name
            dst_label = label_dir / f"{image_path.stem}.txt"
            lines: list[str] = []
            for stable in boxes:
                line = clipped_yolo_line(stable, width, height)
                if line is None:
                    continue
                lines.append(line)
                class_counts[stable.cls] += 1
                confs.append(stable.conf)
                stabilities.append(stable.stability)
                scores.append(stable.score)
                box_rows.append(
                    {
                        "image": str(dst_image.resolve()),
                        "source_image": str(image_path.resolve()),
                        "class_id": stable.cls,
                        "conf": f"{stable.conf:.6f}",
                        "stability": f"{stable.stability:.6f}",
                        "score": f"{stable.score:.6f}",
                        "views": stable.views,
                        "xyxy": " ".join(f"{v:.2f}" for v in stable.xyxy),
                    }
                )
            if not lines:
                continue
            link_or_copy(image_path, dst_image)
            dst_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
            list_images.append(dst_image.resolve())
            keep_image_names.add(dst_image.name)
            keep_label_names.add(dst_label.name)

        for existing in image_dir.glob("*"):
            if existing.name not in keep_image_names:
                existing.unlink()
        for existing in label_dir.glob("*.txt"):
            if existing.name not in keep_label_names:
                existing.unlink()

        train_list = setup.LIST_ROOT / f"pl02_{client_tag}_stable_train.txt"
        train_list.write_text("\n".join(str(path) for path in sorted(list_images)) + ("\n" if list_images else ""), encoding="utf-8")
        if not list_images:
            message = f"No stable pseudo labels were generated for {client_tag}. Lower thresholds or inspect the teacher."
            if args.generate_only:
                print(f"WARNING: {message}")
            else:
                raise RuntimeError(message)

        box_table = args.workspace_root / "stats" / f"02_{client_tag}_stable_boxes.csv"
        write_csv(
            box_table,
            box_rows,
            ["image", "source_image", "class_id", "conf", "stability", "score", "views", "xyxy"],
        )
        client_stats = {
            "client": client_tag,
            "source_images_scanned": len(images),
            "pseudo_images_kept": len(list_images),
            "pseudo_boxes_kept": int(sum(class_counts.values())),
            "boxes_per_kept_image": float(sum(class_counts.values()) / max(len(list_images), 1)),
            "mean_conf": float(np.mean(confs)) if confs else 0.0,
            "mean_stability": float(np.mean(stabilities)) if stabilities else 0.0,
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "class_counts": {str(k): int(v) for k, v in sorted(class_counts.items())},
            "train_list": str(train_list.resolve()),
            "box_table": str(box_table.resolve()),
        }
        all_client_stats[client_tag] = client_stats
        stats_rows.append(
            {
                "client": client_tag,
                "source_images_scanned": client_stats["source_images_scanned"],
                "pseudo_images_kept": client_stats["pseudo_images_kept"],
                "pseudo_boxes_kept": client_stats["pseudo_boxes_kept"],
                "boxes_per_kept_image": f"{client_stats['boxes_per_kept_image']:.4f}",
                "mean_conf": f"{client_stats['mean_conf']:.6f}",
                "mean_stability": f"{client_stats['mean_stability']:.6f}",
                "mean_score": f"{client_stats['mean_score']:.6f}",
                "train_list": client_stats["train_list"],
            }
        )
        print(json.dumps(client_stats, indent=2, ensure_ascii=False))

    csv_path = args.workspace_root / "stats" / "02_pseudo_label_stats.csv"
    write_csv(
        csv_path,
        stats_rows,
        [
            "client",
            "source_images_scanned",
            "pseudo_images_kept",
            "pseudo_boxes_kept",
            "boxes_per_kept_image",
            "mean_conf",
            "mean_stability",
            "mean_score",
            "train_list",
        ],
    )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "teacher": str(warmup.resolve()),
        "params": {
            "imgsz": args.imgsz,
            "conf_thres": args.conf_thres,
            "nms_iou_thres": args.nms_iou_thres,
            "match_iou": args.match_iou,
            "min_views": args.min_views,
            "min_stability": args.min_stability,
            "min_score": args.min_score,
            "max_boxes_per_image": args.max_boxes_per_image,
            "max_class_fraction": args.max_class_fraction,
            "min_class_keep": args.min_class_keep,
            "max_images_per_client": args.max_images_per_client,
        },
        "clients": all_client_stats,
        "csv": str(csv_path.resolve()),
    }
    stats_path = args.workspace_root / "stats" / "02_pseudo_label_stats.json"
    stats_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {stats_path}")
    return payload


def apply_common_training_hyp(cfg: dict[str, Any], variant: Variant) -> None:
    cfg["linear_lr"] = False
    cfg["hyp"]["lr0"] = variant.lr0
    cfg["hyp"]["lrf"] = 1.0
    cfg["hyp"]["warmup_epochs"] = 0
    cfg["hyp"]["mixup"] = 0.0
    cfg["hyp"]["scale"] = 0.35
    cfg["hyp"]["hsv_s"] = 0.45
    cfg["hyp"]["hsv_v"] = 0.25


def train_expr(source_list: Path, pseudo_list: Path, variant: Variant) -> str:
    parts = []
    if variant.source_repeat <= 1:
        parts.append(str(source_list.resolve()))
    else:
        parts.append(f"{source_list.resolve()}*{variant.source_repeat}")
    if variant.pseudo_repeat <= 1:
        parts.append(str(pseudo_list.resolve()))
    else:
        parts.append(f"{pseudo_list.resolve()}*{variant.pseudo_repeat}")
    return "||".join(parts)


def write_client_config(setup, variant: Variant, client: dict[str, Any], start: Path, args: argparse.Namespace) -> Path:
    client_tag = f"client{client['id']}_{client['weather']}"
    pseudo_list = setup.LIST_ROOT / f"pl02_{client_tag}_stable_train.txt"
    source_list = setup.LIST_ROOT / "server_cloudy_train.txt"
    run_name = f"pl02_{variant.name}_{client_tag}"
    cfg = setup.efficientteacher_config(
        name=run_name,
        train=source_list,
        val=setup.LIST_ROOT / "server_cloudy_val.txt",
        target=None,
        weights=str(start.resolve()),
        epochs=variant.epochs,
        train_scope=variant.train_scope,
        orthogonal_weight=variant.orthogonal_weight,
        batch_size=args.batch_size,
        workers=args.workers,
        device=config_device(args),
    )
    cfg["Dataset"]["train"] = train_expr(source_list, pseudo_list, variant)
    cfg["FedSTO"]["unlabeled_only_client"] = False
    cfg["SSOD"] = {"train_domain": False}
    apply_common_training_hyp(cfg, variant)
    return setup.write_config(f"{run_name}.yaml", cfg)


def write_server_repair_config(setup, start: Path, args: argparse.Namespace, variant: Variant) -> Path:
    run_name = f"pl02_{variant.name}_server_repair"
    cfg = setup.efficientteacher_config(
        name=run_name,
        train=setup.LIST_ROOT / "server_cloudy_train.txt",
        val=setup.LIST_ROOT / "server_cloudy_val.txt",
        target=None,
        weights=str(start.resolve()),
        epochs=args.server_repair_epochs,
        train_scope="all",
        orthogonal_weight=0.0,
        batch_size=args.batch_size,
        workers=args.workers,
        device=config_device(args),
    )
    cfg["SSOD"] = {"train_domain": False}
    cfg["linear_lr"] = False
    cfg["hyp"]["lr0"] = args.server_repair_lr
    cfg["hyp"]["lrf"] = 1.0
    cfg["hyp"]["warmup_epochs"] = 0
    cfg["hyp"]["mixup"] = 0.0
    cfg["hyp"]["scale"] = 0.25
    return setup.write_config(f"{run_name}.yaml", cfg)


def run_train(setup, fedsto, config: Path, *, dry_run: bool, gpus: int, master_port: int) -> Path:
    if gpus > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(gpus),
            "--master_port",
            str(master_port),
            "train.py",
            "--cfg",
            str(config.resolve()),
        ]
    else:
        cmd = [sys.executable, "train.py", "--cfg", str(config.resolve())]
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=setup.ET_ROOT, check=True)
    with config.open(encoding="utf-8") as f:
        run_name = yaml.safe_load(f)["name"]
    return fedsto.checkpoint_path(run_name)


def reusable_checkpoint(fedsto, path: Path, force: bool) -> bool:
    if force or not fedsto.checkpoint_present(path):
        return False
    ok, reason = fedsto.validate_checkpoint(path)
    if ok:
        print(f"Reusing checkpoint: {path}")
        return True
    print(f"Ignoring invalid checkpoint ({reason}): {path}")
    return False


def save_checkpoint_record(records: list[dict[str, str]], label: str, path: Path, kind: str, variant: str = "", client: str = "") -> None:
    records.append(
        {
            "label": label,
            "kind": kind,
            "variant": variant,
            "client": client,
            "path": str(path.resolve()),
        }
    )


def run_variant(
    setup,
    fedsto,
    variant: Variant,
    warmup: Path,
    args: argparse.Namespace,
    clients: list[dict[str, Any]],
    port_offset: int,
) -> tuple[list[dict[str, str]], int]:
    records: list[dict[str, str]] = []
    local_paths: list[Path] = []

    for client in clients:
        client_tag = f"client{client['id']}_{client['weather']}"
        start = fedsto.CLIENT_STATE_DIR / f"pl02_{variant.name}_{client_tag}_start.pt"
        run_name = f"pl02_{variant.name}_{client_tag}"
        raw_ckpt = fedsto.checkpoint_path(run_name)
        final_ckpt = args.workspace_root / "checkpoints" / f"{variant.name}_{client_tag}.pt"

        if not args.dry_run and not fedsto.checkpoint_matches_protocol(start, PROTOCOL_VERSION):
            fedsto.make_start_checkpoint(
                warmup,
                start,
                protocol=PROTOCOL_VERSION,
                stage=f"{variant.name}_{client_tag}_start",
            )

        if not reusable_checkpoint(fedsto, final_ckpt, args.force):
            cfg = write_client_config(setup, variant, client, start, args)
            raw_ckpt = run_train(
                setup,
                fedsto,
                cfg,
                dry_run=args.dry_run,
                gpus=args.gpus,
                master_port=args.master_port + port_offset,
            )
            port_offset += 1
            if not args.dry_run:
                fedsto.mark_checkpoint_protocol(raw_ckpt, PROTOCOL_VERSION, f"{variant.name}_{client_tag}_raw")
                fedsto.make_start_checkpoint(
                    raw_ckpt,
                    final_ckpt,
                    protocol=PROTOCOL_VERSION,
                    stage=f"{variant.name}_{client_tag}",
                )

        local_paths.append(final_ckpt)
        save_checkpoint_record(records, f"{variant.name}_{client_tag}", final_ckpt, "client", variant.name, client_tag)

    aggregate_scope = variant.aggregate_scope
    aggregate = args.workspace_root / "checkpoints" / f"{variant.name}_aggregate_{aggregate_scope}.pt"
    if not args.dry_run and not reusable_checkpoint(fedsto, aggregate, args.force):
        fedsto.aggregate_checkpoints(local_paths, warmup, aggregate, backbone_only=(aggregate_scope == "backbone"))
        fedsto.mark_checkpoint_protocol(aggregate, PROTOCOL_VERSION, f"{variant.name}_aggregate_{aggregate_scope}")
    save_checkpoint_record(records, f"{variant.name}_aggregate_{aggregate_scope}", aggregate, "aggregate", variant.name)

    if args.server_repair_epochs > 0:
        repair_start = fedsto.GLOBAL_DIR / f"{variant.name}_server_repair_start.pt"
        repair = args.workspace_root / "checkpoints" / f"{variant.name}_server_repair.pt"
        repair_raw_name = f"pl02_{variant.name}_server_repair"
        repair_raw = fedsto.checkpoint_path(repair_raw_name)
        if not args.dry_run and not reusable_checkpoint(fedsto, repair, args.force):
            fedsto.make_start_checkpoint(
                aggregate,
                repair_start,
                protocol=PROTOCOL_VERSION,
                stage=f"{variant.name}_server_repair_start",
            )
            cfg = write_server_repair_config(setup, repair_start, args, variant)
            repair_raw = run_train(
                setup,
                fedsto,
                cfg,
                dry_run=args.dry_run,
                gpus=args.gpus,
                master_port=args.master_port + port_offset,
            )
            port_offset += 1
            if not args.dry_run:
                fedsto.mark_checkpoint_protocol(repair_raw, PROTOCOL_VERSION, f"{variant.name}_server_repair_raw")
                fedsto.make_start_checkpoint(
                    repair_raw,
                    repair,
                    protocol=PROTOCOL_VERSION,
                    stage=f"{variant.name}_server_repair",
                )
        save_checkpoint_record(records, f"{variant.name}_server_repair", repair, "server_repair", variant.name)

    return records, port_offset


def write_checkpoint_table(path: Path, records: list[dict[str, str]]) -> None:
    write_csv(path, records, ["label", "kind", "variant", "client", "path"])


def run_evaluation(args: argparse.Namespace, records: list[dict[str, str]]) -> None:
    checkpoints = [record for record in records if record["kind"] in {"warmup", "aggregate", "server_repair", "client"}]
    cmd = [
        sys.executable,
        str((PROJECT_ROOT / "scripts" / "evaluate_scene_protocol.py").resolve()),
        "--workspace",
        str(args.workspace_root.resolve()),
        "--splits",
        args.eval_splits,
        "--batch-size",
        str(args.val_batch_size),
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.no_eval_plots:
        cmd.append("--no-plots")
    if args.classwise:
        cmd.append("--verbose")
    if args.dry_run:
        cmd.append("--dry-run")
    for record in checkpoints:
        cmd.extend(["--checkpoint", f"{record['label']}={record['path']}"])
    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT / "output" / "02_stable_pseudogt")
    parser.add_argument("--warmup-checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "round000_warmup.pt")
    parser.add_argument("--clients", default="all")
    parser.add_argument("--variants", default="stable_mix_backbone,stable_mix_all_lowlr")
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=30421)
    parser.add_argument("--device", default="")
    parser.add_argument("--server-repair-epochs", type=int, default=1)
    parser.add_argument("--server-repair-lr", type=float, default=0.0008)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.18)
    parser.add_argument("--nms-iou-thres", type=float, default=0.65)
    parser.add_argument("--match-iou", type=float, default=0.55)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--min-stability", type=float, default=0.58)
    parser.add_argument("--min-score", type=float, default=0.16)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-boxes-per-image", type=int, default=80)
    parser.add_argument("--max-images-per-client", type=int, default=0)
    parser.add_argument("--max-class-fraction", type=float, default=0.55)
    parser.add_argument("--min-class-keep", type=int, default=500)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--skip-pseudo-generation", action="store_true")
    parser.add_argument("--force-pseudo", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-splits", default="highway,citystreet,residential,total")
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--classwise", action="store_true")
    parser.add_argument("--no-eval-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.workspace_root = args.workspace_root.expanduser().resolve()
    ensure_dirs(args.workspace_root)
    setup, fedsto = configure_modules(args.workspace_root)
    setup_payload = setup.build_base_configs()
    manifest = setup_payload.get("manifest") if isinstance(setup_payload, dict) else None
    if manifest is None:
        manifest_path = args.workspace_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    clients = resolve_clients(args.clients, setup)
    warmup = copy_warmup_to_workspace(args.warmup_checkpoint, args.workspace_root, args.force)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Workspace: {args.workspace_root}")
    print(f"Warmup: {warmup}")
    print(json.dumps({"clients": clients, "server": manifest.get("server")}, indent=2, ensure_ascii=False))

    if args.setup_only:
        print("Setup complete.")
        return 0

    pseudo_stats = generate_pseudo_labels(setup, warmup, args, clients)
    if args.generate_only:
        print("Pseudo label generation complete.")
        return 0

    args.gpus = fedsto.resolve_gpus(args.gpus)
    if not args.dry_run:
        fedsto.check_runtime_dependencies()

    variants = resolve_variants(args.variants, args.epochs_override)
    all_records: list[dict[str, str]] = []
    save_checkpoint_record(all_records, "warmup_global", warmup, "warmup")
    port_offset = 0
    for variant in variants:
        print(f"\n=== Running variant: {variant.name} ===")
        print(json.dumps(asdict(variant), indent=2))
        records, port_offset = run_variant(setup, fedsto, variant, warmup, args, clients, port_offset)
        all_records.extend(records)

    stats_root = args.workspace_root / "stats"
    table_path = stats_root / "02_checkpoints.csv"
    write_checkpoint_table(table_path, all_records)

    manifest_path = stats_root / "02_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "protocol": PROTOCOL_VERSION,
                "project_root": str(PROJECT_ROOT),
                "workspace": str(args.workspace_root),
                "warmup_source": str(args.warmup_checkpoint.expanduser().resolve()),
                "warmup_workspace": str(warmup),
                "pseudo_stats": pseudo_stats,
                "variants": [asdict(variant) for variant in variants],
                "checkpoints": all_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved: {table_path}")
    print(f"Saved: {manifest_path}")

    if args.evaluate:
        run_evaluation(args, all_records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
