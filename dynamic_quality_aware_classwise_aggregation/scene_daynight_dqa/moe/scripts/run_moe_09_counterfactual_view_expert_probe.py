#!/usr/bin/env python3
"""Probe counterfactual-view pseudoGT experts for MoE x DQA.

09 changes the pseudoGT-MoE unit from "which client/domain gets this box?" to
"which observation condition made this box learnable?"  The concrete probe is:

* predict each target image under original and illumination-enhanced views;
* cluster boxes across views in the original coordinate system;
* split pseudo boxes into clean-original and illumination-rescued experts;
* optionally train one short rescued-view expert from the 03 aggregate.

The default scan is intentionally bounded so the notebook can finish in about
one hour while still producing real evidence about the 05 failure mode.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import torch
import yaml


MOE_ROOT = Path(__file__).resolve().parents[1]
SCENE_ROOT = MOE_ROOT.parent
REPO_ROOT = SCENE_ROOT.parents[1]
PSEUDOGT_SCRIPTS = REPO_ROOT / "pseudogt_learnability" / "scripts"
PROTOCOL_VERSION = "scene_daynight_dqa_moe_09_counterfactual_view_expert_probe_v1"

for path in (SCENE_ROOT / "scripts", SCENE_ROOT.parent, PSEUDOGT_SCRIPTS, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_pseudogt_learnability_02 as pl02  # noqa: E402
import run_pseudogt_learnability_03 as pl03  # noqa: E402
import run_scene_daynight_dqa_01 as dqa01  # noqa: E402
import run_scene_daynight_dqa_01_0 as base01_0  # noqa: E402


DEFAULT_WORKSPACE = MOE_ROOT / "output" / "09_counterfactual_view_expert_probe"
DEFAULT_SOURCE_WORKSPACE = SCENE_ROOT / "output" / "03_main_bn_residual_dqa_experiment"
DEFAULT_ROUTER_WORKSPACE = SCENE_ROOT / "output" / "05_expert_choice_pseudogt_router_dqa"
DEFAULT_TEACHER = (
    DEFAULT_SOURCE_WORKSPACE
    / "bn_residual_dqa"
    / "checkpoints"
    / "round030_bn_residual_dqa_aggregate.pt"
)
DEFAULT_ROUTER_TEACHER = (
    DEFAULT_ROUTER_WORKSPACE
    / "checkpoints"
    / "round030_expert_choice_pseudogt_router_aggregate.pt"
)


ORIGINAL_VIEWS = {"identity", "identity_hflip"}
ILLUMINATION_VIEWS = {"bright", "bright_hflip", "clahe", "clahe_hflip"}


@dataclass(frozen=True)
class ViewExpertBox:
    image_path: Path
    cls: int
    conf: float
    stability: float
    score: float
    views: str
    expert: str
    xyxy: tuple[float, float, float, float]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def seconds_to_hms(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return ""
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def round_tag(round_idx: int) -> str:
    return f"round{round_idx:03d}"


def is_night_client(client_tag: str) -> bool:
    return "night" in client_tag


def brighten(image: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=1.28, beta=28)


def clahe_lightness(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return cv2.convertScaleAbs(enhanced, alpha=1.08, beta=8)


def view_images(image: np.ndarray) -> list[tuple[str, np.ndarray, bool]]:
    bright = brighten(image)
    clahe = clahe_lightness(image)
    return [
        ("identity", image, False),
        ("identity_hflip", cv2.flip(image, 1), True),
        ("bright", bright, False),
        ("bright_hflip", cv2.flip(bright, 1), True),
        ("clahe", clahe, False),
        ("clahe_hflip", cv2.flip(clahe, 1), True),
    ]


@torch.no_grad()
def predict_counterfactual_views(
    labeler: pl02.StableAugPseudoLabeler,
    image_path: Path,
) -> tuple[list[pl02.BoxPrediction], tuple[int, int]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    predictions: list[pl02.BoxPrediction] = []
    for view_name, view_image, flipped in view_images(image):
        call_view = "hflip" if flipped else view_name
        outputs = labeler._predict_view(view_image, call_view, width)  # noqa: SLF001
        if flipped:
            outputs = [replace(item, view=view_name) for item in outputs]
        predictions.extend(outputs)
    return predictions, (width, height)


def yolo_line(box: ViewExpertBox, width: int, height: int) -> str | None:
    stable = pl02.StableBox(
        cls=box.cls,
        conf=box.conf,
        stability=box.stability,
        score=box.score,
        views=box.views,
        xyxy=box.xyxy,
    )
    return pl02.clipped_yolo_line(stable, width, height)


def classify_expert(views: set[str], min_views: int) -> str:
    original_count = len(views & ORIGINAL_VIEWS)
    illumination_count = len(views & ILLUMINATION_VIEWS)
    if original_count >= min_views:
        return "clean_original"
    if illumination_count >= min_views:
        return "illumination_rescued"
    if original_count >= 1 and illumination_count >= 1:
        return "cross_view_bridge"
    return "unstable"


def cluster_counterfactual_boxes(
    image_path: Path,
    predictions: list[pl02.BoxPrediction],
    *,
    match_iou: float,
    min_stability: float,
    min_score: float,
    min_views: int,
    max_boxes_per_image: int,
) -> list[ViewExpertBox]:
    by_class: dict[int, list[pl02.BoxPrediction]] = defaultdict(list)
    for pred in predictions:
        by_class[pred.cls].append(pred)

    clustered: list[ViewExpertBox] = []
    for cls, cls_preds in by_class.items():
        cls_preds = sorted(cls_preds, key=lambda item: item.conf, reverse=True)
        if not cls_preds:
            continue
        used = [False] * len(cls_preds)
        boxes = np.array([pred.xyxy for pred in cls_preds], dtype=np.float32)
        for index, seed in enumerate(cls_preds):
            if used[index]:
                continue
            seed_box = np.array(seed.xyxy, dtype=np.float32)
            ious = pl02.box_iou_one(seed_box, boxes)
            group_indices = [i for i, iou in enumerate(ious) if not used[i] and iou >= match_iou]
            for group_index in group_indices:
                used[group_index] = True
            group = [cls_preds[i] for i in group_indices]
            views = sorted({item.view for item in group})
            expert = classify_expert(set(views), min_views)
            if expert == "unstable":
                continue
            group_boxes = np.array([item.xyxy for item in group], dtype=np.float32)
            group_confs = np.array([item.conf for item in group], dtype=np.float32)
            weights = group_confs / max(float(group_confs.sum()), 1e-9)
            weighted_box = (group_boxes * weights[:, None]).sum(axis=0)
            stability = pl02.mean_iou_to_reference(weighted_box, group_boxes)
            conf = float(group_confs.mean())
            score = conf * stability
            if stability < min_stability or score < min_score:
                continue
            clustered.append(
                ViewExpertBox(
                    image_path=image_path,
                    cls=cls,
                    conf=conf,
                    stability=stability,
                    score=score,
                    views=",".join(views),
                    expert=expert,
                    xyxy=tuple(float(v) for v in weighted_box.tolist()),
                )
            )

    clustered.sort(key=lambda item: item.score, reverse=True)
    return clustered[:max_boxes_per_image]


def prepare_workspace(args: argparse.Namespace):
    pl03.ensure_dirs(args.workspace_root)
    setup, fedsto = dqa01.configure_modules(args.workspace_root, args.client_limit)
    setup_payload = setup.build_base_configs()
    manifest = setup_payload.get("manifest") if isinstance(setup_payload, dict) else {}
    clients = pl02.resolve_clients(args.clients, setup)
    return setup, fedsto, manifest, clients


def link_or_copy_image(src: Path, dst: Path) -> None:
    pl02.link_or_copy(src, dst)


def write_expert_datasets(
    setup,
    args: argparse.Namespace,
    client_tag: str,
    boxes_by_image: Mapping[Path, list[ViewExpertBox]],
    dimensions: Mapping[Path, tuple[int, int]],
) -> dict[str, Any]:
    root = args.workspace_root / "pseudo_dataset" / "09_counterfactual_view_experts" / client_tag
    if args.force_pseudo and root.exists():
        shutil.rmtree(root)

    experts = ("clean_original", "illumination_rescued", "cross_view_bridge", "hybrid_all")
    expert_images: dict[str, set[Path]] = {expert: set() for expert in experts}
    expert_box_counts: Counter[str] = Counter()
    expert_class_counts: dict[str, Counter[int]] = {expert: Counter() for expert in experts}

    for image_path, image_boxes in sorted(boxes_by_image.items(), key=lambda item: str(item[0])):
        width, height = dimensions[image_path]
        by_expert: dict[str, list[ViewExpertBox]] = defaultdict(list)
        for box in image_boxes:
            by_expert[box.expert].append(box)
            by_expert["hybrid_all"].append(box)

        for expert, expert_boxes in by_expert.items():
            image_dir = root / expert / "images" / "train"
            label_dir = root / expert / "labels" / "train"
            dst_image = image_dir / image_path.name
            dst_label = label_dir / f"{image_path.stem}.txt"
            lines: list[str] = []
            for box in expert_boxes:
                line = yolo_line(box, width, height)
                if line is None:
                    continue
                lines.append(line)
                expert_box_counts[expert] += 1
                expert_class_counts[expert][box.cls] += 1
            if not lines:
                continue
            link_or_copy_image(image_path, dst_image)
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
            expert_images[expert].add(dst_image.resolve())

    train_lists: dict[str, str] = {}
    for expert in experts:
        train_list = setup.LIST_ROOT / f"pl09_{client_tag}_{expert}_train.txt"
        images = sorted(expert_images[expert])
        train_list.write_text("\n".join(str(path) for path in images) + ("\n" if images else ""), encoding="utf-8")
        train_lists[expert] = str(train_list.resolve())

    return {
        "train_lists": train_lists,
        "images": {expert: len(expert_images[expert]) for expert in experts},
        "boxes": {expert: int(expert_box_counts[expert]) for expert in experts},
        "class_counts": {
            expert: {str(k): int(v) for k, v in sorted(expert_class_counts[expert].items())}
            for expert in experts
        },
    }


def write_combined_train_lists(setup, args: argparse.Namespace, client_stats: Mapping[str, Any]) -> dict[str, str]:
    experts = ("clean_original", "illumination_rescued", "cross_view_bridge", "hybrid_all")
    combined: dict[str, str] = {}
    for expert in experts:
        image_paths: list[str] = []
        for stats in client_stats.values():
            train_list = Path(stats["dataset"][expert])
            if train_list.exists():
                image_paths.extend([line.strip() for line in train_list.read_text(encoding="utf-8").splitlines() if line.strip()])
        out = setup.LIST_ROOT / f"pl09_all_{expert}_train.txt"
        unique = sorted(set(image_paths))
        out.write_text("\n".join(unique) + ("\n" if unique else ""), encoding="utf-8")
        combined[expert] = str(out.resolve())
    return combined


def scan_counterfactual_views(args: argparse.Namespace, setup, clients: list[dict[str, Any]]) -> dict[str, Any]:
    start_time = time.monotonic()
    labeler = pl02.StableAugPseudoLabeler(
        weights=args.teacher_checkpoint,
        device=args.device,
        imgsz=args.imgsz,
        conf_thres=args.conf_thres,
        iou_thres=args.nms_iou_thres,
        max_det=args.max_det,
    )

    client_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    client_stats: dict[str, Any] = {}

    for client in clients:
        client_tag = f"client{client['id']}_{client['weather']}"
        source_list = setup.LIST_ROOT / f"client_{client['id']}_{client['weather']}_target.txt"
        images = pl02.read_image_list(source_list, args.max_images_per_client)
        boxes_by_image: dict[Path, list[ViewExpertBox]] = {}
        dimensions: dict[Path, tuple[int, int]] = {}
        expert_counts: Counter[str] = Counter()
        class_counts: dict[str, Counter[int]] = defaultdict(Counter)
        view_counts: Counter[str] = Counter()

        for idx, image_path in enumerate(images, start=1):
            predictions, (width, height) = predict_counterfactual_views(labeler, image_path)
            for pred in predictions:
                view_counts[pred.view] += 1
            boxes = cluster_counterfactual_boxes(
                image_path,
                predictions,
                match_iou=args.match_iou,
                min_stability=args.min_stability,
                min_score=args.min_score,
                min_views=args.min_views,
                max_boxes_per_image=args.max_boxes_per_image,
            )
            if boxes:
                boxes_by_image[image_path] = boxes
                dimensions[image_path] = (width, height)
                for box in boxes:
                    expert_counts[box.expert] += 1
                    class_counts[box.expert][box.cls] += 1
                    box_rows.append(
                        {
                            "client": client_tag,
                            "image": str(image_path.resolve()),
                            "expert": box.expert,
                            "class_id": box.cls,
                            "conf": f"{box.conf:.6f}",
                            "stability": f"{box.stability:.6f}",
                            "score": f"{box.score:.6f}",
                            "views": box.views,
                            "xyxy": " ".join(f"{value:.2f}" for value in box.xyxy),
                        }
                    )
            if idx == 1 or idx % args.progress_every == 0 or idx == len(images):
                print(
                    f"09 {client_tag}: scan {idx}/{len(images)} images, "
                    f"clean={expert_counts['clean_original']} "
                    f"rescued={expert_counts['illumination_rescued']} "
                    f"bridge={expert_counts['cross_view_bridge']}"
                )

        dataset = write_expert_datasets(setup, args, client_tag, boxes_by_image, dimensions)
        clean = int(expert_counts["clean_original"])
        rescued = int(expert_counts["illumination_rescued"])
        bridge = int(expert_counts["cross_view_bridge"])
        total = clean + rescued + bridge
        row = {
            "client": client_tag,
            "is_night": is_night_client(client_tag),
            "source_images_scanned": len(images),
            "images_with_boxes": len(boxes_by_image),
            "clean_original_boxes": clean,
            "illumination_rescued_boxes": rescued,
            "cross_view_bridge_boxes": bridge,
            "total_clustered_boxes": total,
            "rescued_ratio": rescued / max(1, total),
            "non_original_ratio": (rescued + bridge) / max(1, total),
            "view_counts": dict(view_counts),
            "class_counts": {
                expert: {str(k): int(v) for k, v in sorted(counts.items())}
                for expert, counts in class_counts.items()
            },
            "dataset": {
                "clean_original": dataset["train_lists"]["clean_original"],
                "illumination_rescued": dataset["train_lists"]["illumination_rescued"],
                "cross_view_bridge": dataset["train_lists"]["cross_view_bridge"],
                "hybrid_all": dataset["train_lists"]["hybrid_all"],
            },
            "dataset_stats": dataset,
        }
        client_stats[client_tag] = row
        client_rows.append(
            {
                "client": client_tag,
                "is_night": row["is_night"],
                "source_images_scanned": row["source_images_scanned"],
                "images_with_boxes": row["images_with_boxes"],
                "clean_original_boxes": clean,
                "illumination_rescued_boxes": rescued,
                "cross_view_bridge_boxes": bridge,
                "total_clustered_boxes": total,
                "rescued_ratio": f"{row['rescued_ratio']:.6f}",
                "non_original_ratio": f"{row['non_original_ratio']:.6f}",
                "clean_train_list": row["dataset"]["clean_original"],
                "illumination_rescued_train_list": row["dataset"]["illumination_rescued"],
                "hybrid_train_list": row["dataset"]["hybrid_all"],
            }
        )

    combined_lists = write_combined_train_lists(setup, args, client_stats)
    write_csv(
        args.workspace_root / "stats" / "09_view_expert_probe_client_stats.csv",
        client_rows,
        [
            "client",
            "is_night",
            "source_images_scanned",
            "images_with_boxes",
            "clean_original_boxes",
            "illumination_rescued_boxes",
            "cross_view_bridge_boxes",
            "total_clustered_boxes",
            "rescued_ratio",
            "non_original_ratio",
            "clean_train_list",
            "illumination_rescued_train_list",
            "hybrid_train_list",
        ],
    )
    write_csv(
        args.workspace_root / "stats" / "09_view_expert_probe_boxes.csv",
        box_rows,
        ["client", "image", "expert", "class_id", "conf", "stability", "score", "views", "xyxy"],
    )

    elapsed = time.monotonic() - start_time
    summary = summarize_scan(client_stats, combined_lists, elapsed, args)
    (args.workspace_root / "stats" / "09_view_expert_probe_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def avg(rows: Iterable[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values)) if values else 0.0


def summarize_scan(
    client_stats: Mapping[str, Any],
    combined_lists: Mapping[str, str],
    elapsed: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = list(client_stats.values())
    day_rows = [row for row in rows if not row["is_night"]]
    night_rows = [row for row in rows if row["is_night"]]
    total_clean = sum(int(row["clean_original_boxes"]) for row in rows)
    total_rescued = sum(int(row["illumination_rescued_boxes"]) for row in rows)
    total_bridge = sum(int(row["cross_view_bridge_boxes"]) for row in rows)
    total = total_clean + total_rescued + total_bridge
    day_rescued_ratio = avg(day_rows, "rescued_ratio")
    night_rescued_ratio = avg(night_rows, "rescued_ratio")
    day_non_original = avg(day_rows, "non_original_ratio")
    night_non_original = avg(night_rows, "non_original_ratio")
    signal = "strong" if night_rescued_ratio > day_rescued_ratio + 0.08 and total_rescued >= 50 else "moderate" if total_rescued >= 20 else "weak"
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "workspace": str(args.workspace_root.resolve()),
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "router_teacher_checkpoint": str(args.router_teacher_checkpoint.resolve()) if args.router_teacher_checkpoint.exists() else "",
        "elapsed_seconds": elapsed,
        "elapsed_hms": seconds_to_hms(elapsed),
        "scan_params": {
            "max_images_per_client": args.max_images_per_client,
            "imgsz": args.imgsz,
            "conf_thres": args.conf_thres,
            "nms_iou_thres": args.nms_iou_thres,
            "match_iou": args.match_iou,
            "min_views": args.min_views,
            "min_stability": args.min_stability,
            "min_score": args.min_score,
            "max_boxes_per_image": args.max_boxes_per_image,
        },
        "totals": {
            "clean_original_boxes": total_clean,
            "illumination_rescued_boxes": total_rescued,
            "cross_view_bridge_boxes": total_bridge,
            "total_clustered_boxes": total,
            "rescued_ratio": total_rescued / max(1, total),
            "non_original_ratio": (total_rescued + total_bridge) / max(1, total),
        },
        "day_night_signal": {
            "day_rescued_ratio": day_rescued_ratio,
            "night_rescued_ratio": night_rescued_ratio,
            "night_minus_day_rescued_ratio": night_rescued_ratio - day_rescued_ratio,
            "day_non_original_ratio": day_non_original,
            "night_non_original_ratio": night_non_original,
            "night_minus_day_non_original_ratio": night_non_original - day_non_original,
            "signal_strength": signal,
        },
        "combined_train_lists": dict(combined_lists),
        "clients": client_stats,
    }


def train_expr(source_list: Path, pseudo_list: Path, pseudo_repeat: int) -> str:
    parts = [str(source_list.resolve())]
    if pseudo_repeat <= 1:
        parts.append(str(pseudo_list.resolve()))
    else:
        parts.append(f"{pseudo_list.resolve()}*{pseudo_repeat}")
    return "||".join(parts)


def write_probe_train_config(setup, args: argparse.Namespace, start: Path, pseudo_list: Path) -> Path:
    cfg = setup.efficientteacher_config(
        name="pl09_counterfactual_illumination_rescued_probe",
        train=setup.LIST_ROOT / "server_cloudy_train.txt",
        val=setup.LIST_ROOT / "server_cloudy_val.txt",
        target=None,
        weights=str(start.resolve()),
        epochs=args.probe_epochs,
        train_scope=args.probe_train_scope,
        orthogonal_weight=args.probe_orthogonal_weight,
        batch_size=args.batch_size,
        workers=args.workers,
        device=pl03.config_device(args),
    )
    cfg["Dataset"]["train"] = train_expr(setup.LIST_ROOT / "server_cloudy_train.txt", pseudo_list, args.probe_pseudo_repeat)
    cfg["FedSTO"]["unlabeled_only_client"] = False
    cfg["SSOD"] = {"train_domain": False}
    cfg["linear_lr"] = False
    cfg["hyp"]["lr0"] = args.probe_lr
    cfg["hyp"]["lrf"] = 1.0
    cfg["hyp"]["warmup_epochs"] = 0
    cfg["hyp"]["mixup"] = 0.0
    cfg["hyp"]["scale"] = 0.25
    cfg["hyp"]["hsv_s"] = 0.35
    cfg["hyp"]["hsv_v"] = 0.20
    if args.probe_loss_box is not None:
        cfg.setdefault("Loss", {})
        cfg["Loss"]["box"] = float(args.probe_loss_box)
    return setup.write_config("pl09_counterfactual_illumination_rescued_probe.yaml", cfg)


def run_training_probe(args: argparse.Namespace, setup, fedsto, summary: Mapping[str, Any]) -> dict[str, Any]:
    pseudo_list = Path(summary["combined_train_lists"]["illumination_rescued"])
    images = [line for line in pseudo_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(images) < args.min_rescued_images_for_training:
        payload = {
            "status": "skipped",
            "reason": f"rescued image count {len(images)} < {args.min_rescued_images_for_training}",
            "rescued_images": len(images),
        }
        (args.workspace_root / "stats" / "09_training_probe_summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return payload

    args.gpus = fedsto.resolve_gpus(args.gpus)
    fedsto.check_runtime_dependencies()
    start = fedsto.GLOBAL_DIR / "pl09_counterfactual_illumination_rescued_probe_start.pt"
    final_ckpt = args.workspace_root / "checkpoints" / "09_counterfactual_illumination_rescued_probe.pt"
    if not fedsto.checkpoint_matches_protocol(start, PROTOCOL_VERSION):
        fedsto.make_start_checkpoint(
            args.teacher_checkpoint,
            start,
            protocol=PROTOCOL_VERSION,
            stage="09_counterfactual_illumination_rescued_probe_start",
        )
    if not pl03.reusable_checkpoint(fedsto, final_ckpt, args.force):
        cfg = write_probe_train_config(setup, args, start, pseudo_list)
        raw_ckpt = pl03.run_train(
            setup,
            fedsto,
            cfg,
            dry_run=args.dry_run,
            gpus=args.gpus,
            master_port=args.master_port,
        )
        if not args.dry_run:
            fedsto.mark_checkpoint_protocol(raw_ckpt, PROTOCOL_VERSION, "09_counterfactual_illumination_rescued_probe_raw")
            fedsto.make_start_checkpoint(
                raw_ckpt,
                final_ckpt,
                protocol=PROTOCOL_VERSION,
                stage="09_counterfactual_illumination_rescued_probe",
            )
            pl03.cleanup_training_artifacts(raw_ckpt, start)

    records = [
        {
            "condition": "03_reference",
            "label": "03_bn_residual_dqa_aggregate",
            "kind": "aggregate",
            "round": "30",
            "client": "",
            "variant": "reference",
            "path": str(args.teacher_checkpoint.resolve()),
        },
        {
            "condition": "05_reference",
            "label": "05_expert_choice_router_aggregate",
            "kind": "aggregate",
            "round": "30",
            "client": "",
            "variant": "reference",
            "path": str(args.router_teacher_checkpoint.resolve()),
        },
        {
            "condition": "09_counterfactual_view_expert",
            "label": "09_illumination_rescued_probe",
            "kind": "aggregate",
            "round": "",
            "client": "",
            "variant": f"{args.probe_train_scope}:pseudo_repeat={args.probe_pseudo_repeat}",
            "path": str(final_ckpt.resolve()),
        },
    ]
    write_csv(
        args.workspace_root / "stats" / "09_training_probe_checkpoints.csv",
        records,
        ["condition", "label", "kind", "round", "client", "variant", "path"],
    )
    if args.evaluate:
        base01_0.run_evaluation(args, records)
        metric_rows = write_probe_metrics(args)
    else:
        metric_rows = []
    payload = {
        "status": "ok",
        "checkpoint": str(final_ckpt.resolve()),
        "rescued_images": len(images),
        "metrics": metric_rows,
    }
    (args.workspace_root / "stats" / "09_training_probe_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def write_probe_metrics(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_csv(args.workspace_root / "validation_reports" / "paper_protocol_eval_summary.csv")
        if row.get("status") == "ok"
    ]
    total_rows = [row for row in rows if row.get("split") in {"scene_daynight_total", "total"}]
    by_label_total = {row["checkpoint_label"]: row for row in total_rows}
    by_label_split = {(row["checkpoint_label"], row["split"]): row for row in rows}
    reference = as_float(by_label_total.get("03_bn_residual_dqa_aggregate", {}).get("map50_95"), 0.0) or 0.0
    router = as_float(by_label_total.get("05_expert_choice_router_aggregate", {}).get("map50_95"), 0.0) or 0.0

    metric_rows: list[dict[str, Any]] = []
    for label, total in by_label_total.items():
        m95 = as_float(total.get("map50_95"), 0.0) or 0.0
        m50 = as_float(total.get("map50"), 0.0) or 0.0
        split_gap = base01_0.split_gap_metrics(by_label_split, label)
        metric_rows.append(
            {
                "checkpoint_label": label,
                "precision": total.get("precision", ""),
                "recall": total.get("recall", ""),
                "map50": f"{m50:.6f}",
                "map50_95": f"{m95:.6f}",
                "delta_vs_03_map50_95": f"{m95 - reference:.6f}",
                "delta_vs_05_router_map50_95": f"{m95 - router:.6f}",
                **split_gap,
            }
        )
    write_csv(
        args.workspace_root / "stats" / "09_training_probe_metrics.csv",
        metric_rows,
        [
            "checkpoint_label",
            "precision",
            "recall",
            "map50",
            "map50_95",
            "delta_vs_03_map50_95",
            "delta_vs_05_router_map50_95",
            "worst_split",
            "worst_split_map50_95",
            "day_avg_map50_95",
            "night_avg_map50_95",
            "day_night_gap_map50_95",
        ],
    )
    return metric_rows


def write_report(args: argparse.Namespace, summary: Mapping[str, Any], training: Mapping[str, Any]) -> None:
    signal = summary["day_night_signal"]
    totals = summary["totals"]
    lines = [
        "# MoE x DQA 09: Counterfactual View Expert Probe",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- protocol: {PROTOCOL_VERSION}",
        "- idea: create experts by the view condition that made pseudoGT appear, not by preserving a domain bucket",
        "",
        "## Scan Result",
        "",
        f"- scanned images per client: {summary['scan_params']['max_images_per_client']}",
        f"- clean original boxes: {totals['clean_original_boxes']}",
        f"- illumination-rescued boxes: {totals['illumination_rescued_boxes']}",
        f"- cross-view bridge boxes: {totals['cross_view_bridge_boxes']}",
        f"- rescued ratio: {totals['rescued_ratio']:.3f}",
        f"- day rescued ratio: {signal['day_rescued_ratio']:.3f}",
        f"- night rescued ratio: {signal['night_rescued_ratio']:.3f}",
        f"- night-day rescued gap: {signal['night_minus_day_rescued_ratio']:.3f}",
        f"- signal strength: {signal['signal_strength']}",
        "",
        "## Training Probe",
        "",
        f"- status: {training.get('status')}",
        f"- checkpoint: {training.get('checkpoint', '')}",
        f"- rescued images: {training.get('rescued_images', '')}",
    ]
    metrics = training.get("metrics") or []
    if metrics:
        lines.extend(
            [
                "",
                "| checkpoint | mAP50 | mAP50:95 | delta vs 03 | delta vs 05 router | day avg | night avg |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in metrics:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["checkpoint_label"],
                        row["map50"],
                        row["map50_95"],
                        row["delta_vs_03_map50_95"],
                        row["delta_vs_05_router_map50_95"],
                        row["day_avg_map50_95"],
                        row["night_avg_map50_95"],
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If illumination-rescued boxes are concentrated in night clients, this supports the pseudoGT*MoE thesis: the expert should represent the observation condition that makes pseudo labels learnable.  If the optional training probe improves night or total mAP over 05, the next full notebook should train clean, rescued, and bridge experts separately before aggregation.",
        ]
    )
    (args.workspace_root / "09_counterfactual_view_expert_probe_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def notify(args: argparse.Namespace, message: str, *, title: str, status: str = "", error: str | None = None) -> None:
    try:
        from notebook_notify import notify_discord

        context: dict[str, Any] = {
            "workspace": str(args.workspace_root.expanduser().resolve()),
            "status": status,
            "report": str((args.workspace_root.expanduser().resolve() / "09_counterfactual_view_expert_probe_report.md")),
        }
        if error:
            context["error"] = error[:500]
        summary_path = args.workspace_root.expanduser().resolve() / "stats" / "09_view_expert_probe_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            context["scan_summary"] = json.dumps(
                {
                    "totals": summary.get("totals"),
                    "day_night_signal": summary.get("day_night_signal"),
                },
                ensure_ascii=False,
            )[:1500]
        metrics_path = args.workspace_root.expanduser().resolve() / "stats" / "09_training_probe_metrics.csv"
        if metrics_path.exists():
            context["metrics_csv"] = str(metrics_path)
        print(notify_discord(message, title=title, context=context, fail_silently=True))
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}")


def str2bool(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-workspace", type=Path, default=DEFAULT_SOURCE_WORKSPACE)
    parser.add_argument("--router-workspace", type=Path, default=DEFAULT_ROUTER_WORKSPACE)
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--router-teacher-checkpoint", type=Path, default=DEFAULT_ROUTER_TEACHER)
    parser.add_argument("--client-limit", type=int, default=1500)
    parser.add_argument("--clients", default="all")
    parser.add_argument("--max-images-per-client", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.20)
    parser.add_argument("--nms-iou-thres", type=float, default=0.65)
    parser.add_argument("--match-iou", type=float, default=0.55)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--min-stability", type=float, default=0.55)
    parser.add_argument("--min-score", type=float, default=0.10)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-boxes-per-image", type=int, default=24)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=32941)
    parser.add_argument("--train-probe", action="store_true")
    parser.add_argument("--probe-epochs", type=int, default=1)
    parser.add_argument("--probe-train-scope", choices=["neck_head", "all"], default="neck_head")
    parser.add_argument("--probe-lr", type=float, default=0.0006)
    parser.add_argument("--probe-pseudo-repeat", type=int, default=2)
    parser.add_argument("--probe-loss-box", type=float, default=0.003)
    parser.add_argument("--probe-orthogonal-weight", type=float, default=1e-4)
    parser.add_argument("--min-rescued-images-for-training", type=int, default=20)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument(
        "--eval-splits",
        default="highway_day,highway_night,citystreet_day,citystreet_night,residential_day,residential_night,total",
    )
    parser.add_argument("--eval-clients", action="store_true")
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--classwise", action="store_true")
    parser.add_argument("--no-eval-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-pseudo", action="store_true")
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--notify-start", action="store_true")
    parser.add_argument("--notify-end", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.teacher_checkpoint = args.teacher_checkpoint.expanduser().resolve()
    args.router_teacher_checkpoint = args.router_teacher_checkpoint.expanduser().resolve()
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    (args.workspace_root / "stats").mkdir(parents=True, exist_ok=True)

    setup, fedsto, manifest, clients = prepare_workspace(args)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "workspace": str(args.workspace_root),
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "router_teacher_checkpoint": str(args.router_teacher_checkpoint),
        "server": manifest.get("server"),
        "clients": clients,
        "train_probe": args.train_probe,
        "evaluate": args.evaluate,
    }
    (args.workspace_root / "stats" / "09_counterfactual_view_expert_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = scan_counterfactual_views(args, setup, clients)
    if args.train_probe:
        training = run_training_probe(args, setup, fedsto, summary)
    else:
        training = {"status": "not_requested", "metrics": []}
    write_report(args, summary, training)
    return summary, training


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.notify or args.notify_start:
        notify(args, "MoE x DQA 09 counterfactual-view expert probe started.", title="DQA MoE 09 start", status="started")
    status = "success"
    error: str | None = None
    try:
        summary, training = run(args)
        print(json.dumps({"summary": summary["day_night_signal"], "training": training.get("status")}, indent=2, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = str(exc)
        raise
    finally:
        if args.notify or args.notify_end:
            notify(
                args,
                f"MoE x DQA 09 counterfactual-view expert probe finished with status={status}.",
                title="DQA MoE 09 finish",
                status=status,
                error=error,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
