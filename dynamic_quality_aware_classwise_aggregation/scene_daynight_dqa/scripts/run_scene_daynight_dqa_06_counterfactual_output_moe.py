#!/usr/bin/env python3
"""Run 06 Counterfactual pseudoGT routing + output-space MoE DQA.

This is the production version of the 06 idea:

* generate pseudo labels from counterfactual views, reusing the 09 scan logic;
* split pseudo labels by the observation condition that made them stable;
* train independent detector experts from the same 03 teacher checkpoint;
* evaluate each expert with the scene-daynight paper protocol;
* fuse expert predictions with an image-level output-space MoE router.

The implementation deliberately avoids residual checkpoint mixing.  Expert
specialization is preserved as separate checkpoints and combined at prediction
time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
NAV_ROOT = REPO_ROOT / "navigating_data_heterogeneity"
PSEUDOGT_SCRIPTS = REPO_ROOT / "pseudogt_learnability" / "scripts"
MOE_SCRIPTS = PROJECT_ROOT / "moe" / "scripts"
ET_ROOT = NAV_ROOT / "vendor" / "efficientteacher"
PROTOCOL_VERSION = "scene_daynight_dqa_06_counterfactual_output_moe_v1"

for path in (PROJECT_ROOT / "scripts", MOE_SCRIPTS, PROJECT_ROOT.parent, NAV_ROOT, PSEUDOGT_SCRIPTS, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_paper_protocol as shared_eval  # noqa: E402
import run_moe_09_counterfactual_view_expert_probe as moe09  # noqa: E402
import run_pseudogt_learnability_03 as pl03  # noqa: E402
import run_scene_daynight_dqa_01 as dqa01  # noqa: E402
import run_scene_daynight_dqa_01_0 as base01_0  # noqa: E402


DEFAULT_WORKSPACE = PROJECT_ROOT / "output" / "06_counterfactual_output_moe_dqa"
DEFAULT_SOURCE_WORKSPACE = PROJECT_ROOT / "output" / "03_main_bn_residual_dqa_experiment"
DEFAULT_ROUTER_WORKSPACE = PROJECT_ROOT / "output" / "05_expert_choice_pseudogt_router_dqa"
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
EXPERT_ORDER = ("clean_original", "illumination_rescued", "cross_view_bridge")
DEFAULT_EVAL_SPLITS = "highway_day,highway_night,citystreet_day,citystreet_night,residential_day,residential_night,total"


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    pseudo_repeat: int
    lr: float
    loss_box: float
    train_scope: str = "neck_head"
    orthogonal_weight: float = 1e-4


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


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._") or "run"


def count_list(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def resolve_experts(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(EXPERT_ORDER)
    experts = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(experts) - set(EXPERT_ORDER) - {"hybrid_all"})
    if unknown:
        raise ValueError(f"Unknown experts: {unknown}. Available: {', '.join((*EXPERT_ORDER, 'hybrid_all'))}")
    return experts


def expert_specs(args: argparse.Namespace) -> dict[str, ExpertSpec]:
    return {
        "clean_original": ExpertSpec(
            name="clean_original",
            pseudo_repeat=args.clean_pseudo_repeat,
            lr=args.clean_lr,
            loss_box=args.clean_loss_box,
            train_scope=args.expert_train_scope,
            orthogonal_weight=args.expert_orthogonal_weight,
        ),
        "illumination_rescued": ExpertSpec(
            name="illumination_rescued",
            pseudo_repeat=args.illumination_pseudo_repeat,
            lr=args.illumination_lr,
            loss_box=args.illumination_loss_box,
            train_scope=args.expert_train_scope,
            orthogonal_weight=args.expert_orthogonal_weight,
        ),
        "cross_view_bridge": ExpertSpec(
            name="cross_view_bridge",
            pseudo_repeat=args.bridge_pseudo_repeat,
            lr=args.bridge_lr,
            loss_box=args.bridge_loss_box,
            train_scope=args.expert_train_scope,
            orthogonal_weight=args.expert_orthogonal_weight,
        ),
        "hybrid_all": ExpertSpec(
            name="hybrid_all",
            pseudo_repeat=args.hybrid_pseudo_repeat,
            lr=args.hybrid_lr,
            loss_box=args.hybrid_loss_box,
            train_scope=args.expert_train_scope,
            orthogonal_weight=args.expert_orthogonal_weight,
        ),
    }


def prepare_workspace(args: argparse.Namespace):
    pl03.ensure_dirs(args.workspace_root)
    setup, fedsto = dqa01.configure_modules(args.workspace_root, args.client_limit)
    setup_payload = setup.build_base_configs()
    manifest = setup_payload.get("manifest") if isinstance(setup_payload, dict) else {}
    clients = moe09.pl02.resolve_clients(args.clients, setup)
    return setup, fedsto, manifest, clients


def train_expr(source_list: Path, pseudo_list: Path, pseudo_repeat: int) -> str:
    parts = [str(source_list.resolve())]
    if pseudo_repeat <= 1:
        parts.append(str(pseudo_list.resolve()))
    else:
        parts.append(f"{pseudo_list.resolve()}*{pseudo_repeat}")
    return "||".join(parts)


def write_expert_train_config(
    setup,
    args: argparse.Namespace,
    start: Path,
    pseudo_list: Path,
    spec: ExpertSpec,
) -> Path:
    run_name = f"sdn06_counterfactual_{spec.name}_expert"
    cfg = setup.efficientteacher_config(
        name=run_name,
        train=setup.LIST_ROOT / "server_cloudy_train.txt",
        val=setup.LIST_ROOT / "server_cloudy_val.txt",
        target=None,
        weights=str(start.resolve()),
        epochs=args.expert_epochs,
        train_scope=spec.train_scope,
        orthogonal_weight=spec.orthogonal_weight,
        batch_size=args.batch_size,
        workers=args.workers,
        device=pl03.config_device(args),
    )
    cfg["Dataset"]["train"] = train_expr(setup.LIST_ROOT / "server_cloudy_train.txt", pseudo_list, spec.pseudo_repeat)
    cfg["FedSTO"]["unlabeled_only_client"] = False
    cfg["SSOD"] = {"train_domain": False}
    cfg["linear_lr"] = False
    cfg["hyp"]["lr0"] = spec.lr
    cfg["hyp"]["lrf"] = 1.0
    cfg["hyp"]["warmup_epochs"] = 0
    cfg["hyp"]["mixup"] = 0.0
    cfg["hyp"]["scale"] = args.expert_scale_aug
    cfg["hyp"]["hsv_s"] = args.expert_hsv_s
    cfg["hyp"]["hsv_v"] = args.expert_hsv_v
    cfg.setdefault("Loss", {})
    cfg["Loss"]["box"] = float(spec.loss_box)
    return setup.write_config(f"{run_name}.yaml", cfg)


def train_experts(
    args: argparse.Namespace,
    setup,
    fedsto,
    scan_summary: Mapping[str, Any],
    experts: list[str],
) -> tuple[list[dict[str, str]], dict[str, Path]]:
    args.gpus = fedsto.resolve_gpus(args.gpus)
    fedsto.check_runtime_dependencies()
    specs = expert_specs(args)
    records: list[dict[str, str]] = []
    checkpoints: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []

    combined_lists = scan_summary.get("combined_train_lists", {})
    for index, expert in enumerate(experts):
        spec = specs[expert]
        pseudo_list = Path(combined_lists[expert])
        pseudo_images = count_list(pseudo_list)
        final_ckpt = args.workspace_root / "checkpoints" / f"06_counterfactual_{expert}_expert.pt"
        start = fedsto.GLOBAL_DIR / f"06_counterfactual_{expert}_start.pt"
        rows.append(
            {
                "expert": expert,
                "pseudo_list": str(pseudo_list),
                "pseudo_images": pseudo_images,
                "pseudo_repeat": spec.pseudo_repeat,
                "lr": spec.lr,
                "loss_box": spec.loss_box,
                "train_scope": spec.train_scope,
                "checkpoint": str(final_ckpt),
            }
        )
        if pseudo_images < args.min_expert_images:
            print(f"06 skip expert={expert}: pseudo images {pseudo_images} < {args.min_expert_images}")
            continue
        if not fedsto.checkpoint_matches_protocol(start, PROTOCOL_VERSION):
            fedsto.make_start_checkpoint(
                args.teacher_checkpoint,
                start,
                protocol=PROTOCOL_VERSION,
                stage=f"06_counterfactual_{expert}_start",
            )
        if not pl03.reusable_checkpoint(fedsto, final_ckpt, args.force):
            cfg = write_expert_train_config(setup, args, start, pseudo_list, spec)
            raw_ckpt = pl03.run_train(
                setup,
                fedsto,
                cfg,
                dry_run=args.dry_run,
                gpus=args.gpus,
                master_port=args.master_port + index,
            )
            if not args.dry_run:
                fedsto.mark_checkpoint_protocol(raw_ckpt, PROTOCOL_VERSION, f"06_counterfactual_{expert}_raw")
                fedsto.make_start_checkpoint(
                    raw_ckpt,
                    final_ckpt,
                    protocol=PROTOCOL_VERSION,
                    stage=f"06_counterfactual_{expert}_expert",
                )
                pl03.cleanup_training_artifacts(raw_ckpt, start)
        checkpoints[expert] = final_ckpt
        records.append(
            {
                "condition": "06_counterfactual_output_moe",
                "label": f"06_{expert}_expert",
                "kind": "aggregate",
                "round": "",
                "client": "",
                "variant": f"{spec.train_scope}:repeat={spec.pseudo_repeat}:box={spec.loss_box}",
                "path": str(final_ckpt.resolve()),
            }
        )

    write_csv(
        args.workspace_root / "stats" / "06_expert_training_checkpoints.csv",
        rows,
        ["expert", "pseudo_list", "pseudo_images", "pseudo_repeat", "lr", "loss_box", "train_scope", "checkpoint"],
    )
    return records, checkpoints


def build_eval_records(args: argparse.Namespace, expert_records: list[dict[str, str]]) -> list[dict[str, str]]:
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
    ]
    if args.router_teacher_checkpoint.exists():
        records.append(
            {
                "condition": "05_reference",
                "label": "05_expert_choice_router_aggregate",
                "kind": "aggregate",
                "round": "30",
                "client": "",
                "variant": "reference",
                "path": str(args.router_teacher_checkpoint.resolve()),
            }
        )
    records.extend(expert_records)
    write_csv(
        args.workspace_root / "stats" / "06_individual_checkpoints.csv",
        records,
        ["condition", "label", "kind", "round", "client", "variant", "path"],
    )
    return records


def write_individual_metrics(args: argparse.Namespace) -> list[dict[str, Any]]:
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
        args.workspace_root / "stats" / "06_individual_expert_metrics.csv",
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


def split_names(args: argparse.Namespace) -> list[str]:
    return [item.strip() for item in args.eval_splits.split(",") if item.strip()]


def split_config_path(args: argparse.Namespace, split: str) -> Path:
    config_root = args.workspace_root / "validation_reports" / "paper_protocol_configs"
    direct = config_root / f"{split}.yaml"
    if direct.exists():
        return direct
    if split == "total":
        scene_total = config_root / "scene_daynight_total.yaml"
        if scene_total.exists():
            return scene_total
    return direct


def evaluation_ready(args: argparse.Namespace) -> bool:
    summary = args.workspace_root / "validation_reports" / "paper_protocol_eval_summary.csv"
    if not summary.exists():
        return False
    return all(split_config_path(args, split).exists() for split in split_names(args))


def select_val_python() -> Path:
    return shared_eval.select_val_python(None)


def run_val_for_predictions(
    args: argparse.Namespace,
    checkpoint_label: str,
    checkpoint: Path,
    split: str,
    cfg: Path,
) -> Path:
    pred_root = args.workspace_root / "validation_reports" / "06_output_moe_predictions"
    save_dir = pred_root / safe_name(f"{checkpoint_label}_{split}")
    labels_dir = save_dir / "labels"
    if args.reuse_predictions and labels_dir.exists():
        return labels_dir
    if save_dir.exists():
        shutil.rmtree(save_dir)
    cmd = [
        str(select_val_python()),
        "val.py",
        "--weights",
        str(checkpoint.resolve()),
        "--cfg",
        str(cfg.resolve()),
        "--batch-size",
        str(args.val_batch_size),
        "--imgsz",
        str(args.imgsz),
        "--conf-thres",
        str(args.output_conf_thres),
        "--iou-thres",
        str(args.output_nms_iou_thres),
        "--project",
        str(pred_root.resolve()),
        "--name",
        save_dir.name,
        "--exist-ok",
        "--save-txt",
        "--save-conf",
        "--no-plots",
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    log_file = args.workspace_root / "validation_reports" / "06_output_moe_prediction_logs" / f"{save_dir.name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(cmd))
    if args.dry_run:
        return labels_dir
    result = subprocess.run(cmd, cwd=ET_ROOT, capture_output=True, text=True)
    log_file.write_text(result.stdout + "\nSTDERR\n" + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"val.py prediction export failed for {checkpoint_label}/{split}: {result.stderr[-1000:]}")
    return labels_dir


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        idx = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"Image path does not contain /images/: {image_path}") from exc
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def xywh_to_xyxy(box: Iterable[float]) -> np.ndarray:
    x, y, w, h = [float(value) for value in box]
    return np.array([x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0], dtype=np.float32)


def read_gt_label(path: Path) -> list[tuple[int, np.ndarray]]:
    if not path.exists():
        return []
    rows: list[tuple[int, np.ndarray]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        rows.append((int(float(parts[0])), xywh_to_xyxy([float(v) for v in parts[1:5]])))
    return rows


def read_pred_label(path: Path) -> list[tuple[int, np.ndarray, float]]:
    if not path.exists():
        return []
    rows: list[tuple[int, np.ndarray, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        rows.append((int(float(parts[0])), xywh_to_xyxy([float(v) for v in parts[1:5]]), float(parts[5])))
    return rows


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-9)


def image_router_weights(
    image_path: Path,
    expert_preds: Mapping[str, list[tuple[int, np.ndarray, float]]],
    enabled: tuple[str, ...],
) -> dict[str, float]:
    image = cv2.imread(str(image_path))
    if image is None:
        base = {expert: 1.0 for expert in enabled}
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_v = float(hsv[:, :, 2].mean()) / 255.0
        contrast = float(gray.std()) / 255.0
        low_light = float(np.clip((0.46 - mean_v) / 0.28, 0.0, 1.0))
        low_contrast = float(np.clip((0.18 - contrast) / 0.18, 0.0, 1.0))
        base = {
            "clean_original": 0.70 - 0.35 * low_light,
            "illumination_rescued": 0.20 + 0.55 * max(low_light, low_contrast),
            "cross_view_bridge": 0.10 + 0.25 * low_contrast,
        }

    clean_count = len(expert_preds.get("clean_original", []))
    illum_count = len(expert_preds.get("illumination_rescued", []))
    bridge_count = len(expert_preds.get("cross_view_bridge", []))
    if "illumination_rescued" in enabled and illum_count > clean_count:
        base["illumination_rescued"] = base.get("illumination_rescued", 0.0) + 0.15
    if "cross_view_bridge" in enabled and abs(illum_count - clean_count) >= 3 and bridge_count > 0:
        base["cross_view_bridge"] = base.get("cross_view_bridge", 0.0) + 0.10

    weights = {expert: max(0.0, float(base.get(expert, 0.0))) for expert in enabled}
    total = sum(weights.values())
    if total <= 0:
        return {expert: 1.0 / len(enabled) for expert in enabled}
    return {expert: value / total for expert, value in weights.items()}


def weighted_nms_fusion(
    boxes: list[tuple[int, np.ndarray, float, str]],
    *,
    iou_thr: float,
    score_thr: float,
) -> list[tuple[int, np.ndarray, float]]:
    fused: list[tuple[int, np.ndarray, float]] = []
    by_class: dict[int, list[tuple[np.ndarray, float, str]]] = defaultdict(list)
    for cls, box, score, expert in boxes:
        if score >= score_thr:
            by_class[cls].append((box, score, expert))

    for cls, cls_boxes in by_class.items():
        remaining = sorted(cls_boxes, key=lambda item: item[1], reverse=True)
        while remaining:
            seed_box, seed_score, _ = remaining.pop(0)
            if not remaining:
                fused.append((cls, seed_box, seed_score))
                continue
            rem_boxes = np.stack([item[0] for item in remaining], axis=0)
            ious = box_iou(seed_box, rem_boxes)
            group = [(seed_box, seed_score)]
            next_remaining: list[tuple[np.ndarray, float, str]] = []
            for item, iou in zip(remaining, ious):
                if float(iou) >= iou_thr:
                    group.append((item[0], item[1]))
                else:
                    next_remaining.append(item)
            weights = np.array([score for _, score in group], dtype=np.float32)
            box_stack = np.stack([box for box, _ in group], axis=0)
            fused_box = (box_stack * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1e-9)
            fused_score = min(1.0, max(score for _, score in group) + 0.04 * (len(group) - 1))
            fused.append((cls, fused_box.astype(np.float32), float(fused_score)))
            remaining = sorted(next_remaining, key=lambda item: item[1], reverse=True)
    return fused


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    return float(np.mean([mpre[mrec >= threshold].max() if np.any(mrec >= threshold) else 0.0 for threshold in np.linspace(0, 1, 101)]))


def evaluate_predictions(
    images: list[Path],
    predictions: Mapping[Path, list[tuple[int, np.ndarray, float]]],
    iou_thresholds: Iterable[float] | None = None,
) -> dict[str, float]:
    thresholds = np.array(list(iou_thresholds or np.arange(0.5, 1.0, 0.05)), dtype=np.float32)
    gt_by_image_class: dict[tuple[Path, int], list[np.ndarray]] = defaultdict(list)
    det_by_class: dict[int, list[tuple[Path, np.ndarray, float]]] = defaultdict(list)
    classes: set[int] = set()
    for image_path in images:
        for cls, box in read_gt_label(label_path_for_image(image_path)):
            gt_by_image_class[(image_path, cls)].append(box)
            classes.add(cls)
        for cls, box, score in predictions.get(image_path, []):
            det_by_class[cls].append((image_path, box, score))
            classes.add(cls)

    ap_by_thr: dict[float, list[float]] = {float(thr): [] for thr in thresholds}
    precision50: list[float] = []
    recall50: list[float] = []
    for cls in sorted(classes):
        gt_count = sum(len(v) for (image_path, gt_cls), v in gt_by_image_class.items() if gt_cls == cls)
        if gt_count <= 0:
            continue
        detections = sorted(det_by_class.get(cls, []), key=lambda item: item[2], reverse=True)
        for threshold in thresholds:
            matched: dict[Path, np.ndarray] = {
                image_path: np.zeros(len(gt_boxes), dtype=bool)
                for (image_path, gt_cls), gt_boxes in gt_by_image_class.items()
                if gt_cls == cls
            }
            tp = np.zeros(len(detections), dtype=np.float32)
            fp = np.zeros(len(detections), dtype=np.float32)
            for det_idx, (image_path, det_box, _score) in enumerate(detections):
                gt_boxes = gt_by_image_class.get((image_path, cls), [])
                if not gt_boxes:
                    fp[det_idx] = 1.0
                    continue
                gt_stack = np.stack(gt_boxes, axis=0)
                ious = box_iou(det_box, gt_stack)
                best_idx = int(ious.argmax()) if ious.size else -1
                if best_idx >= 0 and float(ious[best_idx]) >= float(threshold) and not matched[image_path][best_idx]:
                    matched[image_path][best_idx] = True
                    tp[det_idx] = 1.0
                else:
                    fp[det_idx] = 1.0
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / max(float(gt_count), 1e-9)
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
            ap_by_thr[float(threshold)].append(compute_ap(recall, precision))
            if abs(float(threshold) - 0.5) < 1e-6 and recall.size:
                precision50.append(float(precision[-1]))
                recall50.append(float(recall[-1]))

    map50 = float(np.mean(ap_by_thr.get(0.5, [0.0]))) if ap_by_thr.get(0.5) else 0.0
    map5095 = float(np.mean([ap for values in ap_by_thr.values() for ap in values])) if any(ap_by_thr.values()) else 0.0
    return {
        "images": float(len(images)),
        "precision": float(np.mean(precision50)) if precision50 else 0.0,
        "recall": float(np.mean(recall50)) if recall50 else 0.0,
        "map50": map50,
        "map50_95": map5095,
    }


def run_output_moe(args: argparse.Namespace, expert_checkpoints: Mapping[str, Path]) -> list[dict[str, Any]]:
    enabled_experts = tuple(expert for expert in EXPERT_ORDER if expert in expert_checkpoints)
    if len(enabled_experts) < 2:
        return []
    result_rows: list[dict[str, Any]] = []
    pred_dirs: dict[tuple[str, str], Path] = {}
    for expert, checkpoint in expert_checkpoints.items():
        if expert not in EXPERT_ORDER:
            continue
        for split in split_names(args):
            cfg = split_config_path(args, split)
            if not cfg.exists():
                raise FileNotFoundError(f"Missing eval config for split={split}: {cfg}")
            pred_dirs[(expert, split)] = run_val_for_predictions(args, f"06_{expert}_expert", checkpoint, split, cfg)

    variants: dict[str, tuple[str, ...]] = {
        "06_output_moe_clean_illum": tuple(expert for expert in ("clean_original", "illumination_rescued") if expert in enabled_experts),
        "06_output_moe_clean_illum_bridge": tuple(expert for expert in EXPERT_ORDER if expert in enabled_experts),
    }
    for label, variant_experts in variants.items():
        if len(variant_experts) < 2:
            continue
        by_split_rows: list[dict[str, Any]] = []
        for split in split_names(args):
            cfg = split_config_path(args, split)
            image_list = Path(yaml.safe_load(cfg.read_text(encoding="utf-8"))["Dataset"]["val"])
            images = [Path(line.strip()) for line in image_list.read_text(encoding="utf-8").splitlines() if line.strip()]
            predictions: dict[Path, list[tuple[int, np.ndarray, float]]] = {}
            for image_idx, image_path in enumerate(images, start=1):
                per_expert_preds: dict[str, list[tuple[int, np.ndarray, float]]] = {}
                for expert in variant_experts:
                    pred_path = pred_dirs[(expert, split)] / f"{image_path.stem}.txt"
                    preds = read_pred_label(pred_path)
                    if args.output_max_preds_per_expert_image > 0:
                        preds = sorted(preds, key=lambda item: item[2], reverse=True)[
                            : args.output_max_preds_per_expert_image
                        ]
                    per_expert_preds[expert] = preds
                weights = image_router_weights(image_path, per_expert_preds, variant_experts)
                weighted_boxes: list[tuple[int, np.ndarray, float, str]] = []
                for expert, preds in per_expert_preds.items():
                    for cls, box, score in preds:
                        weighted_boxes.append((cls, box, score * weights[expert], expert))
                fused = weighted_nms_fusion(
                    weighted_boxes,
                    iou_thr=args.output_wbf_iou,
                    score_thr=args.output_score_thres,
                )
                if args.output_max_fused_per_image > 0:
                    fused = sorted(fused, key=lambda item: item[2], reverse=True)[: args.output_max_fused_per_image]
                predictions[image_path] = fused
                if args.progress_every and image_idx % args.progress_every == 0:
                    print(f"[output-moe] {label}/{split}: fused {image_idx}/{len(images)} images")
            metrics = evaluate_predictions(images, predictions)
            row = {
                "checkpoint_label": label,
                "split": split,
                "status": "ok",
                "images": f"{metrics['images']:.0f}",
                "precision": f"{metrics['precision']:.6f}",
                "recall": f"{metrics['recall']:.6f}",
                "map50": f"{metrics['map50']:.6f}",
                "map50_95": f"{metrics['map50_95']:.6f}",
                "experts": ",".join(variant_experts),
            }
            by_split_rows.append(row)
            result_rows.append(row)

        by_label_split = {(row["checkpoint_label"], row["split"]): row for row in by_split_rows}
        total = next((row for row in by_split_rows if row["split"] in {"total", "scene_daynight_total"}), None)
        if total:
            gap = base01_0.split_gap_metrics(by_label_split, total["checkpoint_label"])
            result_rows.append(
                {
                    "checkpoint_label": total["checkpoint_label"],
                    "split": "summary",
                    "status": "ok",
                    "images": total["images"],
                    "precision": total["precision"],
                    "recall": total["recall"],
                    "map50": total["map50"],
                    "map50_95": total["map50_95"],
                    "experts": total["experts"],
                    **gap,
                }
            )

    fieldnames = [
        "checkpoint_label",
        "split",
        "status",
        "images",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "experts",
        "worst_split",
        "worst_split_map50_95",
        "day_avg_map50_95",
        "night_avg_map50_95",
        "day_night_gap_map50_95",
    ]
    write_csv(args.workspace_root / "stats" / "06_output_moe_metrics.csv", result_rows, fieldnames)
    return result_rows


def write_report(
    args: argparse.Namespace,
    scan_summary: Mapping[str, Any],
    individual_metrics: list[dict[str, Any]],
    output_moe_metrics: list[dict[str, Any]],
    elapsed_seconds: float,
) -> None:
    totals = scan_summary.get("totals", {})
    signal = scan_summary.get("day_night_signal", {})
    lines = [
        "# Scene-Daynight DQA 06: Counterfactual Output-MoE",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- protocol: {PROTOCOL_VERSION}",
        f"- elapsed: {seconds_to_hms(elapsed_seconds)}",
        f"- workspace: `{args.workspace_root}`",
        f"- teacher: `{args.teacher_checkpoint}`",
        "",
        "## Counterfactual pseudoGT scan",
        "",
        f"- clean_original boxes: {totals.get('clean_original_boxes', '')}",
        f"- illumination_rescued boxes: {totals.get('illumination_rescued_boxes', '')}",
        f"- cross_view_bridge boxes: {totals.get('cross_view_bridge_boxes', '')}",
        f"- rescued ratio: {as_float(totals.get('rescued_ratio'), 0.0):.3f}",
        f"- day rescued ratio: {as_float(signal.get('day_rescued_ratio'), 0.0):.3f}",
        f"- night rescued ratio: {as_float(signal.get('night_rescued_ratio'), 0.0):.3f}",
        f"- night-day rescued gap: {as_float(signal.get('night_minus_day_rescued_ratio'), 0.0):.3f}",
        "",
        "## Individual checkpoints",
        "",
        "| checkpoint | mAP50 | mAP50:95 | day avg | night avg | worst split |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in individual_metrics:
        lines.append(
            f"| {row['checkpoint_label']} | {row['map50']} | {row['map50_95']} | "
            f"{row['day_avg_map50_95']} | {row['night_avg_map50_95']} | {row['worst_split']} |"
        )

    summaries = [row for row in output_moe_metrics if row.get("split") == "summary"]
    lines.extend(
        [
            "",
            "## Output-space MoE",
            "",
            "| output MoE | mAP50 | mAP50:95 | day avg | night avg | worst split | experts |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['checkpoint_label']} | {row['map50']} | {row['map50_95']} | "
            f"{row.get('day_avg_map50_95', '')} | {row.get('night_avg_map50_95', '')} | "
            f"{row.get('worst_split', '')} | {row.get('experts', '')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation hook",
            "",
            "06 succeeds if the output-space MoE improves over the best individual expert and closes the 05 night-domain drop without relying on residual checkpoint mixing.  If `illumination_rescued` is strong alone but fusion is weak, the next move is router calibration.  If all individual experts are weak, the bottleneck is expert data construction rather than output MoE.",
        ]
    )
    (args.workspace_root / "06_counterfactual_output_moe_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def notify(args: argparse.Namespace, message: str, *, title: str, status: str = "", error: str | None = None) -> None:
    try:
        from notebook_notify import notify_discord

        context: dict[str, Any] = {
            "workspace": str(args.workspace_root.expanduser().resolve()),
            "status": status,
            "report": str(args.workspace_root.expanduser().resolve() / "06_counterfactual_output_moe_report.md"),
        }
        if error:
            context["error"] = error[:500]
        metrics_path = args.workspace_root.expanduser().resolve() / "stats" / "06_output_moe_metrics.csv"
        if metrics_path.exists():
            context["output_moe_metrics_csv"] = str(metrics_path)
        individual_path = args.workspace_root.expanduser().resolve() / "stats" / "06_individual_expert_metrics.csv"
        if individual_path.exists():
            context["individual_metrics_csv"] = str(individual_path)
        print(notify_discord(message, title=title, context=context, fail_silently=True))
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}")


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
    parser.add_argument("--experts", default="clean_original,illumination_rescued,cross_view_bridge")
    parser.add_argument("--expert-epochs", type=int, default=1)
    parser.add_argument("--expert-train-scope", choices=["neck_head", "all"], default="neck_head")
    parser.add_argument("--expert-orthogonal-weight", type=float, default=1e-4)
    parser.add_argument("--clean-pseudo-repeat", type=int, default=1)
    parser.add_argument("--illumination-pseudo-repeat", type=int, default=2)
    parser.add_argument("--bridge-pseudo-repeat", type=int, default=2)
    parser.add_argument("--hybrid-pseudo-repeat", type=int, default=1)
    parser.add_argument("--clean-lr", type=float, default=0.0007)
    parser.add_argument("--illumination-lr", type=float, default=0.0006)
    parser.add_argument("--bridge-lr", type=float, default=0.0005)
    parser.add_argument("--hybrid-lr", type=float, default=0.0006)
    parser.add_argument("--clean-loss-box", type=float, default=0.005)
    parser.add_argument("--illumination-loss-box", type=float, default=0.003)
    parser.add_argument("--bridge-loss-box", type=float, default=0.002)
    parser.add_argument("--hybrid-loss-box", type=float, default=0.003)
    parser.add_argument("--expert-scale-aug", type=float, default=0.25)
    parser.add_argument("--expert-hsv-s", type=float, default=0.35)
    parser.add_argument("--expert-hsv-v", type=float, default=0.20)
    parser.add_argument("--min-expert-images", type=int, default=20)
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=33161)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--output-moe", action="store_true")
    parser.add_argument("--eval-splits", default=DEFAULT_EVAL_SPLITS)
    parser.add_argument("--eval-clients", action="store_true")
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--classwise", action="store_true")
    parser.add_argument("--no-eval-plots", action="store_true")
    parser.add_argument("--output-conf-thres", type=float, default=0.001)
    parser.add_argument("--output-nms-iou-thres", type=float, default=0.60)
    parser.add_argument("--output-wbf-iou", type=float, default=0.55)
    parser.add_argument("--output-score-thres", type=float, default=0.001)
    parser.add_argument("--output-max-preds-per-expert-image", type=int, default=100)
    parser.add_argument("--output-max-fused-per-image", type=int, default=150)
    parser.add_argument("--reuse-predictions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-pseudo", action="store_true")
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--notify-start", action="store_true")
    parser.add_argument("--notify-end", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    start_time = time.monotonic()
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.teacher_checkpoint = args.teacher_checkpoint.expanduser().resolve()
    args.router_teacher_checkpoint = args.router_teacher_checkpoint.expanduser().resolve()
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    (args.workspace_root / "stats").mkdir(parents=True, exist_ok=True)

    setup, fedsto, manifest, clients = prepare_workspace(args)
    selected_experts = resolve_experts(args.experts)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "workspace": str(args.workspace_root),
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "router_teacher_checkpoint": str(args.router_teacher_checkpoint),
        "server": manifest.get("server"),
        "clients": clients,
        "experts": selected_experts,
        "evaluate": args.evaluate,
        "output_moe": args.output_moe,
    }
    (args.workspace_root / "stats" / "06_counterfactual_output_moe_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    scan_path = args.workspace_root / "stats" / "06_counterfactual_scan_summary.json"
    if scan_path.exists() and not args.force_pseudo:
        print(f"Reusing counterfactual scan summary: {scan_path}")
        scan_summary = json.loads(scan_path.read_text(encoding="utf-8"))
    else:
        scan_summary = moe09.scan_counterfactual_views(args, setup, clients)
        scan_path.write_text(
            json.dumps(scan_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    expert_records, expert_checkpoints = train_experts(args, setup, fedsto, scan_summary, selected_experts)
    records = build_eval_records(args, expert_records)
    individual_metrics: list[dict[str, Any]] = []
    output_moe_metrics: list[dict[str, Any]] = []

    if (args.evaluate or args.output_moe) and not evaluation_ready(args):
        base01_0.run_evaluation(args, records)
    elif args.evaluate or args.output_moe:
        print("Reusing existing paper-protocol evaluation summary and configs.")
    if args.evaluate or args.output_moe:
        individual_metrics = write_individual_metrics(args)
    if args.output_moe:
        output_moe_metrics = run_output_moe(args, expert_checkpoints)

    elapsed = time.monotonic() - start_time
    write_report(args, scan_summary, individual_metrics, output_moe_metrics, elapsed)
    return {
        "status": "ok",
        "elapsed_seconds": elapsed,
        "elapsed_hms": seconds_to_hms(elapsed),
        "workspace": str(args.workspace_root),
        "individual_metrics": individual_metrics,
        "output_moe_metrics": output_moe_metrics,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.notify or args.notify_start:
        notify(args, "Scene-daynight DQA 06 counterfactual output-MoE started.", title="DQA 06 start", status="started")
    status = "success"
    error: str | None = None
    try:
        result = run(args)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = str(exc)
        raise
    finally:
        if args.notify or args.notify_end:
            notify(
                args,
                f"Scene-daynight DQA 06 counterfactual output-MoE finished with status={status}.",
                title="DQA 06 finish",
                status=status,
                error=error,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
