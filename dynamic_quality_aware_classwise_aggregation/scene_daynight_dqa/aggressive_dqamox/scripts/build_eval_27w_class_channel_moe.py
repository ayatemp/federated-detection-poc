#!/usr/bin/env python3
"""Evaluate 27w class-channel MoE for scene-daynight DQA."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

import build_eval_27h_model_level_moe as h27
import build_eval_27t_path_domain_routed_moe as t27
import build_eval_27u_coco_bridge_moe as u27
import build_eval_27v_consensus_wbf_moe as v27


SCRIPT_PATH = Path(__file__).resolve()
AGG_ROOT = SCRIPT_PATH.parents[1]
REPORTS_ROOT = AGG_ROOT / "reports"
DEFAULT_WORKSPACE = AGG_ROOT / "output" / "27w_class_channel_moe"
NOTEBOOK_PATH = AGG_ROOT / "notebooks" / "research_loop_until_060" / "017_27w_class_channel_moe.ipynb"

if str(h27.ET_ROOT) not in sys.path:
    sys.path.insert(0, str(h27.ET_ROOT))

from configs.defaults import get_cfg  # noqa: E402
from utils.datasets import LoadImagesAndLabels  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402


CLASS_NAMES = [
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

SOURCE_ORDER = [
    "ref",
    "ref_softnms",
    "union_strict_s020",
    "union_strict_s035",
    "union_strict_s050",
    "union_riderdup_s030",
    "coco_strict_s050_only",
    "coco_rare_rider_s045_only",
]

SOURCE_IDEAS = {
    "ref": "current best 1024px brightness-routed DQA-MoE",
    "ref_softnms": "same routed DQA-MoE, but Gaussian Soft-NMS keeps close boxes alive",
    "union_strict_s020": "DQA reference plus low-score strict COCO bridge x0.20",
    "union_strict_s035": "DQA reference plus moderate strict COCO bridge x0.35",
    "union_strict_s050": "DQA reference plus aggressive strict COCO bridge x0.50",
    "union_riderdup_s030": "DQA reference plus COCO remap with person->rider duplicate x0.30",
    "coco_strict_s050_only": "COCO bridge only, strict BDD remap x0.50",
    "coco_rare_rider_s045_only": "COCO bridge only for rare/rider-like BDD classes x0.45",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--target-map50", type=float, default=0.60)
    parser.add_argument("--previous-best-map50", type=float, default=0.529594)
    parser.add_argument("--previous-best-map50-95", type=float, default=0.296321)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--merge-iou", type=float, default=0.50)
    parser.add_argument("--device", default="")
    parser.add_argument("--gate-images", type=int, default=240)
    parser.add_argument("--min-gate-gain", type=float, default=0.012)
    parser.add_argument("--class-min-ap-gain", type=float, default=0.015)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--no-discord", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    return parser.parse_args(argv)


def scale_det(det: torch.Tensor, scale: float) -> torch.Tensor:
    if det.numel() == 0:
        return det.reshape(0, 6)
    out = det.clone()
    out[:, 4] *= float(scale)
    return out


def nms_or_empty(parts: list[torch.Tensor], merge_iou: float, max_det: int) -> torch.Tensor:
    parts = [part for part in parts if part.numel()]
    if not parts:
        return torch.zeros((0, 6), dtype=torch.float32)
    merged = torch.cat(parts, dim=0)
    return t27.nms_detections(merged, merge_iou, max_det).float().cpu()


def infer_base_parts(
    *,
    models: dict,
    image: np.ndarray,
    imgsz: int,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        ref = t27.infer_bgr_batch(
            model=models["routed_day_light_night_hard"],
            images=[image],
            imgsz=imgsz,
            device=device,
            conf_thres=args.conf_thres,
            iou_thres=0.50,
            max_det=args.max_det,
        )[0].detach()
        coco_raw = t27.infer_bgr_batch(
            model=models["coco80_efficient_yolov5l"],
            images=[image],
            imgsz=imgsz,
            device=device,
            conf_thres=args.conf_thres,
            iou_thres=0.50,
            max_det=args.max_det,
        )[0].detach()
    return {
        "ref": ref,
        "coco_strict": u27.remap_coco_detections(coco_raw, u27.STRICT_COCO_TO_BDD),
        "coco_riderdup": u27.remap_coco_detections(coco_raw, u27.RIDER_DUP_COCO_TO_BDD),
    }


def build_source_predictions(parts: dict[str, torch.Tensor], args: argparse.Namespace) -> dict[str, torch.Tensor]:
    ref = parts["ref"]
    coco_strict = parts["coco_strict"]
    coco_riderdup = parts["coco_riderdup"]
    rare_classes = [0, 1, 5, 6, 7, 8, 9]
    return {
        "ref": nms_or_empty([ref], args.merge_iou, args.max_det),
        "ref_softnms": v27.soft_nms_detections(
            ref.clone(),
            {"soft_nms_sigma": 0.50, "soft_nms_score_thr": 0.0001},
            args.max_det,
        ).float().cpu(),
        "union_strict_s020": nms_or_empty([ref, scale_det(coco_strict, 0.20)], args.merge_iou, args.max_det),
        "union_strict_s035": nms_or_empty([ref, scale_det(coco_strict, 0.35)], args.merge_iou, args.max_det),
        "union_strict_s050": nms_or_empty([ref, scale_det(coco_strict, 0.50)], args.merge_iou, args.max_det),
        "union_riderdup_s030": nms_or_empty([ref, scale_det(coco_riderdup, 0.30)], args.merge_iou, args.max_det),
        "coco_strict_s050_only": nms_or_empty([scale_det(coco_strict, 0.50)], args.merge_iou, args.max_det),
        "coco_rare_rider_s045_only": nms_or_empty(
            [scale_det(v27.class_filter_detections(coco_riderdup, rare_classes), 0.45)],
            args.merge_iou,
            args.max_det,
        ),
    }


def predictions_from_class_policy(
    source_predictions: dict[str, torch.Tensor],
    class_policy: dict[int, str],
    args: argparse.Namespace,
) -> torch.Tensor:
    parts = []
    for cls, source in class_policy.items():
        det = source_predictions[source]
        if det.numel() == 0:
            continue
        cls_det = det[det[:, 5].long() == int(cls)]
        if cls_det.numel():
            parts.append(cls_det)
    return nms_or_empty(parts, args.merge_iou, args.max_det)


def packed_stats(stats: list, niou: int) -> list[np.ndarray]:
    if stats:
        return [np.concatenate(x, 0) for x in zip(*stats)]
    return [np.zeros((0, niou), dtype=bool), np.array([]), np.array([]), np.array([])]


def summarize_stats_with_classes(stats: list, niou: int, nc: int = 10) -> tuple[dict, list[dict]]:
    packed = packed_stats(stats, niou)
    label_counts = np.bincount(packed[3].astype(int), minlength=nc) if packed[3].size else np.zeros(nc, dtype=int)
    class_rows = [
        {
            "class_id": cls,
            "class_name": CLASS_NAMES[cls],
            "label_count": int(label_counts[cls]),
            "ap50": 0.0,
            "ap50_95": 0.0,
        }
        for cls in range(nc)
    ]
    if len(packed) and packed[0].any():
        p, r, ap, _f1, ap_class, _cls_thr = t27.ap_per_class(*packed, plot=False, names={})
        ap50 = ap[:, 0]
        ap_all = ap.mean(1)
        mp, mr, map50, map95 = p.mean(), r.mean(), ap50.mean(), ap_all.mean()
        for row_idx, cls in enumerate(ap_class.astype(int)):
            if 0 <= cls < nc:
                class_rows[cls]["ap50"] = round(float(ap50[row_idx]), 6)
                class_rows[cls]["ap50_95"] = round(float(ap_all[row_idx]), 6)
        summary = {
            "precision": round(float(mp), 6),
            "recall": round(float(mr), 6),
            "map50": round(float(map50), 6),
            "map50_95": round(float(map95), 6),
            "num_ap_classes": int(len(ap_class)),
        }
    else:
        summary = {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map50_95": 0.0, "num_ap_classes": 0}
    return summary, class_rows


def evaluate_sources(
    *,
    models: dict,
    dataset: LoadImagesAndLabels,
    indices: list[int],
    imgsz: int,
    device: torch.device,
    args: argparse.Namespace,
    phase: str,
) -> tuple[list[dict], list[dict]]:
    iouv = torch.linspace(0.5, 0.95, 10)
    niou = iouv.numel()
    total_stats = {source: [] for source in SOURCE_ORDER}
    split_stats = {source: {split: [] for split in t27.DOMAIN_SPLITS} for source in SOURCE_ORDER}
    split_counts = {split: {"images": 0, "labels": 0} for split in t27.DOMAIN_SPLITS}
    total_labels = 0

    for idx in tqdm(indices, desc=phase):
        path = dataset.img_files[idx]
        split = t27.split_from_path(path)
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        h0, w0 = image.shape[:2]
        labelsn = t27.native_labels(dataset.labels[idx], (w0, h0))
        total_labels += len(labelsn)
        split_counts[split]["images"] += 1
        split_counts[split]["labels"] += len(labelsn)
        source_predictions = build_source_predictions(
            infer_base_parts(models=models, image=image, imgsz=imgsz, device=device, args=args),
            args,
        )
        for source, pred in source_predictions.items():
            t27.add_stats(total_stats[source], pred, labelsn, iouv)
            t27.add_stats(split_stats[source][split], pred, labelsn, iouv)

    rows: list[dict] = []
    class_rows: list[dict] = []
    for source in SOURCE_ORDER:
        summary, cls_rows = summarize_stats_with_classes(total_stats[source], niou)
        rows.append(
            {
                "candidate": source,
                "phase": phase,
                "split": "scene_daynight_total",
                "images": len(indices),
                "labels": int(total_labels),
                **summary,
                "imgsz": imgsz,
                "merge_iou": args.merge_iou,
                "class_policy": "",
                "route_summary": SOURCE_IDEAS[source],
                "idea": "Source candidate for gate-learned class-channel MoE.",
            }
        )
        for cls_row in cls_rows:
            class_rows.append(
                {
                    "phase": phase,
                    "candidate": source,
                    "class_id": cls_row["class_id"],
                    "class_name": cls_row["class_name"],
                    "label_count": cls_row["label_count"],
                    "ap50": cls_row["ap50"],
                    "ap50_95": cls_row["ap50_95"],
                    "selected_source": "",
                    "ap50_gain_vs_ref": "",
                }
            )
        for split in t27.DOMAIN_SPLITS:
            split_summary, _ = summarize_stats_with_classes(split_stats[source][split], niou)
            rows.append(
                {
                    "candidate": source,
                    "phase": phase,
                    "split": split,
                    "images": split_counts[split]["images"],
                    "labels": int(split_counts[split]["labels"]),
                    **split_summary,
                    "imgsz": imgsz,
                    "merge_iou": args.merge_iou,
                    "class_policy": "",
                    "route_summary": SOURCE_IDEAS[source],
                    "idea": "Source candidate for gate-learned class-channel MoE.",
                }
            )
    return rows, class_rows


def class_ap_lookup(class_rows: list[dict]) -> dict[str, dict[int, dict]]:
    lookup: dict[str, dict[int, dict]] = {}
    for row in class_rows:
        lookup.setdefault(row["candidate"], {})[int(row["class_id"])] = row
    return lookup


def select_class_policy(
    *,
    lookup: dict[str, dict[int, dict]],
    min_gain: float,
    guarded: bool,
) -> tuple[dict[int, str], list[dict]]:
    policy: dict[int, str] = {}
    details: list[dict] = []
    for cls in range(len(CLASS_NAMES)):
        ref_row = lookup["ref"][cls]
        ref_ap = float(ref_row["ap50"])
        if int(ref_row["label_count"]) <= 0:
            selected = "ref"
            best_gain = 0.0
        else:
            best = max(
                (lookup[source][cls] for source in SOURCE_ORDER),
                key=lambda row: (float(row["ap50"]), float(row["ap50_95"])),
            )
            best_gain = float(best["ap50"]) - ref_ap
            selected = str(best["candidate"])
            if guarded and best_gain < min_gain:
                selected = "ref"
        policy[cls] = selected
        details.append(
            {
                "class_id": cls,
                "class_name": CLASS_NAMES[cls],
                "label_count": int(ref_row["label_count"]),
                "selected_source": selected,
                "ref_ap50": round(ref_ap, 6),
                "selected_ap50": round(float(lookup[selected][cls]["ap50"]), 6),
                "ap50_gain_vs_ref": round(float(lookup[selected][cls]["ap50"]) - ref_ap, 6),
            }
        )
    return policy, details


def summarize_policy(policy: dict[int, str]) -> str:
    groups: dict[str, list[str]] = {}
    for cls, source in policy.items():
        groups.setdefault(source, []).append(CLASS_NAMES[cls])
    return "; ".join(f"{source}: {', '.join(classes)}" for source, classes in sorted(groups.items()))


def evaluate_class_policy(
    *,
    models: dict,
    dataset: LoadImagesAndLabels,
    indices: list[int],
    imgsz: int,
    device: torch.device,
    args: argparse.Namespace,
    phase: str,
    candidate_label: str,
    policy: dict[int, str],
) -> tuple[list[dict], list[dict]]:
    iouv = torch.linspace(0.5, 0.95, 10)
    niou = iouv.numel()
    total_stats: list = []
    split_stats = {split: [] for split in t27.DOMAIN_SPLITS}
    split_counts = {split: {"images": 0, "labels": 0} for split in t27.DOMAIN_SPLITS}
    total_labels = 0

    for idx in tqdm(indices, desc=f"{phase}_{candidate_label}"):
        path = dataset.img_files[idx]
        split = t27.split_from_path(path)
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        h0, w0 = image.shape[:2]
        labelsn = t27.native_labels(dataset.labels[idx], (w0, h0))
        total_labels += len(labelsn)
        split_counts[split]["images"] += 1
        split_counts[split]["labels"] += len(labelsn)
        source_predictions = build_source_predictions(
            infer_base_parts(models=models, image=image, imgsz=imgsz, device=device, args=args),
            args,
        )
        pred = predictions_from_class_policy(source_predictions, policy, args)
        t27.add_stats(total_stats, pred, labelsn, iouv)
        t27.add_stats(split_stats[split], pred, labelsn, iouv)

    summary, cls_rows = summarize_stats_with_classes(total_stats, niou)
    route_summary = summarize_policy(policy)
    rows = [
        {
            "candidate": candidate_label,
            "phase": phase,
            "split": "scene_daynight_total",
            "images": len(indices),
            "labels": int(total_labels),
            **summary,
            "imgsz": imgsz,
            "merge_iou": args.merge_iou,
            "class_policy": route_summary,
            "route_summary": route_summary,
            "idea": "Gate-learned class-channel MoE: choose the best output source independently per BDD class.",
        }
    ]
    for split in t27.DOMAIN_SPLITS:
        split_summary, _ = summarize_stats_with_classes(split_stats[split], niou)
        rows.append(
            {
                "candidate": candidate_label,
                "phase": phase,
                "split": split,
                "images": split_counts[split]["images"],
                "labels": int(split_counts[split]["labels"]),
                **split_summary,
                "imgsz": imgsz,
                "merge_iou": args.merge_iou,
                "class_policy": route_summary,
                "route_summary": route_summary,
                "idea": "Gate-learned class-channel MoE: choose the best output source independently per BDD class.",
            }
        )
    class_rows = [
        {
            "phase": phase,
            "candidate": candidate_label,
            "class_id": row["class_id"],
            "class_name": row["class_name"],
            "label_count": row["label_count"],
            "ap50": row["ap50"],
            "ap50_95": row["ap50_95"],
            "selected_source": policy[int(row["class_id"])],
            "ap50_gain_vs_ref": "",
        }
        for row in cls_rows
    ]
    return rows, class_rows


def append_research_summary(
    *,
    workspace: Path,
    best: dict,
    status: str,
    args: argparse.Namespace,
    rationale_suffix: str = "",
) -> None:
    path = REPORTS_ROOT / "27_research_loop_mAP_summary.csv"
    fieldnames = [
        "trial",
        "status",
        "best_map50",
        "best_map50_95",
        "warmup_map50",
        "repair_map50",
        "dqa_aggregate_map50",
        "dqa_repair_map50",
        "workspace",
        "notebook",
        "log",
        "finished_utc",
        "rationale",
    ]
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "trial": workspace.name,
                "status": status,
                "best_map50": best.get("map50", ""),
                "best_map50_95": best.get("map50_95", ""),
                "warmup_map50": args.previous_best_map50,
                "repair_map50": "",
                "dqa_aggregate_map50": "",
                "dqa_repair_map50": "",
                "workspace": str(workspace),
                "notebook": str(NOTEBOOK_PATH),
                "log": str(workspace / "stats" / "27w_class_channel_moe_metrics.csv"),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "rationale": (
                    "27v showed WBF/Soft-NMS consensus improved only +0.007 on the gate and was too slow. "
                    "27w uses class-channel routing: per-class gate AP chooses among reference DQA, Soft-NMS, "
                    "COCO bridge unions, and COCO-only rare-class sources before a guarded full evaluation."
                    + rationale_suffix
                ),
            }
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    stats_dir = workspace / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    setup = h27.load_scene_setup(workspace)
    manifest = setup.build_data_lists()
    split_specs = h27.select_split_specs(manifest["paper_evaluation"], h27.PAPER_SPLITS)
    total_split = split_specs[-1]
    split_cfg = h27.write_eval_config(setup, workspace, total_split, args)
    val_cfg = get_cfg()
    val_cfg.merge_from_file(str(split_cfg))

    specs, missing, routed_ref = u27.make_model_specs(workspace)
    if missing:
        raise RuntimeError(f"Missing checkpoint inputs: {missing}")

    device = select_device(args.device, batch_size=1)
    dataset = LoadImagesAndLabels(
        val_cfg.Dataset.val,
        img_size=args.imgsz,
        batch_size=1,
        rect=False,
        stride=32,
        pad=0.0,
        cfg=val_cfg,
        prefix="27w: ",
    )
    gate_indices = t27.even_indices(len(dataset), args.gate_images)
    labels = ["routed_day_light_night_hard", "coco80_efficient_yolov5l"]
    models, imgsz = u27.load_models(specs=specs, labels=labels, device=device, data=split_cfg, imgsz=args.imgsz)

    metrics_csv = stats_dir / "27w_class_channel_moe_metrics.csv"
    class_csv = stats_dir / "27w_class_channel_moe_class_ap.csv"
    policy_json = stats_dir / "27w_class_channel_moe_policy.json"
    fieldnames = [
        "candidate",
        "phase",
        "split",
        "images",
        "labels",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "num_ap_classes",
        "imgsz",
        "merge_iou",
        "class_policy",
        "route_summary",
        "idea",
    ]
    class_fieldnames = [
        "phase",
        "candidate",
        "class_id",
        "class_name",
        "label_count",
        "ap50",
        "ap50_95",
        "selected_source",
        "ap50_gain_vs_ref",
    ]

    manifest_path = stats_dir / "27w_class_channel_moe_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "workspace": str(workspace),
                "dataset_size": len(dataset),
                "gate_images": len(gate_indices),
                "routed_reference": str(routed_ref),
                "coco_expert": str(u27.COCO_EFFICIENT_YOLOV5L),
                "source_order": SOURCE_ORDER,
                "papers": [
                    "Weighted Boxes Fusion and late-fusion detector ensembles motivate preserving per-detector outputs.",
                    "Model-level detection MoE and FedDG-MoE motivate routing by domain or task rather than averaging weights.",
                    "27u/27v results motivate class-conditional routing because COCO helped some gate slices while hurting full precision.",
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    if not args.no_discord:
        t27.notify(
            "27w started: class-channel gate-learned MoE. "
            f"Gate={len(gate_indices)} images; full run only if class policy gate gain >= {args.min_gate_gain:.3f}.",
            "DQA-MoX 27w started",
        )

    rows: list[dict] = []
    class_rows: list[dict] = []
    source_rows, source_class_rows = evaluate_sources(
        models=models,
        dataset=dataset,
        indices=gate_indices,
        imgsz=imgsz,
        device=device,
        args=args,
        phase="gate_sources",
    )
    rows.extend(source_rows)
    class_rows.extend(source_class_rows)
    t27.write_rows(metrics_csv, rows, fieldnames)
    t27.write_rows(class_csv, class_rows, class_fieldnames)

    lookup = class_ap_lookup(source_class_rows)
    raw_policy, raw_details = select_class_policy(lookup=lookup, min_gain=0.0, guarded=False)
    guarded_policy, guarded_details = select_class_policy(
        lookup=lookup,
        min_gain=args.class_min_ap_gain,
        guarded=True,
    )
    strict_policy, strict_details = select_class_policy(lookup=lookup, min_gain=args.class_min_ap_gain * 2.0, guarded=True)

    policies = {
        "class_channel_raw": (raw_policy, raw_details),
        f"class_channel_guarded_gain{int(args.class_min_ap_gain * 1000):03d}": (guarded_policy, guarded_details),
        f"class_channel_strict_gain{int(args.class_min_ap_gain * 2000):03d}": (strict_policy, strict_details),
    }
    policy_json.write_text(
        json.dumps(
            {
                "raw": raw_details,
                "guarded": guarded_details,
                "strict": strict_details,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    for label, (policy, details) in policies.items():
        policy_rows, policy_class_rows = evaluate_class_policy(
            models=models,
            dataset=dataset,
            indices=gate_indices,
            imgsz=imgsz,
            device=device,
            args=args,
            phase=f"gate_{label}",
            candidate_label=label,
            policy=policy,
        )
        for row in policy_class_rows:
            selected = policy[int(row["class_id"])]
            ref_ap = float(lookup["ref"][int(row["class_id"])]["ap50"])
            row["selected_source"] = selected
            row["ap50_gain_vs_ref"] = round(float(lookup[selected][int(row["class_id"])]["ap50"]) - ref_ap, 6)
        rows.extend(policy_rows)
        class_rows.extend(policy_class_rows)
        t27.write_rows(metrics_csv, rows, fieldnames)
        t27.write_rows(class_csv, class_rows, class_fieldnames)
        total = next(row for row in policy_rows if row["split"] == "scene_daynight_total")
        print(f"gate {label} mAP50={total['map50']:.6f} policy={summarize_policy(policy)}")

    ref_gate = next(row for row in rows if row["candidate"] == "ref" and row["split"] == "scene_daynight_total")
    policy_gate_rows = [
        row
        for row in rows
        if row["candidate"].startswith("class_channel_") and row["split"] == "scene_daynight_total"
    ]
    best_gate = max(policy_gate_rows, key=lambda row: (float(row["map50"]), float(row["map50_95"])))
    best_gain = float(best_gate["map50"]) - float(ref_gate["map50"])
    best_policy = policies[best_gate["candidate"]][0]
    best = {
        "candidate": "previous_full_best",
        "phase": "previous_full_best_27u",
        "map50": round(float(args.previous_best_map50), 6),
        "map50_95": round(float(args.previous_best_map50_95), 6),
    }
    status = "aborted_gate_no_gain"
    full_evaluated = False

    if best_gain >= args.min_gate_gain:
        full_rows, full_class_rows = evaluate_class_policy(
            models=models,
            dataset=dataset,
            indices=list(range(len(dataset))),
            imgsz=imgsz,
            device=device,
            args=args,
            phase=f"full_{best_gate['candidate']}",
            candidate_label=best_gate["candidate"],
            policy=best_policy,
        )
        rows.extend(full_rows)
        class_rows.extend(full_class_rows)
        t27.write_rows(metrics_csv, rows, fieldnames)
        t27.write_rows(class_csv, class_rows, class_fieldnames)
        best = next(row for row in full_rows if row["split"] == "scene_daynight_total")
        status = "target_reached" if float(best["map50"]) >= args.target_map50 else "completed"
        full_evaluated = True

    t27.release_models(models)

    rationale_suffix = (
        f" Gate ref={float(ref_gate['map50']):.6f}; best class policy={best_gate['candidate']} "
        f"gate mAP50={float(best_gate['map50']):.6f} gain={best_gain:+.6f}."
    )
    if not full_evaluated:
        rationale_suffix += " No full evaluation was run because the gate gain was below threshold."
    if not args.no_summary:
        append_research_summary(workspace=workspace, best=best, status=status, args=args, rationale_suffix=rationale_suffix)

    target_reached = full_evaluated and float(best["map50"]) >= args.target_map50
    message = "\n".join(
        [
            f"27w finished. Status={status}",
            f"- gate reference mAP50={float(ref_gate['map50']):.6f}",
            f"- best gate={best_gate['candidate']} mAP50={float(best_gate['map50']):.6f}; gate gain={best_gain:+.6f}",
            f"- policy={summarize_policy(best_policy)}",
            f"- reported best={best['candidate']} phase={best['phase']} mAP50={best['map50']} / mAP50:95={best['map50_95']}",
            f"- metrics CSV: {metrics_csv}",
            f"- class AP CSV: {class_csv}",
            "Decision: target reached." if target_reached else "Decision: target 0.600 not reached; continue with a different strategy.",
        ]
    )
    print(message)
    if not args.no_discord:
        t27.notify(message, "DQA-MoX 27w result")
    return 0 if target_reached else 2


if __name__ == "__main__":
    raise SystemExit(main())
