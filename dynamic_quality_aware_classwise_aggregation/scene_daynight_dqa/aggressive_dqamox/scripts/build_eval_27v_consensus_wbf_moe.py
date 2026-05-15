#!/usr/bin/env python3
"""Evaluate 27v consensus/WBF output MoE for scene-daynight DQA."""

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


SCRIPT_PATH = Path(__file__).resolve()
AGG_ROOT = SCRIPT_PATH.parents[1]
SCENE_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = SCRIPT_PATH.parents[4]
ET_ROOT = h27.ET_ROOT
REPORTS_ROOT = AGG_ROOT / "reports"
DEFAULT_WORKSPACE = AGG_ROOT / "output" / "27v_consensus_wbf_moe"
NOTEBOOK_PATH = AGG_ROOT / "notebooks" / "research_loop_until_060" / "016_27v_consensus_wbf_moe.ipynb"
COCO_EFFICIENT_YOLOV5L = SCENE_ROOT / "output" / "08_full_latent_dqamox_from_warmup" / "weights" / "efficient-yolov5l.pt"

if str(ET_ROOT) not in sys.path:
    sys.path.insert(0, str(ET_ROOT))

from configs.defaults import get_cfg  # noqa: E402
from utils.datasets import LoadImagesAndLabels  # noqa: E402
from utils.detect_multi_backend import DetectMultiBackend  # noqa: E402
from utils.general import check_img_size  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402


STRICT_COCO_TO_BDD = {
    0: [(0, 1.00)],  # person -> person
    1: [(5, 1.00)],  # bicycle -> bike
    2: [(2, 1.00)],  # car -> car
    3: [(6, 1.00)],  # motorcycle -> motor
    5: [(3, 1.00)],  # bus -> bus
    6: [(9, 1.00)],  # train -> train
    7: [(4, 1.00)],  # truck -> truck
    9: [(7, 1.00)],  # traffic light -> traffic light
    11: [(8, 1.00)],  # stop sign -> traffic sign
}
RIDER_DUP_COCO_TO_BDD = {
    **STRICT_COCO_TO_BDD,
    0: [(0, 1.00), (1, 0.22)],  # rider recall probe: low-score duplicate from COCO person
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--target-map50", type=float, default=0.60)
    parser.add_argument("--previous-best-map50", type=float, default=0.52939)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--merge-iou", type=float, default=0.50)
    parser.add_argument("--device", default="")
    parser.add_argument("--gate-images", type=int, default=1200)
    parser.add_argument("--min-gate-gain", type=float, default=0.010)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--no-discord", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    return parser.parse_args(argv)


def remap_coco_detections(det: torch.Tensor, mapping: dict[int, list[tuple[int, float]]]) -> torch.Tensor:
    if det.numel() == 0:
        return det.reshape(0, 6)
    parts = []
    classes = det[:, 5].long()
    for coco_cls, targets in mapping.items():
        mask = classes == coco_cls
        if not mask.any():
            continue
        src = det[mask]
        for bdd_cls, scale in targets:
            mapped = src.clone()
            mapped[:, 4] *= scale
            mapped[:, 5] = float(bdd_cls)
            parts.append(mapped)
    if not parts:
        return torch.zeros((0, 6), device=det.device)
    return torch.cat(parts, dim=0)


def model_mapping(candidate: dict, label: str) -> dict[int, list[tuple[int, float]]] | None:
    if label not in candidate.get("coco_remap_models", {}):
        return None
    mode = candidate["coco_remap_models"][label]
    if mode == "strict":
        return STRICT_COCO_TO_BDD
    if mode == "rider_dup":
        return RIDER_DUP_COCO_TO_BDD
    raise ValueError(f"Unknown COCO remap mode: {mode}")


def class_filter_detections(det: torch.Tensor, allowed_classes: list[int] | None) -> torch.Tensor:
    if allowed_classes is None or det.numel() == 0:
        return det
    allowed = torch.tensor(allowed_classes, device=det.device, dtype=torch.long)
    keep = (det[:, 5].long().unsqueeze(1) == allowed.unsqueeze(0)).any(1)
    return det[keep]


def pairwise_iou_xyxy(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), device=box.device)
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    box_area = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    boxes_area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return inter / (box_area + boxes_area - inter + 1e-7)


def filter_by_anchor_overlap(det: torch.Tensor, anchor: torch.Tensor, min_iou: float) -> torch.Tensor:
    if det.numel() == 0 or anchor.numel() == 0:
        return torch.zeros((0, 6), device=det.device)
    keep = []
    for row in det:
        same_cls = anchor[anchor[:, 5] == row[5]]
        if same_cls.numel() == 0:
            keep.append(False)
            continue
        keep.append(bool((pairwise_iou_xyxy(row[:4], same_cls[:, :4]) >= min_iou).any()))
    if not keep:
        return torch.zeros((0, 6), device=det.device)
    return det[torch.tensor(keep, device=det.device, dtype=torch.bool)]


def weighted_box_fusion(
    model_dets: list[tuple[str, torch.Tensor]],
    candidate: dict,
    max_det: int,
) -> torch.Tensor:
    rows = []
    iou_thr = float(candidate.get("wbf_iou", candidate["merge_iou"]))
    bonus = float(candidate.get("wbf_agreement_bonus", 0.0))
    weights = candidate.get("wbf_model_weights", {})
    topk_per_class = int(candidate.get("wbf_topk_per_class", 60))
    for cls in range(10):
        boxes = []
        for model_label, det in model_dets:
            if det.numel() == 0:
                continue
            cls_det = det[det[:, 5].long() == cls]
            if topk_per_class > 0 and cls_det.shape[0] > topk_per_class:
                cls_det = cls_det[cls_det[:, 4].argsort(descending=True)[:topk_per_class]]
            for row in cls_det:
                boxes.append(
                    {
                        "box": row[:4].clone(),
                        "score": float(row[4]),
                        "model": model_label,
                        "weight": float(weights.get(model_label, 1.0)),
                    }
                )
        boxes.sort(key=lambda item: item["score"], reverse=True)
        clusters: list[dict] = []
        for item in boxes:
            best_idx = -1
            best_iou = 0.0
            for idx, cluster in enumerate(clusters):
                iou = float(pairwise_iou_xyxy(item["box"], cluster["fused"].unsqueeze(0))[0])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= iou_thr:
                cluster = clusters[best_idx]
                cluster["items"].append(item)
            else:
                cluster = {"items": [item], "fused": item["box"].clone()}
                clusters.append(cluster)
            scores = torch.tensor([x["score"] * x["weight"] for x in cluster["items"]], device=item["box"].device)
            coords = torch.stack([x["box"] for x in cluster["items"]])
            cluster["fused"] = (coords * scores[:, None]).sum(0) / scores.sum().clamp(min=1e-7)
        for cluster in clusters:
            items = cluster["items"]
            unique_models = {x["model"] for x in items}
            max_score = max(x["score"] for x in items)
            avg_score = sum(x["score"] * x["weight"] for x in items) / max(sum(x["weight"] for x in items), 1e-7)
            if candidate.get("wbf_score_mode", "max_bonus") == "avg":
                score = avg_score
            else:
                score = max_score + bonus * max(len(unique_models) - 1, 0)
            rows.append(torch.cat([cluster["fused"], torch.tensor([min(score, 0.999), float(cls)], device=cluster["fused"].device)]))
    if not rows:
        return torch.zeros((0, 6), dtype=torch.float32)
    fused = torch.stack(rows)
    fused = fused[fused[:, 4].argsort(descending=True)]
    if max_det > 0:
        fused = fused[:max_det]
    return fused.float().cpu()


def soft_nms_detections(det: torch.Tensor, candidate: dict, max_det: int) -> torch.Tensor:
    if det.numel() == 0:
        return det.reshape(0, 6)
    sigma = float(candidate.get("soft_nms_sigma", 0.5))
    score_thr = float(candidate.get("soft_nms_score_thr", 0.0001))
    out = []
    for cls in det[:, 5].unique():
        cls_det = det[det[:, 5] == cls].clone()
        while cls_det.numel() and len(out) < max_det:
            order = cls_det[:, 4].argsort(descending=True)
            cls_det = cls_det[order]
            best = cls_det[0].clone()
            out.append(best)
            rest = cls_det[1:]
            if rest.numel() == 0:
                break
            ious = pairwise_iou_xyxy(best[:4], rest[:, :4])
            rest[:, 4] *= torch.exp(-(ious * ious) / sigma)
            cls_det = rest[rest[:, 4] >= score_thr]
    if not out:
        return torch.zeros((0, 6), dtype=torch.float32)
    merged = torch.stack(out)
    merged = merged[merged[:, 4].argsort(descending=True)]
    if max_det > 0:
        merged = merged[:max_det]
    return merged.float().cpu()


def predict_candidate(
    *,
    models: dict[str, DetectMultiBackend],
    image: np.ndarray,
    split: str,
    candidate: dict,
    imgsz: int,
    device: torch.device,
    args: argparse.Namespace,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    model_parts: list[tuple[str, torch.Tensor]] = []
    for label in t27.route_for_candidate(candidate, split):
        with torch.no_grad():
            det = t27.infer_bgr_batch(
                model=models[label],
                images=[image],
                imgsz=imgsz,
                device=device,
                conf_thres=args.conf_thres,
                iou_thres=candidate["pre_iou"],
                max_det=args.max_det,
            )[0]
        det = det.detach()
        mapping = model_mapping(candidate, label)
        if mapping is not None:
            det = remap_coco_detections(det, mapping)
        det = class_filter_detections(det, candidate.get("class_filters", {}).get(label))
        if det.numel() == 0:
            continue
        det = det.clone()
        det[:, 4] *= t27.scale_for_model(candidate, label)
        model_parts.append((label, det))
        parts.append(det)
    if not parts:
        return torch.zeros((0, 6), dtype=torch.float32)
    for label, policy in candidate.get("overlap_filters", {}).items():
        anchor_label = policy["anchor"]
        anchor = next((det for model_label, det in model_parts if model_label == anchor_label), torch.zeros((0, 6)))
        filtered = []
        for model_label, det in model_parts:
            if model_label != label:
                filtered.append((model_label, det))
            else:
                filtered.append((model_label, filter_by_anchor_overlap(det, anchor.to(det.device), float(policy["iou"]))))
        model_parts = filtered
    if candidate.get("fuser") == "wbf":
        return weighted_box_fusion(model_parts, candidate, args.max_det)
    merged = torch.cat(parts, dim=0)
    if candidate.get("fuser") == "soft_nms":
        return soft_nms_detections(merged, candidate, args.max_det)
    merged = t27.nms_detections(merged, candidate["merge_iou"], args.max_det)
    return merged.float().cpu()


def evaluate_candidate(
    *,
    models: dict[str, DetectMultiBackend],
    dataset: LoadImagesAndLabels,
    candidate: dict,
    indices: list[int],
    imgsz: int,
    device: torch.device,
    args: argparse.Namespace,
    phase: str,
) -> list[dict]:
    iouv = torch.linspace(0.5, 0.95, 10)
    niou = iouv.numel()
    total_stats = []
    split_stats = {split: [] for split in t27.DOMAIN_SPLITS}
    split_counts = {split: {"images": 0, "labels": 0} for split in t27.DOMAIN_SPLITS}
    total_images = 0
    total_labels = 0

    for idx in tqdm(indices, desc=f"{phase}_{candidate['label']}"):
        path = dataset.img_files[idx]
        split = t27.split_from_path(path)
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        h0, w0 = image.shape[:2]
        labelsn = t27.native_labels(dataset.labels[idx], (w0, h0))
        nl = len(labelsn)
        pred = predict_candidate(
            models=models,
            image=image,
            split=split,
            candidate=candidate,
            imgsz=imgsz,
            device=device,
            args=args,
        )
        t27.add_stats(total_stats, pred, labelsn, iouv)
        t27.add_stats(split_stats[split], pred, labelsn, iouv)
        total_images += 1
        total_labels += nl
        split_counts[split]["images"] += 1
        split_counts[split]["labels"] += nl

    rows: list[dict] = []
    base = {
        "candidate": candidate["label"],
        "phase": phase,
        "imgsz": imgsz,
        "pre_iou": candidate["pre_iou"],
        "merge_iou": candidate["merge_iou"],
        "route_summary": candidate["route_summary"],
        "idea": candidate["idea"],
    }
    rows.append(
        {
            **base,
            "split": "scene_daynight_total",
            "images": total_images,
            "labels": int(total_labels),
            **t27.summarize_stats(total_stats, niou),
        }
    )
    for split in t27.DOMAIN_SPLITS:
        rows.append(
            {
                **base,
                "split": split,
                "images": split_counts[split]["images"],
                "labels": int(split_counts[split]["labels"]),
                **t27.summarize_stats(split_stats[split], niou),
            }
        )
    return rows


def make_model_specs(workspace: Path) -> tuple[dict[str, Path], list[str], Path]:
    specs, missing, routed_ref = t27.make_model_specs(workspace)
    specs = {
        "routed_day_light_night_hard": specs["routed_day_light_night_hard"],
        "coco80_efficient_yolov5l": COCO_EFFICIENT_YOLOV5L,
    }
    missing = sorted(label for label, path in specs.items() if not path.exists())
    return specs, missing, routed_ref


def make_candidates(args: argparse.Namespace) -> list[dict]:
    base = {"pre_iou": 0.50, "merge_iou": args.merge_iou}
    return [
        {
            **base,
            "label": "full1024_reference",
            "routes": t27.all_routes(["routed_day_light_night_hard"]),
            "route_summary": "all images -> current best brightness-routed DQA-MoE",
            "idea": "Reference for comparing consensus fusion policies.",
        },
        {
            **base,
            "label": "reference_softnms_sigma050",
            "routes": t27.all_routes(["routed_day_light_night_hard"]),
            "fuser": "soft_nms",
            "soft_nms_sigma": 0.50,
            "route_summary": "all images -> reference routed DQA-MoE with Gaussian Soft-NMS",
            "idea": "Soft-NMS keeps close objects alive while decaying duplicate scores instead of hard suppression.",
        },
        {
            **base,
            "label": "wbf_coco_tail_s020_b004",
            "routes": t27.all_routes(["routed_day_light_night_hard", "coco80_efficient_yolov5l"]),
            "coco_remap_models": {"coco80_efficient_yolov5l": "strict"},
            "score_scales": {"routed_day_light_night_hard": 1.00, "coco80_efficient_yolov5l": 0.20},
            "fuser": "wbf",
            "wbf_iou": 0.55,
            "wbf_agreement_bonus": 0.04,
            "wbf_topk_per_class": 60,
            "route_summary": "reference + strict COCO x0.20 -> WBF, small agreement bonus",
            "idea": "WBF should improve localization/ranking when the broad COCO expert agrees, while tail boxes stay low.",
        },
        {
            **base,
            "label": "wbf_coco_consensus_s035_b008",
            "routes": t27.all_routes(["routed_day_light_night_hard", "coco80_efficient_yolov5l"]),
            "coco_remap_models": {"coco80_efficient_yolov5l": "strict"},
            "score_scales": {"routed_day_light_night_hard": 1.00, "coco80_efficient_yolov5l": 0.35},
            "fuser": "wbf",
            "wbf_iou": 0.55,
            "wbf_agreement_bonus": 0.08,
            "wbf_topk_per_class": 60,
            "route_summary": "reference + strict COCO x0.35 -> WBF, consensus score boost",
            "idea": "Consensus MoE: agreement between in-domain DQA and out-of-domain COCO should outrank single-expert boxes.",
        },
        {
            **base,
            "label": "wbf_coco_rare_rider_s045_b010",
            "routes": t27.all_routes(["routed_day_light_night_hard", "coco80_efficient_yolov5l"]),
            "coco_remap_models": {"coco80_efficient_yolov5l": "rider_dup"},
            "score_scales": {"routed_day_light_night_hard": 1.00, "coco80_efficient_yolov5l": 0.45},
            "class_filters": {"coco80_efficient_yolov5l": [0, 1, 5, 6, 7, 8, 9]},
            "fuser": "wbf",
            "wbf_iou": 0.50,
            "wbf_agreement_bonus": 0.10,
            "wbf_topk_per_class": 60,
            "route_summary": "reference + rare-class COCO/rider duplicate x0.45 -> WBF",
            "idea": "Class-channel MoE: let COCO help underfit rare classes while keeping car/truck/bus dominated by DQA.",
        },
        {
            **base,
            "label": "wbf_coco_agreeonly_s060_b012",
            "routes": t27.all_routes(["routed_day_light_night_hard", "coco80_efficient_yolov5l"]),
            "coco_remap_models": {"coco80_efficient_yolov5l": "strict"},
            "score_scales": {"routed_day_light_night_hard": 1.00, "coco80_efficient_yolov5l": 0.60},
            "overlap_filters": {"coco80_efficient_yolov5l": {"anchor": "routed_day_light_night_hard", "iou": 0.20}},
            "fuser": "wbf",
            "wbf_iou": 0.55,
            "wbf_agreement_bonus": 0.12,
            "wbf_topk_per_class": 60,
            "route_summary": "reference + only COCO boxes overlapping reference -> WBF, high consensus boost",
            "idea": "Conservative consensus: use COCO as a verifier/localizer, not as a free-box generator.",
        },
    ]


def load_models(
    *,
    specs: dict[str, Path],
    labels: list[str],
    device: torch.device,
    data: Path,
    imgsz: int,
) -> tuple[dict[str, DetectMultiBackend], int]:
    models: dict[str, DetectMultiBackend] = {}
    max_stride = 32
    for label in labels:
        print(f"loading expert {label}: {specs[label]}")
        model = DetectMultiBackend(str(specs[label]), device=device, data=str(data), fp16=True)
        model.eval()
        for module in model.model.modules():
            if module.__class__.__name__ == "Detect":
                if not hasattr(module, "class_skew_enabled"):
                    module.class_skew_enabled = False
                if not hasattr(module, "use_residual"):
                    module.use_residual = False
        stride = max(int(model.stride), 32)
        max_stride = max(max_stride, stride)
        models[label] = model
    imgsz = check_img_size(imgsz, s=max_stride)
    for model in models.values():
        with torch.no_grad():
            model.warmup(imgsz=(1, 3, imgsz, imgsz))
    return models, imgsz


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
                "log": str(workspace / "stats" / "27v_consensus_wbf_moe_metrics.csv"),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "rationale": (
                    "27u showed external COCO boxes can create large gate gains but do not generalize to full mAP. "
                    "27v therefore switches from additive NMS to WBF/Soft-NMS consensus MoE, boosting boxes only "
                    "when experts agree or when COCO targets rare classes."
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

    specs, missing, routed_ref = make_model_specs(workspace)
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
        prefix="27v: ",
    )
    gate_indices = t27.even_indices(len(dataset), args.gate_images)
    candidates = make_candidates(args)
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
        "pre_iou",
        "merge_iou",
        "route_summary",
        "idea",
    ]
    metrics_csv = stats_dir / "27v_consensus_wbf_moe_metrics.csv"
    manifest_path = stats_dir / "27v_consensus_wbf_moe_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "workspace": str(workspace),
                "dataset_size": len(dataset),
                "gate_images": len(gate_indices),
                "routed_reference": str(routed_ref),
                "coco_expert": str(COCO_EFFICIENT_YOLOV5L),
                "model_specs": {label: str(path) for label, path in specs.items()},
                "candidate_labels": [candidate["label"] for candidate in candidates],
                "papers": [
                    "Weighted Boxes Fusion (Solovyev et al., 2019) motivates averaging boxes from different detectors instead of hard suppression.",
                    "Soft-NMS (Bodla et al., 2017) motivates score decay rather than deleting overlapping candidates.",
                    "27u's COCO bridge result motivates consensus filtering to reduce full-set false-positive drift.",
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
            "27v started: consensus WBF/Soft-NMS output MoE. "
            f"Gate={len(gate_indices)} images; full run only if gate gain >= {args.min_gate_gain:.3f}.",
            "DQA-MoX 27v started",
        )

    rows: list[dict] = []
    gate_total_rows: list[dict] = []
    baseline_map50 = None
    for candidate in candidates:
        labels = t27.required_labels(candidate)
        models, imgsz = load_models(specs=specs, labels=labels, device=device, data=split_cfg, imgsz=args.imgsz)
        try:
            candidate_rows = evaluate_candidate(
                models=models,
                dataset=dataset,
                candidate=candidate,
                indices=gate_indices,
                imgsz=imgsz,
                device=device,
                args=args,
                phase=f"gate_{candidate['label']}",
            )
        finally:
            t27.release_models(models)
        rows.extend(candidate_rows)
        t27.write_rows(metrics_csv, rows, fieldnames)
        total_row = next(row for row in candidate_rows if row["split"] == "scene_daynight_total")
        gate_total_rows.append(total_row)
        if candidate["label"] == "full1024_reference":
            baseline_map50 = float(total_row["map50"])
            print(f"gate reference mAP50={baseline_map50:.6f} mAP50:95={total_row['map50_95']}")
        else:
            assert baseline_map50 is not None
            gain = float(total_row["map50"]) - baseline_map50
            print(f"gate {candidate['label']} mAP50={total_row['map50']:.6f} gain={gain:+.6f}")

    assert baseline_map50 is not None
    non_reference = [row for row in gate_total_rows if row["candidate"] != "full1024_reference"]
    best_gate = max(non_reference, key=lambda row: (float(row["map50"]), float(row["map50_95"])))
    best_gain = float(best_gate["map50"]) - baseline_map50
    best = best_gate
    status = "aborted_gate_no_gain"
    full_evaluated = False
    if best_gain >= args.min_gate_gain:
        best_candidate = next(candidate for candidate in candidates if candidate["label"] == best_gate["candidate"])
        labels = t27.required_labels(best_candidate)
        models, imgsz = load_models(specs=specs, labels=labels, device=device, data=split_cfg, imgsz=args.imgsz)
        try:
            full_rows = evaluate_candidate(
                models=models,
                dataset=dataset,
                candidate=best_candidate,
                indices=list(range(len(dataset))),
                imgsz=imgsz,
                device=device,
                args=args,
                phase=f"full_{best_candidate['label']}",
            )
        finally:
            t27.release_models(models)
        rows.extend(full_rows)
        t27.write_rows(metrics_csv, rows, fieldnames)
        best = next(row for row in full_rows if row["split"] == "scene_daynight_total")
        status = "target_reached" if float(best["map50"]) >= args.target_map50 else "completed"
        full_evaluated = True

    reported_best = best
    rationale_suffix = ""
    if not full_evaluated:
        reported_best = {
            "candidate": "previous_full_best",
            "phase": "previous_full_best_27u",
            "map50": round(float(args.previous_best_map50), 6),
            "map50_95": "",
        }
        rationale_suffix = (
            f" Gate-only best was {best_gate['candidate']} mAP50={best_gate['map50']:.6f} "
            f"(gain={best_gain:+.6f}), below the continuation threshold, so no full evaluation was run."
        )
    if not args.no_summary:
        append_research_summary(
            workspace=workspace,
            best=reported_best,
            status=status,
            args=args,
            rationale_suffix=rationale_suffix,
        )
    target_reached = full_evaluated and float(best["map50"]) >= args.target_map50
    message = "\n".join(
        [
            f"27v finished. Status={status}",
            f"- gate reference mAP50={baseline_map50:.6f}",
            f"- best gate={best_gate['candidate']} mAP50={best_gate['map50']:.6f}; gate gain={best_gain:+.6f}",
            f"- reported best={reported_best['candidate']} phase={reported_best['phase']} mAP50={reported_best['map50']} / mAP50:95={reported_best['map50_95']}",
            f"- metrics CSV: {metrics_csv}",
            "Decision: target reached." if target_reached else "Decision: target 0.600 not reached; continue with a different strategy.",
        ]
    )
    print(message)
    if not args.no_discord:
        t27.notify(message, "DQA-MoX 27v result")
    return 0 if target_reached else 2


if __name__ == "__main__":
    raise SystemExit(main())
