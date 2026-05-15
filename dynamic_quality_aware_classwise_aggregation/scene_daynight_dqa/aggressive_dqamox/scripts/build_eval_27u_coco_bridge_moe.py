#!/usr/bin/env python3
"""Evaluate 27u COCO-pretrained bridge expert fusion for scene-daynight DQA."""

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
DEFAULT_WORKSPACE = AGG_ROOT / "output" / "27u_coco_bridge_moe"
NOTEBOOK_PATH = AGG_ROOT / "notebooks" / "research_loop_until_060" / "015_27u_coco_bridge_moe.ipynb"
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
        if det.numel() == 0:
            continue
        det = det.clone()
        det[:, 4] *= t27.scale_for_model(candidate, label)
        parts.append(det)
    if not parts:
        return torch.zeros((0, 6), dtype=torch.float32)
    merged = torch.cat(parts, dim=0)
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
            "idea": "Reference for comparing external generalist bridge expert fusion.",
        },
        {
            **base,
            "label": "coco80_strict_remap_only",
            "routes": t27.all_routes(["coco80_efficient_yolov5l"]),
            "coco_remap_models": {"coco80_efficient_yolov5l": "strict"},
            "route_summary": "all images -> COCO80 Efficient-YOLOv5l remapped to BDD10",
            "idea": "Diagnostic for whether COCO pretraining still detects BDD road objects without DQA fine-tuning.",
        },
        {
            **base,
            "label": "reference_plus_coco_strict_s020",
            "routes": t27.all_routes(["routed_day_light_night_hard", "coco80_efficient_yolov5l"]),
            "coco_remap_models": {"coco80_efficient_yolov5l": "strict"},
            "score_scales": {"routed_day_light_night_hard": 1.00, "coco80_efficient_yolov5l": 0.20},
            "route_summary": "routed DQA-MoE + strict COCO remap, COCO score x0.20",
            "idea": "Low-risk external teacher boxes only enter the tail of the AP ranking.",
        },
        {
            **base,
            "label": "reference_plus_coco_strict_s035",
            "routes": t27.all_routes(["routed_day_light_night_hard", "coco80_efficient_yolov5l"]),
            "coco_remap_models": {"coco80_efficient_yolov5l": "strict"},
            "score_scales": {"routed_day_light_night_hard": 1.00, "coco80_efficient_yolov5l": 0.35},
            "route_summary": "routed DQA-MoE + strict COCO remap, COCO score x0.35",
            "idea": "Moderate COCO bridge: enough score to rescue high-confidence missed vehicles and traffic lights.",
        },
        {
            **base,
            "label": "reference_plus_coco_riderdup_s030",
            "routes": t27.all_routes(["routed_day_light_night_hard", "coco80_efficient_yolov5l"]),
            "coco_remap_models": {"coco80_efficient_yolov5l": "rider_dup"},
            "score_scales": {"routed_day_light_night_hard": 1.00, "coco80_efficient_yolov5l": 0.30},
            "route_summary": "routed DQA-MoE + COCO remap with low-score person->rider duplicate",
            "idea": "Aggressive rider-recall probe: COCO person boxes may cover riders that BDD pseudo labels miss.",
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
                "log": str(workspace / "stats" / "27u_coco_bridge_moe_metrics.csv"),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "rationale": (
                    "27t showed domain routing over existing DQA experts gives only +0.002 gate mAP. "
                    "27u treats the COCO-pretrained Efficient-YOLOv5l initialization as an external generalist "
                    "MoE expert, remaps its 80 classes to BDD10, and fuses low-score boxes to recover teacher misses."
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
        prefix="27u: ",
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
    metrics_csv = stats_dir / "27u_coco_bridge_moe_metrics.csv"
    manifest_path = stats_dir / "27u_coco_bridge_moe_manifest.json"
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
                    "MixPL and related SSOD work motivate using stronger complementary teachers when pseudo labels amplify detector blind spots.",
                    "YOLO-World/open-vocabulary detection work motivates reusing broad visual vocabulary as an inference-time bridge expert.",
                    "Model-level detection MoE work motivates preserving expert outputs and fusing predictions instead of averaging weights.",
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
            "27u started: COCO-pretrained bridge expert. "
            f"Gate={len(gate_indices)} images; full run only if gate gain >= {args.min_gate_gain:.3f}.",
            "DQA-MoX 27u started",
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

    if not args.no_summary:
        append_research_summary(workspace=workspace, best=best, status=status, args=args)
    message = "\n".join(
        [
            f"27u finished. Status={status}",
            f"- gate reference mAP50={baseline_map50:.6f}",
            f"- best gate={best_gate['candidate']} mAP50={best_gate['map50']:.6f}; gate gain={best_gain:+.6f}",
            f"- reported best={best['candidate']} phase={best['phase']} mAP50={best['map50']} / mAP50:95={best['map50_95']}",
            f"- metrics CSV: {metrics_csv}",
            "Decision: target reached." if float(best["map50"]) >= args.target_map50 else "Decision: target 0.600 not reached; continue with a different strategy.",
        ]
    )
    print(message)
    if not args.no_discord:
        t27.notify(message, "DQA-MoX 27u result")
    return 0 if float(best["map50"]) >= args.target_map50 else 2


if __name__ == "__main__":
    raise SystemExit(main())
