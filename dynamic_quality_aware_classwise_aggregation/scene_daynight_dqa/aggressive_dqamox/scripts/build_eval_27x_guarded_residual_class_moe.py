#!/usr/bin/env python3
"""Evaluate 27x guarded residual class MoE distilled from 27w."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

import build_eval_27h_model_level_moe as h27
import build_eval_27t_path_domain_routed_moe as t27
import build_eval_27u_coco_bridge_moe as u27
import build_eval_27w_class_channel_moe as w27


SCRIPT_PATH = Path(__file__).resolve()
AGG_ROOT = SCRIPT_PATH.parents[1]
REPORTS_ROOT = AGG_ROOT / "reports"
DEFAULT_WORKSPACE = AGG_ROOT / "output" / "27x_guarded_residual_class_moe"
NOTEBOOK_PATH = AGG_ROOT / "notebooks" / "research_loop_until_060" / "018_27x_guarded_residual_class_moe.ipynb"

if str(h27.ET_ROOT) not in sys.path:
    sys.path.insert(0, str(h27.ET_ROOT))

from configs.defaults import get_cfg  # noqa: E402
from utils.datasets import LoadImagesAndLabels  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402


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
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--no-discord", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    return parser.parse_args(argv)


def candidate_specs() -> list[dict]:
    return [
        {
            "label": "ref1024_reference",
            "class_sources": {},
            "idea": "Full-set reference inside the residual class-MoE evaluator.",
        },
        {
            "label": "bike_coco_s020",
            "class_sources": {5: "coco_s020"},
            "idea": "Only bike uses COCO strict remap at low score; all other classes stay routed DQA.",
        },
        {
            "label": "bike_coco_s035",
            "class_sources": {5: "coco_s035"},
            "idea": "Bike-only COCO residual at medium score, distilled from 27w's strongest class AP gain.",
        },
        {
            "label": "bike_coco_s050",
            "class_sources": {5: "coco_s050"},
            "idea": "Bike-only COCO residual at the 27w gate-selected score.",
        },
        {
            "label": "bike_union_s035",
            "class_sources": {5: "union_s035"},
            "idea": "Bike keeps both DQA and COCO boxes, letting NMS choose instead of replacing the class.",
        },
        {
            "label": "bike_union_s050",
            "class_sources": {5: "union_s050"},
            "idea": "Aggressive bike union while protecting all non-bike classes.",
        },
        {
            "label": "bike_coco_s035_bus_union_s035",
            "class_sources": {3: "union_s035", 5: "coco_s035"},
            "idea": "Guarded 27w policy: bus gets a COCO union residual, bike gets COCO-only medium score.",
        },
        {
            "label": "bike_coco_s050_bus_union_s050",
            "class_sources": {3: "union_s050", 5: "coco_s050"},
            "idea": "Gate-selected guarded 27w policy with only bus and bike allowed to deviate from reference.",
        },
        {
            "label": "bike_union_s035_bus_union_s035",
            "class_sources": {3: "union_s035", 5: "union_s035"},
            "idea": "No class replacement: bus and bike both use DQA+COCO union at medium score.",
        },
    ]


def build_sources(parts: dict[str, torch.Tensor], args: argparse.Namespace) -> dict[str, torch.Tensor]:
    ref = parts["ref"]
    coco = parts["coco_strict"]
    return {
        "ref": w27.nms_or_empty([ref], args.merge_iou, args.max_det),
        "coco_s020": w27.nms_or_empty([w27.scale_det(coco, 0.20)], args.merge_iou, args.max_det),
        "coco_s035": w27.nms_or_empty([w27.scale_det(coco, 0.35)], args.merge_iou, args.max_det),
        "coco_s050": w27.nms_or_empty([w27.scale_det(coco, 0.50)], args.merge_iou, args.max_det),
        "union_s020": w27.nms_or_empty([ref, w27.scale_det(coco, 0.20)], args.merge_iou, args.max_det),
        "union_s035": w27.nms_or_empty([ref, w27.scale_det(coco, 0.35)], args.merge_iou, args.max_det),
        "union_s050": w27.nms_or_empty([ref, w27.scale_det(coco, 0.50)], args.merge_iou, args.max_det),
    }


def predict_spec(sources: dict[str, torch.Tensor], spec: dict, args: argparse.Namespace) -> torch.Tensor:
    parts = []
    class_sources = spec["class_sources"]
    for cls in range(len(w27.CLASS_NAMES)):
        source = class_sources.get(cls, "ref")
        det = sources[source]
        if det.numel() == 0:
            continue
        cls_det = det[det[:, 5].long() == cls]
        if cls_det.numel():
            parts.append(cls_det)
    return w27.nms_or_empty(parts, args.merge_iou, args.max_det)


def evaluate_specs(
    *,
    models: dict,
    dataset: LoadImagesAndLabels,
    indices: list[int],
    imgsz: int,
    device: torch.device,
    args: argparse.Namespace,
    specs: list[dict],
) -> list[dict]:
    iouv = torch.linspace(0.5, 0.95, 10)
    niou = iouv.numel()
    total_stats = {spec["label"]: [] for spec in specs}
    split_stats = {spec["label"]: {split: [] for split in t27.DOMAIN_SPLITS} for spec in specs}
    split_counts = {split: {"images": 0, "labels": 0} for split in t27.DOMAIN_SPLITS}
    total_labels = 0

    for idx in tqdm(indices, desc="27x_full_guarded_residual"):
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
        sources = build_sources(
            w27.infer_base_parts(models=models, image=image, imgsz=imgsz, device=device, args=args),
            args,
        )
        for spec in specs:
            pred = predict_spec(sources, spec, args)
            t27.add_stats(total_stats[spec["label"]], pred, labelsn, iouv)
            t27.add_stats(split_stats[spec["label"]][split], pred, labelsn, iouv)

    rows: list[dict] = []
    for spec in specs:
        label = spec["label"]
        summary, _ = w27.summarize_stats_with_classes(total_stats[label], niou)
        policy = "; ".join(
            f"{w27.CLASS_NAMES[cls]}->{source}" for cls, source in sorted(spec["class_sources"].items())
        ) or "all->ref"
        rows.append(
            {
                "candidate": label,
                "phase": "full_guarded_residual",
                "split": "scene_daynight_total",
                "images": len(indices),
                "labels": int(total_labels),
                **summary,
                "imgsz": imgsz,
                "merge_iou": args.merge_iou,
                "class_policy": policy,
                "route_summary": policy,
                "idea": spec["idea"],
            }
        )
        for split in t27.DOMAIN_SPLITS:
            split_summary, _ = w27.summarize_stats_with_classes(split_stats[label][split], niou)
            rows.append(
                {
                    "candidate": label,
                    "phase": "full_guarded_residual",
                    "split": split,
                    "images": split_counts[split]["images"],
                    "labels": int(split_counts[split]["labels"]),
                    **split_summary,
                    "imgsz": imgsz,
                    "merge_iou": args.merge_iou,
                    "class_policy": policy,
                    "route_summary": policy,
                    "idea": spec["idea"],
                }
            )
    return rows


def append_research_summary(*, workspace: Path, best: dict, status: str, args: argparse.Namespace) -> None:
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
                "log": str(workspace / "stats" / "27x_guarded_residual_class_moe_metrics.csv"),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "rationale": (
                    "27w class-channel raw overfit the gate and regressed full mAP. 27x distills only the robust "
                    "bike and bus residual signals into guarded full-set class policies, comparing score scales in one pass."
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

    model_specs, missing, routed_ref = u27.make_model_specs(workspace)
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
        prefix="27x: ",
    )
    indices = list(range(len(dataset)))
    if args.limit_images and args.limit_images > 0:
        indices = t27.even_indices(len(dataset), args.limit_images)

    labels = ["routed_day_light_night_hard", "coco80_efficient_yolov5l"]
    models, imgsz = u27.load_models(specs=model_specs, labels=labels, device=device, data=split_cfg, imgsz=args.imgsz)
    specs = candidate_specs()
    metrics_csv = stats_dir / "27x_guarded_residual_class_moe_metrics.csv"
    manifest_path = stats_dir / "27x_guarded_residual_class_moe_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "workspace": str(workspace),
                "dataset_size": len(dataset),
                "evaluated_images": len(indices),
                "routed_reference": str(routed_ref),
                "coco_expert": str(u27.COCO_EFFICIENT_YOLOV5L),
                "candidate_labels": [spec["label"] for spec in specs],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    if not args.no_discord:
        t27.notify(
            f"27x started: guarded residual class MoE on {len(indices)} images. "
            "Only bike/bus may deviate from the routed DQA reference.",
            "DQA-MoX 27x started",
        )

    rows = evaluate_specs(models=models, dataset=dataset, indices=indices, imgsz=imgsz, device=device, args=args, specs=specs)
    t27.release_models(models)
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
    t27.write_rows(metrics_csv, rows, fieldnames)

    total_rows = [row for row in rows if row["split"] == "scene_daynight_total"]
    best = max(total_rows, key=lambda row: (float(row["map50"]), float(row["map50_95"])))
    status = "target_reached" if float(best["map50"]) >= args.target_map50 else "completed"
    if not args.no_summary:
        append_research_summary(workspace=workspace, best=best, status=status, args=args)

    ref = next(row for row in total_rows if row["candidate"] == "ref1024_reference")
    message = "\n".join(
        [
            f"27x finished. Status={status}",
            f"- full reference mAP50={float(ref['map50']):.6f} / mAP50:95={float(ref['map50_95']):.6f}",
            f"- best={best['candidate']} mAP50={float(best['map50']):.6f} / mAP50:95={float(best['map50_95']):.6f}",
            f"- delta vs 27u best={float(best['map50']) - float(args.previous_best_map50):+.6f}",
            f"- metrics CSV: {metrics_csv}",
            "Decision: target reached." if float(best["map50"]) >= args.target_map50 else "Decision: target 0.600 not reached; continue with a different strategy.",
        ]
    )
    print(message)
    if not args.no_discord:
        t27.notify(message, "DQA-MoX 27x result")
    return 0 if float(best["map50"]) >= args.target_map50 else 2


if __name__ == "__main__":
    raise SystemExit(main())
