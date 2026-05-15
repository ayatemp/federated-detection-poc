#!/usr/bin/env python3
"""Evaluate split-routed high-resolution/SAHI policies for self-only DQA-MoE."""

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
import build_eval_27o_asymmetric_score_routed_moe as o27
import build_eval_27s_sahi_routed_moe as s27
import build_eval_27t_path_domain_routed_moe as t27


SCRIPT_PATH = Path(__file__).resolve()
AGG_ROOT = SCRIPT_PATH.parents[1]
SCENE_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = SCRIPT_PATH.parents[4]
REPORTS_ROOT = AGG_ROOT / "reports"
DEFAULT_WORKSPACE = AGG_ROOT / "output" / "30_split_policy_highres_sahi_moe"
NOTEBOOK_PATH = SCENE_ROOT / "notebooks" / "30_split_policy_highres_sahi_moe.ipynb"

if str(h27.ET_ROOT) not in sys.path:
    sys.path.insert(0, str(h27.ET_ROOT))

from configs.defaults import get_cfg  # noqa: E402
from utils.datasets import LoadImagesAndLabels  # noqa: E402
from utils.detect_multi_backend import DetectMultiBackend  # noqa: E402
from utils.metrics import ap_per_class  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402
from val import process_batch  # noqa: E402


DAY_SPLITS = {"highway_day", "citystreet_day", "residential_day"}
NIGHT_SPLITS = {"highway_night", "citystreet_night", "residential_night"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--target-map50", type=float, default=0.55)
    parser.add_argument("--previous-best-map50", type=float, default=0.52939)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gate-images", type=int, default=360)
    parser.add_argument("--min-gate-gain", type=float, default=0.008)
    parser.add_argument("--tile-batch-size", type=int, default=8)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--no-discord", action="store_true")
    return parser.parse_args(argv)


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def notify(message: str, title: str) -> None:
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from notebook_notify import notify_discord

        print(notify_discord(message, title=title, fail_silently=True))
    except Exception as exc:
        print(f"Discord notification skipped: {exc}")


def full_cfg(label: str, imgsz: int, iou: float, merge_iou: float = 0.50) -> dict:
    return {
        "label": label,
        "slice_size": 0,
        "tile_imgsz": imgsz,
        "full_imgsz": imgsz,
        "overlap": 0.0,
        "tile_iou": iou,
        "merge_iou": merge_iou,
        "include_full": True,
        "core": False,
        "idea": label,
    }


def sahi_cfg(label: str, slice_size: int, tile_imgsz: int, overlap: float, merge_iou: float = 0.50) -> dict:
    return {
        "label": label,
        "slice_size": slice_size,
        "tile_imgsz": tile_imgsz,
        "full_imgsz": 1024,
        "overlap": overlap,
        "tile_iou": 0.50,
        "merge_iou": merge_iou,
        "include_full": True,
        "core": True,
        "idea": label,
    }


def policy(default: dict, overrides: dict[str, dict] | None = None) -> dict[str, dict]:
    overrides = overrides or {}
    return {split: overrides.get(split, default) for split in t27.DOMAIN_SPLITS}


def group_policy(default: dict, *, day: dict | None = None, night: dict | None = None, overrides: dict[str, dict] | None = None) -> dict[str, dict]:
    out = {}
    overrides = overrides or {}
    for split in t27.DOMAIN_SPLITS:
        if split in overrides:
            out[split] = overrides[split]
        elif split in DAY_SPLITS and day is not None:
            out[split] = day
        elif split in NIGHT_SPLITS and night is not None:
            out[split] = night
        else:
            out[split] = default
    return out


def make_candidates() -> list[dict]:
    f1024_055 = full_cfg("full1024_iou055", 1024, 0.55)
    f1024_060 = full_cfg("full1024_iou060", 1024, 0.60)
    f1152_055 = full_cfg("full1152_iou055", 1152, 0.55)
    f896_055 = full_cfg("full896_iou055", 896, 0.55)
    sahi768 = sahi_cfg("sahi768_o25_full1024", 768, 768, 0.25)
    sahi896 = sahi_cfg("sahi896_o25_full1024", 896, 896, 0.25)
    sahi640 = sahi_cfg("sahi640_o30_full1024", 640, 640, 0.30, merge_iou=0.45)
    return [
        {
            "label": "all_full1024_iou055_ref",
            "split_configs": policy(f1024_055),
            "idea": "Reference: best known non-tiled 1024px tight-NMS routed MoE.",
        },
        {
            "label": "all_full1152_iou055",
            "split_configs": policy(f1152_055),
            "idea": "Check whether 1152px helps only under this custom evaluator/gate.",
        },
        {
            "label": "day1152_night1024",
            "split_configs": group_policy(f1024_055, day=f1152_055, night=f1024_055),
            "idea": "Day splits have higher precision; spend resolution on day while keeping night conservative.",
        },
        {
            "label": "cityres1152_highway896",
            "split_configs": policy(
                f1024_055,
                {
                    "highway_day": f896_055,
                    "highway_night": f896_055,
                    "citystreet_day": f1152_055,
                    "citystreet_night": f1152_055,
                    "residential_day": f1152_055,
                    "residential_night": f1024_055,
                },
            ),
            "idea": "Route by scene: highway is precision fragile, city/residential get more small-object resolution.",
        },
        {
            "label": "night_sahi768_day1024",
            "split_configs": group_policy(f1024_055, day=f1024_055, night=sahi768),
            "idea": "Use SAHI only where full-image night recall is weak, avoid disturbing day rankings.",
        },
        {
            "label": "resnight_sahi896_city1152",
            "split_configs": policy(
                f1024_055,
                {
                    "citystreet_day": f1152_055,
                    "citystreet_night": f1152_055,
                    "residential_day": f1152_055,
                    "residential_night": sahi896,
                },
            ),
            "idea": "Residential night was weak in 28; test targeted large-tile rescue while city uses high-res full.",
        },
        {
            "label": "hardnight_sahi640_rest1024",
            "split_configs": policy(
                f1024_055,
                {
                    "highway_night": sahi640,
                    "residential_night": sahi640,
                    "citystreet_night": f1024_060,
                },
            ),
            "idea": "Most aggressive night-only small-object rescue, with tighter merge to control duplicates.",
        },
    ]


def summarize_policy(candidate: dict) -> str:
    parts = []
    for split in t27.DOMAIN_SPLITS:
        cfg = candidate["split_configs"][split]
        parts.append(f"{split}:{cfg['label']}")
    return "; ".join(parts)


def evaluate_candidate(
    *,
    model: DetectMultiBackend,
    dataset: LoadImagesAndLabels,
    candidate: dict,
    indices: list[int],
    device: torch.device,
    half: bool,
    args: argparse.Namespace,
    phase: str,
) -> list[dict]:
    iouv = torch.linspace(0.5, 0.95, 10)
    niou = iouv.numel()
    total_stats = []
    split_stats = {split: [] for split in t27.DOMAIN_SPLITS}
    split_counts = {split: {"images": 0, "labels": 0} for split in t27.DOMAIN_SPLITS}

    for idx in tqdm(indices, desc=f"{phase}_{candidate['label']}"):
        path = dataset.img_files[idx]
        split = t27.split_from_path(path)
        cfg = candidate["split_configs"][split]
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        h0, w0 = image.shape[:2]
        labelsn = s27.native_labels(dataset.labels[idx], (w0, h0))
        pred = s27.predict_candidate(model=model, image=image, candidate=cfg, device=device, half=half, args=args)
        tcls = labelsn[:, 0].tolist() if len(labelsn) else []
        if len(pred) == 0:
            if len(labelsn):
                stat = (torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls)
                total_stats.append(stat)
                split_stats[split].append(stat)
        else:
            correct = process_batch(pred, labelsn, iouv) if len(labelsn) else torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stat = (correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls)
            total_stats.append(stat)
            split_stats[split].append(stat)
        split_counts[split]["images"] += 1
        split_counts[split]["labels"] += len(labelsn)

    def summarize(stats: list) -> dict:
        if stats:
            packed = [np.concatenate(x, 0) for x in zip(*stats)]
        else:
            packed = [np.zeros((0, niou), dtype=bool), np.array([]), np.array([]), np.array([])]
        if len(packed) and packed[0].any():
            p, r, ap, _f1, ap_class, _cls_thr = ap_per_class(*packed, plot=False, names={})
            return {
                "precision": round(float(p.mean()), 6),
                "recall": round(float(r.mean()), 6),
                "map50": round(float(ap[:, 0].mean()), 6),
                "map50_95": round(float(ap.mean(1).mean()), 6),
                "num_ap_classes": len(ap_class),
            }
        return {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map50_95": 0.0, "num_ap_classes": 0}

    base = {
        "candidate": candidate["label"],
        "phase": phase,
        "policy": summarize_policy(candidate),
        "idea": candidate["idea"],
    }
    rows = [
        {
            **base,
            "split": "scene_daynight_total",
            "images": sum(v["images"] for v in split_counts.values()),
            "labels": sum(v["labels"] for v in split_counts.values()),
            **summarize(total_stats),
        }
    ]
    for split in t27.DOMAIN_SPLITS:
        rows.append(
            {
                **base,
                "split": split,
                "images": split_counts[split]["images"],
                "labels": split_counts[split]["labels"],
                **summarize(split_stats[split]),
            }
        )
    return rows


def append_summary(*, workspace: Path, best: dict, status: str, args: argparse.Namespace) -> None:
    path = REPORTS_ROOT / "30_split_policy_highres_sahi_summary.csv"
    fieldnames = [
        "trial",
        "status",
        "best_candidate",
        "best_phase",
        "best_map50",
        "best_map50_95",
        "previous_best_map50",
        "target_map50",
        "workspace",
        "notebook",
        "metrics_csv",
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
                "best_candidate": best.get("candidate", ""),
                "best_phase": best.get("phase", ""),
                "best_map50": best.get("map50", ""),
                "best_map50_95": best.get("map50_95", ""),
                "previous_best_map50": args.previous_best_map50,
                "target_map50": args.target_map50,
                "workspace": str(workspace),
                "notebook": str(NOTEBOOK_PATH),
                "metrics_csv": str(workspace / "stats" / "30_split_policy_highres_sahi_metrics.csv"),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "rationale": (
                    "Self-only history shows the largest gains came from high-resolution routed MoE and tiled inference, "
                    "not from adding uncalibrated client experts. This run treats scene/day-night as the router and "
                    "selects high-res/SAHI policies per split using only self-generated DQA-MoE checkpoints."
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

    routed = {candidate["label"]: candidate for candidate in o27.make_candidates(workspace)}["day_light_night_hard"]
    if routed["missing"]:
        raise RuntimeError(f"Missing routed checkpoint inputs: {routed['missing']}")
    weights = routed["paths"][0]
    device = select_device(args.device, batch_size=1)
    model = DetectMultiBackend(str(weights), device=device, data=str(split_cfg), fp16=True)
    half = bool(model.fp16)
    model.eval()
    model.warmup(imgsz=(1, 3, 1152, 1152))

    dataset = LoadImagesAndLabels(
        val_cfg.Dataset.val,
        img_size=1024,
        batch_size=1,
        rect=False,
        stride=max(int(model.stride), 32),
        pad=0.0,
        cfg=val_cfg,
        prefix="30: ",
    )
    gate_indices = s27.even_indices(len(dataset), args.gate_images)
    candidates = make_candidates()
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
        "policy",
        "idea",
    ]
    metrics_csv = stats_dir / "30_split_policy_highres_sahi_metrics.csv"
    manifest_path = stats_dir / "30_split_policy_highres_sahi_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "workspace": str(workspace),
                "weights": str(weights),
                "dataset_size": len(dataset),
                "gate_images": len(gate_indices),
                "target_map50": args.target_map50,
                "candidate_labels": [candidate["label"] for candidate in candidates],
                "papers": [
                    "FedMoX (arXiv:2508.16568): routing and Soft-Mixture motivate explicit split-conditioned policy selection.",
                    "SAHI (arXiv:2202.06934): sliced inference motivates targeted small-object rescue.",
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    if not args.no_discord:
        notify(
            "\n".join(
                [
                    "30 started: self-only split-routed high-res/SAHI policy MoE.",
                    f"- target mAP50={args.target_map50:.3f}",
                    f"- gate images={len(gate_indices)} / dataset={len(dataset)}",
                    f"- previous self-only gate/full reference={args.previous_best_map50:.5f}",
                ]
            ),
            "DQA-MoX 30 started",
        )

    rows: list[dict] = []
    gate_totals: list[dict] = []
    for candidate in candidates:
        candidate_rows = evaluate_candidate(
            model=model,
            dataset=dataset,
            candidate=candidate,
            indices=gate_indices,
            device=device,
            half=half,
            args=args,
            phase=f"gate_{candidate['label']}",
        )
        rows.extend(candidate_rows)
        write_rows(metrics_csv, rows, fieldnames)
        total_row = next(row for row in candidate_rows if row["split"] == "scene_daynight_total")
        gate_totals.append(total_row)
        print(f"gate {candidate['label']} mAP50={total_row['map50']:.6f} mAP50:95={total_row['map50_95']:.6f}")

    reference = next(row for row in gate_totals if row["candidate"] == "all_full1024_iou055_ref")
    reference_map50 = float(reference["map50"])
    best_gate = max(gate_totals, key=lambda row: (float(row["map50"]), float(row["map50_95"])))
    best_gain = float(best_gate["map50"]) - reference_map50
    status = "aborted_gate_no_gain"
    best = best_gate
    if best_gate["candidate"] != reference["candidate"] and best_gain >= args.min_gate_gain:
        best_candidate = next(candidate for candidate in candidates if candidate["label"] == best_gate["candidate"])
        full_rows = evaluate_candidate(
            model=model,
            dataset=dataset,
            candidate=best_candidate,
            indices=list(range(len(dataset))),
            device=device,
            half=half,
            args=args,
            phase=f"full_{best_candidate['label']}",
        )
        rows.extend(full_rows)
        write_rows(metrics_csv, rows, fieldnames)
        best = next(row for row in full_rows if row["split"] == "scene_daynight_total")
        status = "target_reached" if float(best["map50"]) >= args.target_map50 else "completed_below_target"

    append_summary(workspace=workspace, best=best, status=status, args=args)
    target_reached = status == "target_reached"
    message = "\n".join(
        [
            f"30 finished. Status={status}",
            f"- gate reference mAP50={reference_map50:.6f}",
            f"- best gate={best_gate['candidate']} mAP50={best_gate['map50']:.6f}; gain={best_gain:+.6f}",
            f"- reported best={best['candidate']} phase={best['phase']} mAP50={best['map50']} / mAP50:95={best['map50_95']}",
            f"- metrics CSV: {metrics_csv}",
            "Decision: target reached." if target_reached else f"Decision: target {args.target_map50:.3f} not reached; continue with a different strategy.",
        ]
    )
    print(message)
    if not args.no_discord:
        notify(message, "DQA-MoX 30 result")
    return 0 if target_reached else 2


if __name__ == "__main__":
    raise SystemExit(main())
