#!/usr/bin/env python3
"""Evaluate self-only counterfactual-view MoE at inference time.

The experiment treats deterministic test-time views as experts: original image,
hflip, night-enhanced image, and SAHI-style tiled views. It keeps the weights
self-only and lets the scene/day-night path router decide which view experts are
allowed to speak.
"""

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
DEFAULT_WORKSPACE = AGG_ROOT / "output" / "33_counterfactual_view_moe"
NOTEBOOK_PATH = SCENE_ROOT / "notebooks" / "33_counterfactual_view_moe.ipynb"

if str(h27.ET_ROOT) not in sys.path:
    sys.path.insert(0, str(h27.ET_ROOT))

from configs.defaults import get_cfg  # noqa: E402
from utils.datasets import LoadImagesAndLabels  # noqa: E402
from utils.detect_multi_backend import DetectMultiBackend  # noqa: E402
from utils.general import check_img_size  # noqa: E402
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
    parser.add_argument("--gate-images", type=int, default=720)
    parser.add_argument("--min-gate-gain", type=float, default=0.006)
    parser.add_argument("--tile-batch-size", type=int, default=8)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--no-discord", action="store_true")
    return parser.parse_args(argv)


def notify(message: str, title: str) -> None:
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from notebook_notify import notify_discord

        print(notify_discord(message, title=title, fail_silently=True))
    except Exception as exc:
        print(f"Discord notification skipped: {exc}")


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def even_indices(n: int, limit: int | None) -> list[int]:
    if limit is None or limit <= 0 or limit >= n:
        return list(range(n))
    positions = np.linspace(0, n - 1, num=limit, dtype=np.int64)
    return sorted(set(int(x) for x in positions))


def full_cfg(label: str, imgsz: int, iou: float = 0.55, merge_iou: float = 0.50) -> dict:
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


def gamma_bgr(image: np.ndarray, gamma: float) -> np.ndarray:
    inv = 1.0 / max(float(gamma), 1e-6)
    table = np.array([(i / 255.0) ** inv * 255.0 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def clahe_bgr(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = cv2.merge([clahe.apply(l_chan), a_chan, b_chan])
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def unflip_detections(det: torch.Tensor, width: int) -> torch.Tensor:
    if det.numel() == 0:
        return det.reshape(0, 6)
    out = det.clone()
    x1 = out[:, 0].clone()
    x2 = out[:, 2].clone()
    out[:, 0] = width - x2
    out[:, 2] = width - x1
    return out


def scale_scores(det: torch.Tensor, scale: float) -> torch.Tensor:
    if det.numel() == 0:
        return det.reshape(0, 6)
    out = det.clone()
    out[:, 4] *= float(scale)
    return out


def view_applies(view: dict, split: str) -> bool:
    applies = view.get("applies", "all")
    if applies == "all":
        return True
    if applies == "day":
        return split in DAY_SPLITS
    if applies == "night":
        return split in NIGHT_SPLITS
    return split == applies


def run_view(
    *,
    model: DetectMultiBackend,
    image: np.ndarray,
    view: dict,
    device: torch.device,
    half: bool,
    args: argparse.Namespace,
) -> torch.Tensor:
    cfg = view["cfg"]
    mode = view["mode"]
    height, width = image.shape[:2]
    if mode == "orig":
        view_image = image
        pred = s27.predict_candidate(model=model, image=view_image, candidate=cfg, device=device, half=half, args=args)
    elif mode == "hflip":
        view_image = cv2.flip(image, 1)
        pred = s27.predict_candidate(model=model, image=view_image, candidate=cfg, device=device, half=half, args=args)
        pred = unflip_detections(pred, width)
    elif mode == "gamma":
        view_image = gamma_bgr(image, view.get("gamma", 1.35))
        pred = s27.predict_candidate(model=model, image=view_image, candidate=cfg, device=device, half=half, args=args)
    elif mode == "clahe":
        view_image = clahe_bgr(image)
        pred = s27.predict_candidate(model=model, image=view_image, candidate=cfg, device=device, half=half, args=args)
    else:
        raise ValueError(f"Unknown view mode: {mode}")
    if pred.numel():
        pred[:, [0, 2]].clamp_(0, width)
        pred[:, [1, 3]].clamp_(0, height)
    return scale_scores(pred.float().cpu(), view.get("score_scale", 1.0))


def make_candidates() -> list[dict]:
    f1024 = full_cfg("full1024_iou055", 1024, 0.55)
    f1152 = full_cfg("full1152_iou055", 1152, 0.55)
    f1024_loose = full_cfg("full1024_iou050", 1024, 0.50)
    sahi768 = sahi_cfg("sahi768_o25_full1024", 768, 768, 0.25)
    sahi896 = sahi_cfg("sahi896_o25_full1024", 896, 896, 0.25)
    return [
        {
            "label": "full1024_ref",
            "views": [{"mode": "orig", "cfg": f1024}],
            "merge_iou": 0.50,
            "idea": "Reference view expert: original 1024px routed self model.",
        },
        {
            "label": "full1024_hflip_s090",
            "views": [
                {"mode": "orig", "cfg": f1024},
                {"mode": "hflip", "cfg": f1024, "score_scale": 0.90},
            ],
            "merge_iou": 0.50,
            "idea": "Counterfactual view MoE: original + horizontal-flip expert, damped to avoid duplicate domination.",
        },
        {
            "label": "full1152_hflip_s088",
            "views": [
                {"mode": "orig", "cfg": f1152},
                {"mode": "hflip", "cfg": f1152, "score_scale": 0.88},
            ],
            "merge_iou": 0.50,
            "idea": "Higher-resolution view MoE for small-object recall, with a conservative hflip expert.",
        },
        {
            "label": "night_gamma_clahe_day1024",
            "views": [
                {"mode": "orig", "cfg": f1024},
                {"mode": "hflip", "cfg": f1024, "score_scale": 0.88, "applies": "night"},
                {"mode": "gamma", "cfg": f1024_loose, "score_scale": 0.72, "gamma": 1.40, "applies": "night"},
                {"mode": "clahe", "cfg": f1024_loose, "score_scale": 0.70, "applies": "night"},
            ],
            "merge_iou": 0.48,
            "idea": "Night-only quality router: add brightness counterfactual experts only where pseudoGT quality was weakest.",
        },
        {
            "label": "sahi768_hflip_s080",
            "views": [
                {"mode": "orig", "cfg": sahi768},
                {"mode": "hflip", "cfg": sahi768, "score_scale": 0.80},
            ],
            "merge_iou": 0.48,
            "idea": "SAHI view MoE: tiled original + tiled hflip, guarded by lower augmented score.",
        },
        {
            "label": "day1152_night_sahi896_gamma",
            "views": [
                {"mode": "orig", "cfg": f1152, "applies": "day"},
                {"mode": "hflip", "cfg": f1152, "score_scale": 0.85, "applies": "day"},
                {"mode": "orig", "cfg": sahi896, "applies": "night"},
                {"mode": "gamma", "cfg": sahi896, "score_scale": 0.65, "gamma": 1.35, "applies": "night"},
            ],
            "merge_iou": 0.48,
            "idea": "Path/domain MoE: day gets high-res view agreement; night gets large-tile recall plus lightened view.",
        },
    ]


def predict_candidate33(
    *,
    model: DetectMultiBackend,
    image: np.ndarray,
    split: str,
    candidate: dict,
    device: torch.device,
    half: bool,
    args: argparse.Namespace,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for view in candidate["views"]:
        if not view_applies(view, split):
            continue
        pred = run_view(model=model, image=image, view=view, device=device, half=half, args=args)
        if pred.numel():
            parts.append(pred)
    if not parts:
        return torch.zeros((0, 6), dtype=torch.float32)
    return s27.nms_detections(torch.cat(parts, dim=0), candidate["merge_iou"], args.max_det).float().cpu()


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

    with torch.no_grad():
        for idx in tqdm(indices, desc=f"{phase}_{candidate['label']}"):
            path = dataset.img_files[idx]
            split = t27.split_from_path(path)
            image = cv2.imread(path)
            if image is None:
                raise FileNotFoundError(path)
            h0, w0 = image.shape[:2]
            labelsn = s27.native_labels(dataset.labels[idx], (w0, h0))
            pred = predict_candidate33(
                model=model,
                image=image,
                split=split,
                candidate=candidate,
                device=device,
                half=half,
                args=args,
            )
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
            p, r, ap, _f1, ap_class, _cls_thr = t27.ap_per_class(*packed, plot=False, names={})
            return {
                "precision": round(float(p.mean()), 6),
                "recall": round(float(r.mean()), 6),
                "map50": round(float(ap[:, 0].mean()), 6),
                "map50_95": round(float(ap.mean(1).mean()), 6),
                "num_ap_classes": int(len(ap_class)),
            }
        return {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map50_95": 0.0, "num_ap_classes": 0}

    rows = []
    total = summarize(total_stats)
    rows.append(
        {
            "phase": phase,
            "candidate": candidate["label"],
            "split": "scene_daynight_total",
            "images": sum(v["images"] for v in split_counts.values()),
            "labels": sum(v["labels"] for v in split_counts.values()),
            **total,
            "merge_iou": candidate["merge_iou"],
            "views": "+".join(f"{v.get('applies', 'all')}:{v['mode']}:{v['cfg']['label']}@{v.get('score_scale', 1.0)}" for v in candidate["views"]),
            "idea": candidate["idea"],
        }
    )
    for split in t27.DOMAIN_SPLITS:
        summary = summarize(split_stats[split])
        rows.append(
            {
                "phase": phase,
                "candidate": candidate["label"],
                "split": split,
                "images": split_counts[split]["images"],
                "labels": split_counts[split]["labels"],
                **summary,
                "merge_iou": candidate["merge_iou"],
                "views": "+".join(f"{v.get('applies', 'all')}:{v['mode']}:{v['cfg']['label']}@{v.get('score_scale', 1.0)}" for v in candidate["views"]),
                "idea": candidate["idea"],
            }
        )
    return rows


def append_summary(*, workspace: Path, best: dict, status: str, args: argparse.Namespace, full_evaluated: bool) -> None:
    path = REPORTS_ROOT / "33_counterfactual_view_moe_summary.csv"
    fieldnames = [
        "trial",
        "status",
        "best_candidate",
        "best_phase",
        "best_map50",
        "best_map50_95",
        "previous_best_map50",
        "target_map50",
        "full_evaluated",
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
                "full_evaluated": full_evaluated,
                "workspace": str(workspace),
                "notebook": str(NOTEBOOK_PATH),
                "metrics_csv": str(workspace / "stats" / "33_counterfactual_view_moe_metrics.csv"),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "rationale": (
                    "Self-only counterfactual-view MoE inspired by FedMoX/FedMoE domain routing and SSOD "
                    "view consistency: no external teacher, only deterministic views of the same self model."
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

    route_candidates = {candidate["label"]: candidate for candidate in o27.make_candidates(workspace)}
    routed = route_candidates["day_light_night_hard"]
    if routed["missing"]:
        raise RuntimeError(f"Missing routed checkpoint inputs: {routed['missing']}")
    weights = routed["paths"][0]

    device = select_device(args.device, batch_size=1)
    model = DetectMultiBackend(str(weights), device=device, data=str(split_cfg), fp16=True)
    stride = max(int(model.stride), 32)
    candidates = make_candidates()
    for candidate in candidates:
        for view in candidate["views"]:
            cfg = view["cfg"]
            cfg["full_imgsz"] = check_img_size(cfg["full_imgsz"], s=stride)
            if cfg["tile_imgsz"]:
                cfg["tile_imgsz"] = check_img_size(cfg["tile_imgsz"], s=stride)
    half = bool(model.fp16)
    model.eval()
    model.warmup(imgsz=(1, 3, 1024, 1024))

    dataset = LoadImagesAndLabels(
        val_cfg.Dataset.val,
        img_size=1024,
        batch_size=1,
        rect=False,
        stride=stride,
        pad=0.0,
        cfg=val_cfg,
        prefix="33: ",
    )
    gate_indices = even_indices(len(dataset), args.gate_images)
    metrics_csv = stats_dir / "33_counterfactual_view_moe_metrics.csv"
    fieldnames = [
        "phase",
        "candidate",
        "split",
        "images",
        "labels",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "num_ap_classes",
        "merge_iou",
        "views",
        "idea",
    ]
    manifest_path = stats_dir / "33_counterfactual_view_moe_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "workspace": str(workspace),
                "weights": str(weights),
                "dataset_size": len(dataset),
                "gate_images": len(gate_indices),
                "candidate_labels": [c["label"] for c in candidates],
                "literature_hooks": [
                    "FedMoX: soft mixture/MoE-style adaptation without relying on a single averaged model.",
                    "(FL)2: confirmation bias suggests using carefully selected high-confidence consistency signals.",
                    "MixPL: pseudo labels miss small/tail objects; view/scale mixing can recover missed boxes.",
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
            "33 started: self-only counterfactual-view MoE. It mixes original/hflip/night-enhanced/SAHI views "
            f"through path-domain routing. Gate={len(gate_indices)} images; full only if gate gain >= {args.min_gate_gain:.3f}.",
            "DQA-MoX 33 started",
        )

    rows: list[dict] = []
    reference_rows = evaluate_candidate(
        model=model,
        dataset=dataset,
        candidate=candidates[0],
        indices=gate_indices,
        device=device,
        half=half,
        args=args,
        phase="gate_full1024_ref",
    )
    rows.extend(reference_rows)
    write_rows(metrics_csv, rows, fieldnames)
    reference_total = next(row for row in reference_rows if row["split"] == "scene_daynight_total")
    baseline_map50 = float(reference_total["map50"])
    print(f"gate reference mAP50={baseline_map50:.6f} mAP50:95={reference_total['map50_95']}")

    gate_totals = []
    for candidate in candidates[1:]:
        cand_rows = evaluate_candidate(
            model=model,
            dataset=dataset,
            candidate=candidate,
            indices=gate_indices,
            device=device,
            half=half,
            args=args,
            phase=f"gate_{candidate['label']}",
        )
        rows.extend(cand_rows)
        write_rows(metrics_csv, rows, fieldnames)
        total = next(row for row in cand_rows if row["split"] == "scene_daynight_total")
        gate_totals.append(total)
        gain = float(total["map50"]) - baseline_map50
        print(f"gate {candidate['label']} mAP50={float(total['map50']):.6f} gain={gain:+.6f}")

    best_gate = max(gate_totals, key=lambda row: (float(row["map50"]), float(row["map50_95"])))
    best_gain = float(best_gate["map50"]) - baseline_map50
    best = best_gate
    full_evaluated = False
    status = "aborted_gate_no_gain"
    if best_gain >= args.min_gate_gain and float(best_gate["map50"]) >= args.previous_best_map50:
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
        full_evaluated = True
        status = "target_reached" if float(best["map50"]) >= args.target_map50 else "completed_below_target"

    append_summary(workspace=workspace, best=best, status=status, args=args, full_evaluated=full_evaluated)
    message = "\n".join(
        [
            f"33 finished. Status={status}",
            f"- gate reference mAP50={baseline_map50:.6f}",
            f"- best gate={best_gate['candidate']} mAP50={float(best_gate['map50']):.6f}; gate gain={best_gain:+.6f}",
            f"- reported best={best['candidate']} phase={best['phase']} mAP50={best['map50']} / mAP50:95={best['map50_95']}",
            f"- full_evaluated={full_evaluated}",
            f"- metrics CSV: {metrics_csv}",
            "Decision: target reached." if float(best["map50"]) >= args.target_map50 else "Decision: target 0.550 not reached; continue with a different strategy.",
        ]
    )
    print(message)
    if not args.no_discord:
        notify(message, "DQA-MoX 33 result")
    return 0 if float(best["map50"]) >= args.target_map50 else 2


if __name__ == "__main__":
    raise SystemExit(main())
