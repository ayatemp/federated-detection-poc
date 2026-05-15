#!/usr/bin/env python3
"""Evaluate 27s SAHI-style tiled inference for the routed DQA-MoX model."""

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
import torchvision
from tqdm import tqdm

import build_eval_27h_model_level_moe as h27
import build_eval_27o_asymmetric_score_routed_moe as o27


SCRIPT_PATH = Path(__file__).resolve()
AGG_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[4]
ET_ROOT = h27.ET_ROOT
REPORTS_ROOT = AGG_ROOT / "reports"
DEFAULT_WORKSPACE = AGG_ROOT / "output" / "27s_sahi_routed_moe"
NOTEBOOK_PATH = AGG_ROOT / "notebooks" / "research_loop_until_060" / "013_27s_sahi_routed_moe.ipynb"

if str(ET_ROOT) not in sys.path:
    sys.path.insert(0, str(ET_ROOT))

from configs.defaults import get_cfg  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402
from utils.datasets import LoadImagesAndLabels  # noqa: E402
from utils.detect_multi_backend import DetectMultiBackend  # noqa: E402
from utils.general import check_img_size, clip_coords, non_max_suppression, scale_coords  # noqa: E402
from utils.metrics import ap_per_class  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402
from val import process_batch, unwrap_detector_output  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--target-map50", type=float, default=0.60)
    parser.add_argument("--previous-best-map50", type=float, default=0.523)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gate-images", type=int, default=1200)
    parser.add_argument("--min-gate-gain", type=float, default=0.010)
    parser.add_argument("--tile-batch-size", type=int, default=8)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--python-executable", type=Path, default=None)
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


def native_labels(labels: np.ndarray, shape_wh: tuple[float, float]) -> torch.Tensor:
    if labels.size == 0:
        return torch.zeros((0, 5), dtype=torch.float32)
    w, h = shape_wh
    out = torch.zeros((labels.shape[0], 5), dtype=torch.float32)
    src = torch.from_numpy(labels[:, :5]).float()
    out[:, 0] = src[:, 0]
    out[:, 1] = (src[:, 1] - src[:, 3] / 2.0) * w
    out[:, 2] = (src[:, 2] - src[:, 4] / 2.0) * h
    out[:, 3] = (src[:, 1] + src[:, 3] / 2.0) * w
    out[:, 4] = (src[:, 2] + src[:, 4] / 2.0) * h
    return out


def make_tiles(width: int, height: int, slice_size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    step = max(1, int(round(slice_size * (1.0 - overlap))))

    def starts(length: int) -> list[int]:
        if length <= slice_size:
            return [0]
        values = list(range(0, max(length - slice_size, 0) + 1, step))
        last = length - slice_size
        if values[-1] != last:
            values.append(last)
        return values

    tiles = []
    for y1 in starts(height):
        for x1 in starts(width):
            tiles.append((x1, y1, min(x1 + slice_size, width), min(y1 + slice_size, height)))
    return tiles


def nms_detections(det: torch.Tensor, iou_thres: float, max_det: int) -> torch.Tensor:
    if det.numel() == 0:
        return det.reshape(0, 6)
    det = det[det[:, 4].argsort(descending=True)]
    if det.shape[0] > 30000:
        det = det[:30000]
    offsets = det[:, 5:6] * 7680
    keep = torchvision.ops.nms(det[:, :4] + offsets, det[:, 4], iou_thres)
    if max_det > 0:
        keep = keep[:max_det]
    return det[keep]


def infer_bgr_batch(
    *,
    model: DetectMultiBackend,
    images: list[np.ndarray],
    imgsz: int,
    device: torch.device,
    half: bool,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
) -> list[torch.Tensor]:
    processed = []
    meta = []
    for image in images:
        img, ratio, pad = letterbox(image, new_shape=(imgsz, imgsz), auto=False, scaleup=True)
        processed.append(img)
        meta.append((image.shape[:2], ratio, pad))
    batch = np.stack([im.transpose((2, 0, 1))[::-1] for im in processed])
    batch = np.ascontiguousarray(batch)
    tensor = torch.from_numpy(batch).to(device)
    tensor = tensor.half() if half else tensor.float()
    tensor /= 255.0

    outputs = model(tensor)
    out = unwrap_detector_output(outputs)
    pred = non_max_suppression(out, conf_thres, iou_thres, multi_label=True, max_det=max_det)
    results = []
    for det, (shape0, ratio, pad) in zip(pred, meta):
        if det is not None and len(det):
            det = det.clone()
            scale_coords(tensor.shape[2:], det[:, :4], shape0, ratio_pad=(ratio, pad))
        else:
            det = torch.zeros((0, 6), device=device)
        results.append(det)
    return results


def core_filter(
    det: torch.Tensor,
    *,
    tile: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    margin: int,
) -> torch.Tensor:
    if det.numel() == 0 or margin <= 0:
        return det
    x1, y1, x2, y2 = tile
    height, width = image_shape
    local_w, local_h = x2 - x1, y2 - y1
    left = margin if x1 > 0 else 0
    top = margin if y1 > 0 else 0
    right = local_w - margin if x2 < width else local_w
    bottom = local_h - margin if y2 < height else local_h
    cx = (det[:, 0] + det[:, 2]) / 2.0
    cy = (det[:, 1] + det[:, 3]) / 2.0
    keep = (cx >= left) & (cx <= right) & (cy >= top) & (cy <= bottom)
    return det[keep]


def predict_candidate(
    *,
    model: DetectMultiBackend,
    image: np.ndarray,
    candidate: dict,
    device: torch.device,
    half: bool,
    args: argparse.Namespace,
) -> torch.Tensor:
    height, width = image.shape[:2]
    parts: list[torch.Tensor] = []
    if candidate["include_full"]:
        full = infer_bgr_batch(
            model=model,
            images=[image],
            imgsz=candidate["full_imgsz"],
            device=device,
            half=half,
            conf_thres=args.conf_thres,
            iou_thres=candidate["tile_iou"],
            max_det=args.max_det,
        )[0]
        parts.append(full)

    if candidate["slice_size"] > 0:
        tiles = make_tiles(width, height, candidate["slice_size"], candidate["overlap"])
        margin = int(round(candidate["slice_size"] * candidate["overlap"] * 0.5)) if candidate["core"] else 0
        for start in range(0, len(tiles), args.tile_batch_size):
            chunk = tiles[start : start + args.tile_batch_size]
            crop_images = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in chunk]
            dets = infer_bgr_batch(
                model=model,
                images=crop_images,
                imgsz=candidate["tile_imgsz"],
                device=device,
                half=half,
                conf_thres=args.conf_thres,
                iou_thres=candidate["tile_iou"],
                max_det=args.max_det,
            )
            for det, tile in zip(dets, chunk):
                if det.numel() == 0:
                    continue
                det = core_filter(det, tile=tile, image_shape=(height, width), margin=margin)
                if det.numel() == 0:
                    continue
                x1, y1, _, _ = tile
                det = det.clone()
                det[:, [0, 2]] += x1
                det[:, [1, 3]] += y1
                clip_coords(det[:, :4], (height, width))
                parts.append(det)

    if not parts:
        return torch.zeros((0, 6), dtype=torch.float32)
    merged = torch.cat(parts, dim=0)
    merged = nms_detections(merged, candidate["merge_iou"], args.max_det)
    return merged.float().cpu()


def evaluate_candidate(
    *,
    model: DetectMultiBackend,
    dataset: LoadImagesAndLabels,
    candidate: dict,
    indices: list[int],
    device: torch.device,
    half: bool,
    args: argparse.Namespace,
    desc: str,
) -> dict:
    iouv = torch.linspace(0.5, 0.95, 10)
    niou = iouv.numel()
    stats = []
    seen = 0
    nt_total = 0
    for idx in tqdm(indices, desc=desc):
        path = dataset.img_files[idx]
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        h0, w0 = image.shape[:2]
        labelsn = native_labels(dataset.labels[idx], (w0, h0))
        nl = len(labelsn)
        nt_total += nl
        tcls = labelsn[:, 0].tolist() if nl else []
        pred = predict_candidate(model=model, image=image, candidate=candidate, device=device, half=half, args=args)
        seen += 1
        if len(pred) == 0:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue
        if nl:
            correct = process_batch(pred, labelsn, iouv)
        else:
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    if stats:
        packed = [np.concatenate(x, 0) for x in zip(*stats)]
    else:
        packed = [np.zeros((0, niou), dtype=bool), np.array([]), np.array([]), np.array([])]
    if len(packed) and packed[0].any():
        p, r, ap, _f1, ap_class, _cls_thr = ap_per_class(*packed, plot=False, names={})
        ap50 = ap[:, 0]
        ap_all = ap.mean(1)
        mp, mr, map50, map95 = p.mean(), r.mean(), ap50.mean(), ap_all.mean()
    else:
        mp = mr = map50 = map95 = 0.0
        ap_class = []
    return {
        "candidate": candidate["label"],
        "phase": desc,
        "images": seen,
        "labels": int(nt_total),
        "precision": round(float(mp), 6),
        "recall": round(float(mr), 6),
        "map50": round(float(map50), 6),
        "map50_95": round(float(map95), 6),
        "num_ap_classes": len(ap_class),
        "slice_size": candidate["slice_size"],
        "tile_imgsz": candidate["tile_imgsz"],
        "full_imgsz": candidate["full_imgsz"],
        "overlap": candidate["overlap"],
        "tile_iou": candidate["tile_iou"],
        "merge_iou": candidate["merge_iou"],
        "include_full": candidate["include_full"],
        "core": candidate["core"],
        "idea": candidate["idea"],
    }


def make_candidates() -> list[dict]:
    base = {
        "tile_iou": 0.50,
        "merge_iou": 0.50,
        "include_full": True,
        "full_imgsz": 1024,
        "core": True,
    }
    return [
        {
            **base,
            "label": "full1024_reference",
            "slice_size": 0,
            "tile_imgsz": 1024,
            "overlap": 0.0,
            "core": False,
            "idea": "Custom evaluator reference for the current 1024px routed MoE.",
        },
        {
            **base,
            "label": "sahi896_overlap25_full1024_core",
            "slice_size": 896,
            "tile_imgsz": 896,
            "overlap": 0.25,
            "idea": "Cheap width-sliced inference: keep full context and add two-ish large tiles for small objects.",
        },
        {
            **base,
            "label": "sahi768_overlap25_full1024_core",
            "slice_size": 768,
            "tile_imgsz": 768,
            "overlap": 0.25,
            "idea": "Balanced SAHI-style tiling, using core-window filtering to avoid duplicate border boxes.",
        },
        {
            **base,
            "label": "sahi640_overlap30_full1024_core",
            "slice_size": 640,
            "tile_imgsz": 640,
            "overlap": 0.30,
            "merge_iou": 0.45,
            "idea": "Aggressive small-object rescue with denser tiles and tighter final duplicate suppression.",
        },
    ]


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
                "log": str(workspace / "stats" / "27s_sahi_routed_moe_metrics.csv"),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "rationale": (
                    "27p/27q showed the largest gain came from higher-resolution inference, while routing and score "
                    "recalibration saturated near mAP50 0.523. 27s tests SAHI/ASAHI-inspired tiled inference around "
                    "the current best routed MoE to recover small objects without retraining."
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
        candidate["full_imgsz"] = check_img_size(candidate["full_imgsz"], s=stride)
        if candidate["tile_imgsz"]:
            candidate["tile_imgsz"] = check_img_size(candidate["tile_imgsz"], s=stride)
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
        prefix="27s: ",
    )
    gate_indices = even_indices(len(dataset), args.gate_images)
    fieldnames = [
        "candidate",
        "phase",
        "images",
        "labels",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "num_ap_classes",
        "slice_size",
        "tile_imgsz",
        "full_imgsz",
        "overlap",
        "tile_iou",
        "merge_iou",
        "include_full",
        "core",
        "idea",
    ]
    rows: list[dict] = []
    metrics_csv = stats_dir / "27s_sahi_routed_moe_metrics.csv"
    manifest_path = stats_dir / "27s_sahi_routed_moe_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "workspace": str(workspace),
                "weights": str(weights),
                "dataset_size": len(dataset),
                "gate_images": len(gate_indices),
                "candidate_labels": [c["label"] for c in candidates],
                "papers": [
                    "SAHI: Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection (arXiv:2202.06934)",
                    "ASAHI/DAHI-style adaptive or density-aided slicing papers motivate checking tiled inference after the 1024px plateau.",
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if not args.no_discord:
        notify(
            "27s started: SAHI-style tiled inference around the current best routed MoE. "
            f"Gate={len(gate_indices)} images; full run only if gate gain >= {args.min_gate_gain:.3f}.",
            "DQA-MoX 27s started",
        )

    reference = evaluate_candidate(
        model=model,
        dataset=dataset,
        candidate=candidates[0],
        indices=gate_indices,
        device=device,
        half=half,
        args=args,
        desc="gate_full1024_reference",
    )
    rows.append(reference)
    write_rows(metrics_csv, rows, fieldnames)
    baseline_map50 = float(reference["map50"])
    print(f"gate reference mAP50={baseline_map50:.6f} mAP50:95={reference['map50_95']}")

    gate_rows = []
    for candidate in candidates[1:]:
        row = evaluate_candidate(
            model=model,
            dataset=dataset,
            candidate=candidate,
            indices=gate_indices,
            device=device,
            half=half,
            args=args,
            desc=f"gate_{candidate['label']}",
        )
        gate_rows.append(row)
        rows.append(row)
        write_rows(metrics_csv, rows, fieldnames)
        gain = float(row["map50"]) - baseline_map50
        print(f"gate {candidate['label']} mAP50={row['map50']:.6f} gain={gain:+.6f}")

    best_gate = max(gate_rows, key=lambda row: (float(row["map50"]), float(row["map50_95"])))
    best_gain = float(best_gate["map50"]) - baseline_map50
    status = "aborted_gate_no_gain"
    best = best_gate
    if best_gain >= args.min_gate_gain:
        best_candidate = next(c for c in candidates if c["label"] == best_gate["candidate"])
        full_row = evaluate_candidate(
            model=model,
            dataset=dataset,
            candidate=best_candidate,
            indices=list(range(len(dataset))),
            device=device,
            half=half,
            args=args,
            desc=f"full_{best_candidate['label']}",
        )
        rows.append(full_row)
        write_rows(metrics_csv, rows, fieldnames)
        best = full_row
        status = "target_reached" if float(full_row["map50"]) >= args.target_map50 else "completed"

    append_research_summary(workspace=workspace, best=best, status=status, args=args)
    message = "\n".join(
        [
            f"27s finished. Status={status}",
            f"- gate reference mAP50={baseline_map50:.6f}",
            f"- best gate={best_gate['candidate']} mAP50={best_gate['map50']:.6f}; gate gain={best_gain:+.6f}",
            f"- reported best={best['candidate']} phase={best['phase']} mAP50={best['map50']} / mAP50:95={best['map50_95']}",
            f"- metrics CSV: {metrics_csv}",
            "Decision: target reached." if float(best["map50"]) >= args.target_map50 else "Decision: target 0.600 not reached; continue with a different strategy.",
        ]
    )
    print(message)
    if not args.no_discord:
        notify(message, "DQA-MoX 27s result")
    return 0 if float(best["map50"]) >= args.target_map50 else 2


if __name__ == "__main__":
    raise SystemExit(main())
