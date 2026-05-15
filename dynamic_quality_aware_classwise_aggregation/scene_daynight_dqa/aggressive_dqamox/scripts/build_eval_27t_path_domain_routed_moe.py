#!/usr/bin/env python3
"""Evaluate 27t path/domain-routed expert MoE policies for scene-daynight DQA."""

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
import build_eval_27k_brightness_routed_moe as k27
import build_eval_27o_asymmetric_score_routed_moe as o27


SCRIPT_PATH = Path(__file__).resolve()
AGG_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[4]
ET_ROOT = h27.ET_ROOT
REPORTS_ROOT = AGG_ROOT / "reports"
DEFAULT_WORKSPACE = AGG_ROOT / "output" / "27t_path_domain_routed_moe"
NOTEBOOK_PATH = AGG_ROOT / "notebooks" / "research_loop_until_060" / "014_27t_path_domain_routed_moe.ipynb"

if str(ET_ROOT) not in sys.path:
    sys.path.insert(0, str(ET_ROOT))

from configs.defaults import get_cfg  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402
from utils.datasets import LoadImagesAndLabels  # noqa: E402
from utils.detect_multi_backend import DetectMultiBackend  # noqa: E402
from utils.general import check_img_size, non_max_suppression, scale_coords  # noqa: E402
from utils.metrics import ap_per_class  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402
from val import process_batch, unwrap_detector_output  # noqa: E402


DOMAIN_SPLITS = [
    "highway_day",
    "highway_night",
    "citystreet_day",
    "citystreet_night",
    "residential_day",
    "residential_night",
]
SPLIT_CLIENTS = {
    "highway_day": "27g_client0_highway_day",
    "highway_night": "27g_client1_highway_night",
    "citystreet_day": "27g_client2_citystreet_day",
    "citystreet_night": "27g_client3_citystreet_night",
    "residential_day": "27g_client4_residential_day",
    "residential_night": "27g_client5_residential_night",
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


def split_from_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    for split in DOMAIN_SPLITS:
        if f"/{split}/" in normalized:
            return split
    raise ValueError(f"Could not infer scene_daynight split from path: {path}")


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
    tensor = tensor.half() if bool(model.fp16) else tensor.float()
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


def route_for_candidate(candidate: dict, split: str) -> list[str]:
    routes = candidate["routes"]
    if split in routes:
        return routes[split]
    return routes["*"]


def scale_for_model(candidate: dict, label: str) -> float:
    if label in candidate.get("score_scales", {}):
        return float(candidate["score_scales"][label])
    for prefix, scale in candidate.get("score_prefix_scales", {}).items():
        if label.startswith(prefix):
            return float(scale)
    return float(candidate.get("default_score_scale", 1.0))


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
    for label in route_for_candidate(candidate, split):
        det = infer_bgr_batch(
            model=models[label],
            images=[image],
            imgsz=imgsz,
            device=device,
            conf_thres=args.conf_thres,
            iou_thres=candidate["pre_iou"],
            max_det=args.max_det,
        )[0]
        if det.numel() == 0:
            continue
        det = det.clone()
        det[:, 4] *= scale_for_model(candidate, label)
        parts.append(det)
    if not parts:
        return torch.zeros((0, 6), dtype=torch.float32)
    merged = torch.cat(parts, dim=0)
    merged = nms_detections(merged, candidate["merge_iou"], args.max_det)
    return merged.float().cpu()


def add_stats(stats: list, pred: torch.Tensor, labelsn: torch.Tensor, iouv: torch.Tensor) -> None:
    nl = len(labelsn)
    tcls = labelsn[:, 0].tolist() if nl else []
    if len(pred) == 0:
        if nl:
            stats.append((torch.zeros(0, iouv.numel(), dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
        return
    if nl:
        correct = process_batch(pred, labelsn, iouv)
    else:
        correct = torch.zeros(pred.shape[0], iouv.numel(), dtype=torch.bool)
    stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))


def summarize_stats(stats: list, niou: int) -> dict:
    if stats:
        packed = [np.concatenate(x, 0) for x in zip(*stats)]
    else:
        packed = [np.zeros((0, niou), dtype=bool), np.array([]), np.array([]), np.array([])]
    if len(packed) and packed[0].any():
        p, r, ap, _f1, ap_class, _cls_thr = ap_per_class(*packed, plot=False, names={})
        ap50 = ap[:, 0]
        ap_all = ap.mean(1)
        mp, mr, map50, map95 = p.mean(), r.mean(), ap50.mean(), ap_all.mean()
        num_ap_classes = len(ap_class)
    else:
        mp = mr = map50 = map95 = 0.0
        num_ap_classes = 0
    return {
        "precision": round(float(mp), 6),
        "recall": round(float(mr), 6),
        "map50": round(float(map50), 6),
        "map50_95": round(float(map95), 6),
        "num_ap_classes": num_ap_classes,
    }


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
    split_stats = {split: [] for split in DOMAIN_SPLITS}
    split_counts = {split: {"images": 0, "labels": 0} for split in DOMAIN_SPLITS}
    total_images = 0
    total_labels = 0

    for idx in tqdm(indices, desc=f"{phase}_{candidate['label']}"):
        path = dataset.img_files[idx]
        split = split_from_path(path)
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        h0, w0 = image.shape[:2]
        labelsn = native_labels(dataset.labels[idx], (w0, h0))
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
        add_stats(total_stats, pred, labelsn, iouv)
        add_stats(split_stats[split], pred, labelsn, iouv)
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
    total_row = {
        **base,
        "split": "scene_daynight_total",
        "images": total_images,
        "labels": int(total_labels),
        **summarize_stats(total_stats, niou),
    }
    rows.append(total_row)
    for split in DOMAIN_SPLITS:
        rows.append(
            {
                **base,
                "split": split,
                "images": split_counts[split]["images"],
                "labels": int(split_counts[split]["labels"]),
                **summarize_stats(split_stats[split], niou),
            }
        )
    return rows


def make_model_specs(workspace: Path) -> tuple[dict[str, Path], list[str], Path]:
    pool = k27.checkpoint_pool()
    route_candidates = {candidate["label"]: candidate for candidate in o27.make_candidates(workspace)}
    routed = route_candidates["day_light_night_hard"]
    specs: dict[str, Path] = {
        "routed_day_light_night_hard": routed["paths"][0],
        "25a_r1_repair": pool["25a_r1_repair"],
        "warmup": pool["warmup"],
        "27g_repair": pool["27g_repair"],
    }
    for label in SPLIT_CLIENTS.values():
        specs[label] = pool[label]
    missing = sorted(label for label, path in specs.items() if not path.exists())
    return specs, missing, routed["paths"][0]


def all_routes(labels: list[str]) -> dict[str, list[str]]:
    return {"*": labels}


def split_client_routes(prefix: list[str], suffix: list[str] | None = None) -> dict[str, list[str]]:
    suffix = suffix or []
    return {split: [*prefix, SPLIT_CLIENTS[split], *suffix] for split in DOMAIN_SPLITS}


def day_ref_night_anchor_routes() -> dict[str, list[str]]:
    routes = {}
    for split in DOMAIN_SPLITS:
        if split.endswith("_day"):
            routes[split] = ["routed_day_light_night_hard"]
        else:
            routes[split] = ["25a_r1_repair", SPLIT_CLIENTS[split]]
    return routes


def make_candidates(args: argparse.Namespace) -> list[dict]:
    base = {"pre_iou": 0.50, "merge_iou": args.merge_iou}
    client_scale = {label: 0.70 for label in SPLIT_CLIENTS.values()}
    night_client_scale = {label: 0.80 for split, label in SPLIT_CLIENTS.items() if split.endswith("_night")}
    return [
        {
            **base,
            "label": "full1024_reference",
            "routes": all_routes(["routed_day_light_night_hard"]),
            "route_summary": "all images -> current best brightness-routed MoE",
            "idea": "Custom evaluator reference matching the 1024px routed MoE plateau.",
        },
        {
            **base,
            "label": "path_domain_client_only",
            "routes": split_client_routes([]),
            "route_summary": "image path split -> matching 27g client expert",
            "idea": "Upper-bound diagnostic for whether 27g local clients became useful domain experts.",
        },
        {
            **base,
            "label": "path_domain_25a_client_s070",
            "routes": split_client_routes(["25a_r1_repair"]),
            "score_scales": {"25a_r1_repair": 1.05, **client_scale},
            "route_summary": "image path split -> 25a anchor + matching 27g client, client score x0.70",
            "idea": "Model-level MoE with explicit domain routing and a conservative client residual.",
        },
        {
            **base,
            "label": "path_domain_25a_27g_client_s065",
            "routes": split_client_routes(["25a_r1_repair", "27g_repair"]),
            "score_scales": {"25a_r1_repair": 1.05, "27g_repair": 0.90, **{v: 0.65 for v in SPLIT_CLIENTS.values()}},
            "route_summary": "image path split -> 25a + global 27g repair + matching client, client score x0.65",
            "idea": "Three-expert domain router: strong anchor, global repair, and damped split specialist.",
        },
        {
            **base,
            "label": "path_domain_dayref_nightclient_s080",
            "routes": day_ref_night_anchor_routes(),
            "score_scales": {"25a_r1_repair": 1.05, **night_client_scale},
            "route_summary": "day -> routed reference; night -> 25a anchor + matching night client x0.80",
            "idea": "Protect day gains from 27o while giving night images a path-domain specialist residual.",
        },
    ]


def required_labels(candidate: dict) -> list[str]:
    labels = set()
    for route in candidate["routes"].values():
        labels.update(route)
    return sorted(labels)


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
        stride = max(int(model.stride), 32)
        max_stride = max(max_stride, stride)
        models[label] = model
    imgsz = check_img_size(imgsz, s=max_stride)
    for model in models.values():
        model.warmup(imgsz=(1, 3, imgsz, imgsz))
    return models, imgsz


def release_models(models: dict[str, DetectMultiBackend]) -> None:
    models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
                "log": str(workspace / "stats" / "27t_path_domain_routed_moe_metrics.csv"),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "rationale": (
                    "27s showed SAHI-like tiling gives only +0.006 gate mAP. 27t uses recent model-level detection "
                    "MoE and FedDG-MoE ideas more aggressively: the validation path reveals scene/day-night domain, "
                    "so this probes an oracle domain router over existing 25a, 27g repair, and 27g client experts."
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
        prefix="27t: ",
    )
    gate_indices = even_indices(len(dataset), args.gate_images)
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
    metrics_csv = stats_dir / "27t_path_domain_routed_moe_metrics.csv"
    manifest_path = stats_dir / "27t_path_domain_routed_moe_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "workspace": str(workspace),
                "dataset_size": len(dataset),
                "gate_images": len(gate_indices),
                "routed_reference": str(routed_ref),
                "model_specs": {label: str(path) for label, path in specs.items()},
                "candidate_labels": [candidate["label"] for candidate in candidates],
                "papers": [
                    "Domain-Specialized Object Detection via Model-Level Mixtures of Experts (arXiv:2604.18256)",
                    "YOLO Meets Mixture-of-Experts: Adaptive Expert Routing for Robust Object Detection (arXiv:2511.13344)",
                    "FedDG-MoE: Test-Time Mixture-of-Experts Fusion for Federated Domain Generalization (CVPRW 2025)",
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
            "27t started: path/domain-routed model-level MoE. "
            f"Gate={len(gate_indices)} images; full run only if gate gain >= {args.min_gate_gain:.3f}.",
            "DQA-MoX 27t started",
        )

    rows: list[dict] = []
    gate_total_rows: list[dict] = []
    baseline_map50 = None
    for candidate in candidates:
        labels = required_labels(candidate)
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
            release_models(models)
        rows.extend(candidate_rows)
        write_rows(metrics_csv, rows, fieldnames)
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
        labels = required_labels(best_candidate)
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
            release_models(models)
        rows.extend(full_rows)
        write_rows(metrics_csv, rows, fieldnames)
        best = next(row for row in full_rows if row["split"] == "scene_daynight_total")
        status = "target_reached" if float(best["map50"]) >= args.target_map50 else "completed"

    append_research_summary(workspace=workspace, best=best, status=status, args=args)
    message = "\n".join(
        [
            f"27t finished. Status={status}",
            f"- gate reference mAP50={baseline_map50:.6f}",
            f"- best gate={best_gate['candidate']} mAP50={best_gate['map50']:.6f}; gate gain={best_gain:+.6f}",
            f"- reported best={best['candidate']} phase={best['phase']} mAP50={best['map50']} / mAP50:95={best['map50_95']}",
            f"- metrics CSV: {metrics_csv}",
            "Decision: target reached." if float(best["map50"]) >= args.target_map50 else "Decision: target 0.600 not reached; continue with a different strategy.",
        ]
    )
    print(message)
    if not args.no_discord:
        notify(message, "DQA-MoX 27t result")
    return 0 if float(best["map50"]) >= args.target_map50 else 2


if __name__ == "__main__":
    raise SystemExit(main())
