#!/usr/bin/env python3
"""Evaluate trained domain-6 MoE experts with a tiny client/domain router prior.

38 trains explicit scene/day-night experts, but the final paper-protocol
checkpoint is still evaluated with an image-only router.  This diagnostic keeps
the learned checkpoint and adds only a small per-domain router prior at
deployment time.  In FL terms, the client stores a tiny router adapter rather
than another detector.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SCRIPT_PATH = Path(__file__).resolve()
AGG_ROOT = SCRIPT_PATH.parents[1]
SCENE_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = SCENE_ROOT.parents[1]
ET_ROOT = REPO_ROOT / "navigating_data_heterogeneity" / "vendor" / "efficientteacher"
SCENE_SCRIPTS = SCENE_ROOT / "scripts"
for item in (str(ET_ROOT), str(SCENE_SCRIPTS)):
    if item not in sys.path:
        sys.path.insert(0, item)

import run_scene_daynight_dqa_06_counterfactual_output_moe as output_moe  # noqa: E402


RUN_LABEL = "39"
DEFAULT_SOURCE = AGG_ROOT / "output" / "38_domain6_client_cycle_dqamox_loop" / "38b_domain6_cycle_strict_source_guarded"
DEFAULT_OUTPUT = AGG_ROOT / "output" / "39_domain_prior_router_eval_loop"
SUMMARY_CSV = AGG_ROOT / "reports" / "39_domain_prior_router_eval_loop_summary.csv"
FINAL_METRICS = "39_domain_prior_router_metrics.csv"

SPLIT_TO_EXPERT = {
    "highway_day": 0,
    "highway_night": 1,
    "citystreet_day": 2,
    "citystreet_night": 3,
    "residential_day": 4,
    "residential_night": 5,
}
EVAL_SPLITS = tuple(SPLIT_TO_EXPERT)


@dataclass(frozen=True)
class RouterPrior:
    name: str
    mode: str
    bias: float


PRIORS = (
    RouterPrior("add_bias4", "add", 4.0),
    RouterPrior("force_bias8", "force", 8.0),
)


def notify(message: str, title: str) -> None:
    try:
        from notebook_notify import notify_discord

        print(notify_discord(message, title=title, fail_silently=True))
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}", flush=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def latest_eval_checkpoints(source: Path) -> dict[str, Path]:
    records = source / "stats" / "18_eval_checkpoints.csv"
    if not records.exists():
        raise FileNotFoundError(f"Missing 18_eval_checkpoints.csv in {source}")
    by_label = {row["label"]: Path(row["path"]) for row in read_csv(records)}
    wanted = {
        label: by_label[label]
        for label in ("latent_dqamox_final_aggregate", "latent_dqamox_final_repair")
        if label in by_label and by_label[label].exists()
    }
    if not wanted:
        raise FileNotFoundError(f"No final DQA-MoX checkpoints found in {records}")
    return wanted


def prepare_eval_configs(source: Path, workspace: Path) -> None:
    src = source / "validation_reports" / "paper_protocol_configs"
    dst = workspace / "validation_reports" / "paper_protocol_configs"
    if not src.exists():
        raise FileNotFoundError(f"Missing paper protocol configs: {src}")
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def patch_router_prior(
    source_checkpoint: Path,
    output_checkpoint: Path,
    *,
    expert_idx: int,
    prior: RouterPrior,
) -> Path:
    if output_checkpoint.exists():
        return output_checkpoint
    import torch

    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)

    def patch_model(model: Any) -> None:
        if model is None or not hasattr(model, "state_dict"):
            return
        state = model.state_dict()
        for key, tensor in state.items():
            if not (key.startswith("head.router.") and key.endswith(".bias")):
                continue
            if tensor.ndim != 1 or expert_idx >= tensor.numel():
                continue
            if prior.mode == "force":
                tensor.fill_(-float(prior.bias))
                tensor[expert_idx] = float(prior.bias)
            else:
                tensor[expert_idx] = tensor[expert_idx] + float(prior.bias)
        if prior.mode == "force":
            for key, tensor in state.items():
                if key.startswith("head.router.") and key.endswith(".weight"):
                    tensor.zero_()

    patch_model(checkpoint.get("model"))
    patch_model(checkpoint.get("ema"))
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_checkpoint)
    return output_checkpoint


def split_config(workspace: Path, split: str) -> Path:
    path = workspace / "validation_reports" / "paper_protocol_configs" / f"{split}.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def image_list_from_config(config: Path) -> list[Path]:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    image_list = Path(cfg["Dataset"]["val"])
    return [Path(line.strip()) for line in image_list.read_text(encoding="utf-8").splitlines() if line.strip()]


def export_predictions(
    args: argparse.Namespace,
    label: str,
    checkpoint: Path,
    split: str,
) -> Path:
    runner_args = argparse.Namespace(
        workspace_root=args.workspace_root,
        val_batch_size=args.val_batch_size,
        imgsz=args.imgsz,
        output_conf_thres=args.output_conf_thres,
        output_nms_iou_thres=args.output_nms_iou_thres,
        device=args.device,
        reuse_predictions=args.reuse_predictions,
        dry_run=False,
    )
    return output_moe.run_val_for_predictions(
        runner_args,
        label,
        checkpoint,
        split,
        split_config(args.workspace_root, split),
    )


def evaluate_domain_prior_variant(
    args: argparse.Namespace,
    base_label: str,
    base_checkpoint: Path,
    prior: RouterPrior,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_images: list[Path] = []
    all_predictions: dict[Path, list[tuple[int, Any, float]]] = {}
    for split, expert_idx in SPLIT_TO_EXPERT.items():
        routed_checkpoint = patch_router_prior(
            base_checkpoint,
            args.workspace_root / "checkpoints" / f"39_{base_label}_{prior.name}_{split}_expert{expert_idx}.pt",
            expert_idx=expert_idx,
            prior=prior,
        )
        label = f"39_{base_label}_{prior.name}_{split}"
        labels_dir = export_predictions(args, label, routed_checkpoint, split)
        images = image_list_from_config(split_config(args.workspace_root, split))
        predictions: dict[Path, list[tuple[int, Any, float]]] = {}
        for image_path in images:
            pred_path = labels_dir / f"{image_path.stem}.txt"
            preds = output_moe.read_pred_label(pred_path)
            if args.max_preds_per_image > 0:
                preds = sorted(preds, key=lambda item: item[2], reverse=True)[: args.max_preds_per_image]
            predictions[image_path] = preds
            all_predictions[image_path] = preds
        metrics = output_moe.evaluate_predictions(images, predictions)
        row = {
            "checkpoint_label": f"39_{base_label}_{prior.name}",
            "source_checkpoint": str(base_checkpoint),
            "split": split,
            "expert": expert_idx,
            "prior": prior.name,
            "mode": prior.mode,
            "bias": prior.bias,
            "images": f"{metrics['images']:.0f}",
            "precision": f"{metrics['precision']:.6f}",
            "recall": f"{metrics['recall']:.6f}",
            "map50": f"{metrics['map50']:.6f}",
            "map50_95": f"{metrics['map50_95']:.6f}",
        }
        rows.append(row)
        all_images.extend(images)
    total = output_moe.evaluate_predictions(all_images, all_predictions)
    rows.append(
        {
            "checkpoint_label": f"39_{base_label}_{prior.name}",
            "source_checkpoint": str(base_checkpoint),
            "split": "domain_prior_total",
            "expert": "split_routed",
            "prior": prior.name,
            "mode": prior.mode,
            "bias": prior.bias,
            "images": f"{total['images']:.0f}",
            "precision": f"{total['precision']:.6f}",
            "recall": f"{total['recall']:.6f}",
            "map50": f"{total['map50']:.6f}",
            "map50_95": f"{total['map50_95']:.6f}",
        }
    )
    return rows


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, elapsed: float) -> dict[str, Any]:
    totals = [row for row in rows if row.get("split") == "domain_prior_total"]
    best = max(totals, key=lambda row: float(row["map50"])) if totals else {}
    report = args.workspace_root / "39_domain_prior_router_eval_report.md"
    lines = [
        "# DQA-MoX 39 Domain-Prior Router Evaluation",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- source_workspace: `{args.source_workspace}`",
        f"- target_mAP50: {args.target_map50:.3f}",
        f"- elapsed_seconds: {elapsed:.1f}",
        "",
        "## Best",
        "",
        f"- label: `{best.get('checkpoint_label', '')}`",
        f"- mAP50: {best.get('map50', '')}",
        f"- mAP50:95: {best.get('map50_95', '')}",
        "",
        "## Total Rows",
        "",
        "| label | mAP50 | mAP50:95 | mode | bias |",
        "|---|---:|---:|---|---:|",
    ]
    for row in totals:
        lines.append(
            f"| {row['checkpoint_label']} | {row['map50']} | {row['map50_95']} | "
            f"{row['mode']} | {row['bias']} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "best": best,
        "report": report,
        "target_reached": bool(best) and float(best["map50"]) >= args.target_map50,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workspace", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-map50", type=float, default=0.55)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--output-conf-thres", type=float, default=0.001)
    parser.add_argument("--output-nms-iou-thres", type=float, default=0.60)
    parser.add_argument("--max-preds-per-image", type=int, default=300)
    parser.add_argument("--device", default="")
    parser.add_argument("--reuse-predictions", action="store_true")
    parser.add_argument("--no-discord", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.source_workspace = args.source_workspace.expanduser().resolve()
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    (args.workspace_root / "stats").mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    if not args.no_discord:
        notify(
            "\n".join(
                [
                    "39 started: domain/client router prior on trained 38 MoE experts.",
                    f"- source={args.source_workspace}",
                    "- idea: shared learned MoE checkpoint + tiny domain prior, not six full client models.",
                ]
            ),
            "DQA-MoX 39 started",
        )

    prepare_eval_configs(args.source_workspace, args.workspace_root)
    checkpoint_map = latest_eval_checkpoints(args.source_workspace)
    all_rows: list[dict[str, Any]] = []
    for base_label, checkpoint in checkpoint_map.items():
        for prior in PRIORS:
            all_rows.extend(evaluate_domain_prior_variant(args, base_label, checkpoint, prior))

    fields = [
        "checkpoint_label",
        "source_checkpoint",
        "split",
        "expert",
        "prior",
        "mode",
        "bias",
        "images",
        "precision",
        "recall",
        "map50",
        "map50_95",
    ]
    metrics_path = args.workspace_root / "stats" / FINAL_METRICS
    write_csv(metrics_path, all_rows, fields)
    elapsed = time.monotonic() - start
    summary = summarize(all_rows, args, elapsed)
    best = summary["best"]
    row = {
        "run": RUN_LABEL,
        "status": "target_reached" if summary["target_reached"] else "below_target",
        "best_label": best.get("checkpoint_label", ""),
        "best_map50": best.get("map50", ""),
        "best_map50_95": best.get("map50_95", ""),
        "target_map50": f"{args.target_map50:.6f}",
        "workspace": str(args.workspace_root),
        "source_workspace": str(args.source_workspace),
        "metrics_csv": str(metrics_path),
        "report": str(summary["report"]),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": f"{elapsed:.1f}",
    }
    write_csv(SUMMARY_CSV, [row], list(row.keys()))
    message = "\n".join(
        [
            "39 finished: domain-prior router eval.",
            f"- status={row['status']}",
            f"- best={row['best_label']} mAP50={row['best_map50']} / mAP50:95={row['best_map50_95']}",
            f"- metrics={metrics_path}",
            f"- report={summary['report']}",
        ]
    )
    print(message)
    if not args.no_discord:
        notify(message, "DQA-MoX 39 result")
    return 0 if summary["target_reached"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
