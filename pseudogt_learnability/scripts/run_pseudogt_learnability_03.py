#!/usr/bin/env python3
"""Run PseudoGT Learnability 03.

03 reframes pseudo-GT learning as a repair-oriented multi-round procedure.
Each round generates stable target pseudo labels from the current repaired
global model, trains source-anchored clients, aggregates them, repairs the
aggregate on supervised source-cloudy GT, and carries the repaired checkpoint
into the next round.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SCRIPT_ROOT = Path(__file__).resolve().parent
PROTOCOL_VERSION = "pseudogt_learnability_03_repair_oriented_v1"

if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import run_pseudogt_learnability_02 as pl02  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    train_scope: str
    aggregate_scope: str
    client_epochs: int
    client_lr0: float
    source_repeat: int
    pseudo_repeat: int
    orthogonal_weight: float
    note: str = ""


VARIANTS: dict[str, Variant] = {
    "repair_oriented_all_lowlr": Variant(
        name="repair_oriented_all_lowlr",
        train_scope="all",
        aggregate_scope="all",
        client_epochs=1,
        client_lr0=0.0005,
        source_repeat=2,
        pseudo_repeat=1,
        orthogonal_weight=1e-4,
        note=(
            "Low-LR full-model client adaptation. Source GT is repeated to keep "
            "the detector close enough for server repair, while stricter stable "
            "pseudo boxes inject target-domain signal."
        ),
    ),
    "repair_oriented_backbone": Variant(
        name="repair_oriented_backbone",
        train_scope="backbone",
        aggregate_scope="backbone",
        client_epochs=1,
        client_lr0=0.0015,
        source_repeat=2,
        pseudo_repeat=1,
        orthogonal_weight=0.0,
        note="A safer diagnostic variant that only adapts the backbone.",
    ),
}


def ensure_dirs(workspace: Path) -> None:
    for relative in (
        "runs",
        "configs",
        "stats",
        "validation_reports",
        "logs",
        "checkpoints",
        "client_states",
        "global_checkpoints",
        "weights",
        "pseudo_dataset",
    ):
        (workspace / relative).mkdir(parents=True, exist_ok=True)


def resolve_variant(raw: str, epochs_override: int | None) -> Variant:
    raw = raw.strip()
    if raw not in VARIANTS:
        raise ValueError(f"Unknown variant {raw!r}. Available: {', '.join(VARIANTS)}")
    variant = VARIANTS[raw]
    if epochs_override is not None:
        variant = replace(variant, client_epochs=epochs_override)
    return variant


def config_device(args: argparse.Namespace) -> str:
    return "" if args.gpus > 1 else args.device


def repeated_expr(path: Path, repeat: int) -> str:
    return str(path.resolve()) if repeat <= 1 else f"{path.resolve()}*{repeat}"


def train_expr(source_list: Path, pseudo_list: Path, variant: Variant) -> str:
    return "||".join(
        (
            repeated_expr(source_list, variant.source_repeat),
            repeated_expr(pseudo_list, variant.pseudo_repeat),
        )
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_round_pseudo_labels(
    setup,
    teacher: Path,
    args: argparse.Namespace,
    clients: list[dict[str, Any]],
    round_idx: int,
) -> dict[str, Any]:
    round_tag = f"round{round_idx:03d}"
    pseudo_root = args.workspace_root / "pseudo_dataset" / f"03_{round_tag}_stable_aug"
    if args.force_pseudo and pseudo_root.exists():
        shutil.rmtree(pseudo_root)

    labeler = pl02.StableAugPseudoLabeler(
        weights=teacher,
        device=args.device,
        imgsz=args.imgsz,
        conf_thres=args.conf_thres,
        iou_thres=args.nms_iou_thres,
        max_det=args.max_det,
    )

    stats_rows: list[dict[str, Any]] = []
    all_client_stats: dict[str, Any] = {}

    for client in clients:
        client_tag = f"client{client['id']}_{client['weather']}"
        source_list = setup.LIST_ROOT / f"client_{client['id']}_{client['weather']}_target.txt"
        images = pl02.read_image_list(source_list, args.max_images_per_client)
        image_boxes: dict[Path, list[pl02.StableBox]] = {}
        dimensions: dict[Path, tuple[int, int]] = {}

        for idx, image_path in enumerate(images, start=1):
            predictions, (width, height) = labeler.predict_views(image_path)
            stable_boxes = pl02.cluster_stable_boxes(
                predictions,
                match_iou=args.match_iou,
                min_views=args.min_views,
                min_stability=args.min_stability,
                min_score=args.min_score,
                max_boxes_per_image=args.max_boxes_per_image,
            )
            if stable_boxes:
                image_boxes[image_path] = stable_boxes
                dimensions[image_path] = (width, height)
            if idx == 1 or idx % args.progress_every == 0 or idx == len(images):
                kept = sum(len(v) for v in image_boxes.values())
                print(f"{round_tag} {client_tag}: pseudo scan {idx}/{len(images)} images, kept {kept} boxes")

        image_boxes = pl02.apply_class_cap(image_boxes, args.max_class_fraction, args.min_class_keep)

        image_dir = pseudo_root / client_tag / "images" / "train"
        label_dir = pseudo_root / client_tag / "labels" / "train"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        list_images: list[Path] = []
        box_rows: list[dict[str, Any]] = []
        class_counts: Counter[int] = Counter()
        confs: list[float] = []
        stabilities: list[float] = []
        scores: list[float] = []

        for image_path, boxes in sorted(image_boxes.items(), key=lambda item: str(item[0])):
            width, height = dimensions[image_path]
            dst_image = image_dir / image_path.name
            dst_label = label_dir / f"{image_path.stem}.txt"
            lines: list[str] = []
            for stable in boxes:
                line = pl02.clipped_yolo_line(stable, width, height)
                if line is None:
                    continue
                lines.append(line)
                class_counts[stable.cls] += 1
                confs.append(stable.conf)
                stabilities.append(stable.stability)
                scores.append(stable.score)
                box_rows.append(
                    {
                        "round": round_tag,
                        "image": str(dst_image.resolve()),
                        "source_image": str(image_path.resolve()),
                        "class_id": stable.cls,
                        "conf": f"{stable.conf:.6f}",
                        "stability": f"{stable.stability:.6f}",
                        "score": f"{stable.score:.6f}",
                        "views": stable.views,
                        "xyxy": " ".join(f"{v:.2f}" for v in stable.xyxy),
                    }
                )
            if not lines:
                continue
            pl02.link_or_copy(image_path, dst_image)
            dst_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
            list_images.append(dst_image.resolve())

        train_list = setup.LIST_ROOT / f"pl03_{round_tag}_{client_tag}_stable_train.txt"
        train_list.write_text("\n".join(str(path) for path in sorted(list_images)) + ("\n" if list_images else ""), encoding="utf-8")
        if not list_images:
            raise RuntimeError(f"No stable pseudo labels were generated for {round_tag} {client_tag}.")

        box_table = args.workspace_root / "stats" / f"03_{round_tag}_{client_tag}_stable_boxes.csv"
        write_csv(
            box_table,
            box_rows,
            ["round", "image", "source_image", "class_id", "conf", "stability", "score", "views", "xyxy"],
        )
        client_stats = {
            "round": round_tag,
            "client": client_tag,
            "teacher": str(teacher.resolve()),
            "source_images_scanned": len(images),
            "pseudo_images_kept": len(list_images),
            "pseudo_boxes_kept": int(sum(class_counts.values())),
            "boxes_per_kept_image": float(sum(class_counts.values()) / max(len(list_images), 1)),
            "mean_conf": float(np.mean(confs)) if confs else 0.0,
            "mean_stability": float(np.mean(stabilities)) if stabilities else 0.0,
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "class_counts": {str(k): int(v) for k, v in sorted(class_counts.items())},
            "train_list": str(train_list.resolve()),
            "box_table": str(box_table.resolve()),
        }
        all_client_stats[client_tag] = client_stats
        stats_rows.append(
            {
                "round": round_tag,
                "client": client_tag,
                "source_images_scanned": client_stats["source_images_scanned"],
                "pseudo_images_kept": client_stats["pseudo_images_kept"],
                "pseudo_boxes_kept": client_stats["pseudo_boxes_kept"],
                "boxes_per_kept_image": f"{client_stats['boxes_per_kept_image']:.4f}",
                "mean_conf": f"{client_stats['mean_conf']:.6f}",
                "mean_stability": f"{client_stats['mean_stability']:.6f}",
                "mean_score": f"{client_stats['mean_score']:.6f}",
                "train_list": client_stats["train_list"],
            }
        )
        print(json.dumps(client_stats, indent=2, ensure_ascii=False))

    csv_path = args.workspace_root / "stats" / f"03_{round_tag}_pseudo_label_stats.csv"
    write_csv(
        csv_path,
        stats_rows,
        [
            "round",
            "client",
            "source_images_scanned",
            "pseudo_images_kept",
            "pseudo_boxes_kept",
            "boxes_per_kept_image",
            "mean_conf",
            "mean_stability",
            "mean_score",
            "train_list",
        ],
    )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "round": round_tag,
        "teacher": str(teacher.resolve()),
        "params": {
            "imgsz": args.imgsz,
            "conf_thres": args.conf_thres,
            "nms_iou_thres": args.nms_iou_thres,
            "match_iou": args.match_iou,
            "min_views": args.min_views,
            "min_stability": args.min_stability,
            "min_score": args.min_score,
            "max_boxes_per_image": args.max_boxes_per_image,
            "max_class_fraction": args.max_class_fraction,
            "min_class_keep": args.min_class_keep,
            "max_images_per_client": args.max_images_per_client,
        },
        "clients": all_client_stats,
        "csv": str(csv_path.resolve()),
    }
    json_path = args.workspace_root / "stats" / f"03_{round_tag}_pseudo_label_stats.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    return payload


def apply_client_hyp(cfg: dict[str, Any], variant: Variant) -> None:
    cfg["linear_lr"] = False
    cfg["hyp"]["lr0"] = variant.client_lr0
    cfg["hyp"]["lrf"] = 1.0
    cfg["hyp"]["warmup_epochs"] = 0
    cfg["hyp"]["mixup"] = 0.0
    cfg["hyp"]["scale"] = 0.25
    cfg["hyp"]["hsv_s"] = 0.35
    cfg["hyp"]["hsv_v"] = 0.20


def write_client_config(
    setup,
    variant: Variant,
    client: dict[str, Any],
    start: Path,
    args: argparse.Namespace,
    round_idx: int,
) -> Path:
    round_tag = f"round{round_idx:03d}"
    client_tag = f"client{client['id']}_{client['weather']}"
    run_name = f"pl03_{round_tag}_{variant.name}_{client_tag}"
    source_list = setup.LIST_ROOT / "server_cloudy_train.txt"
    pseudo_list = setup.LIST_ROOT / f"pl03_{round_tag}_{client_tag}_stable_train.txt"
    cfg = setup.efficientteacher_config(
        name=run_name,
        train=source_list,
        val=setup.LIST_ROOT / "server_cloudy_val.txt",
        target=None,
        weights=str(start.resolve()),
        epochs=variant.client_epochs,
        train_scope=variant.train_scope,
        orthogonal_weight=variant.orthogonal_weight,
        batch_size=args.batch_size,
        workers=args.workers,
        device=config_device(args),
    )
    cfg["Dataset"]["train"] = train_expr(source_list, pseudo_list, variant)
    cfg["FedSTO"]["unlabeled_only_client"] = False
    cfg["SSOD"] = {"train_domain": False}
    apply_client_hyp(cfg, variant)
    return setup.write_config(f"{run_name}.yaml", cfg)


def write_server_repair_config(
    setup,
    variant: Variant,
    start: Path,
    args: argparse.Namespace,
    round_idx: int,
) -> Path:
    round_tag = f"round{round_idx:03d}"
    run_name = f"pl03_{round_tag}_{variant.name}_server_repair"
    cfg = setup.efficientteacher_config(
        name=run_name,
        train=setup.LIST_ROOT / "server_cloudy_train.txt",
        val=setup.LIST_ROOT / "server_cloudy_val.txt",
        target=None,
        weights=str(start.resolve()),
        epochs=args.server_repair_epochs,
        train_scope="all",
        orthogonal_weight=0.0,
        batch_size=args.batch_size,
        workers=args.workers,
        device=config_device(args),
    )
    cfg["SSOD"] = {"train_domain": False}
    cfg["linear_lr"] = False
    cfg["hyp"]["lr0"] = args.server_repair_lr
    cfg["hyp"]["lrf"] = 1.0
    cfg["hyp"]["warmup_epochs"] = 0
    cfg["hyp"]["mixup"] = 0.0
    cfg["hyp"]["scale"] = 0.25
    return setup.write_config(f"{run_name}.yaml", cfg)


def run_train(setup, fedsto, config: Path, *, dry_run: bool, gpus: int, master_port: int) -> Path:
    if gpus > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(gpus),
            "--master_port",
            str(master_port),
            "train.py",
            "--cfg",
            str(config.resolve()),
        ]
    else:
        cmd = [sys.executable, "train.py", "--cfg", str(config.resolve())]
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=setup.ET_ROOT, check=True)
    with config.open(encoding="utf-8") as f:
        run_name = yaml.safe_load(f)["name"]
    return fedsto.checkpoint_path(run_name)


def reusable_checkpoint(fedsto, path: Path, force: bool) -> bool:
    if force or not fedsto.checkpoint_present(path):
        return False
    ok, reason = fedsto.validate_checkpoint(path)
    if ok:
        print(f"Reusing checkpoint: {path}")
        return True
    print(f"Ignoring invalid checkpoint ({reason}): {path}")
    return False


def save_checkpoint_record(
    records: list[dict[str, str]],
    label: str,
    path: Path,
    kind: str,
    variant: str = "",
    client: str = "",
    round_idx: int | None = None,
) -> None:
    records.append(
        {
            "label": label,
            "kind": kind,
            "variant": variant,
            "client": client,
            "round": "" if round_idx is None else str(round_idx),
            "path": str(path.resolve()),
        }
    )


def run_round(
    setup,
    fedsto,
    variant: Variant,
    current_global: Path,
    args: argparse.Namespace,
    clients: list[dict[str, Any]],
    round_idx: int,
    port_offset: int,
) -> tuple[list[dict[str, str]], Path, dict[str, Any], int]:
    round_tag = f"round{round_idx:03d}"
    print(f"\n=== {round_tag}: pseudo labels from {current_global} ===")
    pseudo_stats = generate_round_pseudo_labels(setup, current_global, args, clients, round_idx)

    records: list[dict[str, str]] = []
    local_paths: list[Path] = []

    for client in clients:
        client_tag = f"client{client['id']}_{client['weather']}"
        start = fedsto.CLIENT_STATE_DIR / f"pl03_{round_tag}_{variant.name}_{client_tag}_start.pt"
        run_name = f"pl03_{round_tag}_{variant.name}_{client_tag}"
        raw_ckpt = fedsto.checkpoint_path(run_name)
        final_ckpt = args.workspace_root / "checkpoints" / f"{round_tag}_{variant.name}_{client_tag}.pt"

        if not args.dry_run and not fedsto.checkpoint_matches_protocol(start, PROTOCOL_VERSION):
            fedsto.make_start_checkpoint(
                current_global,
                start,
                protocol=PROTOCOL_VERSION,
                stage=f"{round_tag}_{variant.name}_{client_tag}_start",
            )

        if not reusable_checkpoint(fedsto, final_ckpt, args.force):
            cfg = write_client_config(setup, variant, client, start, args, round_idx)
            raw_ckpt = run_train(
                setup,
                fedsto,
                cfg,
                dry_run=args.dry_run,
                gpus=args.gpus,
                master_port=args.master_port + port_offset,
            )
            port_offset += 1
            if not args.dry_run:
                fedsto.mark_checkpoint_protocol(raw_ckpt, PROTOCOL_VERSION, f"{round_tag}_{variant.name}_{client_tag}_raw")
                fedsto.make_start_checkpoint(
                    raw_ckpt,
                    final_ckpt,
                    protocol=PROTOCOL_VERSION,
                    stage=f"{round_tag}_{variant.name}_{client_tag}",
                )

        local_paths.append(final_ckpt)
        save_checkpoint_record(records, f"{round_tag}_{client_tag}", final_ckpt, "client", variant.name, client_tag, round_idx)

    aggregate = args.workspace_root / "checkpoints" / f"{round_tag}_{variant.name}_aggregate_{variant.aggregate_scope}.pt"
    if not args.dry_run and not reusable_checkpoint(fedsto, aggregate, args.force):
        fedsto.aggregate_checkpoints(local_paths, current_global, aggregate, backbone_only=(variant.aggregate_scope == "backbone"))
        fedsto.mark_checkpoint_protocol(aggregate, PROTOCOL_VERSION, f"{round_tag}_{variant.name}_aggregate_{variant.aggregate_scope}")
    save_checkpoint_record(records, f"{round_tag}_aggregate_{variant.aggregate_scope}", aggregate, "aggregate", variant.name, round_idx=round_idx)

    repair_start = fedsto.GLOBAL_DIR / f"{round_tag}_{variant.name}_server_repair_start.pt"
    repair = args.workspace_root / "checkpoints" / f"{round_tag}_{variant.name}_server_repair.pt"
    repair_raw_name = f"pl03_{round_tag}_{variant.name}_server_repair"
    repair_raw = fedsto.checkpoint_path(repair_raw_name)
    if args.server_repair_epochs > 0:
        if not args.dry_run and not reusable_checkpoint(fedsto, repair, args.force):
            fedsto.make_start_checkpoint(
                aggregate,
                repair_start,
                protocol=PROTOCOL_VERSION,
                stage=f"{round_tag}_{variant.name}_server_repair_start",
            )
            cfg = write_server_repair_config(setup, variant, repair_start, args, round_idx)
            repair_raw = run_train(
                setup,
                fedsto,
                cfg,
                dry_run=args.dry_run,
                gpus=args.gpus,
                master_port=args.master_port + port_offset,
            )
            port_offset += 1
            if not args.dry_run:
                fedsto.mark_checkpoint_protocol(repair_raw, PROTOCOL_VERSION, f"{round_tag}_{variant.name}_server_repair_raw")
                fedsto.make_start_checkpoint(
                    repair_raw,
                    repair,
                    protocol=PROTOCOL_VERSION,
                    stage=f"{round_tag}_{variant.name}_server_repair",
                )
        save_checkpoint_record(records, f"{round_tag}_server_repair", repair, "server_repair", variant.name, round_idx=round_idx)
        next_global = repair
    else:
        next_global = aggregate

    return records, next_global, pseudo_stats, port_offset


def write_checkpoint_table(path: Path, records: list[dict[str, str]]) -> None:
    write_csv(path, records, ["label", "kind", "variant", "client", "round", "path"])


def run_evaluation(args: argparse.Namespace, records: list[dict[str, str]]) -> None:
    allowed = {"warmup", "aggregate", "server_repair"}
    if args.eval_clients:
        allowed.add("client")
    checkpoints = [record for record in records if record["kind"] in allowed]
    cmd = [
        sys.executable,
        str((PROJECT_ROOT / "scripts" / "evaluate_scene_protocol.py").resolve()),
        "--workspace",
        str(args.workspace_root.resolve()),
        "--splits",
        args.eval_splits,
        "--batch-size",
        str(args.val_batch_size),
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.no_eval_plots:
        cmd.append("--no-plots")
    if args.classwise:
        cmd.append("--verbose")
    if args.dry_run:
        cmd.append("--dry-run")
    for record in checkpoints:
        cmd.extend(["--checkpoint", f"{record['label']}={record['path']}"])
    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def read_eval_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def write_round_metrics(args: argparse.Namespace, rounds: int, last_n: int) -> None:
    summary_path = args.workspace_root / "validation_reports" / "paper_protocol_eval_summary.csv"
    rows = [
        row
        for row in read_eval_rows(summary_path)
        if row.get("status") == "ok" and row.get("split") in {"scene_total", "total"}
    ]
    if not rows:
        print(f"No scene-total evaluation rows found for round metrics: {summary_path}")
        return

    by_label = {row["checkpoint_label"]: row for row in rows}
    warmup = by_label.get("warmup_global")
    warmup_m50 = as_float(warmup.get("map50")) if warmup else None
    warmup_m95 = as_float(warmup.get("map50_95")) if warmup else None

    metric_rows: list[dict[str, Any]] = []
    previous_repaired_m95: float | None = None

    for round_idx in range(1, rounds + 1):
        round_tag = f"round{round_idx:03d}"
        aggregate = by_label.get(f"{round_tag}_aggregate_all") or by_label.get(f"{round_tag}_aggregate_backbone")
        repaired = by_label.get(f"{round_tag}_server_repair")
        if not repaired:
            continue

        agg_m50 = as_float(aggregate.get("map50")) if aggregate else None
        agg_m95 = as_float(aggregate.get("map50_95")) if aggregate else None
        rep_m50 = as_float(repaired.get("map50"))
        rep_m95 = as_float(repaired.get("map50_95"))

        client_m95_values = []
        for label, row in by_label.items():
            if label.startswith(f"{round_tag}_client"):
                value = as_float(row.get("map50_95"))
                if value is not None:
                    client_m95_values.append(value)
        avg_client_m95 = float(np.mean(client_m95_values)) if client_m95_values else None

        metric_rows.append(
            {
                "round": round_idx,
                "aggregate_map50": "" if agg_m50 is None else f"{agg_m50:.6f}",
                "aggregate_map50_95": "" if agg_m95 is None else f"{agg_m95:.6f}",
                "repaired_map50": "" if rep_m50 is None else f"{rep_m50:.6f}",
                "repaired_map50_95": "" if rep_m95 is None else f"{rep_m95:.6f}",
                "repair_gain_map50_95": "" if rep_m95 is None or agg_m95 is None else f"{rep_m95 - agg_m95:.6f}",
                "retained_gain_map50_95": "" if rep_m95 is None or warmup_m95 is None else f"{rep_m95 - warmup_m95:.6f}",
                "round_delta_map50_95": "" if rep_m95 is None or previous_repaired_m95 is None else f"{rep_m95 - previous_repaired_m95:.6f}",
                "avg_client_map50_95": "" if avg_client_m95 is None else f"{avg_client_m95:.6f}",
                "client_to_repair_gap_map50_95": "" if avg_client_m95 is None or rep_m95 is None else f"{rep_m95 - avg_client_m95:.6f}",
            }
        )
        if rep_m95 is not None:
            previous_repaired_m95 = rep_m95

    metrics_path = args.workspace_root / "stats" / "03_round_metrics.csv"
    write_csv(
        metrics_path,
        metric_rows,
        [
            "round",
            "aggregate_map50",
            "aggregate_map50_95",
            "repaired_map50",
            "repaired_map50_95",
            "repair_gain_map50_95",
            "retained_gain_map50_95",
            "round_delta_map50_95",
            "avg_client_map50_95",
            "client_to_repair_gap_map50_95",
        ],
    )

    repaired_values = [as_float(row["repaired_map50_95"]) for row in metric_rows]
    repaired_values = [value for value in repaired_values if value is not None]
    tail = repaired_values[-last_n:] if last_n > 0 else repaired_values
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "warmup_map50": warmup_m50,
        "warmup_map50_95": warmup_m95,
        "last_n": last_n,
        "final_repaired_map50_95": repaired_values[-1] if repaired_values else None,
        "last_n_avg_repaired_map50_95": float(np.mean(tail)) if tail else None,
        "last_n_min_repaired_map50_95": float(np.min(tail)) if tail else None,
        "round_metrics_csv": str(metrics_path.resolve()),
    }
    json_path = args.workspace_root / "stats" / "03_round_metrics_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {json_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT / "output" / "03_repair_oriented_multiround")
    parser.add_argument("--warmup-checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "round000_warmup.pt")
    parser.add_argument("--clients", default="all")
    parser.add_argument("--variant", default="repair_oriented_all_lowlr")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=30531)
    parser.add_argument("--device", default="")
    parser.add_argument("--server-repair-epochs", type=int, default=1)
    parser.add_argument("--server-repair-lr", type=float, default=0.0008)
    parser.add_argument("--last-n", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--nms-iou-thres", type=float, default=0.65)
    parser.add_argument("--match-iou", type=float, default=0.60)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--min-stability", type=float, default=0.72)
    parser.add_argument("--min-score", type=float, default=0.28)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-boxes-per-image", type=int, default=12)
    parser.add_argument("--max-images-per-client", type=int, default=0)
    parser.add_argument("--max-class-fraction", type=float, default=0.45)
    parser.add_argument("--min-class-keep", type=int, default=250)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--force-pseudo", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-clients", action="store_true")
    parser.add_argument("--eval-splits", default="highway,citystreet,residential,total")
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--classwise", action="store_true")
    parser.add_argument("--no-eval-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.workspace_root = args.workspace_root.expanduser().resolve()
    ensure_dirs(args.workspace_root)
    setup, fedsto = pl02.configure_modules(args.workspace_root)
    setup_payload = setup.build_base_configs()
    manifest = setup_payload.get("manifest") if isinstance(setup_payload, dict) else None
    if manifest is None:
        manifest_path = args.workspace_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    clients = pl02.resolve_clients(args.clients, setup)
    variant = resolve_variant(args.variant, args.epochs_override)
    warmup = pl02.copy_warmup_to_workspace(args.warmup_checkpoint, args.workspace_root, args.force)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Workspace: {args.workspace_root}")
    print(f"Warmup: {warmup}")
    print(json.dumps({"clients": clients, "server": manifest.get("server"), "variant": asdict(variant)}, indent=2, ensure_ascii=False))

    if args.setup_only:
        print("Setup complete.")
        return 0

    args.gpus = fedsto.resolve_gpus(args.gpus)
    if not args.dry_run:
        fedsto.check_runtime_dependencies()

    all_records: list[dict[str, str]] = []
    save_checkpoint_record(all_records, "warmup_global", warmup, "warmup")
    pseudo_history: list[dict[str, Any]] = []
    current_global = warmup
    port_offset = 0

    for round_idx in range(1, args.rounds + 1):
        records, current_global, pseudo_stats, port_offset = run_round(
            setup,
            fedsto,
            variant,
            current_global,
            args,
            clients,
            round_idx,
            port_offset,
        )
        all_records.extend(records)
        pseudo_history.append(pseudo_stats)
        table_path = args.workspace_root / "stats" / "03_checkpoints.csv"
        write_checkpoint_table(table_path, all_records)
        print(f"Saved: {table_path}")

    stats_root = args.workspace_root / "stats"
    table_path = stats_root / "03_checkpoints.csv"
    write_checkpoint_table(table_path, all_records)
    manifest_path = stats_root / "03_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "protocol": PROTOCOL_VERSION,
                "project_root": str(PROJECT_ROOT),
                "workspace": str(args.workspace_root),
                "warmup_source": str(args.warmup_checkpoint.expanduser().resolve()),
                "warmup_workspace": str(warmup),
                "variant": asdict(variant),
                "rounds": args.rounds,
                "pseudo_history": pseudo_history,
                "checkpoints": all_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved: {table_path}")
    print(f"Saved: {manifest_path}")

    if args.evaluate:
        run_evaluation(args, all_records)
        write_round_metrics(args, args.rounds, args.last_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
