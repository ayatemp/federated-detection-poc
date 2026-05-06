#!/usr/bin/env python3
"""Run scene-daynight baseline comparison for DQA 01.

01_0 separates the source-repair effect from the pseudoGT/DQA effect:

* repair_only: warmup -> source repair repeated for N rounds.
* pseudo_fedavg: pseudoGT clients -> plain FedAvg -> source repair.
* pseudo_dqa: pseudoGT clients -> server-anchored DQA-CWA v2 -> source repair.

Each condition uses its own sub-workspace so pseudo labels, configs, and runs do
not collide.  The root workspace stores a combined metrics table.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
NAV_ROOT = REPO_ROOT / "navigating_data_heterogeneity"
DQA_ROOT = PROJECT_ROOT.parent
PSEUDOGT_SCRIPTS = REPO_ROOT / "pseudogt_learnability" / "scripts"
PROTOCOL_VERSION = "scene_daynight_dqa_01_0_baseline_comparison_v1"

for path in (NAV_ROOT, DQA_ROOT, PSEUDOGT_SCRIPTS, PROJECT_ROOT / "scripts", REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dqa_cwa_aggregation_v2 as dqa_v2  # noqa: E402
import run_pseudogt_learnability_02 as pl02  # noqa: E402
import run_pseudogt_learnability_03 as pl03  # noqa: E402
import run_scene_daynight_dqa_01 as dqa01  # noqa: E402


CONDITION_ORDER = ("repair_only", "pseudo_fedavg", "pseudo_dqa")
SPLIT_NAMES = (
    "highway_day",
    "highway_night",
    "citystreet_day",
    "citystreet_night",
    "residential_day",
    "residential_night",
)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def round_tag(round_idx: int) -> str:
    return f"round{round_idx:03d}"


def resolve_conditions(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(CONDITION_ORDER)
    conditions = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(conditions) - set(CONDITION_ORDER))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}. Available: {', '.join(CONDITION_ORDER)}")
    return conditions


def condition_workspace(root: Path, condition: str) -> Path:
    return root / condition


def condition_args(args: argparse.Namespace, workspace: Path) -> argparse.Namespace:
    copied = copy.copy(args)
    copied.workspace_root = workspace
    return copied


def normalize_records(records: list[dict[str, Any]], condition: str) -> list[dict[str, str]]:
    normalized = []
    for record in records:
        normalized.append(
            {
                "condition": condition,
                "label": str(record.get("label", "")),
                "kind": str(record.get("kind", "")),
                "round": str(record.get("round", "")),
                "client": str(record.get("client", "")),
                "variant": str(record.get("variant", "")),
                "path": str(record.get("path", "")),
            }
        )
    return normalized


def write_server_repair_config(
    setup,
    variant: pl03.Variant,
    condition: str,
    start: Path,
    args: argparse.Namespace,
    round_idx: int,
) -> Path:
    tag = round_tag(round_idx)
    run_name = f"sdn010_{condition}_{tag}_server_repair"
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
        device=pl03.config_device(args),
    )
    cfg["SSOD"] = {"train_domain": False}
    cfg["linear_lr"] = False
    cfg["hyp"]["lr0"] = args.server_repair_lr
    cfg["hyp"]["lrf"] = 1.0
    cfg["hyp"]["warmup_epochs"] = 0
    cfg["hyp"]["mixup"] = 0.0
    cfg["hyp"]["scale"] = 0.25
    return setup.write_config(f"{run_name}.yaml", cfg)


def run_repair_only_round(
    setup,
    fedsto,
    variant: pl03.Variant,
    current_global: Path,
    args: argparse.Namespace,
    round_idx: int,
    port_offset: int,
) -> tuple[list[dict[str, str]], Path, int]:
    tag = round_tag(round_idx)
    print(f"\n=== {tag}: repair-only source update ===")
    records: list[dict[str, str]] = []
    repair_start = fedsto.GLOBAL_DIR / f"{tag}_repair_only_start.pt"
    repair = args.workspace_root / "checkpoints" / f"{tag}_repair_only_server_repair.pt"

    if args.server_repair_epochs <= 0:
        return records, current_global, port_offset

    if not args.dry_run and not fedsto.checkpoint_matches_protocol(repair_start, PROTOCOL_VERSION):
        fedsto.make_start_checkpoint(
            current_global,
            repair_start,
            protocol=PROTOCOL_VERSION,
            stage=f"{tag}_repair_only_start",
        )

    if not pl03.reusable_checkpoint(fedsto, repair, args.force):
        cfg = write_server_repair_config(setup, variant, "repair_only", repair_start, args, round_idx)
        raw_repair = pl03.run_train(
            setup,
            fedsto,
            cfg,
            dry_run=args.dry_run,
            gpus=args.gpus,
            master_port=args.master_port + port_offset,
        )
        port_offset += 1
        if not args.dry_run:
            fedsto.mark_checkpoint_protocol(raw_repair, PROTOCOL_VERSION, f"{tag}_repair_only_raw")
            fedsto.make_start_checkpoint(
                raw_repair,
                repair,
                protocol=PROTOCOL_VERSION,
                stage=f"{tag}_repair_only",
            )

    records.append(
        {
            "label": f"{tag}_server_repair",
            "kind": "server_repair",
            "round": str(round_idx),
            "client": "",
            "variant": "repair_only",
            "path": str(repair.resolve()),
        }
    )
    return records, repair, port_offset


def run_evaluation(args: argparse.Namespace, records: list[dict[str, str]]) -> None:
    allowed = {"warmup", "aggregate", "server_repair"}
    if args.eval_clients:
        allowed.add("client")
    cmd = [
        sys.executable,
        str((PROJECT_ROOT / "scripts" / "evaluate_scene_daynight_protocol.py").resolve()),
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
    for record in records:
        if record["kind"] in allowed:
            cmd.extend(["--checkpoint", f"{record['label']}={record['path']}"])
    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def aggregate_label(condition: str, tag: str) -> str | None:
    if condition == "repair_only":
        return None
    if condition == "pseudo_fedavg":
        return f"{tag}_aggregate_all"
    if condition == "pseudo_dqa":
        return f"{tag}_dqa_aggregate"
    raise ValueError(condition)


def split_gap_metrics(by_label_split: dict[tuple[str, str], dict[str, str]], label: str) -> dict[str, Any]:
    split_values: dict[str, float] = {}
    for split in SPLIT_NAMES:
        row = by_label_split.get((label, split))
        value = as_float(row.get("map50_95")) if row else None
        if value is not None:
            split_values[split] = value
    if not split_values:
        return {
            "worst_split": "",
            "worst_split_map50_95": "",
            "day_avg_map50_95": "",
            "night_avg_map50_95": "",
            "day_night_gap_map50_95": "",
        }
    worst_split = min(split_values, key=split_values.get)
    day_values = [value for split, value in split_values.items() if split.endswith("_day")]
    night_values = [value for split, value in split_values.items() if split.endswith("_night")]
    day_avg = float(np.mean(day_values)) if day_values else None
    night_avg = float(np.mean(night_values)) if night_values else None
    return {
        "worst_split": worst_split,
        "worst_split_map50_95": f"{split_values[worst_split]:.6f}",
        "day_avg_map50_95": "" if day_avg is None else f"{day_avg:.6f}",
        "night_avg_map50_95": "" if night_avg is None else f"{night_avg:.6f}",
        "day_night_gap_map50_95": "" if day_avg is None or night_avg is None else f"{day_avg - night_avg:.6f}",
    }


def write_condition_metrics(condition: str, workspace: Path, rounds: int, last_n: int) -> list[dict[str, Any]]:
    summary_path = workspace / "validation_reports" / "paper_protocol_eval_summary.csv"
    rows = [row for row in read_csv(summary_path) if row.get("status") == "ok"]
    total_rows = [row for row in rows if row.get("split") in {"scene_daynight_total", "total"}]
    by_label = {row["checkpoint_label"]: row for row in total_rows}
    by_label_split = {(row["checkpoint_label"], row["split"]): row for row in rows}
    warm = by_label.get("warmup_global")
    warm_m50 = as_float(warm.get("map50")) if warm else None
    warm_m95 = as_float(warm.get("map50_95")) if warm else None

    metric_rows: list[dict[str, Any]] = []
    previous_repaired_m95: float | None = None
    for idx in range(1, rounds + 1):
        tag = round_tag(idx)
        agg_label = aggregate_label(condition, tag)
        rep_label = f"{tag}_server_repair"
        agg = by_label.get(agg_label) if agg_label else None
        rep = by_label.get(rep_label)
        if not rep:
            continue

        agg_m50 = as_float(agg.get("map50")) if agg else None
        agg_m95 = as_float(agg.get("map50_95")) if agg else None
        rep_m50 = as_float(rep.get("map50"))
        rep_m95 = as_float(rep.get("map50_95"))
        gap = split_gap_metrics(by_label_split, rep_label)

        metric_rows.append(
            {
                "condition": condition,
                "round": idx,
                "aggregate_map50": "" if agg_m50 is None else f"{agg_m50:.6f}",
                "aggregate_map50_95": "" if agg_m95 is None else f"{agg_m95:.6f}",
                "repaired_map50": "" if rep_m50 is None else f"{rep_m50:.6f}",
                "repaired_map50_95": "" if rep_m95 is None else f"{rep_m95:.6f}",
                "repair_gain_map50_95": "" if rep_m95 is None or agg_m95 is None else f"{rep_m95 - agg_m95:.6f}",
                "retained_gain_map50_95": "" if rep_m95 is None or warm_m95 is None else f"{rep_m95 - warm_m95:.6f}",
                "round_delta_map50_95": ""
                if rep_m95 is None or previous_repaired_m95 is None
                else f"{rep_m95 - previous_repaired_m95:.6f}",
                **gap,
            }
        )
        if rep_m95 is not None:
            previous_repaired_m95 = rep_m95

    metrics_path = workspace / "stats" / "01_0_condition_metrics.csv"
    fieldnames = [
        "condition",
        "round",
        "aggregate_map50",
        "aggregate_map50_95",
        "repaired_map50",
        "repaired_map50_95",
        "repair_gain_map50_95",
        "retained_gain_map50_95",
        "round_delta_map50_95",
        "worst_split",
        "worst_split_map50_95",
        "day_avg_map50_95",
        "night_avg_map50_95",
        "day_night_gap_map50_95",
    ]
    write_csv(metrics_path, metric_rows, fieldnames)

    repaired_values = [as_float(str(row["repaired_map50_95"])) for row in metric_rows]
    repaired_values = [value for value in repaired_values if value is not None]
    tail = repaired_values[-last_n:] if last_n > 0 else repaired_values
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "condition": condition,
        "warmup_map50": warm_m50,
        "warmup_map50_95": warm_m95,
        "final_repaired_map50_95": repaired_values[-1] if repaired_values else None,
        "last_n": last_n,
        "last_n_avg_repaired_map50_95": float(np.mean(tail)) if tail else None,
        "last_n_min_repaired_map50_95": float(np.min(tail)) if tail else None,
        "metrics_csv": str(metrics_path.resolve()),
    }
    (workspace / "stats" / "01_0_condition_metrics_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metric_rows


def write_combined_metrics(root: Path, all_rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "condition",
        "round",
        "aggregate_map50",
        "aggregate_map50_95",
        "repaired_map50",
        "repaired_map50_95",
        "repair_gain_map50_95",
        "retained_gain_map50_95",
        "round_delta_map50_95",
        "worst_split",
        "worst_split_map50_95",
        "day_avg_map50_95",
        "night_avg_map50_95",
        "day_night_gap_map50_95",
    ]
    metrics_path = root / "stats" / "01_0_all_condition_metrics.csv"
    write_csv(metrics_path, all_rows, fieldnames)

    final_rows = {}
    for row in all_rows:
        final_rows[str(row["condition"])] = row
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_csv": str(metrics_path.resolve()),
        "final_by_condition": final_rows,
    }
    (root / "stats" / "01_0_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved: {metrics_path}")


def prepare_condition(args: argparse.Namespace, condition: str):
    workspace = condition_workspace(args.workspace_root, condition)
    pl03.ensure_dirs(workspace)
    setup, fedsto = dqa01.configure_modules(workspace, args.client_limit)
    setup_payload = setup.build_base_configs()
    manifest = setup_payload.get("manifest") if isinstance(setup_payload, dict) else {}
    clients = pl02.resolve_clients(args.clients, setup)
    variant = pl03.resolve_variant("repair_oriented_all_lowlr", args.epochs_override)
    warmup = pl02.copy_warmup_to_workspace(args.warmup_checkpoint, workspace, args.force)
    return workspace, setup, fedsto, manifest, clients, variant, warmup


def run_condition(args: argparse.Namespace, condition: str, condition_index: int) -> list[dict[str, Any]]:
    workspace = condition_workspace(args.workspace_root, condition)
    completed_metrics = workspace / "stats" / "01_0_condition_metrics.csv"
    completed_summary = workspace / "stats" / "01_0_condition_metrics_summary.json"
    if args.evaluate and not args.force and completed_metrics.exists() and completed_summary.exists():
        print(f"\n\n######## condition={condition} ########")
        print(f"Reusing completed condition metrics: {completed_metrics}")
        return read_csv(completed_metrics)

    workspace, setup, fedsto, manifest, clients, variant, warmup = prepare_condition(args, condition)
    cargs = condition_args(args, workspace)

    print(f"\n\n######## condition={condition} ########")
    print(f"Workspace: {workspace}")
    print(f"Warmup: {warmup}")
    print(json.dumps({"clients": clients, "server": manifest.get("server"), "variant": asdict(variant)}, indent=2, ensure_ascii=False))

    if args.setup_only:
        print(f"Setup complete for {condition}.")
        return []

    cargs.gpus = fedsto.resolve_gpus(cargs.gpus)
    if not cargs.dry_run:
        fedsto.check_runtime_dependencies()

    records: list[dict[str, str]] = [
        {
            "condition": condition,
            "label": "warmup_global",
            "kind": "warmup",
            "round": "",
            "client": "",
            "variant": "",
            "path": str(warmup.resolve()),
        }
    ]
    current_global = warmup
    pseudo_history: list[dict[str, Any]] = []
    dqa_history: list[dict[str, Any]] = []
    port_offset = condition_index * args.condition_port_stride

    for idx in range(1, args.rounds + 1):
        if condition == "repair_only":
            round_records, current_global, port_offset = run_repair_only_round(
                setup,
                fedsto,
                variant,
                current_global,
                cargs,
                idx,
                port_offset,
            )
        elif condition == "pseudo_fedavg":
            round_records, current_global, pseudo_stats, port_offset = pl03.run_round(
                setup,
                fedsto,
                variant,
                current_global,
                cargs,
                clients,
                idx,
                port_offset,
            )
            pseudo_history.append(pseudo_stats)
        elif condition == "pseudo_dqa":
            round_records, current_global, pseudo_stats, dqa_state, port_offset = dqa01.run_round(
                setup,
                fedsto,
                variant,
                current_global,
                cargs,
                clients,
                idx,
                port_offset,
            )
            pseudo_history.append(pseudo_stats)
            dqa_history.append({"round": idx, "state": dqa_state})
        else:
            raise ValueError(condition)

        records.extend(normalize_records(round_records, condition))
        write_csv(
            workspace / "stats" / "01_0_checkpoints.csv",
            records,
            ["condition", "label", "kind", "round", "client", "variant", "path"],
        )

    manifest_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "condition": condition,
        "workspace": str(workspace.resolve()),
        "root_workspace": str(args.workspace_root.resolve()),
        "rounds": args.rounds,
        "client_limit": args.client_limit,
        "warmup_source": str(args.warmup_checkpoint.expanduser().resolve()),
        "warmup_workspace": str(warmup),
        "pseudo_history": pseudo_history,
        "dqa_history": dqa_history,
        "checkpoints": records,
        "dqa_config": asdict(dqa01.dqa_config(cargs, len(setup.BDD_NAMES))) if condition == "pseudo_dqa" else None,
    }
    (workspace / "stats" / "01_0_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.evaluate:
        run_evaluation(cargs, records)
        return write_condition_metrics(condition, workspace, args.rounds, args.last_n)
    return []


def notify(args: argparse.Namespace, message: str, *, title: str, status: str | None = None, error: str | None = None) -> None:
    try:
        from notebook_notify import notify_discord

        context = {
            "workspace": str(args.workspace_root.resolve()),
            "conditions": args.conditions,
            "rounds": args.rounds,
        }
        if status:
            context["status"] = status
        if error:
            context["error"] = error[:500]
        summary_path = args.workspace_root / "stats" / "01_0_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            context["summary"] = str(summary.get("final_by_condition", {}))[:900]
        result = notify_discord(
            message,
            title=title,
            context=context,
            fail_silently=True,
        )
        print(result)
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}")


def run(args: argparse.Namespace) -> None:
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    (args.workspace_root / "stats").mkdir(parents=True, exist_ok=True)
    conditions = resolve_conditions(args.conditions)
    all_rows: list[dict[str, Any]] = []
    for index, condition in enumerate(conditions):
        all_rows.extend(run_condition(args, condition, index))
    if all_rows:
        write_combined_metrics(args.workspace_root, all_rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT / "output" / "01_0_repair_baseline_comparison")
    parser.add_argument("--warmup-checkpoint", type=Path, default=REPO_ROOT / "pseudogt_learnability" / "checkpoints" / "round000_warmup.pt")
    parser.add_argument("--conditions", default="all")
    parser.add_argument("--client-limit", type=int, default=1500)
    parser.add_argument("--clients", default="all")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=30741)
    parser.add_argument("--condition-port-stride", type=int, default=100)
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
    parser.add_argument("--dqa-count-ema", type=float, default=0.65)
    parser.add_argument("--dqa-quality-ema", type=float, default=0.65)
    parser.add_argument("--dqa-alpha-ema", type=float, default=0.40)
    parser.add_argument("--dqa-temperature", type=float, default=2.50)
    parser.add_argument("--dqa-uniform-mix", type=float, default=0.05)
    parser.add_argument("--dqa-classwise-blend", type=float, default=0.08)
    parser.add_argument("--dqa-residual-blend", type=float, default=0.05)
    parser.add_argument("--dqa-min-server-alpha", type=float, default=0.82)
    parser.add_argument("--dqa-server-anchor", type=float, default=20.0)
    parser.add_argument("--dqa-stability-lambda", type=float, default=0.70)
    parser.add_argument("--dqa-min-effective-count", type=float, default=5.0)
    parser.add_argument("--dqa-min-quality", type=float, default=0.10)
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-pseudo", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-clients", action="store_true")
    parser.add_argument(
        "--eval-splits",
        default="highway_day,highway_night,citystreet_day,citystreet_night,residential_day,residential_night,total",
    )
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--classwise", action="store_true")
    parser.add_argument("--no-eval-plots", action="store_true")
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--notify-start", action="store_true")
    parser.add_argument("--notify-end", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    do_start_notify = args.notify or args.notify_start
    do_end_notify = args.notify or args.notify_end
    if do_start_notify:
        notify(args, "Scene-Daynight DQA 01_0 started.", title="Scene-Daynight DQA 01_0 start")

    status = "success"
    error: str | None = None
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = str(exc)
        raise
    finally:
        if do_end_notify:
            notify(
                args,
                f"Scene-Daynight DQA 01_0 finished with status={status}.",
                title="Scene-Daynight DQA 01_0 finish",
                status=status,
                error=error,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
