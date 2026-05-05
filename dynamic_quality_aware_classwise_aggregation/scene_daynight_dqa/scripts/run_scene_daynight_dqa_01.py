#!/usr/bin/env python3
"""Run repair-oriented DQA on BDD100K scene x day/night clients."""

from __future__ import annotations

import argparse
import csv
import importlib
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
PROTOCOL_VERSION = "scene_daynight_dqa_01_repair_oriented_v1"

for path in (NAV_ROOT, DQA_ROOT, PSEUDOGT_SCRIPTS, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dqa_cwa_aggregation as dqa_v1  # noqa: E402
import dqa_cwa_aggregation_v2 as dqa_v2  # noqa: E402
import run_pseudogt_learnability_02 as pl02  # noqa: E402
import run_pseudogt_learnability_03 as pl03  # noqa: E402


def configure_modules(workspace: Path, client_limit: int):
    setup = importlib.import_module("setup_scene_daynight")
    setup.WORK_ROOT = workspace
    setup.LIST_ROOT = workspace / "data_lists"
    setup.CONFIG_ROOT = workspace / "configs"
    setup.RUN_ROOT = workspace / "runs"
    setup.CLIENT_LIMIT = client_limit

    fedsto = importlib.import_module("run_fedsto_efficientteacher_exact")
    fedsto.setup = setup
    fedsto.PRETRAINED_PATH = workspace / "weights" / "efficient-yolov5l.pt"
    fedsto.GLOBAL_DIR = workspace / "global_checkpoints"
    fedsto.CLIENT_STATE_DIR = workspace / "client_states"
    fedsto.HISTORY_PATH = workspace / "history.json"
    return setup, fedsto


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def round_tag(round_idx: int) -> str:
    return f"round{round_idx:03d}"


def pseudo_stats_to_dqa_stats(pseudo_stats: dict[str, Any], num_classes: int) -> list[dqa_v1.ClientClassStats]:
    rows = []
    for client_tag, stats in pseudo_stats["clients"].items():
        counts = [0.0] * num_classes
        confidence_sums = [0.0] * num_classes
        localization_sums = [0.0] * num_classes
        quality_sums = [0.0] * num_classes

        box_table = Path(stats["box_table"])
        with box_table.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                cls = int(row["class_id"])
                conf = float(row["conf"])
                stability = float(row["stability"])
                score = float(row["score"])
                counts[cls] += 1.0
                confidence_sums[cls] += conf
                localization_sums[cls] += stability
                quality_sums[cls] += score

        rows.append(
            {
                "client_id": client_tag,
                "counts": counts,
                "confidence_sums": confidence_sums,
                "objectness_sums": confidence_sums,
                "class_confidence_sums": confidence_sums,
                "localization_sums": localization_sums,
                "quality_sums": quality_sums,
            }
        )

    return [
        dqa_v1.ClientClassStats.from_mapping(row, num_classes, default_id=f"client{idx}")
        for idx, row in enumerate(rows)
    ]


def dqa_config(args: argparse.Namespace, num_classes: int) -> dqa_v2.AggregationConfig:
    return dqa_v2.AggregationConfig(
        num_classes=num_classes,
        count_ema=args.dqa_count_ema,
        quality_ema=args.dqa_quality_ema,
        alpha_ema=args.dqa_alpha_ema,
        temperature=args.dqa_temperature,
        uniform_mix=args.dqa_uniform_mix,
        classwise_blend=args.dqa_classwise_blend,
        stability_lambda=args.dqa_stability_lambda,
        min_effective_count=args.dqa_min_effective_count,
        min_quality=args.dqa_min_quality,
        max_quality=1.0,
        server_anchor=args.dqa_server_anchor,
        localize_bn=True,
        min_server_alpha=args.dqa_min_server_alpha,
        residual_blend=args.dqa_residual_blend,
    )


def save_checkpoint_record(
    records: list[dict[str, str]],
    label: str,
    path: Path,
    kind: str,
    *,
    round_idx: int | None = None,
    client: str = "",
) -> None:
    records.append(
        {
            "label": label,
            "kind": kind,
            "round": "" if round_idx is None else str(round_idx),
            "client": client,
            "path": str(path.resolve()),
        }
    )


def run_round(
    setup,
    fedsto,
    variant: pl03.Variant,
    current_global: Path,
    args: argparse.Namespace,
    clients: list[dict[str, Any]],
    round_idx: int,
    port_offset: int,
) -> tuple[list[dict[str, str]], Path, dict[str, Any], dict[str, Any], int]:
    tag = round_tag(round_idx)
    print(f"\n=== {tag}: DQA scene-daynight round ===")
    pseudo_stats = pl03.generate_round_pseudo_labels(setup, current_global, args, clients, round_idx)

    records: list[dict[str, str]] = []
    local_paths: list[Path] = []
    for client in clients:
        client_tag = f"client{client['id']}_{client['weather']}"
        start = fedsto.CLIENT_STATE_DIR / f"dqa01_{tag}_{client_tag}_start.pt"
        run_name = f"pl03_{tag}_{variant.name}_{client_tag}"
        raw_ckpt = fedsto.checkpoint_path(run_name)
        final_ckpt = args.workspace_root / "checkpoints" / f"{tag}_dqa01_{client_tag}.pt"

        if not args.dry_run and not fedsto.checkpoint_matches_protocol(start, PROTOCOL_VERSION):
            fedsto.make_start_checkpoint(current_global, start, protocol=PROTOCOL_VERSION, stage=f"{tag}_{client_tag}_start")

        if not pl03.reusable_checkpoint(fedsto, final_ckpt, args.force):
            cfg = pl03.write_client_config(setup, variant, client, start, args, round_idx)
            raw_ckpt = pl03.run_train(
                setup,
                fedsto,
                cfg,
                dry_run=args.dry_run,
                gpus=args.gpus,
                master_port=args.master_port + port_offset,
            )
            port_offset += 1
            if not args.dry_run:
                fedsto.mark_checkpoint_protocol(raw_ckpt, PROTOCOL_VERSION, f"{tag}_{client_tag}_raw")
                fedsto.make_start_checkpoint(raw_ckpt, final_ckpt, protocol=PROTOCOL_VERSION, stage=f"{tag}_{client_tag}")
        local_paths.append(final_ckpt)
        save_checkpoint_record(records, f"{tag}_{client_tag}", final_ckpt, "client", round_idx=round_idx, client=client_tag)

    stats = pseudo_stats_to_dqa_stats(pseudo_stats, num_classes=len(setup.BDD_NAMES))
    aggregate = args.workspace_root / "checkpoints" / f"{tag}_dqa_cwa_v2_aggregate.pt"
    state_path = args.workspace_root / "stats" / "01_dqa_state.json"
    config = dqa_config(args, len(setup.BDD_NAMES))
    dqa_state: dict[str, Any] = {}
    if not args.dry_run and not pl03.reusable_checkpoint(fedsto, aggregate, args.force):
        _, dqa_state = dqa_v2.aggregate_checkpoints(
            client_checkpoints=local_paths,
            server_checkpoint=current_global,
            output_checkpoint=aggregate,
            stats=stats,
            state_path=state_path,
            config=config,
            repo_root=REPO_ROOT,
        )
        fedsto.mark_checkpoint_protocol(aggregate, PROTOCOL_VERSION, f"{tag}_dqa_cwa_v2_aggregate")
    elif state_path.exists():
        dqa_state = json.loads(state_path.read_text(encoding="utf-8"))
    save_checkpoint_record(records, f"{tag}_dqa_aggregate", aggregate, "aggregate", round_idx=round_idx)

    repair_start = fedsto.GLOBAL_DIR / f"{tag}_dqa_server_repair_start.pt"
    repair = args.workspace_root / "checkpoints" / f"{tag}_dqa_server_repair.pt"
    if args.server_repair_epochs > 0:
        if not args.dry_run and not pl03.reusable_checkpoint(fedsto, repair, args.force):
            fedsto.make_start_checkpoint(aggregate, repair_start, protocol=PROTOCOL_VERSION, stage=f"{tag}_server_repair_start")
            cfg = pl03.write_server_repair_config(setup, variant, repair_start, args, round_idx)
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
                fedsto.mark_checkpoint_protocol(raw_repair, PROTOCOL_VERSION, f"{tag}_server_repair_raw")
                fedsto.make_start_checkpoint(raw_repair, repair, protocol=PROTOCOL_VERSION, stage=f"{tag}_server_repair")
        save_checkpoint_record(records, f"{tag}_server_repair", repair, "server_repair", round_idx=round_idx)
        next_global = repair
    else:
        next_global = aggregate

    return records, next_global, pseudo_stats, dqa_state, port_offset


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


def read_eval_rows(path: Path) -> list[dict[str, str]]:
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


def write_round_metrics(args: argparse.Namespace, rounds: int) -> None:
    summary_path = args.workspace_root / "validation_reports" / "paper_protocol_eval_summary.csv"
    rows = [
        row
        for row in read_eval_rows(summary_path)
        if row.get("status") == "ok" and row.get("split") in {"scene_daynight_total", "total"}
    ]
    by_label = {row["checkpoint_label"]: row for row in rows}
    warm = by_label.get("warmup_global")
    warm_m95 = as_float(warm.get("map50_95")) if warm else None
    metric_rows = []
    prev = None
    for idx in range(1, rounds + 1):
        tag = round_tag(idx)
        agg = by_label.get(f"{tag}_dqa_aggregate")
        rep = by_label.get(f"{tag}_server_repair")
        if not rep:
            continue
        agg_m95 = as_float(agg.get("map50_95")) if agg else None
        rep_m50 = as_float(rep.get("map50"))
        rep_m95 = as_float(rep.get("map50_95"))
        metric_rows.append(
            {
                "round": idx,
                "aggregate_map50_95": "" if agg_m95 is None else f"{agg_m95:.6f}",
                "repaired_map50": "" if rep_m50 is None else f"{rep_m50:.6f}",
                "repaired_map50_95": "" if rep_m95 is None else f"{rep_m95:.6f}",
                "repair_gain_map50_95": "" if rep_m95 is None or agg_m95 is None else f"{rep_m95 - agg_m95:.6f}",
                "retained_gain_map50_95": "" if rep_m95 is None or warm_m95 is None else f"{rep_m95 - warm_m95:.6f}",
                "round_delta_map50_95": "" if rep_m95 is None or prev is None else f"{rep_m95 - prev:.6f}",
            }
        )
        if rep_m95 is not None:
            prev = rep_m95

    metrics_path = args.workspace_root / "stats" / "01_round_metrics.csv"
    write_csv(
        metrics_path,
        metric_rows,
        [
            "round",
            "aggregate_map50_95",
            "repaired_map50",
            "repaired_map50_95",
            "repair_gain_map50_95",
            "retained_gain_map50_95",
            "round_delta_map50_95",
        ],
    )
    values = [as_float(row["repaired_map50_95"]) for row in metric_rows]
    values = [value for value in values if value is not None]
    tail = values[-args.last_n :] if args.last_n > 0 else values
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "warmup_map50_95": warm_m95,
        "final_repaired_map50_95": values[-1] if values else None,
        "last_n": args.last_n,
        "last_n_avg_repaired_map50_95": float(np.mean(tail)) if tail else None,
        "last_n_min_repaired_map50_95": float(np.min(tail)) if tail else None,
        "metrics_csv": str(metrics_path.resolve()),
    }
    summary_json = args.workspace_root / "stats" / "01_round_metrics_summary.json"
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {summary_json}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT / "output" / "01_repair_oriented_scene_daynight_dqa")
    parser.add_argument("--warmup-checkpoint", type=Path, default=REPO_ROOT / "pseudogt_learnability" / "checkpoints" / "round000_warmup.pt")
    parser.add_argument("--client-limit", type=int, default=1500)
    parser.add_argument("--clients", default="all")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=30641)
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.workspace_root = args.workspace_root.expanduser().resolve()
    pl03.ensure_dirs(args.workspace_root)
    setup, fedsto = configure_modules(args.workspace_root, args.client_limit)
    setup_payload = setup.build_base_configs()
    manifest = setup_payload.get("manifest") if isinstance(setup_payload, dict) else {}
    clients = pl02.resolve_clients(args.clients, setup)
    variant = pl03.resolve_variant("repair_oriented_all_lowlr", None)
    warmup = pl02.copy_warmup_to_workspace(args.warmup_checkpoint, args.workspace_root, args.force)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Workspace: {args.workspace_root}")
    print(f"Warmup: {warmup}")
    print(json.dumps({"clients": clients, "server": manifest.get("server"), "dqa": asdict(dqa_config(args, len(setup.BDD_NAMES)))}, indent=2, ensure_ascii=False))

    if args.setup_only:
        print("Setup complete.")
        return 0

    args.gpus = fedsto.resolve_gpus(args.gpus)
    if not args.dry_run:
        fedsto.check_runtime_dependencies()

    records: list[dict[str, str]] = []
    save_checkpoint_record(records, "warmup_global", warmup, "warmup")
    current_global = warmup
    pseudo_history = []
    dqa_history = []
    port_offset = 0

    for idx in range(1, args.rounds + 1):
        round_records, current_global, pseudo_stats, dqa_state, port_offset = run_round(
            setup, fedsto, variant, current_global, args, clients, idx, port_offset
        )
        records.extend(round_records)
        pseudo_history.append(pseudo_stats)
        dqa_history.append({"round": idx, "state": dqa_state})
        write_csv(args.workspace_root / "stats" / "01_checkpoints.csv", records, ["label", "kind", "round", "client", "path"])

    manifest_path = args.workspace_root / "stats" / "01_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "protocol": PROTOCOL_VERSION,
                "project_root": str(PROJECT_ROOT),
                "workspace": str(args.workspace_root),
                "warmup_source": str(args.warmup_checkpoint.expanduser().resolve()),
                "warmup_workspace": str(warmup),
                "client_limit": args.client_limit,
                "rounds": args.rounds,
                "dqa_config": asdict(dqa_config(args, len(setup.BDD_NAMES))),
                "pseudo_history": pseudo_history,
                "dqa_history": dqa_history,
                "checkpoints": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved: {manifest_path}")

    if args.evaluate:
        run_evaluation(args, records)
        write_round_metrics(args, args.rounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
