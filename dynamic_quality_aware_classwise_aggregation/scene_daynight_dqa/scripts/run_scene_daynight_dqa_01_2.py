#!/usr/bin/env python3
"""Run scene-daynight DQA 01_2 SSOD pivot.

If fixed pseudoGT supervision still cannot beat source repair, 01_2 pivots to a
more FedSTO-faithful client update: clients train with EfficientTeacher SSOD on
their unlabeled target images.  Stable pseudo boxes are generated only as a DQA
quality signal for aggregation, not as fixed training labels.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
NAV_ROOT = REPO_ROOT / "navigating_data_heterogeneity"
DQA_ROOT = PROJECT_ROOT.parent
PSEUDOGT_SCRIPTS = REPO_ROOT / "pseudogt_learnability" / "scripts"
PROTOCOL_VERSION = "scene_daynight_dqa_01_2_ssod_pivot_v1"

for path in (NAV_ROOT, DQA_ROOT, PSEUDOGT_SCRIPTS, PROJECT_ROOT / "scripts", REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dqa_cwa_aggregation_v2 as dqa_v2  # noqa: E402
import run_pseudogt_learnability_02 as pl02  # noqa: E402
import run_pseudogt_learnability_03 as pl03  # noqa: E402
import run_scene_daynight_dqa_01 as dqa01  # noqa: E402
import run_scene_daynight_dqa_01_0 as base01_0  # noqa: E402


SPLIT_NAMES = base01_0.SPLIT_NAMES


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    mode: str
    note: str
    train_scope: str = "all"
    aggregate_scope: str = "all"
    client_lr0: float = 0.001
    orthogonal_weight: float = 1e-4
    ssod_box_loss_weight: float = 0.05
    dqa_min_server_alpha: float | None = None
    dqa_server_anchor: float | None = None
    dqa_residual_blend: float | None = None
    dqa_classwise_blend: float | None = None


CONDITION_SPECS: dict[str, ConditionSpec] = {
    "repair_only": ConditionSpec(
        name="repair_only",
        mode="repair_only",
        note="Control: warmup plus repeated supervised source repair only.",
    ),
    "ssod_fedavg": ConditionSpec(
        name="ssod_fedavg",
        mode="ssod_fedavg",
        note="FedSTO-like SSOD client training with plain FedAvg aggregation.",
        client_lr0=0.001,
    ),
    "ssod_dqa": ConditionSpec(
        name="ssod_dqa",
        mode="ssod_dqa",
        note="SSOD client training; stable pseudo boxes are used only for DQA aggregation quality.",
        client_lr0=0.001,
        dqa_min_server_alpha=0.72,
        dqa_server_anchor=12.0,
        dqa_residual_blend=0.12,
        dqa_classwise_blend=0.14,
    ),
    "ssod_dqa_head": ConditionSpec(
        name="ssod_dqa_head",
        mode="ssod_dqa",
        note="Head-only SSOD target adaptation with weak bbox loss.",
        train_scope="head",
        client_lr0=0.0015,
        ssod_box_loss_weight=0.01,
        dqa_min_server_alpha=0.70,
        dqa_server_anchor=10.0,
        dqa_residual_blend=0.14,
        dqa_classwise_blend=0.16,
    ),
    "ssod_dqa_nonbackbone": ConditionSpec(
        name="ssod_dqa_nonbackbone",
        mode="ssod_dqa",
        note="Neck/head SSOD target adaptation with reduced bbox loss.",
        train_scope="non_backbone",
        client_lr0=0.0012,
        ssod_box_loss_weight=0.02,
        dqa_min_server_alpha=0.70,
        dqa_server_anchor=10.0,
        dqa_residual_blend=0.14,
        dqa_classwise_blend=0.16,
    ),
}

DEFAULT_CONDITIONS = ("repair_only", "ssod_fedavg", "ssod_dqa", "ssod_dqa_head", "ssod_dqa_nonbackbone")


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


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def round_tag(round_idx: int) -> str:
    return f"round{round_idx:03d}"


def resolve_conditions(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(DEFAULT_CONDITIONS)
    conditions = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(conditions) - set(CONDITION_SPECS))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}. Available: {', '.join(CONDITION_SPECS)}")
    return conditions


def condition_workspace(root: Path, condition: str) -> Path:
    return root / condition


def condition_args(args: argparse.Namespace, workspace: Path, spec: ConditionSpec) -> argparse.Namespace:
    copied = copy.copy(args)
    copied.workspace_root = workspace
    if spec.dqa_min_server_alpha is not None:
        copied.dqa_min_server_alpha = spec.dqa_min_server_alpha
    if spec.dqa_server_anchor is not None:
        copied.dqa_server_anchor = spec.dqa_server_anchor
    if spec.dqa_residual_blend is not None:
        copied.dqa_residual_blend = spec.dqa_residual_blend
    if spec.dqa_classwise_blend is not None:
        copied.dqa_classwise_blend = spec.dqa_classwise_blend
    return copied


def config_device(args: argparse.Namespace) -> str:
    return "" if args.gpus > 1 else args.device


def write_ssod_client_config(
    setup,
    spec: ConditionSpec,
    client: dict[str, Any],
    start: Path,
    args: argparse.Namespace,
    round_idx: int,
) -> Path:
    tag = round_tag(round_idx)
    client_tag = f"client{client['id']}_{client['weather']}"
    run_name = f"sds012_{tag}_{spec.name}_{client_tag}"
    source_list = setup.LIST_ROOT / "server_cloudy_train.txt"
    target_list = setup.LIST_ROOT / f"client_{client['id']}_{client['weather']}_target.txt"
    cfg = setup.efficientteacher_config(
        name=run_name,
        train=source_list,
        val=setup.LIST_ROOT / "server_cloudy_val.txt",
        target=target_list,
        weights=str(start.resolve()),
        epochs=args.client_epochs,
        train_scope=spec.train_scope,
        orthogonal_weight=spec.orthogonal_weight,
        batch_size=args.batch_size,
        workers=args.workers,
        device=config_device(args),
    )
    cfg["linear_lr"] = False
    cfg["hyp"]["lr0"] = spec.client_lr0
    cfg["hyp"]["lrf"] = 1.0
    cfg["hyp"]["warmup_epochs"] = 0
    cfg["hyp"]["mixup"] = 0.0
    cfg["hyp"]["scale"] = 0.25
    cfg["hyp"]["hsv_s"] = 0.35
    cfg["hyp"]["hsv_v"] = 0.20
    cfg["FedSTO"]["unlabeled_only_client"] = True
    cfg["SSOD"]["train_domain"] = True
    cfg["SSOD"]["box_loss_weight"] = float(spec.ssod_box_loss_weight)
    cfg["SSOD"]["teacher_loss_weight"] = 1.0
    cfg["SSOD"]["pseudo_label_with_bbox"] = True
    cfg["SSOD"]["pseudo_label_with_cls"] = True
    cfg["SSOD"]["pseudo_label_with_obj"] = True
    return setup.write_config(f"{run_name}.yaml", cfg)


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


def save_checkpoint_record(
    records: list[dict[str, str]],
    label: str,
    path: Path,
    kind: str,
    *,
    round_idx: int | None = None,
    client: str = "",
    variant: str = "",
) -> None:
    records.append(
        {
            "label": label,
            "kind": kind,
            "round": "" if round_idx is None else str(round_idx),
            "client": client,
            "variant": variant,
            "path": str(path.resolve()),
        }
    )


def aggregate_label(spec: ConditionSpec, tag: str) -> str | None:
    if spec.mode == "repair_only":
        return None
    if spec.mode == "ssod_fedavg":
        return f"{tag}_ssod_aggregate_{spec.aggregate_scope}"
    if spec.mode == "ssod_dqa":
        return f"{tag}_ssod_dqa_aggregate"
    raise ValueError(spec.mode)


def run_ssod_round(
    setup,
    fedsto,
    spec: ConditionSpec,
    current_global: Path,
    args: argparse.Namespace,
    clients: list[dict[str, Any]],
    round_idx: int,
    port_offset: int,
) -> tuple[list[dict[str, str]], Path, dict[str, Any] | None, dict[str, Any] | None, int]:
    tag = round_tag(round_idx)
    print(f"\n=== {tag}: {spec.name} SSOD client round ===")
    records: list[dict[str, str]] = []
    local_paths: list[Path] = []
    pseudo_stats: dict[str, Any] | None = None
    dqa_state: dict[str, Any] | None = None

    if spec.mode == "ssod_dqa":
        pseudo_stats = pl03.generate_round_pseudo_labels(setup, current_global, args, clients, round_idx)

    for client in clients:
        client_tag = f"client{client['id']}_{client['weather']}"
        start = fedsto.CLIENT_STATE_DIR / f"sds012_{tag}_{spec.name}_{client_tag}_start.pt"
        run_name = f"sds012_{tag}_{spec.name}_{client_tag}"
        raw_ckpt = fedsto.checkpoint_path(run_name)
        final_ckpt = args.workspace_root / "checkpoints" / f"{tag}_{spec.name}_{client_tag}.pt"

        if not args.dry_run and not fedsto.checkpoint_matches_protocol(start, PROTOCOL_VERSION):
            fedsto.make_start_checkpoint(current_global, start, protocol=PROTOCOL_VERSION, stage=f"{tag}_{spec.name}_{client_tag}_start")

        if not pl03.reusable_checkpoint(fedsto, final_ckpt, args.force):
            cfg = write_ssod_client_config(setup, spec, client, start, args, round_idx)
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
                fedsto.mark_checkpoint_protocol(raw_ckpt, PROTOCOL_VERSION, f"{tag}_{spec.name}_{client_tag}_raw")
                fedsto.make_start_checkpoint(raw_ckpt, final_ckpt, protocol=PROTOCOL_VERSION, stage=f"{tag}_{spec.name}_{client_tag}")

        local_paths.append(final_ckpt)
        save_checkpoint_record(records, f"{tag}_{client_tag}", final_ckpt, "client", round_idx=round_idx, client=client_tag, variant=spec.name)

    if spec.mode == "ssod_fedavg":
        aggregate = args.workspace_root / "checkpoints" / f"{tag}_{spec.name}_aggregate_{spec.aggregate_scope}.pt"
        if not args.dry_run and not pl03.reusable_checkpoint(fedsto, aggregate, args.force):
            fedsto.aggregate_checkpoints(local_paths, current_global, aggregate, backbone_only=(spec.aggregate_scope == "backbone"))
            fedsto.mark_checkpoint_protocol(aggregate, PROTOCOL_VERSION, f"{tag}_{spec.name}_aggregate_{spec.aggregate_scope}")
        save_checkpoint_record(records, f"{tag}_ssod_aggregate_{spec.aggregate_scope}", aggregate, "aggregate", round_idx=round_idx, variant=spec.name)
    else:
        assert pseudo_stats is not None
        stats = dqa01.pseudo_stats_to_dqa_stats(pseudo_stats, num_classes=len(setup.BDD_NAMES))
        aggregate = args.workspace_root / "checkpoints" / f"{tag}_{spec.name}_dqa_aggregate.pt"
        state_path = args.workspace_root / "stats" / f"01_2_{spec.name}_dqa_state.json"
        config = dqa01.dqa_config(args, len(setup.BDD_NAMES))
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
            fedsto.mark_checkpoint_protocol(aggregate, PROTOCOL_VERSION, f"{tag}_{spec.name}_dqa_aggregate")
        elif state_path.exists():
            dqa_state = json.loads(state_path.read_text(encoding="utf-8"))
        save_checkpoint_record(records, f"{tag}_ssod_dqa_aggregate", aggregate, "aggregate", round_idx=round_idx, variant=spec.name)

    repair_start = fedsto.GLOBAL_DIR / f"{tag}_{spec.name}_server_repair_start.pt"
    repair = args.workspace_root / "checkpoints" / f"{tag}_{spec.name}_server_repair.pt"
    if args.server_repair_epochs > 0:
        if not args.dry_run and not pl03.reusable_checkpoint(fedsto, repair, args.force):
            fedsto.make_start_checkpoint(aggregate, repair_start, protocol=PROTOCOL_VERSION, stage=f"{tag}_{spec.name}_server_repair_start")
            variant = pl03.Variant(spec.name, spec.train_scope, spec.aggregate_scope, 1, spec.client_lr0, 1, 1, spec.orthogonal_weight)
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
                fedsto.mark_checkpoint_protocol(raw_repair, PROTOCOL_VERSION, f"{tag}_{spec.name}_server_repair_raw")
                fedsto.make_start_checkpoint(raw_repair, repair, protocol=PROTOCOL_VERSION, stage=f"{tag}_{spec.name}_server_repair")
        save_checkpoint_record(records, f"{tag}_server_repair", repair, "server_repair", round_idx=round_idx, variant=spec.name)
        next_global = repair
    else:
        next_global = aggregate

    return records, next_global, pseudo_stats, dqa_state, port_offset


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


def write_condition_metrics(spec: ConditionSpec, workspace: Path, rounds: int, last_n: int) -> list[dict[str, Any]]:
    rows = [row for row in read_csv(workspace / "validation_reports" / "paper_protocol_eval_summary.csv") if row.get("status") == "ok"]
    total_rows = [row for row in rows if row.get("split") in {"scene_daynight_total", "total"}]
    by_label = {row["checkpoint_label"]: row for row in total_rows}
    by_label_split = {(row["checkpoint_label"], row["split"]): row for row in rows}
    warm = by_label.get("warmup_global")
    warm_m50 = as_float(warm.get("map50")) if warm else None
    warm_m95 = as_float(warm.get("map50_95")) if warm else None

    metric_rows: list[dict[str, Any]] = []
    prev_m95: float | None = None
    for idx in range(1, rounds + 1):
        tag = round_tag(idx)
        agg_label = aggregate_label(spec, tag)
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
                "condition": spec.name,
                "mode": spec.mode,
                "round": idx,
                "aggregate_map50": "" if agg_m50 is None else f"{agg_m50:.6f}",
                "aggregate_map50_95": "" if agg_m95 is None else f"{agg_m95:.6f}",
                "aggregate_survival_vs_warmup_map50_95": "" if agg_m95 is None or warm_m95 is None else f"{agg_m95 - warm_m95:.6f}",
                "repaired_map50": "" if rep_m50 is None else f"{rep_m50:.6f}",
                "repaired_map50_95": "" if rep_m95 is None else f"{rep_m95:.6f}",
                "repair_gain_map50_95": "" if rep_m95 is None or agg_m95 is None else f"{rep_m95 - agg_m95:.6f}",
                "retained_gain_map50_95": "" if rep_m95 is None or warm_m95 is None else f"{rep_m95 - warm_m95:.6f}",
                "round_delta_map50_95": "" if rep_m95 is None or prev_m95 is None else f"{rep_m95 - prev_m95:.6f}",
                **gap,
            }
        )
        if rep_m95 is not None:
            prev_m95 = rep_m95

    fieldnames = [
        "condition",
        "mode",
        "round",
        "aggregate_map50",
        "aggregate_map50_95",
        "aggregate_survival_vs_warmup_map50_95",
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
    metrics_path = workspace / "stats" / "01_2_condition_metrics.csv"
    write_csv(metrics_path, metric_rows, fieldnames)
    repaired_values = [as_float(row["repaired_map50_95"]) for row in metric_rows]
    repaired_values = [value for value in repaired_values if value is not None]
    tail = repaired_values[-last_n:] if last_n > 0 else repaired_values
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "condition": spec.name,
        "mode": spec.mode,
        "warmup_map50": warm_m50,
        "warmup_map50_95": warm_m95,
        "final_repaired_map50_95": repaired_values[-1] if repaired_values else None,
        "last_n": last_n,
        "last_n_avg_repaired_map50_95": float(np.mean(tail)) if tail else None,
        "last_n_min_repaired_map50_95": float(np.min(tail)) if tail else None,
        "metrics_csv": str(metrics_path.resolve()),
    }
    (workspace / "stats" / "01_2_condition_metrics_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metric_rows


def apply_repair_only_reference(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repair_rows = [row for row in all_rows if row.get("condition") == "repair_only"]
    by_round = {str(row.get("round")): row for row in repair_rows}
    enriched = []
    for row in all_rows:
        copied = dict(row)
        ref = by_round.get(str(row.get("round")))
        rep = as_float(row.get("repaired_map50_95"))
        ref_rep = as_float(ref.get("repaired_map50_95")) if ref else None
        worst = as_float(row.get("worst_split_map50_95"))
        ref_worst = as_float(ref.get("worst_split_map50_95")) if ref else None
        night = as_float(row.get("night_avg_map50_95"))
        ref_night = as_float(ref.get("night_avg_map50_95")) if ref else None
        copied["delta_vs_repair_only_map50_95"] = "" if rep is None or ref_rep is None else f"{rep - ref_rep:.6f}"
        copied["worst_delta_vs_repair_only_map50_95"] = "" if worst is None or ref_worst is None else f"{worst - ref_worst:.6f}"
        copied["night_delta_vs_repair_only_map50_95"] = "" if night is None or ref_night is None else f"{night - ref_night:.6f}"
        enriched.append(copied)
    return enriched


def write_combined_metrics(root: Path, all_rows: list[dict[str, Any]]) -> None:
    all_rows = apply_repair_only_reference(all_rows)
    fieldnames = [
        "condition",
        "mode",
        "round",
        "aggregate_map50",
        "aggregate_map50_95",
        "aggregate_survival_vs_warmup_map50_95",
        "repaired_map50",
        "repaired_map50_95",
        "delta_vs_repair_only_map50_95",
        "repair_gain_map50_95",
        "retained_gain_map50_95",
        "round_delta_map50_95",
        "worst_split",
        "worst_split_map50_95",
        "worst_delta_vs_repair_only_map50_95",
        "day_avg_map50_95",
        "night_avg_map50_95",
        "night_delta_vs_repair_only_map50_95",
        "day_night_gap_map50_95",
    ]
    metrics_path = root / "stats" / "01_2_all_condition_metrics.csv"
    write_csv(metrics_path, all_rows, fieldnames)
    final_rows = {}
    for row in all_rows:
        final_rows[str(row["condition"])] = row
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_csv": str(metrics_path.resolve()),
        "final_by_condition": final_rows,
        "condition_specs": {name: asdict(spec) for name, spec in CONDITION_SPECS.items()},
    }
    (root / "stats" / "01_2_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {metrics_path}")


def prepare_condition(args: argparse.Namespace, spec: ConditionSpec):
    workspace = condition_workspace(args.workspace_root, spec.name)
    pl03.ensure_dirs(workspace)
    setup, fedsto = dqa01.configure_modules(workspace, args.client_limit)
    setup_payload = setup.build_base_configs()
    manifest = setup_payload.get("manifest") if isinstance(setup_payload, dict) else {}
    clients = pl02.resolve_clients(args.clients, setup)
    warmup = pl02.copy_warmup_to_workspace(args.warmup_checkpoint, workspace, args.force)
    return workspace, setup, fedsto, manifest, clients, warmup


def run_condition(args: argparse.Namespace, spec: ConditionSpec, condition_index: int) -> list[dict[str, Any]]:
    workspace = condition_workspace(args.workspace_root, spec.name)
    completed_metrics = workspace / "stats" / "01_2_condition_metrics.csv"
    completed_summary = workspace / "stats" / "01_2_condition_metrics_summary.json"
    if args.evaluate and not args.force and completed_metrics.exists() and completed_summary.exists():
        print(f"\n\n######## condition={spec.name} ########")
        print(f"Reusing completed condition metrics: {completed_metrics}")
        return read_csv(completed_metrics)

    workspace, setup, fedsto, manifest, clients, warmup = prepare_condition(args, spec)
    cargs = condition_args(args, workspace, spec)
    variant = pl03.Variant(spec.name, spec.train_scope, spec.aggregate_scope, 1, spec.client_lr0, 1, 1, spec.orthogonal_weight)

    print(f"\n\n######## condition={spec.name} mode={spec.mode} ########")
    print(f"Workspace: {workspace}")
    print(f"Warmup: {warmup}")
    print(json.dumps({"clients": clients, "server": manifest.get("server"), "spec": asdict(spec)}, indent=2, ensure_ascii=False))

    if args.setup_only:
        print(f"Setup complete for {spec.name}.")
        return []

    cargs.gpus = fedsto.resolve_gpus(cargs.gpus)
    if not cargs.dry_run:
        fedsto.check_runtime_dependencies()

    records: list[dict[str, str]] = [
        {
            "condition": spec.name,
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
        if spec.mode == "repair_only":
            round_records, current_global, port_offset = base01_0.run_repair_only_round(
                setup, fedsto, variant, current_global, cargs, idx, port_offset
            )
            pseudo_stats = None
            dqa_state = None
        else:
            round_records, current_global, pseudo_stats, dqa_state, port_offset = run_ssod_round(
                setup, fedsto, spec, current_global, cargs, clients, idx, port_offset
            )
        records.extend(normalize_records(round_records, spec.name))
        if pseudo_stats is not None:
            pseudo_history.append(pseudo_stats)
        if dqa_state is not None:
            dqa_history.append({"round": idx, "state": dqa_state})
        write_csv(
            workspace / "stats" / "01_2_checkpoints.csv",
            records,
            ["condition", "label", "kind", "round", "client", "variant", "path"],
        )

    manifest_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "condition": spec.name,
        "spec": asdict(spec),
        "workspace": str(workspace.resolve()),
        "root_workspace": str(args.workspace_root.resolve()),
        "rounds": args.rounds,
        "client_limit": args.client_limit,
        "warmup_source": str(args.warmup_checkpoint.expanduser().resolve()),
        "warmup_workspace": str(warmup),
        "pseudo_history": pseudo_history,
        "dqa_history": dqa_history,
        "checkpoints": records,
        "dqa_config": asdict(dqa01.dqa_config(cargs, len(setup.BDD_NAMES))) if spec.mode == "ssod_dqa" else None,
    }
    (workspace / "stats" / "01_2_manifest.json").write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.evaluate:
        base01_0.run_evaluation(cargs, records)
        return write_condition_metrics(spec, workspace, args.rounds, args.last_n)
    return []


def notify(args: argparse.Namespace, message: str, *, title: str, status: str | None = None, error: str | None = None) -> None:
    try:
        from notebook_notify import notify_discord

        context = {
            "workspace": str(args.workspace_root.resolve()),
            "conditions": args.conditions,
            "rounds": args.rounds,
            "client_limit": args.client_limit,
        }
        if status:
            context["status"] = status
        if error:
            context["error"] = error[:500]
        summary_path = args.workspace_root / "stats" / "01_2_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            context["summary"] = str(summary.get("final_by_condition", {}))[:900]
        result = notify_discord(message, title=title, context=context, fail_silently=True)
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
        all_rows.extend(run_condition(args, CONDITION_SPECS[condition], index))
    if all_rows:
        write_combined_metrics(args.workspace_root, all_rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT / "output" / "01_2_ssod_pivot")
    parser.add_argument("--warmup-checkpoint", type=Path, default=REPO_ROOT / "pseudogt_learnability" / "checkpoints" / "round000_warmup.pt")
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--client-limit", type=int, default=800)
    parser.add_argument("--clients", default="all")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--client-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=31041)
    parser.add_argument("--condition-port-stride", type=int, default=100)
    parser.add_argument("--device", default="0")
    parser.add_argument("--server-repair-epochs", type=int, default=1)
    parser.add_argument("--server-repair-lr", type=float, default=0.0008)
    parser.add_argument("--last-n", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--nms-iou-thres", type=float, default=0.65)
    parser.add_argument("--match-iou", type=float, default=0.60)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--min-stability", type=float, default=0.72)
    parser.add_argument("--min-score", type=float, default=0.28)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-boxes-per-image", type=int, default=12)
    parser.add_argument("--max-images-per-client", type=int, default=800)
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
        notify(args, "Scene-Daynight DQA 01_2 started.", title="Scene-Daynight DQA 01_2 start")

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
                f"Scene-Daynight DQA 01_2 finished with status={status}.",
                title="Scene-Daynight DQA 01_2 finish",
                status=status,
                error=error,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
