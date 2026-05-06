#!/usr/bin/env python3
"""Run scene-daynight DQA 01_1 diagnostic sweep.

01_1 is designed to make the next claim testable:

* repair-only tells us how much source repair alone explains.
* FedAvg target-heavy tells us whether pseudoGT client updates are destructive.
* DQA variants test whether constrained target adaptation can beat repair-only
  or at least preserve aggregate performance before repair.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
NAV_ROOT = REPO_ROOT / "navigating_data_heterogeneity"
DQA_ROOT = PROJECT_ROOT.parent
PSEUDOGT_SCRIPTS = REPO_ROOT / "pseudogt_learnability" / "scripts"
PROTOCOL_VERSION = "scene_daynight_dqa_01_1_diagnostic_sweep_v1"

for path in (NAV_ROOT, DQA_ROOT, PSEUDOGT_SCRIPTS, PROJECT_ROOT / "scripts", REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
    client_lr0: float = 0.0005
    source_repeat: int = 2
    pseudo_repeat: int = 1
    orthogonal_weight: float = 1e-4
    loss_box: float | None = None
    dqa_min_server_alpha: float | None = None
    dqa_server_anchor: float | None = None
    dqa_residual_blend: float | None = None
    dqa_classwise_blend: float | None = None

    def variant(self) -> pl03.Variant:
        return pl03.Variant(
            name=self.name,
            train_scope=self.train_scope,
            aggregate_scope=self.aggregate_scope,
            client_epochs=1,
            client_lr0=self.client_lr0,
            source_repeat=self.source_repeat,
            pseudo_repeat=self.pseudo_repeat,
            orthogonal_weight=self.orthogonal_weight,
            note=self.note,
        )


CONDITION_SPECS: dict[str, ConditionSpec] = {
    "repair_only": ConditionSpec(
        name="repair_only",
        mode="repair_only",
        note="Control: warmup plus repeated supervised source repair only.",
    ),
    "dqa_current": ConditionSpec(
        name="dqa_current",
        mode="dqa",
        note="01_0-style DQA: source-heavy full-model client update.",
    ),
    "dqa_source_light": ConditionSpec(
        name="dqa_source_light",
        mode="dqa",
        note="Ablation: reduce source dominance so client domain signal can move the model.",
        source_repeat=1,
        pseudo_repeat=1,
        client_lr0=0.00045,
    ),
    "dqa_target_double": ConditionSpec(
        name="dqa_target_double",
        mode="dqa",
        note="Target-heavy DQA: target pseudoGT appears twice, source once.",
        source_repeat=1,
        pseudo_repeat=2,
        client_lr0=0.00035,
        loss_box=0.03,
        dqa_min_server_alpha=0.76,
        dqa_server_anchor=14.0,
        dqa_residual_blend=0.10,
        dqa_classwise_blend=0.12,
    ),
    "dqa_head_lowbox": ConditionSpec(
        name="dqa_head_lowbox",
        mode="dqa",
        note=(
            "Head-only target adaptation with weak bbox loss. Tests whether pseudoGT "
            "is useful as class/objectness/domain signal without trusting box regression."
        ),
        train_scope="head",
        aggregate_scope="all",
        source_repeat=1,
        pseudo_repeat=2,
        client_lr0=0.0008,
        loss_box=0.005,
        dqa_min_server_alpha=0.72,
        dqa_server_anchor=12.0,
        dqa_residual_blend=0.12,
        dqa_classwise_blend=0.14,
    ),
    "dqa_nonbackbone_lowbox": ConditionSpec(
        name="dqa_nonbackbone_lowbox",
        mode="dqa",
        note="Neck/head adaptation with weak bbox loss. Tests target-specific non-backbone adaptation.",
        train_scope="non_backbone",
        aggregate_scope="all",
        source_repeat=1,
        pseudo_repeat=2,
        client_lr0=0.00055,
        loss_box=0.01,
        dqa_min_server_alpha=0.74,
        dqa_server_anchor=13.0,
        dqa_residual_blend=0.10,
        dqa_classwise_blend=0.12,
    ),
    "fedavg_target_double": ConditionSpec(
        name="fedavg_target_double",
        mode="fedavg",
        note="FedAvg stress test with the same target-heavy client recipe as dqa_target_double.",
        source_repeat=1,
        pseudo_repeat=2,
        client_lr0=0.00035,
        loss_box=0.03,
    ),
}

DEFAULT_CONDITIONS = (
    "repair_only",
    "dqa_current",
    "dqa_source_light",
    "dqa_target_double",
    "dqa_head_lowbox",
    "dqa_nonbackbone_lowbox",
    "fedavg_target_double",
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


@contextmanager
def patched_client_config(loss_box: float | None) -> Iterator[None]:
    original = pl03.write_client_config
    if loss_box is None:
        yield
        return

    def wrapped(setup, variant, client, start, args, round_idx):  # noqa: ANN001
        path = original(setup, variant, client, start, args, round_idx)
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg.setdefault("Loss", {})
        cfg["Loss"]["box"] = float(loss_box)
        cfg["Loss"]["pseudo_box_downweight_note"] = (
            "Set by scene_daynight_dqa 01_1 to reduce pseudoGT localization pressure."
        )
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return path

    pl03.write_client_config = wrapped
    try:
        yield
    finally:
        pl03.write_client_config = original


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


def aggregate_label(spec: ConditionSpec, tag: str) -> str | None:
    if spec.mode == "repair_only":
        return None
    if spec.mode == "fedavg":
        return f"{tag}_aggregate_{spec.aggregate_scope}"
    if spec.mode == "dqa":
        return f"{tag}_dqa_aggregate"
    raise ValueError(spec.mode)


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
        aggregate_survival = "" if agg_m95 is None or warm_m95 is None else f"{agg_m95 - warm_m95:.6f}"

        metric_rows.append(
            {
                "condition": spec.name,
                "mode": spec.mode,
                "round": idx,
                "aggregate_map50": "" if agg_m50 is None else f"{agg_m50:.6f}",
                "aggregate_map50_95": "" if agg_m95 is None else f"{agg_m95:.6f}",
                "aggregate_survival_vs_warmup_map50_95": aggregate_survival,
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
    metrics_path = workspace / "stats" / "01_1_condition_metrics.csv"
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
    (workspace / "stats" / "01_1_condition_metrics_summary.json").write_text(
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
    metrics_path = root / "stats" / "01_1_all_condition_metrics.csv"
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
    (root / "stats" / "01_1_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved: {metrics_path}")


def prepare_condition(args: argparse.Namespace, spec: ConditionSpec):
    workspace = condition_workspace(args.workspace_root, spec.name)
    pl03.ensure_dirs(workspace)
    setup, fedsto = dqa01.configure_modules(workspace, args.client_limit)
    setup_payload = setup.build_base_configs()
    manifest = setup_payload.get("manifest") if isinstance(setup_payload, dict) else {}
    clients = pl02.resolve_clients(args.clients, setup)
    warmup = pl02.copy_warmup_to_workspace(args.warmup_checkpoint, workspace, args.force)
    return workspace, setup, fedsto, manifest, clients, spec.variant(), warmup


def run_condition(args: argparse.Namespace, spec: ConditionSpec, condition_index: int) -> list[dict[str, Any]]:
    workspace = condition_workspace(args.workspace_root, spec.name)
    completed_metrics = workspace / "stats" / "01_1_condition_metrics.csv"
    completed_summary = workspace / "stats" / "01_1_condition_metrics_summary.json"
    if args.evaluate and not args.force and completed_metrics.exists() and completed_summary.exists():
        print(f"\n\n######## condition={spec.name} ########")
        print(f"Reusing completed condition metrics: {completed_metrics}")
        return read_csv(completed_metrics)

    workspace, setup, fedsto, manifest, clients, variant, warmup = prepare_condition(args, spec)
    cargs = condition_args(args, workspace, spec)

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

    with patched_client_config(spec.loss_box):
        for idx in range(1, args.rounds + 1):
            if spec.mode == "repair_only":
                round_records, current_global, port_offset = base01_0.run_repair_only_round(
                    setup,
                    fedsto,
                    variant,
                    current_global,
                    cargs,
                    idx,
                    port_offset,
                )
            elif spec.mode == "fedavg":
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
            elif spec.mode == "dqa":
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
                raise ValueError(spec.mode)

            records.extend(normalize_records(round_records, spec.name))
            write_csv(
                workspace / "stats" / "01_1_checkpoints.csv",
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
        "dqa_config": asdict(dqa01.dqa_config(cargs, len(setup.BDD_NAMES))) if spec.mode == "dqa" else None,
    }
    (workspace / "stats" / "01_1_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

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
            "max_images_per_client": args.max_images_per_client,
        }
        if status:
            context["status"] = status
        if error:
            context["error"] = error[:500]
        summary_path = args.workspace_root / "stats" / "01_1_summary.json"
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
        all_rows.extend(run_condition(args, CONDITION_SPECS[condition], index))
    if all_rows:
        write_combined_metrics(args.workspace_root, all_rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT / "output" / "01_1_dqa_diagnostic_sweep")
    parser.add_argument("--warmup-checkpoint", type=Path, default=REPO_ROOT / "pseudogt_learnability" / "checkpoints" / "round000_warmup.pt")
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--client-limit", type=int, default=1500)
    parser.add_argument("--clients", default="all")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=30941)
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
        notify(args, "Scene-Daynight DQA 01_1 started.", title="Scene-Daynight DQA 01_1 start")

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
                f"Scene-Daynight DQA 01_1 finished with status={status}.",
                title="Scene-Daynight DQA 01_1 finish",
                status=status,
                error=error,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
