#!/usr/bin/env python3
"""Run scene-daynight DQA 02 head-to-full long schedule.

This runner tests the current main hypothesis:

* Phase 1 is long and conservative: head/neck-only client updates create stable
  client/class/domain differences for DQA to aggregate.
* Phase 2 is short and stronger: a low-LR full-model burst injects the learned
  target-domain signal into the whole detector without giving pseudoGT enough
  time to dominate the model.
* Evaluation is final-focused by default to keep a 30+2 round pilot practical.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
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
PROTOCOL_VERSION = "scene_daynight_dqa_02_head_to_full_long_v1"

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
class PhaseSpec:
    name: str
    display_name: str
    note: str
    train_scope: str
    aggregate_scope: str = "all"
    client_lr0: float = 0.0005
    source_repeat: int = 1
    pseudo_repeat: int = 1
    orthogonal_weight: float = 1e-4
    loss_box: float | None = None
    dqa_min_server_alpha: float = 0.76
    dqa_server_anchor: float = 14.0
    dqa_residual_blend: float = 0.10
    dqa_classwise_blend: float = 0.12

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


def default_phase1_spec(args: argparse.Namespace) -> PhaseSpec:
    return PhaseSpec(
        name="phase1_head",
        display_name="Phase 1 head-only DQA",
        note=(
            "Long head/neck-only source-anchored pseudoGT adaptation. "
            "The goal is stable client/class/domain differentiation, not large "
            "bbox movement."
        ),
        train_scope="neck_head",
        client_lr0=args.phase1_client_lr,
        source_repeat=args.phase1_source_repeat,
        pseudo_repeat=args.phase1_pseudo_repeat,
        loss_box=args.phase1_loss_box,
        dqa_min_server_alpha=args.phase1_dqa_min_server_alpha,
        dqa_server_anchor=args.phase1_dqa_server_anchor,
        dqa_residual_blend=args.phase1_dqa_residual_blend,
        dqa_classwise_blend=args.phase1_dqa_classwise_blend,
    )


def default_phase2_spec(args: argparse.Namespace) -> PhaseSpec:
    return PhaseSpec(
        name="phase2_full",
        display_name="Phase 2 full-model burst DQA",
        note=(
            "Short full-model low-LR target burst. The goal is to let the "
            "Phase-1 target signal touch backbone/neck/head while keeping the "
            "pseudoGT exposure short."
        ),
        train_scope="all",
        client_lr0=args.phase2_client_lr,
        source_repeat=args.phase2_source_repeat,
        pseudo_repeat=args.phase2_pseudo_repeat,
        loss_box=args.phase2_loss_box,
        dqa_min_server_alpha=args.phase2_dqa_min_server_alpha,
        dqa_server_anchor=args.phase2_dqa_server_anchor,
        dqa_residual_blend=args.phase2_dqa_residual_blend,
        dqa_classwise_blend=args.phase2_dqa_classwise_blend,
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


def seconds_to_hms(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return ""
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


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
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return path

    pl03.write_client_config = wrapped
    try:
        yield
    finally:
        pl03.write_client_config = original


def phase_args(args: argparse.Namespace, workspace: Path, spec: PhaseSpec) -> argparse.Namespace:
    copied = copy.copy(args)
    copied.workspace_root = workspace
    copied.dqa_min_server_alpha = spec.dqa_min_server_alpha
    copied.dqa_server_anchor = spec.dqa_server_anchor
    copied.dqa_residual_blend = spec.dqa_residual_blend
    copied.dqa_classwise_blend = spec.dqa_classwise_blend
    return copied


def save_checkpoint_record(
    records: list[dict[str, str]],
    label: str,
    path: Path,
    kind: str,
    *,
    phase: str,
    phase_round: int | str = "",
    global_round: int | str = "",
    client: str = "",
    variant: str = "",
) -> None:
    records.append(
        {
            "label": label,
            "kind": kind,
            "phase": str(phase),
            "phase_round": str(phase_round),
            "global_round": str(global_round),
            "client": client,
            "variant": variant,
            "path": str(path.resolve()),
        }
    )


def run_dqa_round(
    setup,
    fedsto,
    spec: PhaseSpec,
    current_global: Path,
    args: argparse.Namespace,
    clients: list[dict[str, Any]],
    *,
    phase_round: int,
    global_round: int,
    port_offset: int,
) -> tuple[list[dict[str, str]], Path, dict[str, Any], dict[str, Any], int]:
    tag = round_tag(global_round)
    print(f"\n=== {spec.display_name}: {tag} phase_round={phase_round} ===")
    pseudo_stats = pl03.generate_round_pseudo_labels(setup, current_global, args, clients, global_round)

    records: list[dict[str, str]] = []
    local_paths: list[Path] = []
    variant = spec.variant()

    with patched_client_config(spec.loss_box):
        for client in clients:
            client_tag = f"client{client['id']}_{client['weather']}"
            start = fedsto.CLIENT_STATE_DIR / f"02_{tag}_{spec.name}_{client_tag}_start.pt"
            run_name = f"sdn02htf_{tag}_{spec.name}_{client_tag}"
            raw_ckpt = fedsto.checkpoint_path(run_name)
            final_ckpt = args.workspace_root / "checkpoints" / f"{tag}_{spec.name}_{client_tag}.pt"

            if not args.dry_run and not fedsto.checkpoint_matches_protocol(start, PROTOCOL_VERSION):
                fedsto.make_start_checkpoint(
                    current_global,
                    start,
                    protocol=PROTOCOL_VERSION,
                    stage=f"{tag}_{spec.name}_{client_tag}_start",
                )

            if not pl03.reusable_checkpoint(fedsto, final_ckpt, args.force):
                cfg = pl03.write_client_config(setup, variant, client, start, args, global_round)
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
                    fedsto.make_start_checkpoint(
                        raw_ckpt,
                        final_ckpt,
                        protocol=PROTOCOL_VERSION,
                        stage=f"{tag}_{spec.name}_{client_tag}",
                    )
                    pl03.cleanup_training_artifacts(raw_ckpt, start)

            local_paths.append(final_ckpt)
            save_checkpoint_record(
                records,
                f"{spec.name}_{tag}_{client_tag}",
                final_ckpt,
                "client",
                phase=spec.name,
                phase_round=phase_round,
                global_round=global_round,
                client=client_tag,
                variant=spec.name,
            )

    stats = dqa01.pseudo_stats_to_dqa_stats(pseudo_stats, num_classes=len(setup.BDD_NAMES))
    aggregate = args.workspace_root / "checkpoints" / f"{tag}_{spec.name}_dqa_aggregate.pt"
    state_path = args.workspace_root / "stats" / "02_head_to_full_dqa_state.json"
    config = dqa01.dqa_config(args, len(setup.BDD_NAMES))
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
        fedsto.mark_checkpoint_protocol(aggregate, PROTOCOL_VERSION, f"{tag}_{spec.name}_dqa_aggregate")
    elif state_path.exists():
        dqa_state = json.loads(state_path.read_text(encoding="utf-8"))

    save_checkpoint_record(
        records,
        f"{spec.name}_{tag}_dqa_aggregate",
        aggregate,
        "aggregate",
        phase=spec.name,
        phase_round=phase_round,
        global_round=global_round,
        variant=spec.name,
    )

    repair_start = fedsto.GLOBAL_DIR / f"02_{tag}_{spec.name}_server_repair_start.pt"
    repair = args.workspace_root / "checkpoints" / f"{tag}_{spec.name}_server_repair.pt"
    if args.server_repair_epochs > 0:
        if not args.dry_run and not pl03.reusable_checkpoint(fedsto, repair, args.force):
            fedsto.make_start_checkpoint(
                aggregate,
                repair_start,
                protocol=PROTOCOL_VERSION,
                stage=f"{tag}_{spec.name}_server_repair_start",
            )
            cfg = pl03.write_server_repair_config(setup, variant, repair_start, args, global_round)
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
                fedsto.make_start_checkpoint(
                    raw_repair,
                    repair,
                    protocol=PROTOCOL_VERSION,
                    stage=f"{tag}_{spec.name}_server_repair",
                )
                pl03.cleanup_training_artifacts(raw_repair, repair_start)
        save_checkpoint_record(
            records,
            f"{spec.name}_{tag}_server_repair",
            repair,
            "server_repair",
            phase=spec.name,
            phase_round=phase_round,
            global_round=global_round,
            variant=spec.name,
        )
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


def repair_reference(args: argparse.Namespace) -> dict[str, str] | None:
    path = args.repair_reference_csv
    if not path.exists():
        return None
    rows = read_csv(path)
    repair_rows = [row for row in rows if row.get("condition") == "repair_only"]
    if not repair_rows:
        return None
    repair_rows.sort(key=lambda row: int(row.get("round") or 0))
    return repair_rows[-1]


def selected_eval_records(
    records: list[dict[str, str]],
    *,
    phase1_spec: PhaseSpec,
    phase2_spec: PhaseSpec,
    phase1_rounds: int,
    phase2_rounds: int,
    eval_all_rounds: bool,
) -> list[dict[str, str]]:
    if eval_all_rounds:
        return records

    by_label = {row["label"]: row for row in records}
    phase1_tag = round_tag(phase1_rounds)
    phase2_global_round = phase1_rounds + phase2_rounds
    phase2_tag = round_tag(phase2_global_round)
    wanted = [
        ("warmup_global", "warmup_global"),
        (f"{phase1_spec.name}_{phase1_tag}_dqa_aggregate", "phase1_final_aggregate"),
        (f"{phase1_spec.name}_{phase1_tag}_server_repair", "phase1_final_repair"),
        (f"{phase2_spec.name}_{phase2_tag}_dqa_aggregate", "phase2_final_aggregate"),
        (f"{phase2_spec.name}_{phase2_tag}_server_repair", "phase2_final_repair"),
    ]
    selected: list[dict[str, str]] = []
    for source_label, eval_label in wanted:
        row = by_label.get(source_label)
        if row is None:
            continue
        copied = dict(row)
        copied["label"] = eval_label
        selected.append(copied)
    return selected


def write_final_metrics(args: argparse.Namespace, eval_records: list[dict[str, str]]) -> list[dict[str, Any]]:
    summary_path = args.workspace_root / "validation_reports" / "paper_protocol_eval_summary.csv"
    rows = [row for row in read_csv(summary_path) if row.get("status") == "ok"]
    by_label_split = {(row["checkpoint_label"], row["split"]): row for row in rows}
    total_rows = [row for row in rows if row.get("split") in {"scene_daynight_total", "total"}]
    by_label_total = {row["checkpoint_label"]: row for row in total_rows}
    warm = by_label_total.get("warmup_global")
    warm_m95 = as_float(warm.get("map50_95")) if warm else None
    ref = repair_reference(args)
    ref_m95 = as_float(ref.get("repaired_map50_95")) if ref else None
    ref_worst = as_float(ref.get("worst_split_map50_95")) if ref else None
    ref_night = as_float(ref.get("night_avg_map50_95")) if ref else None

    meta = {row["label"]: row for row in eval_records}
    metric_rows: list[dict[str, Any]] = []
    for label in [row["label"] for row in eval_records]:
        total = by_label_total.get(label)
        if not total:
            continue
        m95 = as_float(total.get("map50_95"))
        gap = split_gap_metrics(by_label_split, label)
        worst = as_float(gap.get("worst_split_map50_95"))
        night = as_float(gap.get("night_avg_map50_95"))
        metric_rows.append(
            {
                "checkpoint_label": label,
                "kind": meta.get(label, {}).get("kind", ""),
                "phase": meta.get(label, {}).get("phase", ""),
                "phase_round": meta.get(label, {}).get("phase_round", ""),
                "global_round": meta.get(label, {}).get("global_round", ""),
                "precision": total.get("precision", ""),
                "recall": total.get("recall", ""),
                "map50": total.get("map50", ""),
                "map50_95": total.get("map50_95", ""),
                "gain_vs_warmup_map50_95": "" if m95 is None or warm_m95 is None else f"{m95 - warm_m95:.6f}",
                "delta_vs_repair_only_r3_map50_95": "" if m95 is None or ref_m95 is None else f"{m95 - ref_m95:.6f}",
                "worst_delta_vs_repair_only_r3_map50_95": "" if worst is None or ref_worst is None else f"{worst - ref_worst:.6f}",
                "night_delta_vs_repair_only_r3_map50_95": "" if night is None or ref_night is None else f"{night - ref_night:.6f}",
                **gap,
            }
        )

    fieldnames = [
        "checkpoint_label",
        "kind",
        "phase",
        "phase_round",
        "global_round",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "gain_vs_warmup_map50_95",
        "delta_vs_repair_only_r3_map50_95",
        "worst_split",
        "worst_split_map50_95",
        "worst_delta_vs_repair_only_r3_map50_95",
        "day_avg_map50_95",
        "night_avg_map50_95",
        "night_delta_vs_repair_only_r3_map50_95",
        "day_night_gap_map50_95",
    ]
    metrics_path = args.workspace_root / "stats" / "02_head_to_full_final_metrics.csv"
    write_csv(metrics_path, metric_rows, fieldnames)

    split_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["checkpoint_label"] not in meta:
            continue
        split_rows.append(
            {
                "checkpoint_label": row["checkpoint_label"],
                "kind": meta[row["checkpoint_label"]].get("kind", ""),
                "phase": meta[row["checkpoint_label"]].get("phase", ""),
                "split": row["split"],
                "images": row.get("images", ""),
                "labels": row.get("labels", ""),
                "precision": row.get("precision", ""),
                "recall": row.get("recall", ""),
                "map50": row.get("map50", ""),
                "map50_95": row.get("map50_95", ""),
            }
        )
    write_csv(
        args.workspace_root / "stats" / "02_head_to_full_split_metrics.csv",
        split_rows,
        [
            "checkpoint_label",
            "kind",
            "phase",
            "split",
            "images",
            "labels",
            "precision",
            "recall",
            "map50",
            "map50_95",
        ],
    )

    print(f"Saved: {metrics_path}")
    return metric_rows


def write_checkpoint_records(path: Path, records: list[dict[str, str]]) -> None:
    write_csv(
        path,
        records,
        ["label", "kind", "phase", "phase_round", "global_round", "client", "variant", "path"],
    )


def update_progress(
    args: argparse.Namespace,
    progress_rows: list[dict[str, Any]],
    *,
    phase: str,
    phase_round: int,
    global_round: int,
    elapsed: float,
    completed: int,
    total_rounds: int,
    checkpoint: Path,
) -> None:
    avg = elapsed / completed if completed else 0.0
    eta = avg * (total_rounds - completed)
    progress_rows.append(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "phase_round": phase_round,
            "global_round": global_round,
            "completed_rounds": completed,
            "total_rounds": total_rounds,
            "elapsed_seconds": f"{elapsed:.3f}",
            "avg_seconds_per_round": f"{avg:.3f}",
            "eta_seconds": f"{eta:.3f}",
            "elapsed_hms": seconds_to_hms(elapsed),
            "eta_hms": seconds_to_hms(eta),
            "checkpoint": str(checkpoint.resolve()),
        }
    )
    write_csv(
        args.workspace_root / "stats" / "02_head_to_full_progress.csv",
        progress_rows,
        [
            "created_utc",
            "phase",
            "phase_round",
            "global_round",
            "completed_rounds",
            "total_rounds",
            "elapsed_seconds",
            "avg_seconds_per_round",
            "eta_seconds",
            "elapsed_hms",
            "eta_hms",
            "checkpoint",
        ],
    )


def tqdm_factory(args: argparse.Namespace, total: int):
    if args.no_progress:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:  # noqa: BLE001
        return None
    return tqdm(total=total, desc="02 head-to-full DQA", unit="round")


def prepare(args: argparse.Namespace):
    args.workspace_root = args.workspace_root.expanduser().resolve()
    pl03.ensure_dirs(args.workspace_root)
    setup, fedsto = dqa01.configure_modules(args.workspace_root, args.client_limit)
    setup_payload = setup.build_base_configs()
    manifest = setup_payload.get("manifest") if isinstance(setup_payload, dict) else {}
    clients = pl02.resolve_clients(args.clients, setup)
    warmup = pl02.copy_warmup_to_workspace(args.warmup_checkpoint, args.workspace_root, args.force)
    return setup, fedsto, manifest, clients, warmup


def estimated_seconds(args: argparse.Namespace) -> float:
    return (
        args.phase1_rounds * args.estimated_phase1_round_minutes
        + args.phase2_rounds * args.estimated_phase2_round_minutes
        + (args.estimated_eval_minutes if args.evaluate else 0.0)
    ) * 60.0


def run(args: argparse.Namespace) -> None:
    setup, fedsto, manifest, clients, warmup = prepare(args)
    phase1 = default_phase1_spec(args)
    phase2 = default_phase2_spec(args)
    total_rounds = args.phase1_rounds + args.phase2_rounds
    est = estimated_seconds(args)

    print(
        json.dumps(
            {
                "protocol": PROTOCOL_VERSION,
                "workspace": str(args.workspace_root.resolve()),
                "clients": clients,
                "server": manifest.get("server"),
                "phase1": asdict(phase1),
                "phase2": asdict(phase2),
                "total_rounds": total_rounds,
                "estimated_runtime": seconds_to_hms(est),
                "final_focused_evaluation": args.evaluate and not args.eval_all_rounds,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    if args.setup_only:
        print("Setup complete.")
        return

    args.gpus = fedsto.resolve_gpus(args.gpus)
    if not args.dry_run:
        fedsto.check_runtime_dependencies()

    records: list[dict[str, str]] = []
    save_checkpoint_record(records, "warmup_global", warmup, "warmup", phase="warmup")
    current_global = warmup
    pseudo_history: list[dict[str, Any]] = []
    dqa_history: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    port_offset = 0
    start_time = time.monotonic()
    progress = tqdm_factory(args, total_rounds)

    def finish_round(spec: PhaseSpec, phase_round: int, global_round: int, checkpoint: Path) -> None:
        completed = global_round
        elapsed = time.monotonic() - start_time
        update_progress(
            args,
            progress_rows,
            phase=spec.name,
            phase_round=phase_round,
            global_round=global_round,
            elapsed=elapsed,
            completed=completed,
            total_rounds=total_rounds,
            checkpoint=checkpoint,
        )
        if progress is not None:
            eta = as_float(progress_rows[-1]["eta_seconds"])
            progress.set_postfix(
                phase=spec.name,
                round=f"{phase_round}",
                elapsed=seconds_to_hms(elapsed),
                eta=seconds_to_hms(eta),
            )
            progress.update(1)

    for phase_round in range(1, args.phase1_rounds + 1):
        global_round = phase_round
        cargs = phase_args(args, args.workspace_root, phase1)
        round_records, current_global, pseudo_stats, dqa_state, port_offset = run_dqa_round(
            setup,
            fedsto,
            phase1,
            current_global,
            cargs,
            clients,
            phase_round=phase_round,
            global_round=global_round,
            port_offset=port_offset,
        )
        records.extend(round_records)
        pseudo_history.append({"phase": phase1.name, "phase_round": phase_round, "global_round": global_round, "stats": pseudo_stats})
        dqa_history.append({"phase": phase1.name, "phase_round": phase_round, "global_round": global_round, "state": dqa_state})
        write_checkpoint_records(args.workspace_root / "stats" / "02_head_to_full_checkpoints.csv", records)
        finish_round(phase1, phase_round, global_round, current_global)

    for phase_round in range(1, args.phase2_rounds + 1):
        global_round = args.phase1_rounds + phase_round
        cargs = phase_args(args, args.workspace_root, phase2)
        round_records, current_global, pseudo_stats, dqa_state, port_offset = run_dqa_round(
            setup,
            fedsto,
            phase2,
            current_global,
            cargs,
            clients,
            phase_round=phase_round,
            global_round=global_round,
            port_offset=port_offset,
        )
        records.extend(round_records)
        pseudo_history.append({"phase": phase2.name, "phase_round": phase_round, "global_round": global_round, "stats": pseudo_stats})
        dqa_history.append({"phase": phase2.name, "phase_round": phase_round, "global_round": global_round, "state": dqa_state})
        write_checkpoint_records(args.workspace_root / "stats" / "02_head_to_full_checkpoints.csv", records)
        finish_round(phase2, phase_round, global_round, current_global)

    if progress is not None:
        progress.close()

    manifest_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "workspace": str(args.workspace_root.resolve()),
        "phase1_rounds": args.phase1_rounds,
        "phase2_rounds": args.phase2_rounds,
        "client_limit": args.client_limit,
        "warmup_source": str(args.warmup_checkpoint.expanduser().resolve()),
        "warmup_workspace": str(warmup.resolve()),
        "phase1": asdict(phase1),
        "phase2": asdict(phase2),
        "pseudo_history": pseudo_history,
        "dqa_history": dqa_history,
        "checkpoints": records,
        "estimated_runtime_seconds": est,
        "actual_runtime_seconds": time.monotonic() - start_time,
        "final_checkpoint": str(current_global.resolve()),
        "final_focused_evaluation": args.evaluate and not args.eval_all_rounds,
        "dqa_config_phase1": asdict(dqa01.dqa_config(phase_args(args, args.workspace_root, phase1), len(setup.BDD_NAMES))),
        "dqa_config_phase2": asdict(dqa01.dqa_config(phase_args(args, args.workspace_root, phase2), len(setup.BDD_NAMES))),
    }
    (args.workspace_root / "stats" / "02_head_to_full_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.evaluate:
        eval_records = selected_eval_records(
            records,
            phase1_spec=phase1,
            phase2_spec=phase2,
            phase1_rounds=args.phase1_rounds,
            phase2_rounds=args.phase2_rounds,
            eval_all_rounds=args.eval_all_rounds,
        )
        write_checkpoint_records(args.workspace_root / "stats" / "02_head_to_full_eval_checkpoints.csv", eval_records)
        base01_0.run_evaluation(args, eval_records)
        write_final_metrics(args, eval_records)


def notify(args: argparse.Namespace, message: str, *, title: str, status: str | None = None, error: str | None = None) -> None:
    try:
        from notebook_notify import notify_discord

        context = {
            "workspace": str(args.workspace_root.expanduser().resolve()),
            "phase1_rounds": args.phase1_rounds,
            "phase2_rounds": args.phase2_rounds,
            "estimated_runtime": seconds_to_hms(estimated_seconds(args)),
        }
        if status:
            context["status"] = status
        if error:
            context["error"] = error[:500]
        metrics_path = args.workspace_root.expanduser().resolve() / "stats" / "02_head_to_full_final_metrics.csv"
        if metrics_path.exists():
            context["final_metrics_csv"] = str(metrics_path)
        result = notify_discord(message, title=title, context=context, fail_silently=True)
        print(result)
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT / "output" / "02_head_to_full_long_dqa")
    parser.add_argument("--warmup-checkpoint", type=Path, default=REPO_ROOT / "pseudogt_learnability" / "checkpoints" / "round000_warmup.pt")
    parser.add_argument("--client-limit", type=int, default=1500)
    parser.add_argument("--clients", default="all")
    parser.add_argument("--phase1-rounds", type=int, default=30)
    parser.add_argument("--phase2-rounds", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=31141)
    parser.add_argument("--device", default="")
    parser.add_argument("--server-repair-epochs", type=int, default=1)
    parser.add_argument("--server-repair-lr", type=float, default=0.0008)
    parser.add_argument("--phase1-client-lr", type=float, default=0.0008)
    parser.add_argument("--phase1-source-repeat", type=int, default=1)
    parser.add_argument("--phase1-pseudo-repeat", type=int, default=2)
    parser.add_argument("--phase1-loss-box", type=float, default=0.005)
    parser.add_argument("--phase1-dqa-min-server-alpha", type=float, default=0.70)
    parser.add_argument("--phase1-dqa-server-anchor", type=float, default=10.0)
    parser.add_argument("--phase1-dqa-residual-blend", type=float, default=0.14)
    parser.add_argument("--phase1-dqa-classwise-blend", type=float, default=0.16)
    parser.add_argument("--phase2-client-lr", type=float, default=0.0003)
    parser.add_argument("--phase2-source-repeat", type=int, default=1)
    parser.add_argument("--phase2-pseudo-repeat", type=int, default=1)
    parser.add_argument("--phase2-loss-box", type=float, default=0.01)
    parser.add_argument("--phase2-dqa-min-server-alpha", type=float, default=0.76)
    parser.add_argument("--phase2-dqa-server-anchor", type=float, default=14.0)
    parser.add_argument("--phase2-dqa-residual-blend", type=float, default=0.10)
    parser.add_argument("--phase2-dqa-classwise-blend", type=float, default=0.12)
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
    parser.add_argument("--eval-all-rounds", action="store_true")
    parser.add_argument("--eval-clients", action="store_true")
    parser.add_argument(
        "--eval-splits",
        default="highway_day,highway_night,citystreet_day,citystreet_night,residential_day,residential_night,total",
    )
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--classwise", action="store_true")
    parser.add_argument("--no-eval-plots", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--estimated-phase1-round-minutes", type=float, default=19.0)
    parser.add_argument("--estimated-phase2-round-minutes", type=float, default=23.0)
    parser.add_argument("--estimated-eval-minutes", type=float, default=60.0)
    parser.add_argument(
        "--repair-reference-csv",
        type=Path,
        default=PROJECT_ROOT / "output" / "01_0_repair_baseline_comparison" / "stats" / "01_0_all_condition_metrics.csv",
    )
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--notify-start", action="store_true")
    parser.add_argument("--notify-end", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    do_start_notify = args.notify or args.notify_start
    do_end_notify = args.notify or args.notify_end
    if do_start_notify:
        notify(args, "Scene-Daynight DQA 02 head-to-full started.", title="DQA 02 head-to-full start")

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
                f"Scene-Daynight DQA 02 head-to-full finished with status={status}.",
                title="DQA 02 head-to-full finish",
                status=status,
                error=error,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
