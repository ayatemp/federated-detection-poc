#!/usr/bin/env python3
"""Run the main scene-daynight comparison experiment.

This runner turns the current best DQA hypothesis into a paper-style comparison:

* warmup only
* warmup + source/server repair
* warmup + BN-inclusive residual DQA + source/server repair

The DQA branch is based on the strongest signal from the MoE/DQA loops: client
experts contain useful scene/day specialization, but the old global aggregator
erases it.  The productionized policy therefore applies the average day-client
residual to the server checkpoint, restricted to neck/head by default and
including BatchNorm tensors inside that scope.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
NAV_ROOT = REPO_ROOT / "navigating_data_heterogeneity"
DQA_ROOT = PROJECT_ROOT.parent
PSEUDOGT_SCRIPTS = REPO_ROOT / "pseudogt_learnability" / "scripts"
PROTOCOL_VERSION = "scene_daynight_dqa_03_main_bn_residual_v1"

for path in (NAV_ROOT, DQA_ROOT, PSEUDOGT_SCRIPTS, PROJECT_ROOT / "scripts", REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dqa_cwa_aggregation as dqa_v1  # noqa: E402
import run_pseudogt_learnability_02 as pl02  # noqa: E402
import run_pseudogt_learnability_03 as pl03  # noqa: E402
import run_scene_daynight_dqa_01 as dqa01  # noqa: E402
import run_scene_daynight_dqa_01_0 as base01_0  # noqa: E402
import run_scene_daynight_dqa_02_head_to_full as htf02  # noqa: E402


SPLIT_NAMES = base01_0.SPLIT_NAMES


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


def is_neck_or_head(key: str) -> bool:
    return key.startswith("neck.") or key.startswith("head.")


def key_filter_from_scope(scope: str) -> Callable[[str], bool] | None:
    if scope == "all":
        return None
    if scope == "neck_head":
        return is_neck_or_head
    raise ValueError(f"Unsupported DQA residual scope: {scope!r}")


def _load(path: Path) -> dict[str, Any]:
    return dqa_v1._load_checkpoint(path, REPO_ROOT)


def _state_dict(ckpt: Mapping[str, Any], key: str) -> dict[str, torch.Tensor] | None:
    if ckpt.get(key) is None:
        return None
    return dqa_v1._model_state_dict(ckpt, key)


def _replace(base: dict[str, Any], state: Mapping[str, torch.Tensor], key: str) -> None:
    dqa_v1._replace_model_state(base, dict(state), key)


def save_checkpoint(base: dict[str, Any], output: Path) -> Path:
    base["epoch"] = -1
    base["optimizer"] = None
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, output)
    return output


def weighted_residual_state(
    base_state: Mapping[str, torch.Tensor],
    source_states: Sequence[Mapping[str, torch.Tensor]],
    anchor_state: Mapping[str, torch.Tensor],
    *,
    weights: Sequence[float],
    beta: float,
    key_filter: Callable[[str], bool] | None,
    include_bn: bool,
) -> dict[str, torch.Tensor]:
    total = float(sum(weights)) or 1.0
    norm = [float(weight) / total for weight in weights]
    result: dict[str, torch.Tensor] = {}
    for key, base_value in base_state.items():
        if not include_bn and dqa_v1._is_batchnorm_key(key):
            result[key] = base_value
        elif key_filter is not None and not key_filter(key):
            result[key] = base_value
        elif torch.is_tensor(base_value) and base_value.dtype.is_floating_point:
            residual = torch.zeros_like(base_value.float())
            for weight, state in zip(norm, source_states):
                residual = residual + weight * (state[key].float() - anchor_state[key].float())
            result[key] = (base_value.float() + beta * residual).to(base_value.dtype)
        else:
            result[key] = base_value
    return result


def residual_dqa_checkpoint(
    *,
    base: Path,
    sources: Sequence[Path],
    anchor: Path,
    output: Path,
    beta: float,
    scope: str,
    include_bn: bool,
) -> Path:
    base_ckpt = _load(base)
    source_ckpts = [_load(path) for path in sources]
    anchor_ckpt = _load(anchor)
    out = copy.deepcopy(base_ckpt)
    weights = [1.0] * len(sources)
    key_filter = key_filter_from_scope(scope)

    model = weighted_residual_state(
        dqa_v1._model_state_dict(base_ckpt, "model"),
        [dqa_v1._model_state_dict(ckpt, "model") for ckpt in source_ckpts],
        dqa_v1._model_state_dict(anchor_ckpt, "model"),
        weights=weights,
        beta=beta,
        key_filter=key_filter,
        include_bn=include_bn,
    )
    _replace(out, model, "model")

    base_ema = _state_dict(base_ckpt, "ema")
    source_emas = [_state_dict(ckpt, "ema") for ckpt in source_ckpts]
    anchor_ema = _state_dict(anchor_ckpt, "ema")
    if base_ema is not None and anchor_ema is not None and all(item is not None for item in source_emas):
        ema = weighted_residual_state(
            base_ema,
            [item for item in source_emas if item is not None],
            anchor_ema,
            weights=weights,
            beta=beta,
            key_filter=key_filter,
            include_bn=include_bn,
        )
        _replace(out, ema, "ema")

    return save_checkpoint(out, output)


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


def save_record(
    records: list[dict[str, str]],
    label: str,
    path: Path,
    kind: str,
    *,
    condition: str,
    round_idx: int | str = "",
    client: str = "",
    variant: str = "",
) -> None:
    records.append(
        {
            "condition": condition,
            "label": label,
            "kind": kind,
            "round": str(round_idx),
            "client": client,
            "variant": variant,
            "path": str(path.resolve()),
        }
    )


def write_checkpoint_records(path: Path, records: list[dict[str, str]]) -> None:
    write_csv(path, records, ["condition", "label", "kind", "round", "client", "variant", "path"])


def workspace_args(args: argparse.Namespace, workspace: Path) -> argparse.Namespace:
    copied = copy.copy(args)
    copied.workspace_root = workspace
    return copied


def prepare_workspace(args: argparse.Namespace, workspace: Path):
    cargs = workspace_args(args, workspace)
    pl03.ensure_dirs(cargs.workspace_root)
    setup, fedsto = dqa01.configure_modules(cargs.workspace_root, cargs.client_limit)
    setup_payload = setup.build_base_configs()
    manifest = setup_payload.get("manifest") if isinstance(setup_payload, dict) else {}
    clients = pl02.resolve_clients(cargs.clients, setup)
    warmup = pl02.copy_warmup_to_workspace(cargs.warmup_checkpoint, cargs.workspace_root, cargs.force)
    return cargs, setup, fedsto, manifest, clients, warmup


def repair_variant(name: str = "server_repair") -> pl03.Variant:
    return pl03.Variant(
        name=name,
        train_scope="all",
        aggregate_scope="all",
        client_epochs=1,
        client_lr0=0.0,
        source_repeat=1,
        pseudo_repeat=1,
        orthogonal_weight=0.0,
        note="Source/server repair only.",
    )


def dqa_variant(args: argparse.Namespace) -> pl03.Variant:
    return pl03.Variant(
        name="bn_residual_dqa_head",
        train_scope=args.dqa_train_scope,
        aggregate_scope=args.dqa_residual_scope,
        client_epochs=1,
        client_lr0=args.dqa_client_lr,
        source_repeat=args.dqa_source_repeat,
        pseudo_repeat=args.dqa_pseudo_repeat,
        orthogonal_weight=args.dqa_orthogonal_weight,
        note=(
            "Head/neck client adaptation with fixed stable pseudoGT. "
            "Aggregation uses day-client residuals with BN included."
        ),
    )


def run_server_repair_round(
    setup,
    fedsto,
    current_global: Path,
    args: argparse.Namespace,
    *,
    condition: str,
    variant: pl03.Variant,
    round_idx: int,
    port_offset: int,
) -> tuple[list[dict[str, str]], Path, int]:
    tag = round_tag(round_idx)
    print(f"\n=== {condition}: {tag} server repair ===")
    records: list[dict[str, str]] = []
    repair_start = fedsto.GLOBAL_DIR / f"03_{condition}_{tag}_server_repair_start.pt"
    repair = args.workspace_root / "checkpoints" / f"{tag}_{condition}_server_repair.pt"

    if args.server_repair_epochs <= 0:
        return records, current_global, port_offset

    if not args.dry_run and not fedsto.checkpoint_matches_protocol(repair_start, PROTOCOL_VERSION):
        fedsto.make_start_checkpoint(
            current_global,
            repair_start,
            protocol=PROTOCOL_VERSION,
            stage=f"{condition}_{tag}_server_repair_start",
        )

    if not pl03.reusable_checkpoint(fedsto, repair, args.force):
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
            fedsto.mark_checkpoint_protocol(raw_repair, PROTOCOL_VERSION, f"{condition}_{tag}_server_repair_raw")
            fedsto.make_start_checkpoint(
                raw_repair,
                repair,
                protocol=PROTOCOL_VERSION,
                stage=f"{condition}_{tag}_server_repair",
            )
            pl03.cleanup_training_artifacts(raw_repair, repair_start)

    save_record(
        records,
        f"{tag}_{condition}_server_repair",
        repair,
        "server_repair",
        condition=condition,
        round_idx=round_idx,
        variant=variant.name,
    )
    return records, repair, port_offset


def is_day_client(client: Mapping[str, Any]) -> bool:
    return str(client.get("timeofday", "")).lower() == "daytime" or str(client.get("weather", "")).endswith("_day")


def run_bn_residual_dqa_round(
    setup,
    fedsto,
    current_global: Path,
    args: argparse.Namespace,
    clients: list[dict[str, Any]],
    *,
    round_idx: int,
    port_offset: int,
) -> tuple[list[dict[str, str]], Path, dict[str, Any], int]:
    tag = round_tag(round_idx)
    variant = dqa_variant(args)
    print(f"\n=== bn_residual_dqa: {tag} client adaptation ===")
    pseudo_stats = pl03.generate_round_pseudo_labels(setup, current_global, args, clients, round_idx)

    records: list[dict[str, str]] = []
    local_paths: list[Path] = []
    day_paths: list[Path] = []

    with patched_client_config(args.dqa_loss_box):
        for client in clients:
            client_tag = f"client{client['id']}_{client['weather']}"
            start = fedsto.CLIENT_STATE_DIR / f"03_{tag}_bn_residual_dqa_{client_tag}_start.pt"
            raw_ckpt = fedsto.checkpoint_path(f"pl03_{tag}_{variant.name}_{client_tag}")
            final_ckpt = args.workspace_root / "checkpoints" / f"{tag}_bn_residual_dqa_{client_tag}.pt"

            if not args.dry_run and not fedsto.checkpoint_matches_protocol(start, PROTOCOL_VERSION):
                fedsto.make_start_checkpoint(
                    current_global,
                    start,
                    protocol=PROTOCOL_VERSION,
                    stage=f"{tag}_bn_residual_dqa_{client_tag}_start",
                )

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
                    fedsto.mark_checkpoint_protocol(raw_ckpt, PROTOCOL_VERSION, f"{tag}_bn_residual_dqa_{client_tag}_raw")
                    fedsto.make_start_checkpoint(
                        raw_ckpt,
                        final_ckpt,
                        protocol=PROTOCOL_VERSION,
                        stage=f"{tag}_bn_residual_dqa_{client_tag}",
                    )
                    pl03.cleanup_training_artifacts(raw_ckpt, start)

            local_paths.append(final_ckpt)
            if is_day_client(client):
                day_paths.append(final_ckpt)
            save_record(
                records,
                f"{tag}_{client_tag}",
                final_ckpt,
                "client",
                condition="bn_residual_dqa",
                round_idx=round_idx,
                client=client_tag,
                variant=variant.name,
            )

    if not day_paths:
        raise RuntimeError("No day clients were resolved; BN residual DQA needs day-client experts.")

    aggregate = args.workspace_root / "checkpoints" / f"{tag}_bn_residual_dqa_aggregate.pt"
    if not args.dry_run and not pl03.reusable_checkpoint(fedsto, aggregate, args.force):
        residual_dqa_checkpoint(
            base=current_global,
            sources=day_paths,
            anchor=current_global,
            output=aggregate,
            beta=args.dqa_residual_beta,
            scope=args.dqa_residual_scope,
            include_bn=args.dqa_include_bn,
        )
        fedsto.mark_checkpoint_protocol(aggregate, PROTOCOL_VERSION, f"{tag}_bn_residual_dqa_aggregate")

    save_record(
        records,
        f"{tag}_bn_residual_dqa_aggregate",
        aggregate,
        "aggregate",
        condition="bn_residual_dqa",
        round_idx=round_idx,
        variant=f"{variant.name}:beta={args.dqa_residual_beta}:scope={args.dqa_residual_scope}:bn={args.dqa_include_bn}",
    )

    repair_records, repaired, port_offset = run_server_repair_round(
        setup,
        fedsto,
        aggregate,
        args,
        condition="bn_residual_dqa",
        variant=variant,
        round_idx=round_idx,
        port_offset=port_offset,
    )
    records.extend(repair_records)
    return records, repaired, pseudo_stats, port_offset


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


def condition_display(label: str) -> str:
    return {
        "warmup_global": "warmup",
        "warmup_server_repair_final": "warmup + server repair",
        "bn_residual_dqa_final_aggregate": "warmup + BN-residual DQA aggregate",
        "bn_residual_dqa_final_repair": "warmup + BN-residual DQA + server repair",
    }.get(label, label)


def write_final_metrics(args: argparse.Namespace, eval_workspace: Path, eval_records: list[dict[str, str]]) -> list[dict[str, Any]]:
    summary_path = eval_workspace / "validation_reports" / "paper_protocol_eval_summary.csv"
    rows = [row for row in read_csv(summary_path) if row.get("status") == "ok"]
    total_rows = [row for row in rows if row.get("split") in {"scene_daynight_total", "total"}]
    by_label_total = {row["checkpoint_label"]: row for row in total_rows}
    by_label_split = {(row["checkpoint_label"], row["split"]): row for row in rows}
    meta = {row["label"]: row for row in eval_records}
    warm = by_label_total.get("warmup_global")
    repair = by_label_total.get("warmup_server_repair_final")
    warm_m95 = as_float(warm.get("map50_95")) if warm else None
    repair_m95 = as_float(repair.get("map50_95")) if repair else None

    ordered = [
        "warmup_global",
        "warmup_server_repair_final",
        "bn_residual_dqa_final_aggregate",
        "bn_residual_dqa_final_repair",
    ]
    metric_rows: list[dict[str, Any]] = []
    for label in ordered:
        total = by_label_total.get(label)
        if not total:
            continue
        m50 = as_float(total.get("map50"))
        m95 = as_float(total.get("map50_95"))
        gap = split_gap_metrics(by_label_split, label)
        metric_rows.append(
            {
                "checkpoint_label": label,
                "condition": condition_display(label),
                "kind": meta.get(label, {}).get("kind", ""),
                "source_condition": meta.get(label, {}).get("condition", ""),
                "round": meta.get(label, {}).get("round", ""),
                "precision": total.get("precision", ""),
                "recall": total.get("recall", ""),
                "map50": "" if m50 is None else f"{m50:.6f}",
                "map50_95": "" if m95 is None else f"{m95:.6f}",
                "gain_vs_warmup_map50_95": "" if m95 is None or warm_m95 is None else f"{m95 - warm_m95:.6f}",
                "delta_vs_server_repair_map50_95": "" if m95 is None or repair_m95 is None else f"{m95 - repair_m95:.6f}",
                **gap,
            }
        )

    fields = [
        "checkpoint_label",
        "condition",
        "kind",
        "source_condition",
        "round",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "gain_vs_warmup_map50_95",
        "delta_vs_server_repair_map50_95",
        "worst_split",
        "worst_split_map50_95",
        "day_avg_map50_95",
        "night_avg_map50_95",
        "day_night_gap_map50_95",
    ]
    write_csv(args.workspace_root / "stats" / "03_main_experiment_final_metrics.csv", metric_rows, fields)

    split_rows: list[dict[str, Any]] = []
    for row in rows:
        label = row["checkpoint_label"]
        if label not in meta:
            continue
        split_rows.append(
            {
                "checkpoint_label": label,
                "condition": condition_display(label),
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
        args.workspace_root / "stats" / "03_main_experiment_split_metrics.csv",
        split_rows,
        ["checkpoint_label", "condition", "split", "images", "labels", "precision", "recall", "map50", "map50_95"],
    )
    return metric_rows


def update_progress(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    *,
    stage: str,
    round_idx: int,
    completed: int,
    total: int,
    start_time: float,
    checkpoint: Path,
) -> None:
    elapsed = time.monotonic() - start_time
    avg = elapsed / completed if completed else 0.0
    eta = avg * (total - completed)
    rows.append(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "round": round_idx,
            "completed_steps": completed,
            "total_steps": total,
            "elapsed_seconds": f"{elapsed:.3f}",
            "eta_seconds": f"{eta:.3f}",
            "elapsed_hms": seconds_to_hms(elapsed),
            "eta_hms": seconds_to_hms(eta),
            "checkpoint": str(checkpoint.resolve()),
        }
    )
    write_csv(
        args.workspace_root / "stats" / "03_main_experiment_progress.csv",
        rows,
        [
            "created_utc",
            "stage",
            "round",
            "completed_steps",
            "total_steps",
            "elapsed_seconds",
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
    return tqdm(total=total, desc="03 main DQA experiment", unit="step")


def estimated_seconds(args: argparse.Namespace) -> float:
    return (
        args.repair_rounds * args.estimated_repair_round_minutes
        + args.dqa_rounds * args.estimated_dqa_round_minutes
        + (args.estimated_eval_minutes if args.evaluate else 0.0)
    ) * 60.0


def run(args: argparse.Namespace) -> None:
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    (args.workspace_root / "stats").mkdir(parents=True, exist_ok=True)

    repair_workspace = args.workspace_root / "warmup_server_repair"
    dqa_workspace = args.workspace_root / "bn_residual_dqa"
    repair_args, repair_setup, repair_fedsto, repair_manifest, _repair_clients, repair_warmup = prepare_workspace(args, repair_workspace)

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "workspace": str(args.workspace_root.resolve()),
        "repair_workspace": str(repair_workspace.resolve()),
        "dqa_workspace": str(dqa_workspace.resolve()),
        "repair_rounds": args.repair_rounds,
        "dqa_rounds": args.dqa_rounds,
        "dqa_policy": {
            "train_scope": args.dqa_train_scope,
            "residual_scope": args.dqa_residual_scope,
            "include_bn": args.dqa_include_bn,
            "residual_beta": args.dqa_residual_beta,
            "client_lr": args.dqa_client_lr,
            "source_repeat": args.dqa_source_repeat,
            "pseudo_repeat": args.dqa_pseudo_repeat,
            "loss_box": args.dqa_loss_box,
        },
        "server": repair_manifest.get("server"),
        "clients": _repair_clients,
        "estimated_runtime": seconds_to_hms(estimated_seconds(args)),
    }

    if args.setup_only:
        _, _, _, dqa_manifest, clients, _ = prepare_workspace(args, dqa_workspace)
        payload["server"] = dqa_manifest.get("server") or payload["server"]
        payload["clients"] = clients
        (args.workspace_root / "stats" / "03_main_experiment_manifest.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("Setup complete.")
        return

    (args.workspace_root / "stats" / "03_main_experiment_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    repair_args.gpus = repair_fedsto.resolve_gpus(repair_args.gpus)
    if not args.dry_run:
        repair_fedsto.check_runtime_dependencies()

    total_steps = args.repair_rounds + args.dqa_rounds
    progress = tqdm_factory(args, total_steps)
    progress_rows: list[dict[str, Any]] = []
    start_time = time.monotonic()
    completed = 0

    repair_records: list[dict[str, str]] = []
    save_record(repair_records, "warmup_global", repair_warmup, "warmup", condition="warmup")
    repair_current = repair_warmup
    repair_port = 0
    for idx in range(1, args.repair_rounds + 1):
        records, repair_current, repair_port = run_server_repair_round(
            repair_setup,
            repair_fedsto,
            repair_current,
            repair_args,
            condition="warmup_server_repair",
            variant=repair_variant("server_repair_baseline"),
            round_idx=idx,
            port_offset=repair_port,
        )
        repair_records.extend(records)
        write_checkpoint_records(repair_workspace / "stats" / "03_repair_checkpoints.csv", repair_records)
        completed += 1
        update_progress(args, progress_rows, stage="repair_baseline", round_idx=idx, completed=completed, total=total_steps, start_time=start_time, checkpoint=repair_current)
        if progress is not None:
            progress.set_postfix(stage="repair", round=idx, eta=progress_rows[-1]["eta_hms"])
            progress.update(1)

    # setup_scene_daynight and the FedSTO runner are imported modules with
    # mutable global paths.  Prepare the DQA workspace only after the repair
    # branch has finished, otherwise the DQA paths overwrite the repair branch.
    dqa_args, dqa_setup, dqa_fedsto, dqa_manifest, clients, dqa_warmup = prepare_workspace(args, dqa_workspace)
    dqa_args.gpus = dqa_fedsto.resolve_gpus(dqa_args.gpus)
    if not args.dry_run:
        dqa_fedsto.check_runtime_dependencies()
    payload["server"] = dqa_manifest.get("server") or payload["server"]
    payload["clients"] = clients
    (args.workspace_root / "stats" / "03_main_experiment_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    dqa_records: list[dict[str, str]] = []
    save_record(dqa_records, "warmup_global", dqa_warmup, "warmup", condition="warmup")
    dqa_current = dqa_warmup
    dqa_port = args.repair_rounds + 1
    pseudo_history: list[dict[str, Any]] = []
    for idx in range(1, args.dqa_rounds + 1):
        records, dqa_current, pseudo_stats, dqa_port = run_bn_residual_dqa_round(
            dqa_setup,
            dqa_fedsto,
            dqa_current,
            dqa_args,
            clients,
            round_idx=idx,
            port_offset=dqa_port,
        )
        dqa_records.extend(records)
        pseudo_history.append({"round": idx, "stats": pseudo_stats})
        write_checkpoint_records(dqa_workspace / "stats" / "03_bn_residual_dqa_checkpoints.csv", dqa_records)
        completed += 1
        update_progress(args, progress_rows, stage="bn_residual_dqa", round_idx=idx, completed=completed, total=total_steps, start_time=start_time, checkpoint=dqa_current)
        if progress is not None:
            progress.set_postfix(stage="dqa", round=idx, eta=progress_rows[-1]["eta_hms"])
            progress.update(1)

    if progress is not None:
        progress.close()

    final_aggregate_label = f"{round_tag(args.dqa_rounds)}_bn_residual_dqa_aggregate"
    final_repair_label = f"{round_tag(args.dqa_rounds)}_bn_residual_dqa_server_repair"
    repair_final_label = f"{round_tag(args.repair_rounds)}_warmup_server_repair_server_repair"

    repair_by_label = {row["label"]: row for row in repair_records}
    dqa_by_label = {row["label"]: row for row in dqa_records}
    eval_records = [
        {
            "condition": "warmup",
            "label": "warmup_global",
            "kind": "warmup",
            "round": "",
            "client": "",
            "variant": "",
            "path": str(dqa_warmup.resolve()),
        },
        {
            **repair_by_label[repair_final_label],
            "label": "warmup_server_repair_final",
        },
        {
            **dqa_by_label[final_aggregate_label],
            "label": "bn_residual_dqa_final_aggregate",
        },
        {
            **dqa_by_label[final_repair_label],
            "label": "bn_residual_dqa_final_repair",
        },
    ]
    write_checkpoint_records(args.workspace_root / "stats" / "03_main_experiment_eval_checkpoints.csv", eval_records)

    run_payload = {
        **payload,
        "actual_runtime_seconds": time.monotonic() - start_time,
        "repair_records": repair_records,
        "dqa_records": dqa_records,
        "pseudo_history": pseudo_history,
        "eval_records": eval_records,
    }
    (args.workspace_root / "stats" / "03_main_experiment_run_manifest.json").write_text(
        json.dumps(run_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.evaluate:
        eval_args = workspace_args(args, dqa_workspace)
        base01_0.run_evaluation(eval_args, eval_records)
        metrics = write_final_metrics(args, dqa_workspace, eval_records)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


def notify(args: argparse.Namespace, message: str, *, title: str, status: str | None = None, error: str | None = None) -> None:
    try:
        from notebook_notify import notify_discord

        context: dict[str, Any] = {
            "workspace": str(args.workspace_root.expanduser().resolve()),
            "repair_rounds": args.repair_rounds,
            "dqa_rounds": args.dqa_rounds,
            "estimated_runtime": seconds_to_hms(estimated_seconds(args)),
        }
        if status:
            context["status"] = status
        if error:
            context["error"] = error[:500]
        metrics_path = args.workspace_root.expanduser().resolve() / "stats" / "03_main_experiment_final_metrics.csv"
        if metrics_path.exists():
            rows = read_csv(metrics_path)
            context["metrics_csv"] = str(metrics_path)
            context["summary"] = str(
                [
                    {
                        "condition": row.get("condition"),
                        "map50": row.get("map50"),
                        "map50_95": row.get("map50_95"),
                        "delta_vs_server_repair": row.get("delta_vs_server_repair_map50_95"),
                    }
                    for row in rows
                ]
            )[:1500]
        print(notify_discord(message, title=title, context=context, fail_silently=True))
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}")


def str2bool(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT / "output" / "03_main_bn_residual_dqa_experiment")
    parser.add_argument("--warmup-checkpoint", type=Path, default=REPO_ROOT / "pseudogt_learnability" / "checkpoints" / "round000_warmup.pt")
    parser.add_argument("--client-limit", type=int, default=1500)
    parser.add_argument("--clients", default="all")
    parser.add_argument("--repair-rounds", type=int, default=30)
    parser.add_argument("--dqa-rounds", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=31841)
    parser.add_argument("--device", default="")
    parser.add_argument("--server-repair-epochs", type=int, default=1)
    parser.add_argument("--server-repair-lr", type=float, default=0.0008)
    parser.add_argument("--dqa-train-scope", choices=["neck_head", "all"], default="neck_head")
    parser.add_argument("--dqa-residual-scope", choices=["neck_head", "all"], default="neck_head")
    parser.add_argument("--dqa-include-bn", type=str2bool, default=True)
    parser.add_argument("--dqa-residual-beta", type=float, default=1.0)
    parser.add_argument("--dqa-client-lr", type=float, default=0.0008)
    parser.add_argument("--dqa-source-repeat", type=int, default=1)
    parser.add_argument("--dqa-pseudo-repeat", type=int, default=2)
    parser.add_argument("--dqa-loss-box", type=float, default=0.005)
    parser.add_argument("--dqa-orthogonal-weight", type=float, default=1e-4)
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-pseudo", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument(
        "--eval-splits",
        default="highway_day,highway_night,citystreet_day,citystreet_night,residential_day,residential_night,total",
    )
    parser.add_argument("--eval-clients", action="store_true")
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--classwise", action="store_true")
    parser.add_argument("--no-eval-plots", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--estimated-repair-round-minutes", type=float, default=4.0)
    parser.add_argument("--estimated-dqa-round-minutes", type=float, default=19.0)
    parser.add_argument("--estimated-eval-minutes", type=float, default=55.0)
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--notify-start", action="store_true")
    parser.add_argument("--notify-end", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    do_start_notify = args.notify or args.notify_start
    do_end_notify = args.notify or args.notify_end
    if do_start_notify:
        notify(args, "Scene-Daynight DQA 03 main experiment started.", title="DQA 03 main experiment start")

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
                f"Scene-Daynight DQA 03 main experiment finished with status={status}.",
                title="DQA 03 main experiment finish",
                status=status,
                error=error,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
