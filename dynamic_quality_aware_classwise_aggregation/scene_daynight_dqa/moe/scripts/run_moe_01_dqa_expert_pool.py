#!/usr/bin/env python3
"""Run scene-daynight DQA-MoE 01 expert-pool schedule.

This runner keeps the 02 head-to-full training schedule, but changes the DQA
aggregation target from one global checkpoint to a K-expert checkpoint pool.

The pilot is intentionally architecture-conservative:

* experts are full YOLO checkpoints initialized from the same warmup model;
* clients are trained once per round from the repaired deployable model;
* DQA routes client updates into domain/hard-case experts;
* a deployable checkpoint is created by soft-mixing expert residuals, then
  source server repair is applied once per round.

The resulting experiment tells us whether preserving specialization is useful
before we invest in a true in-model MoE head/router.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


MOE_ROOT = Path(__file__).resolve().parents[1]
SCENE_ROOT = MOE_ROOT.parent
REPO_ROOT = SCENE_ROOT.parents[1]
SCENE_SCRIPTS = SCENE_ROOT / "scripts"
DQA_ROOT = SCENE_ROOT.parent
PROTOCOL_VERSION = "scene_daynight_dqa_moe_01_expert_pool_v1"

for path in (SCENE_SCRIPTS, DQA_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dqa_cwa_aggregation as dqa_v1  # noqa: E402
import dqa_cwa_aggregation_v2 as dqa_v2  # noqa: E402
import run_scene_daynight_dqa_02_head_to_full as htf  # noqa: E402


SPLIT_NAMES = htf.SPLIT_NAMES


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
    return htf.as_float(value)


def round_tag(round_idx: int) -> str:
    return htf.round_tag(round_idx)


def seconds_to_hms(seconds: float | None) -> str:
    return htf.seconds_to_hms(seconds)


def client_tag(client: Mapping[str, Any]) -> str:
    return f"client{client['id']}_{client['weather']}"


def client_scene(client: Mapping[str, Any]) -> str:
    weather = str(client.get("weather", client.get("name", ""))).lower()
    if weather.startswith("highway"):
        return "highway"
    if weather.startswith("citystreet"):
        return "citystreet"
    if weather.startswith("residential"):
        return "residential"
    scene = str(client.get("scene", "")).lower().replace(" ", "")
    return scene or weather.split("_")[0]


def client_is_night(client: Mapping[str, Any]) -> bool:
    weather = str(client.get("weather", client.get("name", ""))).lower()
    return weather.endswith("_night") or str(client.get("timeofday", "")).lower() == "night"


def expert_names(k: int) -> list[str]:
    base = ["highway", "citystreet", "residential", "night_hard"]
    if k <= len(base):
        return base[:k]
    return base + [f"extra_{idx}" for idx in range(len(base), k)]


def routed_client_indices(
    expert_idx: int,
    clients: Sequence[dict[str, Any]],
    *,
    k: int,
) -> list[int]:
    """Route clients into K experts.

    For the default K=4 scene/day-night setting, the first three experts keep
    scene specialization and the fourth expert preserves night/hard-case signal.
    Clients may belong to more than one expert; this is intentional because the
    final deployable model is a soft mixture rather than a hard partition.
    """

    if k == 4:
        name = expert_names(k)[expert_idx]
        if name in {"highway", "citystreet", "residential"}:
            return [idx for idx, client in enumerate(clients) if client_scene(client) == name]
        if name == "night_hard":
            return [idx for idx, client in enumerate(clients) if client_is_night(client)]

    routed = [idx for idx in range(len(clients)) if idx % k == expert_idx]
    return routed or [expert_idx % len(clients)]


def stats_by_client(stats: Sequence[dqa_v1.ClientClassStats]) -> dict[str, dqa_v1.ClientClassStats]:
    return {item.client_id: item for item in stats}


def route_rows(
    clients: Sequence[dict[str, Any]],
    stats: Sequence[dqa_v1.ClientClassStats],
    *,
    k: int,
    phase: str,
    phase_round: int,
    global_round: int,
) -> list[dict[str, Any]]:
    by_client = stats_by_client(stats)
    rows: list[dict[str, Any]] = []
    names = expert_names(k)
    for expert_idx in range(k):
        selected = routed_client_indices(expert_idx, clients, k=k)
        for idx in selected:
            client = clients[idx]
            tag = client_tag(client)
            item = by_client.get(tag)
            rows.append(
                {
                    "phase": phase,
                    "phase_round": phase_round,
                    "global_round": global_round,
                    "expert_idx": expert_idx,
                    "expert_name": names[expert_idx],
                    "client": tag,
                    "scene": client_scene(client),
                    "is_night": client_is_night(client),
                    "pseudo_count": "" if item is None else f"{sum(item.counts):.0f}",
                    "mean_quality": "" if item is None else f"{float(np.mean(item.mean_quality_scores)):.6f}",
                }
            )
    return rows


def _load(path: Path) -> dict[str, Any]:
    return dqa_v1._load_checkpoint(path, REPO_ROOT)


def _state_dict(ckpt: Mapping[str, Any], key: str) -> dict[str, torch.Tensor] | None:
    if ckpt.get(key) is None:
        return None
    return dqa_v1._model_state_dict(ckpt, key)


def _weighted_residual_state(
    base_state: Mapping[str, torch.Tensor],
    source_states: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float],
    *,
    residual_blend: float,
    localize_bn: bool = True,
) -> dict[str, torch.Tensor]:
    total = float(sum(weights))
    norm = [float(w) / total for w in weights] if total > 0 else [1.0 / len(source_states)] * len(source_states)
    mixed: dict[str, torch.Tensor] = {}
    for key, base_value in base_state.items():
        if localize_bn and dqa_v1._is_batchnorm_key(key):
            mixed[key] = base_value
        elif torch.is_tensor(base_value) and base_value.dtype.is_floating_point:
            base_float = base_value.float()
            residual = torch.zeros_like(base_float)
            for weight, state in zip(norm, source_states):
                residual = residual + float(weight) * (state[key].float() - base_float)
            mixed[key] = (base_float + residual_blend * residual).to(base_value.dtype)
        else:
            mixed[key] = base_value
    return mixed


def softmix_checkpoints(
    *,
    base_checkpoint: Path,
    expert_checkpoints: Sequence[Path],
    output_checkpoint: Path,
    weights: Sequence[float],
    residual_blend: float,
) -> Path:
    base = copy.deepcopy(_load(base_checkpoint))
    expert_ckpts = [_load(path) for path in expert_checkpoints]

    base_model = dqa_v1._model_state_dict(base, "model")
    expert_models = [dqa_v1._model_state_dict(ckpt, "model") for ckpt in expert_ckpts]
    mixed = _weighted_residual_state(base_model, expert_models, weights, residual_blend=residual_blend)
    dqa_v1._replace_model_state(base, mixed, "model")

    base_ema = _state_dict(base, "ema")
    expert_emas = [_state_dict(ckpt, "ema") for ckpt in expert_ckpts]
    if base_ema is not None and all(item is not None for item in expert_emas):
        mixed_ema = _weighted_residual_state(
            base_ema,
            [item for item in expert_emas if item is not None],
            weights,
            residual_blend=residual_blend,
        )
        dqa_v1._replace_model_state(base, mixed_ema, "ema")

    base["epoch"] = -1
    base["optimizer"] = None
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, output_checkpoint)
    return output_checkpoint


def refresh_expert_checkpoint(
    *,
    repaired_checkpoint: Path,
    expert_checkpoint: Path,
    output_checkpoint: Path,
    expert_preserve: float,
) -> Path:
    return softmix_checkpoints(
        base_checkpoint=repaired_checkpoint,
        expert_checkpoints=[expert_checkpoint],
        output_checkpoint=output_checkpoint,
        weights=[1.0],
        residual_blend=expert_preserve,
    )


def init_experts(warmup: Path, args: argparse.Namespace, fedsto) -> list[Path]:  # noqa: ANN001
    paths: list[Path] = []
    for idx, name in enumerate(expert_names(args.experts)):
        path = args.workspace_root / "experts" / f"expert{idx}_{name}" / "round000_warmup.pt"
        if not args.dry_run and not htf.pl03.reusable_checkpoint(fedsto, path, args.force):
            fedsto.make_start_checkpoint(
                warmup,
                path,
                protocol=PROTOCOL_VERSION,
                stage=f"expert{idx}_{name}_round000_warmup",
            )
        paths.append(path)
    return paths


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
    expert_idx: int | str = "",
    expert_name: str = "",
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
            "expert_idx": str(expert_idx),
            "expert_name": expert_name,
            "path": str(path.resolve()),
        }
    )


def write_checkpoint_records(path: Path, records: list[dict[str, str]]) -> None:
    write_csv(
        path,
        records,
        [
            "label",
            "kind",
            "phase",
            "phase_round",
            "global_round",
            "client",
            "variant",
            "expert_idx",
            "expert_name",
            "path",
        ],
    )


def train_clients_once(
    setup,
    fedsto,
    spec: htf.PhaseSpec,
    current_global: Path,
    args: argparse.Namespace,
    clients: list[dict[str, Any]],
    *,
    phase_round: int,
    global_round: int,
    port_offset: int,
) -> tuple[list[dict[str, str]], list[Path], int]:
    tag = round_tag(global_round)
    records: list[dict[str, str]] = []
    local_paths: list[Path] = []
    variant = spec.variant()

    with htf.patched_client_config(spec.loss_box):
        for client in clients:
            tag_client = client_tag(client)
            start = fedsto.CLIENT_STATE_DIR / f"moe01_{tag}_{spec.name}_{tag_client}_start.pt"
            run_name = f"sdnmoe01_{tag}_{spec.name}_{tag_client}"
            final_ckpt = args.workspace_root / "checkpoints" / f"{tag}_{spec.name}_{tag_client}.pt"

            if not args.dry_run and not fedsto.checkpoint_matches_protocol(start, PROTOCOL_VERSION):
                fedsto.make_start_checkpoint(
                    current_global,
                    start,
                    protocol=PROTOCOL_VERSION,
                    stage=f"{tag}_{spec.name}_{tag_client}_start",
                )

            if not htf.pl03.reusable_checkpoint(fedsto, final_ckpt, args.force):
                cfg = htf.pl03.write_client_config(setup, variant, client, start, args, global_round)
                raw_ckpt = htf.pl03.run_train(
                    setup,
                    fedsto,
                    cfg,
                    dry_run=args.dry_run,
                    gpus=args.gpus,
                    master_port=args.master_port + port_offset,
                )
                port_offset += 1
                if not args.dry_run:
                    fedsto.mark_checkpoint_protocol(raw_ckpt, PROTOCOL_VERSION, f"{tag}_{spec.name}_{tag_client}_raw")
                    fedsto.make_start_checkpoint(
                        raw_ckpt,
                        final_ckpt,
                        protocol=PROTOCOL_VERSION,
                        stage=f"{tag}_{spec.name}_{tag_client}",
                    )
                    htf.pl03.cleanup_training_artifacts(raw_ckpt, start)

            local_paths.append(final_ckpt)
            save_checkpoint_record(
                records,
                f"{spec.name}_{tag}_{tag_client}",
                final_ckpt,
                "client",
                phase=spec.name,
                phase_round=phase_round,
                global_round=global_round,
                client=tag_client,
                variant=spec.name,
            )
    return records, local_paths, port_offset


def run_moe_round(
    setup,
    fedsto,
    spec: htf.PhaseSpec,
    current_global: Path,
    expert_paths: list[Path],
    args: argparse.Namespace,
    clients: list[dict[str, Any]],
    *,
    phase_round: int,
    global_round: int,
    port_offset: int,
) -> tuple[list[dict[str, str]], Path, list[Path], dict[str, Any], dict[str, Any], int]:
    tag = round_tag(global_round)
    print(f"\n=== {spec.display_name} DQA-MoE: {tag} phase_round={phase_round} ===")
    pseudo_stats = htf.pl03.generate_round_pseudo_labels(setup, current_global, args, clients, global_round)
    records, local_paths, port_offset = train_clients_once(
        setup,
        fedsto,
        spec,
        current_global,
        args,
        clients,
        phase_round=phase_round,
        global_round=global_round,
        port_offset=port_offset,
    )

    all_stats = htf.dqa01.pseudo_stats_to_dqa_stats(pseudo_stats, num_classes=len(setup.BDD_NAMES))
    by_client = stats_by_client(all_stats)
    by_path = {client_tag(client): path for client, path in zip(clients, local_paths)}
    route_log = route_rows(
        clients,
        all_stats,
        k=args.experts,
        phase=spec.name,
        phase_round=phase_round,
        global_round=global_round,
    )
    route_csv = args.workspace_root / "stats" / "01_moe_routes.csv"
    existing_routes = read_csv(route_csv)
    write_csv(
        route_csv,
        existing_routes + route_log,
        [
            "phase",
            "phase_round",
            "global_round",
            "expert_idx",
            "expert_name",
            "client",
            "scene",
            "is_night",
            "pseudo_count",
            "mean_quality",
        ],
    )

    cargs = htf.phase_args(args, args.workspace_root, spec)
    config = htf.dqa01.dqa_config(cargs, len(setup.BDD_NAMES))
    names = expert_names(args.experts)
    next_raw_experts: list[Path] = []
    dqa_states: dict[str, Any] = {}
    expert_weights: list[float] = []

    for expert_idx, name in enumerate(names):
        selected_indices = routed_client_indices(expert_idx, clients, k=args.experts)
        selected_tags = [client_tag(clients[idx]) for idx in selected_indices]
        selected_paths = [by_path[tag] for tag in selected_tags]
        selected_stats = [by_client[tag] for tag in selected_tags]
        pseudo_mass = float(sum(sum(item.counts) for item in selected_stats))
        expert_weights.append(1.0 if args.moe_softmix_weighting == "uniform" else max(pseudo_mass, 1.0))

        aggregate = args.workspace_root / "experts" / f"expert{expert_idx}_{name}" / f"{tag}_{spec.name}_dqa_aggregate.pt"
        state_path = args.workspace_root / "stats" / f"01_moe_expert{expert_idx}_{name}_dqa_state.json"
        if not args.dry_run and not htf.pl03.reusable_checkpoint(fedsto, aggregate, args.force):
            _, state = dqa_v2.aggregate_checkpoints(
                client_checkpoints=selected_paths,
                server_checkpoint=expert_paths[expert_idx],
                output_checkpoint=aggregate,
                stats=selected_stats,
                state_path=state_path,
                config=config,
                repo_root=REPO_ROOT,
            )
            fedsto.mark_checkpoint_protocol(aggregate, PROTOCOL_VERSION, f"{tag}_{spec.name}_expert{expert_idx}_{name}_dqa_aggregate")
            dqa_states[f"expert{expert_idx}_{name}"] = state
        elif state_path.exists():
            dqa_states[f"expert{expert_idx}_{name}"] = json.loads(state_path.read_text(encoding="utf-8"))

        next_raw_experts.append(aggregate)
        save_checkpoint_record(
            records,
            f"{spec.name}_{tag}_expert{expert_idx}_{name}_aggregate",
            aggregate,
            "expert_aggregate",
            phase=spec.name,
            phase_round=phase_round,
            global_round=global_round,
            variant=spec.name,
            expert_idx=expert_idx,
            expert_name=name,
        )

    softmix = args.workspace_root / "checkpoints" / f"{tag}_{spec.name}_moe_softmix.pt"
    if not args.dry_run and not htf.pl03.reusable_checkpoint(fedsto, softmix, args.force):
        softmix_checkpoints(
            base_checkpoint=current_global,
            expert_checkpoints=next_raw_experts,
            output_checkpoint=softmix,
            weights=expert_weights,
            residual_blend=args.moe_softmix_blend,
        )
        fedsto.mark_checkpoint_protocol(softmix, PROTOCOL_VERSION, f"{tag}_{spec.name}_moe_softmix")
    save_checkpoint_record(
        records,
        f"{spec.name}_{tag}_moe_softmix",
        softmix,
        "moe_softmix",
        phase=spec.name,
        phase_round=phase_round,
        global_round=global_round,
        variant=spec.name,
    )

    repair_start = fedsto.GLOBAL_DIR / f"moe01_{tag}_{spec.name}_server_repair_start.pt"
    repair = args.workspace_root / "checkpoints" / f"{tag}_{spec.name}_server_repair.pt"
    if args.server_repair_epochs > 0:
        if not args.dry_run and not htf.pl03.reusable_checkpoint(fedsto, repair, args.force):
            fedsto.make_start_checkpoint(
                softmix,
                repair_start,
                protocol=PROTOCOL_VERSION,
                stage=f"{tag}_{spec.name}_server_repair_start",
            )
            cfg = htf.pl03.write_server_repair_config(setup, spec.variant(), repair_start, args, global_round)
            raw_repair = htf.pl03.run_train(
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
                htf.pl03.cleanup_training_artifacts(raw_repair, repair_start)
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
        next_global = softmix

    refreshed_experts: list[Path] = []
    if args.server_repair_epochs > 0 and args.expert_preserve_after_repair < 1.0:
        for expert_idx, (name, expert_raw) in enumerate(zip(names, next_raw_experts)):
            refreshed = args.workspace_root / "experts" / f"expert{expert_idx}_{name}" / f"{tag}_{spec.name}_refreshed.pt"
            if not args.dry_run and not htf.pl03.reusable_checkpoint(fedsto, refreshed, args.force):
                refresh_expert_checkpoint(
                    repaired_checkpoint=next_global,
                    expert_checkpoint=expert_raw,
                    output_checkpoint=refreshed,
                    expert_preserve=args.expert_preserve_after_repair,
                )
                fedsto.mark_checkpoint_protocol(refreshed, PROTOCOL_VERSION, f"{tag}_{spec.name}_expert{expert_idx}_{name}_refreshed")
            refreshed_experts.append(refreshed)
            save_checkpoint_record(
                records,
                f"{spec.name}_{tag}_expert{expert_idx}_{name}_refreshed",
                refreshed,
                "expert_refreshed",
                phase=spec.name,
                phase_round=phase_round,
                global_round=global_round,
                variant=spec.name,
                expert_idx=expert_idx,
                expert_name=name,
            )
    else:
        refreshed_experts = next_raw_experts

    return records, next_global, refreshed_experts, pseudo_stats, dqa_states, port_offset


def selected_eval_records(
    records: list[dict[str, str]],
    *,
    phase1: htf.PhaseSpec,
    phase2: htf.PhaseSpec,
    phase1_rounds: int,
    phase2_rounds: int,
    k: int,
    eval_all_rounds: bool,
    eval_experts: bool,
    eval_phase1_experts: bool,
) -> list[dict[str, str]]:
    if eval_all_rounds:
        return records

    by_label = {row["label"]: row for row in records}
    phase1_tag = round_tag(phase1_rounds)
    phase2_global_round = phase1_rounds + phase2_rounds
    phase2_tag = round_tag(phase2_global_round)
    wanted: list[tuple[str, str]] = [
        ("warmup_global", "warmup_global"),
        (f"{phase1.name}_{phase1_tag}_moe_softmix", "phase1_final_moe_softmix"),
        (f"{phase1.name}_{phase1_tag}_server_repair", "phase1_final_repair"),
        (f"{phase2.name}_{phase2_tag}_moe_softmix", "phase2_final_moe_softmix"),
        (f"{phase2.name}_{phase2_tag}_server_repair", "phase2_final_repair"),
    ]
    if eval_experts:
        names = expert_names(k)
        def preferred_expert_label(phase: htf.PhaseSpec, tag: str, idx: int, name: str) -> str:
            refreshed = f"{phase.name}_{tag}_expert{idx}_{name}_refreshed"
            if refreshed in by_label:
                return refreshed
            return f"{phase.name}_{tag}_expert{idx}_{name}_aggregate"

        if eval_phase1_experts:
            wanted.extend(
                (
                    preferred_expert_label(phase1, phase1_tag, idx, name),
                    f"phase1_final_expert{idx}_{name}",
                )
                for idx, name in enumerate(names)
            )
        wanted.extend(
            (
                preferred_expert_label(phase2, phase2_tag, idx, name),
                f"phase2_final_expert{idx}_{name}",
            )
            for idx, name in enumerate(names)
        )

    selected: list[dict[str, str]] = []
    for source_label, eval_label in wanted:
        row = by_label.get(source_label)
        if row is None:
            continue
        copied = dict(row)
        copied["label"] = eval_label
        selected.append(copied)
    return selected


def split_gap_metrics(by_label_split: dict[tuple[str, str], dict[str, str]], label: str) -> dict[str, Any]:
    return htf.split_gap_metrics(by_label_split, label)


def repair_reference(args: argparse.Namespace) -> dict[str, str] | None:
    return htf.repair_reference(args)


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
                "expert_idx": meta.get(label, {}).get("expert_idx", ""),
                "expert_name": meta.get(label, {}).get("expert_name", ""),
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
        "expert_idx",
        "expert_name",
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
    metrics_path = args.workspace_root / "stats" / "01_moe_final_metrics.csv"
    write_csv(metrics_path, metric_rows, fieldnames)

    split_rows: list[dict[str, Any]] = []
    for row in rows:
        label = row["checkpoint_label"]
        if label not in meta:
            continue
        split_rows.append(
            {
                "checkpoint_label": label,
                "kind": meta[label].get("kind", ""),
                "phase": meta[label].get("phase", ""),
                "expert_idx": meta[label].get("expert_idx", ""),
                "expert_name": meta[label].get("expert_name", ""),
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
        args.workspace_root / "stats" / "01_moe_split_metrics.csv",
        split_rows,
        [
            "checkpoint_label",
            "kind",
            "phase",
            "expert_idx",
            "expert_name",
            "split",
            "images",
            "labels",
            "precision",
            "recall",
            "map50",
            "map50_95",
        ],
    )
    write_oracle_split_metrics(args, split_rows)
    print(f"Saved: {metrics_path}")
    return metric_rows


def write_oracle_split_metrics(args: argparse.Namespace, split_rows: list[dict[str, Any]]) -> None:
    expert_rows = [
        row
        for row in split_rows
        if str(row.get("checkpoint_label", "")).startswith("phase2_final_expert")
        and row.get("split") not in {"scene_daynight_total", "total"}
    ]
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in expert_rows:
        by_split.setdefault(str(row["split"]), []).append(row)

    oracle_rows: list[dict[str, Any]] = []
    for split, rows in sorted(by_split.items()):
        best_m95 = max(rows, key=lambda row: as_float(row.get("map50_95")) or -1.0)
        best_m50 = max(rows, key=lambda row: as_float(row.get("map50")) or -1.0)
        oracle_rows.append(
            {
                "split": split,
                "best_map50_95_checkpoint": best_m95["checkpoint_label"],
                "best_map50_95": best_m95.get("map50_95", ""),
                "best_map50_checkpoint": best_m50["checkpoint_label"],
                "best_map50": best_m50.get("map50", ""),
            }
        )

    if oracle_rows:
        write_csv(
            args.workspace_root / "stats" / "01_moe_oracle_split_metrics.csv",
            oracle_rows,
            ["split", "best_map50_95_checkpoint", "best_map50_95", "best_map50_checkpoint", "best_map50"],
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
        args.workspace_root / "stats" / "01_moe_progress.csv",
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
    return tqdm(total=total, desc="01 DQA-MoE expert pool", unit="round")


def estimated_seconds(args: argparse.Namespace) -> float:
    base = htf.estimated_seconds(args)
    aggregation = (args.phase1_rounds + args.phase2_rounds) * args.experts * args.estimated_moe_aggregation_minutes * 60.0
    expert_eval = args.experts * args.estimated_eval_expert_minutes * 60.0 if args.evaluate and args.eval_experts else 0.0
    return base + aggregation + expert_eval


def run(args: argparse.Namespace) -> None:
    setup, fedsto, manifest, clients, warmup = htf.prepare(args)
    args.gpus = fedsto.resolve_gpus(args.gpus)
    phase1 = htf.default_phase1_spec(args)
    phase2 = htf.default_phase2_spec(args)
    phase1 = htf.PhaseSpec(**(asdict(phase1) | {"name": "phase1_moe_head", "display_name": "Phase 1 DQA-MoE head-only"}))
    phase2 = htf.PhaseSpec(**(asdict(phase2) | {"name": "phase2_moe_full", "display_name": "Phase 2 DQA-MoE full burst"}))
    total_rounds = args.phase1_rounds + args.phase2_rounds
    est = estimated_seconds(args)

    print(
        json.dumps(
            {
                "protocol": PROTOCOL_VERSION,
                "workspace": str(args.workspace_root.resolve()),
                "experts": args.experts,
                "expert_names": expert_names(args.experts),
                "clients": clients,
                "server": manifest.get("server"),
                "phase1": asdict(phase1),
                "phase2": asdict(phase2),
                "moe_softmix_blend": args.moe_softmix_blend,
                "expert_preserve_after_repair": args.expert_preserve_after_repair,
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

    if not args.dry_run:
        fedsto.check_runtime_dependencies()

    records: list[dict[str, str]] = []
    save_checkpoint_record(records, "warmup_global", warmup, "warmup", phase="warmup")
    expert_paths = init_experts(warmup, args, fedsto)
    current_global = warmup
    pseudo_history: list[dict[str, Any]] = []
    dqa_history: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    port_offset = 0
    start_time = time.monotonic()
    progress = tqdm_factory(args, total_rounds)

    def finish_round(spec: htf.PhaseSpec, phase_round: int, global_round: int, checkpoint: Path) -> None:
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
        cargs = htf.phase_args(args, args.workspace_root, phase1)
        round_records, current_global, expert_paths, pseudo_stats, dqa_states, port_offset = run_moe_round(
            setup,
            fedsto,
            phase1,
            current_global,
            expert_paths,
            cargs,
            clients,
            phase_round=phase_round,
            global_round=global_round,
            port_offset=port_offset,
        )
        records.extend(round_records)
        pseudo_history.append({"phase": phase1.name, "phase_round": phase_round, "global_round": global_round, "stats": pseudo_stats})
        dqa_history.append({"phase": phase1.name, "phase_round": phase_round, "global_round": global_round, "state": dqa_states})
        write_checkpoint_records(args.workspace_root / "stats" / "01_moe_checkpoints.csv", records)
        finish_round(phase1, phase_round, global_round, current_global)

    for phase_round in range(1, args.phase2_rounds + 1):
        global_round = args.phase1_rounds + phase_round
        cargs = htf.phase_args(args, args.workspace_root, phase2)
        round_records, current_global, expert_paths, pseudo_stats, dqa_states, port_offset = run_moe_round(
            setup,
            fedsto,
            phase2,
            current_global,
            expert_paths,
            cargs,
            clients,
            phase_round=phase_round,
            global_round=global_round,
            port_offset=port_offset,
        )
        records.extend(round_records)
        pseudo_history.append({"phase": phase2.name, "phase_round": phase_round, "global_round": global_round, "stats": pseudo_stats})
        dqa_history.append({"phase": phase2.name, "phase_round": phase_round, "global_round": global_round, "state": dqa_states})
        write_checkpoint_records(args.workspace_root / "stats" / "01_moe_checkpoints.csv", records)
        finish_round(phase2, phase_round, global_round, current_global)

    if progress is not None:
        progress.close()

    manifest_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "workspace": str(args.workspace_root.resolve()),
        "experts": args.experts,
        "expert_names": expert_names(args.experts),
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
        "final_experts": [str(path.resolve()) for path in expert_paths],
        "estimated_runtime_seconds": est,
        "actual_runtime_seconds": time.monotonic() - start_time,
        "final_checkpoint": str(current_global.resolve()),
        "final_focused_evaluation": args.evaluate and not args.eval_all_rounds,
        "moe_softmix_blend": args.moe_softmix_blend,
        "expert_preserve_after_repair": args.expert_preserve_after_repair,
    }
    (args.workspace_root / "stats" / "01_moe_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.evaluate:
        eval_records = selected_eval_records(
            records,
            phase1=phase1,
            phase2=phase2,
            phase1_rounds=args.phase1_rounds,
            phase2_rounds=args.phase2_rounds,
            k=args.experts,
            eval_all_rounds=args.eval_all_rounds,
            eval_experts=args.eval_experts,
            eval_phase1_experts=args.eval_phase1_experts,
        )
        write_checkpoint_records(args.workspace_root / "stats" / "01_moe_eval_checkpoints.csv", eval_records)
        htf.base01_0.run_evaluation(args, eval_records)
        write_final_metrics(args, eval_records)


def notify(args: argparse.Namespace, message: str, *, title: str, status: str | None = None, error: str | None = None) -> None:
    try:
        from notebook_notify import notify_discord

        context = {
            "workspace": str(args.workspace_root.expanduser().resolve()),
            "experts": args.experts,
            "phase1_rounds": args.phase1_rounds,
            "phase2_rounds": args.phase2_rounds,
            "estimated_runtime": seconds_to_hms(estimated_seconds(args)),
        }
        if status:
            context["status"] = status
        if error:
            context["error"] = error[:500]
        metrics_path = args.workspace_root.expanduser().resolve() / "stats" / "01_moe_final_metrics.csv"
        if metrics_path.exists():
            context["final_metrics_csv"] = str(metrics_path)
        result = notify_discord(message, title=title, context=context, fail_silently=True)
        print(result)
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}")


def _has_arg(argv: Sequence[str], name: str) -> bool:
    return any(item == name or item.startswith(f"{name}=") for item in argv)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--experts", type=int, default=4)
    custom.add_argument("--moe-softmix-blend", type=float, default=0.28)
    custom.add_argument("--moe-softmix-weighting", choices=("uniform", "pseudo_mass"), default="uniform")
    custom.add_argument("--expert-preserve-after-repair", type=float, default=0.65)
    custom.add_argument("--eval-experts", action=argparse.BooleanOptionalAction, default=True)
    custom.add_argument("--eval-phase1-experts", action=argparse.BooleanOptionalAction, default=False)
    custom.add_argument("--estimated-moe-aggregation-minutes", type=float, default=0.35)
    custom.add_argument("--estimated-eval-expert-minutes", type=float, default=6.0)
    custom_args, remaining = custom.parse_known_args(raw)

    if not _has_arg(remaining, "--workspace-root"):
        remaining.extend(["--workspace-root", str(MOE_ROOT / "output" / "01_dqa_moe_expert_pool")])
    args = htf.parse_args(remaining)
    args.experts = custom_args.experts
    args.moe_softmix_blend = custom_args.moe_softmix_blend
    args.moe_softmix_weighting = custom_args.moe_softmix_weighting
    args.expert_preserve_after_repair = custom_args.expert_preserve_after_repair
    args.eval_experts = custom_args.eval_experts
    args.eval_phase1_experts = custom_args.eval_phase1_experts
    args.estimated_moe_aggregation_minutes = custom_args.estimated_moe_aggregation_minutes
    args.estimated_eval_expert_minutes = custom_args.estimated_eval_expert_minutes
    if args.experts <= 0:
        raise ValueError("--experts must be positive")
    if not 0.0 <= args.moe_softmix_blend <= 1.0:
        raise ValueError("--moe-softmix-blend must be in [0, 1]")
    if not 0.0 <= args.expert_preserve_after_repair <= 1.0:
        raise ValueError("--expert-preserve-after-repair must be in [0, 1]")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    do_start_notify = args.notify or args.notify_start
    do_end_notify = args.notify or args.notify_end
    if do_start_notify:
        notify(args, "Scene-Daynight DQA-MoE 01 expert-pool started.", title="DQA-MoE 01 start")

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
                f"Scene-Daynight DQA-MoE 01 expert-pool finished with status={status}.",
                title="DQA-MoE 01 finish",
                status=status,
                error=error,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
