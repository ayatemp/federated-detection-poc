#!/usr/bin/env python3
"""Run tighter-specialist latent DQA-MoX from an existing warmup.

10 is the tighter follow-up loop after 09. It keeps the anonymous latent MoE
head, but makes the pseudo-GT learning problem more conservative before client
training:

* generate stable pseudo boxes from the current global model;
* select pseudo boxes through expert-choice buckets that favor clean, rare,
  small/scale-sensitive, and hard-but-stable regions;
* cap head-class pseudo boxes more strongly than 09;
* train clients with source-dominant batches and weaker pseudo bbox loss;
* compute DQA classwise aggregation stats from the selected boxes, not from the
  raw, class-imbalanced pseudoGT pool.

Warmup and repair-only baselines are intentionally skipped here. They should be
reused from 08 so this loop can focus on improving the DQA branch. The default
round count is shorter than the 09 full run because this is meant as the next
debuggable improvement loop after 09 plateaued around mAP50 0.509.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
NAV_ROOT = REPO_ROOT / "navigating_data_heterogeneity"
DQA_ROOT = PROJECT_ROOT.parent
PSEUDOGT_SCRIPTS = REPO_ROOT / "pseudogt_learnability" / "scripts"
PROTOCOL_VERSION = "scene_daynight_dqa_10_tighter_specialist_latent_dqamox_v1"

for path in (NAV_ROOT, DQA_ROOT, PSEUDOGT_SCRIPTS, PROJECT_ROOT / "scripts", REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dqa_cwa_aggregation as dqa_v1  # noqa: E402
import dqa_cwa_aggregation_v2 as dqa_v2  # noqa: E402
import run_pseudogt_learnability_02 as pl02  # noqa: E402
import run_pseudogt_learnability_03 as pl03  # noqa: E402
import run_scene_daynight_dqa_01 as dqa01  # noqa: E402
import run_scene_daynight_dqa_01_0 as base01_0  # noqa: E402
import run_scene_daynight_dqa_05_expert_choice_pseudogt_router as ec05  # noqa: E402


SPLIT_NAMES = base01_0.SPLIT_NAMES


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


def ensure_dirs(workspace: Path) -> None:
    pl03.ensure_dirs(workspace)
    for relative in ("reports",):
        (workspace / relative).mkdir(parents=True, exist_ok=True)


def configure_workspace(args: argparse.Namespace):
    ensure_dirs(args.workspace_root)
    setup, fedsto = dqa01.configure_modules(args.workspace_root, args.client_limit)
    setup_payload = setup.build_base_configs()
    manifest = setup_payload.get("manifest") if isinstance(setup_payload, dict) else {}
    clients = pl02.resolve_clients(args.clients, setup)
    return setup, fedsto, manifest, clients


def patch_latent_moe_config(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg.setdefault("Model", {}).setdefault("Head", {})["name"] = "LatentMoEYoloV5"
    cfg["LatentMoE"] = {
        "enabled": True,
        "num_experts": int(args.num_experts),
        "top_k": int(args.top_k),
        "temperature": float(args.router_temperature),
        "scale": float(args.moe_scale),
        "balance_weight": float(args.router_balance_weight),
        "entropy_weight": float(args.router_entropy_weight),
    }
    cfg.setdefault("ClassSkewFedSTO", {})
    cfg["ClassSkewFedSTO"]["enabled"] = False
    cfg["ClassSkewFedSTO"]["use_residual"] = False
    return cfg


def config_device(args: argparse.Namespace) -> str:
    return ""


def repeated_expr(path: Path, repeat: int) -> str:
    return str(path.resolve()) if repeat <= 1 else f"{path.resolve()}*{repeat}"


def train_expr(source_list: Path, pseudo_list: Path, source_repeat: int, pseudo_repeat: int) -> str:
    return "||".join((repeated_expr(source_list, source_repeat), repeated_expr(pseudo_list, pseudo_repeat)))


def run_train(setup, fedsto, config: Path, *, dry_run: bool, gpus: int, master_port: int) -> Path:
    return pl03.run_train(setup, fedsto, config, dry_run=dry_run, gpus=gpus, master_port=master_port)


def reusable_checkpoint(fedsto, path: Path, args: argparse.Namespace) -> bool:
    return fedsto.checkpoint_matches_protocol(path, PROTOCOL_VERSION) and pl03.reusable_checkpoint(fedsto, path, args.force)


def write_warmup_config(setup, fedsto, args: argparse.Namespace, weights: Path) -> Path:
    cfg = setup.efficientteacher_config(
        name="sdn10_tighter_specialist_latent_dqamox_warmup",
        train=setup.LIST_ROOT / "server_cloudy_train.txt",
        val=setup.LIST_ROOT / "server_cloudy_val.txt",
        target=None,
        weights=str(weights.resolve()),
        epochs=args.warmup_epochs,
        train_scope="all",
        orthogonal_weight=0.0,
        batch_size=args.batch_size,
        workers=args.workers,
        device=config_device(args),
    )
    patch_latent_moe_config(cfg, args)
    cfg["linear_lr"] = args.warmup_epochs > 1
    cfg["hyp"]["lr0"] = args.warmup_lr
    cfg["hyp"]["lrf"] = args.warmup_lrf
    cfg["hyp"]["warmup_epochs"] = min(3, max(0, args.warmup_epochs // 10))
    return setup.write_config("sdn10_tighter_specialist_latent_dqamox_warmup.yaml", cfg)


def materialize_latent_moe_checkpoint(setup, fedsto, args: argparse.Namespace, source: Path, out: Path) -> Path:
    """Convert an existing plain warmup checkpoint into the latent-MoE architecture.

    Existing YOLOv5 weights are copied where shapes match. The new router/expert
    residual branch starts from zero, so the converted detector initially behaves
    like the original warmup model but can train latent experts afterward.
    """

    if reusable_checkpoint(fedsto, out, args):
        return out

    cfg_path = write_warmup_config(setup, fedsto, args, source)
    fedsto.ensure_efficientteacher_import_path()
    from configs.defaults import get_cfg
    from models.detector.yolo import Model
    from utils.torch_utils import intersect_dicts

    cfg = get_cfg()
    cfg.merge_from_file(str(cfg_path))
    cfg.freeze()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    model = Model(cfg)
    src_model = checkpoint.get("model")
    if src_model is None:
        raise RuntimeError(f"Checkpoint has no model: {source}")
    model_state = intersect_dicts(src_model.float().state_dict(), model.state_dict(), exclude=["anchor"])
    model.load_state_dict(model_state, strict=False)

    converted = copy.deepcopy(checkpoint)
    converted["model"] = model.half()
    if checkpoint.get("ema") is not None:
        ema_model = Model(cfg)
        ema_state = intersect_dicts(checkpoint["ema"].float().state_dict(), ema_model.state_dict(), exclude=["anchor"])
        ema_model.load_state_dict(ema_state, strict=False)
        converted["ema"] = ema_model.half()
    else:
        converted["ema"] = None
    converted["epoch"] = -1
    converted["optimizer"] = None
    converted["fedsto_protocol"] = PROTOCOL_VERSION
    converted["fedsto_stage"] = "round000_external_warmup_materialized_as_latent_moe"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, out)
    return out


def train_or_load_warmup(setup, fedsto, args: argparse.Namespace, port_offset: int) -> tuple[Path, int]:
    warmup = args.workspace_root / "checkpoints" / "round000_latent_dqamox_warmup.pt"
    if reusable_checkpoint(fedsto, warmup, args):
        return warmup, port_offset

    if args.skip_warmup_training:
        if args.warmup_checkpoint is None:
            raise ValueError("--skip-warmup-training requires --warmup-checkpoint")
        materialize_latent_moe_checkpoint(setup, fedsto, args, args.warmup_checkpoint, warmup)
        return warmup, port_offset

    weights = args.pretrained_checkpoint
    if weights is None:
        weights = fedsto.download_pretrained(force=args.force_pretrained) if not args.dry_run else fedsto.PRETRAINED_PATH
    cfg = write_warmup_config(setup, fedsto, args, weights)
    raw = run_train(
        setup,
        fedsto,
        cfg,
        dry_run=args.dry_run,
        gpus=args.gpus,
        master_port=args.master_port + port_offset,
    )
    port_offset += 1
    if not args.dry_run:
        fedsto.mark_checkpoint_protocol(raw, PROTOCOL_VERSION, "round000_latent_dqamox_warmup_raw")
        fedsto.make_start_checkpoint(raw, warmup, protocol=PROTOCOL_VERSION, stage="round000_latent_dqamox_warmup")
        pl03.cleanup_training_artifacts(raw, None)
    return warmup, port_offset


def apply_train_hyp(cfg: dict[str, Any], *, lr: float, loss_box: float, args: argparse.Namespace) -> None:
    cfg["linear_lr"] = False
    cfg["hyp"]["lr0"] = float(lr)
    cfg["hyp"]["lrf"] = 1.0
    cfg["hyp"]["warmup_epochs"] = 0
    cfg["hyp"]["mixup"] = 0.0
    cfg["hyp"]["scale"] = 0.25
    cfg["hyp"]["hsv_s"] = 0.35
    cfg["hyp"]["hsv_v"] = 0.20
    cfg.setdefault("Loss", {})
    cfg["Loss"]["box"] = float(loss_box)


def write_client_config(
    setup,
    *,
    condition: str,
    phase: int,
    round_idx: int,
    client: dict[str, Any],
    start: Path,
    train_scope: str,
    epochs: int,
    lr: float,
    source_repeat: int,
    pseudo_repeat: int,
    loss_box: float,
    pseudo_list: Path | None,
    args: argparse.Namespace,
) -> Path:
    tag = round_tag(round_idx)
    client_tag = f"client{client['id']}_{client['weather']}"
    run_name = f"sdn10_{condition}_p{phase}_{tag}_{client_tag}"
    source_list = setup.LIST_ROOT / "server_cloudy_train.txt"
    if pseudo_list is None:
        pseudo_list = setup.LIST_ROOT / f"pl03_{tag}_{client_tag}_stable_train.txt"
    cfg = setup.efficientteacher_config(
        name=run_name,
        train=source_list,
        val=setup.LIST_ROOT / "server_cloudy_val.txt",
        target=None,
        weights=str(start.resolve()),
        epochs=epochs,
        train_scope=train_scope,
        orthogonal_weight=args.orthogonal_weight,
        batch_size=args.batch_size,
        workers=args.workers,
        device=config_device(args),
    )
    patch_latent_moe_config(cfg, args)
    cfg["Dataset"]["train"] = train_expr(source_list, pseudo_list, source_repeat, pseudo_repeat)
    cfg["FedSTO"]["unlabeled_only_client"] = False
    cfg["SSOD"] = {"train_domain": False}
    apply_train_hyp(cfg, lr=lr, loss_box=loss_box, args=args)
    return setup.write_config(f"{run_name}.yaml", cfg)


def write_server_repair_config(
    setup,
    *,
    condition: str,
    phase: int,
    round_idx: int,
    start: Path,
    train_scope: str,
    args: argparse.Namespace,
) -> Path:
    tag = round_tag(round_idx)
    run_name = f"sdn10_{condition}_p{phase}_{tag}_server_repair"
    cfg = setup.efficientteacher_config(
        name=run_name,
        train=setup.LIST_ROOT / "server_cloudy_train.txt",
        val=setup.LIST_ROOT / "server_cloudy_val.txt",
        target=None,
        weights=str(start.resolve()),
        epochs=args.server_repair_epochs,
        train_scope=train_scope,
        orthogonal_weight=0.0,
        batch_size=args.batch_size,
        workers=args.workers,
        device=config_device(args),
    )
    patch_latent_moe_config(cfg, args)
    cfg["SSOD"] = {"train_domain": False}
    apply_train_hyp(cfg, lr=args.server_repair_lr, loss_box=args.server_repair_loss_box, args=args)
    return setup.write_config(f"{run_name}.yaml", cfg)


def pseudo_stats_to_dqa_stats(pseudo_stats: dict[str, Any], num_classes: int) -> list[dqa_v1.ClientClassStats]:
    rows = []
    for client_tag, stats in pseudo_stats["clients"].items():
        counts = [0.0] * num_classes
        confidence_sums = [0.0] * num_classes
        localization_sums = [0.0] * num_classes
        quality_sums = [0.0] * num_classes

        with Path(stats["box_table"]).open(encoding="utf-8", newline="") as f:
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


def attach_selected_box_tables(args: argparse.Namespace, pseudo_stats: dict[str, Any], round_idx: int) -> dict[str, Any]:
    """Point each client stat to the expert-choice selected boxes used for training."""
    tag = round_tag(round_idx)
    selected_path = args.workspace_root / "stats" / f"05_{tag}_expert_choice_boxes.csv"
    rows = read_csv(selected_path)
    by_client: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_client.setdefault(row["client"], []).append(row)

    fields = [
        "round",
        "client",
        "expert_id",
        "image",
        "source_pseudo_image",
        "source_image",
        "class_id",
        "conf",
        "stability",
        "score",
        "line_index",
        "xyxy",
    ]
    for client_tag, stats in pseudo_stats.get("clients", {}).items():
        selected = by_client.get(client_tag, [])
        if not selected:
            continue
        box_table = args.workspace_root / "stats" / f"10_{tag}_{client_tag}_specialist_selected_boxes.csv"
        write_csv(box_table, selected, fields)
        stats["box_table"] = str(box_table.resolve())
        stats["specialist_selection"] = "expert_choice_selected_boxes"
    return pseudo_stats


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


def save_record(
    records: list[dict[str, str]],
    label: str,
    path: Path,
    kind: str,
    *,
    condition: str,
    phase: int | str = "",
    round_idx: int | str = "",
    client: str = "",
    variant: str = "",
) -> None:
    records.append(
        {
            "condition": condition,
            "label": label,
            "kind": kind,
            "phase": str(phase),
            "round": str(round_idx),
            "client": client,
            "variant": variant,
            "path": str(path.resolve()),
        }
    )


def write_checkpoint_records(path: Path, records: list[dict[str, str]]) -> None:
    write_csv(path, records, ["condition", "label", "kind", "phase", "round", "client", "variant", "path"])


def source_repair_baseline_record(args: argparse.Namespace) -> dict[str, str] | None:
    source = args.source_workspace.expanduser().resolve()
    candidates: list[dict[str, str]] = []
    for name in ("08_eval_checkpoints.csv", "08_all_checkpoints.csv", "08_repair_baseline_checkpoints.csv"):
        candidates.extend(read_csv(source / "stats" / name))
    wanted = {
        "warmup_server_repair_final",
        f"repair_baseline_p0_{round_tag(args.source_repair_baseline_rounds)}_server_repair",
    }
    for row in candidates:
        if row.get("label") in wanted and row.get("path") and Path(row["path"]).exists():
            copied = dict(row)
            copied["label"] = "warmup_server_repair_final"
            copied["condition"] = copied.get("condition") or "warmup + server repair from source workspace"
            return copied
    fallback = source / "checkpoints" / f"repair_baseline_p0_{round_tag(args.source_repair_baseline_rounds)}_server_repair.pt"
    if fallback.exists():
        return {
            "condition": "warmup + server repair from source workspace",
            "label": "warmup_server_repair_final",
            "kind": "server_repair",
            "phase": "0",
            "round": str(args.source_repair_baseline_rounds),
            "client": "",
            "variant": "source_workspace",
            "path": str(fallback.resolve()),
        }
    return None


def run_server_repair_round(
    setup,
    fedsto,
    current: Path,
    args: argparse.Namespace,
    *,
    condition: str,
    phase: int,
    round_idx: int,
    train_scope: str,
    port_offset: int,
) -> tuple[list[dict[str, str]], Path, int]:
    if args.server_repair_epochs <= 0:
        return [], current, port_offset
    tag = round_tag(round_idx)
    repair_start = fedsto.GLOBAL_DIR / f"sdn10_{condition}_p{phase}_{tag}_server_repair_start.pt"
    repair = args.workspace_root / "checkpoints" / f"{condition}_p{phase}_{tag}_server_repair.pt"
    records: list[dict[str, str]] = []

    if not args.dry_run and not fedsto.checkpoint_matches_protocol(repair_start, PROTOCOL_VERSION):
        fedsto.make_start_checkpoint(current, repair_start, protocol=PROTOCOL_VERSION, stage=f"{condition}_p{phase}_{tag}_server_repair_start")

    if not reusable_checkpoint(fedsto, repair, args):
        cfg = write_server_repair_config(
            setup,
            condition=condition,
            phase=phase,
            round_idx=round_idx,
            start=repair_start,
            train_scope=train_scope,
            args=args,
        )
        raw = run_train(
            setup,
            fedsto,
            cfg,
            dry_run=args.dry_run,
            gpus=args.gpus,
            master_port=args.master_port + port_offset,
        )
        port_offset += 1
        if not args.dry_run:
            fedsto.mark_checkpoint_protocol(raw, PROTOCOL_VERSION, f"{condition}_p{phase}_{tag}_server_repair_raw")
            fedsto.make_start_checkpoint(raw, repair, protocol=PROTOCOL_VERSION, stage=f"{condition}_p{phase}_{tag}_server_repair")
            pl03.cleanup_training_artifacts(raw, repair_start)

    save_record(
        records,
        f"{condition}_p{phase}_{tag}_server_repair",
        repair,
        "server_repair",
        condition=condition,
        phase=phase,
        round_idx=round_idx,
    )
    return records, repair, port_offset


def run_repair_baseline(
    setup,
    fedsto,
    warmup: Path,
    args: argparse.Namespace,
    *,
    port_offset: int,
) -> tuple[list[dict[str, str]], Path, int]:
    records: list[dict[str, str]] = []
    current = warmup
    for idx in range(1, args.repair_baseline_rounds + 1):
        print(f"\n=== repair baseline {round_tag(idx)} ===")
        round_records, current, port_offset = run_server_repair_round(
            setup,
            fedsto,
            current,
            args,
            condition="repair_baseline",
            phase=0,
            round_idx=idx,
            train_scope="all",
            port_offset=port_offset,
        )
        records.extend(round_records)
        write_checkpoint_records(args.workspace_root / "stats" / "10_repair_baseline_checkpoints.csv", records)
    return records, current, port_offset


def run_dqa_round(
    setup,
    fedsto,
    current: Path,
    args: argparse.Namespace,
    clients: list[dict[str, Any]],
    *,
    phase: int,
    round_idx: int,
    train_scope: str,
    client_epochs: int,
    client_lr: float,
    source_repeat: int,
    pseudo_repeat: int,
    loss_box: float,
    repair_train_scope: str,
    port_offset: int,
    load_bias_state: dict[str, list[float]],
) -> tuple[list[dict[str, str]], Path, dict[str, Any], dict[str, list[float]], int]:
    tag = round_tag(round_idx)
    print(f"\n=== tighter_specialist_latent_dqamox phase {phase} {tag}: pseudo labels ===")
    raw_pseudo_stats = pl03.generate_round_pseudo_labels(setup, current, args, clients, round_idx)
    pseudo_stats, next_load_bias_state = ec05.apply_expert_choice_selection(
        setup,
        args,
        raw_pseudo_stats,
        round_idx,
        load_bias_state,
    )
    pseudo_stats = attach_selected_box_tables(args, pseudo_stats, round_idx)
    records: list[dict[str, str]] = []
    local_paths: list[Path] = []

    for client in clients:
        client_tag = f"client{client['id']}_{client['weather']}"
        start = fedsto.CLIENT_STATE_DIR / f"sdn10_p{phase}_{tag}_{client_tag}_start.pt"
        final = args.workspace_root / "checkpoints" / f"latent_dqamox_p{phase}_{tag}_{client_tag}.pt"
        if not args.dry_run and not fedsto.checkpoint_matches_protocol(start, PROTOCOL_VERSION):
            fedsto.make_start_checkpoint(current, start, protocol=PROTOCOL_VERSION, stage=f"latent_dqamox_p{phase}_{tag}_{client_tag}_start")

        if not reusable_checkpoint(fedsto, final, args):
            cfg = write_client_config(
                setup,
                condition="latent_dqamox",
                phase=phase,
                round_idx=round_idx,
                client=client,
                start=start,
                train_scope=train_scope,
                epochs=client_epochs,
                lr=client_lr,
                source_repeat=source_repeat,
                pseudo_repeat=pseudo_repeat,
                loss_box=loss_box,
                pseudo_list=Path(pseudo_stats["clients"][client_tag]["train_list"]),
                args=args,
            )
            raw = run_train(
                setup,
                fedsto,
                cfg,
                dry_run=args.dry_run,
                gpus=args.gpus,
                master_port=args.master_port + port_offset,
            )
            port_offset += 1
            if not args.dry_run:
                fedsto.mark_checkpoint_protocol(raw, PROTOCOL_VERSION, f"latent_dqamox_p{phase}_{tag}_{client_tag}_raw")
                fedsto.make_start_checkpoint(raw, final, protocol=PROTOCOL_VERSION, stage=f"latent_dqamox_p{phase}_{tag}_{client_tag}")
                pl03.cleanup_training_artifacts(raw, start)

        local_paths.append(final)
        save_record(
            records,
            f"latent_dqamox_p{phase}_{tag}_{client_tag}",
            final,
            "client",
            condition="latent_dqamox",
            phase=phase,
            round_idx=round_idx,
            client=client_tag,
            variant=f"scope={train_scope}:lr={client_lr}:source={source_repeat}:pseudo={pseudo_repeat}:box={loss_box}",
        )

    aggregate = args.workspace_root / "checkpoints" / f"latent_dqamox_p{phase}_{tag}_dqa_aggregate.pt"
    state_path = args.workspace_root / "stats" / "10_latent_dqamox_dqa_state.json"
    if not args.dry_run and not reusable_checkpoint(fedsto, aggregate, args):
        stats = pseudo_stats_to_dqa_stats(pseudo_stats, num_classes=len(setup.BDD_NAMES))
        _, dqa_state = dqa_v2.aggregate_checkpoints(
            client_checkpoints=local_paths,
            server_checkpoint=current,
            output_checkpoint=aggregate,
            stats=stats,
            state_path=state_path,
            config=dqa_config(args, len(setup.BDD_NAMES)),
            repo_root=REPO_ROOT,
        )
        (args.workspace_root / "stats" / f"10_p{phase}_{tag}_dqa_state_snapshot.json").write_text(
            json.dumps(dqa_state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        fedsto.mark_checkpoint_protocol(aggregate, PROTOCOL_VERSION, f"latent_dqamox_p{phase}_{tag}_dqa_aggregate")

    save_record(
        records,
        f"latent_dqamox_p{phase}_{tag}_dqa_aggregate",
        aggregate,
        "aggregate",
        condition="latent_dqamox",
        phase=phase,
        round_idx=round_idx,
    )

    repair_records, repaired, port_offset = run_server_repair_round(
        setup,
        fedsto,
        aggregate,
        args,
        condition="latent_dqamox",
        phase=phase,
        round_idx=round_idx,
        train_scope=repair_train_scope,
        port_offset=port_offset,
    )
    records.extend(repair_records)
    return records, repaired, pseudo_stats, next_load_bias_state, port_offset


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


def write_final_metrics(args: argparse.Namespace, eval_records: list[dict[str, str]]) -> list[dict[str, Any]]:
    summary_path = args.workspace_root / "validation_reports" / "paper_protocol_eval_summary.csv"
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
        "latent_dqamox_final_aggregate",
        "latent_dqamox_final_repair",
    ]
    metric_rows: list[dict[str, Any]] = []
    for label in ordered:
        total = by_label_total.get(label)
        if not total:
            continue
        m50 = as_float(total.get("map50"))
        m95 = as_float(total.get("map50_95"))
        metric_rows.append(
            {
                "checkpoint_label": label,
                "condition": {
                    "warmup_global": "warmup",
                    "warmup_server_repair_final": "warmup + server repair",
                    "latent_dqamox_final_aggregate": "warmup + tighter-specialist latent DQA-MoX aggregate",
                    "latent_dqamox_final_repair": "warmup + tighter-specialist latent DQA-MoX + server repair",
                }.get(label, label),
                "kind": meta.get(label, {}).get("kind", ""),
                "phase": meta.get(label, {}).get("phase", ""),
                "round": meta.get(label, {}).get("round", ""),
                "precision": total.get("precision", ""),
                "recall": total.get("recall", ""),
                "map50": "" if m50 is None else f"{m50:.6f}",
                "map50_95": "" if m95 is None else f"{m95:.6f}",
                "gain_vs_warmup_map50_95": "" if m95 is None or warm_m95 is None else f"{m95 - warm_m95:.6f}",
                "delta_vs_server_repair_map50_95": "" if m95 is None or repair_m95 is None else f"{m95 - repair_m95:.6f}",
                **split_gap_metrics(by_label_split, label),
            }
        )

    fields = [
        "checkpoint_label",
        "condition",
        "kind",
        "phase",
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
    write_csv(args.workspace_root / "stats" / "10_tighter_specialist_latent_dqamox_final_metrics.csv", metric_rows, fields)

    split_rows: list[dict[str, Any]] = []
    for row in rows:
        label = row["checkpoint_label"]
        if label not in meta:
            continue
        split_rows.append(
            {
                "checkpoint_label": label,
                "condition": meta[label]["condition"],
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
        args.workspace_root / "stats" / "10_tighter_specialist_latent_dqamox_split_metrics.csv",
        split_rows,
        ["checkpoint_label", "condition", "split", "images", "labels", "precision", "recall", "map50", "map50_95"],
    )
    return metric_rows


def run_evaluation(args: argparse.Namespace, eval_records: list[dict[str, str]]) -> None:
    base01_0.run_evaluation(args, eval_records)


def write_report(args: argparse.Namespace, metrics: list[dict[str, Any]], run_manifest: dict[str, Any]) -> Path:
    path = args.workspace_root / "10_tighter_specialist_latent_dqamox_report.md"
    lines = [
        "# 10 Tighter-Specialist Latent DQA-MoX Report",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- protocol: `{PROTOCOL_VERSION}`",
        f"- workspace: `{args.workspace_root.resolve()}`",
        f"- experts: K={args.num_experts}, top_k={args.top_k}, temperature={args.router_temperature}",
        f"- schedule: warmup {args.warmup_epochs} epochs, repair baseline {args.repair_baseline_rounds} rounds, phase1 {args.phase1_rounds} rounds, phase2 {args.phase2_rounds} rounds",
        "",
        "## Metrics",
        "",
        "| condition | mAP50 | mAP50:95 | delta vs repair | worst split | worst mAP50:95 |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in metrics:
        lines.append(
            "| {condition} | {map50} | {map50_95} | {delta} | {worst} | {worst_m95} |".format(
                condition=row.get("condition", ""),
                map50=row.get("map50", ""),
                map50_95=row.get("map50_95", ""),
                delta=row.get("delta_vs_server_repair_map50_95", ""),
                worst=row.get("worst_split", ""),
                worst_m95=row.get("worst_split_map50_95", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Hooks",
            "",
            "- This run is not client-id MoE and not hand-labeled output MoE. Expert identities are latent and learned by the router.",
            "- The new part is pseudoGT selection: expert-choice buckets reduce class imbalance before client training and before DQA statistics.",
            "- DQA remains in the selected pseudoGT statistics and classwise server-anchored aggregation; MoE remains inside the detector head.",
            "- The key comparison is `latent_dqamox_final_repair` vs `warmup_server_repair_final` on total and each scene/day-night split.",
            "",
            "## Run Manifest",
            "",
            "```json",
            json.dumps(run_manifest, indent=2, ensure_ascii=False)[:6000],
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def estimated_seconds(args: argparse.Namespace) -> float:
    warmup = args.estimated_warmup_minutes
    repair = args.repair_baseline_rounds * args.estimated_repair_round_minutes
    dqa = args.phase1_rounds * args.estimated_phase1_round_minutes + args.phase2_rounds * args.estimated_phase2_round_minutes
    eval_minutes = args.estimated_eval_minutes if args.evaluate else 0.0
    return (warmup + repair + dqa + eval_minutes) * 60.0


def progress_factory(args: argparse.Namespace, total: int):
    if args.no_progress:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:  # noqa: BLE001
        return None
    return tqdm(total=total, desc="10 tighter-specialist latent DQA-MoX", unit="step")


def run(args: argparse.Namespace) -> None:
    args.workspace_root = args.workspace_root.expanduser().resolve()
    setup, fedsto, manifest, clients = configure_workspace(args)
    args.gpus = fedsto.resolve_gpus(args.gpus)
    if not args.dry_run:
        fedsto.check_runtime_dependencies()

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "workspace": str(args.workspace_root.resolve()),
        "estimated_runtime": seconds_to_hms(estimated_seconds(args)),
        "architecture": {
            "head": "LatentMoEYoloV5",
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "router_temperature": args.router_temperature,
            "moe_scale": args.moe_scale,
            "router_balance_weight": args.router_balance_weight,
            "router_entropy_weight": args.router_entropy_weight,
            "expert_semantics": "latent_anonymous_learned",
        },
        "schedule": {
            "warmup_epochs": args.warmup_epochs,
            "repair_baseline_rounds": args.repair_baseline_rounds,
            "phase1_rounds": args.phase1_rounds,
            "phase2_rounds": args.phase2_rounds,
            "phase1_train_scope": args.phase1_train_scope,
            "phase2_train_scope": args.phase2_train_scope,
        },
        "pseudo_selection": {
            "method": "expert_choice_tighter_specialist_balanced",
            "expert_count": args.expert_count,
            "keep_fraction": args.expert_keep_fraction,
            "max_class_fraction": args.expert_max_class_fraction,
            "load_bias_strength": args.load_bias_strength,
        },
        "server": manifest.get("server"),
        "clients": clients,
    }
    (args.workspace_root / "stats" / "10_tighter_specialist_latent_dqamox_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.setup_only:
        return

    total_steps = 1 + args.repair_baseline_rounds + args.phase1_rounds + args.phase2_rounds + (1 if args.evaluate else 0)
    progress = progress_factory(args, total_steps)
    start_time = time.monotonic()
    next_progress_notify = args.notify_first_progress_hours * 3600.0 if args.notify_first_progress_hours > 0 else None
    port_offset = 0

    def maybe_notify_progress(stage: str, completed: int, checkpoint: Path | None = None) -> None:
        nonlocal next_progress_notify
        if next_progress_notify is None or not (args.notify or args.notify_progress):
            return
        elapsed = time.monotonic() - start_time
        if elapsed < next_progress_notify:
            return
        avg = elapsed / max(completed, 1)
        eta = avg * max(total_steps - completed, 0)
        extra = {
            "stage": stage,
            "completed_steps": completed,
            "total_steps": total_steps,
            "elapsed_hms": seconds_to_hms(elapsed),
            "eta_hms": seconds_to_hms(eta),
        }
        if checkpoint is not None:
            extra["checkpoint"] = str(checkpoint.resolve())
        notify(
            args,
            f"DQA 10 progress: {stage}, ETA {seconds_to_hms(eta)}.",
            title="DQA 10 progress",
            status="running",
            extra_context=extra,
        )
        interval = args.notify_progress_interval_hours if args.notify_progress_interval_hours > 0 else args.notify_first_progress_hours
        next_progress_notify += max(interval, 0.1) * 3600.0

    records: list[dict[str, str]] = []
    warmup, port_offset = train_or_load_warmup(setup, fedsto, args, port_offset)
    save_record(records, "warmup_global", warmup, "warmup", condition="warmup")
    if progress is not None:
        progress.update(1)
    maybe_notify_progress("warmup_done", 1, warmup)

    if args.repair_baseline_rounds > 0:
        repair_records, _, port_offset = run_repair_baseline(setup, fedsto, warmup, args, port_offset=port_offset)
        records.extend(repair_records)
        if progress is not None:
            progress.update(args.repair_baseline_rounds)
    completed_steps = 1 + args.repair_baseline_rounds
    maybe_notify_progress("repair_baseline_done", completed_steps)

    dqa_current = warmup
    dqa_records: list[dict[str, str]] = []
    pseudo_history: list[dict[str, Any]] = []
    load_bias_state: dict[str, list[float]] = {}
    for idx in range(1, args.phase1_rounds + 1):
        round_records, dqa_current, pseudo_stats, load_bias_state, port_offset = run_dqa_round(
            setup,
            fedsto,
            dqa_current,
            args,
            clients,
            phase=1,
            round_idx=idx,
            train_scope=args.phase1_train_scope,
            client_epochs=args.phase1_client_epochs,
            client_lr=args.phase1_client_lr,
            source_repeat=args.phase1_source_repeat,
            pseudo_repeat=args.phase1_pseudo_repeat,
            loss_box=args.phase1_loss_box,
            repair_train_scope=args.phase1_repair_train_scope,
            port_offset=port_offset,
            load_bias_state=load_bias_state,
        )
        dqa_records.extend(round_records)
        pseudo_history.append({"phase": 1, "round": idx, "stats": pseudo_stats})
        write_checkpoint_records(args.workspace_root / "stats" / "10_latent_dqamox_checkpoints.csv", dqa_records)
        if progress is not None:
            elapsed = time.monotonic() - start_time
            done = 1 + args.repair_baseline_rounds + idx
            eta = elapsed / max(done, 1) * max(total_steps - done, 0)
            progress.set_postfix(stage="phase1", round=idx, eta=seconds_to_hms(eta))
            progress.update(1)
        maybe_notify_progress("phase1", done, dqa_current)

    for idx in range(1, args.phase2_rounds + 1):
        global_round_idx = args.phase1_rounds + idx
        round_records, dqa_current, pseudo_stats, load_bias_state, port_offset = run_dqa_round(
            setup,
            fedsto,
            dqa_current,
            args,
            clients,
            phase=2,
            round_idx=global_round_idx,
            train_scope=args.phase2_train_scope,
            client_epochs=args.phase2_client_epochs,
            client_lr=args.phase2_client_lr,
            source_repeat=args.phase2_source_repeat,
            pseudo_repeat=args.phase2_pseudo_repeat,
            loss_box=args.phase2_loss_box,
            repair_train_scope=args.phase2_repair_train_scope,
            port_offset=port_offset,
            load_bias_state=load_bias_state,
        )
        dqa_records.extend(round_records)
        pseudo_history.append({"phase": 2, "round": global_round_idx, "phase_local_round": idx, "stats": pseudo_stats})
        write_checkpoint_records(args.workspace_root / "stats" / "10_latent_dqamox_checkpoints.csv", dqa_records)
        if progress is not None:
            progress.set_postfix(stage="phase2", round=global_round_idx)
            progress.update(1)
        completed_steps = 1 + args.repair_baseline_rounds + args.phase1_rounds + idx
        maybe_notify_progress("phase2", completed_steps, dqa_current)

    records.extend(dqa_records)
    write_checkpoint_records(args.workspace_root / "stats" / "10_all_checkpoints.csv", records)

    final_phase = 2 if args.phase2_rounds > 0 else 1
    final_round = args.phase1_rounds + args.phase2_rounds if args.phase2_rounds > 0 else args.phase1_rounds
    final_aggregate_label = f"latent_dqamox_p{final_phase}_{round_tag(final_round)}_dqa_aggregate"
    final_repair_label = f"latent_dqamox_p{final_phase}_{round_tag(final_round)}_server_repair"

    by_label = {row["label"]: row for row in records}
    repair_final_label = f"repair_baseline_p0_{round_tag(args.repair_baseline_rounds)}_server_repair"
    repair_eval_record = (
        {**by_label[repair_final_label], "label": "warmup_server_repair_final"}
        if repair_final_label in by_label
        else source_repair_baseline_record(args)
    )
    eval_records = [
        by_label["warmup_global"],
        {**by_label[final_aggregate_label], "label": "latent_dqamox_final_aggregate"},
        {**by_label[final_repair_label], "label": "latent_dqamox_final_repair"},
    ]
    if repair_eval_record is not None:
        eval_records.insert(1, repair_eval_record)
    write_checkpoint_records(args.workspace_root / "stats" / "10_eval_checkpoints.csv", eval_records)

    run_manifest = {
        **payload,
        "actual_runtime_seconds": time.monotonic() - start_time,
        "actual_runtime_hms": seconds_to_hms(time.monotonic() - start_time),
        "records": records,
        "eval_records": eval_records,
        "pseudo_history": pseudo_history,
    }
    (args.workspace_root / "stats" / "10_tighter_specialist_latent_dqamox_run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metrics: list[dict[str, Any]] = []
    if args.evaluate:
        run_evaluation(args, eval_records)
        metrics = write_final_metrics(args, eval_records)
        if progress is not None:
            progress.update(1)
    if progress is not None:
        progress.close()

    report = write_report(args, metrics, run_manifest)
    print(f"Saved report: {report}")
    if metrics:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


def notify(
    args: argparse.Namespace,
    message: str,
    *,
    title: str,
    status: str | None = None,
    error: str | None = None,
    extra_context: dict[str, Any] | None = None,
) -> None:
    try:
        from notebook_notify import notify_discord

        context: dict[str, Any] = {
            "workspace": str(args.workspace_root.expanduser().resolve()),
            "estimated_runtime": seconds_to_hms(estimated_seconds(args)),
            "experts": args.num_experts,
            "phase1_rounds": args.phase1_rounds,
            "phase2_rounds": args.phase2_rounds,
        }
        if status:
            context["status"] = status
        if error:
            context["error"] = error[:500]
        if extra_context:
            context.update(extra_context)
        metrics_path = args.workspace_root.expanduser().resolve() / "stats" / "10_tighter_specialist_latent_dqamox_final_metrics.csv"
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
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT / "output" / "10_tighter_specialist_latent_dqamox")
    parser.add_argument("--source-workspace", type=Path, default=PROJECT_ROOT / "output" / "08_full_latent_dqamox_from_warmup")
    parser.add_argument("--source-repair-baseline-rounds", type=int, default=30)
    parser.add_argument("--warmup-checkpoint", type=Path, default=None)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None)
    parser.add_argument("--skip-warmup-training", action="store_true")
    parser.add_argument("--force-pretrained", action="store_true")
    parser.add_argument("--client-limit", type=int, default=1500)
    parser.add_argument("--clients", default="all")
    parser.add_argument("--warmup-epochs", type=int, default=50)
    parser.add_argument("--warmup-lr", type=float, default=0.01)
    parser.add_argument("--warmup-lrf", type=float, default=0.2)
    parser.add_argument("--repair-baseline-rounds", type=int, default=0)
    parser.add_argument("--phase1-rounds", type=int, default=12)
    parser.add_argument("--phase2-rounds", type=int, default=1)
    parser.add_argument("--phase1-train-scope", choices=["neck_head", "all"], default="neck_head")
    parser.add_argument("--phase1-repair-train-scope", choices=["neck_head", "all"], default="neck_head")
    parser.add_argument("--phase1-client-epochs", type=int, default=1)
    parser.add_argument("--phase1-client-lr", type=float, default=0.0006)
    parser.add_argument("--phase1-source-repeat", type=int, default=3)
    parser.add_argument("--phase1-pseudo-repeat", type=int, default=1)
    parser.add_argument("--phase1-loss-box", type=float, default=0.0015)
    parser.add_argument("--phase2-train-scope", choices=["neck_head", "all"], default="all")
    parser.add_argument("--phase2-repair-train-scope", choices=["neck_head", "all"], default="all")
    parser.add_argument("--phase2-client-epochs", type=int, default=1)
    parser.add_argument("--phase2-client-lr", type=float, default=0.0003)
    parser.add_argument("--phase2-source-repeat", type=int, default=2)
    parser.add_argument("--phase2-pseudo-repeat", type=int, default=1)
    parser.add_argument("--phase2-loss-box", type=float, default=0.004)
    parser.add_argument("--server-repair-epochs", type=int, default=1)
    parser.add_argument("--server-repair-lr", type=float, default=0.0007)
    parser.add_argument("--server-repair-loss-box", type=float, default=0.05)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--moe-scale", type=float, default=1.0)
    parser.add_argument("--router-balance-weight", type=float, default=0.01)
    parser.add_argument("--router-entropy-weight", type=float, default=0.001)
    parser.add_argument("--orthogonal-weight", type=float, default=1e-4)
    parser.add_argument("--dqa-count-ema", type=float, default=0.80)
    parser.add_argument("--dqa-quality-ema", type=float, default=0.65)
    parser.add_argument("--dqa-alpha-ema", type=float, default=0.55)
    parser.add_argument("--dqa-temperature", type=float, default=0.70)
    parser.add_argument("--dqa-uniform-mix", type=float, default=0.10)
    parser.add_argument("--dqa-classwise-blend", type=float, default=0.35)
    parser.add_argument("--dqa-stability-lambda", type=float, default=0.35)
    parser.add_argument("--dqa-min-effective-count", type=float, default=20.0)
    parser.add_argument("--dqa-min-quality", type=float, default=0.05)
    parser.add_argument("--dqa-server-anchor", type=float, default=0.55)
    parser.add_argument("--dqa-min-server-alpha", type=float, default=0.50)
    parser.add_argument("--dqa-residual-blend", type=float, default=0.15)
    parser.add_argument("--expert-count", type=int, default=4)
    parser.add_argument("--expert-keep-fraction", type=float, default=0.55)
    parser.add_argument("--expert-max-class-fraction", type=float, default=0.24)
    parser.add_argument("--load-bias-strength", type=float, default=0.45)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=33481)
    parser.add_argument("--device", default="")
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
    parser.add_argument("--estimated-warmup-minutes", type=float, default=180.0)
    parser.add_argument("--estimated-repair-round-minutes", type=float, default=4.0)
    parser.add_argument("--estimated-phase1-round-minutes", type=float, default=19.0)
    parser.add_argument("--estimated-phase2-round-minutes", type=float, default=24.0)
    parser.add_argument("--estimated-eval-minutes", type=float, default=55.0)
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--notify-start", action="store_true")
    parser.add_argument("--notify-end", action="store_true")
    parser.add_argument("--notify-progress", action="store_true")
    parser.add_argument("--notify-first-progress-hours", type=float, default=0.0)
    parser.add_argument("--notify-progress-interval-hours", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    do_start_notify = args.notify or args.notify_start
    do_end_notify = args.notify or args.notify_end
    if do_start_notify:
        notify(args, "Scene-Daynight DQA 10 tighter-specialist latent DQA-MoX started.", title="DQA 10 start")

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
                f"Scene-Daynight DQA 10 tighter-specialist latent DQA-MoX finished with status={status}.",
                title="DQA 10 finish",
                status=status,
                error=error,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
