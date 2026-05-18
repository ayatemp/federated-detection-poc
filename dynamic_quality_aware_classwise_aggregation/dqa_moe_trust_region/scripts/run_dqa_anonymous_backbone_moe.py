#!/usr/bin/env python3
"""FedSTO schedule with DQA-conditioned anonymous upcycled MoE adapters.

This runner keeps the FedSTO reproduction loop, but upcycles a dense YOLO
checkpoint into anonymous backbone/neck adapter MoE plus an optional head MoE
before the selected phase.  Expert slots stay anonymous: the config provides
routing context, not hand-assigned domain expert names.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import gc
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
FEDSTO_SCRIPT_ROOT = REPO_ROOT / "FedSTO" / "scripts"
if str(FEDSTO_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(FEDSTO_SCRIPT_ROOT))

import run_fedsto_paper_reproduction as fedsto  # noqa: E402
import setup_fedsto_paper_reproduction as setup  # noqa: E402


BASE_PROTOCOL_VERSION = "fedsto_public_spec_broadcast_reset_ema_v2+dqa_anonymous_backbone_moe_v4"
PROTOCOL_VERSION = BASE_PROTOCOL_VERSION
DEFAULT_WORKSPACE = (
    REPO_ROOT
    / "dynamic_quality_aware_classwise_aggregation"
    / "dqa_moe_trust_region"
    / "output"
    / "03_dqa_anonymous_backbone_moe"
)
SUMMARY_PATH = DEFAULT_WORKSPACE / "anonymous_backbone_moe_round_summary.csv"
ROUTER_DIAGNOSTIC_PATH = DEFAULT_WORKSPACE / "anonymous_backbone_moe_router_diagnostics.csv"
EXPERT_AGGREGATION_PATH = DEFAULT_WORKSPACE / "expertwise_aggregation_diagnostics.csv"


def apply_workspace_root(workspace_root: Path) -> Path:
    global SUMMARY_PATH, ROUTER_DIAGNOSTIC_PATH, EXPERT_AGGREGATION_PATH
    workspace_root = fedsto.apply_workspace_root(workspace_root)
    SUMMARY_PATH = workspace_root / "anonymous_backbone_moe_round_summary.csv"
    ROUTER_DIAGNOSTIC_PATH = workspace_root / "anonymous_backbone_moe_router_diagnostics.csv"
    EXPERT_AGGREGATION_PATH = workspace_root / "expertwise_aggregation_diagnostics.csv"
    return workspace_root


def read_last_results_row(run_name: str) -> dict[str, str]:
    path = setup.RUN_ROOT / run_name / "results.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))
    return rows[-1] if rows else {}


def metric(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def metric_summary(row: dict[str, Any]) -> dict[str, float]:
    return {
        "precision": metric(row, "metrics/precision"),
        "recall": metric(row, "metrics/recall"),
        "map50": metric(row, "metrics/mAP_0.5"),
        "map50_95": metric(row, "metrics/mAP_0.5:0.95"),
    }


def notify(args: argparse.Namespace, title: str, message: str) -> None:
    if not args.discord:
        return
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from notebook_notify import notify_discord

        notify_discord(message, title=title, context={"workspace": str(args.workspace_root)}, fail_silently=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}")


def append_summary(row: dict[str, Any]) -> None:
    flat = {
        "phase": row["phase"],
        "round": row["round"],
        "global": row["global"],
        "moe_enabled": row["moe_enabled"],
        "server_map50": row["server_metrics"].get("map50"),
        "server_map50_95": row["server_metrics"].get("map50_95"),
        "server_precision": row["server_metrics"].get("precision"),
        "server_recall": row["server_metrics"].get("recall"),
        "router_entropy": row.get("router_diagnostics", {}).get("mean_entropy"),
        "router_max_prob": row.get("router_diagnostics", {}).get("mean_max_prob"),
        "router_active_experts": row.get("router_diagnostics", {}).get("mean_active_experts"),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = SUMMARY_PATH.exists()
    with SUMMARY_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(flat)


def append_router_diagnostics(row: dict[str, Any]) -> None:
    levels = row.get("levels", {})
    flat = {
        "phase": row["phase"],
        "round": row["round"],
        "checkpoint": row["checkpoint"],
        "split": row["split"],
        "images": row["images"],
        "mean_entropy": row.get("mean_entropy"),
        "mean_max_prob": row.get("mean_max_prob"),
        "mean_active_experts": row.get("mean_active_experts"),
        "levels_json": json.dumps(levels, sort_keys=True),
    }
    ROUTER_DIAGNOSTIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = ROUTER_DIAGNOSTIC_PATH.exists()
    with ROUTER_DIAGNOSTIC_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(flat)


def append_expert_aggregation_diagnostics(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "phase",
        "round",
        "scope",
        "num_clients",
        "quality_weights",
        "client_metrics",
        "mean_expert_entropy",
        "min_expert_entropy",
        "max_expert_weight",
        "updated_tensors",
        "clipped_tensors",
        "sample_expert_weights_json",
    ]
    EXPERT_AGGREGATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = EXPERT_AGGREGATION_PATH.exists()
    with EXPERT_AGGREGATION_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def use_backbone_moe(args: argparse.Namespace, phase: int) -> bool:
    if phase <= 0:
        return bool(getattr(args, "warmup_moe", False))
    return phase >= args.moe_start_phase


def use_head_moe(args: argparse.Namespace, phase: int) -> bool:
    if phase <= 0:
        return bool(getattr(args, "warmup_moe", False)) and bool(args.enable_head_moe)
    return bool(args.enable_head_moe) and phase >= args.head_moe_start_phase


def use_neck_moe(args: argparse.Namespace, phase: int) -> bool:
    if phase <= 0:
        return bool(getattr(args, "warmup_moe", False)) and bool(args.enable_neck_moe)
    return bool(args.enable_neck_moe) and phase >= args.neck_moe_start_phase


def dqa_context(args: argparse.Namespace, *, phase: int, round_idx: int, role: str, client: dict[str, Any] | None) -> list[float]:
    client_id = int(client["id"]) if client else -1
    one_hot = [1.0 if client_id == idx else 0.0 for idx in range(len(setup.CLIENTS))]
    rounds = args.phase1_rounds if phase == 1 else args.phase2_rounds
    progress = round_idx / max(rounds, 1)
    context = [
        phase / 2.0,
        progress,
        1.0 if role == "client" else 0.0,
        1.0 if role == "server" else 0.0,
        *one_hot,
        args.dqa_context_bias,
    ]
    if len(context) < args.moe_context_dim:
        context.extend([0.0] * (args.moe_context_dim - len(context)))
    return context[: args.moe_context_dim]


def inject_backbone_moe(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    *,
    phase: int,
    round_idx: int,
    role: str,
    client: dict[str, Any] | None = None,
) -> None:
    cfg["BackboneMoE"] = {
        "enabled": True,
        "num_experts": args.moe_num_experts,
        "top_k": args.moe_top_k,
        "temperature": args.moe_temperature,
        "scale": args.moe_scale,
        "shared_scale": args.moe_shared_scale,
        "adapter_ratio": args.moe_adapter_ratio,
        "min_channels": args.moe_min_channels,
        "levels": args.moe_levels.split(","),
        "kernels": [int(item) for item in args.moe_kernels.split(",") if item.strip()],
        "context_dim": args.moe_context_dim,
        "context": dqa_context(args, phase=phase, round_idx=round_idx, role=role, client=client),
        "quality_dim": args.moe_quality_dim,
        "router_noise_std": args.moe_router_noise_std,
        "balance_weight": args.moe_balance_weight,
        "entropy_weight": args.moe_entropy_weight,
        "sample_entropy_weight": args.moe_sample_entropy_weight,
        "z_loss_weight": args.moe_z_loss_weight,
        "diversity_weight": args.moe_diversity_weight,
        "threshold_routing": args.moe_threshold_routing,
        "threshold": args.moe_threshold,
        "freeze_bn": args.moe_freeze_bn,
        "routing_mode": args.moe_routing_mode,
        "straight_through": args.moe_straight_through,
        "expert_choice_capacity_factor": args.moe_expert_choice_capacity_factor,
        "dqa_prior_strength": args.moe_dqa_prior_strength,
        "router_init_std": args.moe_router_init_std,
    }


def inject_head_moe(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    *,
    phase: int,
    round_idx: int,
    role: str,
    client: dict[str, Any] | None = None,
) -> None:
    cfg["Model"]["Head"]["name"] = "LatentMoE"
    client_id = int(client["id"]) if client else -1
    cfg["LatentMoE"] = {
        "enabled": True,
        "num_experts": args.head_moe_num_experts,
        "top_k": args.head_moe_top_k,
        "temperature": args.head_moe_temperature,
        "scale": args.head_moe_scale,
        "balance_weight": args.head_moe_balance_weight,
        "entropy_weight": args.head_moe_entropy_weight,
        "sample_entropy_weight": args.head_moe_sample_entropy_weight,
        # This is intentionally optional.  The default keeps experts anonymous;
        # enabling it lets a probe softly bias one expert for a client/domain.
        "specialization_weight": args.head_moe_specialization_weight,
        "specialization_target": client_id if args.head_moe_specialize_by_client and client_id >= 0 else -1,
        "routing_mode": args.head_moe_routing_mode,
        "straight_through": args.head_moe_straight_through,
        "expert_choice_capacity_factor": args.head_moe_expert_choice_capacity_factor,
        "router_init_std": args.head_moe_router_init_std,
    }


def inject_neck_moe(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    *,
    phase: int,
    round_idx: int,
    role: str,
    client: dict[str, Any] | None = None,
) -> None:
    cfg["NeckMoE"] = {
        "enabled": True,
        "num_experts": args.neck_moe_num_experts,
        "top_k": args.neck_moe_top_k,
        "temperature": args.neck_moe_temperature,
        "scale": args.neck_moe_scale,
        "shared_scale": args.neck_moe_shared_scale,
        "adapter_ratio": args.neck_moe_adapter_ratio,
        "min_channels": args.neck_moe_min_channels,
        "levels": args.neck_moe_levels.split(","),
        "kernels": [int(item) for item in args.neck_moe_kernels.split(",") if item.strip()],
        "context_dim": args.moe_context_dim,
        "context": dqa_context(args, phase=phase, round_idx=round_idx, role=role, client=client),
        "quality_dim": args.moe_quality_dim,
        "router_noise_std": args.neck_moe_router_noise_std,
        "balance_weight": args.neck_moe_balance_weight,
        "entropy_weight": args.neck_moe_entropy_weight,
        "sample_entropy_weight": args.neck_moe_sample_entropy_weight,
        "z_loss_weight": args.neck_moe_z_loss_weight,
        "diversity_weight": args.neck_moe_diversity_weight,
        "threshold_routing": args.moe_threshold_routing,
        "threshold": args.moe_threshold,
        "freeze_bn": args.moe_freeze_bn,
        "routing_mode": args.neck_moe_routing_mode,
        "straight_through": args.neck_moe_straight_through,
        "expert_choice_capacity_factor": args.neck_moe_expert_choice_capacity_factor,
        "dqa_prior_strength": args.neck_moe_dqa_prior_strength,
        "router_init_std": args.neck_moe_router_init_std,
    }


def write_runtime_config(
    name: str,
    *,
    target: Path | None,
    weights: Path,
    phase: int,
    role: str,
    round_idx: int,
    client: dict[str, Any] | None,
    args: argparse.Namespace,
) -> Path:
    train_scope = args.phase1_train_scope if phase == 1 else args.phase2_train_scope
    if (
        phase == 2
        and args.phase2_late_train_scope
        and round_idx > max(0, args.phase2_head_unfreeze_after_round)
    ):
        train_scope = args.phase2_late_train_scope
    if role == "server":
        server_scope = args.phase1_server_train_scope if phase == 1 else args.phase2_server_train_scope
        if server_scope:
            train_scope = server_scope
    orthogonal_weight = args.phase1_orthogonal_weight if phase == 1 else args.orthogonal_weight
    batch_size = args.batch_size if phase == 1 else args.phase2_batch_size
    cfg = setup.efficientteacher_config(
        name=name,
        train=setup.LIST_ROOT / "server_cloudy_train.txt",
        val=setup.LIST_ROOT / "server_cloudy_val.txt",
        target=target,
        weights=str(weights.resolve()),
        epochs=1,
        train_scope=train_scope,
        orthogonal_weight=orthogonal_weight,
        batch_size=batch_size,
        workers=args.workers,
        device=getattr(args, "config_device", args.device),
    )
    if role == "server":
        cfg["SSOD"] = {
            "train_domain": False,
            "ema_rate": 0.999,
            "cosine_ema": False,
        }
        if phase >= 2 and args.phase2_server_lr0 > 0:
            cfg.setdefault("hyp", {})["lr0"] = float(args.phase2_server_lr0)
            cfg["hyp"]["warmup_bias_lr"] = min(
                float(cfg["hyp"].get("warmup_bias_lr", args.phase2_server_lr0)),
                float(args.phase2_server_lr0),
            )
    if use_backbone_moe(args, phase):
        inject_backbone_moe(cfg, args, phase=phase, round_idx=round_idx, role=role, client=client)
    if use_neck_moe(args, phase):
        inject_neck_moe(cfg, args, phase=phase, round_idx=round_idx, role=role, client=client)
    if use_head_moe(args, phase):
        inject_head_moe(cfg, args, phase=phase, round_idx=round_idx, role=role, client=client)
    return setup.write_config(f"runtime_phase{phase}_{role}_round{round_idx}_{name}.yaml", cfg)


def write_warmup_config(
    name: str,
    *,
    weights: Path,
    args: argparse.Namespace,
) -> Path:
    cfg = setup.efficientteacher_config(
        name=name,
        train=setup.LIST_ROOT / "server_cloudy_train.txt",
        val=setup.LIST_ROOT / "server_cloudy_val.txt",
        target=None,
        weights=str(weights.resolve()),
        epochs=args.warmup_epochs,
        train_scope=args.warmup_train_scope,
        batch_size=args.batch_size,
        workers=args.workers,
        device=getattr(args, "config_device", args.device),
    )
    if use_backbone_moe(args, 0):
        inject_backbone_moe(cfg, args, phase=0, round_idx=0, role="server", client=None)
    if use_neck_moe(args, 0):
        inject_neck_moe(cfg, args, phase=0, round_idx=0, role="server", client=None)
    if use_head_moe(args, 0):
        inject_head_moe(cfg, args, phase=0, round_idx=0, role="server", client=None)
    if args.warmup_lr0 > 0:
        cfg.setdefault("hyp", {})["lr0"] = float(args.warmup_lr0)
    if args.warmup_bias_lr > 0:
        cfg.setdefault("hyp", {})["warmup_bias_lr"] = float(args.warmup_bias_lr)
    if args.warmup_hyp_warmup_epochs >= 0:
        cfg.setdefault("hyp", {})["warmup_epochs"] = int(args.warmup_hyp_warmup_epochs)
    return setup.write_config(f"{name}.yaml", cfg)


def checkpoint_has_backbone_moe(path: Path) -> bool:
    if not fedsto.checkpoint_present(path):
        return False
    try:
        state = fedsto._state_dict(fedsto._load(path), "model")
    except Exception:
        return False
    return any(key.startswith("backbone.adapter_moe.") for key in state)


def checkpoint_has_head_moe(path: Path) -> bool:
    if not fedsto.checkpoint_present(path):
        return False
    try:
        state = fedsto._state_dict(fedsto._load(path), "model")
    except Exception:
        return False
    return any(key.startswith("head.router.") or key.startswith("head.expert_m.") for key in state)


def checkpoint_has_neck_moe(path: Path) -> bool:
    if not fedsto.checkpoint_present(path):
        return False
    try:
        state = fedsto._state_dict(fedsto._load(path), "model")
    except Exception:
        return False
    return any(key.startswith("neck.adapter_moe.") for key in state)


def upcycle_checkpoint(
    source: Path,
    out: Path,
    cfg_path: Path,
    *,
    stage: str,
    require_head_moe: bool = False,
    require_neck_moe: bool = False,
) -> Path:
    if (
        checkpoint_has_backbone_moe(out)
        and fedsto.checkpoint_protocol(out) == PROTOCOL_VERSION
        and (not require_head_moe or checkpoint_has_head_moe(out))
        and (not require_neck_moe or checkpoint_has_neck_moe(out))
    ):
        print(f"Reusing upcycled anonymous MoE checkpoint: {out}")
        return out

    fedsto.ensure_efficientteacher_import_path()
    from configs.defaults import get_cfg
    from models.detector.yolo import Model
    from utils.fedsto_regularization import clear_latent_moe_router_cache

    cfg = get_cfg()
    cfg.merge_from_file(str(cfg_path))
    cfg.freeze()

    base = copy.deepcopy(fedsto._load(source))
    model = Model(cfg).cpu().float()
    source_state = fedsto._state_dict(base, "model")
    target_state = model.state_dict()
    compatible = {
        key: value.float()
        for key, value in source_state.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    }
    model.load_state_dict(compatible, strict=False)
    clear_latent_moe_router_cache(model)
    print(f"Upcycled dense checkpoint into anonymous MoE model: loaded {len(compatible)}/{len(target_state)} tensors")

    base["model"] = model.half()
    base["ema"] = copy.deepcopy(model).half()
    clear_latent_moe_router_cache(base["model"])
    clear_latent_moe_router_cache(base["ema"])
    base["epoch"] = -1
    base["optimizer"] = None
    base["updates"] = 0
    base["fedsto_protocol"] = PROTOCOL_VERSION
    base["fedsto_stage"] = stage
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, out)
    return out


def _router_diagnostic_list(args: argparse.Namespace) -> Path:
    split = args.router_diagnostic_split
    paper_list = setup.LIST_ROOT / f"paper_eval_{split}_val.txt"
    if paper_list.exists():
        return paper_list
    return setup.LIST_ROOT / "server_cloudy_val.txt"


def _load_router_diagnostic_batch(paths: list[Path], image_size: int, device: torch.device) -> torch.Tensor:
    import cv2
    import numpy as np
    from utils.augmentations import letterbox

    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = letterbox(image, image_size, stride=32, auto=False)[0]
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        images.append(torch.from_numpy(image))
    if not images:
        return torch.empty(0, 3, image_size, image_size, device=device)
    return torch.stack(images).to(device=device).float() / 255.0


def _router_usage_from_cached(cached: list[torch.Tensor]) -> dict[str, Any] | None:
    if not cached:
        return None
    probs = torch.cat([item.detach().float().cpu() for item in cached], dim=0)
    if probs.ndim == 2:
        usage = probs.mean(dim=0)
    elif probs.ndim == 4:
        usage = probs.mean(dim=(0, 2, 3))
    else:
        return None
    num_experts = usage.numel()
    entropy = float((-(usage.clamp_min(1e-8) * usage.clamp_min(1e-8).log()).sum() / math.log(max(num_experts, 2))).item())
    max_prob = float(usage.max().item())
    active = float((usage > (0.25 / max(num_experts, 1))).sum().item())
    return {
        "entropy": entropy,
        "max_prob": max_prob,
        "active_experts": active,
        "usage": [float(value) for value in usage.tolist()],
    }


def _collect_adapter_diagnostics(adapter: Any, prefix: str, levels: dict[str, Any]) -> None:
    if adapter is None:
        return
    for level_name, level in getattr(adapter, "levels", {}).items():
        cached = getattr(level, "last_router_hard_probs", []) or getattr(level, "last_router_probs", [])
        summary = _router_usage_from_cached(cached)
        if summary is not None:
            levels[f"{prefix}.{level_name}"] = summary


def _collect_head_diagnostics(head: Any, levels: dict[str, Any]) -> None:
    cached = getattr(head, "last_router_hard_probs", []) or getattr(head, "last_router_probs", [])
    if not cached:
        return
    for level_name, probs in zip(("p3", "p4", "p5"), cached):
        summary = _router_usage_from_cached([probs])
        if summary is not None:
            levels[f"head.{level_name}"] = summary


def collect_router_diagnostics(checkpoint: Path, *, phase: int, round_idx: int, args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_router_diagnostics or args.dry_run:
        return {}
    if not (
        checkpoint_has_backbone_moe(checkpoint)
        or checkpoint_has_neck_moe(checkpoint)
        or checkpoint_has_head_moe(checkpoint)
    ):
        return {}

    image_list = _router_diagnostic_list(args)
    if not image_list.exists():
        return {}
    paths = [Path(line.strip()) for line in image_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    paths = paths[: max(0, args.router_diagnostic_images)]
    if not paths:
        return {}

    fedsto.ensure_efficientteacher_import_path()
    from configs.defaults import get_cfg
    from models.detector.yolo import Model
    from utils.fedsto_regularization import clear_latent_moe_router_cache

    cfg_path = write_runtime_config(
        f"phase{phase}_round{round_idx:03d}_router_diagnostic",
        target=None,
        weights=checkpoint,
        phase=phase,
        role="server",
        round_idx=round_idx,
        client=None,
        args=args,
    )
    cfg = get_cfg()
    cfg.merge_from_file(str(cfg_path))
    cfg.freeze()

    if args.router_diagnostic_device:
        device = torch.device(args.router_diagnostic_device)
    elif args.val_device:
        device = torch.device(args.val_device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpus > 0 else "cpu")

    model = Model(cfg).to(device).float()
    loaded = fedsto._load(checkpoint)
    teacher = loaded.get("ema") if loaded.get("ema") is not None else loaded.get("model")
    state = teacher.float().state_dict()
    target_state = model.state_dict()
    compatible = {
        key: value.float()
        for key, value in state.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    }
    model.load_state_dict(compatible, strict=False)
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
        if hasattr(module, "router_noise_std"):
            module.router_noise_std = 0.0
        if hasattr(module, "last_router_probs") and hasattr(module, "last_router_logits"):
            module.force_router_cache = True
    clear_latent_moe_router_cache(model)

    batch_size = max(1, args.router_diagnostic_batch_size)
    image_size = int(cfg.Dataset.img_size)
    seen = 0
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = _load_router_diagnostic_batch(paths[start:start + batch_size], image_size, device)
            if batch.numel() == 0:
                continue
            model(batch)
            seen += batch.shape[0]

    if seen <= 0:
        clear_latent_moe_router_cache(model)
        return {}

    levels: dict[str, Any] = {}
    backbone = getattr(model, "backbone", None)
    neck = getattr(model, "neck", None)
    head = getattr(model, "head", None)
    _collect_adapter_diagnostics(getattr(backbone, "adapter_moe", None), "backbone", levels)
    _collect_adapter_diagnostics(getattr(neck, "adapter_moe", None), "neck", levels)
    _collect_head_diagnostics(head, levels)

    clear_latent_moe_router_cache(model)
    if not levels:
        return {}
    entropies = [float(item["entropy"]) for item in levels.values()]
    max_probs = [float(item["max_prob"]) for item in levels.values()]
    active_counts = [float(item["active_experts"]) for item in levels.values()]

    row = {
        "phase": phase,
        "round": round_idx,
        "checkpoint": str(checkpoint.resolve()),
        "split": args.router_diagnostic_split,
        "images": seen,
        "mean_entropy": sum(entropies) / len(entropies),
        "mean_max_prob": sum(max_probs) / len(max_probs),
        "mean_active_experts": sum(active_counts) / len(active_counts),
        "levels": levels,
    }
    append_router_diagnostics(row)
    return row


def ensure_moe_global(current_global: Path, phase: int, round_idx: int, args: argparse.Namespace) -> Path:
    needs_backbone = use_backbone_moe(args, phase) and not checkpoint_has_backbone_moe(current_global)
    needs_neck = use_neck_moe(args, phase) and not checkpoint_has_neck_moe(current_global)
    needs_head = use_head_moe(args, phase) and not checkpoint_has_head_moe(current_global)
    if not needs_backbone and not needs_neck and not needs_head:
        return current_global
    cfg = write_runtime_config(
        f"phase{phase}_round{round_idx:03d}_moe_upcycle_probe",
        target=None,
        weights=current_global,
        phase=phase,
        role="server",
        round_idx=round_idx,
        client=None,
        args=args,
    )
    out = fedsto.GLOBAL_DIR / f"phase{phase}_round{round_idx:03d}_moe_upcycled_start.pt"
    return upcycle_checkpoint(
        current_global,
        out,
        cfg,
        stage=f"phase{phase}_round{round_idx:03d}_anonymous_backbone_neck_head_moe_upcycle",
        require_head_moe=use_head_moe(args, phase),
        require_neck_moe=use_neck_moe(args, phase),
    )


def validate_seed_checkpoint(path: Path, description: str) -> Path:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{description} checkpoint not found: {path}")
    ok, reason = fedsto.validate_checkpoint(path)
    if not ok:
        raise RuntimeError(f"{description} checkpoint is invalid: {path} ({reason})")
    return path


def seed_external_checkpoint(source: Path, out: Path, *, stage: str, force: bool) -> Path:
    if out.exists() and not force and fedsto.checkpoint_matches_protocol(out, PROTOCOL_VERSION):
        print(f"Reusing seeded checkpoint: {out}")
        return out
    fedsto.make_start_checkpoint(
        source,
        out,
        protocol=PROTOCOL_VERSION,
        stage=stage,
    )
    print(f"Seeded {out} from {source}")
    return out


def phase1_seed_history(args: argparse.Namespace, phase1_global: Path) -> list[dict[str, Any]]:
    if args.phase1_rounds <= 0:
        return []
    return [
        {
            "phase": 1,
            "round": round_idx,
            "global": str(phase1_global.resolve()),
            "protocol": PROTOCOL_VERSION,
            "seeded_from": str(args.phase1_checkpoint.resolve()),
        }
        for round_idx in range(1, args.phase1_rounds + 1)
    ]


def run_final_eval(args: argparse.Namespace) -> None:
    if not args.run_final_eval:
        return
    cmd = [
        sys.executable,
        str(FEDSTO_SCRIPT_ROOT / "evaluate_paper_protocol.py"),
        "--workspace",
        str(args.workspace_root),
        "--splits",
        args.final_eval_splits,
        "--batch-size",
        str(args.val_batch_size),
        "--best-basis",
        "map50",
    ]
    if args.val_device:
        cmd.extend(["--device", args.val_device])
    if not args.final_eval_plots:
        cmd.append("--no-plots")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def checkpoint_results_row(path: Path) -> dict[str, str]:
    results_path = path.parent.parent / "results.csv"
    if not results_path.exists():
        return {}
    with results_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))
    return rows[-1] if rows else {}


def _score_tensor(values: list[float], *, device: torch.device) -> torch.Tensor:
    clean = torch.tensor(
        [0.0 if value is None or not math.isfinite(float(value)) else float(value) for value in values],
        device=device,
        dtype=torch.float32,
    )
    if clean.numel() <= 1:
        return clean
    return (clean - clean.mean()) / clean.std(unbiased=False).clamp_min(1e-4)


def _softmax_weights(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if logits.numel() <= 1:
        return torch.ones_like(logits)
    return torch.softmax(logits / max(float(temperature), 1e-6), dim=0)


def _client_metric_values(paths: list[Path], args: argparse.Namespace) -> list[float]:
    values = []
    for path in paths:
        metrics = metric_summary(checkpoint_results_row(path))
        value = metrics.get(args.expert_aggregate_metric, float("nan"))
        values.append(value)
    return values


def _client_quality_weights(paths: list[Path], args: argparse.Namespace) -> tuple[torch.Tensor, list[float]]:
    values = _client_metric_values(paths, args)
    logits = _score_tensor(values, device=torch.device("cpu")) * float(args.expert_aggregate_quality_strength)
    weights = _softmax_weights(logits, args.expert_aggregate_temperature)
    return weights.cpu(), values


def _expert_id_from_key(key: str) -> int | None:
    parts = key.split(".")
    if "experts" in parts:
        idx = parts.index("experts")
        if idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return None
    if len(parts) >= 4 and parts[0] == "head" and parts[1] == "expert_m":
        try:
            return int(parts[3])
        except ValueError:
            return None
    return None


def _usage_level_from_key(key: str) -> str | None:
    parts = key.split(".")
    if len(parts) >= 5 and parts[0] in {"backbone", "neck"} and parts[1] == "adapter_moe":
        return f"{parts[0]}.{parts[3]}"
    if len(parts) >= 4 and parts[0] == "head" and parts[1] == "expert_m":
        try:
            level_idx = int(parts[2])
        except ValueError:
            return None
        return ("head.p3", "head.p4", "head.p5")[level_idx] if 0 <= level_idx < 3 else None
    return None


def _anonymous_client_expert_prior(client_id: int, expert_id: int) -> float:
    if client_id < 0 or expert_id < 0:
        return 0.0
    phase = float((client_id + 1) * (expert_id + 1))
    return math.sin(phase * 1.61803398875) + 0.5 * math.cos(phase * 0.754877666)


def _expert_weight_entropy(weights: torch.Tensor) -> float:
    if weights.numel() <= 1:
        return 0.0
    safe = weights.float().clamp_min(1e-8)
    return float((-(safe * safe.log()).sum() / math.log(weights.numel())).item())


def _expert_aggregate_list_for_client(client: dict[str, Any], args: argparse.Namespace) -> Path:
    paper_list = setup.LIST_ROOT / f"paper_eval_{client['weather']}_val.txt"
    if paper_list.exists():
        return paper_list
    target = setup.LIST_ROOT / f"client_{client['id']}_{client['weather']}_target.txt"
    if target.exists():
        return target
    return setup.LIST_ROOT / "server_cloudy_val.txt"


def _collect_client_router_usage(
    checkpoint: Path,
    *,
    phase: int,
    round_idx: int,
    client: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, list[float]]:
    if args.dry_run or not args.expert_aggregate_use_router_usage:
        return {}
    if not (
        checkpoint_has_backbone_moe(checkpoint)
        or checkpoint_has_neck_moe(checkpoint)
        or checkpoint_has_head_moe(checkpoint)
    ):
        return {}
    image_list = _expert_aggregate_list_for_client(client, args)
    if not image_list.exists():
        return {}
    paths = [Path(line.strip()) for line in image_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    paths = paths[: max(0, int(args.expert_aggregate_images))]
    if not paths:
        return {}

    fedsto.ensure_efficientteacher_import_path()
    from configs.defaults import get_cfg
    from models.detector.yolo import Model
    from utils.fedsto_regularization import clear_latent_moe_router_cache

    cfg_path = write_runtime_config(
        f"phase{phase}_round{round_idx:03d}_client{client['id']}_expert_aggregate_router",
        target=None,
        weights=checkpoint,
        phase=phase,
        role="client",
        round_idx=round_idx,
        client=client,
        args=args,
    )
    cfg = get_cfg()
    cfg.merge_from_file(str(cfg_path))
    cfg.freeze()

    if args.expert_aggregate_device:
        device = torch.device(args.expert_aggregate_device)
    elif args.router_diagnostic_device:
        device = torch.device(args.router_diagnostic_device)
    elif args.val_device:
        device = torch.device(args.val_device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpus > 0 else "cpu")

    model = Model(cfg).to(device).float()
    loaded = fedsto._load(checkpoint)
    teacher = loaded.get("ema") if loaded.get("ema") is not None else loaded.get("model")
    state = teacher.float().state_dict()
    target_state = model.state_dict()
    compatible = {
        key: value.float()
        for key, value in state.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    }
    model.load_state_dict(compatible, strict=False)
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
        if hasattr(module, "router_noise_std"):
            module.router_noise_std = 0.0
        if hasattr(module, "last_router_probs") and hasattr(module, "last_router_logits"):
            module.force_router_cache = True
    clear_latent_moe_router_cache(model)

    image_size = int(cfg.Dataset.img_size)
    batch_size = max(1, int(args.expert_aggregate_batch_size))
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = _load_router_diagnostic_batch(paths[start:start + batch_size], image_size, device)
            if batch.numel() == 0:
                continue
            model(batch)

    levels: dict[str, Any] = {}
    _collect_adapter_diagnostics(getattr(getattr(model, "backbone", None), "adapter_moe", None), "backbone", levels)
    _collect_adapter_diagnostics(getattr(getattr(model, "neck", None), "adapter_moe", None), "neck", levels)
    _collect_head_diagnostics(getattr(model, "head", None), levels)
    clear_latent_moe_router_cache(model)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return {level: [float(value) for value in item.get("usage", [])] for level, item in levels.items()}


def _collect_router_usages_for_clients(
    paths: list[Path],
    *,
    phase: int,
    round_idx: int,
    args: argparse.Namespace,
) -> list[dict[str, list[float]]]:
    usages: list[dict[str, list[float]]] = []
    for path, client in zip(paths, setup.CLIENTS):
        usages.append(
            _collect_client_router_usage(
                path,
                phase=phase,
                round_idx=round_idx,
                client=client,
                args=args,
            )
        )
    return usages


def _expertwise_client_weights(
    *,
    key: str,
    expert_id: int | None,
    base_value: torch.Tensor,
    client_values: list[torch.Tensor],
    quality_weights: torch.Tensor,
    metric_values: list[float],
    router_usages: list[dict[str, list[float]]],
    args: argparse.Namespace,
) -> torch.Tensor:
    logits = quality_weights.clamp_min(1e-8).log().float()

    if expert_id is not None:
        level = _usage_level_from_key(key)
        usage_logits = []
        for usage_by_level in router_usages:
            usage = usage_by_level.get(level or "", [])
            value = usage[expert_id] if 0 <= expert_id < len(usage) else 1.0 / max(args.moe_num_experts, 1)
            usage_logits.append(math.log(max(float(value), 1e-8)))
        if usage_logits:
            logits = logits + float(args.expert_aggregate_router_strength) * torch.tensor(usage_logits, dtype=torch.float32)

        delta_norms = [
            float((client_value.float() - base_value.float()).norm().item())
            for client_value in client_values
        ]
        logits = logits + float(args.expert_aggregate_delta_strength) * _score_tensor(delta_norms, device=torch.device("cpu"))

        priors = [
            _anonymous_client_expert_prior(int(client["id"]), expert_id)
            for client in setup.CLIENTS[: len(client_values)]
        ]
        logits = logits + float(args.expert_aggregate_prior_strength) * _score_tensor(priors, device=torch.device("cpu"))

    if args.expert_aggregate_metric_gate > 0:
        metric_scores = _score_tensor(metric_values, device=torch.device("cpu"))
        keep = (metric_scores >= -float(args.expert_aggregate_metric_gate)).float()
        if keep.sum() > 0:
            logits = logits + keep.clamp_min(1e-8).log()

    return _softmax_weights(logits, args.expert_aggregate_temperature)


def should_mix_key(key: str, scope: str) -> bool:
    if "anchors" in key:
        return False
    is_adapter = key.startswith("backbone.adapter_moe.") or key.startswith("neck.adapter_moe.")
    is_backbone = key.startswith("backbone.")
    if scope == "all":
        return True
    if scope == "backbone":
        return is_backbone
    if scope == "adapter":
        return is_adapter
    if scope == "adapter_head":
        return is_adapter or key.startswith("head.")
    if scope == "adapter_neck_head":
        return is_adapter or key.startswith("neck.") or key.startswith("head.")
    raise ValueError(f"Unsupported aggregate scope: {scope}")


def should_phase2_mix_key(key: str, scope: str) -> bool:
    return should_mix_key(key, scope)


def clipped_delta(
    base_value: torch.Tensor,
    update_value: torch.Tensor,
    *,
    max_relative_update: float,
    max_absolute_update: float,
) -> tuple[torch.Tensor, bool]:
    delta = update_value.float() - base_value.float()
    raw_norm = float(delta.norm().item())
    max_norm = (
        float(max_relative_update) * max(float(base_value.float().norm().item()), 1e-12)
        + float(max_absolute_update)
    )
    if max_norm > 0 and raw_norm > max_norm:
        return delta * (max_norm / max(raw_norm, 1e-12)), True
    return delta, False


def _aggregate_settings(phase: int, args: argparse.Namespace) -> tuple[bool, float, float, float, str]:
    if phase == 1:
        return (
            bool(args.phase1_soft_aggregate),
            float(args.phase1_aggregate_lambda),
            float(args.phase1_max_relative_update),
            float(args.phase1_max_absolute_update),
            args.phase1_aggregate_scope,
        )
    return (
        True,
        float(args.phase2_aggregate_lambda),
        float(args.phase2_max_relative_update),
        float(args.phase2_max_absolute_update),
        args.phase2_aggregate_scope,
    )


def aggregate_phase_checkpoints(
    paths: list[Path],
    base_path: Path,
    out: Path,
    *,
    phase: int,
    args: argparse.Namespace,
) -> Path:
    use_soft, mix_lambda, max_relative_update, max_absolute_update, aggregate_scope = _aggregate_settings(phase, args)
    if phase == 1 and not use_soft:
        return fedsto.aggregate_checkpoints(paths, base_path, out, backbone_only=True)

    base = fedsto._load(base_path)
    client_states = [fedsto._state_dict(fedsto._load(path), "model") for path in paths]
    model = base["model"].float()
    base_state = model.state_dict()
    mixed_state: dict[str, torch.Tensor] = {}
    updated = 0
    clipped = 0
    quality_weights, metric_values = _client_quality_weights(paths, args)
    router_usages = (
        _collect_router_usages_for_clients(paths, phase=phase, round_idx=int(out.stem.split("_round")[-1].split("_")[0]), args=args)
        if args.expert_wise_aggregate
        else []
    )
    expert_weight_entropies: list[float] = []
    expert_weight_maxes: list[float] = []
    sample_expert_weights: dict[str, list[float]] = {}

    for key, value in base_state.items():
        if value.dtype.is_floating_point and should_mix_key(key, aggregate_scope):
            client_values = [state[key].float() for state in client_states]
            if args.expert_wise_aggregate:
                expert_id = _expert_id_from_key(key)
                weights = _expertwise_client_weights(
                    key=key,
                    expert_id=expert_id,
                    base_value=value,
                    client_values=client_values,
                    quality_weights=quality_weights,
                    metric_values=metric_values,
                    router_usages=router_usages,
                    args=args,
                ).to(dtype=torch.float32)
                avg = torch.zeros_like(client_values[0].float())
                for weight, client_value in zip(weights, client_values):
                    avg = avg + client_value.float() * float(weight.item())
                if expert_id is not None:
                    expert_weight_entropies.append(_expert_weight_entropy(weights))
                    expert_weight_maxes.append(float(weights.max().item()))
                    if len(sample_expert_weights) < 12:
                        sample_expert_weights[key] = [float(item) for item in weights.tolist()]
            else:
                avg = torch.stack(client_values, dim=0).mean(dim=0)
            delta, was_clipped = clipped_delta(
                value,
                avg,
                max_relative_update=max_relative_update,
                max_absolute_update=max_absolute_update,
            )
            mixed_state[key] = (value.float() + mix_lambda * delta).to(value.dtype)
            updated += 1
            clipped += int(was_clipped)
        else:
            mixed_state[key] = value

    model.load_state_dict(mixed_state, strict=False)
    base["model"] = model.half()
    if base.get("ema") is not None:
        ema = base["ema"].float()
        ema.load_state_dict(mixed_state, strict=False)
        base["ema"] = ema.half()
    base["epoch"] = -1
    base["optimizer"] = None
    base["dqa_anonymous_backbone_moe_aggregate"] = {
        "phase": phase,
        "lambda": mix_lambda,
        "scope": aggregate_scope,
        "max_relative_update": max_relative_update,
        "max_absolute_update": max_absolute_update,
        "updated_tensors": updated,
        "clipped_tensors": clipped,
    }

    from utils.fedsto_regularization import clear_latent_moe_router_cache

    clear_latent_moe_router_cache(base["model"])
    if base.get("ema") is not None:
        clear_latent_moe_router_cache(base["ema"])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, out)
    if args.expert_wise_aggregate:
        append_expert_aggregation_diagnostics(
            [
                {
                    "phase": phase,
                    "round": int(out.stem.split("_round")[-1].split("_")[0]),
                    "scope": aggregate_scope,
                    "num_clients": len(paths),
                    "quality_weights": json.dumps([float(item) for item in quality_weights.tolist()]),
                    "client_metrics": json.dumps([float(item) if math.isfinite(float(item)) else None for item in metric_values]),
                    "mean_expert_entropy": (
                        sum(expert_weight_entropies) / len(expert_weight_entropies)
                        if expert_weight_entropies
                        else None
                    ),
                    "min_expert_entropy": min(expert_weight_entropies) if expert_weight_entropies else None,
                    "max_expert_weight": max(expert_weight_maxes) if expert_weight_maxes else None,
                    "updated_tensors": updated,
                    "clipped_tensors": clipped,
                    "sample_expert_weights_json": json.dumps(sample_expert_weights, sort_keys=True),
                }
            ]
        )
    print(
        f"Phase{phase} soft/trust aggregation: "
        f"lambda={mix_lambda}, scope={aggregate_scope}, "
        f"updated={updated}, clipped={clipped}, expertwise={args.expert_wise_aggregate}"
    )
    return out


def run_protocol(args: argparse.Namespace) -> None:
    global PROTOCOL_VERSION
    if args.protocol_suffix:
        PROTOCOL_VERSION = f"{BASE_PROTOCOL_VERSION}+{args.protocol_suffix}"

    apply_workspace_root(args.workspace_root)
    setup.build_base_configs()
    pretrained = fedsto.PRETRAINED_PATH if args.dry_run else fedsto.download_pretrained()
    if not args.dry_run:
        fedsto.check_runtime_dependencies()
    args.gpus = fedsto.resolve_gpus(args.gpus)
    args.config_device = "" if args.gpus > 1 else args.device
    fedsto.GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    fedsto.CLIENT_STATE_DIR.mkdir(parents=True, exist_ok=True)

    if args.warmup_checkpoint:
        args.warmup_checkpoint = validate_seed_checkpoint(args.warmup_checkpoint, "warmup seed")
    if args.phase1_checkpoint:
        args.phase1_checkpoint = validate_seed_checkpoint(args.phase1_checkpoint, "phase1 seed")

    current_global = fedsto.GLOBAL_DIR / "round000_warmup.pt"
    if args.warmup_checkpoint:
        current_global = seed_external_checkpoint(
            args.warmup_checkpoint,
            current_global,
            stage="round000_canonical_warmup_seed",
            force=args.force_restart or args.force_retrain,
        )
    if not current_global.exists():
        warmup_name = "runtime_server_warmup_moe" if args.warmup_moe else "runtime_server_warmup_dense"
        warmup_cfg = write_warmup_config(
            warmup_name,
            weights=pretrained,
            args=args,
        )
        warmup_ckpt = fedsto.run_train(warmup_cfg, args.dry_run, gpus=args.gpus, master_port=args.master_port)
        if args.dry_run and not fedsto.checkpoint_present(warmup_ckpt):
            if args.warmup_moe:
                warmup_ckpt = upcycle_checkpoint(
                    pretrained,
                    current_global,
                    warmup_cfg,
                    stage="round000_moe_warmup_dry_run_placeholder",
                    require_head_moe=use_head_moe(args, 0),
                    require_neck_moe=use_neck_moe(args, 0),
                )
            else:
                fedsto.make_start_checkpoint(
                    pretrained,
                    current_global,
                    protocol=PROTOCOL_VERSION,
                    stage="round000_dense_warmup_dry_run_placeholder",
                )
                warmup_ckpt = current_global
        fedsto.make_start_checkpoint(
            warmup_ckpt,
            current_global,
            protocol=PROTOCOL_VERSION,
            stage="round000_moe_warmup" if args.warmup_moe else "round000_dense_warmup",
        )

    seeded_phase1_history: list[dict[str, Any]] = []
    seeded_phase1_global: Path | None = None
    if args.phase1_checkpoint and args.phase1_rounds > 0:
        seeded_phase1_global = fedsto.GLOBAL_DIR / f"phase1_round{args.phase1_rounds:03d}_global.pt"
        seed_external_checkpoint(
            args.phase1_checkpoint,
            seeded_phase1_global,
            stage=f"phase1_round{args.phase1_rounds:03d}_canonical_seed",
            force=args.force_restart or args.force_retrain,
        )
        seeded_phase1_history = phase1_seed_history(args, seeded_phase1_global)

    if args.force_restart:
        history = list(seeded_phase1_history)
        if seeded_phase1_global is not None:
            current_global = seeded_phase1_global
            fedsto.write_history(history)
            print(
                "Ignoring existing history because --force-restart was set, "
                f"then seeding phase 1 from {seeded_phase1_global}."
            )
        else:
            print("Ignoring existing history because --force-restart was set.")
    else:
        loaded_history = fedsto.load_history()
        if seeded_phase1_history and (
            not any(int(entry.get("phase", 0)) == 1 for entry in loaded_history)
            or any(
                int(entry.get("phase", 0)) == 1 and entry.get("protocol") != PROTOCOL_VERSION
                for entry in loaded_history
            )
        ):
            loaded_history = list(seeded_phase1_history)
            fedsto.write_history(loaded_history)
        history, current_global, next_round = fedsto.completed_history_prefix(
            loaded_history,
            phase1_rounds=args.phase1_rounds,
            phase2_rounds=args.phase2_rounds,
            warmup_checkpoint=current_global,
            expected_protocol=PROTOCOL_VERSION,
        )
        if loaded_history != history:
            fedsto.write_history(history)
        if next_round is None:
            print("All requested federated rounds are already complete.")
            run_final_eval(args)
            return
        if history:
            print(f"Resuming after {len(history)} completed rounds from {current_global}")
        else:
            print("No completed federated rounds found; starting from phase 1 round 1.")

    if not args.keep_intermediate_checkpoints:
        fedsto.cleanup_completed_intermediates(history)

    notify(
        args,
        "DQA Upcycled Anonymous MoE started",
        (
            f"workspace: `{args.workspace_root}`\n"
            f"schedule: warmup={args.warmup_epochs}, phase1={args.phase1_rounds}, phase2={args.phase2_rounds}\n"
            f"warmup_moe={args.warmup_moe}, "
            f"moe: start_phase={args.moe_start_phase}, backbone_experts={args.moe_num_experts}, "
            f"neck={args.enable_neck_moe}, head={args.enable_head_moe}, top_k={args.moe_top_k}"
        ),
    )

    completed = {(int(entry["phase"]), int(entry["round"])) for entry in history}
    for phase, rounds in [(1, args.phase1_rounds), (2, args.phase2_rounds)]:
        for round_idx in range(1, rounds + 1):
            if (phase, round_idx) in completed:
                continue

            current_global = ensure_moe_global(current_global, phase, round_idx, args)
            local_paths: list[Path] = []
            for client in setup.CLIENTS:
                target = setup.LIST_ROOT / f"client_{client['id']}_{client['weather']}_target.txt"
                run_name = f"phase{phase}_round{round_idx:03d}_client{client['id']}_{client['weather']}"
                start = fedsto.CLIENT_STATE_DIR / f"client_{client['id']}_phase{phase}_round{round_idx:03d}_start.pt"
                previous = fedsto.CLIENT_STATE_DIR / f"client_{client['id']}_latest.pt"
                ckpt = fedsto.reuse_checkpoint_if_valid(
                    fedsto.checkpoint_path(run_name),
                    f"client run {run_name}",
                    force_retrain=args.force_retrain,
                    expected_protocol=PROTOCOL_VERSION,
                )
                if ckpt is not None:
                    local_paths.append(ckpt)
                    fedsto.make_start_checkpoint(
                        ckpt,
                        previous,
                        protocol=PROTOCOL_VERSION,
                        stage=f"client_{client['id']}_latest_observation_only",
                    )
                    continue

                if not fedsto.checkpoint_matches_protocol(start, PROTOCOL_VERSION):
                    fedsto.make_start_checkpoint(
                        current_global,
                        start,
                        protocol=PROTOCOL_VERSION,
                        stage=f"phase{phase}_round{round_idx:03d}_client{client['id']}_start",
                    )
                cfg = write_runtime_config(
                    run_name,
                    target=target,
                    weights=start,
                    phase=phase,
                    role="client",
                    round_idx=round_idx,
                    client=client,
                    args=args,
                )
                ckpt = fedsto.run_train(cfg, args.dry_run, gpus=args.gpus, master_port=args.master_port)
                if args.dry_run and not fedsto.checkpoint_present(ckpt):
                    fedsto.make_start_checkpoint(
                        start,
                        ckpt,
                        protocol=PROTOCOL_VERSION,
                        stage=f"phase{phase}_round{round_idx:03d}_client{client['id']}_dry_run_placeholder",
                    )
                fedsto.mark_checkpoint_protocol(ckpt, PROTOCOL_VERSION, f"phase{phase}_round{round_idx:03d}_client{client['id']}")
                local_paths.append(ckpt)
                fedsto.make_start_checkpoint(
                    ckpt,
                    previous,
                    protocol=PROTOCOL_VERSION,
                    stage=f"client_{client['id']}_latest_observation_only",
                )

            client_aggregate = fedsto.GLOBAL_DIR / f"phase{phase}_round{round_idx:03d}_client_aggregate.pt"
            aggregate_phase_checkpoints(local_paths, current_global, client_aggregate, phase=phase, args=args)
            fedsto.mark_checkpoint_protocol(client_aggregate, PROTOCOL_VERSION, f"phase{phase}_round{round_idx:03d}_client_aggregate")

            server_name = f"phase{phase}_round{round_idx:03d}_server"
            server_start = fedsto.GLOBAL_DIR / f"phase{phase}_round{round_idx:03d}_server_start.pt"
            server_ckpt = fedsto.reuse_checkpoint_if_valid(
                fedsto.checkpoint_path(server_name),
                f"server run {server_name}",
                force_retrain=args.force_retrain,
                expected_protocol=PROTOCOL_VERSION,
            )
            if server_ckpt is None:
                if not fedsto.checkpoint_matches_protocol(server_start, PROTOCOL_VERSION):
                    fedsto.make_start_checkpoint(
                        client_aggregate,
                        server_start,
                        protocol=PROTOCOL_VERSION,
                        stage=f"phase{phase}_round{round_idx:03d}_server_start",
                    )
                cfg = write_runtime_config(
                    server_name,
                    target=None,
                    weights=server_start,
                    phase=phase,
                    role="server",
                    round_idx=round_idx,
                    client=None,
                    args=args,
                )
                server_ckpt = fedsto.run_train(cfg, args.dry_run, gpus=args.gpus, master_port=args.master_port)
                if args.dry_run and not fedsto.checkpoint_present(server_ckpt):
                    fedsto.make_start_checkpoint(
                        server_start,
                        server_ckpt,
                        protocol=PROTOCOL_VERSION,
                        stage=f"phase{phase}_round{round_idx:03d}_server_dry_run_placeholder",
                    )
                fedsto.mark_checkpoint_protocol(server_ckpt, PROTOCOL_VERSION, f"phase{phase}_round{round_idx:03d}_server_update")

            next_global = fedsto.GLOBAL_DIR / f"phase{phase}_round{round_idx:03d}_global.pt"
            fedsto.make_start_checkpoint(
                server_ckpt,
                next_global,
                protocol=PROTOCOL_VERSION,
                stage=f"phase{phase}_round{round_idx:03d}_anonymous_backbone_moe_global",
            )
            current_global = next_global
            server_metrics = metric_summary(read_last_results_row(server_name))
            router_diagnostics = collect_router_diagnostics(
                current_global,
                phase=phase,
                round_idx=round_idx,
                args=args,
            )
            row = {
                "phase": phase,
                "round": round_idx,
                "global": str(current_global.resolve()),
                "moe_enabled": use_backbone_moe(args, phase),
                "server_metrics": server_metrics,
                "router_diagnostics": router_diagnostics,
            }
            history.append(
                {
                    "phase": phase,
                    "round": round_idx,
                    "global": str(current_global.resolve()),
                    "protocol": PROTOCOL_VERSION,
                }
            )
            fedsto.write_history(history)
            append_summary(row)
            completed.add((phase, round_idx))
            print(
                f"Completed phase {phase} round {round_idx}: "
                f"mAP50={server_metrics.get('map50'):.5f}, mAP50:95={server_metrics.get('map50_95'):.5f}"
            )
            if args.discord and (round_idx == rounds or use_backbone_moe(args, phase)):
                notify(
                    args,
                    "DQA Upcycled Anonymous MoE round update",
                    (
                        f"phase={phase}, round={round_idx}, moe={use_backbone_moe(args, phase)}\n"
                        f"cloudy mAP50={server_metrics.get('map50'):.5f}, "
                        f"mAP50:95={server_metrics.get('map50_95'):.5f}\n"
                        f"router entropy={router_diagnostics.get('mean_entropy', float('nan')):.3f}, "
                        f"active experts={router_diagnostics.get('mean_active_experts', float('nan')):.2f}"
                    ),
                )
            if not args.keep_intermediate_checkpoints:
                fedsto.cleanup_round_intermediates(phase, round_idx)

    run_final_eval(args)
    notify(
        args,
        "DQA Upcycled Anonymous MoE finished",
        f"Final global: `{current_global}`\nSummary CSV: `{SUMMARY_PATH}`",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--protocol-suffix", default="")
    parser.add_argument("--warmup-checkpoint", type=Path, default=None)
    parser.add_argument("--phase1-checkpoint", type=Path, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=50)
    parser.add_argument("--warmup-moe", action="store_true")
    parser.add_argument("--warmup-train-scope", default="all")
    parser.add_argument("--warmup-lr0", type=float, default=0.0)
    parser.add_argument("--warmup-bias-lr", type=float, default=0.0)
    parser.add_argument("--warmup-hyp-warmup-epochs", type=float, default=-1.0)
    parser.add_argument("--phase1-rounds", type=int, default=20)
    parser.add_argument("--phase2-rounds", type=int, default=20)
    parser.add_argument("--moe-start-phase", type=int, choices=(1, 2), default=2)
    parser.add_argument("--phase1-train-scope", default="backbone")
    parser.add_argument("--phase1-server-train-scope", default="")
    parser.add_argument("--phase1-orthogonal-weight", type=float, default=0.0)
    parser.add_argument("--phase1-soft-aggregate", action="store_true")
    parser.add_argument("--phase1-aggregate-lambda", type=float, default=1.0)
    parser.add_argument("--phase1-max-relative-update", type=float, default=0.0)
    parser.add_argument("--phase1-max-absolute-update", type=float, default=0.0)
    parser.add_argument(
        "--phase1-aggregate-scope",
        choices=("all", "backbone", "adapter", "adapter_head", "adapter_neck_head"),
        default="backbone",
    )
    parser.add_argument("--phase2-train-scope", default="all")
    parser.add_argument("--phase2-server-train-scope", default="")
    parser.add_argument("--phase2-late-train-scope", default="")
    parser.add_argument("--phase2-head-unfreeze-after-round", type=int, default=0)
    parser.add_argument("--orthogonal-weight", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--phase2-batch-size", type=int, default=64)
    parser.add_argument("--phase2-server-lr0", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=29547)
    parser.add_argument("--device", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-restart", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--keep-intermediate-checkpoints", action="store_true")

    parser.add_argument("--moe-num-experts", type=int, default=4)
    parser.add_argument("--moe-top-k", type=int, default=2)
    parser.add_argument("--moe-temperature", type=float, default=1.0)
    parser.add_argument("--moe-scale", type=float, default=0.25)
    parser.add_argument("--moe-shared-scale", type=float, default=1.0)
    parser.add_argument("--moe-adapter-ratio", type=float, default=0.125)
    parser.add_argument("--moe-min-channels", type=int, default=16)
    parser.add_argument("--moe-levels", default="c3,c4,c5")
    parser.add_argument("--moe-kernels", default="3,5,7")
    parser.add_argument("--moe-context-dim", type=int, default=8)
    parser.add_argument("--moe-quality-dim", type=int, default=4)
    parser.add_argument("--moe-router-noise-std", type=float, default=0.01)
    parser.add_argument("--dqa-context-bias", type=float, default=1.0)
    parser.add_argument("--moe-balance-weight", type=float, default=0.02)
    parser.add_argument("--moe-entropy-weight", type=float, default=0.002)
    parser.add_argument("--moe-sample-entropy-weight", type=float, default=0.0)
    parser.add_argument("--moe-z-loss-weight", type=float, default=0.0001)
    parser.add_argument("--moe-diversity-weight", type=float, default=0.001)
    parser.add_argument("--moe-threshold-routing", action="store_true")
    parser.add_argument("--moe-threshold", type=float, default=0.0)
    parser.add_argument("--moe-freeze-bn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--moe-routing-mode", choices=("soft", "topk", "expert_choice", "balanced", "base"), default="soft")
    parser.add_argument("--moe-straight-through", action="store_true")
    parser.add_argument("--moe-expert-choice-capacity-factor", type=float, default=1.0)
    parser.add_argument("--moe-dqa-prior-strength", type=float, default=0.0)
    parser.add_argument("--moe-router-init-std", type=float, default=0.0)
    parser.add_argument("--enable-head-moe", action="store_true")
    parser.add_argument("--head-moe-start-phase", type=int, choices=(1, 2), default=2)
    parser.add_argument("--head-moe-num-experts", type=int, default=4)
    parser.add_argument("--head-moe-top-k", type=int, default=2)
    parser.add_argument("--head-moe-temperature", type=float, default=1.0)
    parser.add_argument("--head-moe-scale", type=float, default=0.25)
    parser.add_argument("--head-moe-balance-weight", type=float, default=0.01)
    parser.add_argument("--head-moe-entropy-weight", type=float, default=0.001)
    parser.add_argument("--head-moe-sample-entropy-weight", type=float, default=0.0)
    parser.add_argument("--head-moe-specialization-weight", type=float, default=0.0)
    parser.add_argument("--head-moe-specialize-by-client", action="store_true")
    parser.add_argument("--head-moe-routing-mode", choices=("soft", "topk", "expert_choice", "balanced", "base"), default="soft")
    parser.add_argument("--head-moe-straight-through", action="store_true")
    parser.add_argument("--head-moe-expert-choice-capacity-factor", type=float, default=1.0)
    parser.add_argument("--head-moe-router-init-std", type=float, default=0.0)
    parser.add_argument("--enable-neck-moe", action="store_true")
    parser.add_argument("--neck-moe-start-phase", type=int, choices=(1, 2), default=2)
    parser.add_argument("--neck-moe-num-experts", type=int, default=4)
    parser.add_argument("--neck-moe-top-k", type=int, default=2)
    parser.add_argument("--neck-moe-temperature", type=float, default=1.0)
    parser.add_argument("--neck-moe-scale", type=float, default=0.15)
    parser.add_argument("--neck-moe-shared-scale", type=float, default=1.0)
    parser.add_argument("--neck-moe-adapter-ratio", type=float, default=0.0625)
    parser.add_argument("--neck-moe-min-channels", type=int, default=16)
    parser.add_argument("--neck-moe-levels", default="p3,p4,p5")
    parser.add_argument("--neck-moe-kernels", default="3,5")
    parser.add_argument("--neck-moe-router-noise-std", type=float, default=0.005)
    parser.add_argument("--neck-moe-balance-weight", type=float, default=0.02)
    parser.add_argument("--neck-moe-entropy-weight", type=float, default=0.001)
    parser.add_argument("--neck-moe-sample-entropy-weight", type=float, default=0.0)
    parser.add_argument("--neck-moe-z-loss-weight", type=float, default=0.0001)
    parser.add_argument("--neck-moe-diversity-weight", type=float, default=0.001)
    parser.add_argument("--neck-moe-routing-mode", choices=("soft", "topk", "expert_choice", "balanced", "base"), default="soft")
    parser.add_argument("--neck-moe-straight-through", action="store_true")
    parser.add_argument("--neck-moe-expert-choice-capacity-factor", type=float, default=1.0)
    parser.add_argument("--neck-moe-dqa-prior-strength", type=float, default=0.0)
    parser.add_argument("--neck-moe-router-init-std", type=float, default=0.0)
    parser.add_argument("--phase2-aggregate-lambda", type=float, default=0.15)
    parser.add_argument("--phase2-max-relative-update", type=float, default=0.01)
    parser.add_argument("--phase2-max-absolute-update", type=float, default=0.0)
    parser.add_argument(
        "--phase2-aggregate-scope",
        choices=("all", "backbone", "adapter", "adapter_head", "adapter_neck_head"),
        default="all",
    )
    parser.add_argument("--expert-wise-aggregate", action="store_true")
    parser.add_argument(
        "--expert-aggregate-metric",
        choices=("map50", "map50_95", "precision", "recall"),
        default="map50",
    )
    parser.add_argument("--expert-aggregate-temperature", type=float, default=0.7)
    parser.add_argument("--expert-aggregate-quality-strength", type=float, default=1.0)
    parser.add_argument("--expert-aggregate-router-strength", type=float, default=0.75)
    parser.add_argument("--expert-aggregate-delta-strength", type=float, default=0.15)
    parser.add_argument("--expert-aggregate-prior-strength", type=float, default=0.15)
    parser.add_argument("--expert-aggregate-metric-gate", type=float, default=0.0)
    parser.add_argument("--expert-aggregate-use-router-usage", action="store_true")
    parser.add_argument("--expert-aggregate-images", type=int, default=16)
    parser.add_argument("--expert-aggregate-batch-size", type=int, default=4)
    parser.add_argument("--expert-aggregate-device", default="")
    parser.add_argument("--skip-router-diagnostics", action="store_true")
    parser.add_argument("--router-diagnostic-split", default="cloudy")
    parser.add_argument("--router-diagnostic-images", type=int, default=12)
    parser.add_argument("--router-diagnostic-batch-size", type=int, default=4)
    parser.add_argument("--router-diagnostic-device", default="")

    parser.add_argument("--run-final-eval", action="store_true")
    parser.add_argument("--final-eval-splits", default="cloudy,overcast,rainy,snowy,total")
    parser.add_argument("--final-eval-plots", action="store_true")
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--val-device", default="")
    parser.add_argument("--discord", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run_protocol(parse_args(argv))


if __name__ == "__main__":
    main()
