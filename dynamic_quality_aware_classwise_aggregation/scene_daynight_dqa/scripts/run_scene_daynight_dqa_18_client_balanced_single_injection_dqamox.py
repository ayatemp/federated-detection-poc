#!/usr/bin/env python3
"""Run an client-balanced single-injection full-from-warmup DQA-MoX experiment.

This is the full protocol version of the latent DQA-MoX idea. Unlike the
shorter follow-up notebooks, it does not assume that warmup or repair-only
baselines have already been trained. The model starts with a latent MoE YOLO
head during warmup, then runs the repair baseline and the DQA branch under the
same scene/day-night setting.

This is the follow-up loop after 15. With the pseudo-label path fixed, the
first one or two DQA rounds can slightly beat the warmup, but longer repeated
pseudo self-training drifts back down. 18 therefore keeps the FedMoX-style
full-from-warmup protocol and the latent MoE detector, but changes the schedule:
inject DQA/pseudoGT briefly, then consolidate the same MoE detector with source
server repair rounds instead of continuing to trust increasingly self-reinforced
pseudo boxes.

* pseudoGT is used briefly as a class/objectness/router/domain signal;
* pseudoGT box regression is weak and only applied to strict pseudo boxes;
* selected pseudoGT is capped against the actual selected pool from the start,
  so easy vehicle/light classes cannot dominate the client updates;
* post-DQA source consolidation keeps the MoE head while removing further
  pseudoGT drift.

The target remains final total mAP50 >= 0.60, but the first success criterion is
to turn the early DQA gain into a stable final target-domain gain.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import shutil
import subprocess
import sys
import time
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
NAV_ROOT = REPO_ROOT / "navigating_data_heterogeneity"
DQA_ROOT = PROJECT_ROOT.parent
PSEUDOGT_SCRIPTS = REPO_ROOT / "pseudogt_learnability" / "scripts"
PROTOCOL_VERSION = "scene_daynight_dqa_18_client_balanced_single_injection_dqamox_v1"

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


def parse_checkpoint_list(raw: str) -> list[Path]:
    paths = [Path(item.strip()).expanduser().resolve() for item in raw.split(",") if item.strip()]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing pseudo-teacher checkpoint: {missing[0]}")
    return paths


def parse_xyxy(raw: str) -> tuple[float, float, float, float]:
    values = [float(item) for item in str(raw).split()]
    if len(values) != 4:
        raise ValueError(f"Invalid xyxy: {raw!r}")
    return values[0], values[1], values[2], values[3]


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
        "specialization_weight": float(getattr(args, "router_specialization_weight", 0.0)),
        "specialization_target": int(getattr(args, "router_specialization_target", -1)),
    }
    cfg.setdefault("ClassSkewFedSTO", {})
    cfg["ClassSkewFedSTO"]["enabled"] = bool(args.class_skew_residual)
    cfg["ClassSkewFedSTO"]["use_residual"] = bool(args.class_skew_residual)
    cfg["ClassSkewFedSTO"]["orthogonal_weight"] = float(args.class_skew_orthogonal_weight)
    cfg["ClassSkewFedSTO"]["srip_weight"] = float(args.class_skew_srip_weight)
    cfg["ClassSkewFedSTO"]["residual_weight"] = float(args.class_skew_residual_weight)
    return cfg


def config_device(args: argparse.Namespace) -> str:
    return ""


def repeated_expr(path: Path, repeat: int) -> str:
    return str(path.resolve()) if repeat <= 1 else f"{path.resolve()}*{repeat}"


def _client_split_name(client: dict[str, Any]) -> str:
    return str(client.get("weather") or f"{client.get('scene', '')}_{client.get('time', '')}").lower()


def _domain_router_target(client: dict[str, Any], num_experts: int) -> int:
    split = _client_split_name(client)
    if num_experts <= 1:
        return -1
    if num_experts >= 6:
        if "highway" in split and "day" in split:
            return 0
        if "highway" in split and "night" in split:
            return 1
        if "citystreet" in split and "day" in split:
            return 2
        if "citystreet" in split and "night" in split:
            return 3
        if "residential" in split and "day" in split:
            return 4
        if "residential" in split and "night" in split:
            return 5
    if "highway" in split and "day" in split:
        return 0 % num_experts
    if "highway" in split and "night" in split:
        return 1 % num_experts
    if "citystreet" in split and "day" in split:
        return 2 % num_experts
    if "citystreet" in split and "night" in split:
        return 3 % num_experts
    if "residential" in split and "day" in split:
        return 2 % num_experts
    if "residential" in split and "night" in split:
        return 3 % num_experts
    return int(client.get("id", 0)) % num_experts


def _class_group_fractions(stats: dict[str, Any]) -> dict[str, float]:
    raw_counts = stats.get("selected_class_counts") or stats.get("class_counts") or {}
    counts: dict[int, float] = {}
    for key, value in raw_counts.items():
        try:
            counts[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    total = sum(counts.values())
    if total <= 0:
        return {"vru": 0.0, "vehicle": 0.0, "traffic": 0.0}
    vru = sum(counts.get(cls, 0.0) for cls in (0, 1, 5, 6))
    vehicle = sum(counts.get(cls, 0.0) for cls in (2, 3, 4, 9))
    traffic = sum(counts.get(cls, 0.0) for cls in (7, 8))
    return {"vru": vru / total, "vehicle": vehicle / total, "traffic": traffic / total}


def router_specialization_plan(
    client: dict[str, Any],
    stats: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    base_weight = float(getattr(args, "router_specialization_weight", 0.0))
    mode = str(getattr(args, "router_specialization_map", "none")).lower()
    num_experts = int(getattr(args, "num_experts", 4))
    if base_weight <= 0 or mode == "none":
        return {"target": -1, "weight": 0.0, "reason": "disabled"}

    target = _domain_router_target(client, num_experts)
    reason = "domain6" if mode == "domain6" else "domain4"
    fractions = _class_group_fractions(stats)
    if mode in {"client4", "client6"}:
        target = int(client.get("id", 0)) % max(num_experts, 1)
        reason = f"client_id_mod{num_experts}"
    elif mode == "class4":
        if fractions["vru"] >= fractions["vehicle"] and fractions["vru"] >= fractions["traffic"]:
            target, reason = 0 % num_experts, "class_vru"
        elif fractions["traffic"] >= fractions["vehicle"]:
            target, reason = 1 % num_experts, "class_traffic"
        else:
            split = _client_split_name(client)
            target, reason = (3 if "night" in split else 2) % num_experts, "class_vehicle_domain_time"
    elif mode == "hybrid_dqa4":
        threshold = float(getattr(args, "router_specialization_class_threshold", 0.28))
        if fractions["vru"] >= threshold:
            target, reason = 0 % num_experts, "dqa_class_vru"
        elif fractions["traffic"] >= threshold:
            target, reason = 1 % num_experts, "dqa_class_traffic"
        else:
            split = _client_split_name(client)
            target, reason = (3 if "night" in split else 2) % num_experts, "dqa_domain_time"

    mean_score = float(stats.get("mean_score") or 0.0)
    mean_stability = float(stats.get("mean_stability") or 0.0)
    boxes = float(stats.get("pseudo_boxes_kept") or stats.get("target_selected_boxes") or 0.0)
    min_quality = float(getattr(args, "router_specialization_min_quality", 0.55))
    min_boxes = max(float(getattr(args, "router_specialization_min_boxes", 500.0)), 1.0)
    max_weight = float(getattr(args, "router_specialization_max_weight", base_weight))
    quality_signal = 0.7 * mean_score + 0.3 * mean_stability
    quality_gate = max(0.0, min(1.0, (quality_signal - min_quality) / max(1.0 - min_quality, 1e-6)))
    count_gate = max(0.0, min(1.0, math.sqrt(boxes / min_boxes)))
    class_confidence = max(fractions.values()) if fractions else 0.0
    dynamic_weight = min(max_weight, base_weight * quality_gate * count_gate * (0.75 + 0.25 * class_confidence))
    if dynamic_weight <= 0 or target < 0:
        target = -1
    return {
        "target": int(target),
        "weight": float(dynamic_weight),
        "reason": reason,
        "quality_signal": quality_signal,
        "quality_gate": quality_gate,
        "count_gate": count_gate,
        "boxes": boxes,
        **{f"{key}_fraction": value for key, value in fractions.items()},
    }


def train_expr(
    source_list: Path,
    pseudo_list: Path,
    source_repeat: int,
    pseudo_repeat: int,
    *,
    style_list: Path | None = None,
    style_repeat: int = 0,
) -> str:
    parts = [repeated_expr(source_list, source_repeat)]
    if style_list is not None and style_repeat > 0:
        parts.append(repeated_expr(style_list, style_repeat))
    if pseudo_repeat > 0:
        parts.append(repeated_expr(pseudo_list, pseudo_repeat))
    return "||".join(parts)


def image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def fourier_style_transfer(source: Image.Image, target: Image.Image, beta: float) -> Image.Image:
    source_arr = np.asarray(source.convert("RGB"), dtype=np.float32)
    target_arr = np.asarray(target.convert("RGB"), dtype=np.float32)
    if target_arr.shape[:2] != source_arr.shape[:2]:
        target_img = Image.fromarray(np.clip(target_arr, 0, 255).astype(np.uint8))
        target_arr = np.asarray(target_img.resize((source_arr.shape[1], source_arr.shape[0]), Image.BILINEAR), dtype=np.float32)

    beta = max(0.0, float(beta))
    if beta <= 0.0:
        return Image.fromarray(np.clip(source_arr, 0, 255).astype(np.uint8))

    source_fft = np.fft.fft2(source_arr, axes=(0, 1))
    target_fft = np.fft.fft2(target_arr, axes=(0, 1))
    source_amp = np.abs(source_fft)
    source_phase = np.angle(source_fft)
    target_amp = np.abs(target_fft)

    source_amp = np.fft.fftshift(source_amp, axes=(0, 1))
    target_amp = np.fft.fftshift(target_amp, axes=(0, 1))
    height, width = source_arr.shape[:2]
    radius = max(1, int(min(height, width) * beta))
    center_h, center_w = height // 2, width // 2
    h0, h1 = max(0, center_h - radius), min(height, center_h + radius + 1)
    w0, w1 = max(0, center_w - radius), min(width, center_w + radius + 1)
    source_amp[h0:h1, w0:w1, :] = target_amp[h0:h1, w0:w1, :]
    mixed_amp = np.fft.ifftshift(source_amp, axes=(0, 1))
    mixed = np.fft.ifft2(mixed_amp * np.exp(1j * source_phase), axes=(0, 1)).real
    return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8))


def write_stylized_source_list(
    setup,
    args: argparse.Namespace,
    *,
    client: dict[str, Any],
    round_idx: int,
    phase: int,
) -> Path | None:
    if int(getattr(args, "style_source_repeat", 0)) <= 0:
        return None

    tag = round_tag(round_idx)
    client_tag = f"client{client['id']}_{client['weather']}"
    style_name = f"style_p{phase}_{tag}_{client_tag}_fda_b{float(args.style_beta):.4f}_n{int(args.style_source_limit)}"
    list_path = setup.LIST_ROOT / f"{style_name}_train.txt"
    if list_path.exists() and not args.force:
        return list_path

    source_list = setup.LIST_ROOT / "server_cloudy_train.txt"
    target_list = setup.LIST_ROOT / f"client_{client['id']}_{client['weather']}_target.txt"
    source_images = [Path(line.strip()) for line in source_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    target_images = [Path(line.strip()) for line in target_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not source_images:
        raise RuntimeError(f"No source images found for style transfer: {source_list}")
    if not target_images:
        raise RuntimeError(f"No target images found for style transfer: {target_list}")

    rng = random.Random(int(args.style_seed) + round_idx * 1009 + int(client["id"]) * 9173 + phase * 37)
    limit = int(args.style_source_limit)
    if limit > 0 and limit < len(source_images):
        source_images = rng.sample(source_images, limit)
    else:
        source_images = list(source_images)
        rng.shuffle(source_images)

    image_dir = args.workspace_root / "style_dataset" / "images" / "train" / style_name
    label_dir = args.workspace_root / "style_dataset" / "labels" / "train" / style_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    stats_rows: list[dict[str, Any]] = []
    size = int(args.style_imgsz) if int(args.style_imgsz) > 0 else int(args.imgsz)
    for idx, source_path in enumerate(source_images):
        label_path = image_to_label_path(source_path)
        if not label_path.exists():
            continue
        target_path = target_images[rng.randrange(len(target_images))]
        out_name = f"{idx:05d}_{source_path.stem}_to_{target_path.stem}.jpg"
        out_image = image_dir / out_name
        out_label = label_dir / f"{Path(out_name).stem}.txt"
        if args.force or not out_image.exists() or not out_label.exists():
            with Image.open(source_path) as source_img, Image.open(target_path) as target_img:
                source_img = source_img.convert("RGB").resize((size, size), Image.BILINEAR)
                target_img = target_img.convert("RGB").resize((size, size), Image.BILINEAR)
                styled = fourier_style_transfer(source_img, target_img, args.style_beta)
                styled.save(out_image, quality=92)
            shutil.copy2(label_path, out_label)
        rows.append(str(out_image.resolve()))
        stats_rows.append(
            {
                "phase": phase,
                "round": tag,
                "client": client_tag,
                "source_image": str(source_path),
                "target_style_image": str(target_path),
                "styled_image": str(out_image),
                "label": str(out_label),
                "beta": f"{float(args.style_beta):.6f}",
                "imgsz": size,
            }
        )

    if not rows:
        raise RuntimeError(f"No styled source images were generated for {client_tag}")
    list_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    write_csv(
        args.workspace_root / "stats" / f"18_{style_name}_stats.csv",
        stats_rows,
        ["phase", "round", "client", "source_image", "target_style_image", "styled_image", "label", "beta", "imgsz"],
    )
    print(
        json.dumps(
            {
                "style_source": style_name,
                "client": client_tag,
                "images": len(rows),
                "list": str(list_path.resolve()),
                "beta": float(args.style_beta),
                "imgsz": size,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return list_path


def run_train(setup, fedsto, config: Path, *, dry_run: bool, gpus: int, master_port: int) -> Path:
    return pl03.run_train(setup, fedsto, config, dry_run=dry_run, gpus=gpus, master_port=master_port)


def reusable_checkpoint(fedsto, path: Path, args: argparse.Namespace) -> bool:
    return fedsto.checkpoint_matches_protocol(path, PROTOCOL_VERSION) and pl03.reusable_checkpoint(fedsto, path, args.force)


def write_warmup_config(setup, fedsto, args: argparse.Namespace, weights: Path) -> Path:
    cfg = setup.efficientteacher_config(
        name="sdn18_client_balanced_single_injection_dqamox_warmup",
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
    return setup.write_config("sdn18_client_balanced_single_injection_dqamox_warmup.yaml", cfg)


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
        ema_model = Model(cfg)
        ema_state = intersect_dicts(model.float().state_dict(), ema_model.state_dict(), exclude=["anchor"])
        ema_model.load_state_dict(ema_state, strict=False)
        converted["ema"] = ema_model.half()
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


def apply_train_hyp(
    cfg: dict[str, Any],
    *,
    lr: float,
    loss_box: float,
    loss_cls: float | None = None,
    loss_obj: float | None = None,
    args: argparse.Namespace,
) -> None:
    cfg["linear_lr"] = False
    cfg["hyp"]["lr0"] = float(lr)
    cfg["hyp"]["lrf"] = 1.0
    cfg["hyp"]["warmup_epochs"] = 0
    cfg["hyp"]["mosaic"] = float(getattr(args, "client_mosaic", cfg["hyp"].get("mosaic", 1.0)))
    cfg["hyp"]["mixup"] = float(getattr(args, "client_mixup", 0.0))
    cfg["hyp"]["scale"] = float(getattr(args, "client_scale", 0.25))
    cfg["hyp"]["hsv_s"] = float(getattr(args, "client_hsv_s", 0.35))
    cfg["hyp"]["hsv_v"] = float(getattr(args, "client_hsv_v", 0.20))
    cfg.setdefault("Loss", {})
    cfg["Loss"]["box"] = float(loss_box)
    if loss_cls is not None:
        cfg["Loss"]["cls"] = float(loss_cls)
    if loss_obj is not None:
        cfg["Loss"]["obj"] = float(loss_obj)


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
    style_list: Path | None,
    args: argparse.Namespace,
    router_plan: dict[str, Any] | None = None,
) -> Path:
    tag = round_tag(round_idx)
    client_tag = f"client{client['id']}_{client['weather']}"
    run_name = f"sdn18_{condition}_p{phase}_{tag}_{client_tag}"
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
    cfg["Dataset"]["img_size"] = int(args.imgsz)
    patch_latent_moe_config(cfg, args)
    if router_plan is not None:
        cfg["LatentMoE"]["specialization_target"] = int(router_plan.get("target", -1))
        cfg["LatentMoE"]["specialization_weight"] = float(router_plan.get("weight", 0.0))
    cfg["Dataset"]["train"] = train_expr(
        source_list,
        pseudo_list,
        source_repeat,
        pseudo_repeat,
        style_list=style_list,
        style_repeat=args.style_source_repeat,
    )
    cfg["FedSTO"]["unlabeled_only_client"] = False
    cfg["SSOD"] = {"train_domain": False}
    apply_train_hyp(
        cfg,
        lr=lr,
        loss_box=loss_box,
        loss_cls=args.client_loss_cls,
        loss_obj=args.client_loss_obj,
        args=args,
    )
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
    run_name = f"sdn18_{condition}_p{phase}_{tag}_server_repair"
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
    cfg["Dataset"]["img_size"] = int(args.imgsz)
    patch_latent_moe_config(cfg, args)
    cfg["SSOD"] = {"train_domain": False}
    apply_train_hyp(
        cfg,
        lr=args.server_repair_lr,
        loss_box=args.server_repair_loss_box,
        loss_cls=args.server_repair_loss_cls,
        loss_obj=args.server_repair_loss_obj,
        args=args,
    )
    return setup.write_config(f"{run_name}.yaml", cfg)


def pseudo_stats_to_dqa_stats(pseudo_stats: dict[str, Any], num_classes: int, args: argparse.Namespace) -> list[dqa_v1.ClientClassStats]:
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

    if args.dqa_client_balance_stats:
        totals = [sum(row["counts"]) for row in rows if sum(row["counts"]) > 0]
        if totals:
            if args.dqa_client_balance_target == "mean":
                target_total = float(np.mean(totals))
            elif args.dqa_client_balance_target == "max":
                target_total = float(max(totals))
            else:
                target_total = float(np.median(totals))
            for row in rows:
                total = sum(row["counts"])
                if total <= 0:
                    continue
                scale = min(target_total / total, args.dqa_client_balance_max_scale)
                for key in ("counts", "confidence_sums", "objectness_sums", "class_confidence_sums", "localization_sums", "quality_sums"):
                    row[key] = [float(value) * scale for value in row[key]]
                row["client_balance_scale"] = scale
                row["client_balance_target_total"] = target_total

    return [
        dqa_v1.ClientClassStats.from_mapping(row, num_classes, default_id=f"client{idx}")
        for idx, row in enumerate(rows)
    ]


def image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        try:
            import cv2

            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"cv2 could not read {path}")
            height, width = image.shape[:2]
            return int(width), int(height)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Could not read image size for {path}") from exc


def image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-12, None)


def source_class_priors(setup, num_classes: int) -> tuple[list[float], list[float]]:
    source_list = setup.LIST_ROOT / "server_cloudy_train.txt"
    counts = np.ones(num_classes, dtype=np.float64)
    if not source_list.exists():
        priors = counts / counts.sum()
        return priors.tolist(), np.log(priors).tolist()
    for raw in source_list.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text:
            continue
        image_path = Path(text)
        label_path = image_to_label_path(image_path)
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 1:
                continue
            try:
                cls = int(float(parts[0]))
            except ValueError:
                continue
            if 0 <= cls < num_classes:
                counts[cls] += 1.0
    priors = counts / counts.sum()
    return priors.tolist(), np.log(priors).tolist()


def learned_quality_features_for_rows(
    rows: list[dict[str, Any]],
    *,
    source_priors: list[float],
    source_log_priors: list[float],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if not rows:
        return [], np.zeros((0, 4), dtype=np.float32)

    size_cache: dict[str, tuple[int, int]] = {}
    feature_rows: list[dict[str, Any]] = []
    norm_boxes: list[list[float]] = []
    classes: list[int] = []
    image_keys: list[str] = []

    for row in rows:
        image_key = str(row.get("image") or row.get("source_image"))
        source_image = Path(row.get("source_image") or row.get("image"))
        if str(source_image) not in size_cache:
            size_cache[str(source_image)] = image_size(source_image)
        width, height = size_cache[str(source_image)]
        x1, y1, x2, y2 = parse_xyxy(row["xyxy"])
        nx1 = float(np.clip(x1 / max(width, 1), 0.0, 1.0))
        ny1 = float(np.clip(y1 / max(height, 1), 0.0, 1.0))
        nx2 = float(np.clip(x2 / max(width, 1), 0.0, 1.0))
        ny2 = float(np.clip(y2 / max(height, 1), 0.0, 1.0))
        w = max(0.0, nx2 - nx1)
        h = max(0.0, ny2 - ny1)
        x = nx1 + w / 2.0
        y = ny1 + h / 2.0
        area = w * h
        aspect = w / max(h, 1e-6)
        conf = float(row.get("conf") or 0.0)
        cls = int(row["class_id"])
        prior = source_priors[cls] if 0 <= cls < len(source_priors) else np.nan
        log_prior = source_log_priors[cls] if 0 <= cls < len(source_log_priors) else np.nan
        feature_rows.append(
            {
                "cls": cls,
                "conf": conf,
                "conf_logit": math.log(np.clip(conf, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - conf, 1e-6, 1.0)),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": area,
                "log_area": math.log1p(area),
                "aspect": aspect,
                "log_aspect": math.log(max(aspect, 1e-6)),
                "edge_dist": min(x, y, 1.0 - x, 1.0 - y),
                "source_gt_class_prior": prior,
                "source_gt_class_log_prior": log_prior,
                "split_pred_class_prior": np.nan,
                "pred_gt_prior_ratio": np.nan,
                "aug640_iou": np.nan,
                "aug640_conf": np.nan,
                "aug640_matched": 0.0,
                "plain512_iou": np.nan,
                "plain512_conf": np.nan,
                "plain512_matched": 0.0,
                "plain768_iou": np.nan,
                "plain768_conf": np.nan,
                "plain768_matched": 0.0,
                "agreement_iou_mean": np.nan,
                "agreement_match_count": 0.0,
            }
        )
        norm_boxes.append([nx1, ny1, nx2, ny2])
        classes.append(cls)
        image_keys.append(image_key)

    boxes_arr = np.asarray(norm_boxes, dtype=np.float32)
    by_image: dict[str, list[int]] = {}
    for idx, image_key in enumerate(image_keys):
        by_image.setdefault(image_key, []).append(idx)

    for indices in by_image.values():
        local_boxes = boxes_arr[indices]
        local_classes = np.asarray([classes[idx] for idx in indices], dtype=np.int64)
        ious = box_iou_matrix(local_boxes, local_boxes)
        if len(indices):
            np.fill_diagonal(ious, 0.0)
        for local_idx, global_idx in enumerate(indices):
            same = local_classes == local_classes[local_idx]
            same[local_idx] = False
            if same.any():
                max_same = float(ious[local_idx, same].max())
                near_same = int((ious[local_idx, same] >= 0.50).sum())
            else:
                max_same = 0.0
                near_same = 0
            max_any = float(ious[local_idx].max()) if len(indices) > 1 else 0.0
            feature_rows[global_idx]["image_pred_count"] = len(indices)
            feature_rows[global_idx]["class_pred_count"] = int((local_classes == local_classes[local_idx]).sum())
            feature_rows[global_idx]["max_iou_same_pred"] = max_same
            feature_rows[global_idx]["max_iou_any_pred"] = max_any
            feature_rows[global_idx]["near_same_count_50"] = near_same
            feature_rows[global_idx]["near_any_count_50"] = int((ious[local_idx] >= 0.50).sum())
        ranked = sorted(indices, key=lambda idx: float(feature_rows[idx]["conf"]), reverse=True)
        denom = max(len(ranked), 1)
        for rank, global_idx in enumerate(ranked, start=1):
            feature_rows[global_idx]["rank_conf"] = float(rank)
            feature_rows[global_idx]["rank_conf_norm"] = float(rank / denom)

    return feature_rows, boxes_arr


def apply_learned_quality_to_pseudo_stats(
    setup,
    args: argparse.Namespace,
    pseudo_stats: dict[str, Any],
    round_idx: int,
) -> dict[str, Any]:
    if not args.learned_quality_pseudogt:
        return pseudo_stats

    import joblib
    import pandas as pd

    model_path = args.learned_quality_model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Missing learned quality model: {model_path}")
    bundle = joblib.load(model_path)
    features = list(bundle.get("features") or [])
    pipeline = bundle["pipeline"]
    priors, log_priors = source_class_priors(setup, len(setup.BDD_NAMES))
    tag = round_tag(round_idx)
    summary_rows: list[dict[str, Any]] = []

    for client_tag, stats in pseudo_stats.get("clients", {}).items():
        box_table = Path(stats["box_table"])
        rows = read_csv(box_table)
        if not rows:
            continue
        feature_rows, _ = learned_quality_features_for_rows(rows, source_priors=priors, source_log_priors=log_priors)
        frame = pd.DataFrame(feature_rows)
        missing = [feature for feature in features if feature not in frame.columns]
        for feature in missing:
            frame[feature] = np.nan
        learned = pipeline.predict_proba(frame[features])[:, 1].astype(float)
        original_scores = np.asarray([float(row.get("score") or 0.0) for row in rows], dtype=np.float64)
        for row, original, quality in zip(rows, original_scores, learned):
            row["original_score"] = f"{float(original):.6f}"
            row["learned_quality"] = f"{float(quality):.6f}"
            row["score"] = f"{float(quality):.6f}"
            row["learned_quality_model"] = str(model_path)

        fieldnames = [
            "round",
            "image",
            "source_image",
            "class_id",
            "conf",
            "stability",
            "score",
            "original_score",
            "learned_quality",
            "learned_quality_model",
            "views",
            "xyxy",
        ]
        write_csv(box_table, rows, fieldnames)

        stats["learned_quality_pseudogt"] = {
            "enabled": True,
            "model": str(model_path),
            "candidate": bundle.get("candidate", ""),
            "features": features,
            "mean_original_score": float(original_scores.mean()) if len(original_scores) else 0.0,
            "mean_learned_quality": float(learned.mean()) if len(learned) else 0.0,
            "p10_learned_quality": float(np.quantile(learned, 0.10)) if len(learned) else 0.0,
            "p50_learned_quality": float(np.quantile(learned, 0.50)) if len(learned) else 0.0,
            "p90_learned_quality": float(np.quantile(learned, 0.90)) if len(learned) else 0.0,
        }
        stats["mean_score"] = float(learned.mean()) if len(learned) else 0.0
        summary_rows.append(
            {
                "round": tag,
                "client": client_tag,
                "boxes": len(rows),
                "model": str(model_path),
                "candidate": bundle.get("candidate", ""),
                "mean_original_score": f"{stats['learned_quality_pseudogt']['mean_original_score']:.6f}",
                "mean_learned_quality": f"{stats['learned_quality_pseudogt']['mean_learned_quality']:.6f}",
                "p10_learned_quality": f"{stats['learned_quality_pseudogt']['p10_learned_quality']:.6f}",
                "p50_learned_quality": f"{stats['learned_quality_pseudogt']['p50_learned_quality']:.6f}",
                "p90_learned_quality": f"{stats['learned_quality_pseudogt']['p90_learned_quality']:.6f}",
                "box_table": str(box_table.resolve()),
            }
        )

    write_csv(
        args.workspace_root / "stats" / f"28_{tag}_learned_quality_pseudogt_stats.csv",
        summary_rows,
        [
            "round",
            "client",
            "boxes",
            "model",
            "candidate",
            "mean_original_score",
            "mean_learned_quality",
            "p10_learned_quality",
            "p50_learned_quality",
            "p90_learned_quality",
            "box_table",
        ],
    )
    pseudo_stats["learned_quality_pseudogt"] = {
        "enabled": True,
        "model": str(model_path),
        "candidate": bundle.get("candidate", ""),
        "summary_csv": str((args.workspace_root / "stats" / f"28_{tag}_learned_quality_pseudogt_stats.csv").resolve()),
    }
    return pseudo_stats


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
        box_table = args.workspace_root / "stats" / f"18_{tag}_{client_tag}_specialist_selected_boxes.csv"
        write_csv(box_table, selected, fields)
        stats["box_table"] = str(box_table.resolve())
        stats["specialist_selection"] = "expert_choice_selected_boxes"
    return pseudo_stats


def rebalance_selected_box_tables(
    setup,
    args: argparse.Namespace,
    pseudo_stats: dict[str, Any],
    round_idx: int,
) -> dict[str, Any]:
    """Trim selected pseudoGT using the actual selected-count class fraction.

    The inherited expert-choice selector caps each class relative to the target
    number of boxes. When pseudoGT is scarce, especially for night clients, the
    actual selected total can be much smaller than that target, so dominant easy
    classes still become one-third of the training signal. This second pass caps
    classes against the actual selected pool, then rewrites the pseudo dataset.
    """

    if args.actual_max_class_fraction <= 0:
        return attach_selected_box_tables(args, pseudo_stats, round_idx)

    tag = round_tag(round_idx)
    selected_path = args.workspace_root / "stats" / f"05_{tag}_expert_choice_boxes.csv"
    rows = read_csv(selected_path)
    by_client: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_client.setdefault(row["client"], []).append(row)

    all_selected_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for client_tag, client_rows in sorted(by_client.items()):
        if not client_rows:
            continue
        original_class_counts: dict[int, int] = {}
        for row in client_rows:
            cls = int(row["class_id"])
            original_class_counts[cls] = original_class_counts.get(cls, 0) + 1
        max_per_class = max(1, int(np.ceil(len(client_rows) * args.actual_max_class_fraction)))
        for candidate in range(max_per_class, 0, -1):
            projected_total = sum(min(count, candidate) for count in original_class_counts.values())
            if projected_total <= 0 or candidate / projected_total <= args.actual_max_class_fraction:
                max_per_class = candidate
                break
        kept: list[dict[str, Any]] = []
        class_counts: dict[int, int] = {}
        ranked = sorted(
            client_rows,
            key=lambda row: (float(row.get("score") or 0.0), float(row.get("stability") or 0.0), float(row.get("conf") or 0.0)),
            reverse=True,
        )
        for row in ranked:
            cls = int(row["class_id"])
            if class_counts.get(cls, 0) >= max_per_class:
                continue
            rewritten = dict(row)
            rewritten["image"] = row.get("source_pseudo_image") or row["image"]
            kept.append(rewritten)
            class_counts[cls] = class_counts.get(cls, 0) + 1

        # Force rewrite because ec05 already wrote the pre-rebalance dataset
        # into the same round/client pseudo directory.
        writer_args = copy.copy(args)
        writer_args.force_pseudo = True
        train_list, selected_stats, selected_rows = ec05.write_selected_pseudo_dataset(
            setup,
            writer_args,
            client_tag,
            round_idx,
            kept,
        )
        box_table = args.workspace_root / "stats" / f"18_{tag}_{client_tag}_actual_class_balanced_boxes.csv"
        write_csv(
            box_table,
            selected_rows,
            [
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
            ],
        )
        client_stats = pseudo_stats["clients"][client_tag]
        client_stats.update(selected_stats)
        client_stats["train_list"] = str(train_list.resolve())
        client_stats["box_table"] = str(box_table.resolve())
        client_stats["actual_class_rebalance"] = {
            "enabled": True,
            "actual_max_class_fraction": args.actual_max_class_fraction,
            "input_selected_boxes": len(client_rows),
            "output_selected_boxes": len(selected_rows),
            "max_per_class": max_per_class,
        }
        all_selected_rows.extend(selected_rows)
        summary_rows.append(
            {
                "round": tag,
                "client": client_tag,
                "input_selected_boxes": len(client_rows),
                "output_selected_boxes": len(selected_rows),
                "actual_max_class_fraction": args.actual_max_class_fraction,
                "max_per_class": max_per_class,
                "train_list": str(train_list.resolve()),
                "box_table": str(box_table.resolve()),
            }
        )

    write_csv(
        args.workspace_root / "stats" / f"18_{tag}_actual_class_rebalance_stats.csv",
        summary_rows,
        [
            "round",
            "client",
            "input_selected_boxes",
            "output_selected_boxes",
            "actual_max_class_fraction",
            "max_per_class",
            "train_list",
            "box_table",
        ],
    )
    write_csv(
        args.workspace_root / "stats" / f"18_{tag}_actual_class_rebalance_boxes.csv",
        all_selected_rows,
        [
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
        ],
    )
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
        moe_expert_blend=args.dqa_moe_expert_blend,
        moe_router_blend=args.dqa_moe_router_blend,
        bn_blend=args.dqa_bn_blend,
    )


def scheduled_args_for_round(args: argparse.Namespace, *, phase: int, round_idx: int) -> argparse.Namespace:
    """Return the round-local arguments for the curriculum stage.

    Early Phase1 is intentionally close to 11: it tests whether a full-from-
    warmup MoE detector can survive long pseudoGT rounds without collapsing.
    Once it has stabilized, late Phase1 and Phase2 shift from protection to
    expansion by using more selected pseudo boxes and a weaker server anchor.
    """

    round_args = copy.copy(args)
    is_expansion = phase == 2 or (phase == 1 and round_idx >= args.curriculum_start_round)
    if not is_expansion:
        return round_args

    round_args.expert_keep_fraction = args.late_expert_keep_fraction
    round_args.expert_max_class_fraction = args.late_expert_max_class_fraction
    round_args.dqa_server_anchor = args.late_dqa_server_anchor
    round_args.dqa_min_server_alpha = args.late_dqa_min_server_alpha
    round_args.dqa_residual_blend = args.late_dqa_residual_blend
    round_args.min_score = args.late_min_score
    round_args.min_stability = args.late_min_stability
    round_args.actual_max_class_fraction = args.late_actual_max_class_fraction
    return round_args


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
    repair_start = fedsto.GLOBAL_DIR / f"sdn18_{condition}_p{phase}_{tag}_server_repair_start.pt"
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
        write_checkpoint_records(args.workspace_root / "stats" / "18_repair_baseline_checkpoints.csv", records)
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
    round_args = scheduled_args_for_round(args, phase=phase, round_idx=round_idx)
    if phase == 1 and round_idx >= args.curriculum_start_round:
        client_lr = args.late_phase1_client_lr
        source_repeat = args.late_phase1_source_repeat
        pseudo_repeat = args.late_phase1_pseudo_repeat
        loss_box = args.late_phase1_loss_box
    print(f"\n=== fixed_pseudolabel_path_dqamox phase {phase} {tag}: pseudo labels ===")
    pseudo_teacher: Path | list[Path] = current
    if round_args.pseudo_teacher_checkpoints:
        pseudo_teacher = parse_checkpoint_list(round_args.pseudo_teacher_checkpoints)
        print(f"Using external pseudo-teacher checkpoint ensemble with {len(pseudo_teacher)} weights")
    if getattr(round_args, "use_local_ema_teacher", False) and not round_args.pseudo_teacher_checkpoints:
        raw_pseudo_stats = generate_round_pseudo_labels_with_optional_local_ema(
            setup,
            current,
            round_args,
            clients,
            round_idx,
        )
    else:
        raw_pseudo_stats = pl03.generate_round_pseudo_labels(setup, pseudo_teacher, round_args, clients, round_idx)
    raw_pseudo_stats = apply_learned_quality_to_pseudo_stats(setup, round_args, raw_pseudo_stats, round_idx)
    pseudo_stats, next_load_bias_state = ec05.apply_expert_choice_selection(
        setup,
        round_args,
        raw_pseudo_stats,
        round_idx,
        load_bias_state,
    )
    pseudo_stats = rebalance_selected_box_tables(setup, round_args, pseudo_stats, round_idx)
    records: list[dict[str, str]] = []
    local_paths: list[Path] = []
    router_plan_rows: list[dict[str, Any]] = []
    expert_assignments: list[dict[str, Any]] = []

    for client in clients:
        client_tag = f"client{client['id']}_{client['weather']}"
        router_plan = router_specialization_plan(client, pseudo_stats["clients"].get(client_tag, {}), round_args)
        expert_assignments.append(
            {
                "client_id": client_tag,
                "target_expert": int(router_plan.get("target", -1)),
                "weight": float(router_plan.get("weight", 0.0)),
                "quality_signal": float(router_plan.get("quality_signal", 0.0)),
                "reason": router_plan.get("reason", ""),
            }
        )
        router_plan_rows.append(
            {
                "phase": phase,
                "round": tag,
                "client": client_tag,
                "target_expert": router_plan.get("target", -1),
                "specialization_weight": f"{float(router_plan.get('weight', 0.0)):.8f}",
                "reason": router_plan.get("reason", ""),
                "quality_signal": f"{float(router_plan.get('quality_signal', 0.0)):.6f}",
                "quality_gate": f"{float(router_plan.get('quality_gate', 0.0)):.6f}",
                "count_gate": f"{float(router_plan.get('count_gate', 0.0)):.6f}",
                "boxes": f"{float(router_plan.get('boxes', 0.0)):.1f}",
                "vru_fraction": f"{float(router_plan.get('vru_fraction', 0.0)):.6f}",
                "vehicle_fraction": f"{float(router_plan.get('vehicle_fraction', 0.0)):.6f}",
                "traffic_fraction": f"{float(router_plan.get('traffic_fraction', 0.0)):.6f}",
            }
        )
        start = fedsto.CLIENT_STATE_DIR / f"sdn18_p{phase}_{tag}_{client_tag}_start.pt"
        final = args.workspace_root / "checkpoints" / f"latent_dqamox_p{phase}_{tag}_{client_tag}.pt"
        if not round_args.dry_run and not fedsto.checkpoint_matches_protocol(start, PROTOCOL_VERSION):
            local_teacher = client_local_ema_checkpoint(round_args, client)
            fedsto.make_start_checkpoint(
                current,
                start,
                local_teacher if getattr(round_args, "use_local_ema_teacher", False) and local_teacher.exists() else None,
                protocol=PROTOCOL_VERSION,
                stage=f"latent_dqamox_p{phase}_{tag}_{client_tag}_start",
            )

        if not reusable_checkpoint(fedsto, final, round_args):
            style_list = write_stylized_source_list(
                setup,
                round_args,
                client=client,
                round_idx=round_idx,
                phase=phase,
            )
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
                style_list=style_list,
                args=round_args,
                router_plan=router_plan,
            )
            raw = run_train(
                setup,
                fedsto,
                cfg,
                dry_run=round_args.dry_run,
                gpus=round_args.gpus,
                master_port=round_args.master_port + port_offset,
            )
            port_offset += 1
            if not round_args.dry_run:
                fedsto.mark_checkpoint_protocol(raw, PROTOCOL_VERSION, f"latent_dqamox_p{phase}_{tag}_{client_tag}_raw")
                fedsto.make_start_checkpoint(raw, final, protocol=PROTOCOL_VERSION, stage=f"latent_dqamox_p{phase}_{tag}_{client_tag}")
                pl03.cleanup_training_artifacts(raw, start)
        if getattr(round_args, "use_local_ema_teacher", False) and not round_args.dry_run and final.exists():
            fedsto.make_start_checkpoint(
                final,
                client_local_ema_checkpoint(round_args, client),
                protocol=PROTOCOL_VERSION,
                stage=f"{client_tag}_local_ema_teacher_after_p{phase}_{tag}",
            )

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
            variant=(
                f"scope={train_scope}:lr={client_lr}:source={source_repeat}:pseudo={pseudo_repeat}:box={loss_box}:"
                f"router_target={router_plan.get('target', -1)}:router_weight={float(router_plan.get('weight', 0.0)):.6f}"
            ),
        )

    if router_plan_rows:
        write_csv(
            args.workspace_root / "stats" / f"18_p{phase}_{tag}_router_specialization.csv",
            router_plan_rows,
            [
                "phase",
                "round",
                "client",
                "target_expert",
                "specialization_weight",
                "reason",
                "quality_signal",
                "quality_gate",
                "count_gate",
                "boxes",
                "vru_fraction",
                "vehicle_fraction",
                "traffic_fraction",
            ],
        )

    aggregate = args.workspace_root / "checkpoints" / f"latent_dqamox_p{phase}_{tag}_dqa_aggregate.pt"
    state_path = args.workspace_root / "stats" / "18_latent_dqamox_dqa_state.json"
    if not round_args.dry_run and not reusable_checkpoint(fedsto, aggregate, round_args):
        stats = pseudo_stats_to_dqa_stats(pseudo_stats, num_classes=len(setup.BDD_NAMES), args=round_args)
        _, dqa_state = dqa_v2.aggregate_checkpoints(
            client_checkpoints=local_paths,
            server_checkpoint=current,
            output_checkpoint=aggregate,
            stats=stats,
            state_path=state_path,
            config=dqa_config(round_args, len(setup.BDD_NAMES)),
            repo_root=REPO_ROOT,
            expert_assignments=expert_assignments,
        )
        (args.workspace_root / "stats" / f"18_p{phase}_{tag}_dqa_state_snapshot.json").write_text(
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
        round_args,
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
                    "latent_dqamox_final_aggregate": "warmup + fixed-pseudo-label-path DQA-MoX aggregate",
                    "latent_dqamox_final_repair": "warmup + fixed-pseudo-label-path DQA-MoX + server repair",
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
    write_csv(args.workspace_root / "stats" / "18_client_balanced_single_injection_dqamox_final_metrics.csv", metric_rows, fields)

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
        args.workspace_root / "stats" / "18_client_balanced_single_injection_dqamox_split_metrics.csv",
        split_rows,
        ["checkpoint_label", "condition", "split", "images", "labels", "precision", "recall", "map50", "map50_95"],
    )
    return metric_rows


def run_evaluation(args: argparse.Namespace, eval_records: list[dict[str, str]]) -> None:
    original_argv = sys.argv
    try:
        # The shared evaluator defaults to 640.  DQA-MoX 17+ can deliberately
        # test high-resolution learning/evaluation, so pass the requested scale
        # through without changing the older 640-default behavior.
        base01_0.run_evaluation(args, eval_records)
    finally:
        sys.argv = original_argv


def write_report(args: argparse.Namespace, metrics: list[dict[str, Any]], run_manifest: dict[str, Any]) -> Path:
    path = args.workspace_root / "18_client_balanced_single_injection_dqamox_report.md"
    lines = [
        "# 18 Fixed-Path Localization-Curriculum Full-From-Warmup DQA-MoX Report",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- protocol: `{PROTOCOL_VERSION}`",
        f"- workspace: `{args.workspace_root.resolve()}`",
        f"- target_map50: {args.target_map50:.3f}",
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
            "- When router specialization is enabled, client/domain/class assignment is written to `18_p*_router_specialization.csv` and gated by DQA pseudoGT quality.",
            "- The new part is pseudoGT selection: expert-choice buckets reduce class imbalance before client training and before DQA statistics.",
            "- DQA remains in the selected pseudoGT statistics and classwise server-anchored aggregation; MoE remains inside the detector head.",
            "- The key comparison is `latent_dqamox_final_repair` vs `warmup_server_repair_final` on total and each scene/day-night split.",
            "- The target for this run is to push final total mAP50 to at least the configured target.",
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
    post = args.post_dqa_repair_rounds * args.estimated_repair_round_minutes
    eval_minutes = args.estimated_eval_minutes if args.evaluate else 0.0
    return (warmup + repair + dqa + post + eval_minutes) * 60.0


def progress_factory(args: argparse.Namespace, total: int):
    if args.no_progress:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:  # noqa: BLE001
        return None
    return tqdm(total=total, desc="18 Fixed-path localization-curriculum full-from-warmup DQA-MoX", unit="step")


def clients_for_round(args: argparse.Namespace, clients: list[dict[str, Any]], round_idx: int) -> list[dict[str, Any]]:
    """FedMoX-style deterministic online-client sampling.

    FedMoX samples a fraction of clients per communication round.  The default
    ratio is 1.0 to preserve existing notebooks; paper-aligned runs can set
    0.33 so a 6-client scene/day-night split trains 2 clients per round.
    """

    ratio = float(args.client_sampling_ratio)
    if ratio >= 1.0 or len(clients) <= 1:
        return clients
    count = max(1, min(len(clients), round(len(clients) * max(ratio, 0.0))))
    rng = random.Random(int(args.client_sampling_seed) + int(round_idx))
    selected = rng.sample(clients, count)
    return sorted(selected, key=lambda row: int(row.get("id", 0)))


def record_round_clients(args: argparse.Namespace, *, phase: int, round_idx: int, clients: list[dict[str, Any]]) -> None:
    path = args.workspace_root / "stats" / "18_round_client_sampling.csv"
    rows = []
    if path.exists():
        rows = read_csv(path)
    rows.append(
        {
            "phase": str(phase),
            "round": str(round_idx),
            "client_sampling_ratio": f"{float(args.client_sampling_ratio):.6f}",
            "client_count": str(len(clients)),
            "clients": ",".join(f"client{client.get('id')}_{client.get('scene')}_{client.get('time')}" for client in clients),
        }
    )
    write_csv(path, rows, ["phase", "round", "client_sampling_ratio", "client_count", "clients"])


def client_local_ema_checkpoint(args: argparse.Namespace, client: dict[str, Any]) -> Path:
    client_tag = f"client{client['id']}_{client['weather']}"
    return args.workspace_root / "client_states" / f"{client_tag}_local_ema_teacher.pt"


def local_ema_teacher_or_anchor(args: argparse.Namespace, current: Path, client: dict[str, Any]) -> Path:
    if not getattr(args, "use_local_ema_teacher", False):
        return current
    local_teacher = client_local_ema_checkpoint(args, client)
    return local_teacher if local_teacher.exists() else current


def generate_round_pseudo_labels_with_optional_local_ema(
    setup,
    current: Path,
    args: argparse.Namespace,
    clients: list[dict[str, Any]],
    round_idx: int,
) -> dict[str, Any]:
    """Generate per-client pseudo labels from local EMA teachers when enabled.

    The default path remains the historical server-anchor teacher.  With
    --use-local-ema-teacher, each selected client uses its persisted EMA as the
    pseudo-label teacher when available, falling back to the current global
    anchor the first time that client is sampled.
    """

    if not getattr(args, "use_local_ema_teacher", False):
        return pl03.generate_round_pseudo_labels(setup, current, args, clients, round_idx)

    merged: dict[str, Any] | None = None
    teacher_rows: list[dict[str, Any]] = []
    first = True
    for client in clients:
        client_tag = f"client{client['id']}_{client['weather']}"
        teacher = local_ema_teacher_or_anchor(args, current, client)
        one_args = copy.copy(args)
        one_args.force_pseudo = bool(getattr(args, "force_pseudo", False) and first)
        payload = pl03.generate_round_pseudo_labels(setup, teacher, one_args, [client], round_idx)
        if merged is None:
            merged = dict(payload)
            merged["clients"] = {}
            merged["teacher"] = "per_client_local_ema_with_server_anchor_fallback"
            merged["teacher_by_client"] = {}
        merged["clients"].update(payload.get("clients", {}))
        merged["teacher_by_client"][client_tag] = str(teacher.expanduser().resolve())
        teacher_rows.append(
            {
                "round": round_tag(round_idx),
                "client": client_tag,
                "teacher": str(teacher.expanduser().resolve()),
                "teacher_role": "local_ema" if teacher != current else "server_anchor_fallback",
            }
        )
        first = False

    if merged is None:
        raise RuntimeError("No clients were provided for pseudo-label generation.")

    stats_dir = args.workspace_root / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        stats_dir / f"18_{round_tag(round_idx)}_local_ema_teacher_map.csv",
        teacher_rows,
        ["round", "client", "teacher", "teacher_role"],
    )
    (stats_dir / f"18_{round_tag(round_idx)}_merged_local_ema_pseudo_label_stats.json").write_text(
        json.dumps(merged, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return merged


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
            "router_specialization_weight": args.router_specialization_weight,
            "router_specialization_map": args.router_specialization_map,
            "router_specialization_min_quality": args.router_specialization_min_quality,
            "router_specialization_min_boxes": args.router_specialization_min_boxes,
            "router_specialization_max_weight": args.router_specialization_max_weight,
            "router_specialization_class_threshold": args.router_specialization_class_threshold,
            "class_skew_residual": args.class_skew_residual,
            "class_skew_orthogonal_weight": args.class_skew_orthogonal_weight,
            "class_skew_srip_weight": args.class_skew_srip_weight,
            "class_skew_residual_weight": args.class_skew_residual_weight,
            "expert_semantics": "DQA-gated client/domain/class specialization when router_specialization_weight > 0",
        },
        "schedule": {
            "warmup_epochs": args.warmup_epochs,
            "repair_baseline_rounds": args.repair_baseline_rounds,
            "phase1_rounds": args.phase1_rounds,
            "phase2_rounds": args.phase2_rounds,
            "post_dqa_repair_rounds": args.post_dqa_repair_rounds,
            "client_sampling_ratio": args.client_sampling_ratio,
            "client_sampling_seed": args.client_sampling_seed,
            "phase1_train_scope": args.phase1_train_scope,
            "phase2_train_scope": args.phase2_train_scope,
            "curriculum_start_round": args.curriculum_start_round,
            "late_phase1_client_lr": args.late_phase1_client_lr,
            "late_phase1_source_repeat": args.late_phase1_source_repeat,
            "late_phase1_pseudo_repeat": args.late_phase1_pseudo_repeat,
            "late_phase1_loss_box": args.late_phase1_loss_box,
            "client_loss_cls": args.client_loss_cls,
            "client_loss_obj": args.client_loss_obj,
            "server_repair_loss_cls": args.server_repair_loss_cls,
            "server_repair_loss_obj": args.server_repair_loss_obj,
        },
        "target": {
            "metric": "paper_protocol_total_map50",
            "target_map50": args.target_map50,
        },
        "pseudo_selection": {
            "method": "expert_choice_fedmox_full_balanced",
            "imgsz": args.imgsz,
            "pseudo_imgsz": args.pseudo_imgsz or args.imgsz,
            "pseudo_teacher_checkpoints": [
                str(path) for path in parse_checkpoint_list(args.pseudo_teacher_checkpoints)
            ]
            if args.pseudo_teacher_checkpoints
            else [],
            "use_local_ema_teacher": bool(args.use_local_ema_teacher),
            "local_ema_teacher_role": (
                "selected clients persist one local EMA teacher across rounds; "
                "server anchor is used only as fallback/comparison and is not stored as a client model"
            ),
            "expert_count": args.expert_count,
            "keep_fraction": args.expert_keep_fraction,
            "max_class_fraction": args.expert_max_class_fraction,
            "actual_max_class_fraction": args.actual_max_class_fraction,
            "load_bias_strength": args.load_bias_strength,
            "late_keep_fraction": args.late_expert_keep_fraction,
            "late_max_class_fraction": args.late_expert_max_class_fraction,
            "late_actual_max_class_fraction": args.late_actual_max_class_fraction,
            "late_min_score": args.late_min_score,
            "late_min_stability": args.late_min_stability,
            "learned_quality_pseudogt": {
                "enabled": args.learned_quality_pseudogt,
                "model": str(args.learned_quality_model.expanduser().resolve()) if args.learned_quality_pseudogt else "",
                "role": "pseudoGT verifier only; replaces pseudo box score before expert-choice selection and DQA stats",
            },
        },
        "style_source_adaptation": {
            "enabled": args.style_source_repeat > 0,
            "method": "FDA target-style source-GT replay",
            "repeat": args.style_source_repeat,
            "source_limit": args.style_source_limit,
            "beta": args.style_beta,
            "imgsz": args.style_imgsz or args.imgsz,
            "role": "client target appearance is injected into source images while source GT boxes remain the only supervised labels",
        },
        "post_dqa_consolidation": {
            "enabled": args.post_dqa_repair_rounds > 0,
            "rounds": args.post_dqa_repair_rounds,
            "train_scope": args.post_dqa_repair_train_scope,
            "lr": args.server_repair_lr,
            "loss_box": args.server_repair_loss_box,
            "loss_cls": args.server_repair_loss_cls,
            "loss_obj": args.server_repair_loss_obj,
            "reason": "keep the early DQA/MoE specialization but stop repeated pseudoGT self-training drift",
        },
        "client_balanced_dqa_stats": {
            "enabled": args.dqa_client_balance_stats,
            "target": args.dqa_client_balance_target,
            "max_scale": args.dqa_client_balance_max_scale,
            "reason": "night and rare-scene clients have fewer selected pseudo boxes, so raw count-weighted DQA can let easy day/citystreet clients dominate aggregation",
        },
        "aggregation_curriculum": {
            "early_server_anchor": args.dqa_server_anchor,
            "early_min_server_alpha": args.dqa_min_server_alpha,
            "early_residual_blend": args.dqa_residual_blend,
            "moe_expert_blend": args.dqa_moe_expert_blend,
            "moe_router_blend": args.dqa_moe_router_blend,
            "bn_blend": args.dqa_bn_blend,
            "late_server_anchor": args.late_dqa_server_anchor,
            "late_min_server_alpha": args.late_dqa_min_server_alpha,
            "late_residual_blend": args.late_dqa_residual_blend,
        },
        "server": manifest.get("server"),
        "clients": clients,
    }
    (args.workspace_root / "stats" / "18_client_balanced_single_injection_dqamox_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.setup_only:
        return

    if not args.allow_cpu_training and int(args.gpus) > 0:
        visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if visible_gpus < int(args.gpus):
            raise RuntimeError(
                f"Requested {args.gpus} CUDA GPUs, but only {visible_gpus} are visible. "
                "Refusing to start full training on CPU. Restore CUDA/NVML or pass "
                "--allow-cpu-training for an intentional CPU debug run."
            )

    total_steps = (
        1
        + args.repair_baseline_rounds
        + args.phase1_rounds
        + args.phase2_rounds
        + args.post_dqa_repair_rounds
        + (1 if args.evaluate else 0)
    )
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
            f"DQA 18 progress: {stage}, ETA {seconds_to_hms(eta)}.",
            title="DQA 18 progress",
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
        round_clients = clients_for_round(args, clients, idx)
        record_round_clients(args, phase=1, round_idx=idx, clients=round_clients)
        round_records, dqa_current, pseudo_stats, load_bias_state, port_offset = run_dqa_round(
            setup,
            fedsto,
            dqa_current,
            args,
            round_clients,
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
        write_checkpoint_records(args.workspace_root / "stats" / "18_latent_dqamox_checkpoints.csv", dqa_records)
        done = 1 + args.repair_baseline_rounds + idx
        if progress is not None:
            elapsed = time.monotonic() - start_time
            eta = elapsed / max(done, 1) * max(total_steps - done, 0)
            progress.set_postfix(stage="phase1", round=idx, eta=seconds_to_hms(eta))
            progress.update(1)
        maybe_notify_progress("phase1", done, dqa_current)

    for idx in range(1, args.phase2_rounds + 1):
        global_round_idx = args.phase1_rounds + idx
        round_clients = clients_for_round(args, clients, global_round_idx)
        record_round_clients(args, phase=2, round_idx=global_round_idx, clients=round_clients)
        round_records, dqa_current, pseudo_stats, load_bias_state, port_offset = run_dqa_round(
            setup,
            fedsto,
            dqa_current,
            args,
            round_clients,
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
        write_checkpoint_records(args.workspace_root / "stats" / "18_latent_dqamox_checkpoints.csv", dqa_records)
        if progress is not None:
            progress.set_postfix(stage="phase2", round=global_round_idx)
            progress.update(1)
        completed_steps = 1 + args.repair_baseline_rounds + args.phase1_rounds + idx
        maybe_notify_progress("phase2", completed_steps, dqa_current)

    final_phase = 2 if args.phase2_rounds > 0 else 1
    final_round = args.phase1_rounds + args.phase2_rounds if args.phase2_rounds > 0 else args.phase1_rounds
    final_aggregate_label = f"latent_dqamox_p{final_phase}_{round_tag(final_round)}_dqa_aggregate"
    final_repair_label = f"latent_dqamox_p{final_phase}_{round_tag(final_round)}_server_repair"

    consolidation_records: list[dict[str, str]] = []
    for idx in range(1, args.post_dqa_repair_rounds + 1):
        print(f"\n=== post-DQA source consolidation {round_tag(idx)} ===")
        round_records, dqa_current, port_offset = run_server_repair_round(
            setup,
            fedsto,
            dqa_current,
            args,
            condition="latent_dqamox_consolidation",
            phase=3,
            round_idx=idx,
            train_scope=args.post_dqa_repair_train_scope,
            port_offset=port_offset,
        )
        consolidation_records.extend(round_records)
        final_repair_label = f"latent_dqamox_consolidation_p3_{round_tag(idx)}_server_repair"
        write_checkpoint_records(args.workspace_root / "stats" / "18_latent_dqamox_consolidation_checkpoints.csv", consolidation_records)
        if progress is not None:
            progress.set_postfix(stage="post_repair", round=idx)
            progress.update(1)
        completed_steps = 1 + args.repair_baseline_rounds + args.phase1_rounds + args.phase2_rounds + idx
        maybe_notify_progress("post_dqa_consolidation", completed_steps, dqa_current)

    records.extend(dqa_records)
    records.extend(consolidation_records)
    write_checkpoint_records(args.workspace_root / "stats" / "18_all_checkpoints.csv", records)

    by_label = {row["label"]: row for row in records}
    if final_repair_label not in by_label:
        final_repair_label = final_aggregate_label
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
    write_checkpoint_records(args.workspace_root / "stats" / "18_eval_checkpoints.csv", eval_records)

    run_manifest = {
        **payload,
        "actual_runtime_seconds": time.monotonic() - start_time,
        "actual_runtime_hms": seconds_to_hms(time.monotonic() - start_time),
        "records": records,
        "eval_records": eval_records,
        "pseudo_history": pseudo_history,
    }
    (args.workspace_root / "stats" / "18_client_balanced_single_injection_dqamox_run_manifest.json").write_text(
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
            "target_map50": args.target_map50,
            "experts": args.num_experts,
            "phase1_rounds": args.phase1_rounds,
            "phase2_rounds": args.phase2_rounds,
            "post_dqa_repair_rounds": args.post_dqa_repair_rounds,
        }
        if status:
            context["status"] = status
        if error:
            context["error"] = error[:500]
        if extra_context:
            context.update(extra_context)
        metrics_path = args.workspace_root.expanduser().resolve() / "stats" / "18_client_balanced_single_injection_dqamox_final_metrics.csv"
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
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT / "output" / "18_client_balanced_single_injection_dqamox")
    parser.add_argument("--source-workspace", type=Path, default=PROJECT_ROOT / "output" / "08_full_latent_dqamox_from_warmup")
    parser.add_argument("--source-repair-baseline-rounds", type=int, default=30)
    parser.add_argument("--warmup-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--pseudo-teacher-checkpoints",
        default="",
        help="Comma-separated checkpoints used only for pseudo-label generation; training still starts from the current global checkpoint.",
    )
    parser.add_argument(
        "--pseudo-teacher-separate-model-views",
        action="store_true",
        help="When pseudo-teacher-checkpoints contains multiple weights, run each self-checkpoint as its own consensus view before stable-box clustering.",
    )
    parser.add_argument(
        "--use-local-ema-teacher",
        action="store_true",
        help=(
            "Persist one local EMA teacher per client and use it for that client's pseudo labels when sampled; "
            "fallback is the current server/global anchor before the client has an EMA."
        ),
    )
    parser.add_argument(
        "--learned-quality-pseudogt",
        action="store_true",
        help="Replace pseudo box score with a source-calibrated learned verifier before expert-choice pseudoGT selection.",
    )
    parser.add_argument(
        "--learned-quality-model",
        type=Path,
        default=DQA_ROOT / "source_calibrated_localization_quality" / "artifacts" / "rscolq_best.joblib",
        help="Joblib bundle containing the learned pseudoGT quality verifier.",
    )
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None)
    parser.add_argument("--skip-warmup-training", action="store_true")
    parser.add_argument("--force-pretrained", action="store_true")
    parser.add_argument("--client-limit", type=int, default=1500)
    parser.add_argument("--clients", default="all")
    parser.add_argument("--warmup-epochs", type=int, default=50)
    parser.add_argument("--warmup-lr", type=float, default=0.01)
    parser.add_argument("--warmup-lrf", type=float, default=0.2)
    parser.add_argument("--repair-baseline-rounds", type=int, default=0)
    parser.add_argument("--phase1-rounds", type=int, default=1)
    parser.add_argument("--phase2-rounds", type=int, default=0)
    parser.add_argument("--client-sampling-ratio", type=float, default=1.0)
    parser.add_argument("--client-sampling-seed", type=int, default=20260511)
    train_scope_choices = ["backbone", "neck_head", "bn", "moe_head", "bn_moe_head", "backbone_moe_head", "all"]
    parser.add_argument("--phase1-train-scope", choices=train_scope_choices, default="neck_head")
    parser.add_argument("--phase1-repair-train-scope", choices=train_scope_choices, default="neck_head")
    parser.add_argument("--phase1-client-epochs", type=int, default=1)
    parser.add_argument("--phase1-client-lr", type=float, default=0.0006)
    parser.add_argument("--phase1-source-repeat", type=int, default=3)
    parser.add_argument("--phase1-pseudo-repeat", type=int, default=1)
    parser.add_argument("--phase1-loss-box", type=float, default=0.01)
    parser.add_argument("--curriculum-start-round", type=int, default=999)
    parser.add_argument("--late-phase1-client-lr", type=float, default=0.0005)
    parser.add_argument("--late-phase1-source-repeat", type=int, default=2)
    parser.add_argument("--late-phase1-pseudo-repeat", type=int, default=2)
    parser.add_argument("--late-phase1-loss-box", type=float, default=0.0005)
    parser.add_argument("--phase2-train-scope", choices=train_scope_choices, default="all")
    parser.add_argument("--phase2-repair-train-scope", choices=train_scope_choices, default="all")
    parser.add_argument("--phase2-client-epochs", type=int, default=1)
    parser.add_argument("--phase2-client-lr", type=float, default=0.0003)
    parser.add_argument("--phase2-source-repeat", type=int, default=2)
    parser.add_argument("--phase2-pseudo-repeat", type=int, default=1)
    parser.add_argument("--phase2-loss-box", type=float, default=0.003)
    parser.add_argument("--server-repair-epochs", type=int, default=1)
    parser.add_argument("--server-repair-lr", type=float, default=0.0007)
    parser.add_argument("--server-repair-loss-box", type=float, default=0.05)
    parser.add_argument("--client-loss-cls", type=float, default=None)
    parser.add_argument("--client-loss-obj", type=float, default=None)
    parser.add_argument("--server-repair-loss-cls", type=float, default=None)
    parser.add_argument("--server-repair-loss-obj", type=float, default=None)
    parser.add_argument("--post-dqa-repair-rounds", type=int, default=0)
    parser.add_argument("--post-dqa-repair-train-scope", choices=train_scope_choices, default="neck_head")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--router-temperature", type=float, default=1.3)
    parser.add_argument("--moe-scale", type=float, default=1.0)
    parser.add_argument("--router-balance-weight", type=float, default=0.03)
    parser.add_argument("--router-entropy-weight", type=float, default=0.002)
    parser.add_argument("--router-specialization-weight", type=float, default=0.0)
    parser.add_argument("--router-specialization-target", type=int, default=-1)
    parser.add_argument(
        "--router-specialization-map",
        choices=["none", "domain4", "domain6", "client4", "client6", "class4", "hybrid_dqa4"],
        default="none",
    )
    parser.add_argument("--router-specialization-min-quality", type=float, default=0.55)
    parser.add_argument("--router-specialization-min-boxes", type=float, default=500.0)
    parser.add_argument("--router-specialization-max-weight", type=float, default=0.12)
    parser.add_argument("--router-specialization-class-threshold", type=float, default=0.28)
    parser.add_argument("--class-skew-residual", action="store_true")
    parser.add_argument("--class-skew-orthogonal-weight", type=float, default=0.0)
    parser.add_argument("--class-skew-srip-weight", type=float, default=0.0)
    parser.add_argument("--class-skew-residual-weight", type=float, default=0.0)
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
    parser.add_argument("--dqa-server-anchor", type=float, default=0.65)
    parser.add_argument("--dqa-min-server-alpha", type=float, default=0.60)
    parser.add_argument("--dqa-residual-blend", type=float, default=0.10)
    parser.add_argument(
        "--dqa-moe-expert-blend",
        type=float,
        default=0.0,
        help="Blend DQA-targeted latent-MoE expert residuals into each expert during aggregation.",
    )
    parser.add_argument(
        "--dqa-moe-router-blend",
        type=float,
        default=0.0,
        help="Blend DQA-weighted latent-MoE router residuals during aggregation.",
    )
    parser.add_argument(
        "--dqa-bn-blend",
        type=float,
        default=0.0,
        help="Blend DQA-weighted batch-norm statistics/affine residuals during aggregation.",
    )
    parser.add_argument("--dqa-client-balance-stats", action="store_true", default=True)
    parser.add_argument("--no-dqa-client-balance-stats", dest="dqa_client_balance_stats", action="store_false")
    parser.add_argument("--dqa-client-balance-target", choices=["median", "mean", "max"], default="median")
    parser.add_argument("--dqa-client-balance-max-scale", type=float, default=3.0)
    parser.add_argument("--late-dqa-server-anchor", type=float, default=0.35)
    parser.add_argument("--late-dqa-min-server-alpha", type=float, default=0.35)
    parser.add_argument("--late-dqa-residual-blend", type=float, default=0.08)
    parser.add_argument("--expert-count", type=int, default=4)
    parser.add_argument("--expert-keep-fraction", type=float, default=0.45)
    parser.add_argument("--expert-max-class-fraction", type=float, default=0.18)
    parser.add_argument("--actual-max-class-fraction", type=float, default=0.25)
    parser.add_argument("--late-expert-keep-fraction", type=float, default=0.60)
    parser.add_argument("--late-expert-max-class-fraction", type=float, default=0.22)
    parser.add_argument("--late-actual-max-class-fraction", type=float, default=0.28)
    parser.add_argument("--load-bias-strength", type=float, default=0.45)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--allow-cpu-training", action="store_true")
    parser.add_argument("--master-port", type=int, default=36601)
    parser.add_argument("--device", default="")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--pseudo-imgsz",
        type=int,
        default=0,
        help="Use this image size only for self pseudoGT generation; 0 falls back to --imgsz.",
    )
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--nms-iou-thres", type=float, default=0.65)
    parser.add_argument("--match-iou", type=float, default=0.60)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--min-models", type=int, default=0)
    parser.add_argument("--min-stability", type=float, default=0.78)
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--late-min-stability", type=float, default=0.68)
    parser.add_argument("--late-min-score", type=float, default=0.24)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-boxes-per-image", type=int, default=8)
    parser.add_argument("--max-images-per-client", type=int, default=0)
    parser.add_argument("--max-class-fraction", type=float, default=0.45)
    parser.add_argument("--min-class-keep", type=int, default=250)
    parser.add_argument("--client-mosaic", type=float, default=1.0)
    parser.add_argument("--client-mixup", type=float, default=0.0)
    parser.add_argument("--client-scale", type=float, default=0.25)
    parser.add_argument("--client-hsv-s", type=float, default=0.35)
    parser.add_argument("--client-hsv-v", type=float, default=0.20)
    parser.add_argument(
        "--style-source-repeat",
        type=int,
        default=0,
        help="Repeat a per-client source-GT list stylized with target-client Fourier amplitude. 0 disables it.",
    )
    parser.add_argument("--style-source-limit", type=int, default=0, help="Maximum source images to stylize per selected client and round. 0 means all source images.")
    parser.add_argument("--style-beta", type=float, default=0.012, help="Low-frequency Fourier amplitude replacement ratio for target style transfer.")
    parser.add_argument("--style-imgsz", type=int, default=640, help="Square image size used when materializing target-styled source images.")
    parser.add_argument("--style-seed", type=int, default=20260512)
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
    parser.add_argument("--target-map50", type=float, default=0.60)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    do_start_notify = args.notify or args.notify_start
    do_end_notify = args.notify or args.notify_end
    if do_start_notify:
        notify(args, "Scene-Daynight DQA 18 client-balanced single-injection full-from-warmup DQA-MoX started.", title="DQA 18 start")

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
                f"Scene-Daynight DQA 18 client-balanced single-injection full-from-warmup DQA-MoX finished with status={status}.",
                title="DQA 18 finish",
                status=status,
                error=error,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
