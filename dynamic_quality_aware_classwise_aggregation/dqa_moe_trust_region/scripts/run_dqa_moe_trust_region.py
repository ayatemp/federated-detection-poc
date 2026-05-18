#!/usr/bin/env python3
"""FedSTO-compatible DQA x MOE trust-region training loop.

This runner keeps the FedSTO paper reproduction path as the main training
signal.  DQA x MOE is applied only as a small phase-2 residual controller:

    FedSTO server repair S_t
      + small weighted residual from client/domain experts
      + source-validation trust-region acceptance
      -> next broadcast global

The intent is to preserve the monotonic behavior seen in the FedSTO
reproduction while giving DQA a real training-time role.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
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


METRIC_PROTOCOL_VERSION = "fedsto_public_spec_broadcast_reset_ema_v2+dqa_moe_trust_region_v1"
SHARED_ROUTED_PROTOCOL_VERSION = (
    "fedsto_public_spec_broadcast_reset_ema_v2+dqa_moe_trust_region_v2_shared_routed_head"
)
DEFAULT_WORKSPACE = (
    REPO_ROOT
    / "dynamic_quality_aware_classwise_aggregation"
    / "dqa_moe_trust_region"
    / "output"
    / "01_fedsto_dqa_moe_trust_region_12h"
)
PREFERRED_VAL_PYTHONS = [
    Path("/root/micromamba/envs/al_yolov8/bin/python"),
    Path(sys.executable),
    Path("/opt/venv/bin/python"),
]

DQA_HISTORY_PATH = DEFAULT_WORKSPACE / "dqa_moe_history.json"
DQA_STATS_PATH = DEFAULT_WORKSPACE / "dqa_moe_round_summary.csv"
DQA_EVAL_ROOT = DEFAULT_WORKSPACE / "dqa_moe_candidate_eval"
DQA_PSEUDO_STATS_ROOT = DEFAULT_WORKSPACE / "pseudo_stats"


def apply_workspace_root(workspace_root: Path) -> Path:
    global DQA_HISTORY_PATH, DQA_STATS_PATH, DQA_EVAL_ROOT, DQA_PSEUDO_STATS_ROOT
    workspace_root = fedsto.apply_workspace_root(workspace_root)
    DQA_HISTORY_PATH = workspace_root / "dqa_moe_history.json"
    DQA_STATS_PATH = workspace_root / "dqa_moe_round_summary.csv"
    DQA_EVAL_ROOT = workspace_root / "dqa_moe_candidate_eval"
    DQA_PSEUDO_STATS_ROOT = workspace_root / "pseudo_stats"
    return workspace_root


def parse_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def finite_float(value: Any, default: float = 0.0) -> float:
    value = parse_float(value, default)
    return value if math.isfinite(value) else default


def positive_float(value: Any, default: float = 0.0) -> float:
    return max(finite_float(value, default), 0.0)


def selected_protocol_version(args: argparse.Namespace) -> str:
    override = getattr(args, "protocol_version", None)
    if override:
        return str(override)
    if getattr(args, "dqa_router_mode", "metric") == "shared_routed":
        return SHARED_ROUTED_PROTOCOL_VERSION
    return METRIC_PROTOCOL_VERSION


def read_last_results_row(run_name: str) -> dict[str, str]:
    path = setup.RUN_ROOT / run_name / "results.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))
    return rows[-1] if rows else {}


def metric(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    return parse_float(row.get(key), default)


def metric_summary(row: dict[str, Any]) -> dict[str, float]:
    return {
        "precision": metric(row, "metrics/precision"),
        "recall": metric(row, "metrics/recall"),
        "map50": metric(row, "metrics/mAP_0.5"),
        "map50_95": metric(row, "metrics/mAP_0.5:0.95"),
    }


def client_pseudo_stats_path(phase: int, round_idx: int, client: dict[str, Any]) -> Path:
    return (
        DQA_PSEUDO_STATS_ROOT
        / f"phase{phase}_round{round_idx:03d}_client{client['id']}_{client['weather']}.json"
    )


def load_client_pseudo_stats(phase: int, round_idx: int, client: dict[str, Any]) -> dict[str, Any]:
    path = client_pseudo_stats_path(phase, round_idx, client)
    if not path.exists():
        return {"path": str(path), "status": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"path": str(path), "status": "invalid", "error": str(exc)}
    payload["path"] = str(path)
    payload["status"] = "ok"
    return payload


def list_floats(raw: Any, length: int, default: float = 0.0) -> list[float]:
    if not isinstance(raw, list):
        return [default] * length
    values = [finite_float(value, default) for value in raw[:length]]
    if len(values) < length:
        values.extend([default] * (length - len(values)))
    return values


def entropy_score(values: list[float]) -> float:
    total = sum(max(value, 0.0) for value in values)
    if total <= 0:
        return 0.0
    entropy = 0.0
    active = 0
    for value in values:
        value = max(value, 0.0)
        if value <= 0:
            continue
        active += 1
        p = value / total
        entropy -= p * math.log(p)
    if active <= 1:
        return 0.0
    return entropy / math.log(active)


def normalize_router_scores(scores: list[float], *, floor: float, temperature: float) -> list[float]:
    if not scores:
        return []
    scores = [max(finite_float(score, 1e-9), 1e-9) for score in scores]
    temperature = max(finite_float(temperature, 1.0), 1e-6)
    logits = [math.log(score) / temperature for score in scores]
    max_logit = max(logits)
    exp_scores = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exp_scores) or 1.0
    weights = [score / total for score in exp_scores]
    floor = min(max(finite_float(floor, 0.0), 0.0), 1.0 / max(len(weights), 1))
    if floor > 0:
        remaining = max(1.0 - floor * len(weights), 0.0)
        weights = [floor + remaining * weight for weight in weights]
        norm = sum(weights) or 1.0
        weights = [weight / norm for weight in weights]
    return weights


def pseudo_stats_summary(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    nc = len(setup.BDD_NAMES)
    counts = list_floats(payload.get("counts"), nc)
    qualities = list_floats(payload.get("mean_quality_scores"), nc)
    total = sum(max(value, 0.0) for value in counts)
    weighted_quality = 0.0
    for count, quality in zip(counts, qualities, strict=False):
        weighted_quality += max(count, 0.0) * min(max(quality, 0.0), 1.0)
    mean_quality = weighted_quality / total if total > 0 else 0.0
    active_classes = sum(1 for value in counts if value > 0)
    rare_ids = [idx for idx in parse_csv_values(args.dqa_rare_class_ids, int) if 0 <= idx < nc]
    rare_count = sum(max(counts[idx], 0.0) for idx in rare_ids)
    return {
        "counts": counts,
        "mean_quality_scores": qualities,
        "pseudo_total": total,
        "mean_quality": mean_quality,
        "active_classes": active_classes,
        "class_balance": entropy_score(counts),
        "rare_share": rare_count / total if total > 0 else 0.0,
    }


def base_client_metric_score(metrics: dict[str, float]) -> float:
    return (
        positive_float(metrics["map50_95"])
        + 0.25 * positive_float(metrics["map50"])
        + 0.10 * positive_float(metrics["recall"])
        + 0.05 * positive_float(metrics["precision"])
    )


def focus_multiplier(weather: str, focus: str | None, args: argparse.Namespace) -> float:
    if not focus or focus == "balanced":
        return 1.0
    boost = max(finite_float(args.dqa_focus_boost, 1.0), 0.0)
    if focus == weather:
        return boost
    if focus == "bad_weather" and weather in {"rainy", "snowy"}:
        return 1.0 + 0.5 * (boost - 1.0)
    if focus == "quality":
        return 1.0
    return 1.0


def client_expert_weights(
    *,
    phase: int,
    round_idx: int,
    client_paths: list[Path],
    args: argparse.Namespace | None,
    focus: str | None,
) -> tuple[list[float], list[dict[str, Any]], list[list[float]], dict[str, Any]]:
    """Compute a DQA router over client/domain experts.

    The score intentionally uses only self-generated training artifacts:
    each client's source validation metrics plus optional pseudo-label stats
    produced by its own local EMA teacher.  In ``shared_routed`` mode this also
    returns class-specific expert-choice weights for YOLO head output channels.
    """

    priors = {"overcast": 1.00, "rainy": 1.03, "snowy": 1.06}
    nc = len(setup.BDD_NAMES)
    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for client, path in zip(setup.CLIENTS, client_paths, strict=False):
        run_name = f"phase{phase}_round{round_idx:03d}_client{client['id']}_{client['weather']}"
        result = read_last_results_row(run_name)
        m = metric_summary(result)
        base = base_client_metric_score(m)
        stats_payload = load_client_pseudo_stats(phase, round_idx, client) if args else {"status": "disabled"}
        stats = pseudo_stats_summary(stats_payload, args) if args else {}
        score = max(base * priors.get(client["weather"], 1.0), 1e-6)
        if args and args.dqa_router_mode == "shared_routed" and stats_payload.get("status") == "ok":
            quality_gain = 1.0 + args.dqa_pseudo_quality_weight * (stats["mean_quality"] - 0.5)
            balance_gain = 1.0 + args.dqa_pseudo_balance_weight * stats["class_balance"]
            rare_gain = 1.0 + args.dqa_pseudo_rare_weight * stats["rare_share"]
            score *= max(quality_gain, 0.05) * max(balance_gain, 0.05) * max(rare_gain, 0.05)
        if args:
            score *= focus_multiplier(client["weather"], focus, args)
        scores.append(score)
        rows.append(
            {
                "client_id": client["id"],
                "weather": client["weather"],
                "path": str(path),
                "score": score,
                "pseudo_stats_path": stats_payload.get("path", ""),
                "pseudo_stats_status": stats_payload.get("status", "disabled"),
                "pseudo_total": stats.get("pseudo_total"),
                "pseudo_mean_quality": stats.get("mean_quality"),
                "pseudo_active_classes": stats.get("active_classes"),
                "pseudo_class_balance": stats.get("class_balance"),
                "pseudo_rare_share": stats.get("rare_share"),
                **m,
            }
        )

    floor = args.dqa_expert_min_weight if args else 0.0
    temperature = args.dqa_router_temperature if args else 1.0
    weights = normalize_router_scores(scores, floor=floor, temperature=temperature)
    for row, weight in zip(rows, weights, strict=False):
        row["weight"] = weight

    class_weights: list[list[float]] = [list(weights) for _ in range(nc)]
    if args and args.dqa_router_mode == "shared_routed" and args.dqa_classwise_head_routing:
        stats_by_client = [pseudo_stats_summary(load_client_pseudo_stats(phase, round_idx, client), args) for client in setup.CLIENTS]
        for class_idx in range(nc):
            class_scores: list[float] = []
            any_count = False
            max_count = max((stats["counts"][class_idx] for stats in stats_by_client), default=0.0)
            for row, stats in zip(rows, stats_by_client, strict=False):
                count = max(stats["counts"][class_idx], 0.0)
                quality = min(max(stats["mean_quality_scores"][class_idx], 0.0), 1.0)
                any_count = any_count or count > 0
                count_gain = 1.0
                if max_count > 0:
                    count_gain += args.dqa_pseudo_class_count_weight * (
                        math.log1p(count) / max(math.log1p(max_count), 1e-9)
                    )
                quality_gain = 1.0 + args.dqa_pseudo_quality_weight * (quality - 0.5)
                class_scores.append(max(row["score"] * count_gain * max(quality_gain, 0.05), 1e-9))
            if any_count:
                class_weights[class_idx] = normalize_router_scores(
                    class_scores,
                    floor=floor,
                    temperature=temperature,
                )

    routing_proxy = sum(
        float(row.get("weight", 0.0))
        * (
            finite_float(row.get("pseudo_mean_quality"), 0.0)
            + 0.10 * finite_float(row.get("pseudo_class_balance"), 0.0)
            + 0.10 * finite_float(row.get("pseudo_rare_share"), 0.0)
        )
        for row in rows
    )
    return weights, rows, class_weights, {"focus": focus or "balanced", "routing_proxy": routing_proxy}


def dqa_lambda_for_round(args: argparse.Namespace, round_idx: int) -> float:
    if args.phase2_rounds <= args.dqa_start_round:
        return args.dqa_lambda_end
    progress = (round_idx - args.dqa_start_round) / max(args.phase2_rounds - args.dqa_start_round, 1)
    progress = min(max(progress, 0.0), 1.0)
    return args.dqa_lambda_start + progress * (args.dqa_lambda_end - args.dqa_lambda_start)


def parse_csv_values(raw: str, cast):
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(cast(item))
    return values


def safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "candidate"


def candidate_specs(args: argparse.Namespace, round_idx: int) -> list[dict[str, Any]]:
    base_lambda = dqa_lambda_for_round(args, round_idx)
    router_candidates = ["balanced"]
    if args.dqa_router_mode == "shared_routed":
        router_candidates = parse_csv_values(args.dqa_router_candidates, str) or ["balanced"]
    if not args.dqa_search_candidates:
        return [
            {
                "name": f"scope_{args.dqa_scope}_route_{router_candidates[0]}_lambda_1.00",
                "scope": args.dqa_scope,
                "routing_focus": router_candidates[0],
                "lambda": base_lambda,
                "lambda_multiplier": 1.0,
                "max_relative_update": args.dqa_max_relative_update,
            }
        ]

    scopes = parse_csv_values(args.dqa_candidate_scopes, str)
    multipliers = parse_csv_values(args.dqa_candidate_lambda_multipliers, float)
    allowed_scopes = {"head", "head_bn", "head_neck_bn", "neck_head"}
    unknown = [scope for scope in scopes if scope not in allowed_scopes]
    if unknown:
        raise ValueError(f"Unknown DQA candidate scope(s): {unknown}. Allowed: {sorted(allowed_scopes)}")
    max_candidates = max(args.dqa_max_candidates, 1)
    specs: list[dict[str, Any]] = []
    for scope in scopes:
        for multiplier in multipliers:
            for routing_focus in router_candidates:
                specs.append(
                    {
                        "name": f"scope_{scope}_route_{safe_label(routing_focus)}_lambda_{multiplier:.2f}",
                        "scope": scope,
                        "routing_focus": routing_focus,
                        "lambda": base_lambda * multiplier,
                        "lambda_multiplier": multiplier,
                        "max_relative_update": args.dqa_max_relative_update,
                    }
                )
    # Keep candidate search cheap and deterministic.  The default order starts
    # with the safest variants, then explores wider scopes/stronger residuals.
    return specs[:max_candidates]


def should_mix_key(key: str, args: argparse.Namespace) -> bool:
    if "anchors" in key:
        return False
    if args.dqa_scope == "head":
        return key.startswith("head.")
    if args.dqa_scope == "head_bn":
        return key.startswith("head.") or (key.startswith("neck.") and ".bn." in key)
    if args.dqa_scope == "head_neck_bn":
        if key.startswith("head."):
            return True
        if not key.startswith("neck."):
            return False
        return ".bn." in key or key.endswith("running_mean") or key.endswith("running_var")
    if args.dqa_scope == "neck_head":
        return key.startswith("head.") or key.startswith("neck.")
    raise ValueError(f"Unknown DQA scope: {args.dqa_scope}")


def head_output_class_for_rows(key: str, value: torch.Tensor, nc: int) -> list[int | None] | None:
    if not (key.startswith("head.m.") and (key.endswith(".weight") or key.endswith(".bias"))):
        return None
    if value.ndim not in {1, 4}:
        return None
    output_channels = int(value.shape[0])
    stride = nc + 5
    if output_channels <= 0 or output_channels % stride != 0:
        return None
    classes: list[int | None] = []
    for row_idx in range(output_channels):
        offset = row_idx % stride
        classes.append(offset - 5 if offset >= 5 else None)
    return classes


def classwise_head_expert_value(
    *,
    key: str,
    value: torch.Tensor,
    client_states: list[dict[str, torch.Tensor]],
    global_weights: list[float],
    class_weights: list[list[float]],
    nc: int,
) -> torch.Tensor:
    row_classes = head_output_class_for_rows(key, value, nc)
    if row_classes is None:
        expert_value = torch.zeros_like(value.float())
        for weight, state in zip(global_weights, client_states, strict=False):
            expert_value += float(weight) * state[key].float()
        return expert_value

    expert_value = torch.zeros_like(value.float())
    for row_idx, class_idx in enumerate(row_classes):
        weights = global_weights if class_idx is None else class_weights[class_idx]
        for weight, state in zip(weights, client_states, strict=False):
            expert_value[row_idx] += float(weight) * state[key].float()[row_idx]
    return expert_value


def tensor_norm(value: torch.Tensor) -> float:
    if not value.dtype.is_floating_point:
        return 0.0
    return float(value.float().norm().item())


def trust_region_delta(
    *,
    base_value: torch.Tensor,
    delta: torch.Tensor,
    max_relative_update: float,
    max_absolute_update: float,
) -> tuple[torch.Tensor, float, float]:
    delta = delta.float()
    delta_norm = float(delta.norm().item())
    base_norm = float(base_value.float().norm().item())
    max_norm = max_relative_update * max(base_norm, 1e-12) + max_absolute_update
    if delta_norm > max_norm > 0:
        delta = delta * (max_norm / max(delta_norm, 1e-12))
        return delta, delta_norm, max_norm
    return delta, delta_norm, max_norm


def create_dqa_moe_candidate(
    *,
    server_ckpt: Path,
    client_paths: list[Path],
    out: Path,
    phase: int,
    round_idx: int,
    args: argparse.Namespace,
    scope: str | None = None,
    residual_lambda: float | None = None,
    max_relative_update: float | None = None,
    routing_focus: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    protocol = selected_protocol_version(args)
    weights, expert_rows, class_weights, routing_meta = client_expert_weights(
        phase=phase,
        round_idx=round_idx,
        client_paths=client_paths,
        args=args,
        focus=routing_focus,
    )
    lam = dqa_lambda_for_round(args, round_idx) if residual_lambda is None else residual_lambda
    scope = args.dqa_scope if scope is None else scope
    max_relative_update = args.dqa_max_relative_update if max_relative_update is None else max_relative_update

    base = copy.deepcopy(fedsto._load(server_ckpt))
    model = base["model"].float()
    server_state = model.state_dict()
    client_states = [fedsto._state_dict(fedsto._load(path), "model") for path in client_paths]

    mixed_state: dict[str, torch.Tensor] = {}
    updated_keys = 0
    clipped_keys = 0
    delta_norm_sum = 0.0
    max_norm_sum = 0.0
    group_counts: dict[str, int] = {"head": 0, "neck": 0, "other": 0}
    key_args = copy.copy(args)
    key_args.dqa_scope = scope
    nc = len(setup.BDD_NAMES)

    for key, value in server_state.items():
        if value.dtype.is_floating_point and should_mix_key(key, key_args):
            if args.dqa_router_mode == "shared_routed" and args.dqa_classwise_head_routing:
                expert_value = classwise_head_expert_value(
                    key=key,
                    value=value,
                    client_states=client_states,
                    global_weights=weights,
                    class_weights=class_weights,
                    nc=nc,
                )
            else:
                expert_value = torch.zeros_like(value.float())
                for weight, state in zip(weights, client_states, strict=False):
                    expert_value += float(weight) * state[key].float()
            delta, raw_norm, max_norm = trust_region_delta(
                base_value=value,
                delta=expert_value - value.float(),
                max_relative_update=max_relative_update,
                max_absolute_update=args.dqa_max_absolute_update,
            )
            if raw_norm > max_norm > 0:
                clipped_keys += 1
            mixed_state[key] = (value.float() + lam * delta).to(value.dtype)
            updated_keys += 1
            delta_norm_sum += raw_norm
            max_norm_sum += max_norm
            if key.startswith("head."):
                group_counts["head"] += 1
            elif key.startswith("neck."):
                group_counts["neck"] += 1
            else:
                group_counts["other"] += 1
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
    base["fedsto_protocol"] = protocol
    base["fedsto_stage"] = f"phase{phase}_round{round_idx:03d}_dqa_moe_candidate"
    base["dqa_moe"] = {
        "protocol": protocol,
        "lambda": lam,
        "scope": scope,
        "router_mode": args.dqa_router_mode,
        "routing_focus": routing_meta["focus"],
        "routing_proxy": routing_meta["routing_proxy"],
        "classwise_head_routing": bool(args.dqa_classwise_head_routing),
        "max_relative_update": max_relative_update,
        "weights": expert_rows,
        "class_weights": class_weights if args.dqa_router_mode == "shared_routed" else [],
        "updated_keys": updated_keys,
        "clipped_keys": clipped_keys,
        "group_counts": group_counts,
        "delta_norm_sum": delta_norm_sum,
        "max_norm_sum": max_norm_sum,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, out)
    return out, base["dqa_moe"]


def select_val_python(explicit: Path | None = None) -> Path:
    candidates = [explicit] if explicit is not None else []
    for candidate in PREFERRED_VAL_PYTHONS:
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if candidate is None or not candidate.exists():
            continue
        result = subprocess.run(
            [str(candidate), "-c", "import cv2, seaborn, torch, yaml"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return candidate
    return Path(sys.executable)


def write_candidate_eval_config(label: str, batch_size: int) -> Path:
    cfg = setup.efficientteacher_config(
        name=f"dqa_candidate_eval_{label}",
        train=setup.LIST_ROOT / "server_cloudy_train.txt",
        val=setup.LIST_ROOT / "server_cloudy_val.txt",
        target=None,
        weights="",
        epochs=1,
        train_scope="all",
        batch_size=batch_size,
        workers=0,
        device="",
    )
    cfg["Dataset"]["batch_size"] = batch_size
    cfg["Dataset"]["workers"] = 0
    cfg["SSOD"] = {"train_domain": False}
    path = DQA_EVAL_ROOT / "configs" / f"{label}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def parse_val_stdout(stdout: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) >= 7 and parts[0] == "all":
            parsed = {
                "images": parse_float(parts[1]),
                "labels": parse_float(parts[2]),
                "precision": parse_float(parts[3]),
                "recall": parse_float(parts[4]),
                "map50": parse_float(parts[5]),
                "map50_95": parse_float(parts[6]),
            }
    return parsed


def evaluate_cloudy_candidate(checkpoint: Path, label: str, args: argparse.Namespace) -> dict[str, Any]:
    val_python = select_val_python(args.val_python)
    cfg = write_candidate_eval_config(label, args.val_batch_size)
    log_dir = DQA_EVAL_ROOT / "logs"
    run_root = DQA_EVAL_ROOT / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(val_python),
        "val.py",
        "--weights",
        str(checkpoint),
        "--cfg",
        str(cfg),
        "--batch-size",
        str(args.val_batch_size),
        "--imgsz",
        str(args.val_imgsz),
        "--conf-thres",
        str(args.val_conf_thres),
        "--iou-thres",
        str(args.val_iou_thres),
        "--project",
        str(run_root),
        "--name",
        label,
        "--exist-ok",
        "--no-plots",
    ]
    if args.val_device:
        cmd.extend(["--device", args.val_device])

    result = subprocess.run(cmd, cwd=setup.ET_ROOT, capture_output=True, text=True)
    log_path = log_dir / f"{label}.log"
    log_path.write_text(result.stdout + "\nSTDERR\n" + result.stderr, encoding="utf-8")
    row: dict[str, Any] = {
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "log_file": str(log_path),
        "command": " ".join(cmd),
    }
    if result.returncode == 0:
        row.update(parse_val_stdout(result.stdout))
    else:
        row["error"] = result.stderr[-1000:]
    return row


def run_train_with_env(config: Path, args: argparse.Namespace, extra_env: dict[str, str] | None = None) -> Path:
    if args.gpus > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(args.gpus),
            "--master_port",
            str(args.master_port),
            "train.py",
            "--cfg",
            str(config.resolve()),
        ]
    else:
        cmd = [sys.executable, "train.py", "--cfg", str(config.resolve())]

    print(" ".join(cmd))
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run(cmd, cwd=setup.ET_ROOT, env=env, check=True)

    cfg_name = config.stem
    if cfg_name.startswith("runtime_phase"):
        with config.open(encoding="utf-8") as f:
            run_name = yaml.safe_load(f)["name"]
    else:
        run_name = cfg_name
    return fedsto.checkpoint_path(run_name)


def dqa_candidate_is_accepted(
    *,
    server_metrics: dict[str, float],
    candidate_metrics: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[bool, str]:
    if candidate_metrics.get("status") != "ok":
        return False, "candidate_eval_failed"

    candidate_map95 = parse_float(candidate_metrics.get("map50_95"))
    candidate_map50 = parse_float(candidate_metrics.get("map50"))
    server_map95 = parse_float(server_metrics.get("map50_95"))
    server_map50 = parse_float(server_metrics.get("map50"))
    if not (math.isfinite(candidate_map95) and math.isfinite(candidate_map50)):
        return False, "candidate_metrics_nan"
    if not (math.isfinite(server_map95) and math.isfinite(server_map50)):
        return False, "server_metrics_nan"
    if candidate_map95 < server_map95 - args.dqa_acceptance_tolerance_map50_95:
        return False, "source_map50_95_regression"
    if candidate_map50 < server_map50 - args.dqa_acceptance_tolerance_map50:
        return False, "source_map50_regression"
    return True, "accepted"


def dqa_proxy_score(metrics: dict[str, Any], args: argparse.Namespace) -> float:
    return (
        parse_float(metrics.get("map50_95"), -1.0)
        + args.dqa_score_map50_weight * parse_float(metrics.get("map50"), 0.0)
        + args.dqa_score_recall_weight * parse_float(metrics.get("recall"), 0.0)
    )


def select_best_candidate(
    *,
    server_metrics: dict[str, float],
    candidate_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, str]:
    acceptable = []
    rejection_reasons = []
    for row in candidate_rows:
        accepted, reason = dqa_candidate_is_accepted(
            server_metrics=server_metrics,
            candidate_metrics=row.get("candidate_metrics", {}),
            args=args,
        )
        row["accepted_by_trust_region"] = accepted
        row["trust_region_reason"] = reason
        row["proxy_score"] = dqa_proxy_score(row.get("candidate_metrics", {}), args) + (
            args.dqa_router_proxy_weight
            * parse_float(row.get("dqa_meta", {}).get("routing_proxy"), 0.0)
        )
        if accepted:
            acceptable.append(row)
        else:
            rejection_reasons.append(f"{row.get('name')}={reason}")

    if not acceptable:
        return None, "all_candidates_rejected:" + ",".join(rejection_reasons[:6])
    acceptable.sort(key=lambda row: row["proxy_score"], reverse=True)
    best = acceptable[0]
    server_score = dqa_proxy_score(server_metrics, args)
    if best["proxy_score"] < server_score - args.dqa_acceptance_score_tolerance:
        return None, "best_candidate_score_below_server"
    return best, "accepted_best_candidate"


def load_dqa_history() -> list[dict[str, Any]]:
    if not DQA_HISTORY_PATH.exists():
        return []
    data = json.loads(DQA_HISTORY_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Expected {DQA_HISTORY_PATH} to contain a list.")
    return data


def write_dqa_history(rows: list[dict[str, Any]]) -> None:
    DQA_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DQA_HISTORY_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def append_round_summary(row: dict[str, Any]) -> None:
    DQA_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    flat = {
        "phase": row.get("phase"),
        "round": row.get("round"),
        "global": row.get("global"),
        "dqa_applied": row.get("dqa_applied"),
        "dqa_accepted": row.get("dqa_accepted"),
        "dqa_reason": row.get("dqa_reason"),
        "dqa_lambda": row.get("dqa_lambda"),
        "server_map50": row.get("server_metrics", {}).get("map50"),
        "server_map50_95": row.get("server_metrics", {}).get("map50_95"),
        "candidate_map50": row.get("candidate_metrics", {}).get("map50"),
        "candidate_map50_95": row.get("candidate_metrics", {}).get("map50_95"),
        "candidate_count": len(row.get("evaluated_candidates", [])),
        "selected_scope": row.get("dqa_scope"),
        "router_mode": row.get("dqa_meta", {}).get("router_mode"),
        "routing_focus": row.get("dqa_meta", {}).get("routing_focus"),
        "routing_proxy": row.get("dqa_meta", {}).get("routing_proxy"),
        "classwise_head_routing": row.get("dqa_meta", {}).get("classwise_head_routing"),
        "updated_keys": row.get("dqa_meta", {}).get("updated_keys"),
        "clipped_keys": row.get("dqa_meta", {}).get("clipped_keys"),
    }
    for expert in row.get("dqa_meta", {}).get("weights", []):
        flat[f"expert_{expert['weather']}_weight"] = expert.get("weight")
        flat[f"expert_{expert['weather']}_map50"] = expert.get("map50")
        flat[f"expert_{expert['weather']}_map50_95"] = expert.get("map50_95")
        flat[f"expert_{expert['weather']}_pseudo_total"] = expert.get("pseudo_total")
        flat[f"expert_{expert['weather']}_pseudo_mean_quality"] = expert.get("pseudo_mean_quality")
        flat[f"expert_{expert['weather']}_pseudo_class_balance"] = expert.get("pseudo_class_balance")

    exists = DQA_STATS_PATH.exists()
    fieldnames = list(flat.keys())
    if exists:
        with DQA_STATS_PATH.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []
        fieldnames = list(dict.fromkeys([*existing_fields, *fieldnames]))
    rows = []
    if exists:
        with DQA_STATS_PATH.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    rows.append({key: flat.get(key, "") for key in fieldnames})
    with DQA_STATS_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def notify(args: argparse.Namespace, title: str, message: str, context: dict[str, Any] | None = None) -> None:
    if not args.discord:
        return
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from notebook_notify import notify_discord

        notify_discord(
            message,
            title=title,
            context=context,
            fail_silently=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}")


def run_final_eval(args: argparse.Namespace) -> None:
    if not args.run_final_eval:
        return
    eval_script = FEDSTO_SCRIPT_ROOT / "evaluate_paper_protocol.py"
    cmd = [
        sys.executable,
        str(eval_script),
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


def run_protocol(args: argparse.Namespace) -> None:
    protocol = selected_protocol_version(args)
    apply_workspace_root(args.workspace_root)
    setup.build_base_configs()
    if args.setup_only:
        print(f"Setup complete: {args.workspace_root}")
        return
    if args.dry_run:
        print(f"Dry run setup complete: {args.workspace_root}")
        print(
            "Training plan: "
            f"warmup={args.warmup_epochs}, phase1={args.phase1_rounds}, "
            f"phase2={args.phase2_rounds}, dqa_start_round={args.dqa_start_round}, "
            f"dqa_scope={args.dqa_scope}, dqa_search_candidates={args.dqa_search_candidates}, "
            f"max_candidates={args.dqa_max_candidates}, router={args.dqa_router_mode}, "
            f"protocol={protocol}"
        )
        return

    pretrained = fedsto.PRETRAINED_PATH if args.dry_run else fedsto.download_pretrained()
    fedsto.check_runtime_dependencies()
    visible_gpus = torch.cuda.device_count()
    if not args.allow_cpu:
        if visible_gpus <= 0 or not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA GPU is visible. Refusing to run the 12h DQA x MOE experiment on CPU. "
                "Fix the GPU/NVML environment and rerun, or pass --allow-cpu for a tiny debugging run only."
            )
        if args.gpus > visible_gpus:
            raise RuntimeError(
                f"Requested --gpus {args.gpus}, but only {visible_gpus} CUDA device(s) are visible. "
                "Use a lower --gpus value or fix device visibility."
            )
    args.gpus = fedsto.resolve_gpus(args.gpus)
    fedsto.GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    fedsto.CLIENT_STATE_DIR.mkdir(parents=True, exist_ok=True)
    config_device = "" if args.gpus > 1 else args.device

    current_global = fedsto.GLOBAL_DIR / "round000_warmup.pt"
    if current_global.exists() and not args.force_warmup:
        ok, reason = fedsto.validate_checkpoint(current_global)
        if ok:
            print(f"Reusing completed warm-up checkpoint: {current_global}")
        else:
            print(f"Warm-up checkpoint exists but is invalid ({reason}); rerunning warm-up.")
            current_global.unlink()

    if not current_global.exists():
        warmup_cfg = setup.write_config(
            "runtime_server_warmup.yaml",
            setup.efficientteacher_config(
                name="runtime_server_warmup",
                train=setup.LIST_ROOT / "server_cloudy_train.txt",
                val=setup.LIST_ROOT / "server_cloudy_val.txt",
                target=None,
                weights=str(pretrained.resolve()),
                epochs=args.warmup_epochs,
                train_scope="all",
                batch_size=args.batch_size,
                workers=args.workers,
                device=config_device,
            ),
        )
        warmup_ckpt = fedsto.run_train(warmup_cfg, False, gpus=args.gpus, master_port=args.master_port)
        fedsto.make_start_checkpoint(
            warmup_ckpt,
            current_global,
            protocol=protocol,
            stage="round000_warmup",
        )

    if args.force_restart:
        history: list[dict[str, Any]] = []
        dqa_history: list[dict[str, Any]] = []
        print("Ignoring existing history because --force-restart was set.")
    else:
        loaded_history = fedsto.load_history()
        history, current_global, next_round = fedsto.completed_history_prefix(
            loaded_history,
            phase1_rounds=args.phase1_rounds,
            phase2_rounds=args.phase2_rounds,
            warmup_checkpoint=current_global,
            expected_protocol=protocol,
        )
        if loaded_history != history:
            fedsto.write_history(history)
        dqa_history = load_dqa_history()
        if next_round is None:
            total_rounds = args.phase1_rounds + args.phase2_rounds
            print(f"All requested federated rounds are already complete ({total_rounds}/{total_rounds}).")
            print(f"Latest global checkpoint: {current_global}")
            run_final_eval(args)
            return
        if history:
            phase, round_idx = next_round
            print(
                f"Resuming after {len(history)} completed federated rounds "
                f"from phase {phase} round {round_idx}."
            )
            print(f"Current global checkpoint: {current_global}")
        else:
            print("No completed federated rounds found; starting from phase 1 round 1.")

    if not args.keep_intermediate_checkpoints:
        fedsto.cleanup_completed_intermediates(history)

    notify(
        args,
        "DQA x MOE Trust Region started",
        (
            f"workspace: `{args.workspace_root}`\n"
            f"schedule: warmup {args.warmup_epochs}, phase1 {args.phase1_rounds}, phase2 {args.phase2_rounds}\n"
            f"dqa: phase2 round >= {args.dqa_start_round}, scope={args.dqa_scope}, router={args.dqa_router_mode}\n"
            f"protocol: `{protocol}`"
        ),
    )

    completed = {(int(entry["phase"]), int(entry["round"])) for entry in history}
    for phase, rounds in [(1, args.phase1_rounds), (2, args.phase2_rounds)]:
        for round_idx in range(1, rounds + 1):
            if (phase, round_idx) in completed:
                continue

            next_global = fedsto.GLOBAL_DIR / f"phase{phase}_round{round_idx:03d}_global.pt"
            reused_global = fedsto.reuse_checkpoint_if_valid(
                next_global,
                f"global checkpoint for phase {phase} round {round_idx}",
                force_retrain=args.force_retrain,
                expected_protocol=protocol,
            )
            if reused_global is not None:
                current_global = reused_global
                history.append(
                    {
                        "phase": phase,
                        "round": round_idx,
                        "global": str(current_global.resolve()),
                        "protocol": protocol,
                    }
                )
                fedsto.write_history(history)
                completed.add((phase, round_idx))
                print(f"Recovered phase {phase} round {round_idx} from existing global checkpoint.")
                if not args.keep_intermediate_checkpoints:
                    fedsto.cleanup_round_intermediates(phase, round_idx)
                continue

            local_paths: list[Path] = []
            for client in setup.CLIENTS:
                target = setup.LIST_ROOT / f"client_{client['id']}_{client['weather']}_target.txt"
                start = fedsto.CLIENT_STATE_DIR / f"client_{client['id']}_phase{phase}_round{round_idx:03d}_start.pt"
                previous = fedsto.CLIENT_STATE_DIR / f"client_{client['id']}_latest.pt"
                run_name = f"phase{phase}_round{round_idx:03d}_client{client['id']}_{client['weather']}"
                client_stats = client_pseudo_stats_path(phase, round_idx, client)
                stats_required = (
                    phase == 2
                    and (
                        args.dqa_collect_pseudo_stats
                        or args.dqa_router_mode == "shared_routed"
                    )
                )
                ckpt = fedsto.reuse_checkpoint_if_valid(
                    fedsto.checkpoint_path(run_name),
                    f"client run {run_name}",
                    force_retrain=args.force_retrain,
                    expected_protocol=protocol,
                )
                if ckpt is not None and stats_required and not client_stats.exists():
                    print(
                        f"Existing client run {run_name} is valid but pseudo stats are missing "
                        f"({client_stats}); rerunning this client for shared-routed DQA."
                    )
                    ckpt = None
                if ckpt is not None:
                    local_paths.append(ckpt)
                    fedsto.make_start_checkpoint(
                        ckpt,
                        previous,
                        protocol=protocol,
                        stage=f"client_{client['id']}_latest_observation_only",
                    )
                    continue

                if not fedsto.checkpoint_matches_protocol(start, protocol):
                    local_ema_source = previous if args.persist_client_ema_across_rounds else None
                    fedsto.make_start_checkpoint(
                        current_global,
                        start,
                        local_ema_source,
                        reset_ema_to_model=local_ema_source is None,
                        protocol=protocol,
                        stage=(
                            f"phase{phase}_round{round_idx:03d}_client{client['id']}_start_"
                            + ("persistent_local_ema" if local_ema_source else "broadcast_reset_ema")
                        ),
                    )
                cfg = fedsto.write_runtime_config(
                    run_name,
                    target=target,
                    weights=start,
                    phase=phase,
                    role="client",
                    round_idx=round_idx,
                    batch_size=args.batch_size,
                    workers=args.workers,
                    device=config_device,
                )
                extra_env = None
                if stats_required:
                    client_stats.parent.mkdir(parents=True, exist_ok=True)
                    if client_stats.exists():
                        client_stats.unlink()
                    extra_env = {
                        "DQA_PSEUDO_STATS_OUT": str(client_stats.resolve()),
                        "DQA_CLIENT_ID": str(client["id"]),
                        "DQA_PHASE": str(phase),
                        "DQA_ROUND": str(round_idx),
                        "DQA0834_STATS_QUALITY_MODE": args.dqa_pseudo_quality_mode,
                        "DQA_STATS_QUALITY_MODE": args.dqa_pseudo_quality_mode,
                    }
                ckpt = run_train_with_env(cfg, args, extra_env=extra_env) if extra_env else fedsto.run_train(
                    cfg,
                    False,
                    gpus=args.gpus,
                    master_port=args.master_port,
                )
                fedsto.mark_checkpoint_protocol(
                    ckpt,
                    protocol,
                    f"phase{phase}_round{round_idx:03d}_client{client['id']}",
                )
                local_paths.append(ckpt)
                fedsto.make_start_checkpoint(
                    ckpt,
                    previous,
                    protocol=protocol,
                    stage=f"client_{client['id']}_latest_observation_only",
                )

            client_aggregate = fedsto.GLOBAL_DIR / f"phase{phase}_round{round_idx:03d}_client_aggregate.pt"
            fedsto.aggregate_checkpoints(local_paths, current_global, client_aggregate, backbone_only=(phase == 1))
            fedsto.mark_checkpoint_protocol(
                client_aggregate,
                protocol,
                f"phase{phase}_round{round_idx:03d}_client_aggregate",
            )

            server_start = fedsto.GLOBAL_DIR / f"phase{phase}_round{round_idx:03d}_server_start.pt"
            server_name = f"phase{phase}_round{round_idx:03d}_server"
            server_ckpt = fedsto.reuse_checkpoint_if_valid(
                fedsto.checkpoint_path(server_name),
                f"server run {server_name}",
                force_retrain=args.force_retrain,
                expected_protocol=protocol,
            )
            if server_ckpt is None:
                if not fedsto.checkpoint_matches_protocol(server_start, protocol):
                    fedsto.make_start_checkpoint(
                        client_aggregate,
                        server_start,
                        protocol=protocol,
                        stage=f"phase{phase}_round{round_idx:03d}_server_start",
                    )
                server_cfg = fedsto.write_runtime_config(
                    server_name,
                    target=None,
                    weights=server_start,
                    phase=phase,
                    role="server",
                    round_idx=round_idx,
                    batch_size=args.batch_size,
                    workers=args.workers,
                    device=config_device,
                )
                server_ckpt = fedsto.run_train(server_cfg, False, gpus=args.gpus, master_port=args.master_port)
                fedsto.mark_checkpoint_protocol(
                    server_ckpt,
                    protocol,
                    f"phase{phase}_round{round_idx:03d}_server_update",
                )

            server_metrics = metric_summary(read_last_results_row(server_name))
            round_row: dict[str, Any] = {
                "phase": phase,
                "round": round_idx,
                "protocol": protocol,
                "server_checkpoint": str(server_ckpt.resolve()),
                "server_metrics": server_metrics,
                "dqa_applied": False,
                "dqa_accepted": False,
                "dqa_reason": "not_scheduled",
            }

            selected_checkpoint = server_ckpt
            apply_dqa = (
                phase == 2
                and round_idx >= args.dqa_start_round
                and ((round_idx - args.dqa_start_round) % max(args.dqa_eval_every, 1) == 0)
            )
            if apply_dqa:
                evaluated_candidates: list[dict[str, Any]] = []
                for index, spec in enumerate(candidate_specs(args, round_idx), start=1):
                    label = safe_label(
                        f"phase{phase}_round{round_idx:03d}_dqa_moe_{index:02d}_{spec['name']}"
                    )
                    candidate = fedsto.GLOBAL_DIR / f"{label}.pt"
                    candidate, meta = create_dqa_moe_candidate(
                        server_ckpt=server_ckpt,
                        client_paths=local_paths,
                        out=candidate,
                        phase=phase,
                        round_idx=round_idx,
                        args=args,
                        scope=spec["scope"],
                        residual_lambda=spec["lambda"],
                        max_relative_update=spec["max_relative_update"],
                        routing_focus=spec.get("routing_focus"),
                    )
                    candidate_metrics = evaluate_cloudy_candidate(candidate, label, args)
                    evaluated_candidates.append(
                        {
                            "name": spec["name"],
                            "label": label,
                            "path": str(candidate.resolve()),
                            "spec": spec,
                            "dqa_meta": meta,
                            "candidate_metrics": candidate_metrics,
                        }
                    )

                best, reason = select_best_candidate(
                    server_metrics=server_metrics,
                    candidate_rows=evaluated_candidates,
                    args=args,
                )
                accepted = best is not None
                if accepted:
                    selected_checkpoint = Path(best["path"])
                    meta = best["dqa_meta"]
                    candidate_metrics = best["candidate_metrics"]
                    candidate_path = best["path"]
                else:
                    meta = evaluated_candidates[0]["dqa_meta"] if evaluated_candidates else {}
                    candidate_metrics = {}
                    candidate_path = ""
                round_row.update(
                    {
                        "dqa_applied": True,
                        "dqa_accepted": accepted,
                        "dqa_reason": reason,
                        "dqa_candidate": candidate_path,
                        "dqa_lambda": meta.get("lambda"),
                        "dqa_scope": meta.get("scope"),
                        "dqa_meta": meta,
                        "candidate_metrics": candidate_metrics,
                        "evaluated_candidates": evaluated_candidates,
                    }
                )
                print(
                    f"DQA-MOE phase {phase} round {round_idx}: {reason}; "
                    f"server mAP50={server_metrics.get('map50'):.5f}, "
                    f"best mAP50={candidate_metrics.get('map50', float('nan')):.5f}, "
                    f"candidates={len(evaluated_candidates)}"
                )

            fedsto.make_start_checkpoint(
                selected_checkpoint,
                next_global,
                protocol=protocol,
                stage=(
                    f"phase{phase}_round{round_idx:03d}_global_after_"
                    + ("dqa_moe" if round_row.get("dqa_accepted") else "server_update")
                ),
            )
            current_global = next_global
            round_row["global"] = str(current_global.resolve())
            history.append(
                {
                    "phase": phase,
                    "round": round_idx,
                    "global": str(current_global.resolve()),
                    "protocol": protocol,
                    "dqa_applied": round_row.get("dqa_applied"),
                    "dqa_accepted": round_row.get("dqa_accepted"),
                    "dqa_reason": round_row.get("dqa_reason"),
                }
            )
            fedsto.write_history(history)
            dqa_history.append(round_row)
            write_dqa_history(dqa_history)
            append_round_summary(round_row)
            completed.add((phase, round_idx))

            print(f"Completed phase {phase} round {round_idx}: {current_global}")
            if args.discord and (round_row.get("dqa_applied") or round_idx == rounds):
                notify(
                    args,
                    "DQA x MOE round update",
                    (
                        f"phase={phase}, round={round_idx}\n"
                        f"server cloudy mAP50={server_metrics.get('map50'):.5f}, "
                        f"mAP50:95={server_metrics.get('map50_95'):.5f}\n"
                        f"dqa={round_row.get('dqa_reason')}"
                    ),
                    context={"workspace": str(args.workspace_root)},
                )
            if not args.keep_intermediate_checkpoints:
                fedsto.cleanup_round_intermediates(phase, round_idx)

    run_final_eval(args)
    notify(
        args,
        "DQA x MOE Trust Region finished",
        f"Final global: `{current_global}`\nSummary CSV: `{DQA_STATS_PATH}`",
        context={"workspace": str(args.workspace_root)},
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--warmup-epochs", type=int, default=50)
    parser.add_argument("--phase1-rounds", type=int, default=20)
    parser.add_argument("--phase2-rounds", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU-only execution. Intended only for tiny debugging runs, not the 12h experiment.",
    )
    parser.add_argument("--master-port", type=int, default=29541)
    parser.add_argument("--device", default="")
    parser.add_argument("--force-warmup", action="store_true")
    parser.add_argument("--force-restart", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--keep-intermediate-checkpoints", action="store_true")
    parser.add_argument("--persist-client-ema-across-rounds", action="store_true")
    parser.add_argument(
        "--protocol-version",
        default="",
        help=(
            "Optional explicit checkpoint protocol tag. By default the runner uses the original "
            "trust-region protocol for metric mode and the v2 tag for shared_routed mode."
        ),
    )

    parser.add_argument("--dqa-start-round", type=int, default=6)
    parser.add_argument("--dqa-eval-every", type=int, default=1)
    parser.add_argument("--dqa-scope", choices=("head", "head_bn", "head_neck_bn", "neck_head"), default="head_bn")
    parser.add_argument("--dqa-search-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dqa-candidate-scopes",
        default="head,head_bn,neck_head",
        help="Comma-separated scopes searched when --dqa-search-candidates is enabled.",
    )
    parser.add_argument(
        "--dqa-candidate-lambda-multipliers",
        default="0.75,1.00",
        help="Comma-separated residual multipliers searched per scope.",
    )
    parser.add_argument(
        "--dqa-max-candidates",
        type=int,
        default=6,
        help="Maximum DQA candidates evaluated per scheduled round.",
    )
    parser.add_argument("--dqa-lambda-start", type=float, default=0.015)
    parser.add_argument("--dqa-lambda-end", type=float, default=0.05)
    parser.add_argument("--dqa-max-relative-update", type=float, default=0.01)
    parser.add_argument("--dqa-max-absolute-update", type=float, default=1e-4)
    parser.add_argument("--dqa-acceptance-tolerance-map50", type=float, default=0.003)
    parser.add_argument("--dqa-acceptance-tolerance-map50-95", type=float, default=0.002)
    parser.add_argument("--dqa-acceptance-score-tolerance", type=float, default=0.001)
    parser.add_argument("--dqa-score-map50-weight", type=float, default=0.20)
    parser.add_argument("--dqa-score-recall-weight", type=float, default=0.05)
    parser.add_argument("--dqa-router-mode", choices=("metric", "shared_routed"), default="metric")
    parser.add_argument(
        "--dqa-router-candidates",
        default="balanced",
        help=(
            "Comma-separated routing focuses used as DQA candidates in shared_routed mode. "
            "Useful values: balanced,overcast,rainy,snowy,bad_weather,quality."
        ),
    )
    parser.add_argument("--dqa-router-temperature", type=float, default=1.35)
    parser.add_argument("--dqa-expert-min-weight", type=float, default=0.08)
    parser.add_argument("--dqa-focus-boost", type=float, default=1.25)
    parser.add_argument("--dqa-router-proxy-weight", type=float, default=0.0)
    parser.add_argument("--dqa-pseudo-quality-weight", type=float, default=0.35)
    parser.add_argument("--dqa-pseudo-balance-weight", type=float, default=0.15)
    parser.add_argument("--dqa-pseudo-rare-weight", type=float, default=0.20)
    parser.add_argument("--dqa-pseudo-class-count-weight", type=float, default=0.25)
    parser.add_argument("--dqa-rare-class-ids", default="1,3,4,5,6,9")
    parser.add_argument("--dqa-pseudo-quality-mode", default="feature_balanced")
    parser.add_argument("--dqa-collect-pseudo-stats", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dqa-classwise-head-routing", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--val-imgsz", type=int, default=640)
    parser.add_argument("--val-conf-thres", type=float, default=0.001)
    parser.add_argument("--val-iou-thres", type=float, default=0.6)
    parser.add_argument("--val-device", default="")
    parser.add_argument("--val-python", type=Path, default=None)
    parser.add_argument("--run-final-eval", action="store_true")
    parser.add_argument("--final-eval-splits", default="cloudy,overcast,rainy,snowy,total")
    parser.add_argument("--final-eval-plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--discord", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.workspace_root = args.workspace_root.expanduser().resolve()
    run_protocol(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
