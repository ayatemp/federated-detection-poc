#!/usr/bin/env python3
"""Run DQA08_3_9 phase2 with R-SCoLQ-gated pseudo boxes.

R-SCoLQ (Source-Calibrated Localization Quality Judge) replaces teacher
confidence as the pseudo-label gate score.  The original objectness and class
confidence are kept, but bbox/cls-positive assignment is controlled by the
source-calibrated localization score.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import dqa_cwa_aggregation_v2 as v2
import run_dqa_cwa_fedsto as base
import run_dqa_cwa_fedsto_scene_v2_tri_stage_policy as tri_stage
from run_dqa_cwa_fedsto_scene import _prepare_scene_modules


DQA_PROTOCOL_VERSION = "dqa08_3_9_scene_phase2_rscolq_anti_drift_policy_v1"
DEFAULT_WORK_ROOT = base.RESEARCH_ROOT / "efficientteacher_dqa08_3_9_phase2_rscolq_anti_drift_policy"
DEFAULT_SOURCE_WORK_ROOT = base.RESEARCH_ROOT / "efficientteacher_dqa08_scene_tri_stage_policy_8h"
DEFAULT_SOURCE_PHASE1_ROUND = 3
DEFAULT_RSCOLQ_MODEL = (
    base.RESEARCH_ROOT
    / "source_calibrated_localization_quality"
    / "artifacts"
    / "rscolq_best.joblib"
)
ROUND_RE = re.compile(r"phase2_round(?P<round>\d+)")
_RSCOLQ_POLICY_CACHE: dict[str, Any] | None = None


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return int(raw)


def _source_workspace() -> Path:
    return Path(os.environ.get("DQA0839_SOURCE_WORK_ROOT", str(DEFAULT_SOURCE_WORK_ROOT))).resolve()


def _source_phase1_round() -> int:
    return _int_env("DQA0839_SOURCE_PHASE1_ROUND", DEFAULT_SOURCE_PHASE1_ROUND)


def _rscolq_model_path() -> Path:
    return Path(os.environ.get("DQA0839_RSCOLQ_MODEL", str(DEFAULT_RSCOLQ_MODEL))).resolve()


def _round_from_path(path: Path) -> int | None:
    match = ROUND_RE.search(str(path))
    return int(match.group("round")) if match else None


def _round_from_run_name(name: str) -> int | None:
    parsed = tri_stage._parse_run_name(name)
    return int(parsed["round"]) if parsed and parsed.get("round") is not None else None


def _scheduled(prefix: str, round_idx: int | None, default_start: float, default_end: float | None = None) -> float:
    start = _float_env(f"{prefix}_START", default_start)
    end = _float_env(f"{prefix}_END", start if default_end is None else default_end)
    total = max(_int_env("DQA0839_PHASE2_ROUNDS", 10), 1)
    if round_idx is None or total <= 1:
        return start
    progress = min(max((round_idx - 1) / (total - 1), 0.0), 1.0)
    return start + (end - start) * progress


def _scheduled_value(name: str, round_idx: int | None, default: float) -> float:
    base_value = _float_env(name, default)
    return _scheduled(name, round_idx, base_value, base_value)


def _parse_float_list(raw: str | None, nc: int) -> list[float] | None:
    if raw in (None, ""):
        return None
    try:
        values = json.loads(raw)
    except json.JSONDecodeError:
        values = [part.strip() for part in raw.split(",") if part.strip()]
    if not isinstance(values, list) or len(values) != nc:
        raise ValueError(f"Expected {nc} threshold values, got {values}")
    return [float(value) for value in values]


def _rscolq_thresholds(nc: int) -> tuple[list[float], list[float], list[float]]:
    low = _float_env("DQA0839_RSCOLQ_LOW", 0.10)
    mid = _float_env("DQA0839_RSCOLQ_MID", 0.30)
    high = _float_env("DQA0839_RSCOLQ_HIGH", 0.60)
    lows = _parse_float_list(os.environ.get("DQA0839_CLASSWISE_LOW"), nc) or [low] * nc
    mids = _parse_float_list(os.environ.get("DQA0839_CLASSWISE_MID"), nc) or [mid] * nc
    highs = _parse_float_list(os.environ.get("DQA0839_CLASSWISE_HIGH"), nc) or [high] * nc
    out_low: list[float] = []
    out_mid: list[float] = []
    out_high: list[float] = []
    for lo, mi, hi in zip(lows, mids, highs):
        lo = min(max(float(lo), 0.0), 0.98)
        mi = min(max(float(mi), lo), 0.99)
        hi = min(max(float(hi), mi), 0.995)
        out_low.append(round(lo, 4))
        out_mid.append(round(mi, 4))
        out_high.append(round(hi, 4))
    return out_low, out_mid, out_high


def _rscolq_policy() -> dict[str, Any]:
    global _RSCOLQ_POLICY_CACHE
    if _RSCOLQ_POLICY_CACHE is not None:
        return _RSCOLQ_POLICY_CACHE
    try:
        import joblib

        bundle = joblib.load(_rscolq_model_path())
        policy = bundle.get("round_policy", {}) if isinstance(bundle, dict) else {}
    except Exception as exc:
        print(f"Warning: could not load R-SCoLQ round policy ({exc}); using neutral multipliers.")
        policy = {}
    _RSCOLQ_POLICY_CACHE = dict(policy or {})
    return _RSCOLQ_POLICY_CACHE


def _stats_root() -> Path | None:
    raw = os.environ.get("DQA08_STATS_ROOT", "").strip()
    if not raw:
        return None
    return Path(raw).resolve()


def _load_round_summary(round_idx: int | None, nc: int) -> dict[str, Any] | None:
    if round_idx is None or round_idx <= 0:
        return None
    root = _stats_root()
    if root is None:
        return None
    path = root / f"phase2_round{round_idx:03d}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    counts = [0.0] * nc
    quality_sums = [0.0] * nc
    for client in data.get("clients", []):
        for idx, value in enumerate(client.get("counts", [])[:nc]):
            counts[idx] += float(value)
        for idx, value in enumerate(client.get("quality_sums", [])[:nc]):
            quality_sums[idx] += float(value)
    total = sum(counts)
    if total <= 0:
        return None
    return {
        "round": round_idx,
        "path": str(path),
        "pseudo_total": total,
        "mean_quality": sum(quality_sums) / total,
        "class_counts": counts,
    }


def _previous_logged_multiplier(round_idx: int | None) -> float:
    if round_idx is None or round_idx <= 1:
        return 1.0
    raw = os.environ.get("DQA08_THRESHOLD_LOG", "").strip()
    if not raw:
        return 1.0
    path = Path(raw)
    if not path.exists():
        return 1.0
    wanted = round_idx - 1
    value = 1.0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(record.get("round") or 0) == wanted:
            value = float(record.get("global_multiplier", value))
    return min(max(value, 0.0), 1.0)


def _anchor_summary(round_idx: int | None, nc: int) -> tuple[dict[str, Any] | None, str]:
    # 08_3_9 anchors the raw-score scale to the first completed target round.
    # This avoids comparing R-SCoLQ raw probabilities against 08_3_6's
    # SCoLQ-scale source anchor, which caused the 1.0/0.15 oscillation.
    anchor_round = _int_env("DQA0839_RSCOLQ_ANCHOR_ROUND", 1)
    if round_idx is not None and anchor_round < round_idx:
        summary = _load_round_summary(anchor_round, nc)
        if summary is not None:
            return summary, f"target_round{anchor_round:03d}_raw"

    policy = _rscolq_policy()
    anchor_counts_raw = policy.get("anchor_class_counts", {}) or {}
    anchor_total = float(policy.get("anchor_total", 0.0) or 0.0)
    anchor_mean = float(policy.get("anchor_mean_scolq", 0.0) or 0.0)
    if anchor_total <= 0.0:
        return None, "missing_anchor"
    return {
        "round": None,
        "path": "artifact_round_policy",
        "pseudo_total": anchor_total,
        "mean_quality": anchor_mean,
        "class_counts": [float(anchor_counts_raw.get(str(idx), 1.0)) for idx in range(nc)],
    }, "artifact_source_policy"


def _rscolq_round_context(round_idx: int | None, nc: int) -> dict[str, Any]:
    neutral = {
        "round": round_idx,
        "previous_round": None,
        "previous_stats": None,
        "anchor_stats": None,
        "global_multiplier": 1.0,
        "target_multiplier": 1.0,
        "previous_multiplier": 1.0,
        "class_multipliers": [1.0] * nc,
        "mean_class_multiplier": 1.0,
        "reason": "neutral",
    }

    previous_round = (round_idx or 1) - 1
    summary = _load_round_summary(previous_round, nc)
    if summary is None:
        neutral.update({"previous_round": previous_round, "reason": "no_previous_stats"})
        return neutral

    anchor, anchor_reason = _anchor_summary(round_idx, nc)
    if anchor is None:
        neutral.update({"previous_round": previous_round, "previous_stats": summary, "reason": anchor_reason})
        return neutral

    global_budget_margin = _float_env("DQA0839_RSCOLQ_GLOBAL_BUDGET_MARGIN", 0.03)
    quality_margin = _float_env("DQA0839_RSCOLQ_QUALITY_MARGIN", 0.030)
    global_power = _float_env("DQA0839_RSCOLQ_GLOBAL_POWER", 1.20)
    quality_power = _float_env("DQA0839_RSCOLQ_QUALITY_POWER", 1.50)
    class_budget_margin = _float_env("DQA0839_RSCOLQ_CLASS_BUDGET_MARGIN", 0.05)
    class_power = _float_env("DQA0839_RSCOLQ_CLASS_POWER", 0.18)
    min_multiplier = _float_env("DQA0839_RSCOLQ_MIN_MULTIPLIER", 0.55)
    ema = min(max(_float_env("DQA0839_RSCOLQ_MULTIPLIER_EMA", 0.70), 0.0), 0.98)

    anchor_total = max(float(anchor["pseudo_total"]), 1.0)
    anchor_mean = float(anchor["mean_quality"])
    anchor_counts = [max(float(value), 1.0) for value in anchor["class_counts"][:nc]]

    pseudo_inflation = max(
        0.0,
        float(summary["pseudo_total"]) / (anchor_total * (1.0 + global_budget_margin)) - 1.0,
    )
    quality_inflation = max(
        0.0,
        float(summary["mean_quality"]) - anchor_mean - quality_margin,
    )
    target_multiplier = math.exp(-global_power * pseudo_inflation) * math.exp(-quality_power * quality_inflation)
    target_multiplier = min(max(target_multiplier, min_multiplier), 1.0)
    previous_multiplier = _previous_logged_multiplier(round_idx)
    global_multiplier = ema * previous_multiplier + (1.0 - ema) * target_multiplier
    global_multiplier = min(max(global_multiplier, min_multiplier), 1.0)

    class_multipliers: list[float] = []
    for idx in range(nc):
        ratio = float(summary["class_counts"][idx]) / (anchor_counts[idx] * (1.0 + class_budget_margin))
        if ratio > 1.0:
            value = ratio ** (-class_power)
        else:
            value = 1.0
        class_multipliers.append(round(min(max(value, min_multiplier), 1.0), 6))

    return {
        "round": round_idx,
        "previous_round": previous_round,
        "previous_stats": summary,
        "anchor_stats": anchor,
        "anchor_reason": anchor_reason,
        "global_multiplier": round(global_multiplier, 6),
        "target_multiplier": round(target_multiplier, 6),
        "previous_multiplier": round(previous_multiplier, 6),
        "class_multipliers": class_multipliers,
        "mean_class_multiplier": round(sum(class_multipliers) / max(len(class_multipliers), 1), 6),
        "pseudo_inflation": pseudo_inflation,
        "quality_inflation": quality_inflation,
        "reason": "raw_stats_ema_policy",
    }

def _log_rscolq_context(run_name: str, context: dict[str, Any]) -> None:
    raw = os.environ.get("DQA08_THRESHOLD_LOG", "").strip()
    if not raw:
        return
    path = Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"run_name": run_name, "variant": os.environ.get("DQA0839_VARIANT", "unnamed"), **context}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _seed_metadata_path(workspace_root: Path) -> Path:
    return workspace_root / "phase2_seed.json"


def _seed_phase1_checkpoint(workspace_root: Path) -> Path:
    workspace_root = workspace_root.resolve()
    source_root = _source_workspace()
    source_round = _source_phase1_round()
    source = source_root / "global_checkpoints" / f"phase1_round{source_round:03d}_global.pt"
    target = workspace_root / "global_checkpoints" / "round000_warmup.pt"
    metadata_path = _seed_metadata_path(workspace_root)
    force = _bool_env("DQA0839_FORCE_SEED", False)

    if not source.exists():
        raise FileNotFoundError(
            f"Missing source phase1 seed checkpoint: {source}\n"
            "Set DQA0839_SOURCE_WORK_ROOT / DQA0839_SOURCE_PHASE1_ROUND for a different seed."
        )

    wanted: dict[str, Any] = {
        "protocol": DQA_PROTOCOL_VERSION,
        "seed_kind": "dqa08_phase1_global_as_round000_warmup",
        "source_work_root": str(source_root),
        "source_phase1_round": source_round,
        "source_checkpoint": str(source),
        "target_checkpoint": str(target.resolve()),
        "variant": os.environ.get("DQA0839_VARIANT", "unnamed"),
        "rscolq_model": str(_rscolq_model_path()),
    }

    if target.exists() and not force:
        if metadata_path.exists():
            try:
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
            same_seed = (
                existing.get("protocol") == wanted["protocol"]
                and existing.get("source_checkpoint") == wanted["source_checkpoint"]
                and existing.get("variant") == wanted["variant"]
                and existing.get("rscolq_model") == wanted["rscolq_model"]
            )
            if same_seed:
                print(f"Reusing 08_3_9 phase2 seed checkpoint: {target}")
                return target
        raise RuntimeError(
            f"Seed checkpoint already exists with different metadata: {target}\n"
            f"Existing metadata: {metadata_path if metadata_path.exists() else '(missing)'}\n"
            "Use DQA0839_FORCE_SEED=1 only if you intentionally want to overwrite it."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    metadata_path.write_text(json.dumps(wanted, indent=2), encoding="utf-8")
    print(f"Seeded 08_3_9 phase2 from DQA08 phase1 round {source_round}: {target}")
    return target


def _apply_rscolq_ssod_profile(cfg: dict[str, Any], *, name: str, target: Path | None) -> None:
    if target is None:
        return

    nc = int(cfg["Dataset"]["nc"])
    round_idx = _round_from_run_name(name)
    lows, mids, highs = _rscolq_thresholds(nc)
    ssod = cfg.setdefault("SSOD", {})
    ssod.update(copy.deepcopy(tri_stage.BASE_PROFILE))
    ssod["nms_conf_thres"] = _scheduled_value("DQA0839_NMS_CONF_THRES", round_idx, 0.01)
    ssod["ignore_thres_low"] = min(lows)
    ssod["ignore_thres_high"] = min(highs)
    ssod["teacher_loss_weight"] = _scheduled_value("DQA0839_TEACHER_LOSS_WEIGHT", round_idx, 0.32)
    ssod["box_loss_weight"] = _scheduled_value("DQA0839_BOX_LOSS_WEIGHT", round_idx, 0.010)
    ssod["obj_loss_weight"] = _scheduled_value("DQA0839_OBJ_LOSS_WEIGHT", round_idx, 0.32)
    ssod["cls_loss_weight"] = _scheduled_value("DQA0839_CLS_LOSS_WEIGHT", round_idx, 0.08)
    ssod["ignore_obj"] = _bool_env("DQA0839_IGNORE_OBJ", False)
    ssod["pseudo_label_with_obj"] = True
    ssod["pseudo_label_with_bbox"] = False
    ssod["pseudo_label_with_cls"] = False


def _prepare_rscolq_modules(work_root: Path):
    setup, fedsto = _prepare_scene_modules(work_root)
    exact_setup = setup.base
    original_config = getattr(
        exact_setup,
        "_dqa0836_original_efficientteacher_config",
        exact_setup.efficientteacher_config,
    )
    exact_setup._dqa0836_original_efficientteacher_config = original_config

    def rscolq_config(**kwargs):
        if hasattr(setup, "_sync_base_paths"):
            setup._sync_base_paths()
        cfg = original_config(**kwargs)
        target = kwargs.get("target")
        name = str(kwargs.get("name", ""))
        round_idx = _round_from_run_name(name)

        if target is not None:
            cfg.setdefault("FedSTO", {})["train_scope"] = os.environ.get("DQA0839_CLIENT_TRAIN_SCOPE", "all")
            cfg["FedSTO"]["orthogonal_weight"] = _float_env("DQA0839_CLIENT_ORTHOGONAL_WEIGHT", 1e-4)
            cfg.setdefault("hyp", {})["lr0"] = _scheduled_value(
                "DQA0839_CLIENT_LR0",
                round_idx,
                cfg.get("hyp", {}).get("lr0", 0.01),
            )
            cfg["hyp"]["lrf"] = 1.0
            _apply_rscolq_ssod_profile(cfg, name=name, target=target)
        elif name != "runtime_server_warmup":
            cfg.setdefault("FedSTO", {})["train_scope"] = os.environ.get("DQA0839_SERVER_TRAIN_SCOPE", "all")
            cfg["FedSTO"]["orthogonal_weight"] = _float_env("DQA0839_SERVER_ORTHOGONAL_WEIGHT", 1e-4)
            cfg.setdefault("hyp", {})["lr0"] = _scheduled_value(
                "DQA0839_SERVER_LR0",
                round_idx,
                cfg.get("hyp", {}).get("lr0", 0.01),
            )
            cfg["hyp"]["lrf"] = 1.0

        return cfg

    setup.efficientteacher_config = rscolq_config
    exact_setup.efficientteacher_config = rscolq_config
    fedsto.setup.efficientteacher_config = rscolq_config
    return setup, fedsto


# Importing the tri-stage runner patches base.run_train as a side effect.  Use
# the raw runner it saved so R-SCoLQ thresholds are not overwritten by tri-stage
# confidence thresholds.
_ORIGINAL_RUN_TRAIN = getattr(tri_stage, "_ORIGINAL_RUN_TRAIN", base.run_train)


def _rscolq_run_train(fedsto, config: Path, args, *, extra_env: dict[str, str] | None = None):
    env = dict(extra_env or {})
    try:
        import yaml

        run_name = str(yaml.safe_load(config.read_text(encoding="utf-8"))["name"])
    except Exception:
        run_name = config.stem
    parsed = tri_stage._parse_run_name(run_name)

    env.setdefault("ET_SKIP_AFTER_TRAIN_BEST_VAL", os.environ.get("ET_SKIP_AFTER_TRAIN_BEST_VAL", "1"))
    env.setdefault("ET_MAX_PSEUDO_PER_IMAGE", os.environ.get("DQA0839_MAX_PSEUDO_PER_IMAGE", "80"))
    env.setdefault("ET_MAX_PSEUDO_PER_CLASS_IMAGE", os.environ.get("DQA0839_MAX_PSEUDO_PER_CLASS_IMAGE", "25"))
    env["DQA0834_STATS_QUALITY_MODE"] = "rscolq_raw"
    env["DQA_STATS_QUALITY_MODE"] = "rscolq_raw"

    if parsed and parsed["role"].startswith("client") and parsed["phase"] >= 2:
        lows, mids, highs = _rscolq_thresholds(args.num_classes)
        context = _rscolq_round_context(parsed.get("round"), args.num_classes)
        env["DQA0839_RSCOLQ_ENABLE"] = "1"
        env["DQA0839_RSCOLQ_MODEL"] = str(_rscolq_model_path())
        env["DQA0839_RSCOLQ_SCORE_POWER"] = os.environ.get("DQA0839_RSCOLQ_SCORE_POWER", "1.0")
        env["DQA0839_RSCOLQ_RAW_STATS"] = "1"
        env["DQA0839_RSCOLQ_GLOBAL_MULTIPLIER"] = str(context["global_multiplier"])
        env["DQA0839_RSCOLQ_CLASS_MULTIPLIERS"] = json.dumps(context["class_multipliers"])
        env["DQA0836_SCOLQ_ENABLE"] = "1"
        env["DQA0836_SCOLQ_MODEL"] = str(_rscolq_model_path())
        env["DQA0836_SCOLQ_SCORE_POWER"] = os.environ.get("DQA0839_RSCOLQ_SCORE_POWER", "1.0")
        env["DQA06_NMS_CONF_THRES"] = str(_float_env("DQA0839_NMS_CONF_THRES", 0.01))
        env["DQA06_IGNORE_THRES_LOW"] = json.dumps(lows)
        env["DQA08_IGNORE_THRES_MID"] = json.dumps(mids)
        env["DQA06_IGNORE_THRES_HIGH"] = json.dumps(highs)
        env["DQA08_TRI_STAGE_GATE"] = "1"
        env["DQA08_LOW_MID_OBJ_WEIGHT"] = str(_float_env("DQA0839_LOW_MID_OBJ_WEIGHT", 0.25))
        env["DQA08_MID_HIGH_OBJ_WEIGHT"] = str(_float_env("DQA0839_MID_HIGH_OBJ_WEIGHT", 0.70))
        _log_rscolq_context(run_name, context)
        print(
            "DQA08_3_9 R-SCoLQ gate:",
            f"client={parsed['client_id']}",
            f"round={parsed['round']}",
            f"low={lows[0]:.2f}",
            f"mid={mids[0]:.2f}",
            f"high={highs[0]:.2f}",
            f"global_mult={context['global_multiplier']:.3f}",
            f"target_mult={context['target_multiplier']:.3f}",
            f"class_mult_mean={context['mean_class_multiplier']:.3f}",
            f"context={context['reason']}",
            f"model={_rscolq_model_path()}",
        )

    return _ORIGINAL_RUN_TRAIN(fedsto, config, args, extra_env=env or None)


def _dqa_config(args) -> v2.AggregationConfig:
    return v2.AggregationConfig(
        num_classes=args.num_classes,
        count_ema=args.count_ema,
        quality_ema=args.quality_ema,
        alpha_ema=args.alpha_ema,
        temperature=args.temperature,
        uniform_mix=args.uniform_mix,
        classwise_blend=args.classwise_blend,
        stability_lambda=args.stability_lambda,
        min_effective_count=args.min_effective_count,
        server_anchor=args.server_anchor,
        localize_bn=args.localize_bn,
        min_server_alpha=_float_env("DQA0839_MIN_SERVER_ALPHA", 0.70),
        residual_blend=_float_env("DQA0839_RESIDUAL_START", 0.14),
    )


def _aggregate_rscolq(
    client_checkpoints,
    server_checkpoint,
    output_checkpoint,
    stats,
    state_path,
    config,
    repo_root,
):
    round_idx = _round_from_path(output_checkpoint)
    residual = _scheduled("DQA0839_RESIDUAL", round_idx, 0.14, 0.06)
    min_server_alpha = _scheduled(
        "DQA0839_MIN_SERVER_ALPHA",
        round_idx,
        _float_env("DQA0839_MIN_SERVER_ALPHA_START", _float_env("DQA0839_MIN_SERVER_ALPHA", 0.70)),
        _float_env("DQA0839_MIN_SERVER_ALPHA_END", _float_env("DQA0839_MIN_SERVER_ALPHA", 0.74)),
    )
    cfg = replace(
        copy.copy(config),
        residual_blend=residual,
        min_server_alpha=min_server_alpha,
    )
    output, state = v2.aggregate_checkpoints(
        client_checkpoints=client_checkpoints,
        server_checkpoint=server_checkpoint,
        output_checkpoint=output_checkpoint,
        stats=stats,
        state_path=state_path,
        config=cfg,
        repo_root=repo_root,
    )
    lows, mids, highs = _rscolq_thresholds(cfg.num_classes)
    context = _rscolq_round_context(round_idx, cfg.num_classes)
    state.setdefault("phase2_rscolq_anti_drift_policy", []).append(
        {
            "round": round_idx,
            "variant": os.environ.get("DQA0839_VARIANT", "unnamed"),
            "rscolq_model": str(_rscolq_model_path()),
            "rscolq_low": lows,
            "rscolq_mid": mids,
            "rscolq_high": highs,
            "rscolq_global_multiplier": context["global_multiplier"],
            "rscolq_target_multiplier": context["target_multiplier"],
            "rscolq_previous_multiplier": context["previous_multiplier"],
            "rscolq_mean_class_multiplier": context["mean_class_multiplier"],
            "rscolq_class_multipliers": context["class_multipliers"],
            "rscolq_context_reason": context["reason"],
            "rscolq_previous_round": context["previous_round"],
            "rscolq_anchor_reason": context.get("anchor_reason"),
            "rscolq_pseudo_inflation": context.get("pseudo_inflation"),
            "rscolq_quality_inflation": context.get("quality_inflation"),
            "residual_blend": residual,
            "min_server_alpha": cfg.min_server_alpha,
            "classwise_blend": cfg.classwise_blend,
            "server_anchor": cfg.server_anchor,
        }
    )
    v2.save_state(state_path, state)
    print(
        "DQA08_3_9 aggregate:",
        f"variant={os.environ.get('DQA0839_VARIANT', 'unnamed')}",
        f"round={round_idx}",
        f"rscolq_high={highs[0]:.2f}",
        f"rscolq_global_mult={context['global_multiplier']:.3f}",
        f"rscolq_target_mult={context['target_multiplier']:.3f}",
        f"residual_blend={residual:.4f}",
        f"min_server_alpha={cfg.min_server_alpha:.3f}",
    )
    return output, state


base.DQA_PROTOCOL_VERSION = DQA_PROTOCOL_VERSION
base.DEFAULT_DQA_WORK_ROOT = DEFAULT_WORK_ROOT
base._prepare_fedsto_modules = _prepare_rscolq_modules
base.run_train = _rscolq_run_train
base.AggregationConfig = v2.AggregationConfig
base.aggregate_fedavg_checkpoints = v2.aggregate_fedavg_checkpoints
base.aggregate_checkpoints = _aggregate_rscolq
base.compute_reliability = v2.compute_reliability
base.load_round_stats = v2.load_round_stats
base.load_state = v2.load_state
base.save_state = v2.save_state
base._dqa_config = _dqa_config


def main() -> None:
    parsed = base.parse_args()
    if parsed.protocol_version != DQA_PROTOCOL_VERSION:
        parsed.protocol_version = DQA_PROTOCOL_VERSION
    if parsed.workspace_root == DEFAULT_SOURCE_WORK_ROOT:
        parsed.workspace_root = DEFAULT_WORK_ROOT

    os.environ["DQA08_STATS_ROOT"] = str(parsed.stats_root.resolve())
    os.environ["DQA08_PHASE2_ROUNDS"] = str(parsed.phase2_rounds)
    os.environ["DQA0839_PHASE2_ROUNDS"] = str(parsed.phase2_rounds)
    os.environ["DQA08_NUM_CLASSES"] = str(parsed.num_classes)
    os.environ.setdefault("DQA08_POLICY_HORIZON_ROUNDS", str(parsed.phase2_rounds))
    os.environ.setdefault(
        "DQA08_THRESHOLD_LOG",
        str((parsed.stats_root / "phase2_rscolq_anti_drift_policy_schedule.jsonl").resolve()),
    )

    if parsed.setup_only:
        base._prepare_fedsto_modules(parsed.workspace_root)[0].build_base_configs()
        return

    if parsed.phase1_rounds != 0:
        raise ValueError("08_3_9 is phase2-only. Run with --phase1-rounds 0.")

    if not parsed.dry_run:
        if not _rscolq_model_path().exists():
            raise FileNotFoundError(
                f"Missing R-SCoLQ artifact: {_rscolq_model_path()}. "
                "Run source_calibrated_localization_quality/02_train_and_validate_round_stable_scolq.ipynb first."
            )
        _seed_phase1_checkpoint(parsed.workspace_root)

    lows, mids, highs = _rscolq_thresholds(parsed.num_classes)
    print("DQA08_3_9 profile: phase2-only R-SCoLQ-gated pseudo boxes")
    print(f"DQA08_3_9 variant: {os.environ.get('DQA0839_VARIANT', 'unnamed')}")
    print(f"DQA08_3_9 seed: {_source_workspace()} phase1 round {_source_phase1_round()}")
    print(f"DQA08_3_9 R-SCoLQ model: {_rscolq_model_path()}")
    print(f"DQA08_3_9 thresholds: low={lows} mid={mids} high={highs}")
    print(f"DQA08_3_9 client train_scope: {os.environ.get('DQA0839_CLIENT_TRAIN_SCOPE', 'all')}")
    print(
        "DQA08_3_9 residual schedule:",
        _float_env("DQA0839_RESIDUAL_START", 0.14),
        "->",
        _float_env("DQA0839_RESIDUAL_END", 0.06),
    )
    print(
        "DQA08_3_9 pseudo loss:",
        f"teacher={_float_env('DQA0839_TEACHER_LOSS_WEIGHT', 0.32)}",
        f"box={_float_env('DQA0839_BOX_LOSS_WEIGHT', 0.010)}",
        f"obj={_float_env('DQA0839_OBJ_LOSS_WEIGHT', 0.32)}",
        f"cls={_float_env('DQA0839_CLS_LOSS_WEIGHT', 0.08)}",
    )
    print(f"DQA08_3_9 dqa_start_phase: {parsed.dqa_start_phase}")
    base.run_protocol(parsed)


if __name__ == "__main__":
    main()
