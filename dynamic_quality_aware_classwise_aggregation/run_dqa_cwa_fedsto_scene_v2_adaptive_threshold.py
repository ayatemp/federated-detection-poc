#!/usr/bin/env python3
"""Run DQA-CWA v2 with GT-free adaptive pseudo-label thresholds.

This wrapper is for notebook 06.  It keeps the 05 scene/class setup and DQA v2
server-anchored aggregation, but feeds the previous round's pseudo-label stats
back into the next client round.  The feedback changes pseudo-label gates without
using target GT:

- per-client/per-class low and high pseudo-label thresholds are exported through
  environment variables consumed by the EfficientTeacher SSOD loss
- the NMS confidence threshold and pseudo-label loss weights are adjusted as
  scalar config values
- a JSONL threshold log is written for later inspection
"""

from __future__ import annotations

import json
import math
import os
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import dqa_cwa_aggregation_v2 as v2
import run_dqa_cwa_fedsto as base
from run_dqa_cwa_fedsto_scene import _prepare_scene_modules


DQA_PROTOCOL_VERSION = "dqa06_scene_adaptive_pseudogt_v1"
RUN_RE = re.compile(r"dqa_phase(?P<phase>\d+)_round(?P<round>\d+)_(?P<role>client(?P<client>\d+)|server)")

BASE_PROFILE: dict[str, Any] = {
    "nms_conf_thres": 0.35,
    "ignore_thres_low": 0.35,
    "ignore_thres_high": 0.75,
    "teacher_loss_weight": 0.35,
    "box_loss_weight": 0.02,
    "obj_loss_weight": 0.35,
    "cls_loss_weight": 0.10,
    "pseudo_label_with_bbox": False,
    "pseudo_label_with_cls": False,
}


@dataclass(frozen=True)
class ThresholdDecision:
    enabled: bool
    reason: str
    phase: int | None
    round: int | None
    client_id: str | None
    nms_conf_thres: float
    ignore_thres_low: list[float]
    ignore_thres_high: list[float]
    teacher_loss_weight: float
    box_loss_weight: float
    obj_loss_weight: float
    cls_loss_weight: float
    source_stats: str | None = None


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


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))


def _profile_name() -> str:
    return os.environ.get("DQA06_SSOD_PROFILE", "adaptive_strict_low_bbox")


def _stats_root() -> Path | None:
    raw = os.environ.get("DQA06_STATS_ROOT")
    return Path(raw).resolve() if raw else None


def _threshold_log() -> Path | None:
    raw = os.environ.get("DQA06_THRESHOLD_LOG")
    if raw:
        return Path(raw).resolve()
    stats_root = _stats_root()
    return stats_root / "adaptive_threshold_schedule.jsonl" if stats_root else None


def _parse_run_name(name: str) -> dict[str, Any] | None:
    match = RUN_RE.search(name)
    if not match:
        return None
    return {
        "phase": int(match.group("phase")),
        "round": int(match.group("round")),
        "role": "client" if match.group("client") is not None else "server",
        "client_id": match.group("client"),
    }


def _client_stats_path(stats_root: Path, phase: int, round_idx: int, client_id: str) -> Path:
    return stats_root / f"phase{phase}_round{round_idx:03d}_client{client_id}.json"


def _load_client_history(stats_root: Path, phase: int, round_idx: int, client_id: str) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for previous_round in range(1, round_idx):
        path = _client_stats_path(stats_root, phase, previous_round, client_id)
        if not path.exists():
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        item["_path"] = str(path)
        history.append(item)
    return history


def _weighted_percentile(values: list[float], weights: list[float], q: float) -> float:
    pairs = sorted((float(v), max(float(w), 0.0)) for v, w in zip(values, weights))
    total = sum(weight for _, weight in pairs)
    if total <= 0:
        return sorted(values)[min(len(values) - 1, max(0, int(q * (len(values) - 1))))]
    threshold = total * q
    running = 0.0
    for value, weight in pairs:
        running += weight
        if running >= threshold:
            return value
    return pairs[-1][0]


def _round_progress(round_idx: int) -> float:
    start = _int_env("DQA06_ADAPT_START_ROUND", 3)
    ramp = max(_int_env("DQA06_RAMP_ROUNDS", 10), 1)
    if round_idx < start:
        return 0.0
    return _clamp((round_idx - start + 1) / ramp, 0.0, 1.0)


def _base_decision(phase: int | None, round_idx: int | None, client_id: str | None, reason: str) -> ThresholdDecision:
    nc = _int_env("DQA06_NUM_CLASSES", 10)
    low = float(BASE_PROFILE["ignore_thres_low"])
    high = float(BASE_PROFILE["ignore_thres_high"])
    return ThresholdDecision(
        enabled=False,
        reason=reason,
        phase=phase,
        round=round_idx,
        client_id=client_id,
        nms_conf_thres=float(BASE_PROFILE["nms_conf_thres"]),
        ignore_thres_low=[low] * nc,
        ignore_thres_high=[high] * nc,
        teacher_loss_weight=float(BASE_PROFILE["teacher_loss_weight"]),
        box_loss_weight=float(BASE_PROFILE["box_loss_weight"]),
        obj_loss_weight=float(BASE_PROFILE["obj_loss_weight"]),
        cls_loss_weight=float(BASE_PROFILE["cls_loss_weight"]),
    )


def decide_thresholds(name: str, num_classes: int) -> ThresholdDecision:
    parsed = _parse_run_name(name)
    if parsed is None or parsed["role"] != "client" or parsed["phase"] < 2:
        return _base_decision(
            parsed["phase"] if parsed else None,
            parsed["round"] if parsed else None,
            parsed["client_id"] if parsed else None,
            "non-client-or-pre-dqa",
        )

    phase = int(parsed["phase"])
    round_idx = int(parsed["round"])
    client_id = str(parsed["client_id"])
    stats_root = _stats_root()
    start_round = _int_env("DQA06_ADAPT_START_ROUND", 3)
    if stats_root is None:
        return _base_decision(phase, round_idx, client_id, "missing-stats-root")
    if round_idx < start_round:
        return _base_decision(phase, round_idx, client_id, "warm-adaptation-round")

    history = _load_client_history(stats_root, phase, round_idx, client_id)
    if not history:
        return _base_decision(phase, round_idx, client_id, "missing-previous-client-stats")

    latest = history[-1]
    previous = history[-2] if len(history) >= 2 else latest
    latest_counts = [float(x) for x in latest["counts"][:num_classes]]
    latest_quality = [float(x) for x in latest["mean_quality_scores"][:num_classes]]
    previous_counts = [float(x) for x in previous["counts"][:num_classes]]
    previous_quality = [float(x) for x in previous["mean_quality_scores"][:num_classes]]

    progress = _round_progress(round_idx)
    min_low = _float_env("DQA06_MIN_LOW", 0.35)
    max_low = _float_env("DQA06_MAX_LOW", 0.55)
    min_high = _float_env("DQA06_MIN_HIGH", 0.75)
    max_high = _float_env("DQA06_MAX_HIGH", 0.90)
    base_low = _float_env("DQA06_BASE_LOW", 0.35)
    base_high_gap = _float_env("DQA06_HIGH_GAP", 0.40)
    ramp_raise = _float_env("DQA06_RAMP_RAISE", 0.10) * progress
    quality_target = _float_env("DQA06_QUALITY_TARGET", 0.775)
    rare_count = _float_env("DQA06_RARE_COUNT", 200.0)

    lows: list[float] = []
    highs: list[float] = []
    for count, quality, previous_count, previous_q in zip(
        latest_counts,
        latest_quality,
        previous_counts,
        previous_quality,
    ):
        quality_penalty = _clamp((quality_target - quality) * 0.80, -0.03, 0.06)
        quality_drop_penalty = _clamp((previous_q - quality) * 1.50, 0.0, 0.04)
        count_ratio = count / max(previous_count, 1.0)
        count_pressure = _clamp((math.log(max(count_ratio, 1e-6)) - math.log(1.10)) * 0.05, 0.0, 0.04)
        rare_relief = _clamp((rare_count - count) / rare_count, 0.0, 1.0) * _float_env("DQA06_RARE_RELIEF", 0.05)
        low = _clamp(
            base_low + ramp_raise + quality_penalty + quality_drop_penalty + count_pressure - rare_relief,
            min_low,
            max_low,
        )
        high = _clamp(low + base_high_gap + 0.05 * progress, min_high, max_high)
        lows.append(round(low, 4))
        highs.append(round(high, 4))

    nms = _clamp(
        _weighted_percentile(lows, latest_counts, 0.25) - 0.02,
        _float_env("DQA06_MIN_NMS", 0.35),
        _float_env("DQA06_MAX_NMS", 0.45),
    )
    teacher_loss = _clamp(0.35 - 0.10 * progress, 0.22, 0.35)
    box_loss = _clamp(0.02 - 0.005 * progress, 0.012, 0.02)
    obj_loss = _clamp(0.35 - 0.07 * progress, 0.25, 0.35)
    cls_loss = _clamp(0.10 - 0.04 * progress, 0.05, 0.10)

    return ThresholdDecision(
        enabled=True,
        reason="previous-client-pseudo-stats",
        phase=phase,
        round=round_idx,
        client_id=client_id,
        nms_conf_thres=round(nms, 4),
        ignore_thres_low=lows,
        ignore_thres_high=highs,
        teacher_loss_weight=round(teacher_loss, 4),
        box_loss_weight=round(box_loss, 5),
        obj_loss_weight=round(obj_loss, 4),
        cls_loss_weight=round(cls_loss, 4),
        source_stats=str(latest.get("_path")),
    )


def _log_decision(name: str, decision: ThresholdDecision) -> None:
    path = _threshold_log()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"run_name": name, **asdict(decision)}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _prepare_adaptive_scene_modules(work_root: Path):
    setup, fedsto = _prepare_scene_modules(work_root)
    exact_setup = setup.base
    original_config = getattr(
        exact_setup,
        "_dqa06_original_efficientteacher_config",
        exact_setup.efficientteacher_config,
    )
    exact_setup._dqa06_original_efficientteacher_config = original_config
    client_lr0 = _float_env("DQA06_CLIENT_LR0", 3e-4)
    server_lr0 = _float_env("DQA06_SERVER_LR0", 1e-3)

    def adaptive_config(**kwargs):
        if hasattr(setup, "_sync_base_paths"):
            setup._sync_base_paths()
        cfg = original_config(**kwargs)
        target = kwargs.get("target")
        name = str(kwargs.get("name", ""))

        if target is not None:
            cfg.setdefault("SSOD", {}).update(deepcopy(BASE_PROFILE))
            decision = decide_thresholds(name, int(cfg["Dataset"]["nc"]))
            cfg["SSOD"]["nms_conf_thres"] = decision.nms_conf_thres
            cfg["SSOD"]["ignore_thres_low"] = min(decision.ignore_thres_low)
            cfg["SSOD"]["ignore_thres_high"] = min(decision.ignore_thres_high)
            cfg["SSOD"]["teacher_loss_weight"] = decision.teacher_loss_weight
            cfg["SSOD"]["box_loss_weight"] = decision.box_loss_weight
            cfg["SSOD"]["obj_loss_weight"] = decision.obj_loss_weight
            cfg["SSOD"]["cls_loss_weight"] = decision.cls_loss_weight
            cfg.setdefault("hyp", {})["lr0"] = client_lr0
            cfg["hyp"]["lrf"] = 1.0
        elif name != "runtime_server_warmup":
            cfg.setdefault("hyp", {})["lr0"] = server_lr0
            cfg["hyp"]["lrf"] = 1.0

        return cfg

    setup.efficientteacher_config = adaptive_config
    exact_setup.efficientteacher_config = adaptive_config
    fedsto.setup.efficientteacher_config = adaptive_config
    return setup, fedsto


_ORIGINAL_RUN_TRAIN = base.run_train


def _adaptive_run_train(fedsto, config: Path, args, *, extra_env: dict[str, str] | None = None):
    env = dict(extra_env or {})
    if "client" in config.name:
        try:
            import yaml

            run_name = str(yaml.safe_load(config.read_text(encoding="utf-8"))["name"])
        except Exception:
            run_name = config.stem
        decision = decide_thresholds(run_name, _int_env("DQA06_NUM_CLASSES", 10))
        if decision.phase is not None and decision.phase >= 2 and decision.client_id is not None:
            env["DQA06_NMS_CONF_THRES"] = str(decision.nms_conf_thres)
            env["DQA06_IGNORE_THRES_LOW"] = json.dumps(decision.ignore_thres_low)
            env["DQA06_IGNORE_THRES_HIGH"] = json.dumps(decision.ignore_thres_high)
            _log_decision(run_name, decision)
    return _ORIGINAL_RUN_TRAIN(fedsto, config, args, extra_env=env or None)


base.AggregationConfig = v2.AggregationConfig
base.aggregate_fedavg_checkpoints = v2.aggregate_fedavg_checkpoints
base.aggregate_checkpoints = v2.aggregate_checkpoints
base.compute_reliability = v2.compute_reliability
base.load_round_stats = v2.load_round_stats
base.load_state = v2.load_state
base.save_state = v2.save_state
base.DQA_PROTOCOL_VERSION = DQA_PROTOCOL_VERSION
base.DEFAULT_DQA_WORK_ROOT = base.RESEARCH_ROOT / "efficientteacher_dqa06_scene_adaptive_pseudogt_8h"
base._prepare_fedsto_modules = _prepare_adaptive_scene_modules
base.run_train = _adaptive_run_train


def main() -> None:
    parsed = base.parse_args()
    os.environ["DQA06_STATS_ROOT"] = str(parsed.stats_root.resolve())
    os.environ["DQA06_PHASE2_ROUNDS"] = str(parsed.phase2_rounds)
    os.environ["DQA06_NUM_CLASSES"] = str(parsed.num_classes)
    os.environ.setdefault("DQA06_THRESHOLD_LOG", str((parsed.stats_root / "adaptive_threshold_schedule.jsonl").resolve()))
    if parsed.protocol_version == DQA_PROTOCOL_VERSION:
        parsed.protocol_version = DQA_PROTOCOL_VERSION
    if parsed.setup_only:
        base._prepare_fedsto_modules(parsed.workspace_root)[0].build_base_configs()
    else:
        print(f"DQA06 profile: {_profile_name()}")
        base.run_protocol(parsed)


if __name__ == "__main__":
    main()
