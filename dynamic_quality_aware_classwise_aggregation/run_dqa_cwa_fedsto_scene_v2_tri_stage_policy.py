#!/usr/bin/env python3
"""Run scene DQA-CWA with a tri-stage learned adaptive pseudoGT policy.

This wrapper is for notebook 08.  It keeps the 06 scene/class schedule, but
replaces the hand-written pseudoGT gate rule with the lightweight policy trained
from DQA05 logs.  The learned proposal is intentionally guarded:

- thresholds are clipped to a conservative range near the successful 05 fixed
  setting
- per-round movement is limited using the previous logged decision
- rare classes receive an extra ceiling so recall is less likely to collapse

The EfficientTeacher patch reads DQA06_* for low/high compatibility and
DQA08_* for the new mid gate/objectness weights.  This keeps 07 runs
unchanged while enabling the tri-stage pseudoGT gate only for notebook 08.
"""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import dqa_cwa_aggregation_v2 as v2
import run_dqa_cwa_fedsto as base
from run_dqa_cwa_fedsto_scene import _prepare_scene_modules
from threshold_policy_model.threshold_policy import (
    CLASS_NAMES,
    load_bundle,
    predict_policy,
    read_client_stats,
    weighted_percentile,
)


DQA_PROTOCOL_VERSION = "dqa08_scene_tri_stage_pseudogt_policy_v1"
RUN_RE = re.compile(r"dqa_phase(?P<phase>\d+)_round(?P<round>\d+)_(?P<role>client(?P<client>\d+)|server)")

BASE_PROFILE: dict[str, Any] = {
    "nms_conf_thres": 0.36,
    "ignore_thres_low": 0.38,
    "ignore_thres_high": 0.84,
    "teacher_loss_weight": 0.34,
    "box_loss_weight": 0.02,
    "obj_loss_weight": 0.32,
    "cls_loss_weight": 0.10,
    "pseudo_label_with_bbox": False,
    "pseudo_label_with_cls": False,
}

TRI_STAGE_DEFAULTS: dict[str, float] = {
    "ignore_thres_mid": 0.60,
    "low_mid_obj_weight": 0.45,
    "mid_high_obj_weight": 1.0,
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
    ignore_thres_mid: list[float]
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


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))


def _profile_name() -> str:
    return os.environ.get("DQA08_SSOD_PROFILE", "tri_stage_policy")


def _stats_root() -> Path | None:
    raw = os.environ.get("DQA08_STATS_ROOT")
    return Path(raw).resolve() if raw else None


def _threshold_log() -> Path | None:
    raw = os.environ.get("DQA08_THRESHOLD_LOG")
    if raw:
        return Path(raw).resolve()
    stats_root = _stats_root()
    return stats_root / "tri_stage_policy_schedule.jsonl" if stats_root else None


def _policy_model_path() -> Path:
    raw = os.environ.get("DQA08_POLICY_MODEL")
    if raw:
        return Path(raw).resolve()
    return (
        base.RESEARCH_ROOT
        / "threshold_policy_model"
        / "artifacts"
        / "dqa05_threshold_policy.joblib"
    ).resolve()


@lru_cache(maxsize=1)
def _policy_bundle() -> dict[str, Any]:
    return load_bundle(_policy_model_path())


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


def _base_decision(phase: int | None, round_idx: int | None, client_id: str | None, reason: str) -> ThresholdDecision:
    nc = _int_env("DQA08_NUM_CLASSES", 10)
    low = float(BASE_PROFILE["ignore_thres_low"])
    mid = float(TRI_STAGE_DEFAULTS["ignore_thres_mid"])
    high = float(BASE_PROFILE["ignore_thres_high"])
    return ThresholdDecision(
        enabled=False,
        reason=reason,
        phase=phase,
        round=round_idx,
        client_id=client_id,
        nms_conf_thres=float(BASE_PROFILE["nms_conf_thres"]),
        ignore_thres_low=[low] * nc,
        ignore_thres_mid=[mid] * nc,
        ignore_thres_high=[high] * nc,
        teacher_loss_weight=float(BASE_PROFILE["teacher_loss_weight"]),
        box_loss_weight=float(BASE_PROFILE["box_loss_weight"]),
        obj_loss_weight=float(BASE_PROFILE["obj_loss_weight"]),
        cls_loss_weight=float(BASE_PROFILE["cls_loss_weight"]),
    )


def _previous_logged_decision(phase: int, round_idx: int, client_id: str) -> dict[str, Any] | None:
    path = _threshold_log()
    if path is None or not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            int(record.get("phase", -1)) == phase
            and int(record.get("round", 10**9)) < round_idx
            and str(record.get("client_id")) == str(client_id)
        ):
            return record
    return None


def _step_toward(previous: float, proposed: float, limit: float) -> float:
    if limit <= 0:
        return proposed
    delta = _clamp(proposed - previous, -limit, limit)
    return previous + delta


def _smooth_decision(decision: ThresholdDecision) -> ThresholdDecision:
    if not _bool_env("DQA08_ENABLE_SMOOTHING", True):
        return decision
    if decision.phase is None or decision.round is None or decision.client_id is None:
        return decision

    previous = _previous_logged_decision(decision.phase, decision.round, decision.client_id)
    if previous is None:
        return decision

    low_step = _float_env("DQA08_LOW_STEP_LIMIT", 0.02)
    mid_step = _float_env("DQA08_MID_STEP_LIMIT", 0.025)
    high_step = _float_env("DQA08_HIGH_STEP_LIMIT", 0.035)
    nms_step = _float_env("DQA08_NMS_STEP_LIMIT", 0.02)
    loss_step = _float_env("DQA08_LOSS_STEP_LIMIT", 0.015)

    prev_lows = previous.get("ignore_thres_low") or decision.ignore_thres_low
    prev_mids = previous.get("ignore_thres_mid") or decision.ignore_thres_mid
    prev_highs = previous.get("ignore_thres_high") or decision.ignore_thres_high
    lows = [
        round(_step_toward(float(prev), float(prop), low_step), 4)
        for prev, prop in zip(prev_lows, decision.ignore_thres_low)
    ]
    mids = [
        round(_step_toward(float(prev), float(prop), mid_step), 4)
        for prev, prop in zip(prev_mids, decision.ignore_thres_mid)
    ]
    highs = [
        round(_step_toward(float(prev), float(prop), high_step), 4)
        for prev, prop in zip(prev_highs, decision.ignore_thres_high)
    ]
    nms = round(_step_toward(float(previous.get("nms_conf_thres", decision.nms_conf_thres)), decision.nms_conf_thres, nms_step), 4)
    teacher = round(
        _step_toward(
            float(previous.get("teacher_loss_weight", decision.teacher_loss_weight)),
            decision.teacher_loss_weight,
            loss_step,
        ),
        4,
    )
    box = round(
        _step_toward(float(previous.get("box_loss_weight", decision.box_loss_weight)), decision.box_loss_weight, loss_step),
        5,
    )
    obj = round(
        _step_toward(float(previous.get("obj_loss_weight", decision.obj_loss_weight)), decision.obj_loss_weight, loss_step),
        4,
    )
    cls = round(
        _step_toward(float(previous.get("cls_loss_weight", decision.cls_loss_weight)), decision.cls_loss_weight, loss_step),
        4,
    )
    return ThresholdDecision(
        enabled=decision.enabled,
        reason=f"{decision.reason}+smoothed",
        phase=decision.phase,
        round=decision.round,
        client_id=decision.client_id,
        nms_conf_thres=nms,
        ignore_thres_low=lows,
        ignore_thres_mid=mids,
        ignore_thres_high=highs,
        teacher_loss_weight=teacher,
        box_loss_weight=box,
        obj_loss_weight=obj,
        cls_loss_weight=cls,
        source_stats=decision.source_stats,
    )


def _latest_policy_rows(stats_root: Path, source_round: int, client_id: str):
    bundle = _policy_bundle()
    class_names = bundle.get("class_names") or CLASS_NAMES
    data = read_client_stats(stats_root, class_names=class_names)
    horizon = max(_int_env("DQA08_POLICY_HORIZON_ROUNDS", _int_env("DQA08_PHASE2_ROUNDS", 24)), source_round)
    data["round_norm"] = (data["round"] - 1) / max(horizon - 1, 1)
    data["apply_round_norm"] = (data["apply_round"] - 1) / max(horizon, 1)
    predictions = predict_policy(bundle, data)
    rows = predictions[(predictions["round"] == source_round) & (predictions["client_id"] == int(client_id))]
    return rows.sort_values("class_index")


def _decision_from_policy(
    *,
    phase: int,
    round_idx: int,
    client_id: str,
    rows,
    num_classes: int,
    source_stats: Path,
) -> ThresholdDecision:
    min_low = _float_env("DQA08_MIN_LOW", 0.38)
    max_low = _float_env("DQA08_MAX_LOW", 0.48)
    min_mid = _float_env("DQA08_MIN_MID", 0.58)
    max_mid = _float_env("DQA08_MAX_MID", 0.72)
    min_high = _float_env("DQA08_MIN_HIGH", 0.84)
    max_high = _float_env("DQA08_MAX_HIGH", 0.92)
    mid_gap = _float_env("DQA08_MID_GAP", 0.20)
    high_mid_gap = _float_env("DQA08_HIGH_MID_GAP", 0.18)
    min_high_gap = _float_env("DQA08_MIN_HIGH_GAP", 0.16)
    min_nms = _float_env("DQA08_MIN_NMS", 0.35)
    max_nms = _float_env("DQA08_MAX_NMS", 0.45)
    low_shift = _float_env("DQA08_LOW_SHIFT", 0.02)
    high_shift = _float_env("DQA08_HIGH_SHIFT", 0.05)
    rare_count = _float_env("DQA08_RARE_COUNT", 250.0)
    rare_max_low = _float_env("DQA08_RARE_MAX_LOW", 0.43)
    rare_max_mid = _float_env("DQA08_RARE_MAX_MID", 0.66)
    rare_max_high = _float_env("DQA08_RARE_MAX_HIGH", 0.89)

    lows = [float(BASE_PROFILE["ignore_thres_low"])] * num_classes
    mids = [float(TRI_STAGE_DEFAULTS["ignore_thres_mid"])] * num_classes
    highs = [float(BASE_PROFILE["ignore_thres_high"])] * num_classes
    counts = [1.0] * num_classes

    for row in rows.to_dict("records"):
        class_idx = int(row["class_index"])
        if class_idx < 0 or class_idx >= num_classes:
            continue
        count = float(row["count"])
        low = _clamp(float(row["pred_low"]) + low_shift, min_low, max_low)
        proposed_high = _clamp(float(row["pred_high"]) + high_shift, min_high, max_high)
        mid = _clamp(max(low + mid_gap, proposed_high - high_mid_gap), min_mid, max_mid)
        high = _clamp(max(proposed_high, mid + min_high_gap), min_high, max_high)
        if count < rare_count:
            low = min(low, rare_max_low)
            mid = min(mid, rare_max_mid)
            high = min(high, rare_max_high)
        mid = _clamp(max(mid, low + 0.08), min_mid, max_mid)
        high = _clamp(max(high, mid + min_high_gap), min_high, max_high)
        lows[class_idx] = round(low, 4)
        mids[class_idx] = round(mid, 4)
        highs[class_idx] = round(high, 4)
        counts[class_idx] = max(count, 1.0)

    nms_values = [min(max(low - 0.02, min_nms), max_nms) for low in lows]
    nms = _clamp(weighted_percentile(nms_values, counts, 0.25), min_nms, max_nms)

    mean_low = sum(lows) / max(len(lows), 1)
    risk = _clamp((mean_low - min_low) / max(max_low - min_low, 1e-6), 0.0, 1.0)
    teacher = _clamp(
        0.34 - _float_env("DQA08_TEACHER_DROP", 0.04) * risk,
        _float_env("DQA08_TEACHER_MIN", 0.28),
        0.34,
    )
    box = _clamp(0.02 - 0.002 * risk, 0.017, 0.02)
    obj = _clamp(0.32 - 0.03 * risk, 0.28, 0.32)
    cls = _clamp(0.10 - 0.02 * risk, 0.075, 0.10)

    decision = ThresholdDecision(
        enabled=True,
        reason="learned-policy-previous-client-stats",
        phase=phase,
        round=round_idx,
        client_id=client_id,
        nms_conf_thres=round(nms, 4),
        ignore_thres_low=lows,
        ignore_thres_mid=mids,
        ignore_thres_high=highs,
        teacher_loss_weight=round(teacher, 4),
        box_loss_weight=round(box, 5),
        obj_loss_weight=round(obj, 4),
        cls_loss_weight=round(cls, 4),
        source_stats=str(source_stats),
    )
    return _smooth_decision(decision)


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
    start_round = _int_env("DQA08_ADAPT_START_ROUND", 3)
    stats_root = _stats_root()
    if stats_root is None:
        return _base_decision(phase, round_idx, client_id, "missing-stats-root")
    if round_idx < start_round:
        return _base_decision(phase, round_idx, client_id, "warm-adaptation-round")

    source_round = round_idx - 1
    source_stats = stats_root / f"phase{phase}_round{source_round:03d}_client{client_id}.json"
    if source_round < 1 or not source_stats.exists():
        return _base_decision(phase, round_idx, client_id, "missing-previous-client-stats")

    rows = _latest_policy_rows(stats_root, source_round, client_id)
    if rows.empty:
        return _base_decision(phase, round_idx, client_id, "missing-learned-policy-rows")
    return _decision_from_policy(
        phase=phase,
        round_idx=round_idx,
        client_id=client_id,
        rows=rows,
        num_classes=num_classes,
        source_stats=source_stats,
    )


def _log_decision(name: str, decision: ThresholdDecision) -> None:
    path = _threshold_log()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"run_name": name, "policy_model": str(_policy_model_path()), **asdict(decision)}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _prepare_tri_stage_scene_modules(work_root: Path):
    setup, fedsto = _prepare_scene_modules(work_root)
    exact_setup = setup.base
    original_config = getattr(
        exact_setup,
        "_dqa08_original_efficientteacher_config",
        exact_setup.efficientteacher_config,
    )
    exact_setup._dqa08_original_efficientteacher_config = original_config
    client_lr0 = _float_env("DQA08_CLIENT_LR0", 3e-4)
    server_lr0 = _float_env("DQA08_SERVER_LR0", 1e-3)

    def tri_stage_config(**kwargs):
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

    setup.efficientteacher_config = tri_stage_config
    exact_setup.efficientteacher_config = tri_stage_config
    fedsto.setup.efficientteacher_config = tri_stage_config
    return setup, fedsto


_ORIGINAL_RUN_TRAIN = base.run_train


def _tri_stage_run_train(fedsto, config: Path, args, *, extra_env: dict[str, str] | None = None):
    env = dict(extra_env or {})
    if "client" in config.name:
        try:
            import yaml

            run_name = str(yaml.safe_load(config.read_text(encoding="utf-8"))["name"])
        except Exception:
            run_name = config.stem
        decision = decide_thresholds(run_name, _int_env("DQA08_NUM_CLASSES", 10))
        if decision.phase is not None and decision.phase >= 2 and decision.client_id is not None:
            env["DQA06_NMS_CONF_THRES"] = str(decision.nms_conf_thres)
            env["DQA06_IGNORE_THRES_LOW"] = json.dumps(decision.ignore_thres_low)
            env["DQA06_IGNORE_THRES_HIGH"] = json.dumps(decision.ignore_thres_high)
            env["DQA08_TRI_STAGE_GATE"] = "1"
            env["DQA08_IGNORE_THRES_MID"] = json.dumps(decision.ignore_thres_mid)
            env["DQA08_LOW_MID_OBJ_WEIGHT"] = str(
                _float_env("DQA08_LOW_MID_OBJ_WEIGHT", float(TRI_STAGE_DEFAULTS["low_mid_obj_weight"]))
            )
            env["DQA08_MID_HIGH_OBJ_WEIGHT"] = str(
                _float_env("DQA08_MID_HIGH_OBJ_WEIGHT", float(TRI_STAGE_DEFAULTS["mid_high_obj_weight"]))
            )
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
base.DEFAULT_DQA_WORK_ROOT = base.RESEARCH_ROOT / "efficientteacher_dqa08_scene_tri_stage_policy_8h"
base._prepare_fedsto_modules = _prepare_tri_stage_scene_modules
base.run_train = _tri_stage_run_train


def main() -> None:
    parsed = base.parse_args()
    os.environ["DQA08_STATS_ROOT"] = str(parsed.stats_root.resolve())
    os.environ["DQA08_PHASE2_ROUNDS"] = str(parsed.phase2_rounds)
    os.environ["DQA08_NUM_CLASSES"] = str(parsed.num_classes)
    os.environ.setdefault("DQA08_POLICY_HORIZON_ROUNDS", str(parsed.phase2_rounds))
    os.environ.setdefault(
        "DQA08_THRESHOLD_LOG",
        str((parsed.stats_root / "tri_stage_policy_schedule.jsonl").resolve()),
    )

    if not parsed.setup_only and not parsed.dry_run and not _policy_model_path().exists():
        raise FileNotFoundError(
            f"Missing learned threshold policy model: {_policy_model_path()}. "
            "Run threshold_policy_model/01_train_dqa05_threshold_policy.ipynb first."
        )

    if parsed.protocol_version == DQA_PROTOCOL_VERSION:
        parsed.protocol_version = DQA_PROTOCOL_VERSION
    if parsed.setup_only:
        base._prepare_fedsto_modules(parsed.workspace_root)[0].build_base_configs()
    else:
        print(f"DQA08 profile: {_profile_name()}")
        print(f"DQA08 policy model: {_policy_model_path()}")
        base.run_protocol(parsed)


if __name__ == "__main__":
    main()
