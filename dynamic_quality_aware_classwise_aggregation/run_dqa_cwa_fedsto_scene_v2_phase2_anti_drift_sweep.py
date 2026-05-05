#!/usr/bin/env python3
"""Run DQA08_3_3 phase2 anti-drift DQA sweeps.

This runner keeps the useful 08_3_2 setup, but focuses on the failure mode we
observed there: DQA variants improve in rounds 1-2, then drift down by round 10.

The extra controls here are deliberately about anti-drift:

- schedule pseudo-label loss weights and objectness softness across rounds
- schedule threshold/nms offsets so later rounds can become stricter
- schedule client/server LR and server anchoring without editing EfficientTeacher
"""

from __future__ import annotations

import copy
import json
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


DQA_PROTOCOL_VERSION = "dqa08_3_3_scene_phase2_anti_drift_sweep_v1"
DEFAULT_WORK_ROOT = base.RESEARCH_ROOT / "efficientteacher_dqa08_3_3_phase2_anti_drift_sweep"
DEFAULT_SOURCE_WORK_ROOT = base.RESEARCH_ROOT / "efficientteacher_dqa08_scene_tri_stage_policy_8h"
DEFAULT_SOURCE_PHASE1_ROUND = 3
ROUND_RE = re.compile(r"phase2_round(?P<round>\d+)")


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


def _profile() -> str:
    return os.environ.get("DQA0833_PROFILE", "dqa_high_light").strip().lower()


def _source_workspace() -> Path:
    return Path(os.environ.get("DQA0833_SOURCE_WORK_ROOT", str(DEFAULT_SOURCE_WORK_ROOT))).resolve()


def _source_phase1_round() -> int:
    return _int_env("DQA0833_SOURCE_PHASE1_ROUND", DEFAULT_SOURCE_PHASE1_ROUND)


def _round_from_path(path: Path) -> int | None:
    match = ROUND_RE.search(str(path))
    return int(match.group("round")) if match else None


def _scheduled(prefix: str, round_idx: int | None, default_start: float, default_end: float | None = None) -> float:
    start = _float_env(f"{prefix}_START", default_start)
    end = _float_env(f"{prefix}_END", start if default_end is None else default_end)
    total = max(_int_env("DQA0833_PHASE2_ROUNDS", 10), 1)
    if round_idx is None or total <= 1:
        return start
    progress = min(max((round_idx - 1) / (total - 1), 0.0), 1.0)
    return start + (end - start) * progress


def _scheduled_value(name: str, round_idx: int | None, default: float) -> float:
    base_value = _float_env(name, default)
    return _scheduled(name, round_idx, base_value, base_value)


def _env_has_schedule(name: str) -> bool:
    return any(key in os.environ for key in (name, f"{name}_START", f"{name}_END"))


def _round_from_run_name(name: str) -> int | None:
    parsed = tri_stage._parse_run_name(name)
    return int(parsed["round"]) if parsed and parsed.get("round") is not None else None


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))


def _seed_metadata_path(workspace_root: Path) -> Path:
    return workspace_root / "phase2_seed.json"


def _seed_phase1_checkpoint(workspace_root: Path) -> Path:
    workspace_root = workspace_root.resolve()
    source_root = _source_workspace()
    source_round = _source_phase1_round()
    source = source_root / "global_checkpoints" / f"phase1_round{source_round:03d}_global.pt"
    target = workspace_root / "global_checkpoints" / "round000_warmup.pt"
    metadata_path = _seed_metadata_path(workspace_root)
    force = _bool_env("DQA0833_FORCE_SEED", False)

    if not source.exists():
        raise FileNotFoundError(
            f"Missing source phase1 seed checkpoint: {source}\n"
            "Set DQA0833_SOURCE_WORK_ROOT / DQA0833_SOURCE_PHASE1_ROUND for a different seed."
        )

    wanted: dict[str, Any] = {
        "protocol": DQA_PROTOCOL_VERSION,
        "seed_kind": "dqa08_phase1_global_as_round000_warmup",
        "source_work_root": str(source_root),
        "source_phase1_round": source_round,
        "source_checkpoint": str(source),
        "target_checkpoint": str(target.resolve()),
        "variant": os.environ.get("DQA0833_VARIANT", "unnamed"),
        "profile": _profile(),
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
                and existing.get("profile") == wanted["profile"]
            )
            if same_seed:
                print(f"Reusing 08_3_3 phase2 seed checkpoint: {target}")
                return target
        raise RuntimeError(
            f"Seed checkpoint already exists with different metadata: {target}\n"
            f"Existing metadata: {metadata_path if metadata_path.exists() else '(missing)'}\n"
            "Use DQA0833_FORCE_SEED=1 only if you intentionally want to overwrite it."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    metadata_path.write_text(json.dumps(wanted, indent=2), encoding="utf-8")
    print(f"Seeded 08_3_3 phase2 from DQA08 phase1 round {source_round}: {target}")
    return target


def _scheduled_override(cfg: dict[str, Any], key: str, env_name: str, round_idx: int | None) -> None:
    if _env_has_schedule(env_name):
        cfg[key] = _scheduled_value(env_name, round_idx, float(cfg[key]))


def _adjust_decision_for_anti_drift(decision: tri_stage.ThresholdDecision) -> tri_stage.ThresholdDecision:
    if decision.phase is None or decision.phase < 2 or decision.round is None:
        return decision

    low_delta = _scheduled_value("DQA0833_LOW_DELTA", decision.round, 0.0)
    mid_delta = _scheduled_value("DQA0833_MID_DELTA", decision.round, 0.0)
    high_delta = _scheduled_value("DQA0833_HIGH_DELTA", decision.round, 0.0)
    nms_delta = _scheduled_value("DQA0833_NMS_DELTA", decision.round, 0.0)
    min_mid_gap = _float_env("DQA0833_SCHEDULE_MIN_MID_GAP", 0.08)
    min_high_gap = _float_env("DQA0833_SCHEDULE_MIN_HIGH_GAP", 0.12)

    lows = [_clamp(x + low_delta, 0.01, 0.98) for x in decision.ignore_thres_low]
    mids = [_clamp(x + mid_delta, 0.01, 0.985) for x in decision.ignore_thres_mid]
    highs = [_clamp(x + high_delta, 0.01, 0.995) for x in decision.ignore_thres_high]
    adjusted_low: list[float] = []
    adjusted_mid: list[float] = []
    adjusted_high: list[float] = []
    for low, mid, high in zip(lows, mids, highs):
        mid = _clamp(max(mid, low + min_mid_gap), 0.01, 0.985)
        high = _clamp(max(high, mid + min_high_gap), 0.01, 0.995)
        adjusted_low.append(round(low, 4))
        adjusted_mid.append(round(mid, 4))
        adjusted_high.append(round(high, 4))

    teacher = _scheduled_value("DQA0833_TEACHER_LOSS_WEIGHT", decision.round, decision.teacher_loss_weight)
    box = _scheduled_value("DQA0833_BOX_LOSS_WEIGHT", decision.round, decision.box_loss_weight)
    obj = _scheduled_value("DQA0833_OBJ_LOSS_WEIGHT", decision.round, decision.obj_loss_weight)
    cls = _scheduled_value("DQA0833_CLS_LOSS_WEIGHT", decision.round, decision.cls_loss_weight)

    return replace(
        decision,
        reason=f"{decision.reason}+anti-drift-scheduled",
        nms_conf_thres=round(_clamp(decision.nms_conf_thres + nms_delta, 0.001, 0.99), 4),
        ignore_thres_low=adjusted_low,
        ignore_thres_mid=adjusted_mid,
        ignore_thres_high=adjusted_high,
        teacher_loss_weight=round(max(teacher, 0.0), 4),
        box_loss_weight=round(max(box, 0.0), 5),
        obj_loss_weight=round(max(obj, 0.0), 4),
        cls_loss_weight=round(max(cls, 0.0), 4),
    )


def _apply_dqa_ssod_profile(cfg: dict[str, Any], *, name: str, target: Path | None) -> None:
    if target is None:
        return

    ssod = cfg.setdefault("SSOD", {})
    ssod.update(copy.deepcopy(tri_stage.BASE_PROFILE))
    decision = _adjust_decision_for_anti_drift(tri_stage.decide_thresholds(name, int(cfg["Dataset"]["nc"])))
    round_idx = decision.round

    ssod["nms_conf_thres"] = decision.nms_conf_thres
    ssod["ignore_thres_low"] = min(decision.ignore_thres_low)
    ssod["ignore_thres_high"] = min(decision.ignore_thres_high)
    ssod["teacher_loss_weight"] = decision.teacher_loss_weight
    ssod["box_loss_weight"] = decision.box_loss_weight
    ssod["obj_loss_weight"] = decision.obj_loss_weight
    ssod["cls_loss_weight"] = decision.cls_loss_weight
    ssod["ignore_obj"] = _bool_env("DQA0833_IGNORE_OBJ", False)
    ssod["pseudo_label_with_obj"] = True
    ssod["pseudo_label_with_bbox"] = _bool_env("DQA0833_PSEUDO_LABEL_WITH_BBOX", True)
    ssod["pseudo_label_with_cls"] = _bool_env("DQA0833_PSEUDO_LABEL_WITH_CLS", True)

    _scheduled_override(ssod, "teacher_loss_weight", "DQA0833_TEACHER_LOSS_WEIGHT", round_idx)
    _scheduled_override(ssod, "box_loss_weight", "DQA0833_BOX_LOSS_WEIGHT", round_idx)
    _scheduled_override(ssod, "obj_loss_weight", "DQA0833_OBJ_LOSS_WEIGHT", round_idx)
    _scheduled_override(ssod, "cls_loss_weight", "DQA0833_CLS_LOSS_WEIGHT", round_idx)

def _prepare_phase2_sweep_modules(work_root: Path):
    setup, fedsto = _prepare_scene_modules(work_root)
    exact_setup = setup.base
    original_config = getattr(
        exact_setup,
        "_dqa0833_original_efficientteacher_config",
        exact_setup.efficientteacher_config,
    )
    exact_setup._dqa0833_original_efficientteacher_config = original_config

    def sweep_config(**kwargs):
        if hasattr(setup, "_sync_base_paths"):
            setup._sync_base_paths()
        cfg = original_config(**kwargs)
        target = kwargs.get("target")
        name = str(kwargs.get("name", ""))
        profile = _profile()
        round_idx = _round_from_run_name(name)

        if target is not None:
            cfg.setdefault("FedSTO", {})["train_scope"] = os.environ.get("DQA0833_CLIENT_TRAIN_SCOPE", "all")
            cfg["FedSTO"]["orthogonal_weight"] = _float_env("DQA0833_CLIENT_ORTHOGONAL_WEIGHT", 1e-4)
            cfg.setdefault("hyp", {})["lr0"] = _scheduled_value(
                "DQA0833_CLIENT_LR0",
                round_idx,
                cfg.get("hyp", {}).get("lr0", 0.01),
            )
            cfg["hyp"]["lrf"] = 1.0
            if profile != "fedsto_exact":
                _apply_dqa_ssod_profile(cfg, name=name, target=target)
        elif name != "runtime_server_warmup":
            cfg.setdefault("FedSTO", {})["train_scope"] = os.environ.get("DQA0833_SERVER_TRAIN_SCOPE", "all")
            cfg["FedSTO"]["orthogonal_weight"] = _float_env("DQA0833_SERVER_ORTHOGONAL_WEIGHT", 1e-4)
            cfg.setdefault("hyp", {})["lr0"] = _scheduled_value(
                "DQA0833_SERVER_LR0",
                round_idx,
                cfg.get("hyp", {}).get("lr0", 0.01),
            )
            cfg["hyp"]["lrf"] = 1.0

        return cfg

    setup.efficientteacher_config = sweep_config
    exact_setup.efficientteacher_config = sweep_config
    fedsto.setup.efficientteacher_config = sweep_config
    return setup, fedsto


_ORIGINAL_RUN_TRAIN = base.run_train


def _sweep_run_train(fedsto, config: Path, args, *, extra_env: dict[str, str] | None = None):
    if _profile() == "fedsto_exact" or "client" not in config.name:
        return _ORIGINAL_RUN_TRAIN(fedsto, config, args, extra_env=extra_env)

    env = dict(extra_env or {})
    try:
        import yaml

        run_name = str(yaml.safe_load(config.read_text(encoding="utf-8"))["name"])
    except Exception:
        run_name = config.stem

    decision = _adjust_decision_for_anti_drift(
        tri_stage.decide_thresholds(run_name, _int_env("DQA08_NUM_CLASSES", 10))
    )
    if decision.phase is not None and decision.phase >= 2 and decision.client_id is not None:
        env["DQA06_NMS_CONF_THRES"] = str(decision.nms_conf_thres)
        env["DQA06_IGNORE_THRES_LOW"] = json.dumps(decision.ignore_thres_low)
        env["DQA06_IGNORE_THRES_HIGH"] = json.dumps(decision.ignore_thres_high)
        env["DQA08_TRI_STAGE_GATE"] = "1" if _bool_env("DQA0833_TRI_STAGE_GATE", True) else "0"
        env["DQA08_IGNORE_THRES_MID"] = json.dumps(decision.ignore_thres_mid)
        env["DQA08_LOW_MID_OBJ_WEIGHT"] = str(
            _scheduled_value("DQA0833_LOW_MID_OBJ_WEIGHT", decision.round, 0.35)
        )
        env["DQA08_MID_HIGH_OBJ_WEIGHT"] = str(
            _scheduled_value("DQA0833_MID_HIGH_OBJ_WEIGHT", decision.round, 0.90)
        )
        tri_stage._log_decision(run_name, decision)

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
        min_server_alpha=_float_env("DQA0833_MIN_SERVER_ALPHA", 0.65),
        residual_blend=_float_env("DQA0833_RESIDUAL_START", 0.16),
    )


def _aggregate_phase2_sweep(
    client_checkpoints,
    server_checkpoint,
    output_checkpoint,
    stats,
    state_path,
    config,
    repo_root,
):
    round_idx = _round_from_path(output_checkpoint)
    residual = _scheduled("DQA0833_RESIDUAL", round_idx, 0.16, 0.08)
    min_server_alpha = _scheduled(
        "DQA0833_MIN_SERVER_ALPHA",
        round_idx,
        _float_env("DQA0833_MIN_SERVER_ALPHA_START", _float_env("DQA0833_MIN_SERVER_ALPHA", 0.65)),
        _float_env("DQA0833_MIN_SERVER_ALPHA_END", _float_env("DQA0833_MIN_SERVER_ALPHA", 0.65)),
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
    state.setdefault("phase2_anti_drift_sweep", []).append(
        {
            "round": round_idx,
            "variant": os.environ.get("DQA0833_VARIANT", "unnamed"),
            "profile": _profile(),
            "residual_blend": residual,
            "min_server_alpha": cfg.min_server_alpha,
            "classwise_blend": cfg.classwise_blend,
            "server_anchor": cfg.server_anchor,
        }
    )
    v2.save_state(state_path, state)
    print(
        "DQA08_3_3 phase2 gate:",
        f"variant={os.environ.get('DQA0833_VARIANT', 'unnamed')}",
        f"profile={_profile()}",
        f"round={round_idx}",
        f"residual_blend={residual:.4f}",
        f"min_server_alpha={cfg.min_server_alpha:.3f}",
        f"classwise_blend={cfg.classwise_blend:.3f}",
        f"server_anchor={cfg.server_anchor:.3f}",
    )
    return output, state


base.DQA_PROTOCOL_VERSION = DQA_PROTOCOL_VERSION
base.DEFAULT_DQA_WORK_ROOT = DEFAULT_WORK_ROOT
base._prepare_fedsto_modules = _prepare_phase2_sweep_modules
base.run_train = _sweep_run_train
base.AggregationConfig = v2.AggregationConfig
base.aggregate_checkpoints = _aggregate_phase2_sweep
base._dqa_config = _dqa_config


def main() -> None:
    parsed = base.parse_args()
    if parsed.protocol_version == tri_stage.DQA_PROTOCOL_VERSION:
        parsed.protocol_version = DQA_PROTOCOL_VERSION
    if parsed.workspace_root == tri_stage.base.RESEARCH_ROOT / "efficientteacher_dqa08_scene_tri_stage_policy_8h":
        parsed.workspace_root = DEFAULT_WORK_ROOT

    os.environ["DQA08_STATS_ROOT"] = str(parsed.stats_root.resolve())
    os.environ["DQA08_PHASE2_ROUNDS"] = str(parsed.phase2_rounds)
    os.environ["DQA0833_PHASE2_ROUNDS"] = str(parsed.phase2_rounds)
    os.environ["DQA08_NUM_CLASSES"] = str(parsed.num_classes)
    os.environ.setdefault("DQA08_POLICY_HORIZON_ROUNDS", str(parsed.phase2_rounds))
    os.environ.setdefault(
        "DQA08_THRESHOLD_LOG",
        str((parsed.stats_root / "phase2_anti_drift_policy_schedule.jsonl").resolve()),
    )

    if parsed.setup_only:
        base._prepare_fedsto_modules(parsed.workspace_root)[0].build_base_configs()
        return

    if parsed.phase1_rounds != 0:
        raise ValueError("08_3_3 is phase2-only. Run with --phase1-rounds 0.")

    if not parsed.dry_run:
        _seed_phase1_checkpoint(parsed.workspace_root)
        if _profile() != "fedsto_exact" and not tri_stage._policy_model_path().exists():
            raise FileNotFoundError(
                f"Missing learned threshold policy model: {tri_stage._policy_model_path()}. "
                "Run threshold_policy_model/01_train_dqa05_threshold_policy.ipynb first."
            )

    print("DQA08_3_3 profile: phase2-only FedSTO/DQA sweep")
    print(f"DQA08_3_3 variant: {os.environ.get('DQA0833_VARIANT', 'unnamed')}")
    print(f"DQA08_3_3 profile mode: {_profile()}")
    print(f"DQA08_3_3 seed: {_source_workspace()} phase1 round {_source_phase1_round()}")
    print(f"DQA08_3_3 client train_scope: {os.environ.get('DQA0833_CLIENT_TRAIN_SCOPE', 'all')}")
    print(
        "DQA08_3_3 residual schedule:",
        _float_env("DQA0833_RESIDUAL_START", 0.16),
        "->",
        _float_env("DQA0833_RESIDUAL_END", 0.08),
    )
    print(
        "DQA08_3_3 pseudo loss schedule:",
        _float_env("DQA0833_TEACHER_LOSS_WEIGHT_START", _float_env("DQA0833_TEACHER_LOSS_WEIGHT", 0.34)),
        "->",
        _float_env("DQA0833_TEACHER_LOSS_WEIGHT_END", _float_env("DQA0833_TEACHER_LOSS_WEIGHT", 0.34)),
        "/ box",
        _float_env("DQA0833_BOX_LOSS_WEIGHT_START", _float_env("DQA0833_BOX_LOSS_WEIGHT", 0.02)),
        "->",
        _float_env("DQA0833_BOX_LOSS_WEIGHT_END", _float_env("DQA0833_BOX_LOSS_WEIGHT", 0.02)),
    )
    print(f"DQA08_3_3 dqa_start_phase: {parsed.dqa_start_phase}")
    base.run_protocol(parsed)


if __name__ == "__main__":
    main()
