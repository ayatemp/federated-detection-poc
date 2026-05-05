#!/usr/bin/env python3
"""Run DQA08_3_5 phase2 with temporal pseudo-label memory.

This experiment keeps the successful 05 conservative SSOD profile, but changes
how phase2 pseudo labels become trusted. New pseudo boxes are capped below the
high threshold, so they only provide weak objectness. A pseudo box is promoted to
full bbox/cls training only when it reappears for the same target image and
class with enough IoU across rounds. The teacher is still the reproducible
FedSTO local EMA teacher carried by each client checkpoint.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import dqa_cwa_aggregation_v2 as v2
import run_dqa_cwa_fedsto as base
from run_dqa_cwa_fedsto_scene import _prepare_scene_modules


DQA_PROTOCOL_VERSION = "dqa08_3_5_scene_phase2_temporal_memory_v1"
DEFAULT_WORK_ROOT = base.RESEARCH_ROOT / "efficientteacher_dqa08_3_5_phase2_temporal_memory"
DEFAULT_SOURCE_WORK_ROOT = base.RESEARCH_ROOT / "efficientteacher_dqa05_scene_class_profile_5h"
FALLBACK_SOURCE_WORK_ROOT = base.RESEARCH_ROOT / "efficientteacher_dqa08_scene_tri_stage_policy_8h"
DEFAULT_SOURCE_PHASE1_ROUND = 8
RUN_RE = re.compile(r"dqa_phase(?P<phase>\d+)_round(?P<round>\d+)_(?P<role>client(?P<client>\d+)|server)")
ROUND_RE = re.compile(r"phase2_round(?P<round>\d+)")

SSOD_PROFILE: dict[str, Any] = {
    "nms_conf_thres": 0.35,
    "ignore_thres_low": 0.35,
    "ignore_thres_high": 0.75,
    "teacher_loss_weight": 0.35,
    "box_loss_weight": 0.02,
    "obj_loss_weight": 0.35,
    "cls_loss_weight": 0.10,
    "pseudo_label_with_obj": True,
    "pseudo_label_with_bbox": False,
    "pseudo_label_with_cls": False,
}


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
    raw = os.environ.get("DQA0835_SOURCE_WORK_ROOT")
    if raw:
        return Path(raw).resolve()
    source_round = _source_phase1_round()
    preferred = DEFAULT_SOURCE_WORK_ROOT.resolve()
    if (preferred / "global_checkpoints" / f"phase1_round{source_round:03d}_global.pt").exists():
        return preferred
    return FALLBACK_SOURCE_WORK_ROOT.resolve()


def _source_phase1_round() -> int:
    return _int_env("DQA0835_SOURCE_PHASE1_ROUND", DEFAULT_SOURCE_PHASE1_ROUND)


def _round_from_path(path: Path) -> int | None:
    match = ROUND_RE.search(str(path))
    return int(match.group("round")) if match else None


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


def _run_name_from_config(config: Path) -> str:
    try:
        import yaml

        payload = yaml.safe_load(config.read_text(encoding="utf-8"))
        return str(payload.get("name", config.stem))
    except Exception:
        return config.stem


def _seed_metadata_path(workspace_root: Path) -> Path:
    return workspace_root / "phase2_seed.json"


def _seed_phase1_checkpoint(workspace_root: Path) -> Path:
    workspace_root = workspace_root.resolve()
    source_root = _source_workspace()
    source_round = _source_phase1_round()
    source = source_root / "global_checkpoints" / f"phase1_round{source_round:03d}_global.pt"
    target = workspace_root / "global_checkpoints" / "round000_warmup.pt"
    metadata_path = _seed_metadata_path(workspace_root)
    force = _bool_env("DQA0835_FORCE_SEED", False)

    if not source.exists():
        raise FileNotFoundError(
            f"Missing source phase1 seed checkpoint: {source}\n"
            "Set DQA0835_SOURCE_WORK_ROOT / DQA0835_SOURCE_PHASE1_ROUND for a different scheduled seed."
        )

    wanted: dict[str, Any] = {
        "protocol": DQA_PROTOCOL_VERSION,
        "seed_kind": "scheduled_phase1_round_as_round000_warmup",
        "source_work_root": str(source_root),
        "source_phase1_round": source_round,
        "source_checkpoint": str(source),
        "target_checkpoint": str(target.resolve()),
        "note": "This is a fixed scheduled initialization, not a validation-selected anchor teacher.",
    }

    if target.exists() and not force:
        if metadata_path.exists():
            try:
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
            same_seed = (
                existing.get("protocol") == wanted["protocol"]
                and existing.get("seed_kind") == wanted["seed_kind"]
                and existing.get("source_checkpoint") == wanted["source_checkpoint"]
            )
            if same_seed:
                print(f"Reusing DQA08_3_5 phase2 seed checkpoint: {target}")
                return target
        raise RuntimeError(
            f"Seed checkpoint already exists with different metadata: {target}\n"
            f"Existing metadata: {metadata_path if metadata_path.exists() else '(missing)'}\n"
            "Use DQA0835_FORCE_SEED=1 only if you intentionally want to overwrite it."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    metadata_path.write_text(json.dumps(wanted, indent=2), encoding="utf-8")
    print(f"Seeded DQA08_3_5 phase2 from scheduled phase1 round {source_round}: {target}")
    return target


def _prepare_temporal_memory_modules(work_root: Path):
    setup, fedsto = _prepare_scene_modules(work_root)
    source_weight = _source_workspace() / "weights" / "efficient-yolov5l.pt"
    if source_weight.exists():
        fedsto.PRETRAINED_PATH = source_weight
    exact_setup = setup.base
    original_config = getattr(
        exact_setup,
        "_dqa0835_original_efficientteacher_config",
        exact_setup.efficientteacher_config,
    )
    exact_setup._dqa0835_original_efficientteacher_config = original_config

    client_lr0 = _float_env("DQA0835_CLIENT_LR0", 3e-4)
    server_lr0 = _float_env("DQA0835_SERVER_LR0", 1e-3)

    def temporal_memory_config(**kwargs):
        if hasattr(setup, "_sync_base_paths"):
            setup._sync_base_paths()
        cfg = original_config(**kwargs)
        target = kwargs.get("target")
        name = str(kwargs.get("name", ""))

        if target is not None:
            cfg.setdefault("SSOD", {}).update(copy.deepcopy(SSOD_PROFILE))
            cfg.setdefault("FedSTO", {})["train_scope"] = os.environ.get("DQA0835_CLIENT_TRAIN_SCOPE", "all")
            cfg["FedSTO"]["orthogonal_weight"] = _float_env("DQA0835_CLIENT_ORTHOGONAL_WEIGHT", 1e-4)
            cfg.setdefault("hyp", {})["lr0"] = client_lr0
            cfg["hyp"]["lrf"] = 1.0
        elif name != "runtime_server_warmup":
            cfg.setdefault("FedSTO", {})["train_scope"] = os.environ.get("DQA0835_SERVER_TRAIN_SCOPE", "all")
            cfg["FedSTO"]["orthogonal_weight"] = _float_env("DQA0835_SERVER_ORTHOGONAL_WEIGHT", 1e-4)
            cfg.setdefault("hyp", {})["lr0"] = server_lr0
            cfg["hyp"]["lrf"] = 1.0

        return cfg

    setup.efficientteacher_config = temporal_memory_config
    exact_setup.efficientteacher_config = temporal_memory_config
    fedsto.setup.efficientteacher_config = temporal_memory_config
    return setup, fedsto


_ORIGINAL_RUN_TRAIN = base.run_train


def _temporal_memory_run_train(fedsto, config: Path, args, *, extra_env: dict[str, str] | None = None):
    env = dict(extra_env or {})
    run_name = _run_name_from_config(config)
    parsed = _parse_run_name(run_name)

    env.setdefault("ET_SKIP_AFTER_TRAIN_BEST_VAL", os.environ.get("ET_SKIP_AFTER_TRAIN_BEST_VAL", "1"))
    env["DQA08_TRI_STAGE_GATE"] = "0"
    env["DQA0834_STATS_QUALITY_MODE"] = os.environ.get("DQA0835_STATS_QUALITY_MODE", "confidence")
    env["DQA_STATS_QUALITY_MODE"] = os.environ.get("DQA0835_STATS_QUALITY_MODE", "confidence")

    if parsed and parsed["role"] == "client" and parsed["phase"] >= 2:
        memory_root = Path(os.environ.get("DQA0835_MEMORY_ROOT", str(args.stats_root / "pseudo_memory"))).resolve()
        memory_root.mkdir(parents=True, exist_ok=True)
        env["DQA0835_PSEUDO_MEMORY"] = "1"
        env["DQA0835_PSEUDO_MEMORY_PATH"] = str(memory_root / f"pseudo_memory_client{parsed['client_id']}.json")
        env.setdefault("DQA0835_MEMORY_IOU", os.environ.get("DQA0835_MEMORY_IOU", "0.55"))
        env.setdefault("DQA0835_MEMORY_MERGE_IOU", os.environ.get("DQA0835_MEMORY_MERGE_IOU", "0.70"))
        env.setdefault("DQA0835_STABLE_ROUNDS", os.environ.get("DQA0835_STABLE_ROUNDS", "2"))
        env.setdefault("DQA0835_NEW_SCORE_CAP", os.environ.get("DQA0835_NEW_SCORE_CAP", "0.70"))
        env.setdefault("DQA0835_MATCHED_SCORE_CAP", os.environ.get("DQA0835_MATCHED_SCORE_CAP", "0.74"))
        env.setdefault("DQA0835_STABLE_SCORE_FLOOR", os.environ.get("DQA0835_STABLE_SCORE_FLOOR", "0.78"))
        env.setdefault("DQA0835_STABLE_OBJ_FLOOR", os.environ.get("DQA0835_STABLE_OBJ_FLOOR", "0.85"))
        env.setdefault("DQA0835_STABLE_CLS_FLOOR", os.environ.get("DQA0835_STABLE_CLS_FLOOR", "0.85"))
        env.setdefault("DQA0835_MAX_ENTRIES_PER_IMAGE", os.environ.get("DQA0835_MAX_ENTRIES_PER_IMAGE", "80"))
        print(
            "DQA08_3_5 temporal memory:",
            f"client={parsed['client_id']}",
            f"phase={parsed['phase']}",
            f"round={parsed['round']}",
            f"memory={env['DQA0835_PSEUDO_MEMORY_PATH']}",
            f"stable_rounds={env['DQA0835_STABLE_ROUNDS']}",
        )

    return _ORIGINAL_RUN_TRAIN(fedsto, config, args, extra_env=env or None)


def _aggregate_temporal_memory(
    client_checkpoints,
    server_checkpoint,
    output_checkpoint,
    stats,
    state_path,
    config,
    repo_root,
):
    output, state = v2.aggregate_checkpoints(
        client_checkpoints=client_checkpoints,
        server_checkpoint=server_checkpoint,
        output_checkpoint=output_checkpoint,
        stats=stats,
        state_path=state_path,
        config=config,
        repo_root=repo_root,
    )
    state.setdefault("phase2_temporal_memory", []).append(
        {
            "round": _round_from_path(output_checkpoint),
            "source_work_root": str(_source_workspace()),
            "source_phase1_round": _source_phase1_round(),
            "memory_iou": _float_env("DQA0835_MEMORY_IOU", 0.55),
            "stable_rounds": _int_env("DQA0835_STABLE_ROUNDS", 2),
            "new_score_cap": _float_env("DQA0835_NEW_SCORE_CAP", 0.70),
            "matched_score_cap": _float_env("DQA0835_MATCHED_SCORE_CAP", 0.74),
            "stable_score_floor": _float_env("DQA0835_STABLE_SCORE_FLOOR", 0.78),
        }
    )
    v2.save_state(state_path, state)
    return output, state


base.DQA_PROTOCOL_VERSION = DQA_PROTOCOL_VERSION
base.DEFAULT_DQA_WORK_ROOT = DEFAULT_WORK_ROOT
base._prepare_fedsto_modules = _prepare_temporal_memory_modules
base.run_train = _temporal_memory_run_train
base.AggregationConfig = v2.AggregationConfig
base.aggregate_fedavg_checkpoints = v2.aggregate_fedavg_checkpoints
base.aggregate_checkpoints = _aggregate_temporal_memory
base.compute_reliability = v2.compute_reliability
base.load_round_stats = v2.load_round_stats
base.load_state = v2.load_state
base.save_state = v2.save_state


def main() -> None:
    parsed = base.parse_args()
    if parsed.protocol_version != DQA_PROTOCOL_VERSION:
        parsed.protocol_version = DQA_PROTOCOL_VERSION

    os.environ["DQA0835_PHASE2_ROUNDS"] = str(parsed.phase2_rounds)
    os.environ.setdefault("DQA0835_MEMORY_ROOT", str((parsed.stats_root / "pseudo_memory").resolve()))

    if parsed.setup_only:
        base._prepare_fedsto_modules(parsed.workspace_root)[0].build_base_configs()
        return

    if parsed.phase1_rounds != 0:
        raise ValueError("08_3_5 is phase2-only. Run with --phase1-rounds 0.")

    if not parsed.dry_run:
        _seed_phase1_checkpoint(parsed.workspace_root)

    print("DQA08_3_5 profile: phase2-only temporal pseudo-label memory")
    print(f"DQA08_3_5 seed: {_source_workspace()} phase1 round {_source_phase1_round()}")
    print(f"DQA08_3_5 memory root: {os.environ['DQA0835_MEMORY_ROOT']}")
    print(f"DQA08_3_5 stable rounds: {_int_env('DQA0835_STABLE_ROUNDS', 2)}")
    print(f"DQA08_3_5 score caps: new<={_float_env('DQA0835_NEW_SCORE_CAP', 0.70)} matched<={_float_env('DQA0835_MATCHED_SCORE_CAP', 0.74)} stable>={_float_env('DQA0835_STABLE_SCORE_FLOOR', 0.78)}")
    print(f"DQA08_3_5 dqa_start_phase: {parsed.dqa_start_phase}")
    base.run_protocol(parsed)


if __name__ == "__main__":
    main()
