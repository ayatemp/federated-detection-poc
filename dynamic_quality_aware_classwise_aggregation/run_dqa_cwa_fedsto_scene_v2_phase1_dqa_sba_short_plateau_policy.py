#!/usr/bin/env python3
"""Run warm-up -> Phase 1 -> Phase 2 with short-plateau DQA-SBA.

08_4_3 keeps the 08_4 idea: DQA starts in Phase 1 and controls only a
server-anchored backbone residual.  Phase 2 is treated as a short impulse:
the first few rounds may adapt, then target influence drops to a near-freeze
plateau so long runs should converge instead of accumulating pseudo-label bias.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch

import dqa_cwa_aggregation as v1
import dqa_cwa_aggregation_v2 as v2
import dqa_sba_aggregation as sba
import run_dqa_cwa_fedsto as base
import run_dqa_cwa_fedsto_scene_v2_tri_stage_policy as tri_stage


DQA_PROTOCOL_VERSION = "dqa08_4_3_scene_phase1_dqa_sba_short_plateau_policy_v1"
DEFAULT_WORK_ROOT = base.RESEARCH_ROOT / "efficientteacher_dqa08_4_3_scene_phase1_dqa_sba_short_plateau_policy"
PHASE_RE = re.compile(r"phase(?P<phase>\d+)_round(?P<round>\d+)")


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


def _phase_round_from_path(path: Path) -> tuple[int | None, int | None]:
    match = PHASE_RE.search(str(path))
    if not match:
        return None, None
    return int(match.group("phase")), int(match.group("round"))


def _scheduled(prefix: str, round_idx: int | None, default_start: float, default_end: float) -> float:
    start = _float_env(f"{prefix}_START", default_start)
    end = _float_env(f"{prefix}_END", default_end)
    total = max(_int_env("DQA0843_PHASE2_DECAY_ROUNDS", 3), 1)
    if round_idx is None or total <= 1:
        return start
    progress = min(max((round_idx - 1) / (total - 1), 0.0), 1.0)
    return start + (end - start) * progress


def _scheduled_value(prefix: str, round_idx: int | None, default_start: float, default_end: float) -> float:
    return _scheduled(prefix, round_idx, default_start, default_end)


def _phase2_config(config: v2.AggregationConfig, round_idx: int | None) -> v2.AggregationConfig:
    classwise_blend = _scheduled_value(
        "DQA0843_PHASE2_CLASSWISE_BLEND",
        round_idx,
        min(float(config.classwise_blend), 0.040),
        0.003,
    )
    residual_blend = _scheduled_value("DQA0843_PHASE2_RESIDUAL", round_idx, 0.040, 0.002)
    min_server_alpha = _scheduled_value("DQA0843_PHASE2_MIN_SERVER_ALPHA", round_idx, 0.88, 0.985)
    server_anchor = _scheduled_value(
        "DQA0843_PHASE2_SERVER_ANCHOR",
        round_idx,
        max(float(config.server_anchor), 24.0),
        80.0,
    )
    temperature = _scheduled_value(
        "DQA0843_PHASE2_TEMPERATURE",
        round_idx,
        max(float(config.temperature), 3.4),
        6.0,
    )
    stability_lambda = _scheduled_value(
        "DQA0843_PHASE2_STABILITY_LAMBDA",
        round_idx,
        max(float(config.stability_lambda), 0.82),
        0.95,
    )
    return replace(
        copy.copy(config),
        classwise_blend=classwise_blend,
        residual_blend=residual_blend,
        min_server_alpha=min_server_alpha,
        server_anchor=server_anchor,
        temperature=temperature,
        stability_lambda=stability_lambda,
    )


def _stable_phase2_fallback(
    client_checkpoints: Sequence[Path],
    server_checkpoint: Path,
    output_checkpoint: Path,
    *,
    repo_root: Path,
    localize_bn: bool,
) -> Path:
    _, round_idx = _phase_round_from_path(output_checkpoint)
    residual_blend = _scheduled_value("DQA0843_PHASE2_RESIDUAL", round_idx, 0.040, 0.002)
    client_ckpts = [v1._load_checkpoint(path, repo_root) for path in client_checkpoints]
    server_ckpt = v1._load_checkpoint(server_checkpoint, repo_root)
    base_ckpt = copy.deepcopy(server_ckpt)

    client_state_dicts = [v1._model_state_dict(ckpt, "model") for ckpt in client_ckpts]
    server_state_dict = v1._model_state_dict(server_ckpt, "model")
    model_state = v2._server_anchored_state_dict(
        client_state_dicts,
        server_state_dict,
        residual_blend=residual_blend,
        localize_bn=localize_bn,
    )
    v1._replace_model_state(base_ckpt, model_state, "model")

    if base_ckpt.get("ema") is not None:
        ema_client_dicts = [v1._model_state_dict(ckpt, "ema") for ckpt in client_ckpts if ckpt.get("ema") is not None]
        server_ema = v1._model_state_dict(server_ckpt, "ema") if server_ckpt.get("ema") is not None else None
        if len(ema_client_dicts) == len(client_ckpts) and server_ema is not None:
            ema_state = v2._server_anchored_state_dict(
                ema_client_dicts,
                server_ema,
                residual_blend=residual_blend,
                localize_bn=localize_bn,
            )
            v1._replace_model_state(base_ckpt, ema_state, "ema")

    base_ckpt["epoch"] = -1
    base_ckpt["optimizer"] = None
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base_ckpt, output_checkpoint)
    return output_checkpoint


def _aggregate_checkpoints(
    client_checkpoints,
    server_checkpoint,
    output_checkpoint,
    stats,
    state_path,
    config,
    repo_root,
):
    phase, round_idx = _phase_round_from_path(output_checkpoint)
    if phase == 1:
        print(
            "DQA08_4_3 Phase 1 aggregation: DQA-SBA "
            f"residual={sba.scheduled_phase1_residual(sba.round_from_path(output_checkpoint)):.4f}"
        )
        return sba.aggregate_phase1_backbone_checkpoints(
            client_checkpoints=client_checkpoints,
            server_checkpoint=server_checkpoint,
            output_checkpoint=output_checkpoint,
            stats=stats,
            state_path=state_path,
            config=config,
            repo_root=repo_root,
        )

    cfg = _phase2_config(config, round_idx)
    output, state = v2.aggregate_checkpoints(
        client_checkpoints=client_checkpoints,
        server_checkpoint=server_checkpoint,
        output_checkpoint=output_checkpoint,
        stats=stats,
        state_path=state_path,
        config=cfg,
        repo_root=repo_root,
    )
    record = {
        "round": round_idx,
        "classwise_blend": cfg.classwise_blend,
        "residual_blend": cfg.residual_blend,
        "min_server_alpha": cfg.min_server_alpha,
        "server_anchor": cfg.server_anchor,
        "temperature": cfg.temperature,
        "stability_lambda": cfg.stability_lambda,
        "phase2_client_train_scope": os.environ.get("DQA0843_PHASE2_CLIENT_TRAIN_SCOPE", "neck_head"),
        "phase2_pseudo_loss_scale": _scheduled_value("DQA0843_PHASE2_PSEUDO_LOSS_SCALE", round_idx, 0.55, 0.08),
    }
    state.setdefault("phase2_stable_plateau", []).append(record)
    state["last_phase2_stable_plateau"] = record
    v2.save_state(state_path, state)
    print(
        "DQA08_4_3 Phase 2 aggregation:",
        f"round={round_idx}",
        f"residual={cfg.residual_blend:.4f}",
        f"classwise={cfg.classwise_blend:.4f}",
        f"min_server_alpha={cfg.min_server_alpha:.3f}",
        f"server_anchor={cfg.server_anchor:.2f}",
    )
    return output, state


def _aggregate_fedavg_checkpoints(
    client_checkpoints,
    server_checkpoint,
    output_checkpoint,
    *,
    repo_root,
    localize_bn=True,
):
    phase, _ = _phase_round_from_path(output_checkpoint)
    if phase == 1:
        print("DQA08_4_3 Phase 1 fallback: conservative backbone residual average")
        return sba.aggregate_phase1_fallback(
            client_checkpoints=client_checkpoints,
            server_checkpoint=server_checkpoint,
            output_checkpoint=output_checkpoint,
            repo_root=repo_root,
            localize_bn=localize_bn,
        )
    print("DQA08_4_3 Phase 2 fallback: scheduled server-anchored residual average")
    return _stable_phase2_fallback(
        client_checkpoints=client_checkpoints,
        server_checkpoint=server_checkpoint,
        output_checkpoint=output_checkpoint,
        repo_root=repo_root,
        localize_bn=localize_bn,
    )


def _prepare_stable_scene_modules(work_root: Path):
    setup, fedsto = tri_stage._prepare_tri_stage_scene_modules(work_root)
    tri_stage_config = setup.efficientteacher_config

    def stable_config(**kwargs):
        cfg = tri_stage_config(**kwargs)
        target = kwargs.get("target")
        name = str(kwargs.get("name", ""))
        parsed = tri_stage._parse_run_name(name)
        phase = int(parsed["phase"]) if parsed else None
        round_idx = int(parsed["round"]) if parsed else None

        if target is not None and phase == 2:
            cfg.setdefault("FedSTO", {})["train_scope"] = os.environ.get(
                "DQA0843_PHASE2_CLIENT_TRAIN_SCOPE",
                "neck_head",
            )
            cfg.setdefault("hyp", {})["lr0"] = _scheduled_value("DQA0843_PHASE2_CLIENT_LR0", round_idx, 1.2e-4, 2.0e-5)
            cfg["hyp"]["lrf"] = 1.0
            scale = _scheduled_value("DQA0843_PHASE2_PSEUDO_LOSS_SCALE", round_idx, 0.55, 0.08)
            ssod = cfg.setdefault("SSOD", {})
            for key in ("teacher_loss_weight", "box_loss_weight", "obj_loss_weight", "cls_loss_weight"):
                if key in ssod:
                    ssod[key] = float(ssod[key]) * scale
            ssod["ignore_obj"] = os.environ.get("DQA0843_PHASE2_IGNORE_OBJ", "1").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            ssod["pseudo_label_with_obj"] = True
            ssod["pseudo_label_with_bbox"] = False
            ssod["pseudo_label_with_cls"] = False
        elif target is None and phase == 2:
            cfg.setdefault("FedSTO", {})["train_scope"] = os.environ.get("DQA0843_PHASE2_SERVER_TRAIN_SCOPE", "all")
            cfg["FedSTO"]["orthogonal_weight"] = _float_env("DQA0843_PHASE2_SERVER_ORTHOGONAL_WEIGHT", 1e-4)
            cfg.setdefault("hyp", {})["lr0"] = _scheduled_value("DQA0843_PHASE2_SERVER_LR0", round_idx, 3.0e-4, 1.0e-4)
            cfg["hyp"]["lrf"] = 1.0

        return cfg

    setup.efficientteacher_config = stable_config
    setup.base.efficientteacher_config = stable_config
    fedsto.setup.efficientteacher_config = stable_config
    return setup, fedsto


_ORIGINAL_RUN_TRAIN = tri_stage._tri_stage_run_train


def _stable_run_train(fedsto, config: Path, args, *, extra_env: dict[str, str] | None = None):
    env = dict(extra_env or {})
    try:
        import yaml

        run_name = str(yaml.safe_load(config.read_text(encoding="utf-8"))["name"])
    except Exception:
        run_name = config.stem
    parsed = tri_stage._parse_run_name(run_name)
    if parsed and parsed["role"].startswith("client") and int(parsed["phase"]) == 2:
        env.setdefault("ET_MAX_PSEUDO_PER_IMAGE", os.environ.get("DQA0843_MAX_PSEUDO_PER_IMAGE", "60"))
        env.setdefault("ET_MAX_PSEUDO_PER_CLASS_IMAGE", os.environ.get("DQA0843_MAX_PSEUDO_PER_CLASS_IMAGE", "15"))
    env.setdefault("ET_SKIP_AFTER_TRAIN_BEST_VAL", os.environ.get("ET_SKIP_AFTER_TRAIN_BEST_VAL", "1"))
    return _ORIGINAL_RUN_TRAIN(fedsto, config, args, extra_env=env or None)


base.AggregationConfig = v2.AggregationConfig
base.aggregate_fedavg_checkpoints = _aggregate_fedavg_checkpoints
base.aggregate_checkpoints = _aggregate_checkpoints
base.compute_reliability = v2.compute_reliability
base.load_round_stats = v2.load_round_stats
base.load_state = v2.load_state
base.save_state = v2.save_state
base.DQA_PROTOCOL_VERSION = DQA_PROTOCOL_VERSION
base.DEFAULT_DQA_WORK_ROOT = DEFAULT_WORK_ROOT
base._prepare_fedsto_modules = _prepare_stable_scene_modules
base.run_train = _stable_run_train


def main() -> None:
    parsed = base.parse_args()
    if parsed.protocol_version != DQA_PROTOCOL_VERSION:
        parsed.protocol_version = DQA_PROTOCOL_VERSION

    if parsed.dqa_start_phase != 1:
        print(f"DQA08_4_3 forces --dqa-start-phase 1; got {parsed.dqa_start_phase}, overriding.")
        parsed.dqa_start_phase = 1

    os.environ["DQA08_STATS_ROOT"] = str(parsed.stats_root.resolve())
    os.environ["DQA08_PHASE2_ROUNDS"] = str(parsed.phase2_rounds)
    os.environ["DQA0843_PHASE2_ROUNDS"] = str(parsed.phase2_rounds)
    os.environ["DQA08_NUM_CLASSES"] = str(parsed.num_classes)
    os.environ["DQA084_PHASE1_ROUNDS"] = str(parsed.phase1_rounds)
    os.environ.setdefault("DQA084_PHASE1_RESIDUAL_START", "0.42")
    os.environ.setdefault("DQA084_PHASE1_RESIDUAL_END", "0.12")
    os.environ.setdefault("DQA084_PHASE1_MAX_RELATIVE_UPDATE", "0.025")
    os.environ.setdefault("DQA08_POLICY_HORIZON_ROUNDS", str(max(parsed.phase2_rounds, 1)))
    os.environ.setdefault(
        "DQA08_THRESHOLD_LOG",
        str((parsed.stats_root / "phase1_dqa_sba_short_plateau_policy_schedule.jsonl").resolve()),
    )

    if not parsed.setup_only and not parsed.dry_run and not tri_stage._policy_model_path().exists():
        raise FileNotFoundError(
            f"Missing learned threshold policy model: {tri_stage._policy_model_path()}. "
            "Run threshold_policy_model/01_train_dqa05_threshold_policy.ipynb first."
        )

    if parsed.setup_only:
        base._prepare_fedsto_modules(parsed.workspace_root)[0].build_base_configs()
    else:
        print("DQA08_4_3 profile: phase1 DQA-SBA + short phase2 plateau")
        print(f"DQA08_4_3 protocol: {DQA_PROTOCOL_VERSION}")
        print(
            "DQA08_4_3 phase1 residual schedule:",
            os.environ.get("DQA084_PHASE1_RESIDUAL_START", "0.42"),
            "->",
            os.environ.get("DQA084_PHASE1_RESIDUAL_END", "0.12"),
        )
        print(
            "DQA08_4_3 phase2 residual schedule:",
            os.environ.get("DQA0843_PHASE2_RESIDUAL_START", "0.040"),
            "->",
            os.environ.get("DQA0843_PHASE2_RESIDUAL_END", "0.002"),
        )
        print(f"DQA08_4_3 phase2 decay rounds: {os.environ.get('DQA0843_PHASE2_DECAY_ROUNDS', '3')}")
        print(f"DQA08_4_3 phase2 client train_scope: {os.environ.get('DQA0843_PHASE2_CLIENT_TRAIN_SCOPE', 'neck_head')}")
        print(f"DQA08_4_3 policy model: {tri_stage._policy_model_path()}")
        base.run_protocol(parsed)


if __name__ == "__main__":
    main()
