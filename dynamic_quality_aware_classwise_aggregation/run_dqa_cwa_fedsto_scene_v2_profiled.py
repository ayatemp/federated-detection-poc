#!/usr/bin/env python3
"""Run DQA-CWA v2 scene-client training with a safer pseudo-label loss profile.

This wrapper is for the 05 class/domain heterogeneity experiment.  It keeps the
scene-client DQA v2 runner shape, but patches the generated EfficientTeacher
configs so client pseudo-label training uses a conservative low-bbox SSOD loss.
The profile is selected by environment variables so the notebook can switch
between a 5h pilot and a later 12h run without editing the shared runner.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import dqa_cwa_aggregation_v2 as v2
import run_dqa_cwa_fedsto as base
from run_dqa_cwa_fedsto_scene import _prepare_scene_modules


DQA_PROTOCOL_VERSION = "dqa05_scene_class_profiled_ssod_v1"


SSOD_PROFILES: dict[str, dict] = {
    # Chosen from the 02/ET probes: keep pseudo labels strict enough to avoid
    # dense noisy boxes, reduce bbox/class damage, but leave enough active
    # classes for DQA's class-wise aggregation to have signal.
    "strict_low_bbox": {
        "nms_conf_thres": 0.35,
        "ignore_thres_low": 0.35,
        "ignore_thres_high": 0.75,
        "teacher_loss_weight": 0.35,
        "box_loss_weight": 0.02,
        "obj_loss_weight": 0.35,
        "cls_loss_weight": 0.10,
        "pseudo_label_with_bbox": False,
        "pseudo_label_with_cls": False,
    },
    "very_strict_low_bbox": {
        "nms_conf_thres": 0.50,
        "ignore_thres_low": 0.50,
        "ignore_thres_high": 0.85,
        "teacher_loss_weight": 0.25,
        "box_loss_weight": 0.015,
        "obj_loss_weight": 0.25,
        "cls_loss_weight": 0.05,
        "pseudo_label_with_bbox": False,
        "pseudo_label_with_cls": False,
    },
    # Not the default because it was not the best tested setting yet, but it is
    # kept here for the next ablation if pseudo-bbox noise still dominates.
    "objectness_only": {
        "nms_conf_thres": 0.35,
        "ignore_thres_low": 0.35,
        "ignore_thres_high": 0.75,
        "teacher_loss_weight": 0.25,
        "box_loss_weight": 0.0,
        "obj_loss_weight": 0.35,
        "cls_loss_weight": 0.0,
        "pseudo_label_with_bbox": False,
        "pseudo_label_with_cls": False,
    },
}


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return float(raw)


def _selected_profile() -> tuple[str, dict]:
    name = os.environ.get("DQA05_SSOD_PROFILE", "strict_low_bbox")
    if name not in SSOD_PROFILES:
        known = ", ".join(sorted(SSOD_PROFILES))
        raise ValueError(f"Unknown DQA05_SSOD_PROFILE={name!r}. Available: {known}")
    return name, deepcopy(SSOD_PROFILES[name])


def _prepare_profiled_scene_modules(work_root: Path):
    setup, fedsto = _prepare_scene_modules(work_root)
    exact_setup = setup.base
    original_config = getattr(
        exact_setup,
        "_dqa05_original_efficientteacher_config",
        exact_setup.efficientteacher_config,
    )
    exact_setup._dqa05_original_efficientteacher_config = original_config
    _, profile = _selected_profile()
    client_lr0 = _float_env("DQA05_CLIENT_LR0", 3e-4)
    server_lr0 = _float_env("DQA05_SERVER_LR0", 1e-3)

    def profiled_config(**kwargs):
        if hasattr(setup, "_sync_base_paths"):
            setup._sync_base_paths()
        cfg = original_config(**kwargs)
        target = kwargs.get("target")
        name = str(kwargs.get("name", ""))

        if target is not None:
            cfg.setdefault("SSOD", {}).update(profile)
            cfg.setdefault("hyp", {})["lr0"] = client_lr0
            cfg["hyp"]["lrf"] = 1.0
        elif name != "runtime_server_warmup":
            cfg.setdefault("hyp", {})["lr0"] = server_lr0
            cfg["hyp"]["lrf"] = 1.0

        return cfg

    setup.efficientteacher_config = profiled_config
    exact_setup.efficientteacher_config = profiled_config
    fedsto.setup.efficientteacher_config = profiled_config
    return setup, fedsto


base.AggregationConfig = v2.AggregationConfig
base.aggregate_fedavg_checkpoints = v2.aggregate_fedavg_checkpoints
base.aggregate_checkpoints = v2.aggregate_checkpoints
base.compute_reliability = v2.compute_reliability
base.load_round_stats = v2.load_round_stats
base.load_state = v2.load_state
base.save_state = v2.save_state
base.DQA_PROTOCOL_VERSION = DQA_PROTOCOL_VERSION
base.DEFAULT_DQA_WORK_ROOT = base.RESEARCH_ROOT / "efficientteacher_dqa05_scene_class_profile_5h"
base._prepare_fedsto_modules = _prepare_profiled_scene_modules


def main() -> None:
    parsed = base.parse_args()
    if parsed.protocol_version == DQA_PROTOCOL_VERSION:
        parsed.protocol_version = DQA_PROTOCOL_VERSION
    if parsed.setup_only:
        base._prepare_fedsto_modules(parsed.workspace_root)[0].build_base_configs()
    else:
        base.run_protocol(parsed)


if __name__ == "__main__":
    main()
