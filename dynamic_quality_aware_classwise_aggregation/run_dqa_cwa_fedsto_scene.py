#!/usr/bin/env python3
"""Run DQA-CWA with BDD100K scene-based clients."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import run_dqa_cwa_fedsto as base


DEFAULT_DQA_WORK_ROOT = base.RESEARCH_ROOT / "efficientteacher_dqa_cwa_scene_12h"


def _prepare_scene_modules(work_root: Path):
    if str(base.NAV_ROOT) not in sys.path:
        sys.path.insert(0, str(base.NAV_ROOT))

    setup = importlib.import_module("setup_fedsto_scene_reproduction")
    setup.WORK_ROOT = work_root
    setup.LIST_ROOT = work_root / "data_lists"
    setup.CONFIG_ROOT = work_root / "configs"
    setup.RUN_ROOT = work_root / "runs"

    fedsto = importlib.import_module("run_fedsto_efficientteacher_exact")
    fedsto.PRETRAINED_PATH = work_root / "weights" / "efficient-yolov5l.pt"
    fedsto.GLOBAL_DIR = work_root / "global_checkpoints"
    fedsto.CLIENT_STATE_DIR = work_root / "client_states"
    fedsto.HISTORY_PATH = work_root / "history.json"
    return setup, fedsto


def main() -> None:
    base.DEFAULT_DQA_WORK_ROOT = DEFAULT_DQA_WORK_ROOT
    base._prepare_fedsto_modules = _prepare_scene_modules
    parsed = base.parse_args()
    if parsed.setup_only:
        base._prepare_fedsto_modules(parsed.workspace_root)[0].build_base_configs()
    else:
        base.run_protocol(parsed)


if __name__ == "__main__":
    main()
