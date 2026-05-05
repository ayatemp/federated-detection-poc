#!/usr/bin/env python3
"""Evaluate PseudoGT Learnability checkpoints on scene-wise validation splits."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
NAV_ROOT = REPO_ROOT / "navigating_data_heterogeneity"

if str(NAV_ROOT) not in sys.path:
    sys.path.insert(0, str(NAV_ROOT))

import evaluate_paper_protocol as shared


def configure_scene_setup(workspace: Path):
    if str(NAV_ROOT) not in sys.path:
        sys.path.insert(0, str(NAV_ROOT))
    setup = importlib.import_module("setup_fedsto_scene_reproduction")
    setup.WORK_ROOT = workspace
    setup.LIST_ROOT = workspace / "data_lists"
    setup.CONFIG_ROOT = workspace / "configs"
    setup.RUN_ROOT = workspace / "runs"
    return setup


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if not any(arg == "--workspace" or arg.startswith("--workspace=") for arg in args):
        args = ["--workspace", str((PROJECT_ROOT / "output").resolve()), *args]
    if not any(arg == "--splits" or arg.startswith("--splits=") for arg in args):
        args = ["--splits", "highway,citystreet,residential,total", *args]

    shared.configure_setup = configure_scene_setup
    return shared.main(args)


if __name__ == "__main__":
    raise SystemExit(main())

