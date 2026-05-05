#!/usr/bin/env python3
"""Evaluate scene-daynight DQA checkpoints on six BDD100K validation splits."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
NAV_ROOT = REPO_ROOT / "navigating_data_heterogeneity"

for path in (NAV_ROOT, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_paper_protocol as shared  # noqa: E402


def configure_scene_daynight_setup(workspace: Path):
    setup = importlib.import_module("setup_scene_daynight")
    setup.WORK_ROOT = workspace
    setup.LIST_ROOT = workspace / "data_lists"
    setup.CONFIG_ROOT = workspace / "configs"
    setup.RUN_ROOT = workspace / "runs"
    return setup


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if not any(arg == "--workspace" or arg.startswith("--workspace=") for arg in args):
        args = ["--workspace", str((PROJECT_ROOT / "output" / "01_repair_oriented_scene_daynight_dqa").resolve()), *args]
    if not any(arg == "--splits" or arg.startswith("--splits=") for arg in args):
        args = [
            "--splits",
            "highway_day,highway_night,citystreet_day,citystreet_night,residential_day,residential_night,total",
            *args,
        ]

    shared.configure_setup = configure_scene_daynight_setup
    return shared.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
