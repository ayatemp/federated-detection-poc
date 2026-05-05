#!/usr/bin/env python3
"""Evaluate single-client EfficientTeacher checkpoints with the shared protocol."""

from __future__ import annotations

import sys
from pathlib import Path


RESEARCH_ROOT = Path(__file__).resolve().parent
NAV_ROOT = RESEARCH_ROOT.parent / "navigating_data_heterogeneity"

if str(NAV_ROOT) not in sys.path:
    sys.path.insert(0, str(NAV_ROOT))

from evaluate_paper_protocol import main as shared_main


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if not any(arg == "--workspace" or arg.startswith("--workspace=") for arg in args):
        args = ["--workspace", str((RESEARCH_ROOT / "efficientteacher_single_client").resolve()), *args]
    return shared_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
