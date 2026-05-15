#!/usr/bin/env python3
"""Create notebook 16 for the night style BN/MoE training probe."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "16_night_style_bn_moe_training_probe.ipynb"


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 16 Night Style BN/MoE Training Probe

15 showed that checkpoint mixing is near its current oracle.  This notebook
returns to learning: night clients only, pseudoGT for DQA/router statistics
only, and target-styled source GT used to train BN/MoE slots.
"""
        ),
        code(
            """
from pathlib import Path
import pandas as pd

ROOT = Path('/app/Object_Detection')
PROJECT = ROOT / 'dynamic_quality_aware_classwise_aggregation' / 'moe_dqa_judger'
OUT = PROJECT / 'output' / '16_night_style_bn_moe_training_probe'
OUT
"""
        ),
        code(
            """
import subprocess, sys

cmd = [
    sys.executable,
    str(PROJECT / 'scripts' / 'run_16_night_style_bn_moe_training_probe.py'),
    '--workspace-root', str(OUT),
    '--phase1-rounds', '1',
    '--client-limit', '1500',
    '--style-source-limit', '1600',
    '--batch-size', '80',
    '--val-batch-size', '32',
    '--workers', '48',
    '--notify-discord',
]
print(' '.join(cmd))
subprocess.run(cmd, cwd=ROOT, check=True)
"""
        ),
        code(
            """
metrics_path = OUT / 'stats' / '16_metrics.csv'
if metrics_path.exists():
    metrics = pd.read_csv(metrics_path)
    display(metrics)
print((OUT / '16_night_style_bn_moe_training_probe_report.md').read_text())
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
