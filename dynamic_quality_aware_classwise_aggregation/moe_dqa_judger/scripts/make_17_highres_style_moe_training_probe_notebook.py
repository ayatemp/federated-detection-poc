#!/usr/bin/env python3
"""Create notebook 17 for the high-res style MoE training probe."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "17_highres_style_moe_training_probe.ipynb"


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 17 High-Res Style MoE Training Probe

16 showed that label-safe target-style learning at 640 preserves the model but
does not improve the paper protocol.  This notebook keeps the same self/source
only design, but moves training and evaluation to high resolution so BN/MoE/head
updates are learned at the small-object scale that helped routed inference.
"""
        ),
        code(
            """
from pathlib import Path
import pandas as pd

ROOT = Path('/app/Object_Detection')
PROJECT = ROOT / 'dynamic_quality_aware_classwise_aggregation' / 'moe_dqa_judger'
OUT = PROJECT / 'output' / '17_highres_style_moe_training_probe_v2'
OUT
"""
        ),
        code(
            """
import subprocess, sys

cmd = [
    sys.executable,
    str(PROJECT / 'scripts' / 'run_17_highres_style_moe_training_probe.py'),
    '--workspace-root', str(OUT),
    '--phase1-rounds', '1',
    '--client-limit', '1200',
    '--style-source-limit', '1000',
    '--imgsz', '960',
    '--pseudo-imgsz', '1280',
    '--style-imgsz', '960',
    '--batch-size', '32',
    '--val-batch-size', '16',
    '--workers', '48',
    '--notify-discord',
]
print(' '.join(cmd))
subprocess.run(cmd, cwd=ROOT, check=True)
"""
        ),
        code(
            """
metrics_path = OUT / 'stats' / '17_metrics.csv'
if metrics_path.exists():
    metrics = pd.read_csv(metrics_path)
    display(metrics)
print((OUT / '17_highres_style_moe_training_probe_report.md').read_text())
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
