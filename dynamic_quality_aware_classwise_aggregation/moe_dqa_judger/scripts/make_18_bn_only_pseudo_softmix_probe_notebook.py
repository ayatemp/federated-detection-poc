#!/usr/bin/env python3
"""Create notebook 18 for the BN-only target-pseudo softmix probe."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "18_bn_only_pseudo_softmix_probe.ipynb"


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 18 BN-Only Target-Pseudo Softmix Probe

17 showed that source-style BN/MoE/head learning is not transferring to target
validation, and client specialists are worse than the warmup.  This notebook
keeps the learning loop but exports only the safest locally learned part:
BatchNorm.  Target pseudoGT is used again, but box regression is disabled and
DQA aggregation is warmup-anchored.
"""
        ),
        code(
            """
from pathlib import Path
import pandas as pd

ROOT = Path('/app/Object_Detection')
PROJECT = ROOT / 'dynamic_quality_aware_classwise_aggregation' / 'moe_dqa_judger'
OUT = PROJECT / 'output' / '18_bn_only_pseudo_softmix_probe'
OUT
"""
        ),
        code(
            """
import subprocess, sys

cmd = [
    sys.executable,
    str(PROJECT / 'scripts' / 'run_18_bn_only_pseudo_softmix_probe.py'),
    '--workspace-root', str(OUT),
    '--phase1-rounds', '1',
    '--client-limit', '700',
    '--imgsz', '960',
    '--pseudo-imgsz', '1280',
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
metrics_path = OUT / 'stats' / '18_metrics.csv'
if metrics_path.exists():
    metrics = pd.read_csv(metrics_path)
    display(metrics)
print((OUT / '18_bn_only_pseudo_softmix_probe_report.md').read_text())
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
