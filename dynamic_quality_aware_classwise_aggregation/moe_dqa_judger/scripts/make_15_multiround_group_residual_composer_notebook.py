#!/usr/bin/env python3
"""Create notebook 15 for multi-round group residual composition."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "15_multiround_group_residual_composer.ipynb"


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 15 Multi-Round Group Residual Composer

14 showed that the selector is now close to the available oracle.  This
notebook therefore creates stronger self-generated candidates by composing
body/head/router/expert residuals from different rounds.
"""
        ),
        code(
            """
from pathlib import Path
import pandas as pd

ROOT = Path('/app/Object_Detection')
PROJECT = ROOT / 'dynamic_quality_aware_classwise_aggregation' / 'moe_dqa_judger'
OUT = PROJECT / 'output' / '15_multiround_group_residual_composer'
OUT
"""
        ),
        code(
            """
import subprocess, sys

cmd = [
    sys.executable,
    str(PROJECT / 'scripts' / 'run_15_multiround_group_residual_composer.py'),
    '--workspace-root', str(OUT),
    '--max-total-drop', '0.0011',
    '--domain-eval-topk', '8',
    '--notify-discord',
]
print(' '.join(cmd))
subprocess.run(cmd, cwd=ROOT, check=True)
"""
        ),
        code(
            """
total = pd.read_csv(OUT / 'stats' / '15_total_eval.csv')
display(total[['label','total_score','map50','map50_95']].sort_values('total_score', ascending=False))

summary = pd.read_csv(OUT / 'stats' / '15_domain_summary.csv')
display(summary[['label','total_score','day_mean_score','night_mean_score','worst_domain_score','group_dro_score']].head(18))

router = pd.read_csv(OUT / 'stats' / '15_domain_router_summary.csv')
display(router)

print((OUT / '15_multiround_group_residual_composer_report.md').read_text())
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
