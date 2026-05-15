#!/usr/bin/env python3
"""Create notebook 10 for the night-targeted delta optimizer."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "10_night_targeted_delta_optimizer.ipynb"


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 10 Night-Targeted Delta Optimizer

07-09 showed that the strongest current rule is an incumbent-rebased soft mix:
keep the best r2 checkpoint and add only carefully selected later-round deltas.
The remaining weak point is night-domain quality.  This notebook searches
group-wise coefficients directly against night mini-slices, then validates the
best candidates on the full total and six day/night domain slices.
"""
        ),
        code(
            """
from pathlib import Path
import pandas as pd

ROOT = Path('/app/Object_Detection')
PROJECT = ROOT / 'dynamic_quality_aware_classwise_aggregation' / 'moe_dqa_judger'
OUT = PROJECT / 'output' / '10_night_targeted_delta_optimizer'
OUT
"""
        ),
        code(
            """
import subprocess, sys

cmd = [
    sys.executable,
    str(PROJECT / 'scripts' / 'run_10_night_targeted_delta_optimizer.py'),
    '--workspace-root', str(OUT),
    '--rounds', '4,15,19,21',
    '--night-mini-images', '512',
    '--random-candidates', '1',
    '--template-topk', '1',
    '--full-eval-topk', '5',
    '--per-domain-topk', '1',
    '--max-full-candidates', '7',
    '--resume',
    '--notify-discord',
]
print(' '.join(cmd))
subprocess.run(cmd, cwd=ROOT, check=True)
"""
        ),
        code(
            """
night = pd.read_csv(OUT / 'stats' / '10_night_probe_summary.csv')
display(night.head(12)[['label','night_objective','night_probe_mean_score','night_probe_worst_score','night_probe_mean_map50']])

summary = pd.read_csv(OUT / 'stats' / '10_full_domain_summary.csv')
display(summary[['label','total_score','day_mean_score','night_mean_score','worst_domain_score','group_dro_score','night_mean_map50']])

router = pd.read_csv(OUT / 'stats' / '10_domain_router_summary.csv')
display(router)

print((OUT / '10_night_targeted_delta_optimizer_report.md').read_text())
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
