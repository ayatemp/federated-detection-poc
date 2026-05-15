#!/usr/bin/env python3
"""Create notebook 11 for the highway-night full optimizer."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "11_highway_night_full_optimizer.ipynb"


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 11 Highway-Night Full Optimizer

10 improved the dynamic night/domain policy a little, but the mini night proxy
was not strong enough.  This notebook directly optimizes candidates on the full
highway_night slice, then validates selected candidates on total and all six
domain slices.  Historical candidates from 08/10 are merged into the final
router pool so the policy can retain known good options.
"""
        ),
        code(
            """
from pathlib import Path
import pandas as pd

ROOT = Path('/app/Object_Detection')
PROJECT = ROOT / 'dynamic_quality_aware_classwise_aggregation' / 'moe_dqa_judger'
OUT = PROJECT / 'output' / '11_highway_night_full_optimizer'
OUT
"""
        ),
        code(
            """
import subprocess, sys

cmd = [
    sys.executable,
    str(PROJECT / 'scripts' / 'run_11_highway_night_full_optimizer.py'),
    '--workspace-root', str(OUT),
    '--rounds', '4,6,9,15,16,19,21',
    '--random-candidates', '0',
    '--template-topk', '1',
    '--full-eval-topk', '6',
    '--resume',
    '--notify-discord',
]
print(' '.join(cmd))
subprocess.run(cmd, cwd=ROOT, check=True)
"""
        ),
        code(
            """
highway = pd.read_csv(OUT / 'stats' / '11_highway_night_eval.csv')
display(highway.sort_values('highway_night_score', ascending=False).head(15)[['label','highway_night_score','map50','map50_95','recall']])

summary = pd.read_csv(OUT / 'stats' / '11_full_domain_summary.csv')
display(summary[['label','total_score','day_mean_score','night_mean_score','worst_domain_score','group_dro_score','night_mean_map50']].head(15))

router = pd.read_csv(OUT / 'stats' / '11_domain_router_summary.csv')
display(router)

print((OUT / '11_highway_night_full_optimizer_report.md').read_text())
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
