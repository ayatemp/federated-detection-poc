#!/usr/bin/env python3
"""Create notebook 08 for the domain-slice judger."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "08_domain_slice_judger.ipynb"


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 08 Domain-Slice Judger

07 removed round drift but several candidates tie on total mAP.  This notebook
evaluates the plateau candidates on six paper domain slices and ranks them with
a night/worst-domain aware objective.
"""
        ),
        code(
            """
from pathlib import Path
import pandas as pd

ROOT = Path('/app/Object_Detection')
PROJECT = ROOT / 'dynamic_quality_aware_classwise_aggregation' / 'moe_dqa_judger'
OUT = PROJECT / 'output' / '08_domain_slice_judger'
OUT
"""
        ),
        code(
            """
import subprocess, sys

cmd = [
    sys.executable,
    str(PROJECT / 'scripts' / 'run_08_domain_slice_judger.py'),
    '--workspace-root', str(OUT),
    '--max-candidates', '9',
    '--val-batch-size', '32',
    '--notify-discord',
]
print(' '.join(cmd))
subprocess.run(cmd, cwd=ROOT, check=True)
"""
        ),
        code(
            """
summary = pd.read_csv(OUT / 'stats' / '08_domain_summary.csv')
display(summary[['label','total_score','day_mean_score','night_mean_score','worst_domain_score','group_dro_score','night_mean_map50']])

domain = pd.read_csv(OUT / 'stats' / '08_domain_eval.csv')
display(domain[['label','domain','group','map50','map50_95','precision','recall','domain_score']].head(60))

print((OUT / '08_domain_slice_judger_report.md').read_text())
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
