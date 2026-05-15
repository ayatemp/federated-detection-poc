#!/usr/bin/env python3
"""Create notebook 05 for the greedy-soup judger."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "05_greedy_soup_judger.ipynb"


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 05 Greedy Soup Judger

This experiment changes the objective: instead of always choosing a fresh round
mixture, it builds a monotonic global model by greedily adding only checkpoints
that improve the proxy score.  This is inspired by model soups and gives the
judger a built-in "do not get worse as rounds/candidates increase" mechanism.
"""
        ),
        code(
            """
from pathlib import Path
import pandas as pd

ROOT = Path('/app/Object_Detection')
PROJECT = ROOT / 'dynamic_quality_aware_classwise_aggregation' / 'moe_dqa_judger'
OUT = PROJECT / 'output' / '05_greedy_soup_judger'
OUT
"""
        ),
        code(
            """
import subprocess, sys

cmd = [
    sys.executable,
    str(PROJECT / 'scripts' / 'run_05_greedy_soup_judger.py'),
    '--workspace-root', str(OUT),
    '--max-round', '6',
    '--mini-images', '512',
    '--max-candidates', '36',
    '--greedy-limit', '24',
    '--full-eval-last', '6',
    '--val-batch-size', '32',
    '--notify-discord',
    '--force',
]
print(' '.join(cmd))
subprocess.run(cmd, cwd=ROOT, check=True)
"""
        ),
        code(
            """
steps = pd.read_csv(OUT / 'stats' / '05_greedy_soup_steps.csv')
display(steps[steps['accepted'] == True][['step','candidate','member_count','map50','map50_95','score']])

full = pd.read_csv(OUT / 'stats' / '05_greedy_soup_full_eval.csv')
display(full[['step','candidate','member_count','map50','map50_95','precision','recall','score']])

print((OUT / '05_greedy_soup_judger_report.md').read_text())
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
