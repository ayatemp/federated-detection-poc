#!/usr/bin/env python3
"""Create notebook 12 for weighted domain-winner soup."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "12_domain_winner_soup.ipynb"


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 12 Domain-Winner Soup

11 found domain-winning checkpoints, especially for highway_night, while keeping
total mAP stable.  This notebook tests whether those winners can be merged into
a single global checkpoint via weighted checkpoint soup.
"""
        ),
        code(
            """
from pathlib import Path
import pandas as pd

ROOT = Path('/app/Object_Detection')
PROJECT = ROOT / 'dynamic_quality_aware_classwise_aggregation' / 'moe_dqa_judger'
OUT = PROJECT / 'output' / '12_domain_winner_soup'
OUT
"""
        ),
        code(
            """
import subprocess, sys

cmd = [
    sys.executable,
    str(PROJECT / 'scripts' / 'run_12_domain_winner_soup.py'),
    '--workspace-root', str(OUT),
    '--max-total-drop', '0.0011',
    '--domain-eval-topk', '6',
    '--resume',
    '--notify-discord',
]
print(' '.join(cmd))
subprocess.run(cmd, cwd=ROOT, check=True)
"""
        ),
        code(
            """
total = pd.read_csv(OUT / 'stats' / '12_total_eval.csv')
display(total.sort_values('total_score', ascending=False)[['label','total_score','map50','map50_95','members']])

summary = pd.read_csv(OUT / 'stats' / '12_domain_summary.csv')
display(summary[['label','total_score','day_mean_score','night_mean_score','worst_domain_score','group_dro_score','night_mean_map50']].head(15))

router = pd.read_csv(OUT / 'stats' / '12_domain_router_summary.csv')
display(router)

print((OUT / '12_domain_winner_soup_report.md').read_text())
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
