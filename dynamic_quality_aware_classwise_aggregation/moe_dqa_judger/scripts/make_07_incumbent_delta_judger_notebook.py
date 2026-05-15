#!/usr/bin/env python3
"""Create notebook 07 for the incumbent-rebased delta judger."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "07_incumbent_delta_judger.ipynb"


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 07 Incumbent-Rebased Delta Judger

06 learned a robust judge, but the original round sequence still drifts after
round 2.  This experiment keeps the best known incumbent and applies only the
late-round update directions onto it:

`I_best + alpha * (A_t - G_t) + beta * (S_t - G_t)`

The judge evaluates body/head/router/expert deltas with the 06 robust proxy and
accepts a full-evaluated candidate only if it improves the incumbent.
"""
        ),
        code(
            """
from pathlib import Path
import pandas as pd

ROOT = Path('/app/Object_Detection')
PROJECT = ROOT / 'dynamic_quality_aware_classwise_aggregation' / 'moe_dqa_judger'
OUT = PROJECT / 'output' / '07_incumbent_delta_judger'
OUT
"""
        ),
        code(
            """
import subprocess, sys

cmd = [
    sys.executable,
    str(PROJECT / 'scripts' / 'run_07_incumbent_delta_judger.py'),
    '--workspace-root', str(OUT),
    '--rounds', '3,4,5,6',
    '--mini-splits', '3',
    '--mini-images', '384',
    '--random-candidates', '4',
    '--full-eval-topk', '2',
    '--val-batch-size', '32',
    '--notify-discord',
]
print(' '.join(cmd))
subprocess.run(cmd, cwd=ROOT, check=True)
"""
        ),
        code(
            """
summary = pd.read_csv(OUT / 'stats' / '07_summary.csv')
display(summary[['round','candidate_id','phase','mean_score','std_score','proxy_lcb_score','pred_full_score','judger_score']].head(20))

full = pd.read_csv(OUT / 'stats' / '07_full_eval.csv')
display(full[['round','candidate_id','mean_score','pred_full_score','map50','map50_95','score']])

accepted = pd.read_csv(OUT / 'stats' / '07_accepted_policy.csv')
display(accepted[['round','candidate_id','accepted','map50','map50_95','score','incumbent_after_score','reason']])

print((OUT / '07_incumbent_delta_judger_report.md').read_text())
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
