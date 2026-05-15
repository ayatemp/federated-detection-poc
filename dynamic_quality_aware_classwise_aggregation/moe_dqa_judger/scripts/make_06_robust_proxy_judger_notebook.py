#!/usr/bin/env python3
"""Create notebook 06 for the robust multi-split proxy judger."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "06_robust_proxy_judger.ipynb"


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 06 Robust Proxy Judger

05 showed that a monotonic greedy soup is not enough: the proxy itself must be
stable.  This notebook evaluates the self-generated candidates from 02/03/04/05
on several mini validation splits, then trains a lightweight calibrator to map
multi-split proxy statistics to full-protocol quality.

The intended judge behavior is conservative but dynamic:

- trust a candidate only if it is good across splits, not just on one lucky split
- learn the relation between proxy evidence and full mAP from previous full evals
- choose a per-round global candidate without adding external teachers
"""
        ),
        code(
            """
from pathlib import Path
import pandas as pd

ROOT = Path('/app/Object_Detection')
PROJECT = ROOT / 'dynamic_quality_aware_classwise_aggregation' / 'moe_dqa_judger'
OUT = PROJECT / 'output' / '06_robust_proxy_judger'
OUT
"""
        ),
        code(
            """
import subprocess, sys

cmd = [
    sys.executable,
    str(PROJECT / 'scripts' / 'run_06_robust_proxy_judger.py'),
    '--workspace-root', str(OUT),
    '--max-round', '6',
    '--mini-splits', '3',
    '--mini-images', '384',
    '--max-candidates', '44',
    '--per-result-file-topk', '12',
    '--select-topk-per-round', '1',
    '--lcb-lambda', '0.75',
    '--pred-lcb-slack', '0.020',
    '--val-batch-size', '32',
    '--notify-discord',
]
print(' '.join(cmd))
subprocess.run(cmd, cwd=ROOT, check=True)
"""
        ),
        code(
            """
summary = pd.read_csv(OUT / 'stats' / '06_robust_proxy_summary.csv')
display(summary[['round','label','source','role','mean_score','std_score','proxy_lcb_score','pred_full_score','known_full_score','judger_score']].head(20))

selected = pd.read_csv(OUT / 'stats' / '06_selected_policy_full.csv')
display(selected[['round','label','source','role','mean_score','std_score','proxy_lcb_score','pred_full_score','map50','map50_95','score','eval_scope']])

print((OUT / '06_robust_proxy_judger_report.md').read_text())
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
