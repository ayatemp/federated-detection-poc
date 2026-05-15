#!/usr/bin/env python3
"""Create notebook 14 for the domain-calibrated delta judger."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "14_domain_calibrated_delta_judger.ipynb"


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 14 Domain-Calibrated Delta Judger

13 learned absolute domain scores and struggled because domain difficulty
dominated the target.  This notebook learns the incumbent-relative delta for
each candidate and then selects the checkpoint/router per domain.
"""
        ),
        code(
            """
from pathlib import Path
import pandas as pd

ROOT = Path('/app/Object_Detection')
PROJECT = ROOT / 'dynamic_quality_aware_classwise_aggregation' / 'moe_dqa_judger'
OUT = PROJECT / 'output' / '14_domain_calibrated_delta_judger'
OUT
"""
        ),
        code(
            """
import subprocess, sys

cmd = [
    sys.executable,
    str(PROJECT / 'scripts' / 'run_14_domain_calibrated_delta_judger.py'),
    '--workspace-root', str(OUT),
    '--max-total-drop', '0.0011',
    '--total-drop-penalty', '0.15',
    '--notify-discord',
]
print(' '.join(cmd))
subprocess.run(cmd, cwd=ROOT, check=True)
"""
        ),
        code(
            """
summary = pd.read_csv(OUT / 'stats' / '14_policy_summary.csv')
display(summary)

policy = pd.read_csv(OUT / 'stats' / '14_policy_rows.csv')
display(policy[['selected_policy','domain','label','delta_score','pred_delta_score','domain_score','total_score']])

print((OUT / '14_domain_calibrated_delta_judger_report.md').read_text())
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
