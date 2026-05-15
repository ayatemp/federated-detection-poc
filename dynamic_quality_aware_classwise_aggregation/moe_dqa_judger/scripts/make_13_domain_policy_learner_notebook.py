#!/usr/bin/env python3
"""Create notebook 13 for the domain policy learner."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "13_domain_policy_learner.ipynb"


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 13 Domain Policy Learner

12 showed that weighted soup does not preserve each domain winner's peak.  This
notebook trains a small machine-learning judger to select a checkpoint/policy
conditioned on domain metadata, using only accumulated self-generated scores.
"""
        ),
        code(
            """
from pathlib import Path
import pandas as pd

ROOT = Path('/app/Object_Detection')
PROJECT = ROOT / 'dynamic_quality_aware_classwise_aggregation' / 'moe_dqa_judger'
OUT = PROJECT / 'output' / '13_domain_policy_learner'
OUT
"""
        ),
        code(
            """
import subprocess, sys

cmd = [
    sys.executable,
    str(PROJECT / 'scripts' / 'run_13_domain_policy_learner.py'),
    '--workspace-root', str(OUT),
    '--max-total-drop', '0.0011',
    '--notify-discord',
]
print(' '.join(cmd))
subprocess.run(cmd, cwd=ROOT, check=True)
"""
        ),
        code(
            """
policy = pd.read_csv(OUT / 'stats' / '13_policy_rows.csv')
display(policy[['domain','label','domain_score','pred_domain_score','map50','total_score']])

summary = pd.read_csv(OUT / 'stats' / '13_policy_summary.csv')
display(summary)

print((OUT / '13_domain_policy_learner_report.md').read_text())
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
