#!/usr/bin/env python3
"""Create notebook 03 for the learned DQA-SoftMoX mix judger."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "03_mix_judger_policy.ipynb"


def code_cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def markdown_cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        markdown_cell(
            """
# 03 Mix Judger Policy

Notebook 02 searched module-wise G/A/S coefficients directly.  This notebook
turns those search traces into a reusable judger: a small model that scores
candidate mixtures from current round features and selects the best body/head/MoE
mix automatically.
"""
        ),
        code_cell(
            """
from pathlib import Path
import json
import pandas as pd

ROOT = Path('/app/Object_Detection')
PROJECT = ROOT / 'dynamic_quality_aware_classwise_aggregation' / 'moe_dqa_judger'
OUT = PROJECT / 'output' / '03_mix_judger_policy'
OUT
"""
        ),
        markdown_cell(
            """
## Train, Select, And Evaluate

The full evaluation is intentionally kept here because the goal is not only to
fit a predictor, but to verify whether its chosen coefficients behave like the
offline optimizer.
"""
        ),
        code_cell(
            """
import subprocess, sys

cmd = [
    sys.executable,
    str(PROJECT / 'scripts' / 'run_03_train_mix_judger.py'),
    '--workspace-root', str(OUT),
    '--rounds', '1,2,3,4,5',
    '--pool-samples', '2200',
    '--observed-templates', '12',
    '--val-batch-size', '32',
    '--evaluate-full',
    '--force',
]
print(' '.join(cmd))
subprocess.run(cmd, cwd=ROOT, check=True)
"""
        ),
        markdown_cell(
            """
## Summaries
"""
        ),
        code_cell(
            """
selected = pd.read_csv(OUT / 'stats' / '03_selected_weights.csv')
display(selected[['round','candidate_id','guard_reason','pred_score','body_g','body_a','body_s','head_g','head_a','head_s','moe_g','moe_a','moe_s','pool_size']])

eval_path = OUT / 'stats' / '03_selected_full_eval.csv'
if eval_path.exists():
    full_eval = pd.read_csv(eval_path)
    display(full_eval[['round','map50','map50_95','precision','recall','score','body_g','body_a','body_s','head_g','head_a','head_s','moe_g','moe_a','moe_s']])

cv = pd.read_csv(OUT / 'stats' / '03_leave_one_round_cv.csv')
display(cv)
"""
        ),
        code_cell(
            """
print((OUT / '03_mix_judger_policy_report.md').read_text())
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
