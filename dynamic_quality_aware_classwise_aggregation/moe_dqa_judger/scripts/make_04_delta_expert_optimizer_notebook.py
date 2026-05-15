#!/usr/bin/env python3
"""Create notebook 04 for anchored delta expert-wise mixing."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "04_delta_expert_optimizer.ipynb"


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 04 Delta Expert Optimizer

This loop goes beyond convex `G/A/S` mixing.  It uses `G` as the parent anchor
and learns how much of the aggregate delta and repair delta to inject:

```text
M = G + alpha(A - G) + beta(S - G)
```

The coefficients are independent for `body`, `head`, `router`, and each MoE
expert.  The purpose is to find a judge that can keep improving across rounds
instead of merely freezing when repeated repair drifts.
"""
        ),
        code(
            """
from pathlib import Path
import pandas as pd

ROOT = Path('/app/Object_Detection')
PROJECT = ROOT / 'dynamic_quality_aware_classwise_aggregation' / 'moe_dqa_judger'
OUT = PROJECT / 'output' / '04_delta_expert_optimizer'
OUT
"""
        ),
        md(
            """
## Run Loop
"""
        ),
        code(
            """
import subprocess, sys

cmd = [
    sys.executable,
    str(PROJECT / 'scripts' / 'run_04_delta_expert_optimizer.py'),
    '--workspace-root', str(OUT),
    '--rounds', '1,2,3,4,5,6',
    '--mini-images', '512',
    '--random-candidates', '10',
    '--surrogate-iterations', '1',
    '--surrogate-pool', '72',
    '--surrogate-evals', '3',
    '--full-eval-topk', '2',
    '--val-batch-size', '32',
    '--notify-discord',
    '--force',
]
print(' '.join(cmd))
subprocess.run(cmd, cwd=ROOT, check=True)
"""
        ),
        md(
            """
## Results
"""
        ),
        code(
            """
best = pd.read_csv(OUT / 'stats' / '04_delta_expert_best_full.csv')
cols = [
    'round','candidate_id','map50','map50_95','precision','recall','score',
    'body_a','body_s','head_a','head_s','router_a','router_s',
    'expert0_a','expert0_s','expert1_a','expert1_s','expert2_a','expert2_s','expert3_a','expert3_s'
]
display(best[cols].head(20))
print((OUT / '04_delta_expert_optimizer_report.md').read_text())
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
