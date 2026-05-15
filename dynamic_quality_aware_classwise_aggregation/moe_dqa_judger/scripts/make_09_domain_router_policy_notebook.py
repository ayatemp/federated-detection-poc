#!/usr/bin/env python3
"""Create notebook 09 for the domain-router policy."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "09_domain_router_policy.ipynb"


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 09 Domain Router Policy

08 showed that the plateau candidates trade off different day/night domains.
This notebook builds an oracle domain-router policy from those results.  It is
not the final deployable method, but it tells us whether dynamic domain-specific
selection has real headroom.
"""
        ),
        code(
            """
from pathlib import Path
import pandas as pd

ROOT = Path('/app/Object_Detection')
PROJECT = ROOT / 'dynamic_quality_aware_classwise_aggregation' / 'moe_dqa_judger'
OUT = PROJECT / 'output' / '09_domain_router_policy'
OUT
"""
        ),
        code(
            """
import subprocess, sys

cmd = [
    sys.executable,
    str(PROJECT / 'scripts' / 'run_09_domain_router_policy.py'),
    '--workspace-root', str(OUT),
    '--min-total-score', '0.5735',
    '--notify-discord',
]
print(' '.join(cmd))
subprocess.run(cmd, cwd=ROOT, check=True)
"""
        ),
        code(
            """
policy = pd.read_csv(OUT / 'stats' / '09_domain_router_policy.csv')
display(policy[['domain','selected_label','selected_score','incumbent_score','delta_score','selected_map50','delta_map50']])

summary = pd.read_csv(OUT / 'stats' / '09_domain_router_summary.csv')
display(summary)

print((OUT / '09_domain_router_policy_report.md').read_text())
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
