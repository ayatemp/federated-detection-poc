#!/usr/bin/env python3
"""Create notebook 39: domain-prior router evaluation."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "39_domain_prior_router_eval.ipynb"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def main() -> None:
    run_code = r'''
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "dynamic_quality_aware_classwise_aggregation").exists():
    REPO_ROOT = Path("/app/Object_Detection")

PROJECT_ROOT = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation" / "scene_daynight_dqa"
RUNNER = PROJECT_ROOT / "aggressive_dqamox" / "scripts" / "run_39_domain_prior_router_eval_loop.py"
LOG_DIR = PROJECT_ROOT / "aggressive_dqamox" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"39_domain_prior_router_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"

cmd = [
    sys.executable,
    str(RUNNER),
    "--target-map50", "0.55",
    "--imgsz", "640",
    "--val-batch-size", "16",
]

print(" ".join(cmd))
print("log:", LOG_PATH)
with LOG_PATH.open("w", encoding="utf-8") as log:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
print("returncode:", proc.returncode)
print(LOG_PATH.read_text(encoding="utf-8", errors="replace")[-8000:])
if proc.returncode not in (0, 2):
    raise SystemExit(proc.returncode)
'''

    summary_code = r'''
from __future__ import annotations

import csv
from pathlib import Path

summary_path = PROJECT_ROOT / "aggressive_dqamox" / "reports" / "39_domain_prior_router_eval_loop_summary.csv"
metrics_path = PROJECT_ROOT / "aggressive_dqamox" / "output" / "39_domain_prior_router_eval_loop" / "stats" / "39_domain_prior_router_metrics.csv"

if summary_path.exists():
    for row in csv.DictReader(summary_path.open(encoding="utf-8")):
        print("summary:", row.get("status"), row.get("best_label"), row.get("best_map50"), row.get("best_map50_95"))

if metrics_path.exists():
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    for row in rows:
        if row.get("split") == "domain_prior_total":
            print(row.get("checkpoint_label"), row.get("map50"), row.get("map50_95"), row.get("mode"), row.get("bias"))
'''

    notebook = {
        "cells": [
            markdown_cell(
                "# 39 Domain-Prior Router Evaluation\n\n"
                "This keeps the trained domain-6 DQA-MoX checkpoint from 38, then evaluates a tiny "
                "client/domain router prior.  The intent is FedMoX-like deployment: shared learned MoE "
                "weights plus a small router adapter per client/domain, rather than six full detectors."
            ),
            code_cell(run_code.strip() + "\n"),
            code_cell(summary_code.strip() + "\n"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
