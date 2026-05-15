#!/usr/bin/env python3
"""Create notebook 41: loss-factorized DQA-MoX loop."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "41_loss_factorized_dqamox_loop.ipynb"


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
RUNNER = PROJECT_ROOT / "aggressive_dqamox" / "scripts" / "run_41_loss_factorized_dqamox_loop.py"
LOG_DIR = PROJECT_ROOT / "aggressive_dqamox" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"41_loss_factorized_dqamox_loop_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"

cmd = [
    sys.executable,
    str(RUNNER),
    "--target-map50", "0.55",
    "--client-limit", "3000",
    "--client-sampling-ratio", "0.333",
    "--gpus", "2",
    "--batch-size", "80",
    "--workers", "8",
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

summary_path = PROJECT_ROOT / "aggressive_dqamox" / "reports" / "41_loss_factorized_dqamox_loop_summary.csv"
rows = list(csv.DictReader(summary_path.open(encoding="utf-8"))) if summary_path.exists() else []
for row in rows[-10:]:
    print(
        row.get("trial"),
        row.get("stage"),
        row.get("status"),
        row.get("best_label"),
        row.get("best_map50"),
        row.get("best_map50_95"),
        row.get("gain_vs_warmup"),
    )
'''

    notebook = {
        "cells": [
            markdown_cell(
                "# 41 Loss-Factorized DQA-MoX\n\n"
                "Learning-first, self-only loop after 40.  40 raised recall slightly but lost localization/AP quality.  "
                "This run keeps FedMoX-style online client sampling and explicit domain experts, while DQA changes the "
                "role of pseudoGT: pseudo boxes mainly drive class/objectness/router specialization, and source GT plus "
                "server anchoring protects geometry."
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
