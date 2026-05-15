#!/usr/bin/env python3
"""Create notebook 33: self-only counterfactual-view MoE."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "33_counterfactual_view_moe.ipynb"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


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
RUNNER = PROJECT_ROOT / "aggressive_dqamox" / "scripts" / "build_eval_33_counterfactual_view_moe.py"
WORKSPACE = PROJECT_ROOT / "aggressive_dqamox" / "output" / "33_counterfactual_view_moe"
LOG_DIR = PROJECT_ROOT / "aggressive_dqamox" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"33_counterfactual_view_moe_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"

cmd = [
    sys.executable,
    str(RUNNER),
    "--workspace", str(WORKSPACE),
    "--target-map50", "0.55",
    "--previous-best-map50", "0.52939",
    "--gate-images", "720",
    "--min-gate-gain", "0.006",
    "--conf-thres", "0.001",
    "--tile-batch-size", "8",
]

print(" ".join(cmd))
print("log:", LOG_PATH)
with LOG_PATH.open("w", encoding="utf-8") as log:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
print("returncode:", proc.returncode)
print(LOG_PATH.read_text(encoding="utf-8", errors="replace")[-6000:])
if proc.returncode not in (0, 2):
    raise SystemExit(proc.returncode)
'''

    summary_code = r'''
from __future__ import annotations

import csv
from pathlib import Path

metrics_path = WORKSPACE / "stats" / "33_counterfactual_view_moe_metrics.csv"
summary_path = PROJECT_ROOT / "aggressive_dqamox" / "reports" / "33_counterfactual_view_moe_summary.csv"
rows = list(csv.DictReader(metrics_path.open(encoding="utf-8"))) if metrics_path.exists() else []
total_rows = [r for r in rows if r.get("split") == "scene_daynight_total"]
gate_rows = [r for r in total_rows if str(r.get("phase", "")).startswith("gate_")]
full_rows = [r for r in total_rows if str(r.get("phase", "")).startswith("full_")]
best_gate = max(gate_rows, key=lambda r: (float(r.get("map50") or 0), float(r.get("map50_95") or 0))) if gate_rows else {}
best_full = max(full_rows, key=lambda r: (float(r.get("map50") or 0), float(r.get("map50_95") or 0))) if full_rows else {}
reported = best_full or best_gate
status = "target_reached" if float(reported.get("map50") or 0) >= 0.55 else "below_target"

message = "\n".join(
    [
        "33 counterfactual-view MoE finished.",
        f"status: {status}",
        f"best gate: {best_gate.get('candidate', '')} mAP50={best_gate.get('map50', '')} mAP50:95={best_gate.get('map50_95', '')}",
        f"best full: {best_full.get('candidate', '')} mAP50={best_full.get('map50', '')} mAP50:95={best_full.get('map50_95', '')}",
        f"workspace: {WORKSPACE}",
        f"metrics: {metrics_path}",
        f"summary: {summary_path}",
    ]
)
print(message)
'''

    notebook = {
        "cells": [
            markdown_cell(
                "# 33 Counterfactual-View MoE\n\n"
                "Self-only inference-time MoE. The same DQA-MoE checkpoint is evaluated through deterministic "
                "view experts: original image, hflip, night brightness correction, and SAHI-style tiling. "
                "The path/domain router controls which views are mixed."
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
