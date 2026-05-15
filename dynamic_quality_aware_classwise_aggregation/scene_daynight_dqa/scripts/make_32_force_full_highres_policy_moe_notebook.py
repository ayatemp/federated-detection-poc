#!/usr/bin/env python3
"""Create notebook 32: force full eval for the best high-res routed MoE policy."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "32_force_full_highres_policy_moe.ipynb"


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

import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "dynamic_quality_aware_classwise_aggregation").exists():
    REPO_ROOT = Path("/app/Object_Detection")

PROJECT_ROOT = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation" / "scene_daynight_dqa"
RUNNER = PROJECT_ROOT / "aggressive_dqamox" / "scripts" / "build_eval_30_split_policy_highres_sahi_moe.py"
WORKSPACE = PROJECT_ROOT / "aggressive_dqamox" / "output" / "32_force_full_highres_policy_moe"
LOG_DIR = PROJECT_ROOT / "aggressive_dqamox" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"32_force_full_highres_policy_moe_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"

cmd = [
    sys.executable,
    str(RUNNER),
    "--workspace", str(WORKSPACE),
    "--target-map50", "0.55",
    "--previous-best-map50", "0.52939",
    "--gate-images", "360",
    "--min-gate-gain", "0.0",
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
import sys
from pathlib import Path

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebook_notify import notify_discord

metrics_path = WORKSPACE / "stats" / "30_split_policy_highres_sahi_metrics.csv"
summary_path = PROJECT_ROOT / "aggressive_dqamox" / "reports" / "30_split_policy_highres_sahi_summary.csv"
rows = list(csv.DictReader(metrics_path.open(encoding="utf-8"))) if metrics_path.exists() else []
total_rows = [r for r in rows if r.get("split") == "scene_daynight_total"]
gate_rows = [r for r in total_rows if str(r.get("phase", "")).startswith("gate_")]
full_rows = [r for r in total_rows if str(r.get("phase", "")).startswith("full_")]
best_gate = max(gate_rows, key=lambda r: (float(r.get("map50") or 0), float(r.get("map50_95") or 0))) if gate_rows else {}
best_full = max(full_rows, key=lambda r: (float(r.get("map50") or 0), float(r.get("map50_95") or 0))) if full_rows else {}
status = "target_reached" if float(best_full.get("map50") or 0) >= 0.55 else "below_target"

message = "\n".join(
    [
        "32 force-full high-res routed MoE finished.",
        f"status: {status}",
        f"best gate: {best_gate.get('candidate', '')} mAP50={best_gate.get('map50', '')} mAP50:95={best_gate.get('map50_95', '')}",
        f"best full: {best_full.get('candidate', '')} mAP50={best_full.get('map50', '')} mAP50:95={best_full.get('map50_95', '')}",
        f"workspace: {WORKSPACE}",
        f"metrics: {metrics_path}",
        f"summary: {summary_path}",
    ]
)
print(message)
print(
    notify_discord(
        message,
        title="DQA-MoX 32 result",
        context={
            "status": status,
            "workspace": str(WORKSPACE),
            "metrics_csv": str(metrics_path),
            "best_gate_map50": best_gate.get("map50", ""),
            "best_full_map50": best_full.get("map50", ""),
        },
        fail_silently=True,
    )
)
'''

    notebook = {
        "cells": [
            markdown_cell(
                "# 32 Force Full High-Res Routed MoE\n\n"
                "Notebook 30 stopped before full evaluation because the gate gain was small. "
                "This run forces the best self-only high-resolution routed MoE policy to run on the full set."
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
