#!/usr/bin/env python3
"""Create notebook 29: self-only Soft-Mixture output MoE from notebook 28."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "29_self_soft_mixture_moe_from_28.ipynb"


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
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "dynamic_quality_aware_classwise_aggregation").exists():
    REPO_ROOT = Path("/app/Object_Detection")

PROJECT_ROOT = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation" / "scene_daynight_dqa"
RUNNER = PROJECT_ROOT / "aggressive_dqamox" / "scripts" / "build_eval_29_self_soft_mixture_moe.py"
INPUT_WORKSPACE = PROJECT_ROOT / "aggressive_dqamox" / "output" / "28_learned_quality_pseudogt_verifier_r1"
WORKSPACE = PROJECT_ROOT / "aggressive_dqamox" / "output" / "29_self_soft_mixture_moe_from_28"
LOG_DIR = PROJECT_ROOT / "aggressive_dqamox" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"29_self_soft_mixture_moe_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"

cmd = [
    sys.executable,
    str(RUNNER),
    "--input-workspace", str(INPUT_WORKSPACE),
    "--workspace", str(WORKSPACE),
    "--target-map50", "0.55",
    "--previous-warmup-map50", "0.460",
    "--gate-images", "180",
    "--min-gate-gain", "0.006",
    "--imgsz", "1024",
    "--batch-size", "1",
    "--conf-thres", "0.001",
    "--merge-iou", "0.50",
]

print(" ".join(cmd))
print("log:", LOG_PATH)
with LOG_PATH.open("w", encoding="utf-8") as log:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
print("returncode:", proc.returncode)
tail = LOG_PATH.read_text(encoding="utf-8", errors="replace")[-6000:]
print(tail)
if proc.returncode not in (0, 2):
    raise SystemExit(proc.returncode)
'''

    summary_code = r'''
from __future__ import annotations

import csv
from pathlib import Path

metrics_path = WORKSPACE / "stats" / "29_self_soft_mixture_moe_metrics.csv"
manifest_path = WORKSPACE / "stats" / "29_self_soft_mixture_moe_manifest.json"
summary_path = PROJECT_ROOT / "aggressive_dqamox" / "reports" / "29_self_soft_mixture_moe_summary.csv"

rows = list(csv.DictReader(metrics_path.open(encoding="utf-8"))) if metrics_path.exists() else []
total_rows = [r for r in rows if r.get("split") == "scene_daynight_total"]
gate_rows = [r for r in total_rows if str(r.get("phase", "")).startswith("gate_")]
full_rows = [r for r in total_rows if str(r.get("phase", "")).startswith("full_")]
best_gate = max(gate_rows, key=lambda r: (float(r.get("map50") or 0), float(r.get("map50_95") or 0))) if gate_rows else {}
best_full = max(full_rows, key=lambda r: (float(r.get("map50") or 0), float(r.get("map50_95") or 0))) if full_rows else {}

print("metrics:", metrics_path)
print("manifest:", manifest_path)
print("summary:", summary_path)
print("best gate:", best_gate.get("candidate"), best_gate.get("map50"), best_gate.get("map50_95"))
if best_full:
    print("best full:", best_full.get("candidate"), best_full.get("map50"), best_full.get("map50_95"))
else:
    print("full evaluation was skipped by the gate rule")
'''

    notebook = {
        "cells": [
            markdown_cell(
                "# 29 Self-Only Soft-Mixture Output MoE\n\n"
                "FedMoXのSoft-Mixtureを、重み平均ではなく推論時の検出結果MoEとして試す。"
                "使うexpertは28で自分自身が生成した warmup / DQA aggregate / server repair / client specialists だけ。"
                "COCO expertや外部teacherは使わない。gate splitでwarmupを十分超えなければfull評価を打ち切る。"
            ),
            code_cell(run_code.strip() + "\n"),
            code_cell(summary_code.strip() + "\n"),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
