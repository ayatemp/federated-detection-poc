#!/usr/bin/env python3
"""Create the DQA-SoftMoX judger probe notebook."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "01_judger_probe.ipynb"


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
    setup_code = r'''
from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path.cwd().resolve()
if not (REPO_ROOT / "dynamic_quality_aware_classwise_aggregation").exists():
    REPO_ROOT = Path("/app/Object_Detection")

JUDGER_ROOT = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation" / "moe_dqa_judger"
RUNNER = JUDGER_ROOT / "scripts" / "run_01_judger_probe.py"
WORKSPACE = JUDGER_ROOT / "output" / "01_judger_probe"
LOG_DIR = JUDGER_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

print("REPO_ROOT:", REPO_ROOT)
print("RUNNER:", RUNNER, RUNNER.exists())
print("WORKSPACE:", WORKSPACE)
'''

    design_code = r'''
design = pd.DataFrame(
    [
        {
            "block": "goal",
            "setting": "learn automatic G/A/S mixing",
            "detail": "Judger predicts body/head/moe weights over previous global, DQA aggregate, and server repair.",
        },
        {
            "block": "inputs",
            "setting": "reuse 01 artifacts",
            "detail": "Warmup and existing round checkpoints are reused so the judger can be developed without re-running full FL.",
        },
        {
            "block": "module split",
            "setting": "body / head / moe",
            "detail": "Body keeps adaptation, head stays calibrated, MoE keeps specialization.",
        },
        {
            "block": "judger v0",
            "setting": "bootstrap ML model",
            "detail": "Historical round features train a tiny random-forest multi-output regressor for softmix weights.",
        },
        {
            "block": "probe policy",
            "setting": "2 rounds first, then up to 5",
            "detail": "Build/evaluate round1-2 first; if promising, run the same notebook cell for round1-5.",
        },
    ]
)
display(design)
'''

    run_two_code = r'''
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"01_judger_probe_r2_{timestamp}.log"
cmd = [
    sys.executable,
    str(RUNNER),
    "--workspace-root", str(WORKSPACE),
    "--history-rounds", "21",
    "--rounds", "1,2",
    "--force",
]
print(" ".join(cmd))
with log_path.open("w", encoding="utf-8") as log:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write(proc.stdout)
print("returncode:", proc.returncode)
print("log:", log_path)
print(proc.stdout[-4000:])
if proc.returncode != 0:
    raise SystemExit(proc.returncode)
'''

    inspect_two_code = r'''
weights_csv = WORKSPACE / "stats" / "01_judger_softmix_rounds.csv"
weights = pd.read_csv(weights_csv)
display(weights[[
    "round",
    "g_map50",
    "a_proxy_map50",
    "s_map50",
    "repair_gain_vs_a",
    "body_g", "body_a", "body_s",
    "head_g", "head_a", "head_s",
    "moe_g", "moe_a", "moe_s",
]])
print("report:", WORKSPACE / "01_judger_probe_report.md")
'''

    run_five_code = r'''
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"01_judger_probe_r5_{timestamp}.log"
cmd = [
    sys.executable,
    str(RUNNER),
    "--workspace-root", str(WORKSPACE),
    "--history-rounds", "21",
    "--rounds", "1,2,3,4,5",
    "--force",
]
print(" ".join(cmd))
with log_path.open("w", encoding="utf-8") as log:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write(proc.stdout)
print("returncode:", proc.returncode)
print("log:", log_path)
print(proc.stdout[-4000:])
if proc.returncode != 0:
    raise SystemExit(proc.returncode)

weights = pd.read_csv(WORKSPACE / "stats" / "01_judger_softmix_rounds.csv")
display(weights[[
    "round",
    "g_map50",
    "a_proxy_map50",
    "s_map50",
    "repair_gain_vs_a",
    "body_g", "body_a", "body_s",
    "head_g", "head_a", "head_s",
    "moe_g", "moe_a", "moe_s",
]])
'''

    eval_note = r'''
Optional total-split evaluation is separated because each checkpoint evaluation is much slower than checkpoint mixing.
Run this cell when the two/five-round weight table looks sane.
'''

    eval_code = r'''
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"01_judger_probe_eval_total_{timestamp}.log"
cmd = [
    sys.executable,
    str(RUNNER),
    "--workspace-root", str(WORKSPACE),
    "--history-rounds", "21",
    "--rounds", "1,2",
    "--evaluate",
    "--eval-splits", "total",
    "--val-batch-size", "32",
]
print(" ".join(cmd))
with log_path.open("w", encoding="utf-8") as log:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write(proc.stdout)
print("returncode:", proc.returncode)
print("log:", log_path)
print(proc.stdout[-4000:])
if proc.returncode != 0:
    raise SystemExit(proc.returncode)

eval_csv = WORKSPACE / "stats" / "01_judger_softmix_eval.csv"
if eval_csv.exists():
    display(pd.read_csv(eval_csv))
'''

    nb = {
        "cells": [
            markdown_cell("# 01 DQA-SoftMoX Judger Probe\n\nBuild the first automatic module-wise softmix judger for `G_t / A_t / S_t`."),
            code_cell(setup_code),
            markdown_cell("## Design"),
            code_cell(design_code),
            markdown_cell("## Round 1-2 Probe\n\nThis is the fast sanity pass. It builds `M_t` checkpoints for the first two rounds."),
            code_cell(run_two_code),
            code_cell(inspect_two_code),
            markdown_cell("## Extend To Five Rounds\n\nRun this when round 1-2 looks sane. This still reuses existing artifacts; it does not start a new FL training loop."),
            code_cell(run_five_code),
            markdown_cell("## Optional Total Evaluation\n\n" + eval_note),
            code_cell(eval_code),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
