#!/usr/bin/env python3
"""Create notebook 01 for the clean MOE x DQA experiment workspace."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "01_dqa_fedmox_yolo_full.ipynb"


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

MOE_DQA_ROOT = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation" / "moe_dqa"
SCENE_DQA_ROOT = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation" / "scene_daynight_dqa"
RUNNER = SCENE_DQA_ROOT / "scripts" / "run_scene_daynight_dqa_18_client_balanced_single_injection_dqamox.py"
WORKSPACE = MOE_DQA_ROOT / "output" / "01_dqa_fedmox_yolo_full"
LOG_DIR = MOE_DQA_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

print("REPO_ROOT:", REPO_ROOT)
print("RUNNER:", RUNNER, RUNNER.exists())
print("WORKSPACE:", WORKSPACE)
'''

    design_code = r'''
design = pd.DataFrame(
    [
        {
            "block": "comparison",
            "setting": "same workspace, fixed final evaluation",
            "detail": "warmup, warmup+server repair, and full DQA-MoE are reported together.",
        },
        {
            "block": "FL length",
            "setting": "50 rounds",
            "detail": "FedMoX-style communication length; no early stop in this notebook.",
        },
        {
            "block": "client sampling",
            "setting": "2 / 6 clients per round",
            "detail": "client_sampling_ratio=0.333 with a fixed seed.",
        },
        {
            "block": "client teacher",
            "setting": "persistent local EMA",
            "detail": "Each client keeps one local EMA teacher; server anchor is not stored on clients.",
        },
        {
            "block": "model",
            "setting": "YOLO with latent MoE head/router",
            "detail": "No full detector bank; MoE lives in the head/router path.",
        },
        {
            "block": "DQA role",
            "setting": "responsibility aggregation",
            "detail": "DQA controls client contribution and expert/router residual blending.",
        },
    ]
)
display(design)
'''

    command_code = r'''
FULL_CMD = [
    sys.executable,
    str(RUNNER),
    "--workspace-root", str(WORKSPACE),
    "--warmup-epochs", "50",
    "--repair-baseline-rounds", "50",
    "--phase1-rounds", "35",
    "--phase2-rounds", "15",
    "--client-sampling-ratio", "0.333",
    "--client-sampling-seed", "20260513",
    "--use-local-ema-teacher",
    "--num-experts", "4",
    "--expert-count", "4",
    "--top-k", "1",
    "--router-temperature", "1.15",
    "--router-balance-weight", "0.03",
    "--router-entropy-weight", "0.002",
    "--router-specialization-map", "hybrid_dqa4",
    "--router-specialization-weight", "0.10",
    "--router-specialization-max-weight", "0.12",
    "--router-specialization-min-quality", "0.50",
    "--router-specialization-min-boxes", "300",
    "--dqa-server-anchor", "0.65",
    "--dqa-min-server-alpha", "0.60",
    "--dqa-residual-blend", "0.10",
    "--dqa-moe-expert-blend", "0.08",
    "--dqa-moe-router-blend", "0.05",
    "--phase1-train-scope", "all",
    "--phase1-repair-train-scope", "all",
    "--phase1-client-epochs", "1",
    "--phase1-client-lr", "0.00025",
    "--phase1-source-repeat", "3",
    "--phase1-pseudo-repeat", "1",
    "--phase1-loss-box", "0.002",
    "--phase2-train-scope", "all",
    "--phase2-repair-train-scope", "all",
    "--phase2-client-epochs", "1",
    "--phase2-client-lr", "0.00018",
    "--phase2-source-repeat", "2",
    "--phase2-pseudo-repeat", "1",
    "--phase2-loss-box", "0.003",
    "--server-repair-epochs", "1",
    "--server-repair-lr", "0.0007",
    "--server-repair-loss-box", "0.05",
    "--client-limit", "1500",
    "--max-images-per-client", "0",
    "--batch-size", "80",
    "--val-batch-size", "32",
    "--workers", "48",
    "--gpus", "2",
    "--evaluate",
    "--no-eval-plots",
    "--notify-start",
    "--notify-end",
    "--notify-progress",
    "--notify-first-progress-hours", "3.0",
    "--notify-progress-interval-hours", "3.0",
    "--target-map50", "0.60",
]

print(" ".join(FULL_CMD))
'''

    setup_check_code = r'''
def without_notify_flags(cmd: list[str]) -> list[str]:
    out: list[str] = []
    skip_next = False
    value_flags = {"--notify-first-progress-hours", "--notify-progress-interval-hours"}
    bare_flags = {"--notify", "--notify-start", "--notify-end", "--notify-progress"}
    for item in cmd:
        if skip_next:
            skip_next = False
            continue
        if item in bare_flags:
            continue
        if item in value_flags:
            skip_next = True
            continue
        out.append(item)
    return out


SETUP_CMD = [*without_notify_flags(FULL_CMD), "--setup-only", "--dry-run", "--no-progress"]

print(" ".join(SETUP_CMD))
subprocess.run(SETUP_CMD, cwd=REPO_ROOT, check=True)
'''

    run_code = r'''
# Full run guard.
# Change this to True when you want to start the no-early-stop full protocol.
RUN_FULL = False

if not RUN_FULL:
    print("RUN_FULL is False. Set it to True to start the full 50-round experiment.")
else:
    LOG_PATH = LOG_DIR / f"01_dqa_fedmox_yolo_full_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    print("log:", LOG_PATH)
    print(" ".join(FULL_CMD))
    with LOG_PATH.open("w", encoding="utf-8") as log:
        proc = subprocess.run(FULL_CMD, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    print("returncode:", proc.returncode)
    tail = LOG_PATH.read_text(encoding="utf-8", errors="replace")[-12000:]
    print(tail)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
'''

    results_code = r'''
metrics_path = WORKSPACE / "stats" / "18_client_balanced_single_injection_dqamox_final_metrics.csv"
split_path = WORKSPACE / "stats" / "18_client_balanced_single_injection_dqamox_split_metrics.csv"
report_path = WORKSPACE / "18_client_balanced_single_injection_dqamox_report.md"

print("metrics:", metrics_path, metrics_path.exists())
print("splits:", split_path, split_path.exists())
print("report:", report_path, report_path.exists())

if metrics_path.exists():
    metrics = pd.read_csv(metrics_path)
    display(metrics)

if split_path.exists():
    splits = pd.read_csv(split_path)
    total_rows = splits[splits["split"].astype(str).str.contains("total", na=False)]
    hard_rows = splits[splits["split"].astype(str).str.contains("highway_night", na=False)]
    display(pd.concat([total_rows, hard_rows], ignore_index=True))
'''

    notify_code = r'''
if metrics_path.exists():
    sys.path.insert(0, str(REPO_ROOT))
    from notebook_notify import notify_discord

    metrics = pd.read_csv(metrics_path)
    message = metrics.to_markdown(index=False)
    notify_discord(
        message,
        title="MOE x DQA 01 finished",
        context={
            "workspace": str(WORKSPACE),
            "metrics": str(metrics_path),
            "splits": str(split_path),
        },
        fail_silently=True,
    )
else:
    print("No metrics yet; run the full experiment first.")
'''

    notebook = {
        "cells": [
            markdown_cell(
                "# 01 DQA-FedMoX-YOLO-Full\n\n"
                "This notebook is the clean `MOE x DQA` starting point.  It intentionally runs a full fixed protocol, "
                "not an early-stop loop.\n\n"
                "The same workspace reports three comparisons:\n\n"
                "1. `warmup_global`: source-GT warmup only\n"
                "2. `warmup_server_repair_final`: warmup plus matched source-GT server repair\n"
                "3. `latent_dqamox_final_repair`: the proposed DQA-routed MoE YOLO full run\n"
            ),
            markdown_cell(
                "## Implementation Notes\n\n"
                "- Client-side storage stays close to FedSTO: each client keeps one persistent local EMA teacher.\n"
                "- The server anchor remains on the server side and is used as fallback/comparison, not as a third client model.\n"
                "- MoE is not a bank of full YOLO detectors.  It uses the existing latent MoE head/router path.\n"
                "- The protocol uses 50 FL rounds with 33% client sampling and fixed final evaluation.\n"
                "- `RUN_FULL` is guarded below so opening the notebook does not accidentally launch a day-long run.\n"
            ),
            code_cell(setup_code.strip() + "\n"),
            markdown_cell("## Design"),
            code_cell(design_code.strip() + "\n"),
            markdown_cell("## Full Command"),
            code_cell(command_code.strip() + "\n"),
            markdown_cell("## Setup Check"),
            code_cell(setup_check_code.strip() + "\n"),
            markdown_cell("## Full Run"),
            code_cell(run_code.strip() + "\n"),
            markdown_cell("## Results"),
            code_cell(results_code.strip() + "\n"),
            markdown_cell("## Discord Notification"),
            code_cell(notify_code.strip() + "\n"),
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
