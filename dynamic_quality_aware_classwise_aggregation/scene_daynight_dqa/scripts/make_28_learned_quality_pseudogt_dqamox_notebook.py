#!/usr/bin/env python3
"""Create notebook 28: learned-quality pseudoGT verifier DQA-MoX."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "28_learned_quality_pseudogt_verifier_dqamox.ipynb"


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
RUNNER = PROJECT_ROOT / "scripts" / "run_scene_daynight_dqa_18_client_balanced_single_injection_dqamox.py"
WORKSPACE = PROJECT_ROOT / "aggressive_dqamox" / "output" / "28_learned_quality_pseudogt_verifier_r1"
SOURCE_WORKSPACE = PROJECT_ROOT / "output" / "08_full_latent_dqamox_from_warmup"
WARMUP = SOURCE_WORKSPACE / "checkpoints" / "round000_latent_dqamox_warmup.pt"
QUALITY_MODEL = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation" / "source_calibrated_localization_quality" / "artifacts" / "rscolq_best.joblib"
LOG_DIR = PROJECT_ROOT / "aggressive_dqamox" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"28_learned_quality_pseudogt_verifier_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"

cmd = [
    sys.executable,
    str(RUNNER),
    "--workspace-root", str(WORKSPACE),
    "--source-workspace", str(SOURCE_WORKSPACE),
    "--skip-warmup-training",
    "--warmup-checkpoint", str(WARMUP),
    "--learned-quality-pseudogt",
    "--learned-quality-model", str(QUALITY_MODEL),
    "--phase1-rounds", "2",
    "--phase2-rounds", "0",
    "--client-limit", "1600",
    "--clients", "all",
    "--client-sampling-ratio", "1.0",
    "--phase1-train-scope", "neck_head",
    "--phase1-repair-train-scope", "neck_head",
    "--phase1-client-epochs", "1",
    "--phase1-client-lr", "0.00045",
    "--phase1-source-repeat", "2",
    "--phase1-pseudo-repeat", "1",
    "--phase1-loss-box", "0.001",
    "--server-repair-epochs", "1",
    "--server-repair-lr", "0.00050",
    "--server-repair-loss-box", "0.030",
    "--top-k", "2",
    "--router-temperature", "1.20",
    "--router-balance-weight", "0.03",
    "--router-entropy-weight", "0.002",
    "--dqa-server-anchor", "0.65",
    "--dqa-min-server-alpha", "0.55",
    "--dqa-residual-blend", "0.06",
    "--dqa-temperature", "0.70",
    "--dqa-uniform-mix", "0.10",
    "--dqa-classwise-blend", "0.35",
    "--expert-keep-fraction", "0.55",
    "--expert-max-class-fraction", "0.24",
    "--actual-max-class-fraction", "0.34",
    "--load-bias-strength", "0.45",
    "--min-stability", "0.58",
    "--min-score", "0.12",
    "--max-boxes-per-image", "10",
    "--max-class-fraction", "0.50",
    "--min-class-keep", "180",
    "--evaluate",
    "--no-eval-plots",
    "--notify-end",
    "--target-map50", "0.60",
    "--estimated-warmup-minutes", "0",
    "--estimated-phase1-round-minutes", "24",
    "--estimated-eval-minutes", "55",
]

print(" ".join(cmd))
print("log:", LOG_PATH)
with LOG_PATH.open("w", encoding="utf-8") as log:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
print("returncode:", proc.returncode)
if proc.returncode != 0:
    print(LOG_PATH.read_text(encoding="utf-8", errors="replace")[-4000:])
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

metrics_path = WORKSPACE / "stats" / "18_client_balanced_single_injection_dqamox_final_metrics.csv"
quality_paths = sorted((WORKSPACE / "stats").glob("28_round*_learned_quality_pseudogt_stats.csv"))
split_path = WORKSPACE / "stats" / "18_client_balanced_single_injection_dqamox_split_metrics.csv"

rows = list(csv.DictReader(metrics_path.open(encoding="utf-8"))) if metrics_path.exists() else []
best = max(rows, key=lambda r: float(r.get("map50") or 0.0)) if rows else {}
quality_lines = []
for path in quality_paths:
    qrows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not qrows:
        continue
    mean_original = sum(float(r["mean_original_score"]) for r in qrows) / len(qrows)
    mean_learned = sum(float(r["mean_learned_quality"]) for r in qrows) / len(qrows)
    quality_lines.append(f"{path.stem}: original={mean_original:.4f}, learned={mean_learned:.4f}")

message = "\n".join(
    [
        "28 learned-quality pseudoGT verifier finished.",
        f"workspace: {WORKSPACE}",
        f"metrics: {metrics_path}",
        f"best: {best.get('checkpoint_label', '')} map50={best.get('map50', '')} map50_95={best.get('map50_95', '')}",
        "quality:",
        *quality_lines[:6],
    ]
)
print(message)
result = notify_discord(
    message,
    title="DQA-MoX 28 result",
    context={
        "workspace": str(WORKSPACE),
        "metrics_csv": str(metrics_path),
        "split_csv": str(split_path),
        "best_map50": best.get("map50", ""),
        "best_map50_95": best.get("map50_95", ""),
    },
    fail_silently=True,
)
print(result)
'''

    notebook = {
        "cells": [
            markdown_cell(
                "# 28 Learned-Quality PseudoGT Verifier DQA-MoX\n\n"
                "R-SCoLQ/GBDT-style verifier is used only as a pseudoGT box-quality scorer. "
                "It replaces the pseudo box `score` before expert-choice selection and DQA stats; "
                "it is not used as a router teacher."
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
