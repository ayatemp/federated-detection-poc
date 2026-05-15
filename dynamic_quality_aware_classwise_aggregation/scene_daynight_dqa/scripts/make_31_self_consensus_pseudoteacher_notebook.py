#!/usr/bin/env python3
"""Create notebook 31: self-consensus pseudo-teacher DQA-MoX."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "31_self_consensus_pseudoteacher_dqamox.ipynb"


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
RUNNER = PROJECT_ROOT / "scripts" / "run_scene_daynight_dqa_18_client_balanced_single_injection_dqamox.py"
WORKSPACE = PROJECT_ROOT / "aggressive_dqamox" / "output" / "31_self_consensus_pseudoteacher_dqamox"
SOURCE_WORKSPACE = PROJECT_ROOT / "output" / "08_full_latent_dqamox_from_warmup"
WARMUP = SOURCE_WORKSPACE / "checkpoints" / "round000_latent_dqamox_warmup.pt"
AGG_OUTPUT = PROJECT_ROOT / "aggressive_dqamox" / "output"
TEACHERS = [
    WARMUP,
    AGG_OUTPUT / "25_paper_round_until_target" / "25a_fedmox50_sto20_30_top1" / "checkpoints" / "latent_dqamox_p1_round001_server_repair.pt",
    AGG_OUTPUT / "27_research_notebook_until_060" / "27e_probe_clean_day_expert_anchor_r2" / "checkpoints" / "latent_dqamox_p1_round002_server_repair.pt",
    AGG_OUTPUT / "27_research_notebook_until_060" / "27g_probe_moe_head_only_router_r1" / "checkpoints" / "latent_dqamox_p1_round001_server_repair.pt",
]
missing = [str(path) for path in [RUNNER, WARMUP, *TEACHERS] if not path.exists()]
if missing:
    raise FileNotFoundError("\n".join(missing))

LOG_DIR = PROJECT_ROOT / "aggressive_dqamox" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"31_self_consensus_pseudoteacher_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"

cmd = [
    sys.executable,
    str(RUNNER),
    "--workspace-root", str(WORKSPACE),
    "--source-workspace", str(SOURCE_WORKSPACE),
    "--skip-warmup-training",
    "--warmup-checkpoint", str(WARMUP),
    "--pseudo-teacher-checkpoints", ",".join(str(path) for path in TEACHERS),
    "--pseudo-teacher-separate-model-views",
    "--phase1-rounds", "1",
    "--phase2-rounds", "0",
    "--client-limit", "1600",
    "--clients", "all",
    "--client-sampling-ratio", "1.0",
    "--phase1-train-scope", "neck_head",
    "--phase1-repair-train-scope", "neck_head",
    "--phase1-client-epochs", "1",
    "--phase1-client-lr", "0.00035",
    "--phase1-source-repeat", "3",
    "--phase1-pseudo-repeat", "1",
    "--phase1-loss-box", "0.0004",
    "--server-repair-epochs", "1",
    "--server-repair-lr", "0.00045",
    "--server-repair-loss-box", "0.020",
    "--top-k", "2",
    "--router-temperature", "1.15",
    "--router-balance-weight", "0.03",
    "--router-entropy-weight", "0.002",
    "--dqa-server-anchor", "0.75",
    "--dqa-min-server-alpha", "0.65",
    "--dqa-residual-blend", "0.04",
    "--dqa-temperature", "0.70",
    "--dqa-uniform-mix", "0.10",
    "--dqa-classwise-blend", "0.30",
    "--expert-keep-fraction", "0.50",
    "--expert-max-class-fraction", "0.24",
    "--actual-max-class-fraction", "0.30",
    "--load-bias-strength", "0.35",
    "--conf-thres", "0.10",
    "--nms-iou-thres", "0.65",
    "--match-iou", "0.55",
    "--min-views", "2",
    "--min-models", "2",
    "--min-stability", "0.50",
    "--min-score", "0.07",
    "--max-boxes-per-image", "12",
    "--max-class-fraction", "0.50",
    "--min-class-keep", "150",
    "--evaluate",
    "--no-eval-plots",
    "--notify-start",
    "--notify-end",
    "--target-map50", "0.55",
    "--estimated-warmup-minutes", "0",
    "--estimated-phase1-round-minutes", "35",
    "--estimated-eval-minutes", "55",
]

print(" ".join(cmd))
print("log:", LOG_PATH)
with LOG_PATH.open("w", encoding="utf-8") as log:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
print("returncode:", proc.returncode)
print(LOG_PATH.read_text(encoding="utf-8", errors="replace")[-6000:])
if proc.returncode != 0:
    raise SystemExit(proc.returncode)
'''

    summary_code = r'''
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebook_notify import notify_discord

metrics_path = WORKSPACE / "stats" / "18_client_balanced_single_injection_dqamox_final_metrics.csv"
split_path = WORKSPACE / "stats" / "18_client_balanced_single_injection_dqamox_split_metrics.csv"
pseudo_path = WORKSPACE / "stats" / "03_round001_pseudo_label_stats.csv"
pseudo_json = WORKSPACE / "stats" / "03_round001_pseudo_label_stats.json"
summary_path = PROJECT_ROOT / "aggressive_dqamox" / "reports" / "31_self_consensus_pseudoteacher_summary.csv"
notebook_path = PROJECT_ROOT / "notebooks" / "31_self_consensus_pseudoteacher_dqamox.ipynb"

rows = list(csv.DictReader(metrics_path.open(encoding="utf-8"))) if metrics_path.exists() else []
best = max(rows, key=lambda r: float(r.get("map50") or 0.0)) if rows else {}
warm = next((r for r in rows if r.get("checkpoint_label") == "warmup_global"), {})
repair = next((r for r in rows if r.get("checkpoint_label") == "warmup_server_repair_final"), {})
final_repair = next((r for r in rows if r.get("checkpoint_label") == "latent_dqamox_final_repair"), {})
pseudo_rows = list(csv.DictReader(pseudo_path.open(encoding="utf-8"))) if pseudo_path.exists() else []
pseudo_boxes = sum(int(float(r.get("pseudo_boxes_kept") or 0)) for r in pseudo_rows)
pseudo_images = sum(int(float(r.get("pseudo_images_kept") or 0)) for r in pseudo_rows)
mean_stability = (
    sum(float(r.get("mean_stability") or 0.0) for r in pseudo_rows) / len(pseudo_rows)
    if pseudo_rows else 0.0
)
mean_score = (
    sum(float(r.get("mean_score") or 0.0) for r in pseudo_rows) / len(pseudo_rows)
    if pseudo_rows else 0.0
)
status = "target_reached" if float(best.get("map50") or 0.0) >= 0.55 else "below_target"

summary_row = {
    "trial": "31_self_consensus_pseudoteacher_dqamox",
    "status": status,
    "best_label": best.get("checkpoint_label", ""),
    "best_map50": best.get("map50", ""),
    "best_map50_95": best.get("map50_95", ""),
    "warmup_map50": warm.get("map50", ""),
    "repair_map50": repair.get("map50", ""),
    "final_repair_map50": final_repair.get("map50", ""),
    "pseudo_boxes": pseudo_boxes,
    "pseudo_images": pseudo_images,
    "pseudo_mean_stability": f"{mean_stability:.6f}",
    "pseudo_mean_score": f"{mean_score:.6f}",
    "target_map50": "0.55",
    "workspace": str(WORKSPACE),
    "notebook": str(notebook_path),
    "metrics_csv": str(metrics_path),
    "pseudo_csv": str(pseudo_path),
    "finished_utc": datetime.now(timezone.utc).isoformat(),
    "rationale": "Self-only checkpoint consensus verifier: pseudo boxes must be supported by at least two prior self checkpoints before DQA-MoX client training.",
}
fields = list(summary_row)
summary_path.parent.mkdir(parents=True, exist_ok=True)
write_header = not summary_path.exists()
with summary_path.open("a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        writer.writeheader()
    writer.writerow(summary_row)

client_lines = []
for row in pseudo_rows[:6]:
    client_lines.append(
        f"- {row.get('client')}: boxes={row.get('pseudo_boxes_kept')} images={row.get('pseudo_images_kept')} "
        f"stability={row.get('mean_stability')} score={row.get('mean_score')}"
    )

message = "\n".join(
    [
        "31 self-consensus pseudo-teacher DQA-MoX finished.",
        f"status: {status}",
        f"best: {best.get('checkpoint_label', '')} mAP50={best.get('map50', '')} mAP50:95={best.get('map50_95', '')}",
        f"warmup: mAP50={warm.get('map50', '')}; repair: mAP50={repair.get('map50', '')}; final_repair: mAP50={final_repair.get('map50', '')}",
        f"pseudo consensus: boxes={pseudo_boxes}, images={pseudo_images}, mean_stability={mean_stability:.4f}, mean_score={mean_score:.4f}",
        "client pseudo stats:",
        *client_lines,
        f"workspace: {WORKSPACE}",
        f"metrics: {metrics_path}",
    ]
)
print(message)
result = notify_discord(
    message,
    title="DQA-MoX 31 result",
    context={
        "status": status,
        "workspace": str(WORKSPACE),
        "metrics_csv": str(metrics_path),
        "split_csv": str(split_path),
        "pseudo_csv": str(pseudo_path),
        "pseudo_json": str(pseudo_json),
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
                "# 31 Self-Consensus Pseudo-Teacher DQA-MoX\n\n"
                "This run keeps the teacher self-only. It turns prior DQA-MoX checkpoints "
                "into separate pseudo-label views, then keeps boxes supported by at least "
                "two self checkpoints before client-side DQA-MoX training."
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
