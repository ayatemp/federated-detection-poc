#!/usr/bin/env python3
"""Build notebook 27j: distill a model-level MoE pseudo-teacher into DQA-MoX."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
AGG_ROOT = SCRIPT_PATH.parents[1]
SCENE_ROOT = SCRIPT_PATH.parents[2]
RESEARCH_ROOT = SCRIPT_PATH.parents[3]
REPO_ROOT = SCRIPT_PATH.parents[4]
RUNNER = SCENE_ROOT / "scripts" / "run_scene_daynight_dqa_18_client_balanced_single_injection_dqamox.py"
NOTEBOOK_ROOT = AGG_ROOT / "notebooks" / "research_loop_until_060"
REPORT_ROOT = AGG_ROOT / "reports"
OUTPUT_ROOT = AGG_ROOT / "output" / "27_research_notebook_until_060"
WORKSPACE = OUTPUT_ROOT / "27j_moe_ensemble_teacher_distill_r1"
NOTEBOOK_PATH = NOTEBOOK_ROOT / "004_27j_moe_ensemble_teacher_distill_r1.ipynb"
SUMMARY_PATH = REPORT_ROOT / "27_research_loop_mAP_summary.csv"


WARMUP_CHECKPOINT = (
    AGG_ROOT
    / "output"
    / "25_paper_round_until_target"
    / "25a_fedmox50_sto20_30_top1"
    / "checkpoints"
    / "latent_dqamox_p1_round001_server_repair.pt"
)

RESEARCH_27 = AGG_ROOT / "output" / "27_research_notebook_until_060"
PSEUDO_TEACHERS = [
    SCENE_ROOT / "output" / "08_full_latent_dqamox_from_warmup" / "checkpoints" / "round000_latent_dqamox_warmup.pt",
    WARMUP_CHECKPOINT,
    RESEARCH_27 / "27d_probe_teacher_residual_mixpl_r2" / "checkpoints" / "latent_dqamox_p1_round002_server_repair.pt",
    RESEARCH_27 / "27e_probe_clean_day_expert_anchor_r2" / "checkpoints" / "latent_dqamox_p1_round002_server_repair.pt",
    RESEARCH_27 / "27g_probe_moe_head_only_router_r1" / "checkpoints" / "latent_dqamox_p1_round001_server_repair.pt",
    RESEARCH_27 / "27g_probe_moe_head_only_router_r1" / "checkpoints" / "latent_dqamox_p1_round001_client1_highway_night.pt",
    RESEARCH_27 / "27g_probe_moe_head_only_router_r1" / "checkpoints" / "latent_dqamox_p1_round001_client3_citystreet_night.pt",
    RESEARCH_27 / "27g_probe_moe_head_only_router_r1" / "checkpoints" / "latent_dqamox_p1_round001_client5_residential_night.pt",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def main() -> int:
    missing = [path for path in [WARMUP_CHECKPOINT, *PSEUDO_TEACHERS] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint: {missing[0]}")

    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    log_path = WORKSPACE / "logs" / "27j_moe_ensemble_teacher_distill_r1_train.log"
    metrics_path = WORKSPACE / "stats" / "18_client_balanced_single_injection_dqamox_final_metrics.csv"
    teachers = ",".join(str(path.resolve()) for path in PSEUDO_TEACHERS)
    cmd = [
        sys.executable,
        str(RUNNER),
        "--workspace-root",
        str(WORKSPACE),
        "--repair-baseline-rounds",
        "0",
        "--source-workspace",
        str(SCENE_ROOT / "output" / "08_full_latent_dqamox_from_warmup"),
        "--source-repair-baseline-rounds",
        "30",
        "--target-map50",
        "0.60",
        "--skip-warmup-training",
        "--warmup-checkpoint",
        str(WARMUP_CHECKPOINT),
        "--pseudo-teacher-checkpoints",
        teachers,
        "--num-experts",
        "4",
        "--top-k",
        "2",
        "--router-temperature",
        "0.90",
        "--router-balance-weight",
        "0.040",
        "--router-entropy-weight",
        "0.0002",
        "--dqa-client-balance-stats",
        "--dqa-client-balance-target",
        "median",
        "--dqa-client-balance-max-scale",
        "4.0",
        "--load-bias-strength",
        "0.22",
        "--batch-size",
        "80",
        "--workers",
        "8",
        "--gpus",
        "2",
        "--client-limit",
        "800",
        "--client-sampling-ratio",
        "1.000",
        "--client-sampling-seed",
        "270613",
        "--phase1-rounds",
        "1",
        "--phase2-rounds",
        "0",
        "--phase1-train-scope",
        "moe_head",
        "--phase1-repair-train-scope",
        "moe_head",
        "--phase1-client-epochs",
        "1",
        "--phase1-client-lr",
        "0.00032",
        "--phase1-source-repeat",
        "4",
        "--phase1-pseudo-repeat",
        "2",
        "--phase1-loss-box",
        "0.00008",
        "--server-repair-epochs",
        "1",
        "--server-repair-lr",
        "0.00020",
        "--server-repair-loss-box",
        "0.0005",
        "--dqa-temperature",
        "0.80",
        "--dqa-uniform-mix",
        "0.10",
        "--dqa-classwise-blend",
        "0.28",
        "--dqa-stability-lambda",
        "0.55",
        "--dqa-server-anchor",
        "0.70",
        "--dqa-min-server-alpha",
        "0.64",
        "--dqa-residual-blend",
        "0.08",
        "--curriculum-start-round",
        "2",
        "--expert-keep-fraction",
        "0.90",
        "--expert-max-class-fraction",
        "0.36",
        "--actual-max-class-fraction",
        "0.46",
        "--min-score",
        "0.14",
        "--min-stability",
        "0.46",
        "--max-boxes-per-image",
        "14",
        "--imgsz",
        "640",
        "--conf-thres",
        "0.20",
        "--nms-iou-thres",
        "0.65",
        "--match-iou",
        "0.58",
        "--min-views",
        "2",
        "--max-images-per-client",
        "0",
        "--master-port",
        "39400",
        "--evaluate",
        "--classwise",
        "--no-eval-plots",
        "--force",
        "--force-pseudo",
        "--notify-start",
        "--notify-end",
    ]

    run_cell = f"""
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path({str(REPO_ROOT)!r})
WORKSPACE = Path({str(WORKSPACE)!r})
LOG_PATH = Path({str(log_path)!r})
CMD = {cmd!r}

WORKSPACE.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
(WORKSPACE / "stats").mkdir(parents=True, exist_ok=True)
(WORKSPACE / "stats" / "27j_notebook_command.json").write_text(
    json.dumps({{"command": CMD}}, indent=2, ensure_ascii=False) + "\\n",
    encoding="utf-8",
)
print(" ".join(CMD))
with LOG_PATH.open("w", encoding="utf-8") as log:
    proc = subprocess.run(CMD, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
print("returncode", proc.returncode)
print("log", LOG_PATH)
if proc.returncode != 0:
    raise SystemExit(proc.returncode)
"""

    result_cell = f"""
import csv
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path({str(REPO_ROOT)!r})
WORKSPACE = Path({str(WORKSPACE)!r})
METRICS_PATH = Path({str(metrics_path)!r})
SUMMARY_PATH = Path({str(SUMMARY_PATH)!r})
NOTEBOOK_PATH = Path({str(NOTEBOOK_PATH)!r})
LOG_PATH = Path({str(log_path)!r})
TRIAL = "27j_moe_ensemble_teacher_distill_r1"
RATIONALE = (
    "27h showed model-level MoE is the best existing predictor, while 27i found 25a_r1_repair "
    "is the best single teacher. 27j uses the single teacher as the student anchor and an "
    "8-checkpoint model-level MoE ensemble only for pseudo-label generation, distilling that "
    "scene/night specialist signal into MoE-head/router slots."
)

rows = list(csv.DictReader(METRICS_PATH.open(encoding="utf-8"))) if METRICS_PATH.exists() else []
for row in rows:
    print(row)

def f(raw):
    try:
        value = float(raw or "nan")
    except ValueError:
        return None
    return value if math.isfinite(value) else None

warm = next((row for row in rows if row.get("kind") == "warmup"), {{}})
best_row = max(rows, key=lambda row: f(row.get("map50")) or -1.0) if rows else {{}}
summary_row = {{
    "trial": TRIAL,
    "status": "target_reached" if (f(best_row.get("map50")) or 0.0) >= 0.60 else "completed",
    "best_map50": best_row.get("map50", ""),
    "best_map50_95": best_row.get("map50_95", ""),
    "warmup_map50": warm.get("map50", ""),
    "repair_map50": "",
    "dqa_aggregate_map50": next((row.get("map50", "") for row in rows if row.get("kind") == "aggregate"), ""),
    "dqa_repair_map50": rows[-1].get("map50", "") if rows else "",
    "workspace": str(WORKSPACE),
    "notebook": str(NOTEBOOK_PATH),
    "log": str(LOG_PATH),
    "finished_utc": datetime.now(timezone.utc).isoformat(),
    "rationale": RATIONALE,
}}
fields = [
    "trial", "status", "best_map50", "best_map50_95", "warmup_map50", "repair_map50",
    "dqa_aggregate_map50", "dqa_repair_map50", "workspace", "notebook", "log",
    "finished_utc", "rationale",
]
SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
existing = list(csv.DictReader(SUMMARY_PATH.open(encoding="utf-8"))) if SUMMARY_PATH.exists() else []
existing = [row for row in existing if row.get("trial") != TRIAL]
existing.append(summary_row)
with SUMMARY_PATH.open("w", encoding="utf-8", newline="") as fobj:
    writer = csv.DictWriter(fobj, fieldnames=fields)
    writer.writeheader()
    writer.writerows(existing)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    from notebook_notify import notify_discord

    msg = "\\n".join([
        "27j finished: MoE-ensemble pseudo-teacher distillation",
        f"best_mAP50={{summary_row['best_map50']}} / mAP50:95={{summary_row['best_map50_95']}}",
        f"warmup_mAP50={{summary_row['warmup_map50']}}",
        f"dqa_aggregate_mAP50={{summary_row['dqa_aggregate_map50']}}",
        f"dqa_repair_mAP50={{summary_row['dqa_repair_map50']}}",
        f"workspace={{WORKSPACE}}",
    ])
    print(notify_discord(msg, title="DQA-MoX 27j result", fail_silently=True))
except Exception as exc:
    print("Discord notification skipped:", exc)
"""

    cells = [
        md(
            f"""# 27j MoE-Ensemble Teacher Distillation

- created_utc: {now()}
- target: scene_daynight_total mAP50 >= 0.600
- workspace: `{WORKSPACE}`
- start/student anchor: `25a_r1_repair` (27i best single checkpoint, mAP50=0.464)
- pseudo teacher: 8-checkpoint model-level MoE ensemble

## What We Learned Before This

- Ordinary DQA aggregation and short MoE probes saturate around mAP50=0.462.
- 27h model-level/test-time MoE reached mAP50=0.464 / mAP50:95=0.262, the best observed output predictor.
- 27i found no hidden stronger old checkpoint; `25a_r1_repair` is the best single teacher at mAP50=0.464 / mAP50:95=0.261.

## Paper Basis

- [Domain-Specialized Object Detection via Model-Level Mixtures of Experts](https://arxiv.org/abs/2604.18256): BDD100K object detectors can benefit from model-level expert fusion and domain-specialized experts.
- [HI-MoE](https://arxiv.org/abs/2604.04908): detection MoE should route at scene and instance/object granularity rather than only image-level routing.
- [STEP-DETR](https://openaccess.thecvf.com/content/ICCV2025/papers/Shehzadi_STEP-DETR_Advancing_DETR-based_Semi-Supervised_Object_Detection_with_Super_Teacher_and_ICCV_2025_paper.pdf): SSOD improves when a stronger teacher explicitly supplies higher-quality pseudo labels and reduces confidence bias.

## Hypothesis

Use the single best checkpoint as a stable student anchor, but generate pseudo labels from a model-level MoE ensemble containing global repair experts and night client specialists. Train only MoE-head/router slots for one round so the model learns the ensemble's domain signal without moving the detector body destructively.
"""
        ),
        code(run_cell.strip() + "\n"),
        code(result_cell.strip() + "\n"),
    ]
    NOTEBOOK_PATH.write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
