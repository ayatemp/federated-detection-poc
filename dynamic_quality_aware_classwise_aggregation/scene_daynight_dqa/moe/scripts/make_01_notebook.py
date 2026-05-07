#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path


MOE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = MOE_ROOT / "notebooks" / "01_dqa_moe_expert_pool_full.ipynb"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    md(
        """
        # 01 DQA-MoE Expert Pool Full

        This is the first MoE-side notebook for scene/day-night DQA.

        The goal is to test whether DQA should preserve client/domain
        specialization instead of collapsing every client update into one global
        checkpoint.

        This pilot is **checkpoint-level MoE**, not a true architectural YOLO
        MoE head yet:

        - `K=4` experts are initialized from the same warmup model.
        - Phase 1 follows the 02 idea: long `neck_head` adaptation.
        - Phase 2 follows the 02 idea: short low-LR full-model burst.
        - DQA routes client updates into scene/hard-case expert checkpoints.
        - The deployable model is a soft mixture of expert residuals followed
          by server repair.
        - Final evaluation includes the deployable model and expert checkpoints
          to estimate whether a real router would be worth implementing.
        """
    ),
    code(
        """
        from pathlib import Path
        import importlib.util
        import subprocess
        import sys

        import pandas as pd

        cwd = Path.cwd().resolve()
        if cwd.name == "notebooks" and cwd.parent.name == "moe":
            MOE_ROOT = cwd.parent
        elif (cwd / "dynamic_quality_aware_classwise_aggregation").exists():
            MOE_ROOT = cwd / "dynamic_quality_aware_classwise_aggregation" / "scene_daynight_dqa" / "moe"
        else:
            MOE_ROOT = cwd

        SCENE_ROOT = MOE_ROOT.parent
        WORKSPACE = MOE_ROOT / "output" / "01_dqa_moe_expert_pool"
        RUNNER = MOE_ROOT / "scripts" / "run_moe_01_dqa_expert_pool.py"

        print("MOE_ROOT", MOE_ROOT)
        print("SCENE_ROOT", SCENE_ROOT)
        print("WORKSPACE", WORKSPACE)
        print("RUNNER", RUNNER)
        """
    ),
    md("## Design Notes"),
    md(
        """
        With `K=4`, the expert routing is:

        | expert | role |
        |---|---|
        | 0 | highway clients |
        | 1 | citystreet clients |
        | 2 | residential clients |
        | 3 | night/hard clients |

        Night clients are intentionally routed both to their scene expert and to
        the night/hard expert.  The final deployable checkpoint uses a soft
        mixture, so overlapping membership is allowed.
        """
    ),
    md("## Runtime Estimate"),
    code(
        """
        EXPERTS = 4
        PHASE1_ROUNDS = 30
        PHASE2_ROUNDS = 2
        PHASE1_MIN_PER_ROUND = 19.0
        PHASE2_MIN_PER_ROUND = 23.0
        FINAL_EVAL_MIN = 60.0
        MOE_AGG_MIN_PER_EXPERT = 0.35
        EVAL_EXPERT_MIN = 6.0

        estimate = pd.DataFrame([
            {
                "stage": "Phase1 head/neck DQA-MoE",
                "rounds": PHASE1_ROUNDS,
                "minutes_per_round": PHASE1_MIN_PER_ROUND,
                "estimated_minutes": PHASE1_ROUNDS * PHASE1_MIN_PER_ROUND,
            },
            {
                "stage": "Phase2 full burst DQA-MoE",
                "rounds": PHASE2_ROUNDS,
                "minutes_per_round": PHASE2_MIN_PER_ROUND,
                "estimated_minutes": PHASE2_ROUNDS * PHASE2_MIN_PER_ROUND,
            },
            {
                "stage": "MoE expert aggregation overhead",
                "rounds": PHASE1_ROUNDS + PHASE2_ROUNDS,
                "minutes_per_round": EXPERTS * MOE_AGG_MIN_PER_EXPERT,
                "estimated_minutes": (PHASE1_ROUNDS + PHASE2_ROUNDS) * EXPERTS * MOE_AGG_MIN_PER_EXPERT,
            },
            {
                "stage": "final-focused eval",
                "rounds": 0,
                "minutes_per_round": None,
                "estimated_minutes": FINAL_EVAL_MIN + EXPERTS * EVAL_EXPERT_MIN,
            },
        ])
        total_minutes = float(estimate["estimated_minutes"].sum())
        display(estimate)
        print(f"Estimated total: {total_minutes / 60:.2f} hours ({total_minutes:.0f} minutes)")
        """
    ),
    md("## Runner Defaults"),
    code(
        """
        spec = importlib.util.spec_from_file_location("run_moe_01_dqa_expert_pool", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)

        args = runner.parse_args([])
        print("experts:", args.experts)
        print("expert names:", runner.expert_names(args.experts))
        print("estimated:", runner.seconds_to_hms(runner.estimated_seconds(args)))
        """
    ),
    md("## Setup Only"),
    code(
        """
        subprocess.run([
            sys.executable,
            str(RUNNER),
            "--workspace-root", str(WORKSPACE),
            "--experts", str(EXPERTS),
            "--client-limit", "1500",
            "--phase1-rounds", str(PHASE1_ROUNDS),
            "--phase2-rounds", str(PHASE2_ROUNDS),
            "--setup-only",
        ], cwd=MOE_ROOT, check=True)
        """
    ),
    md(
        """
        ## Run Full DQA-MoE

        The runner writes live progress to:

        ```text
        moe/output/01_dqa_moe_expert_pool/stats/01_moe_progress.csv
        ```

        Final-focused evaluation includes:

        - warmup
        - Phase1 final MoE softmix
        - Phase1 final server repair
        - Phase2 final MoE softmix
        - Phase2 final server repair
        - Phase2 final expert 0-3 aggregate checkpoints
        """
    ),
    code(
        """
        CLIENT_LIMIT = 1500
        BATCH_SIZE = 160
        WORKERS = 8
        GPUS = 2
        DEVICE = ""
        MAX_IMAGES_PER_CLIENT = 0

        cmd = [
            sys.executable,
            str(RUNNER),
            "--workspace-root", str(WORKSPACE),
            "--experts", str(EXPERTS),
            "--client-limit", str(CLIENT_LIMIT),
            "--phase1-rounds", str(PHASE1_ROUNDS),
            "--phase2-rounds", str(PHASE2_ROUNDS),
            "--batch-size", str(BATCH_SIZE),
            "--workers", str(WORKERS),
            "--gpus", str(GPUS),
            "--device", DEVICE,
            "--master-port", "31241",
            "--max-images-per-client", str(MAX_IMAGES_PER_CLIENT),
            "--estimated-phase1-round-minutes", str(PHASE1_MIN_PER_ROUND),
            "--estimated-phase2-round-minutes", str(PHASE2_MIN_PER_ROUND),
            "--estimated-eval-minutes", str(FINAL_EVAL_MIN),
            "--estimated-moe-aggregation-minutes", str(MOE_AGG_MIN_PER_EXPERT),
            "--estimated-eval-expert-minutes", str(EVAL_EXPERT_MIN),
            "--evaluate",
            "--classwise",
            "--no-eval-plots",
            "--eval-experts",
            "--no-eval-phase1-experts",
            "--notify",
        ]

        print(" ".join(cmd))
        subprocess.run(cmd, cwd=MOE_ROOT, check=True)
        """
    ),
    md("## Progress"),
    code(
        """
        progress_csv = WORKSPACE / "stats" / "01_moe_progress.csv"
        if progress_csv.exists():
            progress_df = pd.read_csv(progress_csv)
            display(progress_df.tail(10))
            latest = progress_df.tail(1).iloc[0]
            print(f"elapsed={latest['elapsed_hms']} eta={latest['eta_hms']} completed={latest['completed_rounds']}/{latest['total_rounds']}")
        else:
            print("No progress CSV yet:", progress_csv)
        """
    ),
    md("## Final Metrics"),
    code(
        """
        final_metrics = WORKSPACE / "stats" / "01_moe_final_metrics.csv"
        if final_metrics.exists():
            final_df = pd.read_csv(final_metrics)
            display(final_df)
        else:
            print("No final metrics yet:", final_metrics)
        """
    ),
    md("## Split-Level Metrics"),
    code(
        """
        split_metrics = WORKSPACE / "stats" / "01_moe_split_metrics.csv"
        if split_metrics.exists():
            split_df = pd.read_csv(split_metrics)
            cols = ["checkpoint_label", "expert_name", "split", "images", "labels", "precision", "recall", "map50", "map50_95"]
            display(split_df[cols].sort_values(["split", "checkpoint_label"]))
        else:
            print("No split metrics yet:", split_metrics)
        """
    ),
    md("## Expert Routing Upper Bound"),
    code(
        """
        oracle_csv = WORKSPACE / "stats" / "01_moe_oracle_split_metrics.csv"
        if oracle_csv.exists():
            oracle_df = pd.read_csv(oracle_csv)
            display(oracle_df)
        else:
            print("No oracle split metrics yet:", oracle_csv)
        """
    ),
    md("## Route Log"),
    code(
        """
        route_csv = WORKSPACE / "stats" / "01_moe_routes.csv"
        if route_csv.exists():
            route_df = pd.read_csv(route_csv)
            display(route_df.tail(30))
        else:
            print("No route log yet:", route_csv)
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
