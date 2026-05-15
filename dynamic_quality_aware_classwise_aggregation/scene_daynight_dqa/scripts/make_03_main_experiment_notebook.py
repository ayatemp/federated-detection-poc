#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "03_main_bn_residual_dqa_experiment.ipynb"


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
        # 03 Main BN-Residual DQA Experiment

        This is the main comparison notebook for the current DQA direction.

        It evaluates the same scene/day-night paper protocol for:

        - **warmup**
        - **warmup + server repair**
        - **warmup + BN-residual DQA + server repair**

        The DQA branch uses the strongest finding from the MoE/DQA loops:
        client expert residuals are useful, but the old global aggregation
        erases them.  Here we train scene/day-night clients, then apply the
        average day-client residual to the server model's neck/head while
        keeping BatchNorm tensors in that scope.
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
        if cwd.name == "notebooks":
            PROJECT_ROOT = cwd.parent
        elif (cwd / "dynamic_quality_aware_classwise_aggregation").exists():
            PROJECT_ROOT = cwd / "dynamic_quality_aware_classwise_aggregation" / "scene_daynight_dqa"
        else:
            PROJECT_ROOT = cwd

        WORKSPACE = PROJECT_ROOT / "output" / "03_main_bn_residual_dqa_experiment"
        RUNNER = PROJECT_ROOT / "scripts" / "run_scene_daynight_dqa_03_main_experiment.py"

        print("PROJECT_ROOT", PROJECT_ROOT)
        print("WORKSPACE", WORKSPACE)
        print("RUNNER", RUNNER)
        """
    ),
    md("## Schedule And Runtime Estimate"),
    code(
        """
        # Main setting.  REPAIR_ROUNDS is intentionally set to DQA_ROUNDS for
        # a fair source-repair budget.  For a quick smoke run, temporarily set
        # both to 1 and MAX_IMAGES_PER_CLIENT to a small number.
        REPAIR_ROUNDS = 30
        DQA_ROUNDS = 30

        REPAIR_MIN_PER_ROUND = 4.0
        DQA_MIN_PER_ROUND = 19.0
        FINAL_EVAL_MIN = 55.0

        estimate = pd.DataFrame([
            {
                "stage": "warmup + server repair baseline",
                "rounds": REPAIR_ROUNDS,
                "minutes_per_round": REPAIR_MIN_PER_ROUND,
                "estimated_minutes": REPAIR_ROUNDS * REPAIR_MIN_PER_ROUND,
            },
            {
                "stage": "BN-residual DQA + server repair",
                "rounds": DQA_ROUNDS,
                "minutes_per_round": DQA_MIN_PER_ROUND,
                "estimated_minutes": DQA_ROUNDS * DQA_MIN_PER_ROUND,
            },
            {
                "stage": "final paper-protocol evaluation",
                "rounds": 0,
                "minutes_per_round": None,
                "estimated_minutes": FINAL_EVAL_MIN,
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
        spec = importlib.util.spec_from_file_location("run_scene_daynight_dqa_03_main_experiment", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)

        args = runner.parse_args([])
        display(pd.DataFrame([
            {
                "repair_rounds": args.repair_rounds,
                "dqa_rounds": args.dqa_rounds,
                "dqa_train_scope": args.dqa_train_scope,
                "dqa_residual_scope": args.dqa_residual_scope,
                "dqa_include_bn": args.dqa_include_bn,
                "dqa_residual_beta": args.dqa_residual_beta,
                "dqa_client_lr": args.dqa_client_lr,
                "dqa_source_repeat": args.dqa_source_repeat,
                "dqa_pseudo_repeat": args.dqa_pseudo_repeat,
                "dqa_loss_box": args.dqa_loss_box,
            }
        ]))
        """
    ),
    md("## Setup Only"),
    code(
        """
        subprocess.run([
            sys.executable,
            str(RUNNER),
            "--workspace-root", str(WORKSPACE),
            "--client-limit", "1500",
            "--repair-rounds", str(REPAIR_ROUNDS),
            "--dqa-rounds", str(DQA_ROUNDS),
            "--setup-only",
        ], cwd=PROJECT_ROOT, check=True)
        """
    ),
    md(
        """
        ## Run Main Experiment

        The runner writes a live progress table to:

        ```text
        output/03_main_bn_residual_dqa_experiment/stats/03_main_experiment_progress.csv
        ```

        Discord notifications are sent at start and finish.
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
        FORCE_RESTART = True

        cmd = [
            sys.executable,
            str(RUNNER),
            "--workspace-root", str(WORKSPACE),
            "--client-limit", str(CLIENT_LIMIT),
            "--repair-rounds", str(REPAIR_ROUNDS),
            "--dqa-rounds", str(DQA_ROUNDS),
            "--batch-size", str(BATCH_SIZE),
            "--workers", str(WORKERS),
            "--gpus", str(GPUS),
            "--device", DEVICE,
            "--master-port", "31841",
            "--max-images-per-client", str(MAX_IMAGES_PER_CLIENT),
            "--dqa-train-scope", "neck_head",
            "--dqa-residual-scope", "neck_head",
            "--dqa-include-bn", "true",
            "--dqa-residual-beta", "1.0",
            "--dqa-client-lr", "0.0008",
            "--dqa-source-repeat", "1",
            "--dqa-pseudo-repeat", "2",
            "--dqa-loss-box", "0.005",
            "--estimated-repair-round-minutes", str(REPAIR_MIN_PER_ROUND),
            "--estimated-dqa-round-minutes", str(DQA_MIN_PER_ROUND),
            "--estimated-eval-minutes", str(FINAL_EVAL_MIN),
            "--evaluate",
            "--classwise",
            "--no-eval-plots",
            "--notify",
        ]
        if FORCE_RESTART:
            cmd.append("--force")

        print(" ".join(cmd))
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        """
    ),
    md("## Progress"),
    code(
        """
        progress_csv = WORKSPACE / "stats" / "03_main_experiment_progress.csv"
        if progress_csv.exists():
            progress_df = pd.read_csv(progress_csv)
            display(progress_df.tail(12))
            latest = progress_df.tail(1).iloc[0]
            print(
                f"elapsed={latest['elapsed_hms']} eta={latest['eta_hms']} "
                f"completed={latest['completed_steps']}/{latest['total_steps']}"
            )
        else:
            print("No progress CSV yet:", progress_csv)
        """
    ),
    md("## Final Metrics"),
    code(
        """
        final_metrics = WORKSPACE / "stats" / "03_main_experiment_final_metrics.csv"
        if final_metrics.exists():
            final_df = pd.read_csv(final_metrics)
            display(final_df)
        else:
            print("No final metrics yet:", final_metrics)
        """
    ),
    md("## Split-Level Evaluation"),
    code(
        """
        split_metrics = WORKSPACE / "stats" / "03_main_experiment_split_metrics.csv"
        if split_metrics.exists():
            split_df = pd.read_csv(split_metrics)
            cols = [
                "checkpoint_label",
                "condition",
                "split",
                "images",
                "labels",
                "precision",
                "recall",
                "map50",
                "map50_95",
            ]
            display(split_df[cols].sort_values(["split", "checkpoint_label"]))
        else:
            print("No split metrics yet:", split_metrics)
        """
    ),
    md("## Quick Comparison"),
    code(
        """
        final_metrics = WORKSPACE / "stats" / "03_main_experiment_final_metrics.csv"
        if final_metrics.exists():
            df = pd.read_csv(final_metrics)
            display(df[[
                "condition",
                "map50",
                "map50_95",
                "gain_vs_warmup_map50_95",
                "delta_vs_server_repair_map50_95",
                "worst_split",
                "worst_split_map50_95",
                "day_avg_map50_95",
                "night_avg_map50_95",
            ]])
        """
    ),
]

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(
    json.dumps(
        {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        },
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)
print(f"Wrote {NOTEBOOK_PATH}")
