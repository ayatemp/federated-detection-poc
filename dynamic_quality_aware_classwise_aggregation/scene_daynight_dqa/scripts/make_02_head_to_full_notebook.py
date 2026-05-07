#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_head_to_full_long_dqa.ipynb"


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
        # 02 Head-to-Full Long DQA

        This notebook tests the new FedSTO-style DQA schedule:

        - **Phase 1:** long head/neck-only DQA adaptation, default 30 rounds.
        - **Phase 2:** short full-model low-LR DQA burst, default 2 rounds.
        - **Evaluation:** final-focused paper-protocol evaluation only, so the
          long run stays practical.

        The design goal is not to keep learning pseudoGT forever.  Phase 1
        creates stable client/class/domain differences for DQA, and Phase 2
        briefly lets that target signal reach the full detector.
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

        WORKSPACE = PROJECT_ROOT / "output" / "02_head_to_full_long_dqa"
        RUNNER = PROJECT_ROOT / "scripts" / "run_scene_daynight_dqa_02_head_to_full.py"

        print("PROJECT_ROOT", PROJECT_ROOT)
        print("WORKSPACE", WORKSPACE)
        print("RUNNER", RUNNER)
        """
    ),
    md("## Schedule And Runtime Estimate"),
    code(
        """
        PHASE1_ROUNDS = 30
        PHASE2_ROUNDS = 2
        PHASE1_MIN_PER_ROUND = 19.0
        PHASE2_MIN_PER_ROUND = 23.0
        FINAL_EVAL_MIN = 60.0

        estimate = pd.DataFrame([
            {
                "stage": "Phase1 head-only",
                "rounds": PHASE1_ROUNDS,
                "minutes_per_round": PHASE1_MIN_PER_ROUND,
                "estimated_minutes": PHASE1_ROUNDS * PHASE1_MIN_PER_ROUND,
            },
            {
                "stage": "Phase2 full burst",
                "rounds": PHASE2_ROUNDS,
                "minutes_per_round": PHASE2_MIN_PER_ROUND,
                "estimated_minutes": PHASE2_ROUNDS * PHASE2_MIN_PER_ROUND,
            },
            {
                "stage": "final-focused paper eval",
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
        spec = importlib.util.spec_from_file_location("run_scene_daynight_dqa_02_head_to_full", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)

        args = runner.parse_args([])
        phase1 = runner.default_phase1_spec(args)
        phase2 = runner.default_phase2_spec(args)
        display(pd.DataFrame([
            {"phase": "phase1", **runner.asdict(phase1)},
            {"phase": "phase2", **runner.asdict(phase2)},
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
            "--phase1-rounds", str(PHASE1_ROUNDS),
            "--phase2-rounds", str(PHASE2_ROUNDS),
            "--setup-only",
        ], cwd=PROJECT_ROOT, check=True)
        """
    ),
    md(
        """
        ## Run Phase1 30 / Phase2 2

        The runner uses `tqdm` internally and writes live progress to:

        ```text
        output/02_head_to_full_long_dqa/stats/02_head_to_full_progress.csv
        ```

        Default evaluation is final-focused: warmup, Phase1 final aggregate,
        Phase1 final repair, Phase2 final aggregate, and Phase2 final repair.
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
            "--client-limit", str(CLIENT_LIMIT),
            "--phase1-rounds", str(PHASE1_ROUNDS),
            "--phase2-rounds", str(PHASE2_ROUNDS),
            "--batch-size", str(BATCH_SIZE),
            "--workers", str(WORKERS),
            "--gpus", str(GPUS),
            "--device", DEVICE,
            "--master-port", "31141",
            "--max-images-per-client", str(MAX_IMAGES_PER_CLIENT),
            "--estimated-phase1-round-minutes", str(PHASE1_MIN_PER_ROUND),
            "--estimated-phase2-round-minutes", str(PHASE2_MIN_PER_ROUND),
            "--estimated-eval-minutes", str(FINAL_EVAL_MIN),
            "--evaluate",
            "--classwise",
            "--no-eval-plots",
            "--notify",
        ]

        print(" ".join(cmd))
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        """
    ),
    md("## Progress"),
    code(
        """
        progress_csv = WORKSPACE / "stats" / "02_head_to_full_progress.csv"
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
        final_metrics = WORKSPACE / "stats" / "02_head_to_full_final_metrics.csv"
        if final_metrics.exists():
            final_df = pd.read_csv(final_metrics)
            display(final_df)
        else:
            print("No final metrics yet:", final_metrics)
        """
    ),
    md("## Split-Level Final Evaluation"),
    code(
        """
        split_metrics = WORKSPACE / "stats" / "02_head_to_full_split_metrics.csv"
        if split_metrics.exists():
            split_df = pd.read_csv(split_metrics)
            cols = ["checkpoint_label", "split", "images", "labels", "precision", "recall", "map50", "map50_95"]
            display(split_df[cols].sort_values(["split", "checkpoint_label"]))
        else:
            print("No split metrics yet:", split_metrics)
        """
    ),
    md("## Compare Against 01_0 Repair-Only Summary"),
    code(
        """
        baseline = PROJECT_ROOT / "output" / "01_0_repair_baseline_comparison" / "stats" / "01_0_all_condition_metrics.csv"
        if final_metrics.exists() and baseline.exists():
            base = pd.read_csv(baseline)
            repair_only = base[base["condition"].eq("repair_only")].sort_values("round").tail(1)
            display(repair_only)
            cols = [
                "checkpoint_label",
                "map50",
                "map50_95",
                "delta_vs_repair_only_r3_map50_95",
                "worst_split",
                "worst_split_map50_95",
                "worst_delta_vs_repair_only_r3_map50_95",
                "night_avg_map50_95",
                "night_delta_vs_repair_only_r3_map50_95",
            ]
            display(final_df[cols])
        else:
            print("Missing final metrics or baseline.")
        """
    ),
    md("## PseudoGT Signal Trend"),
    code(
        """
        pseudo_rows = []
        stats_dir = WORKSPACE / "stats"
        if stats_dir.exists():
            for path in sorted(stats_dir.glob("03_round*_pseudo_label_stats.csv")):
                part = pd.read_csv(path)
                part["source_csv"] = path.name
                pseudo_rows.append(part)
        if pseudo_rows:
            pseudo_df = pd.concat(pseudo_rows, ignore_index=True)
            summary_cols = [c for c in [
                "round", "client", "images", "kept_images", "boxes",
                "mean_conf", "mean_stability", "mean_score", "source_csv",
            ] if c in pseudo_df.columns]
            display(pseudo_df[summary_cols].tail(30))
        else:
            print("No pseudoGT stats yet.")
        """
    ),
]


notebook = {
    "cells": cells,
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

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(NOTEBOOK_PATH)
