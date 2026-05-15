#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "05_expert_choice_pseudogt_router_dqa.ipynb"


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
        # 05 Expert-Choice pseudoGT Router DQA

        This is the first non-residual DQA notebook.

        Instead of composing checkpoints after training, 05 changes pseudoGT
        selection itself.  Each round:

        1. generates stable pseudo boxes from the current global model,
        2. lets virtual experts select fixed-capacity class/scale/density
           buckets,
        3. writes balanced pseudoGT lists,
        4. trains clients on source GT + selected pseudoGT,
        5. aggregates normally and optionally applies server repair.

        The final result table is directly comparable with the 03 and 04
        tables.
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

        WORKSPACE = PROJECT_ROOT / "output" / "05_expert_choice_pseudogt_router_dqa"
        SOURCE_WORKSPACE = PROJECT_ROOT / "output" / "03_main_bn_residual_dqa_experiment"
        RUNNER = PROJECT_ROOT / "scripts" / "run_scene_daynight_dqa_05_expert_choice_pseudogt_router.py"

        print("PROJECT_ROOT", PROJECT_ROOT)
        print("WORKSPACE", WORKSPACE)
        print("SOURCE_WORKSPACE", SOURCE_WORKSPACE)
        print("RUNNER", RUNNER)
        """
    ),
    md("## 03 Baseline"),
    code(
        """
        baseline_csv = SOURCE_WORKSPACE / "stats" / "03_main_experiment_final_metrics.csv"
        baseline = pd.read_csv(baseline_csv)
        display(
            baseline[
                [
                    "checkpoint_label",
                    "condition",
                    "map50",
                    "map50_95",
                    "delta_vs_server_repair_map50_95",
                    "worst_split",
                    "day_avg_map50_95",
                    "night_avg_map50_95",
                ]
            ]
        )
        """
    ),
    md("## Schedule And Runtime Estimate"),
    code(
        """
        ROUNDS = 30
        CLIENT_LIMIT = 1500

        # 05 is usually close to 03 DQA runtime because it regenerates pseudoGT,
        # filters it, trains clients, and runs source repair every round.
        EST_MIN_PER_ROUND = 19.5
        FINAL_EVAL_MIN = 55.0
        total_minutes = ROUNDS * EST_MIN_PER_ROUND + FINAL_EVAL_MIN

        estimate = pd.DataFrame([
            {
                "stage": "Expert-Choice pseudoGT DQA",
                "rounds": ROUNDS,
                "minutes_per_round": EST_MIN_PER_ROUND,
                "estimated_minutes": ROUNDS * EST_MIN_PER_ROUND,
            },
            {
                "stage": "final paper-protocol evaluation",
                "rounds": 0,
                "minutes_per_round": None,
                "estimated_minutes": FINAL_EVAL_MIN,
            },
        ])
        display(estimate)
        print(f"Estimated total: {total_minutes / 60:.2f} hours ({total_minutes:.0f} minutes)")
        """
    ),
    md("## Runner Defaults"),
    code(
        """
        spec = importlib.util.spec_from_file_location("run_scene_daynight_dqa_05", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)

        args = runner.parse_args([])
        display(pd.DataFrame([
            {
                "rounds": args.rounds,
                "expert_count": args.expert_count,
                "expert_keep_fraction": args.expert_keep_fraction,
                "expert_max_class_fraction": args.expert_max_class_fraction,
                "load_bias_strength": args.load_bias_strength,
                "train_scope": args.train_scope,
                "aggregate_scope": args.aggregate_scope,
                "client_lr": args.client_lr,
                "source_repeat": args.source_repeat,
                "pseudo_repeat": args.pseudo_repeat,
                "loss_box": args.loss_box,
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
            "--source-workspace", str(SOURCE_WORKSPACE),
            "--client-limit", str(CLIENT_LIMIT),
            "--rounds", str(ROUNDS),
            "--setup-only",
        ], cwd=PROJECT_ROOT, check=True)
        """
    ),
    md(
        """
        ## Run 05 Full Experiment

        This is the long cell.  It writes live progress to:

        ```text
        output/05_expert_choice_pseudogt_router_dqa/stats/05_expert_choice_progress.csv
        ```
        """
    ),
    code(
        """
        RUN_FULL = True
        BATCH_SIZE = 160
        WORKERS = 8
        GPUS = 2
        DEVICE = ""
        FORCE_RESTART = False

        if RUN_FULL:
            cmd = [
                sys.executable,
                str(RUNNER),
                "--workspace-root", str(WORKSPACE),
                "--source-workspace", str(SOURCE_WORKSPACE),
                "--client-limit", str(CLIENT_LIMIT),
                "--rounds", str(ROUNDS),
                "--batch-size", str(BATCH_SIZE),
                "--workers", str(WORKERS),
                "--gpus", str(GPUS),
                "--device", DEVICE,
                "--master-port", "33941",
                "--train-scope", "neck_head",
                "--aggregate-scope", "all",
                "--client-lr", "0.0008",
                "--source-repeat", "1",
                "--pseudo-repeat", "2",
                "--loss-box", "0.005",
                "--expert-count", "4",
                "--expert-keep-fraction", "0.65",
                "--expert-max-class-fraction", "0.35",
                "--load-bias-strength", "0.20",
                "--evaluate",
                "--classwise",
                "--no-eval-plots",
                "--notify",
            ]
            if FORCE_RESTART:
                cmd.append("--force")
                cmd.append("--force-pseudo")
            print(" ".join(cmd))
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        else:
            print("RUN_FULL is False.")
        """
    ),
    md("## Progress"),
    code(
        """
        progress_csv = WORKSPACE / "stats" / "05_expert_choice_progress.csv"
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
    md("## Expert Choice Stats"),
    code(
        """
        stats_files = sorted((WORKSPACE / "stats").glob("05_round*_expert_choice_stats.csv"))
        if stats_files:
            latest_stats = stats_files[-1]
            print(latest_stats)
            display(pd.read_csv(latest_stats))
        else:
            print("No expert-choice stats yet.")
        """
    ),
    md("## Final Metrics"),
    code(
        """
        final_metrics = WORKSPACE / "stats" / "05_expert_choice_final_metrics.csv"
        if final_metrics.exists():
            final_df = pd.read_csv(final_metrics)
            display(
                final_df[
                    [
                        "experiment",
                        "checkpoint_label",
                        "condition",
                        "map50",
                        "map50_95",
                        "delta_vs_03_dqa_aggregate_map50_95",
                        "delta_vs_server_repair_map50_95",
                        "worst_split",
                        "day_avg_map50_95",
                        "night_avg_map50_95",
                    ]
                ].sort_values("map50_95", ascending=False)
            )
        else:
            print("No final metrics yet:", final_metrics)
        """
    ),
    md("## Split Metrics"),
    code(
        """
        split_metrics = WORKSPACE / "stats" / "05_expert_choice_split_metrics.csv"
        if split_metrics.exists():
            split_df = pd.read_csv(split_metrics)
            display(
                split_df[
                    [
                        "experiment",
                        "checkpoint_label",
                        "split",
                        "images",
                        "precision",
                        "recall",
                        "map50",
                        "map50_95",
                    ]
                ].sort_values(["split", "map50_95"], ascending=[True, False])
            )
        else:
            print("No split metrics yet:", split_metrics)
        """
    ),
    md("## Report"),
    code(
        """
        report = WORKSPACE / "05_expert_choice_pseudogt_router_report.md"
        if report.exists():
            print(report)
            print(report.read_text(encoding="utf-8")[:5000])
        else:
            print("No report yet:", report)
        """
    ),
]


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
