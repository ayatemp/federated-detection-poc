#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_1_dqa_diagnostic_sweep.ipynb"


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
        # 01_1 DQA Improvement Sweep

        This notebook is a diagnostic follow-up to `01_0`.

        Goal:
        - run only the DQA improvement candidates by default
        - test whether DQA improves over the already-measured `01_0` controls
        - avoid re-running comparison-only conditions unless explicitly requested

        The runner sends Discord notifications at start and finish when `--notify` is enabled.
        """
    ),
    code(
        """
        from pathlib import Path
        import subprocess
        import sys

        cwd = Path.cwd().resolve()
        if cwd.name == "notebooks":
            PROJECT_ROOT = cwd.parent
        elif (cwd / "dynamic_quality_aware_classwise_aggregation").exists():
            PROJECT_ROOT = cwd / "dynamic_quality_aware_classwise_aggregation" / "scene_daynight_dqa"
        else:
            PROJECT_ROOT = cwd

        REPO_ROOT = PROJECT_ROOT.parents[1]
        WORKSPACE = PROJECT_ROOT / "output" / "01_1_dqa_diagnostic_sweep"
        print("PROJECT_ROOT", PROJECT_ROOT)
        print("WORKSPACE", WORKSPACE)
        """
    ),
    md(
        """
        ## Conditions

        Default improvement conditions:

        | condition | purpose |
        |---|---|
        | `dqa_current` | 01_0-style source-heavy DQA |
        | `dqa_source_light` | reduce source dominance |
        | `dqa_target_double` | target-heavy pseudoGT with softer DQA anchor |
        | `dqa_head_lowbox` | head-only target adaptation, weak bbox loss |
        | `dqa_nonbackbone_lowbox` | neck/head adaptation, weak bbox loss |

        `repair_only` and `fedavg_target_double` remain callable by explicit
        condition name, but they are not included in the notebook default.  The
        comparison baseline should come from `01_0`.

        The default below follows the 01_0 full-run resource setting:
        3 rounds, all 1500 target images per client, 2 GPUs, batch size 160,
        and 8 workers.
        """
    ),
    code(
        """
        import importlib.util
        import pandas as pd

        runner_path = PROJECT_ROOT / "scripts" / "run_scene_daynight_dqa_01_1.py"
        spec = importlib.util.spec_from_file_location("run_scene_daynight_dqa_01_1", runner_path)
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)

        rows = []
        for name in runner.DEFAULT_CONDITIONS:
            cond = runner.CONDITION_SPECS[name]
            rows.append({
                "condition": name,
                "mode": cond.mode,
                "train_scope": cond.train_scope,
                "source_repeat": cond.source_repeat,
                "pseudo_repeat": cond.pseudo_repeat,
                "loss_box": cond.loss_box,
                "min_server_alpha": cond.dqa_min_server_alpha,
                "residual_blend": cond.dqa_residual_blend,
                "note": cond.note,
            })
        display(pd.DataFrame(rows))
        """
    ),
    md(
        """
        ## Setup Only

        Run this first if you only want to build the six client/eval lists and confirm paths.
        `all` resolves to the DQA improvement conditions only.
        """
    ),
    code(
        """
        CLIENT_LIMIT = 1500
        subprocess.run([
            sys.executable,
            "scripts/run_scene_daynight_dqa_01_1.py",
            "--workspace-root", str(WORKSPACE),
            "--client-limit", str(CLIENT_LIMIT),
            "--conditions", "all",
            "--setup-only",
        ], cwd=PROJECT_ROOT, check=True)
        """
    ),
    md(
        """
        ## Run DQA Improvement Sweep

        Default: 3 rounds, all target images per client, 2 GPUs.

        This is meant to answer:
        - Which DQA variant is the best improvement candidate?
        - Do target-heavy / low-box variants improve night or worst split?
        - Does the aggregate survive before repair?
        """
    ),
    code(
        """
        ROUNDS = 3
        CONDITIONS = "all"  # DQA-only defaults; explicit controls are still supported
        MAX_IMAGES_PER_CLIENT = 0
        BATCH_SIZE = 160
        WORKERS = 8
        GPUS = 2
        DEVICE = ""
        EVAL_CLIENTS = False

        cmd = [
            sys.executable,
            "scripts/run_scene_daynight_dqa_01_1.py",
            "--workspace-root", str(WORKSPACE),
            "--client-limit", str(CLIENT_LIMIT),
            "--conditions", CONDITIONS,
            "--rounds", str(ROUNDS),
            "--batch-size", str(BATCH_SIZE),
            "--workers", str(WORKERS),
            "--gpus", str(GPUS),
            "--device", DEVICE,
            "--master-port", "30941",
            "--server-repair-epochs", "1",
            "--max-images-per-client", str(MAX_IMAGES_PER_CLIENT),
            "--evaluate",
            "--classwise",
            "--no-eval-plots",
            "--notify",
        ]
        if EVAL_CLIENTS:
            cmd.append("--eval-clients")

        print(" ".join(cmd))
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        """
    ),
    md("## Combined Metrics"),
    code(
        """
        metrics_csv = WORKSPACE / "stats" / "01_1_all_condition_metrics.csv"
        if metrics_csv.exists():
            df = pd.read_csv(metrics_csv)
            display(df)
        else:
            print("No combined metrics yet:", metrics_csv)
        """
    ),
    md("## Final-Round Metrics"),
    code(
        """
        if metrics_csv.exists():
            df = pd.read_csv(metrics_csv)
            final = df.sort_values("round").groupby("condition").tail(1)
            cols = [
                "condition", "mode", "round",
                "aggregate_map50_95", "aggregate_survival_vs_warmup_map50_95",
                "repaired_map50", "repaired_map50_95", "delta_vs_repair_only_map50_95",
                "worst_split", "worst_split_map50_95", "worst_delta_vs_repair_only_map50_95",
                "night_avg_map50_95", "night_delta_vs_repair_only_map50_95",
                "day_night_gap_map50_95",
            ]
            display(final[cols].sort_values("repaired_map50_95", ascending=False))
        """
    ),
    md("## Split-Level Final Evaluation"),
    code(
        """
        rows = []
        if metrics_csv.exists():
            for condition in sorted(df["condition"].unique()):
                path = WORKSPACE / condition / "validation_reports" / "paper_protocol_eval_summary.csv"
                if not path.exists():
                    continue
                part = pd.read_csv(path)
                part["condition"] = condition
                rows.append(part)
        if rows:
            eval_df = pd.concat(rows, ignore_index=True)
            ok = eval_df[eval_df["status"].eq("ok")].copy()
            final_rows = ok[ok["checkpoint_label"].eq(f"round{ROUNDS:03d}_server_repair")]
            display(final_rows[["condition", "split", "precision", "recall", "map50", "map50_95"]].sort_values(["split", "condition"]))
        else:
            print("No evaluation summaries yet.")
        """
    ),
    md("## PseudoGT Stats"),
    code(
        """
        pseudo_rows = []
        for condition_dir in sorted(WORKSPACE.glob("*")):
            stats_dir = condition_dir / "stats"
            if not stats_dir.exists():
                continue
            for path in sorted(stats_dir.glob("03_round*_pseudo_label_stats.csv")):
                part = pd.read_csv(path)
                part["condition"] = condition_dir.name
                pseudo_rows.append(part)
        if pseudo_rows:
            pseudo_df = pd.concat(pseudo_rows, ignore_index=True)
            display(pseudo_df[[
                "condition", "round", "client", "source_images_scanned",
                "pseudo_images_kept", "pseudo_boxes_kept", "boxes_per_kept_image",
                "mean_conf", "mean_stability", "mean_score",
            ]])
            summary = pseudo_df.groupby(["condition", "round"]).agg(
                pseudo_images_kept=("pseudo_images_kept", "sum"),
                pseudo_boxes_kept=("pseudo_boxes_kept", "sum"),
                mean_score=("mean_score", "mean"),
            ).reset_index()
            display(summary)
        else:
            print("No pseudoGT stats yet.")
        """
    ),
    md("## Quick Interpretation"),
    code(
        """
        if metrics_csv.exists():
            final = df.sort_values("round").groupby("condition").tail(1).copy()
            final["delta_vs_repair_only_map50_95"] = pd.to_numeric(final["delta_vs_repair_only_map50_95"], errors="coerce")
            final["aggregate_map50_95"] = pd.to_numeric(final["aggregate_map50_95"], errors="coerce")
            final["repaired_map50_95"] = pd.to_numeric(final["repaired_map50_95"], errors="coerce")
            print("Best repaired:")
            display(final.sort_values("repaired_map50_95", ascending=False).head(5)[[
                "condition", "mode", "repaired_map50_95", "delta_vs_repair_only_map50_95",
                "night_delta_vs_repair_only_map50_95", "worst_delta_vs_repair_only_map50_95",
            ]])
            print("Best aggregate survival:")
            display(final.sort_values("aggregate_map50_95", ascending=False).head(5)[[
                "condition", "mode", "aggregate_map50_95", "repaired_map50_95", "repair_gain_map50_95",
            ]])
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
print(f"Wrote {NOTEBOOK_PATH}")
