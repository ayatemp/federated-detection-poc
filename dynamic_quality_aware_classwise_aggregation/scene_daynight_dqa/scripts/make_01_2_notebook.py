#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_2_ssod_pivot_dqa.ipynb"


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
        # 01_2 SSOD Pivot DQA

        This notebook is the fallback plan if `01_1` still cannot make DQA beat
        `repair_only`.

        Hypothesis:
        fixed pseudoGT is too brittle as supervised bbox training data.  Instead,
        client updates should follow the FedSTO/EfficientTeacher style: train on
        unlabeled target images through EMA teacher/student SSOD, and use stable
        pseudo boxes only as a DQA quality signal for aggregation.

        Claims this notebook can test:
        - fixed pseudoGT failure does not imply target data is useless
        - SSOD client updates may extract target/domain signal more safely
        - DQA should protect the aggregate better than SSOD FedAvg
        - worst/night splits are the real target-domain test
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

        WORKSPACE = PROJECT_ROOT / "output" / "01_2_ssod_pivot"
        print("PROJECT_ROOT", PROJECT_ROOT)
        print("WORKSPACE", WORKSPACE)
        """
    ),
    md("## Conditions"),
    code(
        """
        import importlib.util
        import pandas as pd

        runner_path = PROJECT_ROOT / "scripts" / "run_scene_daynight_dqa_01_2.py"
        spec = importlib.util.spec_from_file_location("run_scene_daynight_dqa_01_2", runner_path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)

        rows = []
        for name, cond in runner.CONDITION_SPECS.items():
            rows.append({
                "condition": name,
                "mode": cond.mode,
                "train_scope": cond.train_scope,
                "client_lr0": cond.client_lr0,
                "ssod_box_loss_weight": cond.ssod_box_loss_weight,
                "min_server_alpha": cond.dqa_min_server_alpha,
                "residual_blend": cond.dqa_residual_blend,
                "note": cond.note,
            })
        display(pd.DataFrame(rows))
        """
    ),
    md("## Setup Only"),
    code(
        """
        CLIENT_LIMIT = 800
        subprocess.run([
            sys.executable,
            "scripts/run_scene_daynight_dqa_01_2.py",
            "--workspace-root", str(WORKSPACE),
            "--client-limit", str(CLIENT_LIMIT),
            "--conditions", "all",
            "--setup-only",
        ], cwd=PROJECT_ROOT, check=True)
        """
    ),
    md(
        """
        ## Run SSOD Pivot

        Default is intentionally conservative: 2 rounds, 800 images/client,
        1 GPU, batch size 32.  This avoids the DDP SIGTERM behavior from 01_0.

        Full condition:
        - set `CLIENT_LIMIT = 1500`
        - set `MAX_IMAGES_PER_CLIENT = 1500`
        - set `ROUNDS = 3`
        """
    ),
    code(
        """
        ROUNDS = 2
        CONDITIONS = "all"
        MAX_IMAGES_PER_CLIENT = 800
        BATCH_SIZE = 32
        WORKERS = 4
        GPUS = 1
        DEVICE = "0"
        EVAL_CLIENTS = False

        cmd = [
            sys.executable,
            "scripts/run_scene_daynight_dqa_01_2.py",
            "--workspace-root", str(WORKSPACE),
            "--client-limit", str(CLIENT_LIMIT),
            "--conditions", CONDITIONS,
            "--rounds", str(ROUNDS),
            "--batch-size", str(BATCH_SIZE),
            "--workers", str(WORKERS),
            "--gpus", str(GPUS),
            "--device", DEVICE,
            "--master-port", "31041",
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
        metrics_csv = WORKSPACE / "stats" / "01_2_all_condition_metrics.csv"
        if metrics_csv.exists():
            df = pd.read_csv(metrics_csv)
            display(df)
        else:
            print("No combined metrics yet:", metrics_csv)
        """
    ),
    md("## Final-Round Comparison"),
    code(
        """
        if metrics_csv.exists():
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
    md("## DQA Pseudo Quality Signal"),
    code(
        """
        pseudo_rows = []
        for condition_dir in sorted(WORKSPACE.glob("ssod_dqa*")):
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
            display(pseudo_df.groupby(["condition", "round"]).agg(
                pseudo_images_kept=("pseudo_images_kept", "sum"),
                pseudo_boxes_kept=("pseudo_boxes_kept", "sum"),
                mean_score=("mean_score", "mean"),
            ).reset_index())
        else:
            print("No DQA pseudo quality stats yet.")
        """
    ),
    md("## Interpretation Checklist"),
    code(
        """
        if metrics_csv.exists():
            final = df.sort_values("round").groupby("condition").tail(1).copy()
            numeric_cols = [
                "aggregate_map50_95", "repaired_map50_95", "delta_vs_repair_only_map50_95",
                "night_delta_vs_repair_only_map50_95", "worst_delta_vs_repair_only_map50_95",
            ]
            for col in numeric_cols:
                final[col] = pd.to_numeric(final[col], errors="coerce")

            print("Best repaired scores:")
            display(final.sort_values("repaired_map50_95", ascending=False)[[
                "condition", "mode", "repaired_map50_95", "delta_vs_repair_only_map50_95",
                "night_delta_vs_repair_only_map50_95", "worst_delta_vs_repair_only_map50_95",
            ]])

            print("Best aggregate preservation:")
            display(final.sort_values("aggregate_map50_95", ascending=False)[[
                "condition", "mode", "aggregate_map50_95", "repaired_map50_95", "repair_gain_map50_95",
            ]])

            print("Claim guide:")
            print("- SSOD FedAvg below DQA aggregate => DQA protects target client updates.")
            print("- Any SSOD DQA repaired > repair_only => target data is useful when trained as SSOD.")
            print("- Night/worst delta > 0 => DQA helps domain/client weakness even if total mAP ties.")
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
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
