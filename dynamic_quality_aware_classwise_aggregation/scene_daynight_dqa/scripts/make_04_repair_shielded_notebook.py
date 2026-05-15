#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "04_repair_shielded_local_expert_dqa.ipynb"


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
        # 04 Repair-shielded Local Expert DQA

        This notebook turns the best MoE 06 loop into a direct 03 comparison.

        The selected idea is:

        > server repair should update the shared/source path, while local
        > pseudo-GT expert residuals are kept shielded from the final repair.

        Warmup, server repair, and BN-residual DQA have already been trained in
        `03_main_bn_residual_dqa_experiment`.  This notebook reuses those
        checkpoints, creates repair-shielded local expert candidates, and
        evaluates them with the same scene/day-night protocol.
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

        WORKSPACE = PROJECT_ROOT / "output" / "04_repair_shielded_local_expert_dqa"
        SOURCE_WORKSPACE = PROJECT_ROOT / "output" / "03_main_bn_residual_dqa_experiment"
        RUNNER = PROJECT_ROOT / "scripts" / "run_scene_daynight_dqa_04_repair_shielded_local_expert.py"

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
    md("## Runner Defaults / Candidate Design"),
    code(
        """
        spec = importlib.util.spec_from_file_location("run_scene_daynight_dqa_04", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)

        args = runner.parse_args([])
        display(pd.DataFrame([
            {
                "source_round": args.source_round,
                "candidate_betas": args.candidate_betas,
                "residual_scope": args.residual_scope,
                "include_bn": args.include_bn,
                "source_workspace": str(args.source_workspace),
                "workspace": str(args.workspace_root),
            }
        ]))
        """
    ),
    md("## Setup / Build Candidate Checkpoints"),
    code(
        """
        cmd = [
            sys.executable,
            str(RUNNER),
            "--workspace-root", str(WORKSPACE),
            "--source-workspace", str(SOURCE_WORKSPACE),
            "--candidate-betas", "0.25,0.50,0.75,1.00",
            "--residual-scope", "neck_head",
            "--include-bn", "true",
            "--client-limit", "1500",
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

        candidates = pd.read_csv(WORKSPACE / "stats" / "04_repair_shielded_candidate_checkpoints.csv")
        display(candidates)
        """
    ),
    md(
        """
        ## Run Full Evaluation

        This evaluates only the new 04 candidate checkpoints, then combines the
        results with the already-computed 03 baseline table.  It should be much
        shorter than rerunning 03 training, but it still runs the full
        scene/day-night evaluation protocol.
        """
    ),
    code(
        """
        RUN_EVALUATION = True
        CANDIDATE_BETAS = "0.25,0.50,0.75,1.00"
        VAL_BATCH_SIZE = 16
        DEVICE = ""

        if RUN_EVALUATION:
            cmd = [
                sys.executable,
                str(RUNNER),
                "--workspace-root", str(WORKSPACE),
                "--source-workspace", str(SOURCE_WORKSPACE),
                "--candidate-betas", CANDIDATE_BETAS,
                "--residual-scope", "neck_head",
                "--include-bn", "true",
                "--client-limit", "1500",
                "--val-batch-size", str(VAL_BATCH_SIZE),
                "--device", DEVICE,
                "--evaluate",
                "--classwise",
                "--no-eval-plots",
                "--notify",
            ]
            print(" ".join(cmd))
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        else:
            print("RUN_EVALUATION is False; candidate checkpoints were built only.")
        """
    ),
    md("## Final Metrics"),
    code(
        """
        final_metrics = WORKSPACE / "stats" / "04_repair_shielded_final_metrics.csv"
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
        split_metrics = WORKSPACE / "stats" / "04_repair_shielded_split_metrics.csv"
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
        report = WORKSPACE / "04_repair_shielded_local_expert_report.md"
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
