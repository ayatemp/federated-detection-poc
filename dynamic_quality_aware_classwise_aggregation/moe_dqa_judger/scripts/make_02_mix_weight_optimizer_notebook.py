#!/usr/bin/env python3
"""Create the DQA-SoftMoX mix-weight optimizer notebook."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "02_mix_weight_optimizer.ipynb"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def dedent(source: str) -> str:
    return textwrap.dedent(source).strip() + "\n"


def main() -> None:
    setup_code = dedent(
        r"""
        from __future__ import annotations

        import subprocess
        import sys
        from datetime import datetime, timezone
        from pathlib import Path

        import pandas as pd

        REPO_ROOT = Path.cwd().resolve()
        if not (REPO_ROOT / "dynamic_quality_aware_classwise_aggregation").exists():
            REPO_ROOT = Path("/app/Object_Detection")

        JUDGER_ROOT = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation" / "moe_dqa_judger"
        RUNNER = JUDGER_ROOT / "scripts" / "run_02_mix_weight_optimizer.py"
        WORKSPACE = JUDGER_ROOT / "output" / "02_mix_weight_optimizer"
        LOG_DIR = JUDGER_ROOT / "logs"
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        print("REPO_ROOT:", REPO_ROOT)
        print("RUNNER:", RUNNER, RUNNER.exists())
        print("WORKSPACE:", WORKSPACE)
        """
    )

    design_code = dedent(
        r"""
        design = pd.DataFrame(
            [
                {
                    "part": "candidate models",
                    "choice": "G_t / A_t / S_t",
                    "reason": "Keep the previous global anchor, the DQA aggregate, and the server-repaired model as the only three ingredients.",
                },
                {
                    "part": "granularity",
                    "choice": "body / head / moe",
                    "reason": "Approximate pFedLA-style layer-wise aggregation while keeping the search small enough for YOLO checkpoints.",
                },
                {
                    "part": "learning signal",
                    "choice": "mini total-val mAP surrogate",
                    "reason": "AdaMerging/model-soup style black-box coefficient learning without adding an external teacher.",
                },
                {
                    "part": "optimizer",
                    "choice": "RF surrogate + Dirichlet local search",
                    "reason": "FedAWA/FedLAW-like adaptive aggregation, but learned from observed validation response instead of fixed hand weights.",
                },
                {
                    "part": "guardrail",
                    "choice": "full total-val for top candidates",
                    "reason": "Mini split is only for cheap search; the selected checkpoints are verified on the paper total split.",
                },
            ]
        )
        display(design)
        """
    )

    run_code = dedent(
        r"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"02_mix_weight_optimizer_{timestamp}.log"
        cmd = [
            sys.executable,
            str(RUNNER),
            "--workspace-root", str(WORKSPACE),
            "--rounds", "1,2",
            "--mini-images", "384",
            "--random-candidates", "6",
            "--surrogate-iterations", "2",
            "--surrogate-pool", "48",
            "--surrogate-evals", "3",
            "--full-eval-topk", "2",
            "--val-batch-size", "32",
            "--force",
        ]
        print(" ".join(cmd))
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log.write(proc.stdout)
        print("returncode:", proc.returncode)
        print("log:", log_path)
        print(proc.stdout[-6000:])
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)
        """
    )

    inspect_code = dedent(
        r"""
        trials_csv = WORKSPACE / "stats" / "02_mix_weight_optimizer_trials.csv"
        best_csv = WORKSPACE / "stats" / "02_mix_weight_optimizer_best_full.csv"
        trials = pd.read_csv(trials_csv)
        best = pd.read_csv(best_csv)

        display(best[[
            "round", "candidate_id", "map50", "map50_95", "precision", "recall", "score",
            "body_g", "body_a", "body_s", "head_g", "head_a", "head_s", "moe_g", "moe_a", "moe_s",
        ]].head(12))

        mini_cols = [
            "round", "candidate_id", "phase", "map50", "map50_95", "precision", "recall", "score",
            "body_g", "body_a", "body_s", "head_g", "head_a", "head_s", "moe_g", "moe_a", "moe_s",
        ]
        display(
            trials[trials["eval_scope"].eq("mini")]
            .sort_values(["round", "score"], ascending=[True, False])
            [mini_cols]
            .groupby("round")
            .head(8)
        )
        print("report:", WORKSPACE / "02_mix_weight_optimizer_report.md")
        """
    )

    expanded_code = dedent(
        r"""
        # Optional: turn this on after the first optimizer pass if the mini/full ranking looks stable.
        RUN_EXPANDED_SEARCH = False

        if RUN_EXPANDED_SEARCH:
            expanded_workspace = JUDGER_ROOT / "output" / "02_mix_weight_optimizer_expanded"
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            log_path = LOG_DIR / f"02_mix_weight_optimizer_expanded_{timestamp}.log"
            cmd = [
                sys.executable,
                str(RUNNER),
                "--workspace-root", str(expanded_workspace),
                "--rounds", "1,2,3,4,5",
                "--mini-images", "768",
                "--random-candidates", "12",
                "--surrogate-iterations", "3",
                "--surrogate-pool", "96",
                "--surrogate-evals", "4",
                "--full-eval-topk", "3",
                "--val-batch-size", "32",
                "--force",
            ]
            print(" ".join(cmd))
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                log.write(proc.stdout)
            print("returncode:", proc.returncode)
            print("log:", log_path)
            print(proc.stdout[-6000:])
            if proc.returncode != 0:
                raise SystemExit(proc.returncode)
        """
    )

    nb = {
        "cells": [
            markdown_cell(
                "# 02 DQA-SoftMoX Mix Weight Optimizer\n\n"
                "Learn module-wise `G_t / A_t / S_t` mixing coefficients with a small black-box optimizer."
            ),
            code_cell(setup_code),
            markdown_cell("## Design"),
            code_cell(design_code),
            markdown_cell(
                "## Round 1-2 Optimizer\n\n"
                "This pass searches the first two rounds on a mini total split, then verifies top candidates on the full total split."
            ),
            code_cell(run_code),
            markdown_cell("## Results"),
            code_cell(inspect_code),
            markdown_cell("## Optional Expanded Search"),
            code_cell(expanded_code),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
