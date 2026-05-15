#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path


MOE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = MOE_ROOT / "notebooks" / "07_non_residual_moe_theory_loops.ipynb"


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
        # 07 Non-residual MoE Theory Loops

        This notebook intentionally leaves the checkpoint/residual family.
        The goal is to mine recent MoE ideas from LLM, vision, retrieval, and
        dynamic-compute papers and translate them into DQA hypotheses.

        The output is a fifteen-loop screening table and a selected next
        non-residual full-design candidate.
        """
    ),
    md(
        """
        ## Research Seeds

        - DeepSeek / auxiliary-loss-free load balancing
        - DeepSeekMoE fine-grained expert segmentation
        - BASE / Expert Choice balanced assignment
        - GRIN gradient-informed routing
        - Mixture-of-Depths and Router-Tuning dynamic compute
        - CartesianMoE factorized routing
        - V-MoE adaptive per-image compute
        - RouterRetriever / routing consistency ideas
        """
    ),
    code(
        """
        from pathlib import Path
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
        WORKSPACE = MOE_ROOT / "output" / "07_non_residual_moe_theory_loops"
        SOURCE_WORKSPACE = SCENE_ROOT / "output" / "03_main_bn_residual_dqa_experiment"
        RUNNER = MOE_ROOT / "scripts" / "run_moe_07_non_residual_moe_theory_loops.py"

        print("MOE_ROOT", MOE_ROOT)
        print("WORKSPACE", WORKSPACE)
        print("SOURCE_WORKSPACE", SOURCE_WORKSPACE)
        print("RUNNER", RUNNER)
        """
    ),
    md("## Execute Fifteen Non-residual Loops"),
    code(
        """
        cmd = [
            sys.executable,
            str(RUNNER),
            "--workspace-root", str(WORKSPACE),
            "--source-workspace", str(SOURCE_WORKSPACE),
            "--notify",
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=MOE_ROOT, check=True)
        """
    ),
    md("## Scoreboard"),
    code(
        """
        scoreboard = pd.read_csv(WORKSPACE / "stats" / "07_non_residual_moe_theory_scoreboard.csv")
        display(
            scoreboard[
                [
                    "loop_id",
                    "paper_seed",
                    "screened_projected_map50_95",
                    "screened_delta_map50_95",
                    "rank_score",
                    "confidence",
                    "dqa_translation",
                    "rationale",
                ]
            ]
        )
        """
    ),
    md("## Fifteen-loop Trace"),
    code(
        """
        trace = pd.read_csv(WORKSPACE / "stats" / "07_non_residual_moe_theory_loop_trace.csv")
        display(
            trace[
                [
                    "loop_index",
                    "loop_id",
                    "step_1_research",
                    "step_5_execution",
                    "step_6_result_summary",
                    "step_7_next_direction",
                ]
            ]
        )
        """
    ),
    md("## Selected Candidate"),
    code(
        """
        import json

        selected_path = WORKSPACE / "stats" / "07_selected_non_residual_candidate.json"
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        print(json.dumps(selected, indent=2, ensure_ascii=False))
        """
    ),
    md("## Markdown Report"),
    code(
        """
        report = WORKSPACE / "07_non_residual_moe_theory_report.md"
        print(report)
        print(report.read_text(encoding="utf-8")[:7000])
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
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
