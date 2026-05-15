#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path


MOE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = MOE_ROOT / "notebooks" / "08_pseudogt_router_recovery_fifteen_loops.ipynb"


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
        # 08 pseudoGT Router Recovery Fifteen Loops

        This notebook continues the MoE x DQA research loop after the previous
        pseudoGT router underperformed the BN-residual DQA baseline.

        The specific failure we target here is not "MoE is bad"; it is that the
        pseudoGT router behaved too much like a hard filter.  The night/hard
        pseudo boxes shrank across rounds, so the router removed exactly the
        signal that DQA needed to learn from.

        This is a fast evidence-screening notebook, not fifteen full YOLO
        trainings.  It uses the completed 03/04/05 metrics and pseudo-label
        statistics to rank fifteen router recovery designs.
        """
    ),
    md(
        """
        ## Research Seeds

        - FedMox sparse/spatial router
        - Auxiliary-loss-free load balancing
        - BASE / Expert Choice balanced assignment
        - Soft Mixture of Experts
        - GRIN gradient-informed routing
        - Mixture-of-Depths and Router-Tuning
        - Vision MoE router studies
        - DeepSeekMoE fine-grained experts
        - CartesianMoE factorized routing
        - Soft Teacher style pseudo box stability

        The core translation for DQA is: use the router as an assignment and
        curriculum mechanism first, and only use it as a hard pseudoGT filter
        when there is clear evidence that a box is harmful.
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
        WORKSPACE = MOE_ROOT / "output" / "08_pseudogt_router_recovery_fifteen_loops"
        SOURCE_WORKSPACE = SCENE_ROOT / "output" / "03_main_bn_residual_dqa_experiment"
        ROUTER_WORKSPACE = SCENE_ROOT / "output" / "05_expert_choice_pseudogt_router_dqa"
        RUNNER = MOE_ROOT / "scripts" / "run_moe_08_pseudogt_router_recovery_fifteen_loops.py"

        print("MOE_ROOT", MOE_ROOT)
        print("WORKSPACE", WORKSPACE)
        print("SOURCE_WORKSPACE", SOURCE_WORKSPACE)
        print("ROUTER_WORKSPACE", ROUTER_WORKSPACE)
        print("RUNNER", RUNNER)
        """
    ),
    md("## Execute Fifteen Router-recovery Loops"),
    code(
        """
        cmd = [
            sys.executable,
            str(RUNNER),
            "--workspace-root", str(WORKSPACE),
            "--source-workspace", str(SOURCE_WORKSPACE),
            "--router-workspace", str(ROUTER_WORKSPACE),
            "--notify",
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=MOE_ROOT, check=True)
        """
    ),
    md("## Evidence Tables"),
    code(
        """
        split_evidence = pd.read_csv(WORKSPACE / "stats" / "08_router_split_evidence.csv")
        pseudo_evidence = pd.read_csv(WORKSPACE / "stats" / "08_router_pseudo_evidence.csv")

        display(split_evidence)
        display(pseudo_evidence)
        """
    ),
    md("## Scoreboard"),
    code(
        """
        scoreboard = pd.read_csv(WORKSPACE / "stats" / "08_pseudogt_router_recovery_scoreboard.csv")
        display(
            scoreboard[
                [
                    "loop_id",
                    "paper_seed",
                    "screened_projected_map50_95",
                    "screened_delta_map50_95",
                    "beats_previous_router",
                    "reaches_03_04_target",
                    "confidence",
                    "router_change",
                    "rationale",
                ]
            ]
        )
        """
    ),
    md("## Fifteen-loop Trace"),
    code(
        """
        trace = pd.read_csv(WORKSPACE / "stats" / "08_pseudogt_router_recovery_loop_trace.csv")
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

        selected_path = WORKSPACE / "stats" / "08_selected_router_candidate.json"
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        print(json.dumps(selected, indent=2, ensure_ascii=False))
        """
    ),
    md("## Markdown Report"),
    code(
        """
        report = WORKSPACE / "08_pseudogt_router_recovery_report.md"
        print(report)
        print(report.read_text(encoding="utf-8")[:9000])
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
