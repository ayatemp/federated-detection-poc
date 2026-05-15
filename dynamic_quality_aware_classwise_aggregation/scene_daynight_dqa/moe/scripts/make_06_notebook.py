#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path


MOE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = MOE_ROOT / "notebooks" / "06_spatial_expert_fifteen_loops.ipynb"


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
        # 06 Local-Region Expert MoE x DQA Fifteen Loops

        This notebook tests the next MoE idea:

        > Do not mix models at the client level.  Grow experts around local
        > pseudo-GT regions that are actually learnable.

        This is a fast design/evidence loop, not fifteen full YOLO trainings.
        It uses the completed `03_main_bn_residual_dqa_experiment` and the
        previous MoE router results to rank fifteen local-region expert
        hypotheses, then writes the concrete next full experiment candidate.
        """
    ),
    md(
        """
        ## Research Seeds

        - PSSFL/FedMox: spatial router + Soft-Mixture for practical
          semi-supervised federated object detection.
        - Soft MoE: differentiable soft assignment instead of brittle hard
          token routing.
        - Expert Choice Routing: experts choose fixed-capacity tokens, which
          maps naturally to pseudo-GT quota control.
        - MMoE / PLE: shared-private experts for task/domain relationship
          modeling.
        - DAMEX: detection can benefit from dataset/domain-aware MoE, but
          routing should avoid expert collapse.
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
        WORKSPACE = MOE_ROOT / "output" / "06_spatial_expert_fifteen_loops"
        SOURCE_WORKSPACE = SCENE_ROOT / "output" / "03_main_bn_residual_dqa_experiment"
        PREV_MOE_WORKSPACE = MOE_ROOT / "output" / "05_router_ten_loops"
        RUNNER = MOE_ROOT / "scripts" / "run_moe_06_spatial_expert_fifteen_loops.py"

        print("MOE_ROOT", MOE_ROOT)
        print("SOURCE_WORKSPACE", SOURCE_WORKSPACE)
        print("PREV_MOE_WORKSPACE", PREV_MOE_WORKSPACE)
        print("WORKSPACE", WORKSPACE)
        print("RUNNER", RUNNER)
        """
    ),
    md("## Execute Fifteen Screening Loops"),
    code(
        """
        cmd = [
            sys.executable,
            str(RUNNER),
            "--workspace-root", str(WORKSPACE),
            "--source-workspace", str(SOURCE_WORKSPACE),
            "--prev-moe-workspace", str(PREV_MOE_WORKSPACE),
            "--notify",
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=MOE_ROOT, check=True)
        """
    ),
    md("## Split Evidence"),
    code(
        """
        split_evidence = pd.read_csv(WORKSPACE / "stats" / "06_spatial_expert_split_evidence.csv")
        display(split_evidence)
        """
    ),
    md("## Fifteen Loop Scoreboard"),
    code(
        """
        scoreboard = pd.read_csv(WORKSPACE / "stats" / "06_spatial_expert_scoreboard.csv")
        display(
            scoreboard[
                [
                    "loop_id",
                    "rank_score",
                    "screened_projected_map50_95",
                    "screened_delta_map50_95",
                    "confidence",
                    "implementation_change",
                    "rationale",
                ]
            ]
        )
        """
    ),
    md("## Explicit Fifteen-loop Trace"),
    code(
        """
        trace = pd.read_csv(WORKSPACE / "stats" / "06_spatial_expert_loop_trace.csv")
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
    md("## Selected Full Experiment Candidate"),
    code(
        """
        import json

        candidate_path = WORKSPACE / "stats" / "06_selected_full_experiment_candidate.json"
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        print(json.dumps(candidate, indent=2, ensure_ascii=False))
        """
    ),
    md("## Markdown Report"),
    code(
        """
        report = WORKSPACE / "06_spatial_expert_fifteen_loop_report.md"
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
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
