#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path


MOE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = MOE_ROOT / "notebooks" / "05_router_ten_loops.ipynb"


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
        # 05 Router-focused MoE x DQA Ten Loops

        This notebook continues the MoE direction after `04_ten_research_loops`.
        The previous result showed that day experts and neck/head day residuals
        beat the single DQA aggregate.  This run asks a more router-specific
        question:

        - Can virtual routing explain the remaining gap?
        - Can low-anchor expert mixing keep the best expert performance?
        - Do BN statistics matter for expert behavior?
        - Is top-k expert weighting better than uniform expert averaging?
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
        if cwd.name == "notebooks" and cwd.parent.name == "moe":
            MOE_ROOT = cwd.parent
        elif (cwd / "dynamic_quality_aware_classwise_aggregation").exists():
            MOE_ROOT = cwd / "dynamic_quality_aware_classwise_aggregation" / "scene_daynight_dqa" / "moe"
        else:
            MOE_ROOT = cwd

        SCENE_ROOT = MOE_ROOT.parent
        WORKSPACE = MOE_ROOT / "output" / "05_router_ten_loops"
        SOURCE_WORKSPACE = SCENE_ROOT / "output" / "02_head_to_full_long_dqa"
        PREV_LOOP_WORKSPACE = MOE_ROOT / "output" / "04_ten_research_loops"
        RUNNER = MOE_ROOT / "scripts" / "run_moe_05_router_ten_loops.py"

        print("MOE_ROOT", MOE_ROOT)
        print("SOURCE_WORKSPACE", SOURCE_WORKSPACE)
        print("PREV_LOOP_WORKSPACE", PREV_LOOP_WORKSPACE)
        print("WORKSPACE", WORKSPACE)
        print("RUNNER", RUNNER)
        """
    ),
    md("## Setup / Sanity Check"),
    code(
        """
        spec = importlib.util.spec_from_file_location("run_moe_05_router_ten_loops", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)

        args = runner.parse_args([
            "--workspace-root", str(WORKSPACE),
            "--source-workspace", str(SOURCE_WORKSPACE),
            "--prev-loop-workspace", str(PREV_LOOP_WORKSPACE),
            "--setup-only",
        ])
        runner.run(args)
        """
    ),
    md("## Execute Ten Router Loops"),
    code(
        """
        cmd = [
            sys.executable,
            str(RUNNER),
            "--workspace-root", str(WORKSPACE),
            "--source-workspace", str(SOURCE_WORKSPACE),
            "--prev-loop-workspace", str(PREV_LOOP_WORKSPACE),
            "--client-limit", "1500",
            "--evaluate",
            "--classwise",
            "--no-eval-plots",
            "--notify",
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=MOE_ROOT, check=True)
        """
    ),
    md("## Results"),
    code(
        """
        metrics_path = WORKSPACE / "stats" / "05_router_ten_loop_metrics.csv"
        log_path = WORKSPACE / "stats" / "05_router_ten_loop_log.csv"

        metrics = pd.read_csv(metrics_path)
        display(
            metrics.sort_values("map50_95", ascending=False)[
                [
                    "loop_id",
                    "checkpoint_label",
                    "map50",
                    "map50_95",
                    "gain_vs_warmup_map50_95",
                    "night_avg_map50_95",
                    "worst_split",
                    "worst_split_map50_95",
                    "variant",
                ]
            ].head(40)
        )

        loop_log = pd.read_csv(log_path)
        display(loop_log)
        """
    ),
    md("## Markdown Report"),
    code(
        """
        report = WORKSPACE / "05_router_ten_loop_report.md"
        print(report)
        print(report.read_text(encoding="utf-8")[:5000])
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
