#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path


MOE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = MOE_ROOT / "notebooks" / "03_five_research_loops.ipynb"


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
        # 03 MoE x DQA Five Research Loops

        This notebook is the explicit five-loop sprint:

        1. **FedMox post-hoc**: already executed in MoE/02. Read its result as
           Loop 1.
        2. **Repair residual reinjection**: test whether server repair erases
           useful target residuals.
        3. **FedBN-style BN transplant**: test whether feature-shift batchnorm
           statistics are harming day/night performance.
        4. **Client expert oracle**: evaluate client checkpoints as experts to
           see whether a router has anything useful to choose from.
        5. **DQA re-aggregation sweep**: test whether aggregation policy is the
           immediate bottleneck.

        The goal is not to pretend these are the final paper method.  The goal
        is to make the loop structure explicit and decide what deserves a full
        long training run.
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
        WORKSPACE = MOE_ROOT / "output" / "03_five_research_loops"
        SOURCE_WORKSPACE = SCENE_ROOT / "output" / "02_head_to_full_long_dqa"
        LOOP1_WORKSPACE = MOE_ROOT / "output" / "02_fedmox_posthoc_five_loop"
        RUNNER = MOE_ROOT / "scripts" / "run_moe_03_five_research_loops.py"

        print("MOE_ROOT", MOE_ROOT)
        print("SOURCE_WORKSPACE", SOURCE_WORKSPACE)
        print("LOOP1_WORKSPACE", LOOP1_WORKSPACE)
        print("WORKSPACE", WORKSPACE)
        print("RUNNER", RUNNER)
        """
    ),
    md("## Setup / Sanity Check"),
    code(
        """
        spec = importlib.util.spec_from_file_location("run_moe_03_five_research_loops", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)

        args = runner.parse_args([
            "--workspace-root", str(WORKSPACE),
            "--source-workspace", str(SOURCE_WORKSPACE),
            "--loop1-workspace", str(LOOP1_WORKSPACE),
            "--setup-only",
        ])
        runner.run(args)
        """
    ),
    md("## Execute Loops 2-5 and Combine with Loop 1"),
    code(
        """
        cmd = [
            sys.executable,
            str(RUNNER),
            "--workspace-root", str(WORKSPACE),
            "--source-workspace", str(SOURCE_WORKSPACE),
            "--loop1-workspace", str(LOOP1_WORKSPACE),
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
        metrics_path = WORKSPACE / "stats" / "03_five_research_loop_metrics.csv"
        log_path = WORKSPACE / "stats" / "03_five_research_loop_log.csv"

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
            ].head(30)
        )

        loop_log = pd.read_csv(log_path)
        display(loop_log)
        """
    ),
    md("## Markdown Report"),
    code(
        """
        report = WORKSPACE / "03_five_research_loop_report.md"
        print(report)
        print(report.read_text(encoding="utf-8")[:4000])
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
