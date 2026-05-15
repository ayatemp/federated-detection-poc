#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path


MOE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = MOE_ROOT / "notebooks" / "02_fedmox_posthoc_five_loop.ipynb"


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
        # 02 FedMox Post-hoc Five-loop Sweep

        The first checkpoint-level MoE run did not show a clear gain.  This
        notebook steps back and tests the FedMox-relevant pieces over the
        already trained `02_head_to_full_long_dqa` checkpoints before spending
        another full training day.

        FedMox's important ingredients are:

        - sparse spatial MoE/router in the detection head,
        - Soft-Mixture between the previous server model and aggregated client
          model,
        - task-head-focused adaptation rather than unrestricted pseudoGT
          drift.

        This notebook is not a true YOLO architectural MoE.  It is a fast
        five-plus-loop decision test:

        | loop | candidate |
        |---|---|
        | 1 | Phase1 Soft-Mixture alpha=0.70 |
        | 2 | Phase1 Soft-Mixture alpha=0.85 |
        | 3 | Phase1 class-only DQA blend=0.25 |
        | 4 | Phase1 class-only DQA blend=0.55 |
        | 5 | Phase1 night-only class DQA blend=0.55 |
        | 6 | Phase1 day-only class DQA blend=0.55 |
        | 7 | Phase2 Soft-Mixture alpha=0.90 |

        The success criterion is not just total mAP.  We also inspect day/night
        averages and worst split, because the current DQA failure is mostly
        specialization being erased or harming night.
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
        WORKSPACE = MOE_ROOT / "output" / "02_fedmox_posthoc_five_loop"
        SOURCE_WORKSPACE = SCENE_ROOT / "output" / "02_head_to_full_long_dqa"
        RUNNER = MOE_ROOT / "scripts" / "run_moe_02_fedmox_posthoc_five_loop.py"

        print("MOE_ROOT", MOE_ROOT)
        print("SOURCE_WORKSPACE", SOURCE_WORKSPACE)
        print("WORKSPACE", WORKSPACE)
        print("RUNNER", RUNNER)
        """
    ),
    md("## Setup / Sanity Check"),
    code(
        """
        spec = importlib.util.spec_from_file_location("run_moe_02_fedmox_posthoc_five_loop", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)

        args = runner.parse_args([
            "--workspace-root", str(WORKSPACE),
            "--source-workspace", str(SOURCE_WORKSPACE),
            "--setup-only",
        ])
        runner.run(args)

        source_table = SOURCE_WORKSPACE / "stats" / "02_head_to_full_checkpoints.csv"
        print("source table exists:", source_table.exists(), source_table)
        """
    ),
    md("## Execute Five-plus Loop Evaluation"),
    code(
        """
        cmd = [
            sys.executable,
            str(RUNNER),
            "--workspace-root", str(WORKSPACE),
            "--source-workspace", str(SOURCE_WORKSPACE),
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
        metrics_path = WORKSPACE / "stats" / "02_fedmox_posthoc_metrics.csv"
        split_path = WORKSPACE / "stats" / "02_fedmox_posthoc_split_metrics.csv"

        metrics = pd.read_csv(metrics_path)
        display(
            metrics.sort_values("map50_95", ascending=False)[
                [
                    "checkpoint_label",
                    "map50",
                    "map50_95",
                    "gain_vs_warmup_map50_95",
                    "gain_vs_normal02_phase2_repair_map50_95",
                    "worst_split",
                    "worst_split_map50_95",
                    "day_avg_map50_95",
                    "night_avg_map50_95",
                    "day_night_gap_map50_95",
                    "variant",
                ]
            ]
        )

        split_metrics = pd.read_csv(split_path)
        display(
            split_metrics.pivot_table(
                index="checkpoint_label",
                columns="split",
                values="map50_95",
                aggfunc="first",
            )
        )
        """
    ),
]


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = {
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
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
