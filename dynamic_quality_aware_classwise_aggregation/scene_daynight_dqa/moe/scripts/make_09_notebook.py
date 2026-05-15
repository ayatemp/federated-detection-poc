#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path


MOE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = MOE_ROOT / "notebooks" / "09_counterfactual_view_expert_probe.ipynb"


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
        # 09 Counterfactual View Expert Probe

        05 failed because pseudoGT routing became an easy-domain filter.  This
        notebook tests a different pseudoGT x MoE idea:

        **Do not create experts by preserving a domain bucket.  Create experts
        by the view condition that made a pseudo box appear.**

        A night box that only appears after illumination enhancement is not
        merely a low-confidence night sample.  It is an
        `illumination_rescued` pseudoGT expert sample.
        """
    ),
    md(
        """
        ## What This Runs

        The bounded probe uses the 03 BN-residual DQA aggregate checkpoint.

        For each sampled client image it predicts six views:

        - original
        - original horizontal flip
        - brightness enhanced
        - brightness enhanced horizontal flip
        - CLAHE enhanced
        - CLAHE enhanced horizontal flip

        Boxes are clustered back in original coordinates and split into:

        - `clean_original`: stable in original views
        - `illumination_rescued`: not stable in original, stable in enhanced views
        - `cross_view_bridge`: appears in both original and enhanced views, but not enough original views alone

        The notebook optionally trains a short neck/head probe on the
        `illumination_rescued` expert dataset.
        """
    ),
    code(
        """
        from pathlib import Path
        import json
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
        WORKSPACE = MOE_ROOT / "output" / "09_counterfactual_view_expert_probe"
        RUNNER = MOE_ROOT / "scripts" / "run_moe_09_counterfactual_view_expert_probe.py"

        print("MOE_ROOT", MOE_ROOT)
        print("WORKSPACE", WORKSPACE)
        print("RUNNER", RUNNER)
        """
    ),
    md("## Execute Bounded Probe"),
    code(
        """
        cmd = [
            sys.executable,
            str(RUNNER),
            "--workspace-root", str(WORKSPACE),
            "--max-images-per-client", "80",
            "--train-probe",
            "--evaluate",
            "--notify",
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=MOE_ROOT, check=True)
        """
    ),
    md("## Counterfactual View Statistics"),
    code(
        """
        client_stats = pd.read_csv(WORKSPACE / "stats" / "09_view_expert_probe_client_stats.csv")
        display(client_stats)

        summary = json.loads((WORKSPACE / "stats" / "09_view_expert_probe_summary.json").read_text(encoding="utf-8"))
        print(json.dumps(summary["totals"], indent=2, ensure_ascii=False))
        print(json.dumps(summary["day_night_signal"], indent=2, ensure_ascii=False))
        """
    ),
    md("## Training Probe Metrics"),
    code(
        """
        metrics_path = WORKSPACE / "stats" / "09_training_probe_metrics.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            display(metrics)
        else:
            train_summary_path = WORKSPACE / "stats" / "09_training_probe_summary.json"
            print(train_summary_path.read_text(encoding="utf-8") if train_summary_path.exists() else "No training summary yet.")
        """
    ),
    md("## Report"),
    code(
        """
        report = WORKSPACE / "09_counterfactual_view_expert_probe_report.md"
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
