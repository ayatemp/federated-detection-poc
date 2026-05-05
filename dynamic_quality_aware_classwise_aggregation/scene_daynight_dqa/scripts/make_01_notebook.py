#!/usr/bin/env python3
"""Generate the first scene-daynight DQA notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_repair_oriented_scene_daynight_dqa.ipynb"


def md(source: str) -> dict:
    source = dedent(source).strip()
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.splitlines()]}


def code(source: str) -> dict:
    source = dedent(source).strip()
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in source.splitlines()]}


def main() -> int:
    cells = [
        md(
            """
            # 01 Repair-Oriented Scene-Daynight DQA

            This notebook starts the new organized DQA line.  It keeps the old
            top-level DQA notebooks untouched and runs the next experiment under
            `scene_daynight_dqa/output`.
            """
        ),
        md(
            """
            ## Objective

            We move from 3 scene clients to 6 clients:

            - highway_day / highway_night
            - citystreet_day / citystreet_night
            - residential_day / residential_night

            The method is repair-oriented: client pseudoGT adaptation is allowed
            to be imperfect, but the repaired global checkpoint after supervised
            source repair is the main score.
            """
        ),
        code(
            """
            from pathlib import Path
            import os
            import subprocess
            import sys

            cwd = Path.cwd().resolve()
            if cwd.name == "notebooks":
                PROJECT_ROOT = cwd.parent
            elif (cwd / "dynamic_quality_aware_classwise_aggregation").exists():
                PROJECT_ROOT = cwd / "dynamic_quality_aware_classwise_aggregation" / "scene_daynight_dqa"
            else:
                PROJECT_ROOT = cwd
            os.chdir(PROJECT_ROOT)
            WORKSPACE = PROJECT_ROOT / "output" / "01_repair_oriented_scene_daynight_dqa"
            print(PROJECT_ROOT)
            print(WORKSPACE)
            """
        ),
        md(
            """
            ## Setup Check

            This builds the six client lists and the six validation splits.  The
            default client cap is 1500 images per client to keep clients balanced.
            """
        ),
        code(
            """
            CLIENT_LIMIT = 1500
            subprocess.run([
                sys.executable,
                "scripts/run_scene_daynight_dqa_01.py",
                "--workspace-root", str(WORKSPACE),
                "--client-limit", str(CLIENT_LIMIT),
                "--setup-only",
            ], check=True)
            """
        ),
        md(
            """
            ## Run 01

            Defaults:

            - 3 rounds
            - 6 clients
            - repair-oriented stable pseudoGT
            - DQA-CWA v2 aggregation
            - supervised server repair after every aggregate

            For a quick smoke test, set `MAX_IMAGES_PER_CLIENT = 300` and
            `ROUNDS = 1`.
            """
        ),
        code(
            """
            ROUNDS = 3
            MAX_IMAGES_PER_CLIENT = 0
            EVAL_CLIENTS = False

            cmd = [
                sys.executable,
                "scripts/run_scene_daynight_dqa_01.py",
                "--workspace-root", str(WORKSPACE),
                "--client-limit", str(CLIENT_LIMIT),
                "--rounds", str(ROUNDS),
                "--batch-size", "160",
                "--workers", "8",
                "--gpus", "2",
                "--master-port", "30641",
                "--server-repair-epochs", "1",
                "--evaluate",
                "--classwise",
                "--no-eval-plots",
            ]
            if MAX_IMAGES_PER_CLIENT > 0:
                cmd.extend(["--max-images-per-client", str(MAX_IMAGES_PER_CLIENT)])
            if EVAL_CLIENTS:
                cmd.append("--eval-clients")
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
            """
        ),
        md(
            """
            ## Round Metrics
            """
        ),
        code(
            """
            import json
            import pandas as pd

            metrics_csv = WORKSPACE / "stats" / "01_round_metrics.csv"
            metrics_json = WORKSPACE / "stats" / "01_round_metrics_summary.json"
            if metrics_csv.exists():
                display(pd.read_csv(metrics_csv))
            else:
                print("No metrics yet:", metrics_csv)
            if metrics_json.exists():
                print(json.dumps(json.loads(metrics_json.read_text()), indent=2, ensure_ascii=False))
            """
        ),
        md(
            """
            ## Evaluation Summary
            """
        ),
        code(
            """
            summary_csv = WORKSPACE / "validation_reports" / "paper_protocol_eval_summary.csv"
            if summary_csv.exists():
                df = pd.read_csv(summary_csv)
                ok = df[df["status"].eq("ok")].copy()
                total = ok[ok["split"].isin(["total", "scene_daynight_total"])].copy()
                display(total.sort_values("map50_95", ascending=False)[
                    ["checkpoint_label", "split", "precision", "recall", "map50", "map50_95"]
                ])
                scene = ok[~ok["split"].isin(["total", "scene_daynight_total"])].copy()
                display(scene.sort_values(["split", "map50_95"], ascending=[True, False])[
                    ["checkpoint_label", "split", "precision", "recall", "map50", "map50_95"]
                ])
            else:
                print("No evaluation summary yet:", summary_csv)
            """
        ),
        md(
            """
            ## DQA State
            """
        ),
        code(
            """
            state_path = WORKSPACE / "stats" / "01_dqa_state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                print(json.dumps({
                    "last_sources": state.get("last_sources"),
                    "last_active_classes": state.get("last_active_classes"),
                    "config": state.get("config"),
                }, indent=2, ensure_ascii=False))
            else:
                print("No DQA state yet:", state_path)
            """
        ),
        md(
            """
            ## Discord Notification
            """
        ),
        code(
            """
            try:
                sys.path.insert(0, str(PROJECT_ROOT.parents[1]))
                from notebook_notify import notify_discord

                metric_line = ""
                if metrics_json.exists():
                    payload = json.loads(metrics_json.read_text())
                    metric_line = (
                        f"final mAP50-95={payload.get('final_repaired_map50_95')}, "
                        f"lastN avg={payload.get('last_n_avg_repaired_map50_95')}, "
                        f"lastN min={payload.get('last_n_min_repaired_map50_95')}"
                    )
                result = notify_discord(
                    "Scene-Daynight DQA 01 finished. " + metric_line,
                    title="Scene-Daynight DQA 01",
                    context={"project": str(PROJECT_ROOT), "workspace": str(WORKSPACE)},
                    fail_silently=True,
                )
                print(result)
            except Exception as exc:
                print("Discord notification skipped:", exc)
            """
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
