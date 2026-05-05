#!/usr/bin/env python3
"""Generate notebook 03 for repair-oriented multi-round pseudo-GT."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "03_repair_oriented_multiround_pseudogt.ipynb"


def md(source: str) -> dict:
    source = dedent(source).strip()
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.splitlines()],
    }


def code(source: str) -> dict:
    source = dedent(source).strip()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.splitlines()],
    }


def main() -> int:
    cells = [
        md(
            """
            # 03 Repair-Oriented Multi-Round PseudoGT

            03 changes the objective.  The target is no longer "client pseudo-GT
            training alone should beat warmup."  The target is:

            > pseudo-GT client adaptation + aggregation + supervised server
            > repair should produce a repaired global model that improves or
            > plateaus across multiple rounds.
            """
        ),
        md(
            """
            ## Round Definition

            Each round is:

            1. generate stable pseudo labels from the current repaired global model
            2. train each client with source cloudy GT plus strict target pseudo labels
            3. aggregate client checkpoints
            4. repair the aggregate on supervised source cloudy GT
            5. evaluate the repaired global checkpoint
            6. carry the repaired global checkpoint into the next round

            The main checkpoint is the repaired global model, not the client or
            aggregate checkpoint.
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
            elif (cwd / "pseudogt_learnability").exists():
                PROJECT_ROOT = cwd / "pseudogt_learnability"
            else:
                PROJECT_ROOT = cwd
            os.chdir(PROJECT_ROOT)
            WORKSPACE = PROJECT_ROOT / "output" / "03_repair_oriented_multiround"
            print(PROJECT_ROOT)
            print(WORKSPACE)
            print((PROJECT_ROOT / "checkpoints" / "round000_warmup.pt").exists())
            """
        ),
        md(
            """
            ## Setup Check

            This creates/reuses the scene-client lists and copies the warmup
            checkpoint into the 03 workspace.
            """
        ),
        code(
            """
            subprocess.run([
                sys.executable,
                "scripts/run_pseudogt_learnability_03.py",
                "--workspace-root", str(WORKSPACE),
                "--setup-only",
            ], check=True)
            """
        ),
        md(
            """
            ## Run 03

            Defaults are intentionally more conservative than 02:

            - one client epoch per round
            - source GT repeated twice in client training
            - stricter stability/score thresholds
            - max 12 pseudo boxes per image
            - client checkpoints are not evaluated by default

            For a short smoke test, set `MAX_IMAGES_PER_CLIENT = 1200` and
            `ROUNDS = 2`.  For the real run, keep `MAX_IMAGES_PER_CLIENT = 0`.
            """
        ),
        code(
            """
            ROUNDS = 3
            MAX_IMAGES_PER_CLIENT = 0
            EVAL_CLIENTS = False

            cmd = [
                sys.executable,
                "scripts/run_pseudogt_learnability_03.py",
                "--workspace-root", str(WORKSPACE),
                "--variant", "repair_oriented_all_lowlr",
                "--rounds", str(ROUNDS),
                "--batch-size", "160",
                "--workers", "8",
                "--gpus", "2",
                "--master-port", "30531",
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

            Primary metrics are the repaired global model after each round.
            `aggregate_mAP` and `repair_gain` are diagnostics for how destructive
            the client update was and how much the source repair recovered.
            """
        ),
        code(
            """
            import json
            import pandas as pd

            metrics_csv = WORKSPACE / "stats" / "03_round_metrics.csv"
            metrics_json = WORKSPACE / "stats" / "03_round_metrics_summary.json"
            if metrics_csv.exists():
                metrics = pd.read_csv(metrics_csv)
                display(metrics)
            else:
                print("No round metrics yet:", metrics_csv)

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
            if not summary_csv.exists():
                print("Evaluation summary is not available yet:", summary_csv)
            else:
                df = pd.read_csv(summary_csv)
                ok = df[df["status"].eq("ok")].copy()
                total = ok[ok["split"].isin(["total", "scene_total"])].copy()
                total = total.sort_values("map50_95", ascending=False)
                display(total[["checkpoint_label", "split", "precision", "recall", "map50", "map50_95"]])
            """
        ),
        md(
            """
            ## Pseudo Label Stats
            """
        ),
        code(
            """
            stats_files = sorted((WORKSPACE / "stats").glob("03_round*_pseudo_label_stats.csv"))
            if not stats_files:
                print("No pseudo-label stats yet")
            else:
                stats = pd.concat([pd.read_csv(path) for path in stats_files], ignore_index=True)
                display(stats)
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
                sys.path.insert(0, str(PROJECT_ROOT.parent))
                from notebook_notify import notify_discord

                metric_line = ""
                if metrics_json.exists():
                    payload = json.loads(metrics_json.read_text())
                    metric_line = (
                        f"final mAP50-95={payload.get('final_repaired_map50_95')}, "
                        f"lastN avg={payload.get('last_n_avg_repaired_map50_95')}, "
                        f"lastN min={payload.get('last_n_min_repaired_map50_95')}"
                    )
                msg = "PseudoGT Learnability 03 finished. " + metric_line
                result = notify_discord(
                    msg,
                    title="PseudoGT Learnability 03",
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
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
