#!/usr/bin/env python3
"""Generate notebook 02 for the PseudoGT Learnability project."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_stable_pseudogt_learnability.ipynb"


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
            # 02 Stable PseudoGT Learnability

            01 still used the online EfficientTeacher confidence pipeline.  02
            changes the pseudo-GT source itself: a single warmup teacher predicts
            each target image under two views, identity and horizontal flip, and
            only de-augmented boxes that remain spatially stable are written as
            offline YOLO labels.
            """
        ),
        md(
            """
            ## Hypothesis

            Phase2 gets worse because confidence is an insufficient localization
            quality signal.  A pseudo box should supervise bbox regression only
            when the teacher localizes it consistently under a label-preserving
            augmentation.  The training set is therefore:

            - source labeled cloudy images, as the same supervised anchor used by
              FedSTO/DQA clients
            - stable target pseudo labels from the same scene-client target lists
            - no second teacher and no GT-based checkpoint selection
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
            WORKSPACE = PROJECT_ROOT / "output" / "02_stable_pseudogt"
            print(PROJECT_ROOT)
            print(WORKSPACE)
            print((PROJECT_ROOT / "checkpoints" / "round000_warmup.pt").exists())
            """
        ),
        md(
            """
            ## Setup Check

            This creates/reuses the scene-client lists and copies the warmup
            checkpoint into the 02 workspace.
            """
        ),
        code(
            """
            subprocess.run([
                sys.executable,
                "scripts/run_pseudogt_learnability_02.py",
                "--workspace-root", str(WORKSPACE),
                "--setup-only",
            ], check=True)
            """
        ),
        md(
            """
            ## Run 02

            `MAX_IMAGES_PER_CLIENT = 0` means the full 5000-image target list per
            client.  For a short probe, set it to something like `1200` and rerun
            with `--force-pseudo` when changing thresholds.
            """
        ),
        code(
            """
            MAX_IMAGES_PER_CLIENT = 0
            VARIANTS = "stable_mix_backbone,stable_mix_all_lowlr"

            cmd = [
                sys.executable,
                "scripts/run_pseudogt_learnability_02.py",
                "--workspace-root", str(WORKSPACE),
                "--variants", VARIANTS,
                "--batch-size", "160",
                "--workers", "8",
                "--gpus", "2",
                "--master-port", "30421",
                "--server-repair-epochs", "1",
                "--evaluate",
                "--classwise",
                "--no-eval-plots",
            ]
            if MAX_IMAGES_PER_CLIENT > 0:
                cmd.extend(["--max-images-per-client", str(MAX_IMAGES_PER_CLIENT)])
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
            """
        ),
        md(
            """
            ## Pseudo Label Stats

            The first thing to inspect is not mAP, but whether each client keeps a
            reasonable number of stable boxes without being dominated by one easy
            class.
            """
        ),
        code(
            """
            import json
            import pandas as pd

            stats_csv = WORKSPACE / "stats" / "02_pseudo_label_stats.csv"
            stats_json = WORKSPACE / "stats" / "02_pseudo_label_stats.json"
            if stats_csv.exists():
                stats = pd.read_csv(stats_csv)
                display(stats)
            else:
                print("No pseudo-label stats yet:", stats_csv)

            if stats_json.exists():
                payload = json.loads(stats_json.read_text())
                for client, row in payload.get("clients", {}).items():
                    print(client, row.get("class_counts", {}))
            """
        ),
        md(
            """
            ## Evaluation Summary

            Main success criterion: an aggregate or repaired checkpoint should
            beat `warmup_global` on `total` mAP@0.5:0.95.  Secondary criterion:
            a client checkpoint improves its matching scene split without a large
            total collapse.
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

                warm = total[total["checkpoint_label"].eq("warmup_global")]
                if not warm.empty:
                    base = float(warm.iloc[0]["map50_95"])
                    total["delta_map50_95_vs_warmup"] = total["map50_95"].astype(float) - base
                    display(total[["checkpoint_label", "map50", "map50_95", "delta_map50_95_vs_warmup"]])
            """
        ),
        md(
            """
            ## Scene-Specific View
            """
        ),
        code(
            """
            if summary_csv.exists():
                scene_rows = ok[ok["split"].isin(["highway", "citystreet", "residential"])].copy()
                display(scene_rows.sort_values(["split", "map50_95"], ascending=[True, False])[
                    ["checkpoint_label", "split", "precision", "recall", "map50", "map50_95"]
                ])
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

                msg = "PseudoGT Learnability 02 finished. Check output/02_stable_pseudogt/validation_reports/paper_protocol_eval_summary.md"
                result = notify_discord(
                    msg,
                    title="PseudoGT Learnability 02",
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
