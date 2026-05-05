#!/usr/bin/env python3
"""Generate notebook 01 for the PseudoGT Learnability project."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_best_pseudogt_learnability.ipynb"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def main() -> int:
    cells = [
        md(
            """
            # 01 Best PseudoGT Learnability

            This notebook separates pseudo-GT learning from DQA aggregation.
            Every client starts from the copied 08_4 warmup checkpoint, then
            candidate pseudo-GT training profiles are compared on the same
            scene protocol used by the DQA experiments.
            """
        ),
        md(
            """
            ## Fixed condition

            - Start checkpoint: `checkpoints/round000_warmup.pt`
            - Output root: `output/`
            - Clients: `highway`, `citystreet`, `residential`
            - Evaluation: `highway`, `citystreet`, `residential`, `total`
            - Main question: can pseudo-GT training itself produce useful
              client updates before DQA-style class-wise aggregation is added?
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
            print(PROJECT_ROOT)
            print((PROJECT_ROOT / "checkpoints" / "round000_warmup.pt").exists())
            """
        ),
        md(
            """
            ## Setup check

            This creates/reuses scene data lists in `output/` and verifies that
            the copied warmup checkpoint is visible from the new project.
            """
        ),
        code(
            """
            subprocess.run([
                sys.executable,
                "scripts/run_pseudogt_learnability_01.py",
                "--setup-only",
            ], check=True)
            """
        ),
        md(
            """
            ## Run 01

            The default run tests three profiles:

            - `backbone_obj_safe`
            - `neck_head_high_precision`
            - `all_consistency_lowlr`

            It saves client checkpoints, aggregate checkpoints, one-epoch
            server-repair checkpoints, and then evaluates them scene-wise.
            """
        ),
        code(
            """
            cmd = [
                sys.executable,
                "scripts/run_pseudogt_learnability_01.py",
                "--variants", "backbone_obj_safe,neck_head_high_precision,all_consistency_lowlr",
                "--batch-size", "160",
                "--workers", "8",
                "--gpus", "2",
                "--master-port", "30341",
                "--server-repair-epochs", "1",
                "--evaluate",
                "--classwise",
                "--no-eval-plots",
            ]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
            """
        ),
        md(
            """
            ## Check generated checkpoints
            """
        ),
        code(
            """
            import pandas as pd

            checkpoint_table = PROJECT_ROOT / "output" / "stats" / "01_checkpoints.csv"
            if checkpoint_table.exists():
                ckpts = pd.read_csv(checkpoint_table)
                display(ckpts)
            else:
                print("No checkpoint table yet:", checkpoint_table)
            """
        ),
        md(
            """
            ## Evaluation summary

            The key metric is whether any pseudo-GT profile beats
            `warmup_global` on `total` mAP@0.5:0.95, or at least improves the
            matching client scene without damaging the total split.
            """
        ),
        code(
            """
            import pandas as pd

            summary_csv = PROJECT_ROOT / "output" / "validation_reports" / "paper_protocol_eval_summary.csv"
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
            ## Scene-specific client learnability

            This view is useful even if the aggregate is weak.  If a client
            improves its own scene split, pseudo-GT may still be learnable, and
            the aggregation rule is the next target.  If no client improves its
            own split, the pseudo-GT training recipe itself is the bottleneck.
            """
        ),
        code(
            """
            if summary_csv.exists():
                ok = pd.read_csv(summary_csv)
                ok = ok[ok["status"].eq("ok")].copy()
                scene_rows = ok[ok["split"].isin(["highway", "citystreet", "residential"])].copy()
                display(scene_rows.sort_values(["split", "map50_95"], ascending=[True, False])[
                    ["checkpoint_label", "split", "precision", "recall", "map50", "map50_95"]
                ])
            """
        ),
        md(
            """
            ## Discord notification
            """
        ),
        code(
            """
            try:
                sys.path.insert(0, str(PROJECT_ROOT.parent))
                from notebook_notify import notify_discord

                msg = "PseudoGT Learnability 01 finished. Check output/validation_reports/paper_protocol_eval_summary.md"
                result = notify_discord(
                    msg,
                    title="PseudoGT Learnability 01",
                    context={"project": str(PROJECT_ROOT)},
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

