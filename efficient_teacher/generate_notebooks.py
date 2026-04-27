#!/usr/bin/env python3
"""Generate the EfficientTeacher single-client notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


def replace_placeholders(cells: list[dict], replacements: dict[str, str]) -> list[dict]:
    for cell in cells:
        source = cell.get("source")
        if not isinstance(source, list):
            continue
        next_source = []
        for line in source:
            for key, value in replacements.items():
                line = line.replace(key, value)
            next_source.append(line)
        cell["source"] = next_source
    return cells


def write_notebook(path: Path, cells: list[dict]) -> None:
    payload = {
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
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def training_notebook(
    *,
    title: str = "01 EfficientTeacher Single-Client Training",
    intro: str = "This notebook runs the EfficientTeacher baseline under the same practical conditions as the guarded DQA run: YOLOv5L, BDD100K paper20k, one pseudo-label client epoch per round, and one labeled server-GT epoch after every round. The only intentional change is `NUM_CLIENTS = 1`.",
    work_dir_name: str = "efficientteacher_single_client",
    local_ema: bool = False,
    master_port: int = 29520,
) -> list[dict]:
    cells = [
        md(
            """
            # __TITLE__

            __INTRO__
            """
        ),
        code(
            """
            from __future__ import annotations

            import json
            import subprocess
            import sys
            from datetime import datetime, timezone
            from pathlib import Path
            from typing import Optional

            import pandas as pd


            def find_repo_root(start: Optional[Path] = None) -> Path:
                start = Path.cwd().resolve() if start is None else Path(start).resolve()
                required = (
                    "efficient_teacher/run_efficient_teacher_single_client.py",
                    "navigating_data_heterogeneity/setup_fedsto_exact_reproduction.py",
                    "navigating_data_heterogeneity/vendor/efficientteacher/train.py",
                )
                candidates = []
                for base in (start, *start.parents):
                    candidates.extend([base, base / "Object_Detection"])
                for candidate in candidates:
                    if all((candidate / marker).exists() for marker in required):
                        return candidate.resolve()
                raise FileNotFoundError("Could not locate the Object_Detection repository root.")


            REPO_ROOT = find_repo_root()
            ET_ROOT = REPO_ROOT / "efficient_teacher"
            DQA_ROOT = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation"
            RUN_SCRIPT = ET_ROOT / "run_efficient_teacher_single_client.py"
            WORK_ROOT = ET_ROOT / "__WORK_DIR_NAME__"
            PSEUDO_STATS_ROOT = WORK_ROOT / "pseudo_stats"
            DQA_WARMUP_CHECKPOINT = (
                DQA_ROOT
                / "efficientteacher_dqa_cwa_corrected_12h"
                / "global_checkpoints"
                / "round000_warmup.pt"
            )
            PYTHON_BIN = Path(sys.executable)
            TRAIN_LOG = WORK_ROOT / "efficientteacher_latest.log"

            pd.options.display.max_columns = 200
            print("repo:", REPO_ROOT)
            print("workspace:", WORK_ROOT)
            """
        ),
        md(
            """
            ## 1. DQA-Matched Configuration

            The defaults mirror the current guarded DQA notebook. Change `CLIENT_ID` only if you want a different single weather client.
            """
        ),
        code(
            """
            CLIENT_ID = 0
            CLIENT_WEATHER = None
            WARMUP_EPOCHS = 15
            PHASE1_ROUNDS = 14
            PHASE2_ROUNDS = 27
            BATCH_SIZE = 64
            WORKERS = 0
            REQUESTED_GPUS = 2
            MASTER_PORT = __MASTER_PORT__
            MIN_FREE_GIB = 70
            LOCAL_EMA = __LOCAL_EMA__
            USE_DQA_WARMUP_CHECKPOINT = True
            WARMUP_CHECKPOINT = DQA_WARMUP_CHECKPOINT if USE_DQA_WARMUP_CHECKPOINT else None
            APPEND_TRAIN_LOG = False
            EXTRA_RUN_ARGS = []

            try:
                import torch

                AVAILABLE_CUDA_GPUS = torch.cuda.device_count()
            except Exception as exc:
                AVAILABLE_CUDA_GPUS = 0
                print("Could not inspect CUDA devices:", exc)

            GPUS = min(REQUESTED_GPUS, AVAILABLE_CUDA_GPUS) if AVAILABLE_CUDA_GPUS else 1
            if GPUS != REQUESTED_GPUS:
                print(f"Requested {REQUESTED_GPUS} GPU(s), but {AVAILABLE_CUDA_GPUS} visible; using {GPUS}.")

            config_summary = {
                "client_id": CLIENT_ID,
                "client_weather": CLIENT_WEATHER or "from manifest",
                "warmup_epochs": WARMUP_EPOCHS,
                "phase1_rounds": PHASE1_ROUNDS,
                "phase2_rounds": PHASE2_ROUNDS,
                "batch_size": BATCH_SIZE,
                "workers": WORKERS,
                "gpus": GPUS,
                "local_ema": LOCAL_EMA,
                "warmup_checkpoint": str(WARMUP_CHECKPOINT) if WARMUP_CHECKPOINT else None,
                "workspace": str(WORK_ROOT),
            }
            config_summary
            """
        ),
        md(
            """
            ## 2. Build Data Lists and Configs

            This uses the same `setup_fedsto_exact_reproduction.py` path as DQA/FedSTO, but patches the client list down to one selected client inside the runner.
            """
        ),
        code(
            """
            setup_cmd = [
                str(PYTHON_BIN),
                str(RUN_SCRIPT),
                "--setup-only",
                "--workspace-root",
                str(WORK_ROOT),
                "--pseudo-stats-root",
                str(PSEUDO_STATS_ROOT),
                "--client-id",
                str(CLIENT_ID),
            ]
            if CLIENT_WEATHER:
                setup_cmd.extend(["--client-weather", CLIENT_WEATHER])

            subprocess.run(setup_cmd, cwd=REPO_ROOT, check=True)

            manifest_path = WORK_ROOT / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_summary = {
                "classes": manifest["classes"],
                "server_train_images": manifest["server"]["train_images"],
                "server_val_images": manifest["server"]["val_images"],
                "clients": manifest["clients"],
                "paper_schedule": manifest["paper_schedule"],
            }
            manifest_summary
            """
        ),
        md(
            """
            ## 3. Dependency and Dry-Run Check

            The dry run generates runtime commands and YAMLs without launching training.
            """
        ),
        code(
            """
            import importlib.util

            required = {
                "yaml": "PyYAML",
                "cv2": "opencv-python",
                "thop": "thop",
                "tensorboard": "tensorboard",
                "sklearn": "scikit-learn",
                "pandas": "pandas",
            }
            missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
            if missing:
                raise ModuleNotFoundError("Missing runtime dependencies: " + ", ".join(missing))

            dry_run_cmd = [
                str(PYTHON_BIN),
                str(RUN_SCRIPT),
                "--dry-run",
                "--workspace-root",
                str(WORK_ROOT),
                "--pseudo-stats-root",
                str(PSEUDO_STATS_ROOT),
                "--client-id",
                str(CLIENT_ID),
                "--warmup-epochs",
                str(WARMUP_EPOCHS),
                "--phase1-rounds",
                str(PHASE1_ROUNDS),
                "--phase2-rounds",
                str(PHASE2_ROUNDS),
                "--batch-size",
                str(BATCH_SIZE),
                "--workers",
                str(WORKERS),
                "--gpus",
                str(GPUS),
                "--master-port",
                str(MASTER_PORT),
                "--min-free-gib",
                str(MIN_FREE_GIB),
            ]
            if CLIENT_WEATHER:
                dry_run_cmd.extend(["--client-weather", CLIENT_WEATHER])
            dry_run_cmd.append("--local-ema" if LOCAL_EMA else "--no-local-ema")
            if WARMUP_CHECKPOINT is not None:
                dry_run_cmd.extend(["--warmup-checkpoint", str(WARMUP_CHECKPOINT)])

            subprocess.run(dry_run_cmd, cwd=REPO_ROOT, check=True)
            """
        ),
        md(
            """
            ## 4. Start or Resume Training

            This cell is restartable. The runner reuses valid checkpoints, appends compact status to notebook output, and writes full EfficientTeacher logs to `efficientteacher_latest.log`.

            When `WARMUP_CHECKPOINT` points at the DQA warm-up checkpoint, the run skips raw-pretrained warm-up training and seeds `round000_warmup.pt` from that DQA artifact instead. If this workspace is already running with a raw-pretrained warm-up, stop that old process before rerunning this cell.
            """
        ),
        code(
            """
            RUN_FULL_REPRODUCTION = True
            ALLOW_CPU_TRAINING = False
            FORCE_RESTART = False
            FORCE_WARMUP = False
            FORCE_RETRAIN = False

            train_cmd = [
                str(PYTHON_BIN),
                "-u",
                str(RUN_SCRIPT),
                "--workspace-root",
                str(WORK_ROOT),
                "--pseudo-stats-root",
                str(PSEUDO_STATS_ROOT),
                "--client-id",
                str(CLIENT_ID),
                "--warmup-epochs",
                str(WARMUP_EPOCHS),
                "--phase1-rounds",
                str(PHASE1_ROUNDS),
                "--phase2-rounds",
                str(PHASE2_ROUNDS),
                "--batch-size",
                str(BATCH_SIZE),
                "--workers",
                str(WORKERS),
                "--gpus",
                str(GPUS),
                "--master-port",
                str(MASTER_PORT),
                "--min-free-gib",
                str(MIN_FREE_GIB),
            ]
            if CLIENT_WEATHER:
                train_cmd.extend(["--client-weather", CLIENT_WEATHER])
            train_cmd.append("--local-ema" if LOCAL_EMA else "--no-local-ema")
            if WARMUP_CHECKPOINT is not None:
                train_cmd.extend(["--warmup-checkpoint", str(WARMUP_CHECKPOINT)])
            if APPEND_TRAIN_LOG:
                train_cmd.append("--append-train-log")
            if FORCE_RESTART:
                train_cmd.append("--force-restart")
            if FORCE_WARMUP:
                train_cmd.append("--force-warmup")
            if FORCE_RETRAIN:
                train_cmd.append("--force-retrain")
            train_cmd.extend(EXTRA_RUN_ARGS)

            if RUN_FULL_REPRODUCTION and AVAILABLE_CUDA_GPUS < 1 and not ALLOW_CPU_TRAINING:
                print("No CUDA GPU is visible, so the run was not started.")
            elif RUN_FULL_REPRODUCTION:
                print("starting:", " ".join(train_cmd))
                subprocess.run(train_cmd, cwd=REPO_ROOT, check=True)
            else:
                print("Set RUN_FULL_REPRODUCTION = True to launch or resume training.")

            print("train log:", TRAIN_LOG)
            """
        ),
        md(
            """
            ## 5. Progress Snapshot
            """
        ),
        code(
            """
            def modified_utc(path: Path) -> str:
                if not path.exists():
                    return ""
                return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


            history_path = WORK_ROOT / "history.json"
            history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
            completed_phase1 = sum(1 for entry in history if int(entry.get("phase", 0)) == 1)
            completed_phase2 = sum(1 for entry in history if int(entry.get("phase", 0)) == 2)
            display(
                pd.DataFrame(
                    [
                        {"phase": "phase1", "completed": completed_phase1, "target": PHASE1_ROUNDS},
                        {"phase": "phase2", "completed": completed_phase2, "target": PHASE2_ROUNDS},
                    ]
                )
            )

            artifacts = [
                WORK_ROOT / "manifest.json",
                WORK_ROOT / "history.json",
                WORK_ROOT / "global_checkpoints" / "round000_warmup.pt",
                WORK_ROOT / "global_checkpoints" / f"phase2_round{PHASE2_ROUNDS:03d}_global.pt",
                TRAIN_LOG,
                PSEUDO_STATS_ROOT,
            ]
            display(
                pd.DataFrame(
                    [
                        {
                            "path": str(path),
                            "exists": path.exists(),
                            "bytes": path.stat().st_size if path.exists() and path.is_file() else "",
                            "modified_utc": modified_utc(path),
                        }
                        for path in artifacts
                    ]
                )
            )
            display(pd.DataFrame(history).tail(10) if history else pd.DataFrame())
            """
        ),
        md(
            """
            ## 6. Recent Logs
            """
        ),
        code(
            """
            def tail(path: Path, lines: int = 80) -> str:
                if not path.exists():
                    return f"{path} does not exist yet."
                result = subprocess.run(["tail", "-n", str(lines), str(path)], capture_output=True, text=True)
                return result.stdout


            print(tail(TRAIN_LOG, 80))
            """
        ),
    ]
    return replace_placeholders(
        cells,
        {
            "__TITLE__": title,
            "__INTRO__": intro,
            "__WORK_DIR_NAME__": work_dir_name,
            "__LOCAL_EMA__": str(bool(local_ema)),
            "__MASTER_PORT__": str(master_port),
        },
    )


def localema_training_notebook() -> list[dict]:
    return training_notebook(
        title="00 LocalEMA Single-Client Training",
        intro=(
            "This notebook runs the LocalEMA comparison under the same DQA-matched "
            "single-client EfficientTeacher protocol. The only method change versus "
            "`01_efficient_teacher_training.ipynb` is that each client's previous EMA "
            "teacher is carried into the next round."
        ),
        work_dir_name="efficientteacher_localema",
        local_ema=True,
        master_port=29521,
    )


def evaluation_notebook() -> list[dict]:
    return [
        md(
            """
            # 01_2 EfficientTeacher Single-Client Evaluation

            Read-only analysis for the single-client EfficientTeacher run. It summarizes training curves, pseudo-label quality/counts, paper-protocol validation, and optional comparisons against DQA/FedSTO artifacts when they exist.
            """
        ),
        code(
            """
            from __future__ import annotations

            import json
            import re
            import subprocess
            import sys
            from datetime import datetime, timezone
            from pathlib import Path
            from typing import Optional

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import Image as NotebookImage, display

            try:
                import seaborn as sns
            except ModuleNotFoundError:
                sns = None


            def find_repo_root(start: Optional[Path] = None) -> Path:
                start = Path.cwd().resolve() if start is None else Path(start).resolve()
                required = (
                    "efficient_teacher/run_efficient_teacher_single_client.py",
                    "navigating_data_heterogeneity/evaluate_paper_protocol.py",
                )
                candidates = []
                for base in (start, *start.parents):
                    candidates.extend([base, base / "Object_Detection"])
                for candidate in candidates:
                    if all((candidate / marker).exists() for marker in required):
                        return candidate.resolve()
                raise FileNotFoundError("Could not locate the Object_Detection repository root.")


            REPO_ROOT = find_repo_root()
            ET_ROOT = REPO_ROOT / "efficient_teacher"
            WORK_ROOT = ET_ROOT / "efficientteacher_single_client"
            LOCALEMA_WORK_ROOT = ET_ROOT / "efficientteacher_localema"
            RUNS_ROOT = WORK_ROOT / "runs"
            PSEUDO_STATS_ROOT = WORK_ROOT / "pseudo_stats"
            VALIDATION_ROOT = WORK_ROOT / "validation_reports"
            EVAL_SCRIPT = ET_ROOT / "evaluate_paper_protocol.py"
            PYTHON_BIN = Path(sys.executable)
            METHOD_WORKSPACES = {
                "EfficientTeacher-1C": WORK_ROOT,
                "LocalEMA-1C": LOCALEMA_WORK_ROOT,
            }

            DQA_PAPER_EVAL = (
                REPO_ROOT
                / "dynamic_quality_aware_classwise_aggregation"
                / "efficientteacher_dqa_cwa_corrected_12h"
                / "validation_reports"
                / "paper_protocol_eval_summary.csv"
            )
            FEDSTO_PAPER_EVAL = (
                REPO_ROOT
                / "navigating_data_heterogeneity"
                / "efficientteacher_fedsto"
                / "validation_reports"
                / "paper_protocol_eval_summary.csv"
            )

            if sns is not None:
                sns.set_theme(style="whitegrid", context="talk")
            else:
                plt.style.use("ggplot")
            pd.options.display.max_columns = 200
            print("workspace:", WORK_ROOT)
            print("localema workspace:", LOCALEMA_WORK_ROOT)
            """
        ),
        md(
            """
            ## 1. Artifact Status
            """
        ),
        code(
            """
            def modified_utc(path: Path) -> str:
                if not path.exists():
                    return ""
                return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


            manifest_path = WORK_ROOT / "manifest.json"
            history_path = WORK_ROOT / "history.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
            history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []

            artifacts = []
            for method, workspace in METHOD_WORKSPACES.items():
                artifacts.extend(
                    [
                        (method, "workspace", workspace),
                        (method, "manifest", workspace / "manifest.json"),
                        (method, "history", workspace / "history.json"),
                        (method, "runs", workspace / "runs"),
                        (method, "pseudo_stats", workspace / "pseudo_stats"),
                        (method, "global_checkpoints", workspace / "global_checkpoints"),
                        (method, "paper_eval_summary", workspace / "validation_reports" / "paper_protocol_eval_summary.csv"),
                    ]
                )
            display(
                pd.DataFrame(
                    [
                        {
                            "method": method,
                            "artifact": label,
                            "path": str(path),
                            "exists": path.exists(),
                            "modified_utc": modified_utc(path),
                        }
                        for method, label, path in artifacts
                    ]
                )
            )

            if manifest:
                display(
                    pd.DataFrame(
                        [
                            {
                                "server_train_images": manifest["server"]["train_images"],
                                "server_val_images": manifest["server"]["val_images"],
                                "client": manifest["clients"][0]["weather"] if manifest.get("clients") else "",
                                "client_images": manifest["clients"][0]["images"] if manifest.get("clients") else "",
                            }
                        ]
                    )
                )
            display(pd.DataFrame(history).tail(12) if history else pd.DataFrame())
            """
        ),
        md(
            """
            ## 2. Training Summary Table
            """
        ),
        code(
            """
            def parse_run_name(run_name: str) -> dict:
                if run_name == "runtime_server_warmup":
                    return {"phase": 0, "round": 0, "role": "warmup", "client_id": np.nan, "weather": "server_cloudy"}
                match = re.fullmatch(
                    r"et_phase(?P<phase>[12])_round(?P<round>\\d{3})_(?P<role>server|client(?P<client_id>\\d+)_(?P<weather>.+))",
                    run_name,
                )
                if not match:
                    return {"phase": np.nan, "round": np.nan, "role": "unknown", "client_id": np.nan, "weather": ""}
                role = "client" if match.group("role").startswith("client") else "server"
                return {
                    "phase": int(match.group("phase")),
                    "round": int(match.group("round")),
                    "role": role,
                    "client_id": int(match.group("client_id")) if match.group("client_id") else np.nan,
                    "weather": match.group("weather") or "server_cloudy",
                }


            def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
                rows = []
                summary_rows = []
                for result_path in sorted(RUNS_ROOT.glob("*/results.csv")):
                    run_name = result_path.parent.name
                    df = pd.read_csv(result_path, skipinitialspace=True).rename(columns=lambda col: col.strip())
                    if df.empty:
                        continue
                    meta = parse_run_name(run_name)
                    for _, row in df.iterrows():
                        rows.append({"run_name": run_name, **meta, **row.to_dict()})
                    last = df.iloc[-1].to_dict()
                    best_idx = df["metrics/mAP_0.5"].astype(float).idxmax() if "metrics/mAP_0.5" in df else df.index[-1]
                    best = df.loc[best_idx].to_dict()
                    summary_rows.append(
                        {
                            "run_name": run_name,
                            **meta,
                            "epochs_logged": len(df),
                            "final_epoch": last.get("epoch"),
                            "final_precision": last.get("metrics/precision"),
                            "final_recall": last.get("metrics/recall"),
                            "final_map50": last.get("metrics/mAP_0.5"),
                            "final_map50_95": last.get("metrics/mAP_0.5:0.95"),
                            "best_map50": best.get("metrics/mAP_0.5"),
                            "best_map50_epoch": best.get("epoch"),
                            "train_box_loss": last.get("train/box_loss"),
                            "train_obj_loss": last.get("train/obj_loss"),
                            "train_cls_loss": last.get("train/cls_loss"),
                        }
                    )
                return pd.DataFrame(rows), pd.DataFrame(summary_rows)


            results_df, summary_df = load_results()
            tables_root = VALIDATION_ROOT / "tables"
            tables_root.mkdir(parents=True, exist_ok=True)
            if not summary_df.empty:
                summary_df = summary_df.sort_values(["phase", "round", "role", "run_name"])
                summary_df.to_csv(tables_root / "training_run_summary.csv", index=False)
                display(summary_df.tail(20).round(4))
            else:
                print("No results.csv files found yet.")
            """
        ),
        md(
            """
            ## 3. ET vs LocalEMA Training Summary
            """
        ),
        code(
            """
            def load_results_for_workspace(workspace: Path, method: str) -> pd.DataFrame:
                rows = []
                for result_path in sorted((workspace / "runs").glob("*/results.csv")):
                    run_name = result_path.parent.name
                    df = pd.read_csv(result_path, skipinitialspace=True).rename(columns=lambda col: col.strip())
                    if df.empty:
                        continue
                    meta = parse_run_name(run_name)
                    last = df.iloc[-1].to_dict()
                    best_idx = df["metrics/mAP_0.5"].astype(float).idxmax() if "metrics/mAP_0.5" in df else df.index[-1]
                    best = df.loc[best_idx].to_dict()
                    rows.append(
                        {
                            "method": method,
                            "workspace": str(workspace),
                            "run_name": run_name,
                            **meta,
                            "epochs_logged": len(df),
                            "final_precision": last.get("metrics/precision"),
                            "final_recall": last.get("metrics/recall"),
                            "final_map50": last.get("metrics/mAP_0.5"),
                            "final_map50_95": last.get("metrics/mAP_0.5:0.95"),
                            "best_map50": best.get("metrics/mAP_0.5"),
                            "train_box_loss": last.get("train/box_loss"),
                            "train_obj_loss": last.get("train/obj_loss"),
                            "train_cls_loss": last.get("train/cls_loss"),
                        }
                    )
                return pd.DataFrame(rows)


            method_summaries = [load_results_for_workspace(workspace, method) for method, workspace in METHOD_WORKSPACES.items()]
            method_summary_df = pd.concat([df for df in method_summaries if not df.empty], ignore_index=True) if any(
                not df.empty for df in method_summaries
            ) else pd.DataFrame()
            if not method_summary_df.empty:
                method_summary_df["timeline"] = np.where(
                    method_summary_df["phase"] == 0,
                    0,
                    method_summary_df["phase"].astype(float) * 1000 + method_summary_df["round"].astype(float),
                )
                display(method_summary_df.sort_values(["method", "phase", "round", "role"]).tail(30).round(4))
            else:
                print("No ET/LocalEMA training summaries found yet.")
            """
        ),
        md(
            """
            ## 3. Metric Curves
            """
        ),
        code(
            """
            def plot_line(ax, data: pd.DataFrame, x: str, y: str, hue: str | None = None, marker: str = "o") -> None:
                if data.empty or y not in data:
                    ax.set_title(f"missing {y}")
                    return
                if sns is not None and hue is not None:
                    sns.lineplot(data=data, x=x, y=y, hue=hue, marker=marker, ax=ax)
                elif sns is not None:
                    sns.lineplot(data=data, x=x, y=y, marker=marker, ax=ax)
                else:
                    for label, part in data.groupby(hue) if hue else [("", data)]:
                        ax.plot(part[x], part[y], marker=marker, label=label)
                    if hue:
                        ax.legend()


            if not summary_df.empty:
                curve_df = summary_df.copy()
                curve_df["timeline"] = np.where(
                    curve_df["phase"] == 0,
                    0,
                    curve_df["phase"].astype(float) * 1000 + curve_df["round"].astype(float),
                )
                server_curve = curve_df[curve_df["role"].isin(["warmup", "server"])].copy()
                client_curve = curve_df[curve_df["role"].eq("client")].copy()

                fig, axes = plt.subplots(2, 2, figsize=(17, 11))
                plot_line(axes[0, 0], server_curve, "timeline", "final_map50", hue="role")
                plot_line(axes[0, 1], server_curve, "timeline", "final_map50_95", hue="role")
                plot_line(axes[1, 0], server_curve, "timeline", "final_precision", hue="role")
                plot_line(axes[1, 1], server_curve, "timeline", "final_recall", hue="role")
                axes[0, 0].set_title("server/warmup mAP@0.5")
                axes[0, 1].set_title("server/warmup mAP@0.5:0.95")
                axes[1, 0].set_title("server/warmup precision")
                axes[1, 1].set_title("server/warmup recall")
                plt.tight_layout()
                plt.show()

                if not client_curve.empty:
                    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
                    plot_line(axes[0], client_curve, "timeline", "final_map50", hue="weather")
                    plot_line(axes[1], client_curve, "timeline", "final_recall", hue="weather")
                    axes[0].set_title("client pseudo-label training mAP@0.5")
                    axes[1].set_title("client pseudo-label training recall")
                    plt.tight_layout()
                    plt.show()
            """
        ),
        md(
            """
            ## 4. ET vs LocalEMA Curves
            """
        ),
        code(
            """
            if not method_summary_df.empty:
                server_methods = method_summary_df[method_summary_df["role"].isin(["warmup", "server"])].copy()
                client_methods = method_summary_df[method_summary_df["role"].eq("client")].copy()
                if not server_methods.empty:
                    fig, axes = plt.subplots(1, 2, figsize=(17, 5))
                    plot_line(axes[0], server_methods, "timeline", "final_map50", hue="method")
                    plot_line(axes[1], server_methods, "timeline", "final_map50_95", hue="method")
                    axes[0].set_title("server/warmup mAP@0.5 by method")
                    axes[1].set_title("server/warmup mAP@0.5:0.95 by method")
                    plt.tight_layout()
                    plt.show()
                if not client_methods.empty:
                    fig, axes = plt.subplots(1, 2, figsize=(17, 5))
                    plot_line(axes[0], client_methods, "timeline", "final_map50", hue="method")
                    plot_line(axes[1], client_methods, "timeline", "final_recall", hue="method")
                    axes[0].set_title("client mAP@0.5 by method")
                    axes[1].set_title("client recall by method")
                    plt.tight_layout()
                    plt.show()
            """
        ),
        md(
            """
            ## 5. Loss Curves
            """
        ),
        code(
            """
            if not summary_df.empty:
                losses = ["train_box_loss", "train_obj_loss", "train_cls_loss"]
                plot_df = summary_df[summary_df["role"].isin(["client", "server"])].copy()
                plot_df["timeline"] = plot_df["phase"].astype(float) * 1000 + plot_df["round"].astype(float)
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                for ax, loss_col in zip(axes, losses):
                    plot_line(ax, plot_df, "timeline", loss_col, hue="role")
                    ax.set_title(loss_col)
                plt.tight_layout()
                plt.show()
            """
        ),
        md(
            """
            ## 6. Pseudo-Label Stats
            """
        ),
        code(
            """
            class_names = manifest.get("classes", [str(i) for i in range(10)])
            pseudo_summary_rows = []
            pseudo_class_rows = []
            pattern = re.compile(r"phase(?P<phase>[12])_round(?P<round>\\d{3})_client(?P<client>\\d+)_(?P<weather>.+)\\.json")

            for method, workspace in METHOD_WORKSPACES.items():
                for path in sorted((workspace / "pseudo_stats").glob("phase*_round*_client*.json")):
                    match = pattern.fullmatch(path.name)
                    if not match:
                        continue
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    counts = np.array(payload.get("counts", []), dtype=float)
                    mean_conf = np.array(payload.get("mean_confidences", [0] * len(counts)), dtype=float)
                    mean_quality = np.array(payload.get("mean_quality_scores", mean_conf), dtype=float)
                    total = counts.sum()
                    weighted_conf = float((counts * mean_conf).sum() / total) if total > 0 else 0.0
                    weighted_quality = float((counts * mean_quality).sum() / total) if total > 0 else 0.0
                    meta = {
                        "method": method,
                        "phase": int(match.group("phase")),
                        "round": int(match.group("round")),
                        "client": int(match.group("client")),
                        "weather": match.group("weather"),
                        "file": str(path),
                    }
                    pseudo_summary_rows.append(
                        {
                            **meta,
                            "total_pseudo": float(total),
                            "active_classes": int((counts > 0).sum()),
                            "weighted_confidence": weighted_conf,
                            "weighted_quality": weighted_quality,
                        }
                    )
                    for class_idx, count in enumerate(counts):
                        pseudo_class_rows.append(
                            {
                                **meta,
                                "class_idx": class_idx,
                                "class_name": class_names[class_idx] if class_idx < len(class_names) else str(class_idx),
                                "count": float(count),
                                "mean_confidence": float(mean_conf[class_idx]) if class_idx < len(mean_conf) else 0.0,
                                "mean_quality": float(mean_quality[class_idx]) if class_idx < len(mean_quality) else 0.0,
                            }
                        )

            pseudo_summary = pd.DataFrame(pseudo_summary_rows)
            pseudo_by_class = pd.DataFrame(pseudo_class_rows)
            if not pseudo_summary.empty:
                pseudo_summary["timeline"] = pseudo_summary["phase"] * 1000 + pseudo_summary["round"]
                pseudo_summary.to_csv(tables_root / "pseudo_label_round_summary.csv", index=False)
                pseudo_by_class.to_csv(tables_root / "pseudo_label_class_summary.csv", index=False)
                display(pseudo_summary.tail(20).round(4))

                fig, axes = plt.subplots(1, 3, figsize=(20, 5))
                plot_line(axes[0], pseudo_summary, "timeline", "total_pseudo", hue="method")
                plot_line(axes[1], pseudo_summary, "timeline", "weighted_confidence", hue="method")
                plot_line(axes[2], pseudo_summary, "timeline", "weighted_quality", hue="method")
                axes[0].set_title("pseudo-label count")
                axes[1].set_title("weighted confidence")
                axes[2].set_title("weighted quality")
                plt.tight_layout()
                plt.show()
            else:
                print("No pseudo-label stats found yet.")
            """
        ),
        md(
            """
            ## 7. Class-Wise Pseudo-Label Heatmap
            """
        ),
        code(
            """
            if not pseudo_by_class.empty:
                heatmap_df = pseudo_by_class.copy()
                heatmap_df["round_label"] = (
                    heatmap_df["method"].astype(str)
                    + "_p"
                    + heatmap_df["phase"].astype(str)
                    + "r"
                    + heatmap_df["round"].astype(str).str.zfill(3)
                )
                pivot = heatmap_df.pivot_table(index="class_name", columns="round_label", values="count", aggfunc="sum", fill_value=0)
                plt.figure(figsize=(max(12, 0.35 * pivot.shape[1]), 7))
                if sns is not None:
                    sns.heatmap(np.log1p(pivot), cmap="viridis", cbar_kws={"label": "log1p(count)"})
                else:
                    plt.imshow(np.log1p(pivot), aspect="auto", cmap="viridis")
                    plt.colorbar(label="log1p(count)")
                    plt.yticks(range(len(pivot.index)), pivot.index)
                    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
                plt.title("class-wise pseudo-label counts")
                plt.xlabel("round")
                plt.ylabel("class")
                plt.tight_layout()
                plt.show()
            """
        ),
        md(
            """
            ## 8. Paper-Protocol Evaluation

            Set `RUN_PAPER_EVAL = True` to run the shared per-weather validation for warmup, phase1 final, phase2 best, and phase2 final checkpoints.
            """
        ),
        code(
            """
            RUN_PAPER_EVAL = False
            PAPER_EVAL_SPLITS = "cloudy,overcast,rainy,snowy,total"
            PAPER_EVAL_BATCH_SIZE = 8

            eval_cmds = [
                [
                    str(PYTHON_BIN),
                    str(EVAL_SCRIPT),
                    "--workspace",
                    str(workspace),
                    "--splits",
                    PAPER_EVAL_SPLITS,
                    "--batch-size",
                    str(PAPER_EVAL_BATCH_SIZE),
                ]
                for workspace in METHOD_WORKSPACES.values()
                if workspace.exists()
            ]
            if RUN_PAPER_EVAL:
                for eval_cmd in eval_cmds:
                    subprocess.run(eval_cmd, cwd=REPO_ROOT, check=True)
            else:
                print("Set RUN_PAPER_EVAL = True to run:")
                for eval_cmd in eval_cmds:
                    print(" ".join(eval_cmd))
            """
        ),
        md(
            """
            ## 9. Paper-Protocol Results
            """
        ),
        code(
            """
            paper_summary_path = VALIDATION_ROOT / "paper_protocol_eval_summary.csv"
            paper_frames = []
            for method, workspace in METHOD_WORKSPACES.items():
                path = workspace / "validation_reports" / "paper_protocol_eval_summary.csv"
                if path.exists():
                    df = pd.read_csv(path)
                    df.insert(0, "method", method)
                    paper_frames.append(df)

            if paper_frames:
                paper_df = pd.concat(paper_frames, ignore_index=True)
                display(paper_df.round(4))
                ok_rows = paper_df[paper_df["status"].eq("ok")].copy()
                if not ok_rows.empty:
                    fig, axes = plt.subplots(1, 2, figsize=(17, 5))
                    if sns is not None:
                        sns.barplot(data=ok_rows, x="split", y="map50", hue="method", ax=axes[0])
                        sns.barplot(data=ok_rows, x="split", y="map50_95", hue="method", ax=axes[1])
                    else:
                        ok_rows.pivot_table(index="split", columns="method", values="map50").plot(kind="bar", ax=axes[0])
                        ok_rows.pivot_table(index="split", columns="method", values="map50_95").plot(kind="bar", ax=axes[1])
                    axes[0].set_title("paper eval mAP@0.5")
                    axes[1].set_title("paper eval mAP@0.5:0.95")
                    for ax in axes:
                        ax.tick_params(axis="x", rotation=20)
                    plt.tight_layout()
                    plt.show()
            else:
                print("No ET/LocalEMA paper-protocol summary CSVs yet.")
            """
        ),
        md(
            """
            ## 10. Optional Comparison with LocalEMA, DQA, and FedSTO
            """
        ),
        code(
            """
            frames = []
            for method, path in [
                ("EfficientTeacher-1C", paper_summary_path),
                ("LocalEMA-1C", LOCALEMA_WORK_ROOT / "validation_reports" / "paper_protocol_eval_summary.csv"),
                ("DQA-CWA", DQA_PAPER_EVAL),
                ("FedSTO", FEDSTO_PAPER_EVAL),
            ]:
                if path.exists():
                    df = pd.read_csv(path)
                    df.insert(0, "method", method)
                    frames.append(df)

            if frames:
                comparison = pd.concat(frames, ignore_index=True)
                display(comparison.round(4))
                ok = comparison[comparison["status"].eq("ok")].copy()
                if not ok.empty:
                    total = ok[ok["split"].eq("total")].copy()
                    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
                    if sns is not None:
                        sns.barplot(data=total, x="checkpoint_label", y="map50", hue="method", ax=axes[0])
                        sns.barplot(data=total, x="checkpoint_label", y="map50_95", hue="method", ax=axes[1])
                    else:
                        total.pivot(index="checkpoint_label", columns="method", values="map50").plot(kind="bar", ax=axes[0])
                        total.pivot(index="checkpoint_label", columns="method", values="map50_95").plot(kind="bar", ax=axes[1])
                    axes[0].set_title("total split mAP@0.5")
                    axes[1].set_title("total split mAP@0.5:0.95")
                    for ax in axes:
                        ax.tick_params(axis="x", rotation=35)
                    plt.tight_layout()
                    plt.show()
            else:
                print("No paper-eval summaries found for comparison yet.")
            """
        ),
        md(
            """
            ## 11. Validation Plot Artifacts
            """
        ),
        code(
            """
            plot_rows = []
            for method, workspace in METHOD_WORKSPACES.items():
                val_runs = workspace / "validation_reports" / "paper_protocol_val_runs"
                for run_dir in sorted(val_runs.glob("*")):
                    if not run_dir.is_dir():
                        continue
                    plot_rows.append(
                        {
                            "method": method,
                            "run": run_dir.name,
                            "PR_curve": (run_dir / "PR_curve.png").exists(),
                            "F1_curve": (run_dir / "F1_curve.png").exists(),
                            "P_curve": (run_dir / "P_curve.png").exists(),
                            "R_curve": (run_dir / "R_curve.png").exists(),
                            "confusion_matrix": (run_dir / "confusion_matrix.png").exists(),
                        }
                    )
            display(pd.DataFrame(plot_rows) if plot_rows else pd.DataFrame())

            first_pr = None
            for workspace in METHOD_WORKSPACES.values():
                val_runs = workspace / "validation_reports" / "paper_protocol_val_runs"
                first_pr = next(val_runs.glob("*/PR_curve.png"), None) if val_runs.exists() else None
                if first_pr is not None:
                    break
            if first_pr is not None:
                display(NotebookImage(filename=str(first_pr)))
            """
        ),
    ]


def main() -> None:
    write_notebook(ROOT / "00_localema_training.ipynb", localema_training_notebook())
    write_notebook(ROOT / "01_efficient_teacher_training.ipynb", training_notebook())
    write_notebook(ROOT / "01_2_efficient_teacher_evaluation.ipynb", evaluation_notebook())
    print("Wrote EfficientTeacher notebooks.")


if __name__ == "__main__":
    main()
