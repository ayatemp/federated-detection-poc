#!/usr/bin/env python3
"""Generate the DQA-CWA notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
GENERATOR_PATH = ROOT / "generate_dqa_cwa_notebook.py"


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


def background_run_markdown() -> dict:
    return md(
        """
        ## 4. Start or Resume True DQA

        This launches the runner in the background like the FedSTO notebook does. The runner keeps EfficientTeacher's verbose train logs in `dqa_cwa_latest.log` and writes compact runner messages to the workspace-specific runner log.
        """
    )


def background_run_code(run_default: bool) -> dict:
    text = """
    RUN_DQA = __RUN_DEFAULT__
    ALLOW_CPU_TRAINING = False
    FORCE_RESTART = False
    FORCE_WARMUP = False
    FORCE_RETRAIN = False


    def read_pid(path: Path) -> int | None:
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None


    def pid_state(pid: int | None) -> str:
        if pid is None:
            return "missing"
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True,
            text=True,
        )
        state = result.stdout.strip()
        if result.returncode != 0 or not state:
            return "missing"
        if "Z" in state:
            return "zombie"
        return state


    existing_pid = read_pid(PID_PATH)
    existing_state = pid_state(existing_pid)

    cmd = [
        str(PYTHON_BIN),
        "-u",
        str(RUN_SCRIPT),
        "--workspace-root",
        str(WORK_ROOT),
        "--stats-root",
        str(STATS_ROOT),
        "--warmup-epochs",
        str(WARMUP_EPOCHS),
        "--phase1-rounds",
        str(PHASE1_ROUNDS),
        "--phase2-rounds",
        str(PHASE2_ROUNDS),
        "--dqa-start-phase",
        str(DQA_START_PHASE),
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
    if APPEND_TRAIN_LOG:
        cmd.append("--append-train-log")
    if FORCE_RESTART:
        cmd.append("--force-restart")
    if FORCE_WARMUP:
        cmd.append("--force-warmup")
    if FORCE_RETRAIN:
        cmd.append("--force-retrain")
    cmd.extend(EXTRA_RUN_ARGS)

    if existing_state not in {"missing", "zombie"}:
        print(f"DQA runner already active with PID {existing_pid} (state={existing_state}).")
    elif RUN_DQA and AVAILABLE_CUDA_GPUS < 1 and not ALLOW_CPU_TRAINING:
        print(
            "No CUDA GPU is visible, so the DQA run was not started. "
            "Use a GPU runtime, or set ALLOW_CPU_TRAINING = True if this is only a tiny debug run."
        )
    elif RUN_DQA:
        if PID_PATH.exists():
            PID_PATH.unlink()
        RUNNER_LOG.parent.mkdir(parents=True, exist_ok=True)
        launch = "cd {cwd} && setsid env PYTHONUNBUFFERED=1 {cmd} > {log} 2>&1 < /dev/null & echo $!".format(
            cwd=shlex.quote(str(REPO_ROOT)),
            cmd=" ".join(shlex.quote(part) for part in cmd),
            log=shlex.quote(str(RUNNER_LOG)),
        )
        pid = subprocess.check_output(["bash", "-lc", launch], text=True).strip()
        PID_PATH.write_text(pid + "\\n", encoding="utf-8")
        print("Started true DQA runner:", pid)
    else:
        print("Set RUN_DQA = True and rerun this cell to start or resume true DQA.")

    print("runner log:", RUNNER_LOG)
    print("train log:", TRAIN_LOG)
    """
    return code(text.replace("__RUN_DEFAULT__", str(run_default)))


def blocking_run_markdown() -> dict:
    return md(
        """
        ## 4. Full DQA Reproduction Run

        This notebook is meant to work cleanly with `Run All`: the cell below runs true DQA in the foreground with compact status updates, waits for completion, and leaves the later cells free to inspect metrics and evaluation outputs.
        """
    )


def blocking_run_code(run_default: bool) -> dict:
    text = """
    import re
    import time
    from collections import deque

    RUN_FULL_REPRODUCTION = __RUN_DEFAULT__
    ALLOW_CPU_TRAINING = False
    FORCE_RESTART = False
    FORCE_WARMUP = False
    FORCE_RETRAIN = False
    STATUS_EVERY_SECONDS = 60


    def read_pid(path: Path) -> int | None:
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None


    def pid_state(pid: int | None) -> str:
        if pid is None:
            return "missing"
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True,
            text=True,
        )
        state = result.stdout.strip()
        if result.returncode != 0 or not state:
            return "missing"
        if "Z" in state:
            return "zombie"
        return state


    cmd = [
        str(PYTHON_BIN),
        "-u",
        str(RUN_SCRIPT),
        "--workspace-root",
        str(WORK_ROOT),
        "--stats-root",
        str(STATS_ROOT),
        "--warmup-epochs",
        str(WARMUP_EPOCHS),
        "--phase1-rounds",
        str(PHASE1_ROUNDS),
        "--phase2-rounds",
        str(PHASE2_ROUNDS),
        "--dqa-start-phase",
        str(DQA_START_PHASE),
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
    if APPEND_TRAIN_LOG:
        cmd.append("--append-train-log")
    if FORCE_RESTART:
        cmd.append("--force-restart")
    if FORCE_WARMUP:
        cmd.append("--force-warmup")
    if FORCE_RETRAIN:
        cmd.append("--force-retrain")
    cmd.extend(EXTRA_RUN_ARGS)

    IMPORTANT_OUTPUT = re.compile(
        r"(Resuming DQA-CWA after|No completed DQA-CWA federated rounds|Current global checkpoint|"
        r"Reusing completed (warm-up|DQA-CWA client run|DQA-CWA server run|DQA-CWA global checkpoint)|"
        r"Recovered DQA-CWA phase|Completed DQA-CWA phase|All requested DQA-CWA federated rounds|"
        r"Dry run complete|Training failed|Traceback|RuntimeError|Exception|Error|out of memory|CUDA error)",
        re.IGNORECASE,
    )


    def compact_line(line: str, limit: int = 240) -> str:
        text = line.replace("\\r", "").strip()
        return text if len(text) <= limit else text[: limit - 3] + "..."


    if RUN_FULL_REPRODUCTION and AVAILABLE_CUDA_GPUS < 1 and not ALLOW_CPU_TRAINING:
        print(
            "No CUDA GPU is visible, so the full DQA run was not started. "
            "Use a GPU runtime, or set ALLOW_CPU_TRAINING = True if this is only a tiny debug run."
        )
    elif RUN_FULL_REPRODUCTION:
        existing_pid = read_pid(PID_PATH)
        existing_state = pid_state(existing_pid)
        if existing_state not in {"missing", "zombie"}:
            raise RuntimeError(
                f"DQA runner is already active with PID {existing_pid} (state={existing_state}). "
                "Stop it or reuse that run before launching another foreground run."
            )

        RUNNER_LOG.parent.mkdir(parents=True, exist_ok=True)
        print("Running:", " ".join(cmd))
        print("Runner log:", RUNNER_LOG)
        print("Train log:", TRAIN_LOG)

        recent = deque(maxlen=20)
        last_status = time.monotonic()
        with RUNNER_LOG.open("a", encoding="utf-8", buffering=1) as runner_log:
            runner_log.write("\\n\\n===== DQA-CWA notebook run resumed =====\\n")
            runner_log.write("Running: " + " ".join(cmd) + "\\n")
            process = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            PID_PATH.write_text(str(process.pid) + "\\n", encoding="utf-8")
            assert process.stdout is not None
            for line in process.stdout:
                runner_log.write(line)
                recent.append(line)
                if IMPORTANT_OUTPUT.search(line):
                    print(compact_line(line))
                    last_status = time.monotonic()
                elif time.monotonic() - last_status >= STATUS_EVERY_SECONDS:
                    print(f"[running] full DQA log is still updating: {RUNNER_LOG}")
                    last_status = time.monotonic()
            return_code = process.wait()
        if PID_PATH.exists():
            PID_PATH.unlink()
        if return_code != 0:
            print("Last captured lines:")
            for line in recent:
                text = compact_line(line)
                if text:
                    print(text)
            raise RuntimeError(
                f"DQA runner exited with status {return_code}. "
                f"See {RUNNER_LOG} and {TRAIN_LOG}."
            )
        print("DQA run completed successfully.")
    else:
        print("Set RUN_FULL_REPRODUCTION = True and rerun this cell.")
    """
    return code(text.replace("__RUN_DEFAULT__", str(run_default)))


def eval_code(eval_default: bool) -> dict:
    text = """
    RUN_EVAL = __RUN_EVAL__
    EVAL_SPLITS = "cloudy,overcast,rainy,snowy,total"
    EVAL_EXTRA_ARGS: list[str] = []

    eval_cmd = [
        str(PYTHON_BIN),
        str(EVAL_SCRIPT),
        "--workspace",
        str(WORK_ROOT),
        "--splits",
        EVAL_SPLITS,
    ]
    eval_cmd.extend(EVAL_EXTRA_ARGS)

    has_eval_checkpoint = any((WORK_ROOT / "global_checkpoints").glob("*.pt"))
    if RUN_EVAL and has_eval_checkpoint:
        subprocess.run(eval_cmd, cwd=REPO_ROOT, check=True)
    elif RUN_EVAL:
        print("Skipping evaluation because no global checkpoints exist yet:", WORK_ROOT / "global_checkpoints")

    summary_path = WORK_ROOT / "validation_reports" / "paper_protocol_eval_summary.csv"
    if summary_path.exists():
        eval_summary = pd.read_csv(summary_path)
        display(eval_summary)
    else:
        print("No DQA paper-protocol summary yet:", summary_path)
        print("Set RUN_EVAL = True after the run has produced checkpoints.")
    """
    return code(text.replace("__RUN_EVAL__", str(eval_default)))


def build_notebook(
    *,
    notebook_title: str,
    notebook_path: Path,
    workspace_name: str,
    stats_dir_name: str,
    runner_log_name: str,
    pid_file_name: str,
    warmup_epochs: int,
    phase1_rounds: int,
    phase2_rounds: int,
    batch_size: int,
    workers: int,
    gpus: int,
    master_port: int,
    min_free_gib: int,
    mode_heading: str,
    mode_description: str,
    estimate_note: str | None = None,
    run_mode: str = "background",
    run_default: bool = False,
    eval_default: bool = False,
) -> None:
    cells = [
        md(
            f"""
            # {notebook_title}

            This notebook gives the DQA-CWA runner the same shape as the FedSTO notebook: build the workspace, dry-run the pipeline, start or resume the experiment, inspect progress, and read the paper-style results in one place.

            It assumes the shared BDD100K/FedSTO setup that already lives under `navigating_data_heterogeneity`, but keeps all DQA-specific runs and stats under `dynamic_quality_aware_classwise_aggregation/`.
            """
        ),
        code(
            f"""
            from __future__ import annotations

            import json
            import shlex
            import shutil
            import subprocess
            import sys
            from datetime import datetime, timezone
            from pathlib import Path
            from typing import Optional

            import pandas as pd


            def find_repo_root(start: Optional[Path] = None) -> Path:
                start = Path.cwd().resolve() if start is None else Path(start).resolve()
                required = (
                    "dynamic_quality_aware_classwise_aggregation/run_dqa_cwa_fedsto.py",
                    "navigating_data_heterogeneity/setup_fedsto_exact_reproduction.py",
                )
                candidate_dirs = []
                for base in (start, *start.parents):
                    candidate_dirs.extend(
                        [
                            base,
                            base / "Object_Detection",
                            base / "masters_research" / "Object_Detection",
                        ]
                    )
                for candidate in candidate_dirs:
                    if all((candidate / marker).exists() for marker in required):
                        return candidate.resolve()
                raise FileNotFoundError("Could not locate the Object_Detection repository root.")


            REPO_ROOT = find_repo_root()
            DQA_ROOT = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation"
            NAV_ROOT = REPO_ROOT / "navigating_data_heterogeneity"
            RUN_SCRIPT = DQA_ROOT / "run_dqa_cwa_fedsto.py"
            EVAL_SCRIPT = DQA_ROOT / "evaluate_paper_protocol.py"
            NOTEBOOK_GENERATOR = DQA_ROOT / "{GENERATOR_PATH.name}"
            WORK_ROOT = DQA_ROOT / "{workspace_name}"
            STATS_ROOT = DQA_ROOT / "{stats_dir_name}"
            RUNNER_LOG = DQA_ROOT / "{runner_log_name}"
            PID_PATH = DQA_ROOT / "{pid_file_name}"
            TRAIN_LOG = WORK_ROOT / "dqa_cwa_latest.log"
            FEDSTO_WORK_ROOT = NAV_ROOT / "efficientteacher_fedsto"

            preferred_python = Path("/root/micromamba/envs/al_yolov8/bin/python")
            PYTHON_BIN = preferred_python if preferred_python.exists() else Path(sys.executable)

            print("repo_root:", REPO_ROOT)
            print("dqa_root:", DQA_ROOT)
            print("workspace:", WORK_ROOT)
            print("stats_root:", STATS_ROOT)
            print("python:", PYTHON_BIN)
            """
        ),
        md(
            f"""
            ## 1. {mode_heading}

            {mode_description}
            """
        ),
        code(
            f"""
            WARMUP_EPOCHS = {warmup_epochs}
            PHASE1_ROUNDS = {phase1_rounds}
            PHASE2_ROUNDS = {phase2_rounds}
            DQA_START_PHASE = 1
            BATCH_SIZE = {batch_size}
            WORKERS = {workers}
            REQUESTED_GPUS = {gpus}
            try:
                import torch

                AVAILABLE_CUDA_GPUS = torch.cuda.device_count()
            except Exception as exc:
                AVAILABLE_CUDA_GPUS = 0
                print("Could not inspect CUDA devices:", exc)

            GPUS = min(REQUESTED_GPUS, AVAILABLE_CUDA_GPUS) if AVAILABLE_CUDA_GPUS else 1
            if GPUS != REQUESTED_GPUS:
                print(
                    f"Requested {{REQUESTED_GPUS}} GPUs, but {{AVAILABLE_CUDA_GPUS}} CUDA device(s) are visible. "
                    f"Using GPUS={{GPUS}} to avoid DDP launch failure."
                )
            MASTER_PORT = {master_port}
            MIN_FREE_GIB = {min_free_gib}

            APPEND_TRAIN_LOG = True
            EXTRA_RUN_ARGS: list[str] = []

            {{
                "warmup_epochs": WARMUP_EPOCHS,
                "phase1_rounds": PHASE1_ROUNDS,
                "phase2_rounds": PHASE2_ROUNDS,
                "dqa_start_phase": DQA_START_PHASE,
                "requested_gpus": REQUESTED_GPUS,
                "available_cuda_gpus": AVAILABLE_CUDA_GPUS,
                "batch_size": BATCH_SIZE,
                "workers": WORKERS,
                "gpus": GPUS,
                "master_port": MASTER_PORT,
                "min_free_gib": MIN_FREE_GIB,
                "workspace": str(WORK_ROOT),
                "stats_root": str(STATS_ROOT),
            }}
            """
        ),
    ]

    if estimate_note:
        cells.append(
            md(
                f"""
                ## Runtime Estimate

                {estimate_note}
                """
            )
        )

    cells.extend(
        [
            md(
                """
                ## 2. Build the Shared Data Interface

                This uses the same shared setup logic as FedSTO and refreshes the DQA workspace manifest, list files, and runtime configs.
                """
            ),
            code(
                """
                subprocess.run(
                    [
                        str(PYTHON_BIN),
                        str(RUN_SCRIPT),
                        "--setup-only",
                        "--workspace-root",
                        str(WORK_ROOT),
                        "--stats-root",
                        str(STATS_ROOT),
                    ],
                    cwd=REPO_ROOT,
                    check=True,
                )

                manifest_path = WORK_ROOT / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_summary = {
                    "classes": manifest["classes"],
                    "server_train_images": manifest["server"]["train_images"],
                    "server_val_images": manifest["server"]["val_images"],
                    "clients": [
                        {
                            "id": client["id"],
                            "weather": client["weather"],
                            "images": client["images"],
                        }
                        for client in manifest["clients"]
                    ],
                    "paper_schedule": manifest["paper_schedule"],
                }
                manifest_summary
                """
            ),
            md(
                """
                ## 3. Dependency and Dry-Run Check

                This makes sure the runtime can import the EfficientTeacher stack and that the DQA runner can generate commands without starting a real train job.
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
                    print("Missing packages:", missing)
                    raise ModuleNotFoundError("Missing runtime dependencies: " + ", ".join(missing))

                subprocess.run(
                    [
                        str(PYTHON_BIN),
                        str(RUN_SCRIPT),
                        "--dry-run",
                        "--workspace-root",
                        str(WORK_ROOT),
                        "--stats-root",
                        str(STATS_ROOT),
                        "--warmup-epochs",
                        str(WARMUP_EPOCHS),
                        "--phase1-rounds",
                        str(PHASE1_ROUNDS),
                        "--phase2-rounds",
                        str(PHASE2_ROUNDS),
                        "--dqa-start-phase",
                        str(DQA_START_PHASE),
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
                    ],
                    cwd=REPO_ROOT,
                    check=True,
                )
                """
            ),
        ]
    )

    if run_mode == "blocking":
        cells.extend([blocking_run_markdown(), blocking_run_code(run_default)])
    else:
        cells.extend([background_run_markdown(), background_run_code(run_default)])

    cells.extend(
        [
            md(
                """
                ## 5. Inspect Progress, Stats Coverage, and Logs

                This is the main heartbeat cell. It reports PID state, federated-round progress, global checkpoints, per-client stats coverage, and short tails of both logs.
                """
            ),
            code(
                """
                def tail_lines(path: Path, lines: int = 25) -> list[str]:
                    if not path.exists():
                        return []
                    try:
                        result = subprocess.run(
                            ["tail", "-n", str(lines), str(path)],
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                        return result.stdout.splitlines()
                    except (FileNotFoundError, subprocess.CalledProcessError):
                        from collections import deque

                        with path.open(encoding="utf-8", errors="replace") as f:
                            return [line.rstrip("\\n") for line in deque(f, maxlen=lines)]


                history_path = WORK_ROOT / "history.json"
                if history_path.exists():
                    history = json.loads(history_path.read_text(encoding="utf-8"))
                else:
                    history = []

                pid = read_pid(PID_PATH)
                state = pid_state(pid)
                completed_phase1 = sum(1 for entry in history if int(entry.get("phase", 0)) == 1)
                completed_phase2 = sum(1 for entry in history if int(entry.get("phase", 0)) == 2)
                expected_total = PHASE1_ROUNDS + PHASE2_ROUNDS
                latest_global = Path(history[-1]["global"]) if history else WORK_ROOT / "global_checkpoints" / "round000_warmup.pt"

                free_gib = shutil.disk_usage(WORK_ROOT).free / 1024**3
                stats_rows = []
                client_count = len(manifest["clients"]) if "manifest" in globals() else 3
                for phase, rounds in ((1, PHASE1_ROUNDS), (2, PHASE2_ROUNDS)):
                    for round_idx in range(1, rounds + 1):
                        round_file = STATS_ROOT / f"phase{phase}_round{round_idx:03d}.json"
                        client_files = sorted(STATS_ROOT.glob(f"phase{phase}_round{round_idx:03d}_client*.json"))
                        stats_rows.append(
                            {
                                "phase": phase,
                                "round": round_idx,
                                "round_stats": round_file.exists(),
                                "client_stats": len(client_files),
                                "expected_client_stats": client_count,
                            }
                        )

                status_summary = {
                    "pid": pid,
                    "pid_state": state,
                    "completed_phase1": f"{completed_phase1}/{PHASE1_ROUNDS}",
                    "completed_phase2": f"{completed_phase2}/{PHASE2_ROUNDS}",
                    "completed_total": f"{len(history)}/{expected_total}",
                    "latest_global": str(latest_global),
                    "free_gib": round(free_gib, 2),
                }
                display(pd.DataFrame([status_summary]))

                if stats_rows:
                    stats_df = pd.DataFrame(stats_rows)
                    display(stats_df.tail(10))

                print("Runner log tail:")
                for line in tail_lines(RUNNER_LOG):
                    print(line)

                print("\\nTrain log tail:")
                for line in tail_lines(TRAIN_LOG):
                    print(line)
                """
            ),
            md(
                """
                ## 6. Paper-Style Evaluation

                Once DQA has produced checkpoints, this cell runs the shared per-weather evaluation and then loads the compact summary table.
                """
            ),
            eval_code(eval_default),
            md(
                """
                ## 7. Compare DQA and FedSTO Results

                If both workspaces have paper-protocol summaries, this cell lines them up side by side.
                """
            ),
            code(
                """
                comparison_paths = {
                    "DQA-CWA": WORK_ROOT / "validation_reports" / "paper_protocol_eval_summary.csv",
                    "FedSTO": FEDSTO_WORK_ROOT / "validation_reports" / "paper_protocol_eval_summary.csv",
                }

                frames = []
                for method, path in comparison_paths.items():
                    if path.exists():
                        df = pd.read_csv(path)
                        df.insert(0, "method", method)
                        frames.append(df)

                if not frames:
                    print("No comparable paper-protocol summaries found yet.")
                else:
                    comparison = pd.concat(frames, ignore_index=True)
                    display(comparison)
                """
            ),
            md(
                """
                ## 8. Artifact Index

                Handy links for whatever we usually need next: manifest, logs, history, stats, checkpoints, and evaluation outputs.
                """
            ),
            code(
                """
                def artifact_row(path: Path, label: str) -> dict:
                    exists = path.exists()
                    modified = (
                        datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
                        if exists
                        else ""
                    )
                    return {
                        "label": label,
                        "path": str(path),
                        "exists": exists,
                        "modified_utc": modified,
                    }


                artifact_rows = [
                    artifact_row(DQA_ROOT / "README.md", "readme"),
                    artifact_row(NOTEBOOK_GENERATOR, "notebook_generator"),
                    artifact_row(WORK_ROOT / "manifest.json", "manifest"),
                    artifact_row(WORK_ROOT / "history.json", "history"),
                    artifact_row(RUNNER_LOG, "runner_log"),
                    artifact_row(TRAIN_LOG, "train_log"),
                    artifact_row(WORK_ROOT / "validation_reports" / "paper_protocol_eval_summary.csv", "paper_eval_summary_csv"),
                    artifact_row(WORK_ROOT / "validation_reports" / "paper_protocol_eval_summary.md", "paper_eval_summary_md"),
                    artifact_row(WORK_ROOT / "global_checkpoints", "global_checkpoints_dir"),
                    artifact_row(STATS_ROOT, "stats_dir"),
                ]
                display(pd.DataFrame(artifact_rows))
                """
            ),
        ]
    )

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
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {notebook_path}")


def build_evaluation_notebook(
    *,
    notebook_title: str,
    notebook_path: Path,
    workspace_name: str,
    stats_dir_name: str,
    notebook_description: str,
) -> None:
    setup_text = """
    from __future__ import annotations

    import json
    import re
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
            "dynamic_quality_aware_classwise_aggregation/run_dqa_cwa_fedsto.py",
            "navigating_data_heterogeneity/setup_fedsto_exact_reproduction.py",
        )
        candidate_dirs = []
        for base in (start, *start.parents):
            candidate_dirs.extend(
                [
                    base,
                    base / "Object_Detection",
                    base / "masters_research" / "Object_Detection",
                ]
            )
        for candidate in candidate_dirs:
            if all((candidate / marker).exists() for marker in required):
                return candidate.resolve()
        raise FileNotFoundError("Could not locate the Object_Detection repository root.")


    REPO_ROOT = find_repo_root()
    DQA_ROOT = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation"
    NAV_ROOT = REPO_ROOT / "navigating_data_heterogeneity"
    WORK_ROOT = DQA_ROOT / "__WORKSPACE_NAME__"
    STATS_ROOT = DQA_ROOT / "__STATS_DIR_NAME__"
    RUNS_ROOT = WORK_ROOT / "runs"
    VALIDATION_ROOT = WORK_ROOT / "validation_reports"
    NOTEBOOK_GENERATOR = DQA_ROOT / "__GENERATOR_NAME__"

    FEDSTO_WORK_ROOT = NAV_ROOT / "efficientteacher_fedsto"
    FEDSTO_TRAINING_SUMMARY = FEDSTO_WORK_ROOT / "validation_reports" / "tables" / "training_run_summary.csv"
    FEDSTO_PAPER_EVAL_SUMMARY = FEDSTO_WORK_ROOT / "validation_reports" / "paper_protocol_eval_summary.csv"

    if sns is not None:
        sns.set_theme(style="whitegrid", context="talk")
    else:
        plt.style.use("ggplot")
    pd.options.display.max_columns = 200
    pd.options.display.max_rows = 200


    def plot_line(ax, data: pd.DataFrame, *, x: str, y: str, hue: str | None = None, marker: str | None = None, linewidth: float = 2.0):
        if sns is not None:
            return sns.lineplot(data=data, x=x, y=y, hue=hue, marker=marker, linewidth=linewidth, ax=ax)

        if hue is not None and hue in data.columns:
            for key, frame in data.groupby(hue):
                ordered = frame.sort_values(x)
                ax.plot(ordered[x], ordered[y], marker=marker, linewidth=linewidth, label=str(key))
            ax.legend(title=hue)
        else:
            ordered = data.sort_values(x)
            ax.plot(ordered[x], ordered[y], marker=marker, linewidth=linewidth)
        return ax


    def plot_bar(ax, data: pd.DataFrame, *, x: str, y: str, hue: str | None = None):
        if sns is not None:
            return sns.barplot(data=data, x=x, y=y, hue=hue, ax=ax)

        if hue is not None and hue in data.columns:
            pivot = data.pivot_table(index=x, columns=hue, values=y, aggfunc="mean")
            pivot.plot(kind="bar", ax=ax)
        else:
            grouped = data.groupby(x, as_index=False)[y].mean()
            ax.bar(grouped[x], grouped[y])
        return ax

    print("repo_root:", REPO_ROOT)
    print("workspace:", WORK_ROOT)
    print("runs_root:", RUNS_ROOT)
    print("validation_root:", VALIDATION_ROOT)
    """
    setup_text = (
        setup_text.replace("__WORKSPACE_NAME__", workspace_name)
        .replace("__STATS_DIR_NAME__", stats_dir_name)
        .replace("__GENERATOR_NAME__", GENERATOR_PATH.name)
    )

    cells = [
        md(
            f"""
            # {notebook_title}

            {notebook_description}
            """
        ),
        code(setup_text),
        md(
            """
            ## 1. Workspace and Artifact Status

            Start with a compact snapshot of what exists: manifest, history, checkpoints, stats, and partial or complete paper-eval outputs.
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

            artifact_rows = [
                {"artifact": "workspace", "path": str(WORK_ROOT), "exists": WORK_ROOT.exists(), "modified_utc": modified_utc(WORK_ROOT)},
                {"artifact": "stats_root", "path": str(STATS_ROOT), "exists": STATS_ROOT.exists(), "modified_utc": modified_utc(STATS_ROOT)},
                {"artifact": "manifest", "path": str(manifest_path), "exists": manifest_path.exists(), "modified_utc": modified_utc(manifest_path)},
                {"artifact": "history", "path": str(history_path), "exists": history_path.exists(), "modified_utc": modified_utc(history_path)},
                {"artifact": "global_checkpoints", "path": str(WORK_ROOT / "global_checkpoints"), "exists": (WORK_ROOT / "global_checkpoints").exists(), "modified_utc": modified_utc(WORK_ROOT / "global_checkpoints")},
                {"artifact": "paper_eval_summary_csv", "path": str(VALIDATION_ROOT / "paper_protocol_eval_summary.csv"), "exists": (VALIDATION_ROOT / "paper_protocol_eval_summary.csv").exists(), "modified_utc": modified_utc(VALIDATION_ROOT / "paper_protocol_eval_summary.csv")},
                {"artifact": "paper_eval_manifest_json", "path": str(VALIDATION_ROOT / "paper_protocol_eval_manifest.json"), "exists": (VALIDATION_ROOT / "paper_protocol_eval_manifest.json").exists(), "modified_utc": modified_utc(VALIDATION_ROOT / "paper_protocol_eval_manifest.json")},
            ]
            display(pd.DataFrame(artifact_rows))

            completed_phase1 = sum(1 for entry in history if int(entry.get("phase", 0)) == 1)
            completed_phase2 = sum(1 for entry in history if int(entry.get("phase", 0)) == 2)
            progress_rows = [
                {"phase": "phase1", "completed_rounds": completed_phase1},
                {"phase": "phase2", "completed_rounds": completed_phase2},
                {"phase": "total", "completed_rounds": len(history)},
            ]
            display(pd.DataFrame(progress_rows))

            manifest_summary = {
                "classes": manifest.get("classes", []),
                "server_weather": manifest.get("server", {}).get("weather"),
                "server_train_images": manifest.get("server", {}).get("train_images"),
                "server_val_images": manifest.get("server", {}).get("val_images"),
                "client_weathers": [client.get("weather") for client in manifest.get("clients", [])],
                "run_dirs": len(list(RUNS_ROOT.glob("*"))),
                "results_csv_files": len(list(RUNS_ROOT.glob("*/results.csv"))),
                "phase1_stats_files": len(list(STATS_ROOT.glob("phase1_round*.json"))),
                "phase2_stats_files": len(list(STATS_ROOT.glob("phase2_round*.json"))),
                "paper_eval_logs": len(list((VALIDATION_ROOT / "paper_protocol_logs").glob("*.log"))),
                "paper_eval_run_dirs": len(list((VALIDATION_ROOT / "paper_protocol_val_runs").glob("*"))),
            }
            manifest_summary
            """
        ),
        md(
            """
            ## 2. Build a Compact Training Summary

            Parse every `runs/*/results.csv`, normalize it into one table, and save a reusable `training_run_summary.csv` under `validation_reports/tables/`.
            """
        ),
        code(
            """
            def normalize_results_columns(df: pd.DataFrame) -> pd.DataFrame:
                renamed = df.rename(columns=lambda col: col.strip())
                return renamed


            def parse_run_name(run_name: str) -> dict | None:
                if run_name == "runtime_server_warmup":
                    return {
                        "phase": 0,
                        "round": 0,
                        "role": "warmup",
                        "client_id": np.nan,
                        "weather": "server_cloudy",
                    }

                match = re.fullmatch(
                    r"dqa_phase(?P<phase>[12])_round(?P<round>\\d{3})_(?:(?P<server>server)|client(?P<client_id>\\d+)_(?P<weather>[a-z]+))",
                    run_name,
                )
                if not match:
                    return None

                phase = int(match.group("phase"))
                round_idx = int(match.group("round"))
                if match.group("server"):
                    return {
                        "phase": phase,
                        "round": round_idx,
                        "role": "server",
                        "client_id": np.nan,
                        "weather": "server_cloudy",
                    }
                return {
                    "phase": phase,
                    "round": round_idx,
                    "role": "client",
                    "client_id": int(match.group("client_id")),
                    "weather": match.group("weather"),
                }


            def safe_float(value) -> float:
                if pd.isna(value):
                    return np.nan
                return float(value)


            metric_columns = {
                "precision": "metrics/precision",
                "recall": "metrics/recall",
                "mAP_0.5": "metrics/mAP_0.5",
                "mAP_0.5:0.95": "metrics/mAP_0.5:0.95",
            }
            train_loss_columns = {
                "box_loss": "train/box_loss",
                "obj_loss": "train/obj_loss",
                "cls_loss": "train/cls_loss",
            }
            val_loss_columns = {
                "box_loss": "val/box_loss",
                "obj_loss": "val/obj_loss",
                "cls_loss": "val/cls_loss",
            }


            summary_rows = []
            best_basis = "metrics/mAP_0.5:0.95"
            for results_path in sorted(RUNS_ROOT.glob("*/results.csv")):
                run_name = results_path.parent.name
                meta = parse_run_name(run_name)
                if meta is None:
                    continue

                df = normalize_results_columns(pd.read_csv(results_path))
                best_idx = df[best_basis].astype(float).idxmax()
                final_row = df.iloc[-1]
                best_row = df.loc[best_idx]

                row = {
                    "run_name": run_name,
                    "run_dir": str(results_path.parent),
                    "phase": meta["phase"],
                    "round": meta["round"],
                    "role": meta["role"],
                    "client_id": meta["client_id"],
                    "weather": meta["weather"],
                    "n_logged_rows": len(df),
                    "final_epoch": int(final_row["epoch"]),
                    "best_epoch": int(best_row["epoch"]),
                    "best_basis": best_basis,
                }

                for prefix, source_row in (("final", final_row), ("best", best_row)):
                    for metric_name, column_name in metric_columns.items():
                        row[f"{prefix}_metrics/{metric_name}"] = safe_float(source_row[column_name])
                    for loss_name, column_name in train_loss_columns.items():
                        row[f"{prefix}_train/{loss_name}"] = safe_float(source_row[column_name])
                    for loss_name, column_name in val_loss_columns.items():
                        row[f"{prefix}_val/{loss_name}"] = safe_float(source_row[column_name])

                summary_rows.append(row)

            run_summary = pd.DataFrame(summary_rows)
            if run_summary.empty:
                raise RuntimeError(f"No DQA results.csv files were found under {RUNS_ROOT}")

            run_summary["client_id_sort"] = run_summary["client_id"].fillna(-1)
            run_summary = (
                run_summary
                .sort_values(["phase", "round", "role", "client_id_sort"])
                .drop(columns=["client_id_sort"])
                .reset_index(drop=True)
            )

            validation_table_root = VALIDATION_ROOT / "tables"
            validation_table_root.mkdir(parents=True, exist_ok=True)
            training_summary_path = validation_table_root / "training_run_summary.csv"
            run_summary.to_csv(training_summary_path, index=False)

            server_summary = run_summary[run_summary["role"].isin(["warmup", "server"])].copy()
            client_summary = run_summary[run_summary["role"] == "client"].copy()

            print("Wrote training summary:", training_summary_path)
            display(run_summary.groupby(["phase", "role"], dropna=False).size().reset_index(name="runs"))
            display(run_summary.head(12))
            """
        ),
        md(
            """
            ## 3. Key DQA Checkpoints

            Pull out the warm-up baseline, the best phase-1 server round, the best phase-2 server round, and the final phase-2 server round. Then show the latest client snapshot by weather.
            """
        ),
        code(
            """
            def labeled_checkpoint_rows(summary: pd.DataFrame) -> pd.DataFrame:
                selections = []
                candidates = [
                    ("warmup", summary[summary["phase"] == 0], "best"),
                    ("best_phase1_server", summary[(summary["phase"] == 1) & (summary["role"] == "server")], "best"),
                    ("best_phase2_server", summary[(summary["phase"] == 2) & (summary["role"] == "server")], "best"),
                    ("final_phase2_server", summary[(summary["phase"] == 2) & (summary["role"] == "server")], "final"),
                ]

                for label, frame, source in candidates:
                    if frame.empty:
                        continue
                    if source == "final":
                        chosen = frame.sort_values(["round", "best_epoch"]).iloc[-1].copy()
                    else:
                        chosen = frame.sort_values("best_metrics/mAP_0.5:0.95", ascending=False).iloc[0].copy()
                    chosen["checkpoint"] = label
                    chosen["metric_source"] = source
                    selections.append(chosen)

                if not selections:
                    return pd.DataFrame()
                return pd.DataFrame(selections)


            key_server = labeled_checkpoint_rows(server_summary)
            if not key_server.empty:
                warmup_baseline = key_server.loc[key_server["checkpoint"] == "warmup", "best_metrics/mAP_0.5:0.95"]
                baseline = warmup_baseline.iloc[0] if not warmup_baseline.empty else np.nan
                key_server["delta_vs_warmup_mAP_0.5:0.95"] = key_server["best_metrics/mAP_0.5:0.95"] - baseline
                display(
                    key_server[
                        [
                            "checkpoint",
                            "run_name",
                            "phase",
                            "round",
                            "metric_source",
                            "final_metrics/precision",
                            "final_metrics/recall",
                            "final_metrics/mAP_0.5",
                            "final_metrics/mAP_0.5:0.95",
                            "best_metrics/mAP_0.5",
                            "best_metrics/mAP_0.5:0.95",
                            "delta_vs_warmup_mAP_0.5:0.95",
                        ]
                    ].round(4)
                )

            latest_clients = (
                client_summary
                .sort_values(["phase", "round"])
                .groupby(["phase", "weather"], as_index=False)
                .tail(1)
                .sort_values(["phase", "weather"])
            )
            display(
                latest_clients[
                    [
                        "phase",
                        "weather",
                        "round",
                        "final_metrics/precision",
                        "final_metrics/recall",
                        "final_metrics/mAP_0.5",
                        "final_metrics/mAP_0.5:0.95",
                    ]
                ].round(4)
            )
            """
        ),
        md(
            """
            ## 4. Server Metric Curves

            Plot the warm-up epochs and the federated server rounds on one stitched timeline so the main mAP, precision, and recall trends are easy to scan.
            """
        ),
        code(
            """
            warmup_history_path = RUNS_ROOT / "runtime_server_warmup" / "results.csv"
            warmup_history = normalize_results_columns(pd.read_csv(warmup_history_path))
            warmup_epochs = len(warmup_history)
            phase1_max_round = int(server_summary.loc[server_summary["phase"] == 1, "round"].max()) if (server_summary["phase"] == 1).any() else 0

            warmup_curve = warmup_history[
                [
                    "epoch",
                    "metrics/precision",
                    "metrics/recall",
                    "metrics/mAP_0.5",
                    "metrics/mAP_0.5:0.95",
                ]
            ].copy()
            warmup_curve["timeline"] = warmup_curve["epoch"] + 1
            warmup_curve["stage"] = "warmup"

            def server_curve_frame(phase: int, stage: str, offset: int) -> pd.DataFrame:
                phase_df = server_summary[(server_summary["phase"] == phase) & (server_summary["role"] == "server")].copy()
                if phase_df.empty:
                    return pd.DataFrame(columns=["timeline", "stage", "round", "metrics/precision", "metrics/recall", "metrics/mAP_0.5", "metrics/mAP_0.5:0.95"])
                phase_df["timeline"] = offset + phase_df["round"]
                phase_df["stage"] = stage
                phase_df = phase_df.rename(
                    columns={
                        "final_metrics/precision": "metrics/precision",
                        "final_metrics/recall": "metrics/recall",
                        "final_metrics/mAP_0.5": "metrics/mAP_0.5",
                        "final_metrics/mAP_0.5:0.95": "metrics/mAP_0.5:0.95",
                    }
                )
                return phase_df[
                    [
                        "timeline",
                        "stage",
                        "round",
                        "metrics/precision",
                        "metrics/recall",
                        "metrics/mAP_0.5",
                        "metrics/mAP_0.5:0.95",
                    ]
                ]


            curve_df = pd.concat(
                [
                    warmup_curve[["timeline", "stage", "metrics/precision", "metrics/recall", "metrics/mAP_0.5", "metrics/mAP_0.5:0.95"]],
                    server_curve_frame(1, "phase1", warmup_epochs),
                    server_curve_frame(2, "phase2", warmup_epochs + phase1_max_round),
                ],
                ignore_index=True,
            )

            fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
            metric_specs = [
                ("metrics/mAP_0.5", "mAP@0.5"),
                ("metrics/mAP_0.5:0.95", "mAP@0.5:0.95"),
                ("metrics/precision", "precision"),
                ("metrics/recall", "recall"),
            ]

            for ax, (metric_col, title) in zip(axes.flat, metric_specs):
                plot_line(ax, curve_df, x="timeline", y=metric_col, hue="stage", linewidth=2)
                if warmup_epochs:
                    ax.axvline(warmup_epochs + 0.5, color="gray", linestyle="--", linewidth=1)
                if phase1_max_round:
                    ax.axvline(warmup_epochs + phase1_max_round + 0.5, color="gray", linestyle="--", linewidth=1)
                best_idx = curve_df[metric_col].idxmax()
                best_row = curve_df.loc[best_idx]
                ax.scatter([best_row["timeline"]], [best_row[metric_col]], color="black", s=50, zorder=5)
                ax.annotate(
                    f"{best_row['stage']} peak\\n{best_row[metric_col]:.3f}",
                    (best_row["timeline"], best_row[metric_col]),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontsize=10,
                )
                ax.set_title(title)
                ax.set_xlabel("stitched timeline step")
                ax.set_ylabel(title)

            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 5. Client Weather Trajectories

            The server view is only half the story. These plots keep the client runs split by weather so it is easier to see whether one weather regime is drifting or improving differently from the others.
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=True)
            for ax, phase in zip(axes, [1, 2]):
                phase_df = client_summary[client_summary["phase"] == phase].copy()
                if phase_df.empty:
                    ax.set_visible(False)
                    continue
                plot_line(
                    ax,
                    phase_df,
                    x="round",
                    y="final_metrics/mAP_0.5:0.95",
                    hue="weather",
                    marker="o",
                )
                ax.set_title(f"phase {phase} client mAP@0.5:0.95")
                ax.set_xlabel("round")
                ax.set_ylabel("mAP@0.5:0.95")

            plt.tight_layout()
            plt.show()

            latest_phase2_clients = (
                client_summary[client_summary["phase"] == 2]
                .sort_values(["weather", "round"])
                .groupby("weather", as_index=False)
                .tail(1)
                .sort_values("weather")
            )
            if not latest_phase2_clients.empty:
                plt.figure(figsize=(9, 4))
                plot_bar(
                    plt.gca(),
                    latest_phase2_clients,
                    x="weather",
                    y="final_metrics/mAP_0.5:0.95",
                )
                plt.title("latest phase-2 client mAP@0.5:0.95 by weather")
                plt.xlabel("weather")
                plt.ylabel("mAP@0.5:0.95")
                plt.tight_layout()
                plt.show()
            """
        ),
        md(
            """
            ## 6. DQA vs FedSTO Snapshot

            If the FedSTO baseline summary is present, compare the same headline checkpoints side by side so the server story is easy to read without opening two notebooks.
            """
        ),
        code(
            """
            def comparable_checkpoint_rows(summary: pd.DataFrame, method: str) -> pd.DataFrame:
                rows = []

                def append_frame(label: str, frame: pd.DataFrame, source: str) -> None:
                    if frame.empty:
                        return
                    chosen = frame.iloc[0]
                    prefix = "best" if source == "best" else "final"
                    rows.append(
                        {
                            "method": method,
                            "checkpoint": label,
                            "run_name": chosen["run_name"],
                            "phase": chosen["phase"],
                            "round": chosen["round"],
                            "metric_source": source,
                            "precision": chosen[f"{prefix}_metrics/precision"],
                            "recall": chosen[f"{prefix}_metrics/recall"],
                            "map50": chosen[f"{prefix}_metrics/mAP_0.5"],
                            "map50_95": chosen[f"{prefix}_metrics/mAP_0.5:0.95"],
                        }
                    )

                append_frame("warmup", summary[summary["phase"] == 0].sort_values("best_metrics/mAP_0.5:0.95", ascending=False).head(1), "best")
                append_frame("best_phase1_server", summary[(summary["phase"] == 1) & (summary["role"] == "server")].sort_values("best_metrics/mAP_0.5:0.95", ascending=False).head(1), "best")
                append_frame("best_phase2_server", summary[(summary["phase"] == 2) & (summary["role"] == "server")].sort_values("best_metrics/mAP_0.5:0.95", ascending=False).head(1), "best")
                append_frame("final_phase2_server", summary[(summary["phase"] == 2) & (summary["role"] == "server")].sort_values("round").tail(1), "final")

                return pd.DataFrame(rows)


            comparison_frames = [comparable_checkpoint_rows(run_summary, "DQA-CWA")]
            if FEDSTO_TRAINING_SUMMARY.exists():
                fedsto_summary = pd.read_csv(FEDSTO_TRAINING_SUMMARY)
                comparison_frames.append(comparable_checkpoint_rows(fedsto_summary, "FedSTO"))
            else:
                fedsto_summary = pd.DataFrame()
                print("FedSTO training summary not found:", FEDSTO_TRAINING_SUMMARY)

            checkpoint_comparison = pd.concat(comparison_frames, ignore_index=True)
            display(checkpoint_comparison.round(4))

            if checkpoint_comparison["method"].nunique() > 1:
                fig, axes = plt.subplots(1, 2, figsize=(16, 5))
                plot_bar(axes[0], checkpoint_comparison, x="checkpoint", y="map50", hue="method")
                plot_bar(axes[1], checkpoint_comparison, x="checkpoint", y="map50_95", hue="method")
                axes[0].set_title("server checkpoint comparison: mAP@0.5")
                axes[1].set_title("server checkpoint comparison: mAP@0.5:0.95")
                for ax in axes:
                    ax.tick_params(axis="x", rotation=20)
                    ax.set_xlabel("")
                plt.tight_layout()
                plt.show()
            """
        ),
        md(
            """
            ## 7. Paper-Eval Status and Visual Artifacts

            Use the complete paper-eval summary if it exists. If not, fall back to partial DQA evaluation artifacts so we can still see what has already been produced.
            """
        ),
        code(
            """
            paper_eval_frames = []
            dqa_paper_eval_summary = VALIDATION_ROOT / "paper_protocol_eval_summary.csv"
            if dqa_paper_eval_summary.exists():
                dqa_eval_df = pd.read_csv(dqa_paper_eval_summary)
                dqa_eval_df.insert(0, "method", "DQA-CWA")
                paper_eval_frames.append(dqa_eval_df)

            if FEDSTO_PAPER_EVAL_SUMMARY.exists():
                fedsto_eval_df = pd.read_csv(FEDSTO_PAPER_EVAL_SUMMARY)
                fedsto_eval_df.insert(0, "method", "FedSTO")
                paper_eval_frames.append(fedsto_eval_df)

            if paper_eval_frames:
                paper_eval_comparison = pd.concat(paper_eval_frames, ignore_index=True)
                display(paper_eval_comparison.round(4))

                ok_rows = paper_eval_comparison[paper_eval_comparison["status"] == "ok"].copy()
                if not ok_rows.empty:
                    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
                    plot_bar(axes[0], ok_rows, x="split", y="map50", hue="method")
                    plot_bar(axes[1], ok_rows, x="split", y="map50_95", hue="method")
                    axes[0].set_title("paper eval mAP@0.5")
                    axes[1].set_title("paper eval mAP@0.5:0.95")
                    for ax in axes:
                        ax.tick_params(axis="x", rotation=20)
                        ax.set_xlabel("")
                    plt.tight_layout()
                    plt.show()
            else:
                print("No complete paper-protocol summary CSV is available yet.")

            partial_log_dir = VALIDATION_ROOT / "paper_protocol_logs"
            partial_run_dir = VALIDATION_ROOT / "paper_protocol_val_runs"

            partial_log_rows = [
                {
                    "log_file": path.name,
                    "bytes": path.stat().st_size,
                    "modified_utc": modified_utc(path),
                }
                for path in sorted(partial_log_dir.glob("*.log"))
            ]
            if partial_log_rows:
                display(pd.DataFrame(partial_log_rows))

            partial_run_rows = [
                {
                    "run_dir": path.name,
                    "has_pr_curve": (path / "PR_curve.png").exists(),
                    "has_confusion_matrix": (path / "confusion_matrix.png").exists(),
                    "has_p_curve": (path / "P_curve.png").exists(),
                    "has_r_curve": (path / "R_curve.png").exists(),
                    "modified_utc": modified_utc(path),
                }
                for path in sorted(partial_run_dir.glob("*"))
                if path.is_dir()
            ]
            if partial_run_rows:
                display(pd.DataFrame(partial_run_rows))

            preview_dir = next(
                (
                    path
                    for path in sorted(partial_run_dir.glob("*"))
                    if path.is_dir() and (path / "PR_curve.png").exists() and (path / "confusion_matrix.png").exists()
                ),
                None,
            )
            if preview_dir is not None:
                print("Previewing DQA partial paper-eval artifacts from:", preview_dir.name)
                display(NotebookImage(filename=str(preview_dir / "PR_curve.png"), width=700))
                display(NotebookImage(filename=str(preview_dir / "confusion_matrix.png"), width=700))
            """
        ),
        md(
            """
            ## 8. Artifact Index

            A last table with the main files we usually click next.
            """
        ),
        code(
            """
            def artifact_row(path: Path, label: str) -> dict:
                exists = path.exists()
                return {
                    "label": label,
                    "path": str(path),
                    "exists": exists,
                    "modified_utc": modified_utc(path),
                }


            artifact_rows = [
                artifact_row(DQA_ROOT / "README.md", "readme"),
                artifact_row(NOTEBOOK_GENERATOR, "notebook_generator"),
                artifact_row(WORK_ROOT / "manifest.json", "manifest"),
                artifact_row(WORK_ROOT / "history.json", "history"),
                artifact_row(VALIDATION_ROOT / "tables" / "training_run_summary.csv", "training_run_summary_csv"),
                artifact_row(VALIDATION_ROOT / "paper_protocol_eval_summary.csv", "paper_eval_summary_csv"),
                artifact_row(VALIDATION_ROOT / "paper_protocol_eval_manifest.json", "paper_eval_manifest_json"),
                artifact_row(VALIDATION_ROOT / "paper_protocol_val_runs", "paper_eval_run_dir"),
                artifact_row(WORK_ROOT / "global_checkpoints", "global_checkpoints_dir"),
                artifact_row(STATS_ROOT, "stats_dir"),
            ]
            display(pd.DataFrame(artifact_rows))
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
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {notebook_path}")


def main() -> None:
    build_notebook(
        notebook_title="01 DQA-CWA Reproduction",
        notebook_path=ROOT / "01_dqa_cwa_reproduction.ipynb",
        workspace_name="efficientteacher_dqa_cwa",
        stats_dir_name="stats",
        runner_log_name="dqa_cwa_runner.out",
        pid_file_name="dqa_cwa_runner.pid",
        warmup_epochs=8,
        phase1_rounds=15,
        phase2_rounds=35,
        batch_size=64,
        workers=0,
        gpus=2,
        master_port=29510,
        min_free_gib=80,
        mode_heading="Pilot Configuration",
        mode_description="The defaults below match the fast true-DQA pilot we have been using: roughly 10-12 hours on a 2x RTX 6000 Ada node.",
        estimate_note="This pilot is the quickest way to see whether true DQA is behaving sensibly. It trades paper-scale coverage for shorter turnaround.",
        run_mode="background",
        run_default=False,
        eval_default=False,
    )
    build_notebook(
        notebook_title="02 DQA-CWA Exact Reproduction",
        notebook_path=ROOT / "02_dqa_cwa_exact_reproduction.ipynb",
        workspace_name="efficientteacher_dqa_cwa_exact",
        stats_dir_name="stats_exact",
        runner_log_name="dqa_cwa_exact_runner.out",
        pid_file_name="dqa_cwa_exact_runner.pid",
        warmup_epochs=50,
        phase1_rounds=100,
        phase2_rounds=150,
        batch_size=64,
        workers=0,
        gpus=2,
        master_port=29511,
        min_free_gib=80,
        mode_heading="Exact-Reproduction Configuration",
        mode_description="These defaults target the paper-scale run: 50 warm-up epochs, 100 phase-1 rounds, and 150 phase-2 rounds. This notebook uses a separate workspace and stats directory so it does not mix with the faster 01 pilot run.",
        estimate_note="Using the FedSTO full reproduction log as a baseline, this paper-scale path is roughly a 46-48 hour clean run on the same 2x RTX 6000 Ada setup, and longer if interruptions force retries.",
        run_mode="background",
        run_default=False,
        eval_default=False,
    )
    build_notebook(
        notebook_title="02_2 DQA-CWA 14h Reproduction",
        notebook_path=ROOT / "02_2_dqa_cwa_14h_reproduction.ipynb",
        workspace_name="efficientteacher_dqa_cwa_14h",
        stats_dir_name="stats_14h",
        runner_log_name="dqa_cwa_14h_runner.out",
        pid_file_name="dqa_cwa_14h_runner.pid",
        warmup_epochs=15,
        phase1_rounds=20,
        phase2_rounds=50,
        batch_size=64,
        workers=0,
        gpus=2,
        master_port=29512,
        min_free_gib=80,
        mode_heading="13-14 Hour Configuration",
        mode_description="These defaults are calibrated from the completed FedSTO log to land near a same-day result on the same hardware, while still leaving enough phase-2 rounds for DQA behavior to show up. This notebook is intended to work cleanly with `Run All`.",
        estimate_note="FedSTO logged 50 warm-up epochs in 0.982 hours, phase-1 rounds at about 10.46 minutes each, and phase-2 rounds at about 11.17 minutes each. With 15 warm-up epochs, 20 phase-1 rounds, and 50 phase-2 rounds, the clean-run estimate is about 13.1 hours before modest DQA overhead, so this notebook is aimed at a practical 13-14 hour turnaround.",
        run_mode="blocking",
        run_default=True,
        eval_default=True,
    )
    build_evaluation_notebook(
        notebook_title="02_3 DQA-CWA 14h Evaluation",
        notebook_path=ROOT / "02_3_dqa_cwa_14h_evaluation.ipynb",
        workspace_name="efficientteacher_dqa_cwa_14h",
        stats_dir_name="stats_14h",
        notebook_description="This notebook is a read-only analysis pass for the 13-14 hour DQA-CWA run. It does not launch training by default. Instead it pulls together the finished run artifacts, writes a compact training summary table, and renders the plots that are easiest to read when we want a quick answer about how DQA behaved.",
    )
    build_notebook(
        notebook_title="03 DQA-CWA Corrected 12h Reproduction",
        notebook_path=ROOT / "03_dqa_cwa_corrected_12h_reproduction.ipynb",
        workspace_name="efficientteacher_dqa_cwa_corrected_12h",
        stats_dir_name="stats_corrected_12h",
        runner_log_name="dqa_cwa_corrected_12h_runner.out",
        pid_file_name="dqa_cwa_corrected_12h_runner.pid",
        warmup_epochs=15,
        phase1_rounds=20,
        phase2_rounds=40,
        batch_size=64,
        workers=0,
        gpus=2,
        master_port=29513,
        min_free_gib=80,
        mode_heading="Corrected 12 Hour Configuration",
        mode_description="This run uses the corrected FedSTO Algorithm 1 order: clients train from the current global model, client checkpoints are aggregated, the server updates that aggregate on labeled data, and that server-updated model becomes the next global checkpoint. DQA-CWA starts at phase 1, so every post-warmup federated round uses DQA aggregation rather than FedSTO aggregation.",
        estimate_note="The completed FedSTO log measured 50 warm-up epochs at 0.982 hours, phase-1 rounds at about 10.46 minutes each, and phase-2 rounds at about 11.17 minutes each. With 15 warm-up epochs, 20 phase-1 rounds, and 40 phase-2 rounds, the clean-run estimate is about 11.2 hours before modest DQA overhead, so this is aimed at roughly a 12-hour corrected run.",
        run_mode="blocking",
        run_default=True,
        eval_default=True,
    )
    build_evaluation_notebook(
        notebook_title="03_2 DQA-CWA Corrected 12h Evaluation",
        notebook_path=ROOT / "03_2_dqa_cwa_corrected_12h_evaluation.ipynb",
        workspace_name="efficientteacher_dqa_cwa_corrected_12h",
        stats_dir_name="stats_corrected_12h",
        notebook_description="This notebook is a read-only analysis pass for the corrected 12-hour DQA-CWA run. It does not launch training by default. Instead it pulls together the finished run artifacts, writes a compact training summary table, renders the plots that are easiest to read, and compares DQA-CWA against the corrected FedSTO baseline when both paper-protocol summaries exist.",
    )


if __name__ == "__main__":
    main()
