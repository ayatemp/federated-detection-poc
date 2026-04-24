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


    if RUN_FULL_REPRODUCTION:
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

    if RUN_EVAL:
        subprocess.run(eval_cmd, cwd=REPO_ROOT, check=True)

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
            BATCH_SIZE = {batch_size}
            WORKERS = {workers}
            GPUS = {gpus}
            MASTER_PORT = {master_port}
            MIN_FREE_GIB = {min_free_gib}

            APPEND_TRAIN_LOG = True
            EXTRA_RUN_ARGS: list[str] = []

            {{
                "warmup_epochs": WARMUP_EPOCHS,
                "phase1_rounds": PHASE1_ROUNDS,
                "phase2_rounds": PHASE2_ROUNDS,
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
                    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]


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
                for round_idx in range(1, PHASE2_ROUNDS + 1):
                    round_file = STATS_ROOT / f"phase2_round{round_idx:03d}.json"
                    client_files = sorted(STATS_ROOT.glob(f"phase2_round{round_idx:03d}_client*.json"))
                    stats_rows.append(
                        {
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


if __name__ == "__main__":
    main()
