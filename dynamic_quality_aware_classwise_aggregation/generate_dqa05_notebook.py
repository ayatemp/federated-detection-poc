#!/usr/bin/env python3
"""Generate notebook 05 for the DQA scene/class heterogeneity experiment."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "05_dqa_scene_class_profile_5h.ipynb"


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


cells = [
    md(
        """
        # 05 DQA Scene/Class Profile 5h

        This notebook is the first DQA experiment aligned to the intended claim:

        - server labeled data is limited to the cloudy/partly-cloudy server split
        - clients are unlabeled scene domains: highway, city street, residential
        - DQA should help when each client has different useful class evidence
        - pseudo-label bbox damage is reduced with the best safe profile from the 02/ET probes

        Default runtime target is about 5 hours by reusing an existing scene warmup and running a
        shorter federated schedule.  For the later 12h run, change `PHASE1_ROUNDS` and
        `PHASE2_ROUNDS` in the settings cell.
        """
    ),
    md(
        """
        ## 1. Paths
        """
    ),
    code(
        """
        from __future__ import annotations

        import json
        import os
        import re
        import shutil
        import socket
        import subprocess
        import sys
        from collections import deque
        from pathlib import Path
        from typing import Optional

        import pandas as pd


        def find_repo_root(start: Optional[Path] = None) -> Path:
            start = Path.cwd().resolve() if start is None else Path(start).resolve()
            required = (
                "dynamic_quality_aware_classwise_aggregation/run_dqa_cwa_fedsto_scene_v2_profiled.py",
                "dynamic_quality_aware_classwise_aggregation/evaluate_scene_protocol.py",
                "navigating_data_heterogeneity/setup_fedsto_scene_reproduction.py",
            )
            for base in (start, *start.parents):
                for candidate in (base, base / "Object_Detection"):
                    if all((candidate / marker).exists() for marker in required):
                        return candidate.resolve()
            raise FileNotFoundError("Could not locate /app/Object_Detection")


        REPO_ROOT = find_repo_root()
        DQA_ROOT = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation"
        RUN_SCRIPT = DQA_ROOT / "run_dqa_cwa_fedsto_scene_v2_profiled.py"
        EVAL_SCRIPT = DQA_ROOT / "evaluate_scene_protocol.py"

        WORK_ROOT = DQA_ROOT / "efficientteacher_dqa05_scene_class_profile_5h"
        STATS_ROOT = DQA_ROOT / "stats_dqa05_scene_class_profile_5h"
        RUNNER_LOG = DQA_ROOT / "dqa05_scene_class_profile_5h_runner.out"
        TRAIN_LOG = DQA_ROOT / "dqa05_scene_class_profile_5h_train.log"
        PID_PATH = DQA_ROOT / "dqa05_scene_class_profile_5h_runner.pid"

        SOURCE_WARMUP = (
            DQA_ROOT
            / "efficientteacher_dqa_ver2_scene_12h"
            / "global_checkpoints"
            / "round000_warmup.pt"
        )

        preferred_python = Path("/root/micromamba/envs/al_yolov8/bin/python")
        PYTHON_BIN = preferred_python if preferred_python.exists() else Path(sys.executable)

        print("repo_root:", REPO_ROOT)
        print("workspace:", WORK_ROOT)
        print("stats_root:", STATS_ROOT)
        print("python:", PYTHON_BIN)
        print("source_warmup:", SOURCE_WARMUP)
        """
    ),
    md(
        """
        ## 2. Experiment Settings
        """
    ),
    code(
        """
        # 5h pilot.  For the later 12h run, use PHASE1_ROUNDS=20 and PHASE2_ROUNDS=35.
        WARMUP_EPOCHS = 20
        PHASE1_ROUNDS = 8
        PHASE2_ROUNDS = 14
        DQA_START_PHASE = 2

        BATCH_SIZE = 128
        WORKERS = 0
        REQUESTED_GPUS = 2
        # The runner deletes completed-round intermediates.  The earlier 70 GiB
        # guard is useful for full clean reproductions, but too strict on this
        # shared workspace where only ~14 GiB is currently free.
        MIN_FREE_GIB = 8

        # Loss profile selected by run_dqa_cwa_fedsto_scene_v2_profiled.py.
        # strict_low_bbox was chosen because it reduced pseudo-bbox damage while keeping enough
        # active classes for DQA.  objectness_only is available as a later ablation.
        SSOD_PROFILE = "strict_low_bbox"
        CLIENT_LR0 = 3e-4
        SERVER_LR0 = 1e-3

        SEED_WARMUP_FROM_SOURCE = True
        APPEND_TRAIN_LOG = False
        RUN_TRAINING = True
        RUN_IN_BACKGROUND = False
        STREAM_TRAIN_OUTPUT = True

        try:
            import torch

            AVAILABLE_CUDA_GPUS = torch.cuda.device_count()
        except Exception as exc:
            AVAILABLE_CUDA_GPUS = 0
            print("Could not inspect CUDA devices:", exc)

        GPUS = min(REQUESTED_GPUS, AVAILABLE_CUDA_GPUS) if AVAILABLE_CUDA_GPUS else 1
        if GPUS != REQUESTED_GPUS:
            print(f"Requested {REQUESTED_GPUS} GPU(s), visible={AVAILABLE_CUDA_GPUS}; using GPUS={GPUS}")


        def find_free_port(preferred: int) -> int:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind(("127.0.0.1", preferred))
                    return preferred
                except OSError:
                    pass
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                return int(sock.getsockname()[1])


        MASTER_PORT = find_free_port(29525)

        os.environ["DQA05_SSOD_PROFILE"] = SSOD_PROFILE
        os.environ["DQA05_CLIENT_LR0"] = str(CLIENT_LR0)
        os.environ["DQA05_SERVER_LR0"] = str(SERVER_LR0)

        {
            "phase1_rounds": PHASE1_ROUNDS,
            "phase2_rounds": PHASE2_ROUNDS,
            "dqa_start_phase": DQA_START_PHASE,
            "ssod_profile": SSOD_PROFILE,
            "client_lr0": CLIENT_LR0,
            "server_lr0": SERVER_LR0,
            "batch_size": BATCH_SIZE,
            "gpus": GPUS,
            "master_port": MASTER_PORT,
            "workspace": str(WORK_ROOT),
        }
        """
    ),
    md(
        """
        ## 3. Build Lists and Seed Warmup

        The scene setup creates highway/citystreet/residential clients and scene-wise validation
        lists.  The 5h pilot reuses the existing scene warmup so the budget is spent on DQA.
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
            env=os.environ.copy(),
        )

        warmup_dst = WORK_ROOT / "global_checkpoints" / "round000_warmup.pt"
        if SEED_WARMUP_FROM_SOURCE and SOURCE_WARMUP.exists() and not warmup_dst.exists():
            warmup_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE_WARMUP, warmup_dst)
            print("Seeded warmup:", warmup_dst)
        elif warmup_dst.exists():
            print("Warmup already present:", warmup_dst)
        else:
            print("No warmup seed found; the runner will train warmup from scratch.")

        manifest = json.loads((WORK_ROOT / "manifest.json").read_text(encoding="utf-8"))
        server = manifest["server"]
        clients = manifest["clients"]
        eval_splits = manifest["paper_evaluation"]["splits"]

        display(pd.DataFrame([server]))
        display(pd.DataFrame(clients))
        display(pd.DataFrame(eval_splits)[["name", "raw_scene", "images", "boxes"]])
        """
    ),
    md(
        """
        ## 4. Dry Run
        """
    ),
    code(
        """
        dry_cmd = [
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
            "--classwise-blend",
            "0.35",
            "--server-anchor",
            "1.25",
            "--localize-bn",
            "--enable-dqa-guard",
            "--dqa-drop-ratio-threshold",
            "0.15",
            "--dqa-spike-ratio-threshold",
            "3.0",
        ]

        subprocess.run(dry_cmd, cwd=REPO_ROOT, check=True, env=os.environ.copy())
        """
    ),
    md(
        """
        ## 5. Start or Resume Training
        """
    ),
    code(
        """
        def read_pid(path: Path) -> int | None:
            if not path.exists():
                return None
            try:
                return int(path.read_text(encoding="utf-8").strip())
            except ValueError:
                return None


        def pid_state(pid: int | None) -> str:
            if pid is None:
                return "missing"
            result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
            state = result.stdout.strip()
            if result.returncode != 0 or not state:
                return "missing"
            if "Z" in state:
                return "zombie"
            return state


        train_cmd = [
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
            "--log-file",
            str(TRAIN_LOG),
            "--classwise-blend",
            "0.35",
            "--server-anchor",
            "1.25",
            "--localize-bn",
            "--enable-dqa-guard",
            "--dqa-drop-ratio-threshold",
            "0.15",
            "--dqa-spike-ratio-threshold",
            "3.0",
        ]
        if APPEND_TRAIN_LOG:
            train_cmd.append("--append-train-log")
        if STREAM_TRAIN_OUTPUT:
            train_cmd.append("--stream-train-output")

        current_pid = read_pid(PID_PATH)
        state = pid_state(current_pid)
        print("existing pid:", current_pid, state)
        print(" ".join(train_cmd))

        if RUN_TRAINING and state not in {"missing", "zombie"}:
            print("Training already appears to be running.")
        elif RUN_TRAINING and RUN_IN_BACKGROUND:
            env = os.environ.copy()
            RUNNER_LOG.parent.mkdir(parents=True, exist_ok=True)
            log_mode = "ab" if APPEND_TRAIN_LOG else "wb"
            with RUNNER_LOG.open(log_mode) as out:
                process = subprocess.Popen(
                    train_cmd,
                    cwd=REPO_ROOT,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
            PID_PATH.write_text(str(process.pid), encoding="utf-8")
            print("Started PID:", process.pid)
            print("Runner log:", RUNNER_LOG)
            print("Train log:", TRAIN_LOG)
        elif RUN_TRAINING:
            env = os.environ.copy()
            RUNNER_LOG.parent.mkdir(parents=True, exist_ok=True)
            log_mode = "a" if APPEND_TRAIN_LOG else "w"
            print("Running in foreground; progress will stream in this cell.")
            print("Runner log:", RUNNER_LOG)
            print("Train log:", TRAIN_LOG)
            with RUNNER_LOG.open(log_mode, encoding="utf-8", buffering=1) as runner_log:
                process = subprocess.Popen(
                    train_cmd,
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
                PID_PATH.write_text(str(process.pid), encoding="utf-8")
                print("Started PID:", process.pid)
                runner_log.write(f"Started PID: {process.pid}\\n")
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="")
                    runner_log.write(line)
                return_code = process.wait()
            if PID_PATH.exists() and PID_PATH.read_text(encoding="utf-8").strip() == str(process.pid):
                PID_PATH.unlink()
            if return_code != 0:
                raise RuntimeError(f"Training failed with exit code {return_code}. See {RUNNER_LOG} and {TRAIN_LOG}.")
            print("Training completed.")
        else:
            print("RUN_TRAINING=False, command was not launched.")
        """
    ),
    md(
        """
        ## 6. Status
        """
    ),
    code(
        """
        def tail_lines(path: Path, lines: int = 30) -> list[str]:
            if not path.exists():
                return []
            try:
                result = subprocess.run(["tail", "-n", str(lines), str(path)], capture_output=True, text=True, check=True)
                return result.stdout.splitlines()
            except Exception:
                with path.open(encoding="utf-8", errors="replace") as f:
                    return [line.rstrip("\\n") for line in deque(f, maxlen=lines)]


        history_path = WORK_ROOT / "history.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        pid = read_pid(PID_PATH)

        completed_phase1 = sum(1 for row in history if int(row.get("phase", 0)) == 1)
        completed_phase2 = sum(1 for row in history if int(row.get("phase", 0)) == 2)
        latest_global = Path(history[-1]["global"]) if history else WORK_ROOT / "global_checkpoints" / "round000_warmup.pt"

        display(
            pd.DataFrame(
                [
                    {
                        "pid": pid,
                        "pid_state": pid_state(pid),
                        "completed_phase1": f"{completed_phase1}/{PHASE1_ROUNDS}",
                        "completed_phase2": f"{completed_phase2}/{PHASE2_ROUNDS}",
                        "completed_total": f"{len(history)}/{PHASE1_ROUNDS + PHASE2_ROUNDS}",
                        "latest_global": str(latest_global),
                        "free_gib": round(shutil.disk_usage(WORK_ROOT).free / 1024**3, 2),
                    }
                ]
            )
        )

        print("Runner log tail:")
        for line in tail_lines(RUNNER_LOG, 35):
            print(line)
        print("\\nTrain log tail:")
        for line in tail_lines(TRAIN_LOG, 35):
            print(line)
        """
    ),
    md(
        """
        ## 7. Scene/Class Evaluation

        After the run finishes, set `RUN_EVAL=True`.  This evaluates warmup, final phase 1,
        and final phase 2 on highway/citystreet/residential/total, with per-class AP rows.
        """
    ),
    code(
        """
        RUN_EVAL = False
        EVAL_SPLITS = "highway,citystreet,residential,total"
        EVAL_BATCH_SIZE = 16
        EVAL_DEVICE = ""

        history = json.loads((WORK_ROOT / "history.json").read_text(encoding="utf-8")) if (WORK_ROOT / "history.json").exists() else []
        checkpoints: list[tuple[str, Path]] = []
        warmup = WORK_ROOT / "global_checkpoints" / "round000_warmup.pt"
        if warmup.exists():
            checkpoints.append(("warmup", warmup))
        phase1 = [row for row in history if int(row.get("phase", 0)) == 1]
        phase2 = [row for row in history if int(row.get("phase", 0)) == 2]
        if phase1:
            checkpoints.append((f"phase1_round{int(phase1[-1]['round']):03d}", Path(phase1[-1]["global"])))
        if phase2:
            checkpoints.append((f"phase2_round{int(phase2[-1]['round']):03d}", Path(phase2[-1]["global"])))

        eval_cmd = [
            str(PYTHON_BIN),
            str(EVAL_SCRIPT),
            "--workspace",
            str(WORK_ROOT),
            "--splits",
            EVAL_SPLITS,
            "--batch-size",
            str(EVAL_BATCH_SIZE),
            "--no-plots",
            "--verbose",
        ]
        if EVAL_DEVICE:
            eval_cmd.extend(["--device", EVAL_DEVICE])
        for label, path in checkpoints:
            eval_cmd.extend(["--checkpoint", f"{label}={path}"])

        print("checkpoints:", checkpoints)
        print(" ".join(eval_cmd))
        if RUN_EVAL and checkpoints:
            subprocess.run(eval_cmd, cwd=REPO_ROOT, check=True)
        elif RUN_EVAL:
            print("No checkpoints found to evaluate.")
        else:
            print("RUN_EVAL=False; set True after training finishes.")
        """
    ),
    md(
        """
        ## 8. Read Evaluation Tables
        """
    ),
    code(
        """
        summary_csv = WORK_ROOT / "validation_reports" / "paper_protocol_eval_summary.csv"
        classwise_csv = WORK_ROOT / "validation_reports" / "paper_protocol_classwise_summary.csv"

        if summary_csv.exists():
            summary = pd.read_csv(summary_csv)
            display(summary.sort_values(["checkpoint_label", "split"]))
        else:
            print("No summary yet:", summary_csv)

        if classwise_csv.exists():
            classwise = pd.read_csv(classwise_csv)
            display(classwise.sort_values(["split", "class", "map50_95"], ascending=[True, True, False]).head(80))
            pivot = classwise.pivot_table(
                index=["split", "class"],
                columns="checkpoint_label",
                values="map50_95",
                aggfunc="first",
            )
            display(pivot)
        else:
            print("No classwise summary yet:", classwise_csv)
        """
    ),
    md(
        """
        ## 9. DQA Stats Snapshot
        """
    ),
    code(
        """
        state_path = WORK_ROOT / "dqa_cwa_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            guard = state.get("round_guard", {})
            display(pd.DataFrame(guard.get("history", [])).tail(20))
            alpha = state.get("alpha", {})
            if alpha:
                latest_key = sorted(alpha)[-1]
                alpha_df = pd.DataFrame(alpha[latest_key]).T
                alpha_df.columns = latest_key.split("|")
                alpha_df.insert(0, "class", manifest["classes"])
                display(alpha_df)
        else:
            print("No DQA state yet:", state_path)

        stats_files = sorted(STATS_ROOT.glob("phase*_round*.json"))
        print("stats files:", len(stats_files), "root:", STATS_ROOT)
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


def main() -> None:
    OUT.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
