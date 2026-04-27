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


CORRECTED_12H_SETTING_TABLES = """
## DQA-CWA Setting Tables

以下は、このノートブックが読む `efficientteacher_dqa_cwa_corrected_12h` 実験の実数です。FedSTO論文表に対応させていますが、DQA-CWAはこのリポジトリ上の追加手法なので、論文に存在しないDQA固有値は実装値として書いています。

### まず結論

| 項目 | DQA-CWA corrected 12h の値 |
|---|---:|
| モデル | EfficientTeacher YOLOv5L style, `efficient-yolov5l.pt` |
| Backbone / Neck / Head | YoloV5 / YoloV5 / YoloV5 |
| depth_multiple / width_multiple | 1.00 / 1.00 |
| 入力サイズ | 640 |
| クラス数 | 10 |
| クラス | person / rider / car / bus / truck / bike / motor / traffic light / traffic sign / train |
| 学習データ合計 | 19,881 images |
| labeled | 4,881 images, server側, cloudy扱いの `partly cloudy` |
| unlabeled / pseudoGT対象 | 15,000 images, client側, overcast / rainy / snowy |
| labeled : unlabeled | 4,881 : 15,000 = 1 : 3.073 |
| labeled率 / unlabeled率 | 24.55% / 75.45% |
| client数 | 3 clients |
| client sampling ratio | 3 / 3 = 1.0 |
| warmup | 15 epochs |
| Phase 1 | 14 rounds, FedSTO-style backbone selective training |
| Phase 2 | 27 rounds, Full Parameter Training + Orthogonal + DQA-CWA |
| federated rounds合計 | 41 rounds |
| warmup込みの学習ステップ数 | 15 + 14 + 27 = 56 |
| local epoch | server / client ともに 1 |
| batch size | 64 |
| workers | 0 |
| requested GPUs | 2 |
| learning rate lr0 | 0.01 |
| EMA | 0.999 |
| NMS conf | 0.1 |
| NMS IoU | 0.65 |

### BDD100Kの画像枚数構成

| 設定 | serverのGT画像 | clientのunlabeled画像 | pseudoGT対象 | 比率 |
|---|---:|---:|---:|---:|
| DQA-CWA warmup | 4,881 | 0 | 0 | GTのみ |
| DQA-CWA Phase 1 | 4,881 | 15,000 | 15,000 images | 4,881 : 15,000 = 1 : 3.073 |
| DQA-CWA Phase 2 | 4,881 | 15,000 | 15,000 images | 4,881 : 15,000 = 1 : 3.073 |
| Fully supervised相当 | 19,881 | 0 | 0 | 全部GTなら19,881 |

### フェーズごとの学習

| フェーズ | 数 | 使うデータ | GT / pseudoGT | 更新する場所 | 集約 |
|---|---:|---|---|---|---|
| Warmup | 15 epochs | server labeled 4,881 | GT | 全パラメータ | なし |
| Phase 1 | 14 rounds | client unlabeled 15,000 + server labeled 4,881 | client側 pseudo label, server側 GT | backboneのみ | FedSTO-style backbone aggregation |
| Phase 2 | 27 rounds | client unlabeled 15,000 + server labeled 4,881 | client側 pseudo label, server側 GT | 全パラメータ, non_backboneにorthogonal | DQA-CWA class-wise head aggregation + BN-local FedAvg |

### 1 roundあたりの画像比率

| 場所 | 使う画像 | 画像数 / round | 役割 |
|---|---|---:|---|
| server | server_cloudy_train | 4,881 | GT supervised update |
| client 0 | overcast target | 5,000 | pseudoGT生成 + unsupervised update |
| client 1 | rainy target | 5,000 | pseudoGT生成 + unsupervised update |
| client 2 | snowy target | 5,000 | pseudoGT生成 + unsupervised update |
| clients合計 | overcast + rainy + snowy | 15,000 | pseudoGT対象 |
| 全体 | server + clients | 19,881 | GT : pseudoGT対象 = 4,881 : 15,000 |

### client条件

| 役割 | id | データ | weather条件 | 画像数 |
|---|---:|---|---|---:|
| Server | - | labeled | cloudy扱いの `partly cloudy` | 4,881 train / 738 val |
| Client | 0 | unlabeled | overcast | 5,000 |
| Client | 1 | unlabeled | rainy | 5,000 |
| Client | 2 | unlabeled | snowy | 5,000 |

### 20 clients / 100 clients相当

| 項目 | DQA-CWA corrected 12h の値 |
|---|---:|
| 実行済みclient数 | 3 |
| 20 clients実験 | 0 runs, 未実行 |
| 100 clients実験 | 0 runs, 未実行 |
| client sampling ratio比較 | 未実行 |
| 0.1の場合の参加client数 | N/A |
| 0.2の場合の参加client数 | N/A |
| 0.5の場合の参加client数 | N/A |
| 各client単一weather条件 | 3-client実験では true |

### 設定できる主な値

| 設定項目 | DQA-CWA corrected 12h の値 |
|---|---:|
| total federated rounds | 41 |
| warmup_epochs | 15 |
| Phase 1 rounds | 14 |
| Phase 2 rounds | 27 |
| dqa_start_phase | 2 |
| local epoch | 1 |
| batch_size | 64 |
| workers | 0 |
| gpus | 2 requested |
| optimizer adam | false |
| lr0 | 0.01 |
| lrf | 1.0 |
| momentum | 0.937 |
| weight_decay | 0.0005 |
| hyp warmup_epochs | 0 |
| warmup_momentum | 0.8 |
| warmup_bias_lr | 0.1 |
| class/object balance | class 0.3, object 0.7 |
| SSOD bbox loss weight | 0.05 |
| teacher_loss_weight | 1.0 |
| anchor threshold | 4.0 |
| ignore threshold | 0.1 to 0.6 |
| NMS confidence | 0.1 |
| NMS IoU | 0.65 |
| EMA rate | 0.999 |
| cosine EMA | true |
| img_size | 640 |
| mosaic | 1.0 |
| cutout | 0.5 |
| autoaugment | 0.5 |
| SSOD scale | 0.8 |
| hyp scale | 0.9 |
| mixup | 0.1 |
| hsv_h / hsv_s / hsv_v | 0.015 / 0.7 / 0.4 |
| degrees / shear | 0.0 / 0.0 |
| uncertain_aug | true |
| pseudo_label_with_obj / bbox / cls | true / true / true |
| use_ota / multi_label / ignore_obj | false / false / false |
| with_da_loss / da_loss_weights | false / 0.01 |
| resample_high_percent / resample_low_percent | 0.25 / 0.99 |
| Phase 1 train_scope | backbone |
| Phase 1 orthogonal_weight | 0.0 |
| Phase 2 train_scope | all |
| Phase 2 orthogonal_weight | 0.0001 |
| orthogonal_scope | non_backbone |

### DQA-CWA固有値

| 設定項目 | 値 |
|---|---:|
| count_ema | 0.70 |
| quality_ema | 0.70 |
| alpha_ema | 0.50 |
| temperature | 1.50 |
| uniform_mix | 0.05 |
| classwise_blend | 0.35 |
| stability_lambda | 0.25 |
| min_effective_count | 1.0 |
| min_quality / max_quality | 0.05 / 1.0 |
| server_anchor | 1.25 |
| localize_bn | true |
| enable_dqa_guard | true |
| dqa_min_round_pseudo_count | 1.0 |
| dqa_drop_ratio_threshold | 0.15 |
| dqa_spike_ratio_threshold | 3.0 |
| dqa_guard_count_ema | 0.70 |
| quality formula weights | confidence 0.50, objectness 0.20, class confidence 0.20, localization 0.10 |

### DQAで記録されたpseudoGT box数

| 項目 | 値 |
|---|---:|
| DQA statsがあるphase | Phase 2のみ |
| round-level stats files | 27 |
| client stats files | 81 |
| stats files合計 | 108 |
| DQA使用round | 27 / 27 |
| DQA guard skip | 0 / 27 |
| pseudo box総数 | 14,339,686 |
| pseudo box平均 / round | 531,099.48 |
| pseudo box最小 | 317,369 at phase2 round 1 |
| pseudo box最大 | 819,890 at phase2 round 19 |
| latest phase2 round 27 pseudo box数 | 521,421 |
| mean quality平均 | 0.706280 |
| mean quality最小 | 0.573588 at phase2 round 7 |
| mean quality最大 | 0.803907 at phase2 round 3 |
| latest phase2 round 27 mean quality | 0.738849 |
| active classes | 9 classes in 26 rounds, 10 classes in 1 round |

### Phase 2累積pseudo box数 class/client別

| class | total | client 0 overcast | client 1 rainy | client 2 snowy |
|---|---:|---:|---:|---:|
| person | 1,110,807 | 626,339 | 170,103 | 314,365 |
| rider | 31,564 | 24,374 | 2,034 | 5,156 |
| car | 8,561,515 | 1,506,469 | 2,871,856 | 4,183,190 |
| bus | 59,039 | 49,772 | 4,754 | 4,513 |
| truck | 85,510 | 49,799 | 11,096 | 24,615 |
| bike | 16,601 | 14,006 | 906 | 1,689 |
| motor | 3,906 | 3,369 | 201 | 336 |
| traffic light | 1,451,363 | 118,698 | 847,286 | 485,379 |
| traffic sign | 3,019,368 | 2,453,904 | 208,526 | 356,938 |
| train | 13 | 0 | 0 | 13 |
| all classes | 14,339,686 | 4,846,730 | 4,116,762 | 5,376,194 |
"""


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
    import os
    import re
    import signal
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
        r"Skipping DQA-CWA for phase|Falling back to BN-local FedAvg|"
        r"Dry run complete|Training failed|Traceback|RuntimeError|Exception|Error|out of memory|CUDA error)",
        re.IGNORECASE,
    )


    def compact_line(line: str, limit: int = 240) -> str:
        text = line.replace("\\r", "").strip()
        return text if len(text) <= limit else text[: limit - 3] + "..."


    def tail_lines(path: Path, lines: int = 80) -> list[str]:
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
            with path.open(encoding="utf-8", errors="replace") as f:
                return [line.rstrip("\\n") for line in deque(f, maxlen=lines)]


    def print_tail(path: Path, label: str, lines: int = 80) -> None:
        rows = tail_lines(path, lines)
        if not rows:
            print(f"{label}: not found or empty: {path}")
            return
        print(f"{label}: {path}")
        for row in rows:
            print(compact_line(row, limit=500))


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
        log_mode = "a" if APPEND_TRAIN_LOG else "w"
        with RUNNER_LOG.open(log_mode, encoding="utf-8", buffering=1) as runner_log:
            runner_log.write("\\n\\n===== DQA-CWA notebook run started =====\\n")
            runner_log.write("Running: " + " ".join(cmd) + "\\n")
            process = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            PID_PATH.write_text(str(process.pid) + "\\n", encoding="utf-8")
            assert process.stdout is not None
            try:
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
            except KeyboardInterrupt:
                print("KeyboardInterrupt received; stopping the DQA runner and child training process.")
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        process.kill()
                    process.wait(timeout=30)
                raise
            finally:
                if PID_PATH.exists():
                    PID_PATH.unlink()
        if return_code != 0:
            print("Last captured lines:")
            for line in recent:
                text = compact_line(line)
                if text:
                    print(text)
            print_tail(TRAIN_LOG, "Train log tail", lines=120)
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
    EVAL_SPLITS = "cloudy"
    EVAL_BATCH_SIZE = 32
    EVAL_EXTRA_ARGS: list[str] = ["--no-plots"]

    history_path = WORK_ROOT / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    eval_checkpoints = []
    if history:
        latest = Path(history[-1]["global"])
        eval_checkpoints.append(f"latest_global={latest}")
    elif (WORK_ROOT / "global_checkpoints" / "round000_warmup.pt").exists():
        eval_checkpoints.append(f"warmup_global={WORK_ROOT / 'global_checkpoints' / 'round000_warmup.pt'}")

    eval_cmd = [
        str(PYTHON_BIN),
        str(EVAL_SCRIPT),
        "--workspace",
        str(WORK_ROOT),
        "--splits",
        EVAL_SPLITS,
        "--batch-size",
        str(EVAL_BATCH_SIZE),
    ]
    for spec in eval_checkpoints:
        eval_cmd.extend(["--checkpoint", spec])
    eval_cmd.extend(EVAL_EXTRA_ARGS)

    if RUN_EVAL and eval_checkpoints:
        subprocess.run(eval_cmd, cwd=REPO_ROOT, check=True)
    elif RUN_EVAL:
        print("Skipping evaluation because no global checkpoints exist yet:", WORK_ROOT / "global_checkpoints")

    summary_path = WORK_ROOT / "validation_reports" / "paper_protocol_eval_summary.csv"
    if summary_path.exists():
        eval_summary = pd.read_csv(summary_path)
        display(eval_summary)
    else:
        print("No DQA paper-protocol summary yet:", summary_path)
        print("Default evaluation is a quick cloudy/latest smoke test with plots disabled.")
        print("For the full table, set EVAL_SPLITS = 'cloudy,overcast,rainy,snowy,total' and add more checkpoints.")
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
    runner_script_name: str = "run_dqa_cwa_fedsto.py",
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
        code('print("Hello, World!")'),
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
            import socket
            import subprocess
            import sys
            from datetime import datetime, timezone
            from pathlib import Path
            from typing import Optional

            import pandas as pd


            def find_repo_root(start: Optional[Path] = None) -> Path:
                start = Path.cwd().resolve() if start is None else Path(start).resolve()
                required = (
                    "dynamic_quality_aware_classwise_aggregation/{runner_script_name}",
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
            RUN_SCRIPT = DQA_ROOT / "{runner_script_name}"
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
            DQA_START_PHASE = 2
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


            MASTER_PORT = find_free_port({master_port})
            MIN_FREE_GIB = {min_free_gib}

            APPEND_TRAIN_LOG = False
            EXTRA_RUN_ARGS: list[str] = [
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
                        *EXTRA_RUN_ARGS,
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

                This cell is intentionally quick by default: latest checkpoint, cloudy split, batch size 32, and plots disabled. Use the full split list only when you are ready for the slower paper table.
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
    setting_tables_markdown: str | None = None,
    method_label: str = "DQA-CWA",
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
    METHOD_LABEL = "__METHOD_LABEL__"

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


    def plot_heatmap(
        ax,
        matrix,
        *,
        row_labels: list[str],
        col_labels: list[str],
        title: str,
        cmap: str = "viridis",
        fmt: str = ".2f",
        annotate: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
    ):
        values = np.asarray(matrix, dtype=float)
        if sns is not None:
            sns.heatmap(
                values,
                ax=ax,
                cmap=cmap,
                annot=annotate,
                fmt=fmt,
                xticklabels=col_labels,
                yticklabels=row_labels,
                vmin=vmin,
                vmax=vmax,
            )
        else:
            image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, rotation=45, ha="right")
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels)
            if annotate:
                for row_idx in range(values.shape[0]):
                    for col_idx in range(values.shape[1]):
                        ax.text(
                            col_idx,
                            row_idx,
                            format(values[row_idx, col_idx], fmt),
                            ha="center",
                            va="center",
                            color="white" if values[row_idx, col_idx] > np.nanmean(values) else "black",
                            fontsize=9,
                        )
        ax.set_title(title)
        ax.set_xlabel("class")
        ax.set_ylabel("")
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
        .replace("__METHOD_LABEL__", method_label)
    )

    cells = [
        code('print("Hello, World!")'),
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
            ## 5. Plateau and Loss Diagnostics

            These views are meant to answer the practical question quickly: is phase 2 still improving, or has it already flattened into pseudo-label noise?
            """
        ),
        code(
            """
            warmup_map50 = float(warmup_curve["metrics/mAP_0.5"].max())
            warmup_map50_95 = float(warmup_curve["metrics/mAP_0.5:0.95"].max())

            phase_detail_frames = []
            for phase in [1, 2]:
                phase_df = (
                    server_summary[(server_summary["phase"] == phase) & (server_summary["role"] == "server")]
                    .sort_values("round")
                    .copy()
                )
                if phase_df.empty:
                    continue
                phase_df["cummax_map50"] = phase_df["final_metrics/mAP_0.5"].cummax()
                phase_df["cummax_map50_95"] = phase_df["final_metrics/mAP_0.5:0.95"].cummax()
                phase_df["delta_prev_map50"] = phase_df["final_metrics/mAP_0.5"].diff()
                phase_df["delta_prev_map50_95"] = phase_df["final_metrics/mAP_0.5:0.95"].diff()
                phase_df["delta_vs_warmup_map50"] = phase_df["final_metrics/mAP_0.5"] - warmup_map50
                phase_df["delta_vs_warmup_map50_95"] = phase_df["final_metrics/mAP_0.5:0.95"] - warmup_map50_95
                phase_detail_frames.append(phase_df)

            phase_detail_df = pd.concat(phase_detail_frames, ignore_index=True) if phase_detail_frames else pd.DataFrame()

            if not phase_detail_df.empty:
                display(
                    phase_detail_df[
                        [
                            "phase",
                            "round",
                            "final_metrics/mAP_0.5",
                            "final_metrics/mAP_0.5:0.95",
                            "cummax_map50",
                            "cummax_map50_95",
                            "delta_prev_map50",
                            "delta_prev_map50_95",
                            "delta_vs_warmup_map50",
                            "delta_vs_warmup_map50_95",
                        ]
                    ].tail(10).round(4)
                )

                fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex="col")
                for row_idx, phase in enumerate([1, 2]):
                    phase_df = phase_detail_df[phase_detail_df["phase"] == phase].copy()
                    if phase_df.empty:
                        axes[row_idx, 0].set_visible(False)
                        axes[row_idx, 1].set_visible(False)
                        continue

                    ax_curve = axes[row_idx, 0]
                    ax_curve.plot(phase_df["round"], phase_df["final_metrics/mAP_0.5"], marker="o", label="mAP@0.5")
                    ax_curve.plot(phase_df["round"], phase_df["final_metrics/mAP_0.5:0.95"], marker="o", label="mAP@0.5:0.95")
                    ax_curve.plot(phase_df["round"], phase_df["cummax_map50"], linestyle="--", linewidth=1.8, label="best-so-far @0.5")
                    ax_curve.plot(phase_df["round"], phase_df["cummax_map50_95"], linestyle="--", linewidth=1.8, label="best-so-far @0.5:0.95")
                    ax_curve.axhline(warmup_map50, color="tab:blue", linestyle=":", linewidth=1.2, label="warmup best @0.5")
                    ax_curve.axhline(warmup_map50_95, color="tab:orange", linestyle=":", linewidth=1.2, label="warmup best @0.5:0.95")
                    ax_curve.set_title(f"phase {phase}: server mAP and running best")
                    ax_curve.set_xlabel("round")
                    ax_curve.set_ylabel("score")
                    ax_curve.legend(fontsize=9)

                    ax_delta = axes[row_idx, 1]
                    ax_delta.bar(phase_df["round"] - 0.15, phase_df["delta_prev_map50"].fillna(0.0), width=0.3, label="delta mAP@0.5")
                    ax_delta.bar(phase_df["round"] + 0.15, phase_df["delta_prev_map50_95"].fillna(0.0), width=0.3, label="delta mAP@0.5:0.95")
                    ax_delta.axhline(0.0, color="black", linewidth=1.0)
                    ax_delta.set_title(f"phase {phase}: per-round metric change")
                    ax_delta.set_xlabel("round")
                    ax_delta.set_ylabel("delta from previous round")
                    ax_delta.legend(fontsize=9)

                plt.tight_layout()
                plt.show()

                phase2_detail = phase_detail_df[phase_detail_df["phase"] == 2].copy()
                if not phase2_detail.empty:
                    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
                    loss_specs = [
                        ("box", "box loss"),
                        ("obj", "objectness loss"),
                        ("cls", "class loss"),
                    ]
                    for ax, (loss_key, title) in zip(axes, loss_specs):
                        ax.plot(phase2_detail["round"], phase2_detail[f"final_train/{loss_key}_loss"], marker="o", label="train")
                        ax.plot(phase2_detail["round"], phase2_detail[f"final_val/{loss_key}_loss"], marker="o", label="val")
                        ax.set_title(f"phase 2 server {title}")
                        ax.set_xlabel("round")
                        ax.set_ylabel("loss")
                        ax.legend(fontsize=9)
                    plt.tight_layout()
                    plt.show()
            """
        ),
        md(
            """
            ## 6. DQA Guard and Pseudo-Label Diagnostics

            These plots connect the DQA-specific internals to the accuracy curve: pseudo-label volume, quality, whether the round guard actually skipped DQA, and how the final class-wise weights were distributed.
            """
        ),
        code(
            """
            dqa_state_path = WORK_ROOT / "dqa_cwa_state.json"
            client_weather_map = {str(client.get("id")): client.get("weather", f"client{client.get('id')}") for client in manifest.get("clients", [])}
            class_names = manifest.get("classes", [str(i) for i in range(10)])

            def is_round_stats_file(path: Path) -> bool:
                return bool(re.fullmatch(r"phase\\d+_round\\d+\\.json", path.name))


            def weighted_average(sum_value: float, total_count: float) -> float:
                return float(sum_value) / float(total_count) if total_count > 0 else 0.0


            round_rows = []
            weather_rows = []
            latest_round_clients = []
            round_stats_paths = sorted(path for path in STATS_ROOT.glob("phase*_round*.json") if is_round_stats_file(path))
            latest_round_number = -1

            for path in round_stats_paths:
                match = re.fullmatch(r"phase(?P<phase>\\d+)_round(?P<round>\\d+)\\.json", path.name)
                if not match:
                    continue
                phase = int(match.group("phase"))
                round_idx = int(match.group("round"))
                data = json.loads(path.read_text(encoding="utf-8"))
                clients = data.get("clients", [])

                total_count = 0.0
                quality_sum = 0.0
                confidence_sum = 0.0
                objectness_sum = 0.0
                phase_active_classes = 0
                final_clients_snapshot = []

                for client in clients:
                    counts = [float(x) for x in client.get("counts", [])]
                    qualities = [float(x) for x in client.get("mean_quality_scores", [])]
                    confidences = [float(x) for x in client.get("mean_confidences", [])]
                    objectness = [float(x) for x in client.get("mean_objectness", [])]
                    client_id = str(client.get("id", client.get("client_id", "?")))
                    weather = client_weather_map.get(client_id, f"client{client_id}")
                    client_total = float(sum(counts))

                    total_count += client_total
                    quality_sum += sum(count * quality for count, quality in zip(counts, qualities))
                    confidence_sum += sum(count * confidence for count, confidence in zip(counts, confidences))
                    objectness_sum += sum(count * obj for count, obj in zip(counts, objectness))
                    phase_active_classes = max(phase_active_classes, sum(1 for count in counts if count > 0))

                    weather_rows.append(
                        {
                            "phase": phase,
                            "round": round_idx,
                            "client_id": client_id,
                            "weather": weather,
                            "pseudo_count": client_total,
                            "mean_quality": weighted_average(sum(count * quality for count, quality in zip(counts, qualities)), client_total),
                            "mean_confidence": weighted_average(sum(count * confidence for count, confidence in zip(counts, confidences)), client_total),
                            "mean_objectness": weighted_average(sum(count * obj for count, obj in zip(counts, objectness)), client_total),
                        }
                    )
                    final_clients_snapshot.append({"weather": weather, "counts": counts, "qualities": qualities})

                round_rows.append(
                    {
                        "phase": phase,
                        "round": round_idx,
                        "total_pseudo_count": total_count,
                        "mean_quality": weighted_average(quality_sum, total_count),
                        "mean_confidence": weighted_average(confidence_sum, total_count),
                        "mean_objectness": weighted_average(objectness_sum, total_count),
                        "active_classes_from_stats": phase_active_classes,
                    }
                )

                if phase == 2 and round_idx >= latest_round_number:
                    latest_round_number = round_idx
                    latest_round_clients = final_clients_snapshot

            dqa_round_df = pd.DataFrame(round_rows).sort_values(["phase", "round"])
            dqa_weather_df = pd.DataFrame(weather_rows).sort_values(["phase", "round", "weather"])

            if dqa_state_path.exists():
                state = json.loads(dqa_state_path.read_text(encoding="utf-8"))
                guard_history = pd.DataFrame(state.get("round_guard", {}).get("history", []))
                if not guard_history.empty:
                    guard_history = guard_history.rename(columns={"active_classes": "guard_active_classes"})
                    dqa_round_df = dqa_round_df.merge(
                        guard_history[["phase", "round", "used_dqa", "reason", "total_count", "mean_quality", "guard_active_classes"]],
                        on=["phase", "round"],
                        how="left",
                        suffixes=("", "_guard"),
                    )
                last_sources = state.get("last_sources", [])
                last_alpha = state.get("last_alpha", [])
            else:
                state = {}
                last_sources = []
                last_alpha = []

            if not dqa_round_df.empty:
                display(
                    dqa_round_df[
                        [
                            "phase",
                            "round",
                            "total_pseudo_count",
                            "mean_quality",
                            "mean_confidence",
                            "mean_objectness",
                            "used_dqa",
                            "guard_active_classes",
                        ]
                    ].tail(12).round(4)
                )

                phase2_round_df = dqa_round_df[dqa_round_df["phase"] == 2].copy()
                if not phase2_round_df.empty:
                    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)

                    ax = axes[0, 0]
                    ax.plot(phase2_round_df["round"], phase2_round_df["total_pseudo_count"], marker="o", label="total pseudo count")
                    ax.set_title("phase 2 pseudo-label volume")
                    ax.set_ylabel("pseudo count")
                    ax.legend(fontsize=9)

                    ax = axes[0, 1]
                    ax.plot(phase2_round_df["round"], phase2_round_df["mean_quality"], marker="o", label="mean quality")
                    ax.plot(phase2_round_df["round"], phase2_round_df["mean_confidence"], marker="o", label="mean confidence")
                    ax.plot(phase2_round_df["round"], phase2_round_df["mean_objectness"], marker="o", label="mean objectness")
                    ax.set_title("phase 2 pseudo-label quality")
                    ax.set_ylabel("score")
                    ax.legend(fontsize=9)

                    ax = axes[1, 0]
                    if not dqa_weather_df.empty:
                        phase2_weather = dqa_weather_df[dqa_weather_df["phase"] == 2].copy()
                        plot_line(
                            ax,
                            phase2_weather,
                            x="round",
                            y="pseudo_count",
                            hue="weather",
                            marker="o",
                        )
                    ax.set_title("phase 2 pseudo count by weather")
                    ax.set_xlabel("round")
                    ax.set_ylabel("pseudo count")

                    ax = axes[1, 1]
                    if "used_dqa" in phase2_round_df.columns:
                        ax.step(
                            phase2_round_df["round"],
                            phase2_round_df["used_dqa"].fillna(False).astype(int),
                            where="mid",
                            label="used DQA",
                        )
                    if "guard_active_classes" in phase2_round_df.columns:
                        ax.plot(
                            phase2_round_df["round"],
                            phase2_round_df["guard_active_classes"].fillna(0),
                            marker="o",
                            label="active classes",
                        )
                    ax.set_title("phase 2 guard decisions")
                    ax.set_xlabel("round")
                    ax.set_ylabel("value")
                    ax.legend(fontsize=9)

                    plt.tight_layout()
                    plt.show()

            if last_sources and last_alpha:
                alpha_columns = class_names[: len(last_alpha[0])]
                alpha_index = [source.replace("client:", "client ").replace("server", "server anchor") for source in last_sources]
                alpha_df = pd.DataFrame(last_alpha, index=alpha_index, columns=alpha_columns)
                plt.figure(figsize=(12, 4))
                plot_heatmap(
                    plt.gca(),
                    alpha_df.values,
                    row_labels=list(alpha_df.index),
                    col_labels=list(alpha_df.columns),
                    title="final class-wise aggregation weights",
                    cmap="magma",
                    fmt=".2f",
                    vmin=0.0,
                    vmax=max(float(np.nanmax(alpha_df.values)), 0.35),
                )
                plt.tight_layout()
                plt.show()
                display(alpha_df.round(3))

            if latest_round_clients:
                final_quality_df = pd.DataFrame(
                    [client["qualities"] for client in latest_round_clients],
                    index=[client["weather"] for client in latest_round_clients],
                    columns=class_names[: len(latest_round_clients[0]["qualities"])],
                )
                final_count_df = pd.DataFrame(
                    [client["counts"] for client in latest_round_clients],
                    index=[client["weather"] for client in latest_round_clients],
                    columns=class_names[: len(latest_round_clients[0]["counts"])],
                )
                fig, axes = plt.subplots(1, 2, figsize=(18, 5))
                plot_heatmap(
                    axes[0],
                    final_quality_df.values,
                    row_labels=list(final_quality_df.index),
                    col_labels=list(final_quality_df.columns),
                    title=f"round {latest_round_number} mean pseudo quality by weather/class",
                    cmap="YlGnBu",
                    fmt=".2f",
                    vmin=0.0,
                    vmax=1.0,
                )
                plot_heatmap(
                    axes[1],
                    np.log10(final_count_df.values + 1.0),
                    row_labels=list(final_count_df.index),
                    col_labels=list(final_count_df.columns),
                    title=f"round {latest_round_number} log10 pseudo count by weather/class",
                    cmap="viridis",
                    fmt=".1f",
                )
                plt.tight_layout()
                plt.show()
            """
        ),
        md(
            """
            ## 7. Client Weather Trajectories

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
            ## 8. DQA vs FedSTO Snapshot

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


            comparison_frames = [comparable_checkpoint_rows(run_summary, METHOD_LABEL)]
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
            ## 9. FedSTO vs DQA Round Curves

            The checkpoint snapshot is useful, but the curves matter more for the "should we just add rounds?" question. This section overlays both methods round by round when the FedSTO training summary is available.
            """
        ),
        code(
            """
            if FEDSTO_TRAINING_SUMMARY.exists():
                fedsto_summary = pd.read_csv(FEDSTO_TRAINING_SUMMARY)
                fedsto_server = fedsto_summary[fedsto_summary["role"].isin(["warmup", "server"])].copy()

                compare_rows = []
                for method, frame in [(METHOD_LABEL, server_summary.copy()), ("FedSTO", fedsto_server.copy())]:
                    method_df = frame[frame["role"] == "server"].copy()
                    if method_df.empty:
                        continue
                    for phase in [1, 2]:
                        phase_df = method_df[method_df["phase"] == phase].sort_values("round").copy()
                        if phase_df.empty:
                            continue
                        phase_df["method"] = method
                        compare_rows.append(phase_df)

                if compare_rows:
                    compare_df = pd.concat(compare_rows, ignore_index=True)
                    fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=True)
                    for ax, phase in zip(axes, [1, 2]):
                        phase_df = compare_df[compare_df["phase"] == phase].copy()
                        plot_line(
                            ax,
                            phase_df,
                            x="round",
                            y="final_metrics/mAP_0.5:0.95",
                            hue="method",
                            marker="o",
                        )
                        ax.set_title(f"phase {phase} server mAP@0.5:0.95")
                        ax.set_xlabel("round")
                        ax.set_ylabel("mAP@0.5:0.95")
                    plt.tight_layout()
                    plt.show()

                    fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=True)
                    for ax, phase in zip(axes, [1, 2]):
                        phase_df = compare_df[compare_df["phase"] == phase].copy()
                        plot_line(
                            ax,
                            phase_df,
                            x="round",
                            y="final_metrics/mAP_0.5",
                            hue="method",
                            marker="o",
                        )
                        ax.set_title(f"phase {phase} server mAP@0.5")
                        ax.set_xlabel("round")
                        ax.set_ylabel("mAP@0.5")
                    plt.tight_layout()
                    plt.show()
            else:
                print("FedSTO training summary not found:", FEDSTO_TRAINING_SUMMARY)
            """
        ),
        md(
            """
            ## 10. Paper-Eval Status and Visual Artifacts

            Use the complete paper-eval summary if it exists. If not, fall back to partial DQA evaluation artifacts so we can still see what has already been produced.
            """
        ),
        code(
            """
            paper_eval_frames = []
            dqa_paper_eval_summary = VALIDATION_ROOT / "paper_protocol_eval_summary.csv"
            if dqa_paper_eval_summary.exists():
                dqa_eval_df = pd.read_csv(dqa_paper_eval_summary)
                dqa_eval_df.insert(0, "method", METHOD_LABEL)
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
            ## 11. Artifact Index

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

    if setting_tables_markdown:
        cells.insert(3, md(setting_tables_markdown))

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


def build_paper_eval_notebook(
    *,
    notebook_title: str,
    notebook_path: Path,
    default_workspace_name: str,
    notebook_description: str,
) -> None:
    setup_text = """
    from __future__ import annotations

    import json
    import re
    import shlex
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
    EVAL_SCRIPT = DQA_ROOT / "evaluate_paper_protocol.py"
    NOTEBOOK_GENERATOR = DQA_ROOT / "__GENERATOR_NAME__"

    WORKSPACE_NAME = "__DEFAULT_WORKSPACE_NAME__"
    ALTERNATIVE_WORKSPACES = [
        "efficientteacher_dqa_cwa_corrected_12h",
        "efficientteacher_dqa_cwa_14h",
        "efficientteacher_dqa_cwa_exact",
        "efficientteacher_dqa_cwa",
    ]
    WORK_ROOT = DQA_ROOT / WORKSPACE_NAME
    VALIDATION_ROOT = WORK_ROOT / "validation_reports"
    RUN_ROOT = VALIDATION_ROOT / "paper_protocol_val_runs"
    LOG_ROOT = VALIDATION_ROOT / "paper_protocol_logs"
    FEDSTO_WORK_ROOT = NAV_ROOT / "efficientteacher_fedsto"
    FEDSTO_PAPER_EVAL_SUMMARY = FEDSTO_WORK_ROOT / "validation_reports" / "paper_protocol_eval_summary.csv"

    preferred_python = Path("/root/micromamba/envs/al_yolov8/bin/python")
    PYTHON_BIN = preferred_python if preferred_python.exists() else Path(sys.executable)

    if sns is not None:
        sns.set_theme(style="whitegrid", context="talk")
    else:
        plt.style.use("ggplot")
    pd.options.display.max_columns = 200
    pd.options.display.max_rows = 200


    def modified_utc(path: Path) -> str:
        if not path.exists():
            return ""
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


    def parse_float(value) -> float | None:
        if value in (None, "") or pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


    def best_phase2_round_from_summary(summary_csv: Path, basis: str) -> int | None:
        if not summary_csv.exists():
            return None
        metric_col = {
            "map50": "final_metrics/mAP_0.5",
            "map50_95": "final_metrics/mAP_0.5:0.95",
        }[basis]
        summary = pd.read_csv(summary_csv)
        phase2 = summary[(summary["phase"] == 2) & (summary["role"] == "server")].copy()
        if phase2.empty or metric_col not in phase2.columns:
            return None
        phase2 = phase2.sort_values(metric_col, ascending=False)
        return int(phase2.iloc[0]["round"])


    def auto_checkpoint_specs(workspace: Path, best_basis: str) -> list[tuple[str, Path]]:
        history_path = workspace / "history.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        global_dir = workspace / "global_checkpoints"
        specs: list[tuple[str, Path]] = []
        seen: set[Path] = set()

        def add(label: str, path: Path) -> None:
            resolved = path.resolve()
            if not path.exists() or resolved in seen:
                return
            seen.add(resolved)
            specs.append((label, resolved))

        add("warmup_global", global_dir / "round000_warmup.pt")

        phase1 = [entry for entry in history if int(entry.get("phase", 0)) == 1]
        if phase1:
            add(f"phase1_round{int(phase1[-1]['round']):03d}_global", Path(phase1[-1]["global"]))

        summary_csv = workspace / "validation_reports" / "tables" / "training_run_summary.csv"
        best_round = best_phase2_round_from_summary(summary_csv, best_basis)
        if best_round is not None:
            add(
                f"phase2_best_server_cloudy_round{best_round:03d}_global",
                global_dir / f"phase2_round{best_round:03d}_global.pt",
            )

        phase2 = [entry for entry in history if int(entry.get("phase", 0)) == 2]
        if phase2:
            add(f"phase2_round{int(phase2[-1]['round']):03d}_global", Path(phase2[-1]["global"]))
        return specs


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


    def plot_heatmap(
        ax,
        matrix,
        *,
        row_labels: list[str],
        col_labels: list[str],
        title: str,
        cmap: str = "viridis",
        fmt: str = ".3f",
        annotate: bool = True,
    ):
        values = np.asarray(matrix, dtype=float)
        if sns is not None:
            sns.heatmap(
                values,
                ax=ax,
                cmap=cmap,
                annot=annotate,
                fmt=fmt,
                xticklabels=col_labels,
                yticklabels=row_labels,
            )
        else:
            image = ax.imshow(values, aspect="auto", cmap=cmap)
            ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, rotation=35, ha="right")
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels)
        ax.set_title(title)
        ax.set_xlabel("split")
        ax.set_ylabel("checkpoint")
        return ax


    print("repo_root:", REPO_ROOT)
    print("workspace_name:", WORKSPACE_NAME)
    print("workspace:", WORK_ROOT)
    print("validation_root:", VALIDATION_ROOT)
    print("python:", PYTHON_BIN)
    print("alternatives:", ", ".join(ALTERNATIVE_WORKSPACES))
    """
    setup_text = (
        setup_text.replace("__DEFAULT_WORKSPACE_NAME__", default_workspace_name)
        .replace("__GENERATOR_NAME__", GENERATOR_PATH.name)
    )

    cells = [
        code('print("Hello, World!")'),
        md(
            f"""
            # {notebook_title}

            {notebook_description}
            """
        ),
        code(setup_text),
        md(
            """
            ## 1. Workspace Status and Auto-Selected Checkpoints

            This notebook is focused on the paper-style validation protocol. It first checks whether the selected workspace exists and which checkpoints the shared evaluation script would pick by default.
            """
        ),
        code(
            """
            manifest_path = WORK_ROOT / "manifest.json"
            history_path = WORK_ROOT / "history.json"
            summary_csv = VALIDATION_ROOT / "tables" / "training_run_summary.csv"
            paper_summary_csv = VALIDATION_ROOT / "paper_protocol_eval_summary.csv"
            paper_manifest_json = VALIDATION_ROOT / "paper_protocol_eval_manifest.json"

            history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
            auto_specs_preview = auto_checkpoint_specs(WORK_ROOT, "map50")

            artifact_rows = [
                {"artifact": "workspace", "path": str(WORK_ROOT), "exists": WORK_ROOT.exists(), "modified_utc": modified_utc(WORK_ROOT)},
                {"artifact": "manifest", "path": str(manifest_path), "exists": manifest_path.exists(), "modified_utc": modified_utc(manifest_path)},
                {"artifact": "history", "path": str(history_path), "exists": history_path.exists(), "modified_utc": modified_utc(history_path)},
                {"artifact": "training_summary_csv", "path": str(summary_csv), "exists": summary_csv.exists(), "modified_utc": modified_utc(summary_csv)},
                {"artifact": "paper_eval_summary_csv", "path": str(paper_summary_csv), "exists": paper_summary_csv.exists(), "modified_utc": modified_utc(paper_summary_csv)},
                {"artifact": "paper_eval_manifest_json", "path": str(paper_manifest_json), "exists": paper_manifest_json.exists(), "modified_utc": modified_utc(paper_manifest_json)},
            ]
            display(pd.DataFrame(artifact_rows))

            history_rows = {
                "completed_phase1_rounds": sum(1 for entry in history if int(entry.get("phase", 0)) == 1),
                "completed_phase2_rounds": sum(1 for entry in history if int(entry.get("phase", 0)) == 2),
                "history_entries": len(history),
                "classes": manifest.get("classes", []),
                "client_weathers": [client.get("weather") for client in manifest.get("clients", [])],
            }
            history_rows

            if auto_specs_preview:
                display(
                    pd.DataFrame(
                        [{"checkpoint_label": label, "checkpoint_path": str(path)} for label, path in auto_specs_preview]
                    )
                )
            else:
                print("No auto-selected checkpoints are available yet for this workspace.")
            """
        ),
        md(
            """
            ## 2. Paper-Eval Controls

            The defaults below run the full paper-style split set on the current workspace. Set `RUN_PAPER_EVAL = False` if you only want to inspect already-generated outputs.
            """
        ),
        code(
            """
            RUN_PAPER_EVAL = True
            DRY_RUN = False
            BEST_BASIS = "map50"
            EVAL_SPLITS = "cloudy,overcast,rainy,snowy,total"
            BATCH_SIZE = 8
            IMGSZ = 640
            CONF_THRES = 0.001
            IOU_THRES = 0.6
            DEVICE = ""
            PLOTS = True
            VERBOSE = False

            # Leave this empty to use the script's auto checkpoint selection:
            #   warmup_global, phase1 final, phase2 best-by-BEST_BASIS, phase2 final.
            # Otherwise add items like:
            # EXPLICIT_CHECKPOINTS = [
            #     ("my_label", WORK_ROOT / "global_checkpoints" / "phase2_round024_global.pt"),
            # ]
            EXPLICIT_CHECKPOINTS: list[tuple[str, Path]] = []

            {
                "run_paper_eval": RUN_PAPER_EVAL,
                "dry_run": DRY_RUN,
                "best_basis": BEST_BASIS,
                "eval_splits": EVAL_SPLITS,
                "batch_size": BATCH_SIZE,
                "imgsz": IMGSZ,
                "conf_thres": CONF_THRES,
                "iou_thres": IOU_THRES,
                "device": DEVICE,
                "plots": PLOTS,
                "verbose": VERBOSE,
                "explicit_checkpoints": [(label, str(path)) for label, path in EXPLICIT_CHECKPOINTS],
            }
            """
        ),
        md(
            """
            ## 3. Run the Shared Paper Protocol Evaluation

            This calls the shared `evaluate_paper_protocol.py` entrypoint. It writes the CSV, Markdown summary, manifest, logs, and per-split validation run directories under `validation_reports/`.
            """
        ),
        code(
            """
            checkpoint_specs = EXPLICIT_CHECKPOINTS if EXPLICIT_CHECKPOINTS else auto_checkpoint_specs(WORK_ROOT, BEST_BASIS)

            eval_cmd = [
                str(PYTHON_BIN),
                str(EVAL_SCRIPT),
                "--workspace",
                str(WORK_ROOT),
                "--splits",
                EVAL_SPLITS,
                "--best-basis",
                BEST_BASIS,
                "--batch-size",
                str(BATCH_SIZE),
                "--imgsz",
                str(IMGSZ),
                "--conf-thres",
                str(CONF_THRES),
                "--iou-thres",
                str(IOU_THRES),
            ]
            if DEVICE:
                eval_cmd.extend(["--device", DEVICE])
            if PLOTS:
                eval_cmd.append("--plots")
            else:
                eval_cmd.append("--no-plots")
            if VERBOSE:
                eval_cmd.append("--verbose")
            if DRY_RUN:
                eval_cmd.append("--dry-run")
            for label, path in checkpoint_specs:
                eval_cmd.extend(["--checkpoint", f"{label}={path}"])

            print("Resolved checkpoints:")
            if checkpoint_specs:
                display(pd.DataFrame([{"checkpoint_label": label, "checkpoint_path": str(path)} for label, path in checkpoint_specs]))
            else:
                print("No checkpoints resolved.")

            print("Command:")
            print(" ".join(shlex.quote(part) for part in eval_cmd))

            if RUN_PAPER_EVAL:
                if not WORK_ROOT.exists():
                    raise FileNotFoundError(f"Workspace does not exist: {WORK_ROOT}")
                subprocess.run(eval_cmd, cwd=REPO_ROOT, check=True)
            else:
                print("Set RUN_PAPER_EVAL = True and rerun this cell to execute the paper protocol.")

            if paper_summary_csv.exists():
                display(pd.read_csv(paper_summary_csv))
            else:
                print("No paper summary CSV found yet:", paper_summary_csv)
            """
        ),
        md(
            """
            ## 4. Paper Protocol Results

            This section reshapes the output into the two views that usually answer the question fastest: per-checkpoint totals and checkpoint-by-split heatmaps.
            """
        ),
        code(
            """
            if not paper_summary_csv.exists():
                print("No paper summary CSV found yet:", paper_summary_csv)
            else:
                paper_eval_df = pd.read_csv(paper_summary_csv)
                display(paper_eval_df)

                ok_rows = paper_eval_df[paper_eval_df["status"] == "ok"].copy()
                if ok_rows.empty:
                    print("Paper summary exists, but no successful rows were recorded yet.")
                else:
                    total_rows = (
                        ok_rows[ok_rows["split"] == "total"]
                        .sort_values("map50", ascending=False)
                        .reset_index(drop=True)
                    )
                    if not total_rows.empty:
                        print("Total-split ranking by mAP@0.5")
                        display(total_rows[["checkpoint_label", "precision", "recall", "map50", "map50_95"]].round(4))

                    weather_rows = ok_rows[ok_rows["split"] != "total"].copy()
                    if not weather_rows.empty:
                        mean_weather = (
                            weather_rows.groupby("checkpoint_label", as_index=False)[["precision", "recall", "map50", "map50_95"]]
                            .mean()
                            .sort_values("map50", ascending=False)
                        )
                        print("Mean across paper weather splits")
                        display(mean_weather.round(4))

                    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
                    if not total_rows.empty:
                        plot_bar(axes[0], total_rows, x="checkpoint_label", y="map50")
                        plot_bar(axes[1], total_rows, x="checkpoint_label", y="map50_95")
                        axes[0].set_title("total split mAP@0.5")
                        axes[1].set_title("total split mAP@0.5:0.95")
                        for ax in axes:
                            ax.tick_params(axis="x", rotation=20)
                            ax.set_xlabel("")
                    else:
                        for ax in axes:
                            ax.set_visible(False)
                    plt.tight_layout()
                    plt.show()

                    pivot50 = ok_rows.pivot_table(index="checkpoint_label", columns="split", values="map50", aggfunc="mean")
                    pivot5095 = ok_rows.pivot_table(index="checkpoint_label", columns="split", values="map50_95", aggfunc="mean")

                    if not pivot50.empty:
                        fig, axes = plt.subplots(1, 2, figsize=(18, max(4, 1.4 * len(pivot50))))
                        plot_heatmap(
                            axes[0],
                            pivot50.fillna(np.nan).values,
                            row_labels=list(pivot50.index),
                            col_labels=list(pivot50.columns),
                            title="mAP@0.5 by checkpoint and split",
                        )
                        plot_heatmap(
                            axes[1],
                            pivot5095.fillna(np.nan).values,
                            row_labels=list(pivot5095.index),
                            col_labels=list(pivot5095.columns),
                            title="mAP@0.5:0.95 by checkpoint and split",
                        )
                        plt.tight_layout()
                        plt.show()
            """
        ),
        md(
            """
            ## 5. DQA vs FedSTO Paper-Summary Snapshot

            If the baseline FedSTO paper summary already exists, line up the total split rows so we can quickly see whether the paper protocol changes the story.
            """
        ),
        code(
            """
            frames = []
            if paper_summary_csv.exists():
                dqa_df = pd.read_csv(paper_summary_csv)
                dqa_df.insert(0, "method", "DQA-CWA")
                frames.append(dqa_df)
            if FEDSTO_PAPER_EVAL_SUMMARY.exists():
                fedsto_df = pd.read_csv(FEDSTO_PAPER_EVAL_SUMMARY)
                fedsto_df.insert(0, "method", "FedSTO")
                frames.append(fedsto_df)

            if not frames:
                print("No paper-protocol summary CSVs found yet.")
            else:
                compare_df = pd.concat(frames, ignore_index=True)
                display(compare_df.round(4))

                ok_total = compare_df[(compare_df["status"] == "ok") & (compare_df["split"] == "total")].copy()
                if not ok_total.empty:
                    best_total = (
                        ok_total.sort_values(["method", "map50"], ascending=[True, False])
                        .groupby("method", as_index=False)
                        .head(1)
                        .reset_index(drop=True)
                    )
                    print("Best total split by method")
                    display(best_total[["method", "checkpoint_label", "precision", "recall", "map50", "map50_95"]].round(4))
            """
        ),
        md(
            """
            ## 6. Visual Artifact Preview

            When plots are enabled, the underlying validation runs save PR curves and confusion matrices. This cell previews the newest run that has both files.
            """
        ),
        code(
            """
            preview_dir = next(
                (
                    path
                    for path in sorted(RUN_ROOT.glob("*"), key=lambda candidate: candidate.stat().st_mtime, reverse=True)
                    if path.is_dir() and (path / "PR_curve.png").exists() and (path / "confusion_matrix.png").exists()
                ),
                None,
            )

            if preview_dir is None:
                print("No completed paper-eval artifact directory with PR/confusion plots was found under:", RUN_ROOT)
            else:
                print("Preview directory:", preview_dir)
                display(NotebookImage(filename=str(preview_dir / "PR_curve.png"), width=700))
                display(NotebookImage(filename=str(preview_dir / "confusion_matrix.png"), width=700))
            """
        ),
        md(
            """
            ## 7. Artifact Index

            This is the short click-list for the generated summary CSV, manifest JSON, logs, and validation run directories.
            """
        ),
        code(
            """
            def artifact_row(path: Path, label: str) -> dict:
                return {
                    "label": label,
                    "path": str(path),
                    "exists": path.exists(),
                    "modified_utc": modified_utc(path),
                }


            artifact_rows = [
                artifact_row(EVAL_SCRIPT, "paper_eval_script"),
                artifact_row(NOTEBOOK_GENERATOR, "notebook_generator"),
                artifact_row(WORK_ROOT, "workspace"),
                artifact_row(manifest_path, "manifest"),
                artifact_row(history_path, "history"),
                artifact_row(summary_csv, "training_summary_csv"),
                artifact_row(paper_summary_csv, "paper_eval_summary_csv"),
                artifact_row(paper_manifest_json, "paper_eval_manifest_json"),
                artifact_row(VALIDATION_ROOT / "paper_protocol_eval_summary.md", "paper_eval_summary_md"),
                artifact_row(LOG_ROOT, "paper_eval_logs_dir"),
                artifact_row(RUN_ROOT, "paper_eval_run_dir"),
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
        notebook_title="03 DQA-CWA Guarded 8h Reproduction",
        notebook_path=ROOT / "03_dqa_cwa_corrected_12h_reproduction.ipynb",
        workspace_name="efficientteacher_dqa_cwa_corrected_12h",
        stats_dir_name="stats_corrected_12h",
        runner_log_name="dqa_cwa_corrected_12h_runner.out",
        pid_file_name="dqa_cwa_corrected_12h_runner.pid",
        warmup_epochs=15,
        phase1_rounds=14,
        phase2_rounds=27,
        batch_size=64,
        workers=0,
        gpus=2,
        master_port=29513,
        min_free_gib=70,
        mode_heading="Guarded 8 Hour Configuration",
        mode_description="This run keeps the corrected FedSTO Algorithm 1 order, preserves FedSTO-style phase 1, and starts DQA-CWA in phase 2 only. DQA now uses a stronger server anchor, lower class-wise blend, objectness/localization-aware pseudo-label quality, BN-local fallback behavior, and a round guard that skips DQA if pseudo-label counts collapse or spike.",
        estimate_note="The completed FedSTO log measured 50 warm-up epochs at 0.982 hours, phase-1 rounds at about 10.46 minutes each, and phase-2 rounds at about 11.17 minutes each. With 15 warm-up epochs, 14 phase-1 rounds, and 27 phase-2 rounds, the clean-run estimate is about 7.8 hours before modest DQA overhead, so this notebook is aimed at roughly an 8-hour turnaround.",
        run_mode="blocking",
        run_default=True,
        eval_default=False,
    )
    build_evaluation_notebook(
        notebook_title="03_2 DQA-CWA Guarded 8h Evaluation",
        notebook_path=ROOT / "03_2_dqa_cwa_corrected_12h_evaluation.ipynb",
        workspace_name="efficientteacher_dqa_cwa_corrected_12h",
        stats_dir_name="stats_corrected_12h",
        notebook_description="This notebook is a read-only analysis pass for the guarded 8-hour DQA-CWA run. It does not launch training by default. Instead it pulls together the finished run artifacts, writes a compact training summary table, renders the plots that are easiest to read, and compares DQA-CWA against the corrected FedSTO baseline when both paper-protocol summaries exist.",
        setting_tables_markdown=CORRECTED_12H_SETTING_TABLES,
    )
    build_notebook(
        notebook_title="04 DQA-CWA v2 Server-Anchored Reproduction",
        notebook_path=ROOT / "04_dqa_ver2_reproduction.ipynb",
        workspace_name="efficientteacher_dqa_ver2",
        stats_dir_name="stats_dqa_ver2",
        runner_log_name="dqa_ver2_runner.out",
        pid_file_name="dqa_ver2_runner.pid",
        runner_script_name="run_dqa_cwa_fedsto_v2.py",
        warmup_epochs=15,
        phase1_rounds=14,
        phase2_rounds=27,
        batch_size=64,
        workers=0,
        gpus=2,
        master_port=29513,
        min_free_gib=70,
        mode_heading="DQA v2 Guarded 8 Hour Configuration",
        mode_description="This run keeps the same schedule and non-implementation settings as 03: corrected FedSTO Algorithm 1 order, FedSTO-style phase 1, DQA starting in phase 2, the same guard thresholds, and the same batch/runtime settings. The only intended change is the DQA v2 aggregation implementation: client updates are applied as quality-weighted residuals on top of the labeled server checkpoint, with a minimum server class-wise anchor to reduce late-round drift.",
        estimate_note="This uses the same warm-up, phase-1, and phase-2 schedule as 03, so the clean-run estimate remains about 7.8 hours before modest DQA overhead on the same hardware.",
        run_mode="blocking",
        run_default=True,
        eval_default=False,
    )
    build_evaluation_notebook(
        notebook_title="04_2 DQA-CWA v2 Evaluation",
        notebook_path=ROOT / "04_2_dqa_ver2_evaluation.ipynb",
        workspace_name="efficientteacher_dqa_ver2",
        stats_dir_name="stats_dqa_ver2",
        notebook_description="This notebook is a read-only analysis pass for the DQA-CWA v2 run. It follows the same evaluation layout as 03_2, but points at the v2 workspace and labels the method as DQA-CWA v2 in comparisons.",
        method_label="DQA-CWA v2",
    )
    build_paper_eval_notebook(
        notebook_title="02_4 DQA-CWA Paper Protocol Evaluation",
        notebook_path=ROOT / "02_4_dqa_cwa_paper_protocol_evaluation.ipynb",
        default_workspace_name="efficientteacher_dqa_cwa_corrected_12h",
        notebook_description="This notebook is the dedicated paper-protocol evaluation pass. It runs the shared per-weather validation script against the selected DQA workspace, writes the summary artifacts under `validation_reports/`, and then reshapes the results so it is easy to see whether the paper-style splits tell a different story from the training-time `server_cloudy_val` view.",
    )


if __name__ == "__main__":
    main()
