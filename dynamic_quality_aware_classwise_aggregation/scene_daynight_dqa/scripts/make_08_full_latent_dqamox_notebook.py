#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "08_full_latent_dqamox_from_warmup.ipynb"


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
        # 08 Full Latent DQA-MoX from Warmup

        このノートブックは、途中checkpointを前提にした後付けMoEではなく、**warmupから最後まで同じ検出器を育てる**ための本実験版です。

        重要な設計は次の通りです。

        - expert には `night` や `rare` のような手書きの意味を割り当てない
        - 共有 backbone/neck と共有 YOLO head を残し、head 内に匿名 latent expert を持たせる
        - router は feature map の各場所ごとに expert を選ぶ。学習中は全expertへsoftに流し、推論時だけtop-k化する
        - DQA は pseudoGT stats と classwise server-anchored aggregation に残す
        - 評価は `warmup`, `warmup + server repair`, `warmup + latent DQA-MoX + server repair` を同じ scene/day-night protocol で比較する

        つまり、FedMox 的な MoE の良さを入れつつ、DQA の「クライアントごとの得意領域を集約に使う」思想を、固定 expert 名ではなく学習される router/expert に移します。
        """
    ),
    code(
        """
        from pathlib import Path
        import importlib.util
        import subprocess
        import sys

        import pandas as pd

        cwd = Path.cwd().resolve()
        if cwd.name == "notebooks":
            PROJECT_ROOT = cwd.parent
        elif (cwd / "dynamic_quality_aware_classwise_aggregation").exists():
            PROJECT_ROOT = cwd / "dynamic_quality_aware_classwise_aggregation" / "scene_daynight_dqa"
        else:
            PROJECT_ROOT = cwd

        RUNNER = PROJECT_ROOT / "scripts" / "run_scene_daynight_dqa_08_full_latent_dqamox.py"
        WORKSPACE = PROJECT_ROOT / "output" / "08_full_latent_dqamox_from_warmup"

        print("PROJECT_ROOT:", PROJECT_ROOT)
        print("RUNNER exists:", RUNNER.exists(), RUNNER)
        print("WORKSPACE:", WORKSPACE)
        """
    ),
    md("## Design"),
    code(
        """
        design = pd.DataFrame([
            {
                "part": "architecture",
                "choice": "LatentMoEYoloV5",
                "why": "expertを手書き意味やclient idに固定せず、featureからrouterが選ぶ。学習中はdense soft routingでexpert collapseを避ける",
            },
            {
                "part": "warmup",
                "choice": "server GTでMoE head込みのdetectorを最初から学習",
                "why": "途中でhead構造を変えず、最後まで一貫したモデルにする",
            },
            {
                "part": "Phase 1",
                "choice": "30 rounds, neck/head/router中心",
                "why": "pseudoGTでbackboneを壊さず、expert/routerを十分に分化させる",
            },
            {
                "part": "Phase 2",
                "choice": "2 rounds, full fine-tune",
                "why": "最後だけ全体をdomainに合わせる。長く回してpseudoGT driftさせない",
            },
            {
                "part": "aggregation",
                "choice": "DQA-CWA v2 server-anchored residual aggregation",
                "why": "server repairの土台を残しつつ、classwise/clientwise pseudoGT qualityを使う",
            },
            {
                "part": "comparison",
                "choice": "warmup / warmup+repair / latent DQA-MoX+repair",
                "why": "DQAがserver repairだけを超えたかをfinal中心で見る",
            },
        ])
        display(design)
        """
    ),
    md("## Runtime Estimate"),
    code(
        """
        spec = importlib.util.spec_from_file_location("dqa08_runner", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)

        args = runner.parse_args([])
        estimate = pd.DataFrame([
            {"stage": "warmup latent MoE detector", "count": 1, "minutes_each": args.estimated_warmup_minutes},
            {"stage": "warmup + server repair baseline", "count": args.repair_baseline_rounds, "minutes_each": args.estimated_repair_round_minutes},
            {"stage": "Phase1 neck/head/router DQA", "count": args.phase1_rounds, "minutes_each": args.estimated_phase1_round_minutes},
            {"stage": "Phase2 full DQA", "count": args.phase2_rounds, "minutes_each": args.estimated_phase2_round_minutes},
            {"stage": "final scene/day-night eval", "count": 1, "minutes_each": args.estimated_eval_minutes},
        ])
        estimate["total_minutes"] = estimate["count"] * estimate["minutes_each"]
        display(estimate)
        total_minutes = estimate["total_minutes"].sum()
        print(f"Estimated total: {total_minutes / 60:.1f} hours ({total_minutes:.0f} minutes)")
        """
    ),
    md("## Setup Check"),
    code(
        """
        # データリスト、config schema、runner import の確認だけを行います。
        cmd = [
            sys.executable,
            str(RUNNER),
            "--setup-only",
            "--dry-run",
            "--workspace-root", str(WORKSPACE),
            "--client-limit", "1500",
            "--no-progress",
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        """
    ),
    md(
        """
        ## Full Run

        本番実行セルです。開始・終了時にDiscord通知を送ります。

        出力:

        - `output/08_full_latent_dqamox_from_warmup/stats/08_full_latent_dqamox_final_metrics.csv`
        - `output/08_full_latent_dqamox_from_warmup/stats/08_full_latent_dqamox_split_metrics.csv`
        - `output/08_full_latent_dqamox_from_warmup/08_full_latent_dqamox_report.md`

        まず軽く試したい場合は `MAX_IMAGES_PER_CLIENT = 80`, `PHASE1_ROUNDS = 2`, `REPAIR_BASELINE_ROUNDS = 2` などに落としてください。
        """
    ),
    code(
        """
        RUN_FULL = True

        NUM_EXPERTS = 4
        TOP_K = 2

        WARMUP_EPOCHS = 50
        REPAIR_BASELINE_ROUNDS = 30
        PHASE1_ROUNDS = 30
        PHASE2_ROUNDS = 2

        BATCH_SIZE = 80
        WORKERS = 8
        GPUS = 2
        MAX_IMAGES_PER_CLIENT = 0

        # 途中から実験だけ軽く見る場合は True にして既存warmup checkpointを渡せます。
        SKIP_WARMUP_TRAINING = False
        WARMUP_CHECKPOINT = ""  # e.g. /app/Object_Detection/pseudogt_learnability/checkpoints/round000_warmup.pt

        cmd = [
            sys.executable,
            str(RUNNER),
            "--workspace-root", str(WORKSPACE),
            "--num-experts", str(NUM_EXPERTS),
            "--top-k", str(TOP_K),
            "--warmup-epochs", str(WARMUP_EPOCHS),
            "--repair-baseline-rounds", str(REPAIR_BASELINE_ROUNDS),
            "--phase1-rounds", str(PHASE1_ROUNDS),
            "--phase2-rounds", str(PHASE2_ROUNDS),
            "--batch-size", str(BATCH_SIZE),
            "--workers", str(WORKERS),
            "--gpus", str(GPUS),
            "--max-images-per-client", str(MAX_IMAGES_PER_CLIENT),
            "--master-port", "33481",
            "--evaluate",
            "--classwise",
            "--no-eval-plots",
            "--notify",
            "--notify-progress",
            "--notify-first-progress-hours", "10",
            "--notify-progress-interval-hours", "2",
        ]
        if SKIP_WARMUP_TRAINING:
            cmd.append("--skip-warmup-training")
            if WARMUP_CHECKPOINT:
                cmd.extend(["--warmup-checkpoint", WARMUP_CHECKPOINT])

        print(" ".join(cmd))
        if RUN_FULL:
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        """
    ),
    md("## Results"),
    code(
        """
        metrics_path = WORKSPACE / "stats" / "08_full_latent_dqamox_final_metrics.csv"
        split_path = WORKSPACE / "stats" / "08_full_latent_dqamox_split_metrics.csv"
        report_path = WORKSPACE / "08_full_latent_dqamox_report.md"

        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            display(metrics)
        else:
            print("metrics not found yet:", metrics_path)

        if split_path.exists():
            split = pd.read_csv(split_path)
            display(split)
        else:
            print("split metrics not found yet:", split_path)

        if report_path.exists():
            print(report_path.read_text(encoding="utf-8")[:4000])
        """
    ),
]


NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(
    json.dumps(
        {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)
print(NOTEBOOK_PATH)
