#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "10_tighter_specialist_latent_dqamox.ipynb"


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
        # 10 Tighter-Specialist Latent DQA-MoX

        このノートブックは、09で見えた「pseudoGTの偏りはかなり抑えられたが、mAP50が0.509付近でplateauする」という問題に対する次ループです。

        重要な設計は次の通りです。

        - warmup と warmup+server repair は08の結果を再利用し、再学習しない
        - detector head は08と同じ `LatentMoEYoloV5`
        - raw pseudoGTをそのまま全部使わず、expert-choice bucketで clean / rare / small / hard-stable をより厳しく選ぶ
        - Phase1は source GT を強め、pseudoGT bbox loss を弱める
        - DQA aggregation は server anchor を強めて、pseudoGTだけの過学習に寄りすぎないようにする
        - クライアント学習だけでなく DQA の classwise stats も selected pseudoGT から作る
        - 評価は `warmup`, `warmup + server repair`, `warmup + tighter-specialist latent DQA-MoX + repair` を同じ scene/day-night protocol で比較する

        つまり、MoEの中身は匿名のまま保ちつつ、pseudoGT側を「量を増やす」ではなく「localization errorを増幅しない学習問題」に整えます。
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

        RUNNER = PROJECT_ROOT / "scripts" / "run_scene_daynight_dqa_10_tighter_specialist_latent_dqamox.py"
        WORKSPACE = PROJECT_ROOT / "output" / "10_tighter_specialist_latent_dqamox"
        SOURCE_WORKSPACE = PROJECT_ROOT / "output" / "08_full_latent_dqamox_from_warmup"

        print("PROJECT_ROOT:", PROJECT_ROOT)
        print("RUNNER exists:", RUNNER.exists(), RUNNER)
        print("WORKSPACE:", WORKSPACE)
        print("SOURCE_WORKSPACE:", SOURCE_WORKSPACE)
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
                "choice": "08のwarmup checkpointを再利用",
                "why": "比較済みのwarmupを使い、改善ループではDQA部分だけを見る",
            },
            {
                "part": "Phase 1",
                "choice": "12 rounds, neck/head/router中心 + tighter selected pseudoGT",
                "why": "09のplateauを踏まえ、まず短い改善ループでsource-dominantな安定性を見る",
            },
            {
                "part": "Phase 2",
                "choice": "1 round, full fine-tune",
                "why": "最後だけ全体をdomainに合わせる。長く回してpseudoGT driftさせない",
            },
            {
                "part": "aggregation",
                "choice": "DQA-CWA v2 server-anchored aggregation over selected pseudoGT stats",
                "why": "集約重みもraw pseudoGTではなく、実際に学習したpseudoGTに合わせる",
            },
            {
                "part": "comparison",
                "choice": "08 warmup / 08 warmup+repair / 10 DQA+repair",
                "why": "baseline再実行なしでDQA改善だけを比較する",
            },
        ])
        display(design)
        """
    ),
    md("## Runtime Estimate"),
    code(
        """
        spec = importlib.util.spec_from_file_location("dqa10_runner", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)

        args = runner.parse_args([])
        estimate = pd.DataFrame([
            {"stage": "materialize/reuse warmup", "count": 1, "minutes_each": 2},
            {"stage": "warmup + server repair baseline", "count": args.repair_baseline_rounds, "minutes_each": args.estimated_repair_round_minutes},
            {"stage": "Phase1 tighter-specialist DQA", "count": args.phase1_rounds, "minutes_each": args.estimated_phase1_round_minutes},
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
            "--source-workspace", str(SOURCE_WORKSPACE),
            "--skip-warmup-training",
            "--warmup-checkpoint", str(SOURCE_WORKSPACE / "checkpoints" / "round000_latent_dqamox_warmup.pt"),
            "--client-limit", "1500",
            "--repair-baseline-rounds", "0",
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

        - `output/10_tighter_specialist_latent_dqamox/stats/10_tighter_specialist_latent_dqamox_final_metrics.csv`
        - `output/10_tighter_specialist_latent_dqamox/stats/10_tighter_specialist_latent_dqamox_split_metrics.csv`
        - `output/10_tighter_specialist_latent_dqamox/10_tighter_specialist_latent_dqamox_report.md`

        まず軽く試したい場合は `MAX_IMAGES_PER_CLIENT = 80`, `PHASE1_ROUNDS = 2` などに落としてください。
        09が有望だったので、この10は最初から短めの改善ループとして回します。良ければ同じ設定を30 roundsへ拡張します。
        """
    ),
    code(
        """
        RUN_FULL = True

        NUM_EXPERTS = 4
        TOP_K = 2

        REPAIR_BASELINE_ROUNDS = 0
        PHASE1_ROUNDS = 12
        PHASE2_ROUNDS = 1

        BATCH_SIZE = 80
        WORKERS = 8
        GPUS = 2
        MAX_IMAGES_PER_CLIENT = 0

        EXPERT_KEEP_FRACTION = 0.55
        EXPERT_MAX_CLASS_FRACTION = 0.24
        LOAD_BIAS_STRENGTH = 0.45

        WARMUP_CHECKPOINT = SOURCE_WORKSPACE / "checkpoints" / "round000_latent_dqamox_warmup.pt"

        cmd = [
            sys.executable,
            str(RUNNER),
            "--workspace-root", str(WORKSPACE),
            "--source-workspace", str(SOURCE_WORKSPACE),
            "--skip-warmup-training",
            "--warmup-checkpoint", str(WARMUP_CHECKPOINT),
            "--num-experts", str(NUM_EXPERTS),
            "--top-k", str(TOP_K),
            "--repair-baseline-rounds", str(REPAIR_BASELINE_ROUNDS),
            "--phase1-rounds", str(PHASE1_ROUNDS),
            "--phase2-rounds", str(PHASE2_ROUNDS),
            "--phase1-client-lr", "0.0006",
            "--phase1-source-repeat", "3",
            "--phase1-pseudo-repeat", "1",
            "--phase1-loss-box", "0.0015",
            "--phase2-loss-box", "0.004",
            "--server-repair-lr", "0.0007",
            "--dqa-server-anchor", "0.55",
            "--dqa-min-server-alpha", "0.50",
            "--dqa-residual-blend", "0.15",
            "--expert-keep-fraction", str(EXPERT_KEEP_FRACTION),
            "--expert-max-class-fraction", str(EXPERT_MAX_CLASS_FRACTION),
            "--load-bias-strength", str(LOAD_BIAS_STRENGTH),
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
        print(" ".join(cmd))
        if RUN_FULL:
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        """
    ),
    md("## Results"),
    code(
        """
        metrics_path = WORKSPACE / "stats" / "10_tighter_specialist_latent_dqamox_final_metrics.csv"
        split_path = WORKSPACE / "stats" / "10_tighter_specialist_latent_dqamox_split_metrics.csv"
        report_path = WORKSPACE / "10_tighter_specialist_latent_dqamox_report.md"

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
