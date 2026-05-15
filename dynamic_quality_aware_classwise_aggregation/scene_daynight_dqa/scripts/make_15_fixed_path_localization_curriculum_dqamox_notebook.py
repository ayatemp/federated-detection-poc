#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "15_fixed_path_localization_curriculum_dqamox.ipynb"


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
        # 15 Fixed-Path Localization-Curriculum Full-From-Warmup DQA-MoX

        13の途中確認で、expert-choice pseudo dataset の画像リストが symlink の `.resolve()` によって元画像パスへ戻り、
        YOLO が生成済み pseudo label ではなく元データ側 label を探していることがわかりました。
        このノートブックではその path 問題を直した状態で、warmup から新規に学習します。

        重要な設計は次の通りです。

        - warmupを `LatentMoEYoloV5` で最初から学習する
        - 同じwarmupから `warmup + server repair` baseline も同じworkspace内で学習する
        - DQA branch は Phase1長め、Phase2短めの FedSTO/FedMoX 風スケジュールにする
        - raw pseudoGTをそのまま全部使わず、expert-choice bucketで clean / rare / small / hard-stable を選ぶ
        - 前半は source GT を強め、pseudoGT bbox loss を弱めて崩壊を避ける
        - selected pseudoGT は実選択数ベースの class cap を最初から使う
        - pseudo train list は pseudo dataset 側の image path を保持し、生成済み pseudo label を確実に読む
        - Phase1の pseudoGT bbox loss は 0.01 にして、正しく読めるpseudo labelの局所化信号を弱く使う
        - Phase2も bbox loss を 0.003 に抑え、最後だけ慎重に全体調整する
        - クライアント学習だけでなく DQA の classwise stats も selected pseudoGT から作る
        - 評価は `warmup`, `warmup + server repair`, `warmup + FedMoX-style DQA-MoX + repair` を同じ scene/day-night protocol で比較する

        目標は `scene_daynight_total` の final mAP50 を 0.60 以上にすることです。
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

        RUNNER = PROJECT_ROOT / "scripts" / "run_scene_daynight_dqa_15_fixed_path_localization_curriculum_dqamox.py"
        WORKSPACE = PROJECT_ROOT / "output" / "15_fixed_path_localization_curriculum_dqamox"

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
                "choice": "50 epochs, LatentMoEYoloV5, from pretrained YOLO",
                "why": "MoEを後付けにせず、FedMoX風に最初からexpert/routerを含めて立ち上げる",
            },
            {
                "part": "Phase 1",
                "choice": "12 rounds, box-safe pseudo classification/objectness",
                "why": "pseudo bboxを学習対象から外し、domain/class exposureだけを取り込む",
            },
            {
                "part": "Phase 2",
                "choice": "2 rounds, full fine-tune",
                "why": "最後だけ全体をdomainに合わせる。長く回してpseudoGT driftさせない",
            },
            {
                "part": "aggregation",
                "choice": "DQA-CWA v2 with stable server anchor",
                "why": "12でanchorを弱めると改善しなかったため、server側の局所化能力を残す",
            },
            {
                "part": "comparison",
                "choice": "same-run warmup / same-run warmup+repair / same-run DQA+repair",
                "why": "warmupから一貫して学習し、baseline再利用による違和感を消す",
            },
            {
                "part": "target",
                "choice": "final total mAP50 >= 0.60",
                "why": "ユーザー指定の到達条件。未達なら原因を見て次のfull-from-warmup候補へ進む",
            },
        ])
        display(design)
        """
    ),
    md("## Runtime Estimate"),
    code(
        """
        spec = importlib.util.spec_from_file_location("dqa15_runner", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)

        args = runner.parse_args([])
        estimate = pd.DataFrame([
            {"stage": "warmup from pretrained", "count": 1, "minutes_each": args.estimated_warmup_minutes},
            {"stage": "warmup + server repair baseline", "count": args.repair_baseline_rounds, "minutes_each": args.estimated_repair_round_minutes},
            {"stage": "Phase1 FedMoX-style DQA-MoE", "count": args.phase1_rounds, "minutes_each": args.estimated_phase1_round_minutes},
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
            "--repair-baseline-rounds", "1",
            "--phase1-rounds", "1",
            "--phase2-rounds", "0",
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

        - `output/15_fixed_path_localization_curriculum_dqamox/stats/15_fixed_path_localization_curriculum_dqamox_final_metrics.csv`
        - `output/15_fixed_path_localization_curriculum_dqamox/stats/15_fixed_path_localization_curriculum_dqamox_split_metrics.csv`
        - `output/15_fixed_path_localization_curriculum_dqamox/15_fixed_path_localization_curriculum_dqamox_report.md`

        これは本実験セルです。warmupから学習するので、途中からの再利用はしません。
        GPUが一時的に落ちた場合は同じworkspaceで再実行すれば、既に完了したcheckpointを再利用して続きます。
        """
    ),
    code(
        """
        RUN_FULL = True

        NUM_EXPERTS = 4
        TOP_K = 2

        REPAIR_BASELINE_ROUNDS = 0
        PHASE1_ROUNDS = 8
        PHASE2_ROUNDS = 2
        TARGET_MAP50 = 0.60

        BATCH_SIZE = 80
        WORKERS = 8
        GPUS = 2
        MAX_IMAGES_PER_CLIENT = 0

        EXPERT_KEEP_FRACTION = 0.45
        EXPERT_MAX_CLASS_FRACTION = 0.18
        ACTUAL_MAX_CLASS_FRACTION = 0.25
        LOAD_BIAS_STRENGTH = 0.45
        CURRICULUM_START_ROUND = 999

        cmd = [
            sys.executable,
            str(RUNNER),
            "--workspace-root", str(WORKSPACE),
            "--num-experts", str(NUM_EXPERTS),
            "--top-k", str(TOP_K),
            "--repair-baseline-rounds", str(REPAIR_BASELINE_ROUNDS),
            "--phase1-rounds", str(PHASE1_ROUNDS),
            "--phase2-rounds", str(PHASE2_ROUNDS),
            "--target-map50", str(TARGET_MAP50),
            "--phase1-client-lr", "0.0006",
            "--phase1-source-repeat", "3",
            "--phase1-pseudo-repeat", "1",
            "--phase1-loss-box", "0.01",
            "--curriculum-start-round", str(CURRICULUM_START_ROUND),
            "--late-phase1-client-lr", "0.0005",
            "--late-phase1-source-repeat", "2",
            "--late-phase1-pseudo-repeat", "2",
            "--late-phase1-loss-box", "0.0005",
            "--phase2-loss-box", "0.003",
            "--server-repair-lr", "0.0007",
            "--router-temperature", "1.3",
            "--router-balance-weight", "0.03",
            "--router-entropy-weight", "0.002",
            "--dqa-server-anchor", "0.65",
            "--dqa-min-server-alpha", "0.60",
            "--dqa-residual-blend", "0.10",
            "--late-dqa-server-anchor", "0.35",
            "--late-dqa-min-server-alpha", "0.35",
            "--late-dqa-residual-blend", "0.08",
            "--expert-keep-fraction", str(EXPERT_KEEP_FRACTION),
            "--expert-max-class-fraction", str(EXPERT_MAX_CLASS_FRACTION),
            "--actual-max-class-fraction", str(ACTUAL_MAX_CLASS_FRACTION),
            "--late-expert-keep-fraction", "0.60",
            "--late-expert-max-class-fraction", "0.22",
            "--late-actual-max-class-fraction", "0.28",
            "--late-min-score", "0.24",
            "--late-min-stability", "0.68",
            "--min-score", "0.35",
            "--min-stability", "0.78",
            "--max-boxes-per-image", "8",
            "--load-bias-strength", str(LOAD_BIAS_STRENGTH),
            "--batch-size", str(BATCH_SIZE),
            "--workers", str(WORKERS),
            "--gpus", str(GPUS),
            "--max-images-per-client", str(MAX_IMAGES_PER_CLIENT),
            "--master-port", "36601",
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
        metrics_path = WORKSPACE / "stats" / "15_fixed_path_localization_curriculum_dqamox_final_metrics.csv"
        split_path = WORKSPACE / "stats" / "15_fixed_path_localization_curriculum_dqamox_split_metrics.csv"
        report_path = WORKSPACE / "15_fixed_path_localization_curriculum_dqamox_report.md"

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
