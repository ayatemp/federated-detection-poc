#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "06_counterfactual_output_moe_dqa.ipynb"


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
        # 06 Counterfactual pseudoGT Routing + Output-MoE DQA

        これはメモではなく、本番実験用ノートブックです。

        05 は pseudoGT を virtual expert で選別しましたが、最終モデルは単一 checkpoint に戻るため、
        night 側の hard pseudoGT が削られて mAP が落ちました。09 では、pseudoGT が
        `original / illumination-enhanced / cross-view` のどの観測条件で成立したかを見ると、
        `illumination_rescued` が夜側の不足信号をかなり戻せることがわかりました。

        この 06 では、その発想を本番化します。

        1. counterfactual multi-view で pseudoGT を作る
        2. `clean_original`, `illumination_rescued`, `cross_view_bridge` に分ける
        3. 各 bucket から expert checkpoint を別々に学習する
        4. paper protocol で expert 単体を評価する
        5. 最終出力を image-level router + weighted NMS で MoE fusion する

        residual checkpoint 合成は使いません。専門性を checkpoint と output の両方に残します。
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

        WORKSPACE = PROJECT_ROOT / "output" / "06_counterfactual_output_moe_dqa"
        SOURCE_WORKSPACE = PROJECT_ROOT / "output" / "03_main_bn_residual_dqa_experiment"
        ROUTER_WORKSPACE = PROJECT_ROOT / "output" / "05_expert_choice_pseudogt_router_dqa"
        RUNNER = PROJECT_ROOT / "scripts" / "run_scene_daynight_dqa_06_counterfactual_output_moe.py"

        TEACHER = SOURCE_WORKSPACE / "bn_residual_dqa" / "checkpoints" / "round030_bn_residual_dqa_aggregate.pt"
        ROUTER_TEACHER = ROUTER_WORKSPACE / "checkpoints" / "round030_expert_choice_pseudogt_router_aggregate.pt"

        print("PROJECT_ROOT", PROJECT_ROOT)
        print("WORKSPACE", WORKSPACE)
        print("RUNNER", RUNNER)
        print("TEACHER exists:", TEACHER.exists(), TEACHER)
        print("05 router reference exists:", ROUTER_TEACHER.exists(), ROUTER_TEACHER)
        """
    ),
    md("## Design"),
    code(
        """
        design = pd.DataFrame([
            {
                "stage": "pseudoGT scan",
                "role": "original / bright / CLAHE / hflip viewsからstable boxを作る",
                "output": "view-conditioned pseudoGT buckets",
            },
            {
                "stage": "clean_original expert",
                "role": "通常viewで安定しているpseudoGTを学習する",
                "output": "06_clean_original_expert",
            },
            {
                "stage": "illumination_rescued expert",
                "role": "明るさ補正viewで救われたpseudoGTをbbox弱めで学習する",
                "output": "06_illumination_rescued_expert",
            },
            {
                "stage": "cross_view_bridge expert",
                "role": "original/enhanced間で揺れるboxをbboxさらに弱めで学習する",
                "output": "06_cross_view_bridge_expert",
            },
            {
                "stage": "output MoE",
                "role": "画像の明るさ・contrast・expert出力数で重みを決めてweighted NMS",
                "output": "06_output_moe_* metrics",
            },
        ])
        display(design)
        """
    ),
    md("## Runtime Estimate"),
    code(
        """
        # 本番は全target imageを使います。09 probeは 80 images/client でしたが、
        # 06 fullは client_limit=1500 の全体を使うため、scanが一番重いです。
        CLIENT_LIMIT = 1500
        MAX_IMAGES_PER_CLIENT = 0  # 0 means all images in each client list
        EXPERT_EPOCHS = 1
        EXPERTS = "clean_original,illumination_rescued,cross_view_bridge"

        # 目安。実GPU状況でかなり揺れます。
        SCAN_MINUTES = 180.0
        TRAIN_MINUTES_PER_EXPERT = 45.0
        INDIVIDUAL_EVAL_MINUTES = 35.0
        OUTPUT_MOE_EXPORT_AND_EVAL_MINUTES = 45.0

        estimate = pd.DataFrame([
            {"stage": "counterfactual pseudoGT scan", "estimated_minutes": SCAN_MINUTES},
            {"stage": "3 expert trainings", "estimated_minutes": TRAIN_MINUTES_PER_EXPERT * 3},
            {"stage": "individual paper-protocol evaluation", "estimated_minutes": INDIVIDUAL_EVAL_MINUTES},
            {"stage": "output-MoE prediction export + fusion eval", "estimated_minutes": OUTPUT_MOE_EXPORT_AND_EVAL_MINUTES},
        ])
        display(estimate)
        total = float(estimate["estimated_minutes"].sum())
        print(f"Estimated full runtime: {total / 60:.1f} hours ({total:.0f} minutes)")
        print("For a smoke run, temporarily set MAX_IMAGES_PER_CLIENT = 80.")
        """
    ),
    md("## Runner Defaults"),
    code(
        """
        spec = importlib.util.spec_from_file_location("run_scene_daynight_dqa_06", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)

        args = runner.parse_args([])
        display(pd.DataFrame([
            {
                "max_images_per_client": args.max_images_per_client,
                "experts": args.experts,
                "expert_epochs": args.expert_epochs,
                "expert_train_scope": args.expert_train_scope,
                "clean_repeat": args.clean_pseudo_repeat,
                "illumination_repeat": args.illumination_pseudo_repeat,
                "bridge_repeat": args.bridge_pseudo_repeat,
                "clean_loss_box": args.clean_loss_box,
                "illumination_loss_box": args.illumination_loss_box,
                "bridge_loss_box": args.bridge_loss_box,
                "output_wbf_iou": args.output_wbf_iou,
            }
        ]))
        """
    ),
    md("## Setup Check"),
    code(
        """
        assert RUNNER.exists(), RUNNER
        assert TEACHER.exists(), f"Missing 03 teacher checkpoint: {TEACHER}"
        if not ROUTER_TEACHER.exists():
            print("05 reference checkpoint is missing; 06 can still run, but delta_vs_05 will be blank-ish.")

        WORKSPACE.mkdir(parents=True, exist_ok=True)
        print("Ready:", WORKSPACE)
        """
    ),
    md(
        """
        ## Run 06 Full Experiment

        このセルが本番実行です。Discord通知は開始と終了で飛ばします。

        出力先:

        - `output/06_counterfactual_output_moe_dqa/stats/06_individual_expert_metrics.csv`
        - `output/06_counterfactual_output_moe_dqa/stats/06_output_moe_metrics.csv`
        - `output/06_counterfactual_output_moe_dqa/06_counterfactual_output_moe_report.md`
        """
    ),
    code(
        """
        RUN_FULL = True
        BATCH_SIZE = 128
        WORKERS = 8
        GPUS = 2
        DEVICE = ""
        FORCE_RESTART = False
        FORCE_PSEUDO_RESCAN = False

        cmd = [
            sys.executable,
            str(RUNNER),
            "--workspace-root", str(WORKSPACE),
            "--source-workspace", str(SOURCE_WORKSPACE),
            "--router-workspace", str(ROUTER_WORKSPACE),
            "--teacher-checkpoint", str(TEACHER),
            "--router-teacher-checkpoint", str(ROUTER_TEACHER),
            "--client-limit", str(CLIENT_LIMIT),
            "--max-images-per-client", str(MAX_IMAGES_PER_CLIENT),
            "--experts", EXPERTS,
            "--expert-epochs", str(EXPERT_EPOCHS),
            "--expert-train-scope", "neck_head",
            "--batch-size", str(BATCH_SIZE),
            "--workers", str(WORKERS),
            "--gpus", str(GPUS),
            "--device", DEVICE,
            "--master-port", "33161",
            "--progress-every", "20",
            "--evaluate",
            "--output-moe",
            "--classwise",
            "--no-eval-plots",
            "--notify",
        ]
        if FORCE_RESTART:
            cmd.append("--force")
        if FORCE_PSEUDO_RESCAN:
            cmd.append("--force-pseudo")

        print(" ".join(cmd))
        if RUN_FULL:
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        """
    ),
    md("## PseudoGT Scan Summary"),
    code(
        """
        scan_path = WORKSPACE / "stats" / "06_counterfactual_scan_summary.json"
        if scan_path.exists():
            scan = pd.json_normalize(pd.read_json(scan_path, typ="series").to_dict())
            display(scan.T)
        else:
            print("No scan summary yet:", scan_path)

        client_stats = WORKSPACE / "stats" / "09_view_expert_probe_client_stats.csv"
        if client_stats.exists():
            display(pd.read_csv(client_stats))
        """
    ),
    md("## Individual Expert Metrics"),
    code(
        """
        individual_csv = WORKSPACE / "stats" / "06_individual_expert_metrics.csv"
        if individual_csv.exists():
            individual = pd.read_csv(individual_csv)
            display(individual)
        else:
            print("No individual metrics yet:", individual_csv)
        """
    ),
    md("## Output-MoE Metrics"),
    code(
        """
        output_moe_csv = WORKSPACE / "stats" / "06_output_moe_metrics.csv"
        if output_moe_csv.exists():
            output_moe = pd.read_csv(output_moe_csv)
            display(output_moe)
            display(output_moe[output_moe["split"].eq("summary")])
        else:
            print("No output-MoE metrics yet:", output_moe_csv)
        """
    ),
    md("## Final Report"),
    code(
        """
        report = WORKSPACE / "06_counterfactual_output_moe_report.md"
        if report.exists():
            print(report.read_text(encoding="utf-8"))
        else:
            print("No report yet:", report)
        """
    ),
]


def main() -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
