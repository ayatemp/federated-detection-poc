#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "07_shared_soft_head_moe_dqa.ipynb"


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
        # 07 Shared Detector + Soft Head-MoE + pseudoGT Routing DQA

        06では counterfactual pseudoGT の信号自体は見えたものの、pseudoGT bucketごとに
        独立detector expertを育てると各expertが弱くなり、output-space MoEでも03を戻せませんでした。

        この07では、考え方を変えます。

        - 03の強い共有検出器をbaseとして残す
        - counterfactual pseudoGT bucketごとにhead/neck専門差分だけを学習する
        - 最終checkpointは `base + beta * sum(router_weight * expert_head_delta)` で作る
        - 出力後のWBFではなく、検出器内部のhead/neck側にsoft MoE信号を入れる

        つまり、真のPyTorch MoE層を追加する前の、壊れにくい本実験版です。
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

        WORKSPACE = PROJECT_ROOT / "output" / "07_shared_soft_head_moe_dqa"
        SOURCE_WORKSPACE = PROJECT_ROOT / "output" / "03_main_bn_residual_dqa_experiment"
        SOURCE_06_WORKSPACE = PROJECT_ROOT / "output" / "06_counterfactual_output_moe_dqa"
        ROUTER_WORKSPACE = PROJECT_ROOT / "output" / "05_expert_choice_pseudogt_router_dqa"
        RUNNER = PROJECT_ROOT / "scripts" / "run_scene_daynight_dqa_07_shared_soft_head_moe.py"

        TEACHER = SOURCE_WORKSPACE / "bn_residual_dqa" / "checkpoints" / "round030_bn_residual_dqa_aggregate.pt"
        ROUTER_TEACHER = ROUTER_WORKSPACE / "checkpoints" / "round030_expert_choice_pseudogt_router_aggregate.pt"

        print("PROJECT_ROOT", PROJECT_ROOT)
        print("WORKSPACE", WORKSPACE)
        print("RUNNER exists:", RUNNER.exists(), RUNNER)
        print("03 teacher exists:", TEACHER.exists(), TEACHER)
        print("05 router reference exists:", ROUTER_TEACHER.exists(), ROUTER_TEACHER)
        print("06 assets exist:", SOURCE_06_WORKSPACE.exists(), SOURCE_06_WORKSPACE)
        """
    ),
    md("## Experiment Design"),
    code(
        """
        design = pd.DataFrame([
            {
                "component": "shared detector",
                "implementation": "03 BN-residual DQA aggregate checkpoint",
                "reason": "今まで一番強い土台を壊さない",
            },
            {
                "component": "pseudoGT routing",
                "implementation": "clean_original / illumination_rescued / cross_view_bridge",
                "reason": "06でnight側のillumination signalが確認できた",
            },
            {
                "component": "head experts",
                "implementation": "routeごとにneck_headのみ1 epoch学習",
                "reason": "pseudoGTでbackbone全体を動かさない",
            },
            {
                "component": "soft MoE",
                "implementation": "base + beta * weighted head_delta",
                "reason": "独立detectorではなく共有trunk上のhead専門性として使う",
            },
            {
                "component": "evaluation",
                "implementation": "03 / 05 / 07 variantsを同じpaper protocolで比較",
                "reason": "final中心評価でDQAが本当に上がったかを見る",
            },
        ])
        display(design)
        """
    ),
    md("## Runtime Estimate"),
    code(
        """
        # Full runは06同様にcounterfactual scan + 3 route expert training + evaluationを行います。
        # 既存06 assetsを使う場合は、scan/trainingを再利用して07 checkpoint合成+評価だけになります。
        CLIENT_LIMIT = 1500
        MAX_IMAGES_PER_CLIENT = 0
        EXPERT_EPOCHS = 1
        EXPERTS = "clean_original,illumination_rescued,cross_view_bridge"

        REUSE_06_ASSETS = False  # Trueにすると06のscan/expert checkpointを再利用して短く検証できます

        estimate = pd.DataFrame([
            {"stage": "counterfactual pseudoGT scan", "full_minutes": 180, "reuse_06_minutes": 0},
            {"stage": "3 route expert trainings", "full_minutes": 135, "reuse_06_minutes": 0},
            {"stage": "soft head-MoE checkpoint composition", "full_minutes": 3, "reuse_06_minutes": 3},
            {"stage": "paper protocol eval for 03/05/07 variants", "full_minutes": 45, "reuse_06_minutes": 45},
        ])
        display(estimate)
        col = "reuse_06_minutes" if REUSE_06_ASSETS else "full_minutes"
        total = float(estimate[col].sum())
        print(f"Estimated runtime: {total / 60:.1f} hours ({total:.0f} minutes)")
        print("Smoke/debugなら MAX_IMAGES_PER_CLIENT = 80 か REUSE_06_ASSETS = True にしてください。")
        """
    ),
    md("## Runner Defaults"),
    code(
        """
        spec = importlib.util.spec_from_file_location("run_scene_daynight_dqa_07", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)

        args = runner.parse_args([])
        display(pd.DataFrame([{
            "max_images_per_client": args.max_images_per_client,
            "experts": args.experts,
            "expert_epochs": args.expert_epochs,
            "expert_train_scope": args.expert_train_scope,
            "moe_scope": args.moe_scope,
            "include_bn": args.include_bn,
            "clean_repeat": args.clean_pseudo_repeat,
            "illumination_repeat": args.illumination_pseudo_repeat,
            "bridge_repeat": args.bridge_pseudo_repeat,
            "clean_loss_box": args.clean_loss_box,
            "illumination_loss_box": args.illumination_loss_box,
            "bridge_loss_box": args.bridge_loss_box,
        }]))
        """
    ),
    md("## Setup Check"),
    code(
        """
        assert RUNNER.exists(), RUNNER
        assert TEACHER.exists(), f"Missing 03 teacher checkpoint: {TEACHER}"
        if not ROUTER_TEACHER.exists():
            print("05 reference checkpoint is missing. The experiment can run, but delta_vs_05 will be limited.")
        if REUSE_06_ASSETS:
            required = [
                SOURCE_06_WORKSPACE / "stats" / "06_counterfactual_scan_summary.json",
                SOURCE_06_WORKSPACE / "checkpoints" / "06_counterfactual_clean_original_expert.pt",
                SOURCE_06_WORKSPACE / "checkpoints" / "06_counterfactual_illumination_rescued_expert.pt",
                SOURCE_06_WORKSPACE / "checkpoints" / "06_counterfactual_cross_view_bridge_expert.pt",
            ]
            missing = [path for path in required if not path.exists()]
            assert not missing, missing

        WORKSPACE.mkdir(parents=True, exist_ok=True)
        print("Ready:", WORKSPACE)
        """
    ),
    md(
        """
        ## Run 07 Full Experiment

        本番実行セルです。Discord通知は開始と終了で送ります。

        出力:

        - `output/07_shared_soft_head_moe_dqa/stats/07_soft_head_moe_variants.csv`
        - `output/07_shared_soft_head_moe_dqa/stats/07_soft_head_moe_metrics.csv`
        - `output/07_shared_soft_head_moe_dqa/07_shared_soft_head_moe_report.md`
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
        EVALUATE_ROUTE_EXPERTS = False  # Trueにすると06 route expert単体も再評価します

        cmd = [
            sys.executable,
            str(RUNNER),
            "--workspace-root", str(WORKSPACE),
            "--source-workspace", str(SOURCE_WORKSPACE),
            "--source-06-workspace", str(SOURCE_06_WORKSPACE),
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
            "--master-port", "33271",
            "--progress-every", "100",
            "--evaluate",
            "--classwise",
            "--no-eval-plots",
            "--notify",
        ]
        if REUSE_06_ASSETS:
            cmd.append("--reuse-06-assets")
        if FORCE_RESTART:
            cmd.append("--force")
        if FORCE_PSEUDO_RESCAN:
            cmd.append("--force-pseudo")
        if EVALUATE_ROUTE_EXPERTS:
            cmd.append("--evaluate-route-experts")

        print(" ".join(cmd))
        if RUN_FULL:
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        """
    ),
    md("## Counterfactual Signal"),
    code(
        """
        scan_candidates = [
            WORKSPACE / "stats" / "07_counterfactual_scan_summary.json",
            SOURCE_06_WORKSPACE / "stats" / "06_counterfactual_scan_summary.json",
        ]
        scan_path = next((path for path in scan_candidates if path.exists()), None)
        if scan_path:
            import json
            scan = json.loads(scan_path.read_text(encoding="utf-8"))
            display(pd.json_normalize({
                "clean_original_boxes": scan.get("totals", {}).get("clean_original_boxes"),
                "illumination_rescued_boxes": scan.get("totals", {}).get("illumination_rescued_boxes"),
                "cross_view_bridge_boxes": scan.get("totals", {}).get("cross_view_bridge_boxes"),
                "rescued_ratio": scan.get("totals", {}).get("rescued_ratio"),
                "day_rescued_ratio": scan.get("day_night_signal", {}).get("day_rescued_ratio"),
                "night_rescued_ratio": scan.get("day_night_signal", {}).get("night_rescued_ratio"),
                "night_minus_day_rescued_ratio": scan.get("day_night_signal", {}).get("night_minus_day_rescued_ratio"),
            }))
        else:
            print("No scan summary yet.")
        """
    ),
    md("## Soft Head-MoE Variants"),
    code(
        """
        variant_csv = WORKSPACE / "stats" / "07_soft_head_moe_variants.csv"
        if variant_csv.exists():
            display(pd.read_csv(variant_csv))
        else:
            print("No variant CSV yet:", variant_csv)
        """
    ),
    md("## Metrics"),
    code(
        """
        metrics_csv = WORKSPACE / "stats" / "07_soft_head_moe_metrics.csv"
        if metrics_csv.exists():
            metrics = pd.read_csv(metrics_csv)
            display(metrics)
            display(metrics.sort_values("map50_95", ascending=False).head(10))
        else:
            print("No metrics yet:", metrics_csv)
        """
    ),
    md("## Final Report"),
    code(
        """
        report = WORKSPACE / "07_shared_soft_head_moe_report.md"
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
