#!/usr/bin/env python3
"""Generate the 02 EfficientTeacher pseudo-GT/DQA probe notebook."""

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


def write_notebook(path: Path, cells: list[dict]) -> None:
    payload = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "al_yolov8",
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


def notebook_cells() -> list[dict]:
    return [
        md(
            """
            # 02 ET PseudoGT DQA Probe 2h

            このノートブックは、DQAに入れる前のEfficientTeacher追加学習が本当に有効な更新になっているかを短時間で確認するための検証です。前回の結果ではpseudoGTが多すぎる・クラスが偏る・bbox/cls lossまで強く入れて教師の誤検出を学習してしまう、という疑いが強かったので、ここではETのloss重みとpseudoGTの作り方を絞って比較します。

            見たいことは3つです。まずpseudoGT密度がGT密度に近づくか。次に各client local checkpointがwarmupより壊れていないか。最後にFedAvgではなくDQA集約でmAPが伸びる条件があるかです。
            """
        ),
        code(
            """
            from __future__ import annotations

            import sys
            from pathlib import Path

            import pandas as pd
            from IPython.display import display


            def find_repo_root(start: Path | None = None) -> Path:
                start = Path.cwd().resolve() if start is None else Path(start).resolve()
                markers = (
                    "efficient_teacher/et_pseudogt_dqa_probe_02.py",
                    "dynamic_quality_aware_classwise_aggregation/exploring/dqa_probe_02_2.py",
                    "navigating_data_heterogeneity/vendor/efficientteacher/train.py",
                )
                for base in (start, *start.parents):
                    candidates = (base, base / "Object_Detection")
                    for candidate in candidates:
                        if all((candidate / marker).exists() for marker in markers):
                            return candidate
                raise FileNotFoundError("Object_Detection repository root was not found.")


            REPO_ROOT = find_repo_root()
            ET_ROOT = REPO_ROOT / "efficient_teacher"
            if str(ET_ROOT) not in sys.path:
                sys.path.insert(0, str(ET_ROOT))

            from et_pseudogt_dqa_probe_02 import (  # noqa: E402
                ETPseudoGTDQAProbe,
                ETPseudoGTProbeConfig,
                reset_probe_outputs,
            )

            pd.options.display.max_columns = 200
            pd.options.display.max_rows = 80
            print("repo:", REPO_ROOT)
            """
        ),
        md(
            """
            ## 1. 実行設定

            `RUN_FULL_PROBE=True`で一括実行します。既存結果を再利用したいときは`FORCE_RERUN=False`のままで大丈夫です。完全に作り直す場合だけ`RESET_OUTPUTS=True`にしてください。
            """
        ),
        code(
            """
            RUN_FULL_PROBE = True
            FORCE_RERUN = False
            RESET_OUTPUTS = False
            INCLUDE_LABELMATCH = False

            cfg = ETPseudoGTProbeConfig(
                source_warmup_key="dqa_v2_scene_12h",
                experiment_name="efficientteacher_pseudogt_dqa_probe_2h",
                server_train_limit=1024,
                server_val_limit=512,
                client_target_limit=512,
                max_wall_hours=2.0,
                batch_size=8,
                val_batch_size=4,
                force_rerun=FORCE_RERUN,
                run_client_local_eval=True,
                run_server_calibration=True,
                top_calibration_k=4,
            )

            probe = ETPseudoGTDQAProbe(cfg, include_labelmatch=INCLUDE_LABELMATCH)
            if RESET_OUTPUTS:
                reset_probe_outputs(probe)
                probe = ETPseudoGTDQAProbe(cfg, include_labelmatch=INCLUDE_LABELMATCH)

            display(probe.describe())
            """
        ),
        md(
            """
            ## 2. データ確認

            server側GTとclient側targetの画像数・物体数を先に確認します。pseudoGT密度はこのserver validation GT密度と比べます。
            """
        ),
        code(
            """
            audit = probe.prepare_data()
            cols = [c for c in ["role", "client_id", "weather", "images", "objects", "objects_per_image", "has_gt"] if c in audit.columns]
            display(audit[cols])
            """
        ),
        md(
            """
            ## 3. 比較するET設定

            `et_default_lr1e-3`を今のET/FedSTO寄りの基準にして、残りはpseudoGTを厳しくする・bbox/clsの影響を弱める・画像あたり/クラスあたりのpseudoGT数を制限する、という方向で比較します。
            """
        ),
        code(
            """
            variant_rows = []
            for item in probe.client_variants:
                ssod = item.get("ssod_overrides", {})
                env = item.get("extra_env", {})
                variant_rows.append(
                    {
                        "variant": item["name"],
                        "epochs": item["epochs"],
                        "lr0": item["lr0"],
                        "train_scope": item["train_scope"],
                        "nms_conf": ssod.get("nms_conf_thres", "default"),
                        "low": ssod.get("ignore_thres_low", "default"),
                        "high": ssod.get("ignore_thres_high", "default"),
                        "teacher_w": ssod.get("teacher_loss_weight", "default"),
                        "box_w": ssod.get("box_loss_weight", "default"),
                        "obj_w": ssod.get("obj_loss_weight", "default"),
                        "cls_w": ssod.get("cls_loss_weight", "default"),
                        "uc_bbox": ssod.get("pseudo_label_with_bbox", "default"),
                        "uc_cls": ssod.get("pseudo_label_with_cls", "default"),
                        "cap_img": env.get("ET_MAX_PSEUDO_PER_IMAGE", ""),
                        "cap_cls_img": env.get("ET_MAX_PSEUDO_PER_CLASS_IMAGE", ""),
                        "note": item.get("note", ""),
                    }
                )
            display(pd.DataFrame(variant_rows))
            """
        ),
        md(
            """
            ## 4. 2時間プローブ実行

            実行内容は、warmup評価、clientごとのET追加学習、pseudoGT集計、client local評価、FedAvg/DQA集約評価、上位候補だけのserver GT 1 epoch補正です。残り時間が少なくなると重い処理は自動でskipされます。
            """
        ),
        code(
            """
            if RUN_FULL_PROBE:
                outputs = probe.run_all()
                print("done")
            else:
                outputs = {}
                print("RUN_FULL_PROBE is False; reading existing results only.")
            """
        ),
        md(
            """
            ## 5. 結果テーブル

            主要CSVを読み出します。`recommendation_table.csv`が最終判断用で、pseudoGT密度、client local、FedAvg、DQA、server補正後の結果をvariant単位でまとめています。
            """
        ),
        code(
            """
            RESULT_ROOT = probe.result_root


            def read_result(filename: str) -> pd.DataFrame:
                path = RESULT_ROOT / filename
                if not path.exists():
                    print(f"missing: {path}")
                    return pd.DataFrame()
                df = pd.read_csv(path)
                print(f"loaded: {filename} ({len(df)} rows)")
                return df


            pseudo_density = read_result("pseudo_density_summary.csv")
            client_eval = read_result("client_local_eval_summary.csv")
            aggregation = read_result("aggregation_summary.csv")
            calibration = read_result("server_calibration_summary.csv")
            leaderboard = read_result("overall_leaderboard.csv")
            recommendation = read_result("recommendation_table.csv")

            if len(pseudo_density):
                density_cols = [
                    "variant",
                    "weather",
                    "pseudo_total",
                    "pseudo_per_image",
                    "reference_gt_per_image",
                    "pseudo_to_gt_density",
                    "top_class_share",
                    "active_classes",
                    "mean_quality_active",
                ]
                display(pseudo_density[[c for c in density_cols if c in pseudo_density.columns]].sort_values(["variant", "weather"]))

            if len(aggregation):
                agg_cols = ["variant", "aggregation", "status", "map50_final", "map50_95_final", "delta_map50_95_vs_warmup"]
                display(aggregation[[c for c in agg_cols if c in aggregation.columns]].sort_values("map50_95_final", ascending=False))

            if len(recommendation):
                display(recommendation)
            """
        ),
        md(
            """
            ## 6. 判定

            DQAに渡す価値がある候補かを機械的に判定します。目安は、DQAがFedAvg以上、pseudoGT密度がGTの2倍以内、最大クラス占有率が高すぎないことです。
            """
        ),
        code(
            """
            baseline = None
            if len(leaderboard) and "name" in leaderboard.columns:
                warmup = leaderboard.loc[leaderboard["name"].eq("warmup_same_mini"), "map50_95_final"].dropna()
                if len(warmup):
                    baseline = float(warmup.iloc[0])

            if len(recommendation):
                decision = recommendation.copy()
                if baseline is not None and "aggregate_map50_95_max" in decision.columns:
                    decision["delta_best_aggregate_vs_warmup"] = decision["aggregate_map50_95_max"] - baseline
                needed = {"delta_dqa_vs_fedavg_map50_95", "pseudo_to_gt_density_mean", "top_class_share_mean"}
                if needed.issubset(decision.columns):
                    decision["dqa_candidate"] = (
                        (decision["delta_dqa_vs_fedavg_map50_95"].fillna(-999) >= -0.0005)
                        & (decision["pseudo_to_gt_density_mean"].fillna(999) <= 2.0)
                        & (decision["top_class_share_mean"].fillna(1.0) <= 0.85)
                    )
                sort_cols = [c for c in ["dqa_candidate", "delta_best_aggregate_vs_warmup", "aggregate_map50_95_max"] if c in decision.columns]
                if sort_cols:
                    display(decision.sort_values(sort_cols, ascending=[False] * len(sort_cols)))
                else:
                    display(decision)
                print("warmup map50_95:", baseline)
            else:
                print("recommendation_table.csv is not available yet.")
            """
        ),
        md(
            """
            ## 7. 出力先

            結果は下記に保存されます。追加考察では、`pseudo_density_summary.csv`でpseudoGTの量と偏りを見て、`aggregation_summary.csv`でFedAvg/DQA差分を見て、最後に`recommendation_table.csv`で設定単位の結論を出します。
            """
        ),
        code(
            """
            paths = {
                "experiment_root": probe.exp_root,
                "results": probe.result_root,
                "pseudo_density": probe.result_root / "pseudo_density_summary.csv",
                "client_local_eval": probe.result_root / "client_local_eval_summary.csv",
                "aggregation": probe.result_root / "aggregation_summary.csv",
                "server_calibration": probe.result_root / "server_calibration_summary.csv",
                "recommendation": probe.result_root / "recommendation_table.csv",
                "leaderboard": probe.result_root / "overall_leaderboard.csv",
            }
            for key, path in paths.items():
                print(f"{key}: {path}")
            """
        ),
    ]


def main() -> None:
    write_notebook(ROOT / "02_et_pseudogt_dqa_probe_2h.ipynb", notebook_cells())


if __name__ == "__main__":
    main()
