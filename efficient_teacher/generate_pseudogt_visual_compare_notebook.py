#!/usr/bin/env python3
"""Generate the 02_2 pseudo-GT visual comparison notebook."""

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
            # 02_2 PseudoGT Visual Compare

            GT付き画像に対して、同じ画像を3列で比較します。

            1. 通常のGT
            2. モデルの生推論結果を低閾値NMSで載せたもの
            3. EfficientTeacher設定でpseudoGTとして残るもの

            初期値は30枚です。`NUM_IMAGES`を変えると出力枚数を調整できます。
            """
        ),
        code(
            """
            from __future__ import annotations

            import sys
            from pathlib import Path

            import pandas as pd
            from IPython.display import HTML, display


            def find_repo_root(start: Path | None = None) -> Path:
                start = Path.cwd().resolve() if start is None else Path(start).resolve()
                markers = (
                    "efficient_teacher/pseudogt_visual_compare_02_2.py",
                    "navigating_data_heterogeneity/vendor/efficientteacher/train.py",
                )
                for base in (start, *start.parents):
                    for candidate in (base, base / "Object_Detection"):
                        if all((candidate / marker).exists() for marker in markers):
                            return candidate
                raise FileNotFoundError("Object_Detection repository root was not found.")


            REPO_ROOT = find_repo_root()
            ET_NOTEBOOK_ROOT = REPO_ROOT / "efficient_teacher"
            if str(ET_NOTEBOOK_ROOT) not in sys.path:
                sys.path.insert(0, str(ET_NOTEBOOK_ROOT))

            from pseudogt_visual_compare_02_2 import (  # noqa: E402
                PseudoGTVisualComparer,
                VisualCompareConfig,
                default_image_list,
                warmup_checkpoint,
            )

            pd.options.display.max_columns = 120
            print("repo:", REPO_ROOT)
            """
        ),
        md(
            """
            ## 1. 設定

            `VARIANT`を変えるとpseudoGT側の閾値・cap設定が変わります。GT比較が目的なので、初期値ではserver validation画像を使います。
            """
        ),
        code(
            """
            EXP_ROOT = REPO_ROOT / "efficient_teacher" / "efficientteacher_pseudogt_dqa_probe_2h"

            NUM_IMAGES = 30
            VARIANT = "et_default_lr1e-3"
            CLIENT_ID = 0

            IMAGE_LIST = default_image_list(EXP_ROOT)
            WEIGHTS = warmup_checkpoint()
            OUTPUT_DIR = EXP_ROOT / "results" / "visual_compare" / VARIANT

            START_INDEX = 0
            SAMPLE_MODE = "first"  # "first" or "random"
            RANDOM_SEED = 0

            RAW_CONF_THRES = 0.001
            MAX_RAW_DETECTIONS = 300
            SHOW_IGNORED_PSEUDO = False

            PANEL_WIDTH = 960
            PREVIEW_WIDTH = PANEL_WIDTH * 3 + 16
            DRAW_RAW_LABELS = False
            DRAW_PSEUDO_LABELS = True
            DRAW_GT_LABELS = True

            DEVICE = ""  # "" lets EfficientTeacher choose; use "cpu" if needed
            HALF = False

            cfg = VisualCompareConfig(
                experiment_root=EXP_ROOT,
                variant=VARIANT,
                client_id=CLIENT_ID,
                image_list=IMAGE_LIST,
                weights=WEIGHTS,
                output_dir=OUTPUT_DIR,
                num_images=NUM_IMAGES,
                start_index=START_INDEX,
                sample_mode=SAMPLE_MODE,
                random_seed=RANDOM_SEED,
                raw_conf_thres=RAW_CONF_THRES,
                max_raw_detections=MAX_RAW_DETECTIONS,
                show_ignored_pseudo=SHOW_IGNORED_PSEUDO,
                panel_width=PANEL_WIDTH,
                draw_raw_labels=DRAW_RAW_LABELS,
                draw_pseudo_labels=DRAW_PSEUDO_LABELS,
                draw_gt_labels=DRAW_GT_LABELS,
                device=DEVICE,
                half=HALF,
            )
            comparer = PseudoGTVisualComparer(cfg)
            display(comparer.describe())
            """
        ),
        md(
            """
            ## 2. 画像出力

            実行すると横3列の比較画像が`OUTPUT_DIR`に保存され、summary CSVも同じ場所に保存されます。
            """
        ),
        code(
            """
            summary = comparer.run()
            display(summary)
            print("output_dir:", comparer.output_dir)
            """
        ),
        md(
            """
            ## 3. ノートブック内プレビュー

            出力された比較画像をそのまま表示します。30枚だと縦に長くなります。
            """
        ),
        code(
            """
            def gallery_html(paths, width=PREVIEW_WIDTH):
                rows = []
                for path in paths:
                    rel = Path(path)
                    rows.append(
                        f'<div style="margin: 0 0 22px 0; overflow-x: auto; width: 100%;">'
                        f'<div style="font-family: sans-serif; font-size: 13px; margin-bottom: 4px;">{rel.name}</div>'
                        f'<img src="{rel.as_posix()}" style="width: {width}px; max-width: none; border: 1px solid #ddd;">'
                        f'</div>'
                    )
                return "<div>" + "\\n".join(rows) + "</div>"


            display(HTML(gallery_html(summary["output"].tolist())))
            """
        ),
        md(
            """
            ## 4. 別variantを見る場合

            例として、pseudoGT数を制限した`et_capped_balanced_obj`を見る設定です。必要ならこのセルのコメントを外して実行してください。
            """
        ),
        code(
            """
            # VARIANT = "et_capped_balanced_obj"
            # cfg.variant = VARIANT
            # cfg.output_dir = EXP_ROOT / "results" / "visual_compare" / VARIANT
            # comparer = PseudoGTVisualComparer(cfg)
            # display(comparer.describe())
            # summary = comparer.run()
            # display(summary)
            # display(HTML(gallery_html(summary["output"].tolist())))
            """
        ),
    ]


def main() -> None:
    write_notebook(ROOT / "02_2_pseudogt_visual_compare.ipynb", notebook_cells())


if __name__ == "__main__":
    main()
