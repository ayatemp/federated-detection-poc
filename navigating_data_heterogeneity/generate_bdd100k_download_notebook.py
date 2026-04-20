import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path(
    "/Users/kakuayato/my-company/masters_research/navigating_data_heterogeneity/00_download_and_inspect_bdd100k.ipynb"
)


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
        # 00. Download and Inspect BDD100K for FedSTO

        この notebook は KaggleHub から BDD100K をダウンロードし、FedSTO notebook
        `01_fedsto_ssfod.ipynb` に使えるデータかを確認するためのものです。

        使う Kaggle dataset:

        ```python
        kagglehub.dataset_download("awsaf49/bdd100k-dataset")
        ```

        この notebook で確認すること:
        - dataset の実体パス
        - 画像ディレクトリがあるか
        - detection label JSON があるか
        - label JSON に `attributes.weather` と `labels[*].box2d` があるか
        - 論文寄せの weather split に必要な `cloudy / overcast / rainy / snowy` が揃うか
        - FedSTO notebook の次段階で使える状態か

        注意:
        - BDD100K は巨大なので、初回ダウンロードには時間と容量が必要です。
        - KaggleHub は通常 Kaggle の cache 配下に保存します。この notebook では、そのパスを `raw/bdd100k/kagglehub_path.txt` に記録します。
        - 実際に `data/server` と `data/clients` へ変換する処理は次の notebook / script で行う想定です。
        """
    ),
    md(
        """
        ## 0. Setup

        `kagglehub` が未インストールなら、次のセルの `%pip install` を有効化してください。  
        KaggleHub は Kaggle API token が不要な場合もありますが、環境によっては Kaggle 認証が必要になることがあります。
        """
    ),
    code(
        """
        # 必要な場合だけコメントアウトを外してください。
        # %pip install -q kagglehub pandas pyyaml pillow tqdm

        from __future__ import annotations

        import json
        import os
        import random
        import shutil
        from collections import Counter, defaultdict
        from dataclasses import dataclass
        from pathlib import Path
        from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

        import pandas as pd
        from PIL import Image

        try:
            import kagglehub
            print("kagglehub: OK")
        except ImportError as exc:
            raise ImportError(
                "kagglehub が未インストールです。上の `%pip install -q kagglehub ...` を実行して kernel を再起動してください。"
            ) from exc
        """
    ),
    code(
        """
        @dataclass
        class BDDDownloadConfig:
            dataset_handle: str = "awsaf49/bdd100k-dataset"
            project_root: Path = Path("/Users/kakuayato/my-company/masters_research/navigating_data_heterogeneity")
            raw_root: Path = project_root / "raw" / "bdd100k"
            report_root: Path = project_root / "reports" / "bdd100k_inspection"
            # BDD100K det_v2 uses "partly cloudy" rather than "cloudy".
            # The paper describes the server domain as cloudy and uses about 5k labeled images;
            # "partly cloudy" is the closest Kaggle det_v2 counterpart and has a similar count.
            expected_server_weather: str = "partly cloudy"
            expected_client_weathers: Tuple[str, ...] = ("overcast", "rainy", "snowy")
            # BDD100K source label is "pedestrian"; convert it to model class name "person" later.
            expected_classes: Tuple[str, ...] = ("pedestrian", "car", "bus", "truck", "traffic sign")
            model_classes: Tuple[str, ...] = ("person", "car", "bus", "truck", "traffic sign")
            max_preview_files: int = 80
            max_json_records_for_deep_check: Optional[int] = None


        cfg = BDDDownloadConfig()
        cfg.raw_root.mkdir(parents=True, exist_ok=True)
        cfg.report_root.mkdir(parents=True, exist_ok=True)

        print(cfg)
        """
    ),
    md(
        """
        ## 1. Download Dataset

        実行すると KaggleHub の cache に dataset が保存されます。  
        返ってきた path を `raw/bdd100k/kagglehub_path.txt` に保存します。
        """
    ),
    code(
        """
        path = kagglehub.dataset_download(cfg.dataset_handle)
        dataset_path = Path(path)

        print("Path to dataset files:", dataset_path)
        print("Exists:", dataset_path.exists())

        (cfg.raw_root / "kagglehub_path.txt").write_text(str(dataset_path), encoding="utf-8")
        print("Recorded path:", cfg.raw_root / "kagglehub_path.txt")

        # 便利な参照用 symlink を作ります。失敗しても本体には影響しません。
        link_path = cfg.raw_root / "downloaded"
        try:
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            link_path.symlink_to(dataset_path, target_is_directory=True)
            print("Symlink:", link_path, "->", dataset_path)
        except OSError as exc:
            print("Symlink skipped:", exc)
        """
    ),
    md(
        """
        ## 2. File Structure Inspection

        まずはファイル拡張子、主要 JSON、画像ディレクトリ候補を確認します。
        """
    ),
    code(
        """
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        LABEL_NAME_PRIORITY = [
            "det_train.json",
            "det_val.json",
            "bdd100k_labels_images_train.json",
            "bdd100k_labels_images_val.json",
        ]


        def iter_files(root: Path) -> Iterable[Path]:
            for p in root.rglob("*"):
                if p.is_file():
                    yield p


        def human_size(num_bytes: int) -> str:
            units = ["B", "KB", "MB", "GB", "TB"]
            value = float(num_bytes)
            for unit in units:
                if value < 1024:
                    return f"{value:.1f}{unit}"
                value /= 1024
            return f"{value:.1f}PB"


        def summarize_files(root: Path, max_preview: int = 80) -> pd.DataFrame:
            rows = []
            ext_counts = Counter()
            total_size = 0
            total_files = 0
            for p in iter_files(root):
                total_files += 1
                size = p.stat().st_size
                total_size += size
                ext_counts[p.suffix.lower()] += 1
                if len(rows) < max_preview:
                    rows.append({
                        "relative_path": str(p.relative_to(root)),
                        "suffix": p.suffix.lower(),
                        "size": human_size(size),
                    })
            print("total_files:", total_files)
            print("total_size:", human_size(total_size))
            print("top extensions:", ext_counts.most_common(20))
            return pd.DataFrame(rows)


        def find_label_jsons(root: Path) -> List[Path]:
            jsons = [p for p in iter_files(root) if p.suffix.lower() == ".json"]
            def score(path: Path) -> Tuple[int, str]:
                name = path.name
                priority = LABEL_NAME_PRIORITY.index(name) if name in LABEL_NAME_PRIORITY else len(LABEL_NAME_PRIORITY)
                det_bonus = 0 if "det" in str(path).lower() or "label" in str(path).lower() else 1
                return (priority, det_bonus, str(path))
            return sorted(jsons, key=score)


        def find_image_roots(root: Path, min_images: int = 10) -> pd.DataFrame:
            counts = []
            for d in [p for p in root.rglob("*") if p.is_dir()]:
                direct_images = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
                if len(direct_images) >= min_images:
                    counts.append({
                        "dir": str(d),
                        "num_direct_images": len(direct_images),
                        "basename": d.name,
                    })
            return pd.DataFrame(counts).sort_values("num_direct_images", ascending=False).reset_index(drop=True)


        preview_df = summarize_files(dataset_path, max_preview=cfg.max_preview_files)
        display(preview_df)

        label_jsons = find_label_jsons(dataset_path)
        print("\\nLabel JSON candidates:")
        for p in label_jsons[:20]:
            print("-", p)

        image_roots_df = find_image_roots(dataset_path)
        print("\\nImage directory candidates:")
        display(image_roots_df.head(30))
        """
    ),
    md(
        """
        ## 3. Label JSON Content Check

        BDD100K detection label なら、各 record にだいたい次が含まれます。

        ```json
        {
          "name": "...jpg",
          "attributes": {"weather": "cloudy"},
          "labels": [
            {"category": "car", "box2d": {"x1": ..., "y1": ..., "x2": ..., "y2": ...}}
          ]
        }
        ```
        """
    ),
    code(
        """
        def load_json_records(path: Path) -> List[dict]:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                for key in ["frames", "data", "annotations", "items"]:
                    if key in payload and isinstance(payload[key], list):
                        return payload[key]
            raise ValueError(f"Unsupported JSON structure: {path}")


        def choose_train_val_jsons(label_jsons: Sequence[Path]) -> Dict[str, Optional[Path]]:
            result = {"train": None, "val": None}
            for p in label_jsons:
                lower = str(p).lower()
                if result["train"] is None and ("train" in lower and ("det" in lower or "label" in lower)):
                    result["train"] = p
                if result["val"] is None and (("val" in lower or "valid" in lower) and ("det" in lower or "label" in lower)):
                    result["val"] = p
            if result["train"] is None and label_jsons:
                result["train"] = label_jsons[0]
            return result


        def record_has_box2d(record: dict) -> bool:
            for label in record.get("labels", []) or []:
                if isinstance(label, dict) and "box2d" in label:
                    return True
            return False


        def inspect_records(records: List[dict], expected_classes: Sequence[str], limit: Optional[int] = None) -> Dict[str, Any]:
            if limit is not None:
                records = records[:limit]

            weather_counter = Counter()
            category_counter = Counter()
            box2d_counter = Counter()
            missing_weather = 0
            records_with_box2d = 0

            for rec in records:
                attrs = rec.get("attributes", {}) or {}
                weather = attrs.get("weather")
                if weather is None:
                    missing_weather += 1
                else:
                    weather_counter[str(weather)] += 1

                has_box = False
                for label in rec.get("labels", []) or []:
                    if not isinstance(label, dict):
                        continue
                    cat = label.get("category")
                    if cat:
                        category_counter[str(cat)] += 1
                    if "box2d" in label:
                        has_box = True
                        box2d_counter[str(cat)] += 1
                if has_box:
                    records_with_box2d += 1

            expected_class_counts = {c: box2d_counter.get(c, 0) for c in expected_classes}
            return {
                "num_records_checked": len(records),
                "missing_weather": missing_weather,
                "records_with_box2d": records_with_box2d,
                "weather_counter": weather_counter,
                "category_counter": category_counter,
                "box2d_counter": box2d_counter,
                "expected_class_counts": expected_class_counts,
            }


        json_choice = choose_train_val_jsons(label_jsons)
        print("Chosen JSONs:", json_choice)

        if json_choice["train"] is None:
            raise FileNotFoundError("No label JSON found. This Kaggle dataset may not include detection labels.")

        train_records = load_json_records(json_choice["train"])
        print("train json:", json_choice["train"])
        print("num train records:", len(train_records))
        print("first record keys:", train_records[0].keys() if train_records else None)
        print("first record sample:")
        print(json.dumps(train_records[0], ensure_ascii=False, indent=2)[:2000] if train_records else "EMPTY")

        train_summary = inspect_records(
            train_records,
            expected_classes=cfg.expected_classes,
            limit=cfg.max_json_records_for_deep_check,
        )

        print("\\nWeather counts:")
        print(train_summary["weather_counter"].most_common())

        print("\\nExpected class box2d counts:")
        print(train_summary["expected_class_counts"])

        print("\\nTop categories with box2d:")
        print(train_summary["box2d_counter"].most_common(20))
        """
    ),
    md(
        """
        ## 4. Image Link Check

        label JSON の `name` と実際の画像ファイルが対応しているかを少数サンプルで確認します。
        """
    ),
    code(
        """
        def build_image_name_index(image_roots: pd.DataFrame, max_roots: int = 10) -> Dict[str, Path]:
            index = {}
            for _, row in image_roots.head(max_roots).iterrows():
                root = Path(row["dir"])
                for img in root.iterdir():
                    if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                        index.setdefault(img.name, img)
            return index


        def check_image_links(records: List[dict], image_index: Dict[str, Path], sample_size: int = 100) -> pd.DataFrame:
            sample = records[:sample_size]
            rows = []
            for rec in sample:
                name = rec.get("name") or rec.get("filename") or rec.get("image")
                rows.append({
                    "name": name,
                    "exists": name in image_index if name else False,
                    "path": str(image_index.get(name, "")) if name else "",
                    "weather": (rec.get("attributes", {}) or {}).get("weather"),
                    "has_box2d": record_has_box2d(rec),
                })
            return pd.DataFrame(rows)


        image_index = build_image_name_index(image_roots_df, max_roots=20)
        print("indexed image names:", len(image_index))

        link_check_df = check_image_links(train_records, image_index, sample_size=100)
        display(link_check_df.head(20))
        print("sample image exists ratio:", link_check_df["exists"].mean() if len(link_check_df) else 0)
        """
    ),
    md(
        """
        ## 5. FedSTO Readiness Check

        `01_fedsto_ssfod.ipynb` に進めるための条件を判定します。
        """
    ),
    code(
        """
        required_weathers = [cfg.expected_server_weather, *cfg.expected_client_weathers]
        weather_counts = train_summary["weather_counter"]
        class_counts = train_summary["expected_class_counts"]

        readiness_rows = [
            {
                "check": "label_json_found",
                "ok": json_choice["train"] is not None,
                "detail": str(json_choice["train"]),
            },
            {
                "check": "images_found",
                "ok": len(image_index) > 0,
                "detail": f"{len(image_index)} image names indexed",
            },
            {
                "check": "json_has_weather",
                "ok": len(weather_counts) > 0 and train_summary["missing_weather"] < train_summary["num_records_checked"],
                "detail": dict(weather_counts),
            },
            {
                "check": "json_has_box2d",
                "ok": train_summary["records_with_box2d"] > 0,
                "detail": f"{train_summary['records_with_box2d']} records with box2d",
            },
            {
                "check": "required_weathers_present",
                "ok": all(weather_counts.get(w, 0) > 0 for w in required_weathers),
                "detail": {w: weather_counts.get(w, 0) for w in required_weathers},
            },
            {
                "check": "expected_classes_present",
                "ok": all(class_counts.get(c, 0) > 0 for c in cfg.expected_classes),
                "detail": class_counts,
            },
            {
                "check": "sample_images_match_labels",
                "ok": bool(len(link_check_df) and link_check_df["exists"].mean() > 0.8),
                "detail": f"{link_check_df['exists'].mean():.1%} matched in first {len(link_check_df)} records" if len(link_check_df) else "no sample",
            },
        ]

        readiness_df = pd.DataFrame(readiness_rows)
        display(readiness_df)

        report_path = cfg.report_root / "readiness_report.csv"
        readiness_df.to_csv(report_path, index=False)
        print("Saved:", report_path)

        if readiness_df["ok"].all():
            print("\\n✅ This Kaggle BDD100K dataset looks usable for the FedSTO preparation step.")
        else:
            print("\\n⚠️ Some checks failed. Inspect the table above before moving to FedSTO conversion.")
        """
    ),
    md(
        """
        ## 6. Next Expected Layout

        次の変換ステップでは、この raw dataset から以下を作ります。

        ```text
        masters_research/navigating_data_heterogeneity/data/
        ├── server/
        │   ├── images/train/
        │   ├── images/val/
        │   ├── labels/train/
        │   ├── labels/val/
        │   └── data.yaml
        └── clients/
            ├── client_0/images_unlabeled/   # overcast
            ├── client_1/images_unlabeled/   # rainy
            └── client_2/images_unlabeled/   # snowy
        ```

        論文寄せの対応:
        - server: `cloudy` の labeled images
        - client_0: `overcast` の unlabeled images
        - client_1: `rainy` の unlabeled images
        - client_2: `snowy` の unlabeled images

        この notebook で readiness が OK になったら、次に `prepare_bdd100k_for_fedsto.py` を作って変換します。
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


NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote notebook: {NOTEBOOK_PATH}")
