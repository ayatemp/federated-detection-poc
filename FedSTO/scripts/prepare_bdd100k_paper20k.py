from __future__ import annotations

import json
from pathlib import Path

import prepare_bdd100k_for_fedsto as prep


PAPER_DATA_ROOT = prep.PROJECT_ROOT / "data_paper20k"


def main() -> None:
    # Paper setup: one labeled cloudy server and three unlabeled non-IID clients
    # from overcast, rainy, and snowy BDD100K weather conditions. The BDD100K
    # Kaggle mirror uses "partly cloudy" for the cloudy split.
    prep.DATA_ROOT = PAPER_DATA_ROOT
    cfg = prep.PrepConfig(
        server_weather="partly cloudy",
        client_weathers=("overcast", "rainy", "snowy"),
        symlink_images=True,
        overwrite=True,
        max_server_train=5000,
        max_server_val=None,
        max_client_images=5000,
    )

    root = prep.dataset_root()
    summary = prep.prepare_dataset(root, cfg)

    server_train = next(item for item in summary["server"] if item["split"] == "train")
    server_val = next(item for item in summary["server"] if item["split"] == "val")
    unlabeled_total = sum(item["num_images"] for item in summary["clients"])
    summary["paper_reproduction"] = {
        "target_total_train_images": 20000,
        "actual_labeled_server_train": server_train["num_images"],
        "actual_labeled_server_val": server_val["num_images"],
        "actual_unlabeled_client_train": unlabeled_total,
        "actual_total_train_images": server_train["num_images"] + unlabeled_total,
        "note": (
            "The paper describes roughly 5K labeled cloudy images and 15K unlabeled "
            "client images. This local BDD100K mirror contains 4,881 train images "
            "with weather='partly cloudy' and YOLO-mapped labels, so all are used."
        ),
    }

    out = PAPER_DATA_ROOT / "dataset_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Paper reproduction dataset ready: {PAPER_DATA_ROOT}")


if __name__ == "__main__":
    main()
