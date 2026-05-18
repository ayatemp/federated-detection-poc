"""Check that local FedMox defaults match the paper trace."""

import tomllib
from pathlib import Path

from fedmox import FedMoxPaperConfig


def main() -> None:
    cfg = FedMoxPaperConfig()
    assert cfg.warmup_epochs == 50
    assert cfg.federated_rounds == 50
    assert cfg.server_epochs_per_round == 1
    assert cfg.client_epochs_per_round == 1
    assert cfg.server_resolution == (1280, 720)
    assert cfg.client_resolution == (640, 360)
    assert cfg.soft_mixture_alpha is None
    assert cfg.optimizer == "AdamW"
    assert cfg.learning_rate == 1e-4
    assert cfg.weight_decay == 0.05
    assert cfg.unsupervised_weight == 4.0
    assert cfg.initial_score_threshold == 0.5
    assert cfg.rpn_pseudo_threshold == 0.9
    assert cfg.cls_pseudo_threshold == 0.9
    assert cfg.reg_pseudo_threshold == 0.02
    assert cfg.jitter_times == 10
    assert cfg.jitter_scale == 0.06
    assert cfg.teacher_proposals is False
    assert cfg.pseudo_box_min_size is None
    assert cfg.train_batch_size == 2
    assert cfg.backbone_name == "ViT-Adapter-Small"
    assert cfg.backbone_image_size == 518
    assert cfg.backbone_patch_size == 14
    assert cfg.backbone_embed_dim == 384
    assert cfg.backbone_num_heads == 6
    assert cfg.fpn_levels == 5
    assert cfg.fpn_out_channels == 256
    assert cfg.rpn_positive_iou_threshold == 0.7
    assert cfg.rpn_negative_iou_threshold == 0.3
    assert cfg.rpn_samples_per_image == 256
    assert cfg.rpn_positive_fraction == 0.5
    assert cfg.rpn_nms_iou_threshold == 0.7
    assert cfg.rpn_pre_nms_topk == 2000
    assert cfg.rpn_post_nms_topk == 1000
    assert cfg.roi_positive_iou_threshold == 0.5
    assert cfg.roi_negative_iou_threshold == 0.5
    assert cfg.roi_samples_per_image == 512
    assert cfg.roi_positive_fraction == 0.25
    assert cfg.roi_add_gt_as_proposals is True
    assert cfg.mask_size == 28
    assert cfg.experts_for_dataset("BDD100K") == 4
    assert cfg.experts_for_dataset("SODA10M") == 3
    assert cfg.experts_for_dataset("Cityscapes") == 3

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    bdd = tomllib.loads((config_dir / "paper_bdd100k.toml").read_text())
    soda = tomllib.loads((config_dir / "paper_soda10m.toml").read_text())
    city = tomllib.loads((config_dir / "paper_cityscapes.toml").read_text())
    results = tomllib.loads((config_dir / "paper_results.toml").read_text())
    assert bdd["server_domain"] == "Cloudy"
    assert bdd["client_unlabeled_images_by_domain"] == {"Overcast": 5000, "Rainy": 5000, "Snowy": 8000}
    assert bdd["test_images_by_domain"] == {"Cloudy": 300, "Overcast": 600, "Rainy": 300, "Snowy": 300}
    assert soda["server_domain"] == "Clear"
    assert soda["client_unlabeled_images_by_domain"] == {"Overcast": 10000, "Rainy": 8000}
    assert soda["test_images_by_domain"] == {"Clear": 1000, "Overcast": 1000, "Rainy": 300}
    assert city["server_labeled_images"] == 2000
    assert city["client_unlabeled_images"] == 18000
    assert city["test_images"] == 500
    assert results["bdd100k"]["n3"]["ours"]["total"] == 0.486
    assert results["bdd100k"]["n9"]["ours"]["total"] == 0.475
    assert results["soda10m"]["n3"]["ours"]["total"] == 0.267
    assert results["soda10m"]["n9"]["ours"]["total"] == 0.261
    assert results["cityscapes"]["n3"]["ours"]["total"] == 0.464
    assert results["cityscapes"]["n9"]["ours"]["total"] == 0.455
    assert results["scaling"]["bdd100k_r67_n9"]["ours"]["total"] == 0.478
    assert results["scaling"]["soda10m_n100"]["ours"]["total"] == 0.258
    assert results["ablation"]["bdd100k_n9_moe_sm"]["total"] == 0.475
    print("FedMox paper-trace checks passed.")


if __name__ == "__main__":
    main()
