"""FedMox paper protocol defaults."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FedMoxPaperConfig:
    """Defaults extracted from arXiv:2508.16568."""

    warmup_epochs: int = 50
    federated_rounds: int = 50
    server_epochs_per_round: int = 1
    client_epochs_per_round: int = 1
    client_sampling_ratio: float = 0.33
    soft_mixture_alpha: float | None = None
    server_resolution: tuple[int, int] = (1280, 720)
    client_resolution: tuple[int, int] = (640, 360)
    optimizer: str = "AdamW"
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    lr_schedule: str = "cosine_annealing_with_5_epoch_warmup"
    unsupervised_weight: float = 4.0
    initial_score_threshold: float = 0.5
    rpn_pseudo_threshold: float = 0.9
    cls_pseudo_threshold: float = 0.9
    reg_pseudo_threshold: float = 0.02
    jitter_times: int = 10
    jitter_scale: float = 0.06
    teacher_proposals: bool = False
    pseudo_box_min_size: int | None = None
    fedprox_mu: float = 0.001
    fedsto_mu: float = 0.001
    train_batch_size: int = 2
    backbone_name: str = "ViT-Adapter-Small"
    backbone_pretraining: str = "DINOv2 with adapter pre-trained on MS-COCO"
    backbone_image_size: int = 518
    backbone_patch_size: int = 14
    backbone_embed_dim: int = 384
    backbone_num_heads: int = 6
    fpn_levels: int = 5
    fpn_out_channels: int = 256
    rpn_positive_iou_threshold: float = 0.7
    rpn_negative_iou_threshold: float = 0.3
    rpn_samples_per_image: int = 256
    rpn_positive_fraction: float = 0.5
    rpn_nms_iou_threshold: float = 0.7
    rpn_pre_nms_topk: int = 2000
    rpn_post_nms_topk: int = 1000
    roi_positive_iou_threshold: float = 0.5
    roi_negative_iou_threshold: float = 0.5
    roi_samples_per_image: int = 512
    roi_positive_fraction: float = 0.25
    roi_add_gt_as_proposals: bool = True
    mask_size: int = 28

    def experts_for_dataset(self, dataset: str) -> int:
        normalized = dataset.lower()
        if normalized == "bdd100k":
            return 4
        if normalized in {"soda10m", "cityscapes"}:
            return 3
        raise ValueError(f"unknown FedMox dataset: {dataset}")
