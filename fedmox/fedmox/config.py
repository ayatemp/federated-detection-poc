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
    fedprox_mu: float = 0.001
    fedsto_mu: float = 0.001

    def experts_for_dataset(self, dataset: str) -> int:
        normalized = dataset.lower()
        if normalized == "bdd100k":
            return 4
        if normalized in {"soda10m", "cityscapes"}:
            return 3
        raise ValueError(f"unknown FedMox dataset: {dataset}")
