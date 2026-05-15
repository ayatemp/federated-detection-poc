"""Check that local FedMox defaults match the paper trace."""

from fedmox import FedMoxPaperConfig


def main() -> None:
    cfg = FedMoxPaperConfig()
    assert cfg.warmup_epochs == 50
    assert cfg.federated_rounds == 50
    assert cfg.server_epochs_per_round == 1
    assert cfg.client_epochs_per_round == 1
    assert cfg.server_resolution == (1280, 720)
    assert cfg.client_resolution == (640, 360)
    assert cfg.unsupervised_weight == 4.0
    assert cfg.experts_for_dataset("BDD100K") == 4
    assert cfg.experts_for_dataset("SODA10M") == 3
    assert cfg.experts_for_dataset("Cityscapes") == 3
    print("FedMox paper-trace checks passed.")


if __name__ == "__main__":
    main()
