"""Orchestration skeleton for Practical Semi-Supervised Federated Learning."""

from __future__ import annotations

import copy
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from .aggregation import fedavg, soft_mixture
from .config import FedMoxPaperConfig


@dataclass(frozen=True)
class ClientUpdate:
    client_id: int
    sample_count: int
    state_dict: Mapping[str, torch.Tensor]


class PSSFLRunner:
    """Paper-order FedMox loop.

    The runner intentionally delegates MMDetection/SoftTeacher training to
    callables so the algorithmic order remains testable and framework-neutral:
    warm-up -> client unsupervised training -> FedAvg -> Soft Mixture ->
    server high-resolution supervised training.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        client_ids: Sequence[int],
        config: FedMoxPaperConfig | None = None,
        soft_mixture_alpha: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.client_ids = list(client_ids)
        self.config = config or FedMoxPaperConfig()
        self.soft_mixture_alpha = soft_mixture_alpha
        self.rng = random.Random(seed)

    def select_clients(self) -> list[int]:
        online = max(1, round(self.config.client_sampling_ratio * len(self.client_ids)))
        return self.rng.sample(self.client_ids, online)

    def run(
        self,
        *,
        warmup_train: Callable[[torch.nn.Module, int], None],
        client_train: Callable[[int, Mapping[str, torch.Tensor], int], ClientUpdate],
        server_train: Callable[[torch.nn.Module, int], None],
        rounds: int | None = None,
    ) -> torch.nn.Module:
        warmup_train(self.model, self.config.warmup_epochs)
        total_rounds = self.config.federated_rounds if rounds is None else rounds

        for _round_idx in range(total_rounds):
            previous_server = copy.deepcopy(self.model.state_dict())
            selected_clients = self.select_clients()
            updates = [
                client_train(client_id, previous_server, self.config.client_epochs_per_round)
                for client_id in selected_clients
            ]
            aggregated = fedavg(
                [update.state_dict for update in updates],
                [update.sample_count for update in updates],
            )
            mixed = soft_mixture(previous_server, aggregated, self.soft_mixture_alpha)
            self.model.load_state_dict(mixed, strict=True)
            server_train(self.model, self.config.server_epochs_per_round)
        return self.model
