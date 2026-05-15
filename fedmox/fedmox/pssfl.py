"""Orchestration skeleton for Practical Semi-Supervised Federated Learning."""

from __future__ import annotations

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


def default_task_head_key_filter(key: str) -> bool:
    """Exclude frozen backbone state from federated exchange.

    The paper sends the backbone once before FL and uses the task head weights
    as the global model w. This conservative default filters common MMDetection
    backbone key names while still allowing custom predicates for unusual models.
    """

    return "backbone" not in key.split(".")


def clone_selected_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    key_filter: Callable[[str], bool] = default_task_head_key_filter,
) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in state_dict.items() if key_filter(key)}


def load_selected_state_dict(
    model: torch.nn.Module,
    partial_state: Mapping[str, torch.Tensor],
) -> None:
    current_state = model.state_dict()
    unexpected = set(partial_state) - set(current_state)
    if unexpected:
        raise KeyError(f"partial_state contains unknown model keys: {sorted(unexpected)}")
    current_state.update(partial_state)
    model.load_state_dict(current_state, strict=True)


def freeze_backbone(model: torch.nn.Module) -> int:
    """Freeze parameters whose module path contains a `backbone` component."""

    frozen = 0
    for name, parameter in model.named_parameters():
        if "backbone" in name.split("."):
            parameter.requires_grad_(False)
            frozen += parameter.numel()
    return frozen


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
        federated_key_filter: Callable[[str], bool] = default_task_head_key_filter,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.client_ids = list(client_ids)
        self.config = config or FedMoxPaperConfig()
        self.soft_mixture_alpha = soft_mixture_alpha
        self.federated_key_filter = federated_key_filter
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
            previous_server = clone_selected_state_dict(self.model.state_dict(), self.federated_key_filter)
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
            load_selected_state_dict(self.model, mixed)
            server_train(self.model, self.config.server_epochs_per_round)
        return self.model
