"""Federated aggregation and FedMox Soft Mixture."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

StateDict = Mapping[str, torch.Tensor]


def _normalize_weights(sample_counts: Sequence[int | float]) -> list[float]:
    total = float(sum(sample_counts))
    if total <= 0:
        raise ValueError("sample_counts must sum to a positive value")
    return [float(count) / total for count in sample_counts]


def fedavg(client_states: Sequence[StateDict], sample_counts: Sequence[int | float]) -> dict[str, torch.Tensor]:
    """Weighted FedAvg: sum_i n_i / n * w_i."""

    if len(client_states) == 0:
        raise ValueError("client_states must not be empty")
    if len(client_states) != len(sample_counts):
        raise ValueError("client_states and sample_counts must have the same length")

    weights = _normalize_weights(sample_counts)
    keys = set(client_states[0].keys())
    if any(set(state.keys()) != keys for state in client_states):
        raise ValueError("all client state dicts must contain identical keys")

    averaged: dict[str, torch.Tensor] = {}
    for key in client_states[0].keys():
        value = torch.zeros_like(client_states[0][key], dtype=torch.float32)
        for state, weight in zip(client_states, weights, strict=True):
            value = value + state[key].detach().to(dtype=torch.float32) * weight
        averaged[key] = value.to(dtype=client_states[0][key].dtype)
    return averaged


def soft_mixture(
    previous_server_state: StateDict,
    aggregated_client_state: StateDict,
    alpha: float,
) -> dict[str, torch.Tensor]:
    """FedMox Soft Mixture: alpha * w_t + (1 - alpha) * wbar_{t+1}."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if set(previous_server_state.keys()) != set(aggregated_client_state.keys()):
        raise ValueError("state dicts must contain identical keys")

    mixed: dict[str, torch.Tensor] = {}
    for key in previous_server_state.keys():
        server = previous_server_state[key].detach().to(dtype=torch.float32)
        client = aggregated_client_state[key].detach().to(dtype=torch.float32)
        value = alpha * server + (1.0 - alpha) * client
        mixed[key] = value.to(dtype=previous_server_state[key].dtype)
    return mixed
