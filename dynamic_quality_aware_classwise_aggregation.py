#!/usr/bin/env python3
"""Dynamic Quality-Aware Class-wise Aggregation for FedSTO-style SSFOD.

This module implements a conservative version of the research idea:

* regular FedAvg for backbone, neck, objectness, and box regression rows;
* dynamic class-wise weighted aggregation for classification-head rows only;
* reliability from pseudo-label quantity and quality, smoothed across rounds;
* optional server-head anchoring so labeled server updates are not erased by
  noisy client pseudo labels.

It is intentionally standalone so it can be dropped into the current FedSTO
runner after the baseline reproduction finishes.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


EPS = 1e-12
YOLOV5_HEAD_RE = re.compile(r"(^|\.)head\.m\.\d+\.(weight|bias)$")
CLASS_ONLY_HEAD_RE = re.compile(
    r"(^|\.)head\.(residual_m|cls_preds|cv3|classification)\.\d+.*\.(weight|bias)$"
)


@dataclass
class AggregationConfig:
    num_classes: int
    count_ema: float = 0.70
    quality_ema: float = 0.70
    alpha_ema: float = 0.50
    temperature: float = 1.50
    uniform_mix: float = 0.05
    classwise_blend: float = 0.75
    stability_lambda: float = 0.25
    min_effective_count: float = 1.0
    min_quality: float = 0.05
    max_quality: float = 1.0
    server_anchor: float = 0.50

    def validate(self) -> None:
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        for name in ("count_ema", "quality_ema", "alpha_ema", "uniform_mix", "classwise_blend"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.server_anchor < 0:
            raise ValueError("server_anchor must be non-negative")


@dataclass
class ClientClassStats:
    client_id: str
    counts: list[float]
    mean_confidences: list[float]

    @classmethod
    def from_mapping(cls, item: Mapping[str, Any], num_classes: int, default_id: str) -> "ClientClassStats":
        client_id = str(item.get("client_id", item.get("id", item.get("name", default_id))))
        counts = _class_vector(item, num_classes, ("counts", "count", "n", "pseudo_counts", "num_pseudo"))
        mean_confidences = _confidence_vector(item, num_classes, counts)
        return cls(client_id=client_id, counts=counts, mean_confidences=mean_confidences)


def _class_vector(item: Mapping[str, Any], num_classes: int, keys: Sequence[str]) -> list[float]:
    for key in keys:
        if key not in item:
            continue
        value = item[key]
        if isinstance(value, Mapping):
            vector = [float(value.get(str(i), value.get(i, 0.0))) for i in range(num_classes)]
        else:
            vector = [float(x) for x in value]
        if len(vector) != num_classes:
            raise ValueError(f"{key} length must be {num_classes}, got {len(vector)}")
        return vector
    raise ValueError(f"missing one of {keys}")


def _confidence_vector(item: Mapping[str, Any], num_classes: int, counts: Sequence[float]) -> list[float]:
    for key in ("mean_confidences", "mean_confidence", "confidences", "confidence", "quality", "scores"):
        if key in item:
            return _class_vector(item, num_classes, (key,))

    for sum_key in ("confidence_sums", "confidence_sum", "score_sums", "score_sum"):
        if sum_key in item:
            sums = _class_vector(item, num_classes, (sum_key,))
            return [float(total) / max(float(count), EPS) for total, count in zip(sums, counts)]

    raise ValueError("missing mean confidence or confidence_sum fields")


def load_round_stats(path: Path, num_classes: int) -> list[ClientClassStats]:
    """Load client class statistics.

    Supported JSON shapes:

    {"clients": [{"id": "client0", "counts": [...], "mean_confidences": [...]}]}
    [{"client_id": "client0", "counts": [...], "confidence_sums": [...]}]
    {"client0": {"counts": [...], "mean_confidences": [...]}, ...}
    """

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping) and "clients" in raw:
        items = raw["clients"]
    elif isinstance(raw, list):
        items = raw
    elif isinstance(raw, Mapping):
        items = []
        for key, value in raw.items():
            if not isinstance(value, Mapping):
                continue
            merged = dict(value)
            merged.setdefault("client_id", key)
            items.append(merged)
    else:
        raise ValueError(f"unsupported stats JSON root: {type(raw).__name__}")

    stats = [
        ClientClassStats.from_mapping(item, num_classes, default_id=f"client{idx}")
        for idx, item in enumerate(items)
    ]
    if not stats:
        raise ValueError(f"no client stats found in {path}")
    return stats


def _empty_client_state(num_classes: int) -> dict[str, list[float]]:
    return {
        "count_ema": [0.0] * num_classes,
        "quality_ema": [0.0] * num_classes,
        "prev_quality_ema": [0.0] * num_classes,
    }


def load_state(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"clients": {}, "alpha": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    state.setdefault("clients", {})
    state.setdefault("alpha", {})
    return state


def save_state(path: Path | None, state: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def compute_reliability(
    stats: Sequence[ClientClassStats],
    state: dict[str, Any],
    config: AggregationConfig,
) -> tuple[dict[str, Any], torch.Tensor, list[str], torch.Tensor]:
    """Return updated state, alpha[source, class], source ids, and active class mask."""

    config.validate()
    client_ids = [item.client_id for item in stats]
    reliability = torch.zeros((len(stats), config.num_classes), dtype=torch.float64)
    effective_counts = torch.zeros(config.num_classes, dtype=torch.float64)

    client_state = state.setdefault("clients", {})
    for client_idx, item in enumerate(stats):
        previous = client_state.setdefault(item.client_id, _empty_client_state(config.num_classes))
        if len(previous["count_ema"]) != config.num_classes:
            previous = _empty_client_state(config.num_classes)
            client_state[item.client_id] = previous

        next_count = []
        next_quality = []
        next_prev_quality = []
        for class_idx in range(config.num_classes):
            raw_count = max(float(item.counts[class_idx]), 0.0)
            raw_quality = min(max(float(item.mean_confidences[class_idx]), 0.0), 1.0)

            old_count = float(previous["count_ema"][class_idx])
            old_quality = float(previous["quality_ema"][class_idx])
            count_ema = config.count_ema * old_count + (1.0 - config.count_ema) * raw_count
            quality_ema = config.quality_ema * old_quality + (1.0 - config.quality_ema) * raw_quality

            count_term = math.log1p(max(count_ema, 0.0))
            if count_term > 0:
                quality_term = min(max(quality_ema, config.min_quality), config.max_quality)
                drift = abs(quality_ema - old_quality)
                stability = math.exp(-config.stability_lambda * drift)
                reliability[client_idx, class_idx] = count_term * quality_term * stability

            effective_counts[class_idx] += count_ema
            next_count.append(count_ema)
            next_quality.append(quality_ema)
            next_prev_quality.append(old_quality)

        previous["count_ema"] = next_count
        previous["quality_ema"] = next_quality
        previous["prev_quality_ema"] = next_prev_quality

    active = effective_counts >= config.min_effective_count
    source_ids = [f"client:{client_id}" for client_id in client_ids]

    if config.server_anchor > 0:
        server_reliability = torch.zeros((1, config.num_classes), dtype=torch.float64)
        for class_idx in range(config.num_classes):
            positives = reliability[:, class_idx][reliability[:, class_idx] > 0]
            if positives.numel() > 0:
                server_reliability[0, class_idx] = positives.mean() * config.server_anchor
        reliability = torch.cat([reliability, server_reliability], dim=0)
        source_ids.append("server")

    alpha_raw = torch.zeros_like(reliability)
    for class_idx in range(config.num_classes):
        column = reliability[:, class_idx].clone()
        if not active[class_idx] or float(column.sum()) <= EPS:
            alpha_raw[:, class_idx] = 1.0 / reliability.shape[0]
            continue
        powered = torch.pow(torch.clamp(column, min=EPS), 1.0 / config.temperature)
        alpha_raw[:, class_idx] = powered / powered.sum()
        if config.uniform_mix > 0:
            alpha_raw[:, class_idx] = (
                (1.0 - config.uniform_mix) * alpha_raw[:, class_idx]
                + config.uniform_mix / reliability.shape[0]
            )

    alpha_key = "|".join(source_ids)
    previous_alpha = state.setdefault("alpha", {}).get(alpha_key)
    if previous_alpha is not None:
        prev = torch.tensor(previous_alpha, dtype=torch.float64)
        if prev.shape == alpha_raw.shape:
            alpha = config.alpha_ema * prev + (1.0 - config.alpha_ema) * alpha_raw
            alpha = alpha / torch.clamp(alpha.sum(dim=0, keepdim=True), min=EPS)
        else:
            alpha = alpha_raw
    else:
        alpha = alpha_raw

    state["alpha"] = {alpha_key: alpha.tolist()}
    state["config"] = asdict(config)
    return state, alpha.float(), source_ids, active


def _prepare_efficientteacher_imports(repo_root: Path) -> None:
    candidate = repo_root / "navigating_data_heterogeneity" / "vendor" / "efficientteacher"
    if candidate.exists():
        sys.path.insert(0, str(candidate.resolve()))


def _load_checkpoint(path: Path, repo_root: Path) -> dict[str, Any]:
    _prepare_efficientteacher_imports(repo_root)
    return torch.load(path, map_location="cpu", weights_only=False)


def _model_state_dict(checkpoint: Mapping[str, Any], key: str = "model") -> dict[str, torch.Tensor]:
    model = checkpoint[key]
    if hasattr(model, "float") and hasattr(model, "state_dict"):
        return model.float().state_dict()
    if isinstance(model, Mapping):
        return {name: value.float() if torch.is_tensor(value) else value for name, value in model.items()}
    raise TypeError(f"checkpoint[{key!r}] is neither a module nor a state dict")


def _replace_model_state(checkpoint: dict[str, Any], state_dict: Mapping[str, torch.Tensor], key: str = "model") -> None:
    model = checkpoint[key]
    if hasattr(model, "float") and hasattr(model, "load_state_dict"):
        model = model.float()
        model.load_state_dict(state_dict, strict=False)
        checkpoint[key] = model.half()
    elif isinstance(model, Mapping):
        checkpoint[key] = dict(state_dict)
    else:
        raise TypeError(f"checkpoint[{key!r}] is neither a module nor a state dict")


def _fedavg_state_dicts(state_dicts: Sequence[Mapping[str, torch.Tensor]], base: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    averaged: dict[str, torch.Tensor] = {}
    for key, value in base.items():
        if torch.is_tensor(value) and value.dtype.is_floating_point:
            averaged[key] = torch.stack([state[key].float() for state in state_dicts], dim=0).mean(dim=0).to(value.dtype)
        else:
            averaged[key] = value
    return averaged


def _yolov5_class_rows(out_channels: int, num_classes: int) -> list[list[int]] | None:
    no = num_classes + 5
    if out_channels % no != 0:
        return None
    anchors = out_channels // no
    return [[anchor * no + 5 + class_idx for anchor in range(anchors)] for class_idx in range(num_classes)]


def _class_only_rows(out_channels: int, num_classes: int) -> list[list[int]] | None:
    if out_channels % num_classes != 0:
        return None
    anchors = out_channels // num_classes
    return [[anchor * num_classes + class_idx for anchor in range(anchors)] for class_idx in range(num_classes)]


def _classification_rows(key: str, value: torch.Tensor, num_classes: int) -> list[list[int]] | None:
    lowered = key.lower()
    if value.ndim not in (1, 4):
        return None
    out_channels = int(value.shape[0])

    if YOLOV5_HEAD_RE.search(lowered):
        return _yolov5_class_rows(out_channels, num_classes)
    if CLASS_ONLY_HEAD_RE.search(lowered):
        return _class_only_rows(out_channels, num_classes)
    return None


def apply_dynamic_classwise_head(
    averaged: dict[str, torch.Tensor],
    source_state_dicts: Sequence[Mapping[str, torch.Tensor]],
    alpha: torch.Tensor,
    active: torch.Tensor,
    config: AggregationConfig,
) -> dict[str, torch.Tensor]:
    """Blend dynamic class-wise head aggregation into a FedAvg state dict."""

    result = dict(averaged)
    for key, fedavg_value in averaged.items():
        if not torch.is_tensor(fedavg_value) or not fedavg_value.dtype.is_floating_point:
            continue
        rows_by_class = _classification_rows(key, fedavg_value, config.num_classes)
        if rows_by_class is None:
            continue

        updated = fedavg_value.float().clone()
        source_values = [state[key].float() for state in source_state_dicts]
        for class_idx, rows in enumerate(rows_by_class):
            if not bool(active[class_idx]):
                continue
            weights = alpha[:, class_idx].to(updated.dtype)
            for row in rows:
                stacked = torch.stack([value[row] for value in source_values], dim=0)
                dynamic_row = torch.sum(stacked * weights.view(-1, *([1] * (stacked.ndim - 1))), dim=0)
                updated[row] = (
                    (1.0 - config.classwise_blend) * updated[row]
                    + config.classwise_blend * dynamic_row
                )
        result[key] = updated.to(fedavg_value.dtype)
    return result


def aggregate_checkpoints(
    client_checkpoints: Sequence[Path],
    server_checkpoint: Path,
    output_checkpoint: Path,
    stats: Sequence[ClientClassStats],
    state_path: Path | None,
    config: AggregationConfig,
    repo_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Aggregate checkpoints with DQA-CWA and save the next global checkpoint."""

    if len(client_checkpoints) != len(stats):
        raise ValueError(
            f"client checkpoint count ({len(client_checkpoints)}) must match stats count ({len(stats)})"
        )

    state = load_state(state_path)
    state, alpha, source_ids, active = compute_reliability(stats, state, config)

    client_ckpts = [_load_checkpoint(path, repo_root) for path in client_checkpoints]
    server_ckpt = _load_checkpoint(server_checkpoint, repo_root)
    base = copy.deepcopy(server_ckpt)

    source_ckpts = client_ckpts + [server_ckpt]
    source_state_dicts = [_model_state_dict(ckpt, "model") for ckpt in source_ckpts]
    dynamic_state_dicts = source_state_dicts if len(source_ids) == len(source_state_dicts) else source_state_dicts[: len(source_ids)]
    base_state = _model_state_dict(base, "model")

    averaged = _fedavg_state_dicts(source_state_dicts, base_state)
    dynamic = apply_dynamic_classwise_head(averaged, dynamic_state_dicts, alpha, active, config)
    _replace_model_state(base, dynamic, "model")

    if base.get("ema") is not None:
        ema_source_dicts = [_model_state_dict(ckpt, "ema") for ckpt in source_ckpts if ckpt.get("ema") is not None]
        if len(ema_source_dicts) == len(source_ckpts):
            ema_dynamic_dicts = ema_source_dicts if len(source_ids) == len(ema_source_dicts) else ema_source_dicts[: len(source_ids)]
            ema_base = _model_state_dict(base, "ema")
            ema_avg = _fedavg_state_dicts(ema_source_dicts, ema_base)
            ema_dynamic = apply_dynamic_classwise_head(ema_avg, ema_dynamic_dicts, alpha, active, config)
            _replace_model_state(base, ema_dynamic, "ema")

    base["epoch"] = -1
    base["optimizer"] = None
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, output_checkpoint)

    state["last_sources"] = source_ids
    state["last_alpha"] = alpha.tolist()
    state["last_active_classes"] = [bool(x) for x in active.tolist()]
    save_state(state_path, state)
    return output_checkpoint, state


def _self_test() -> None:
    num_classes = 2
    cfg = AggregationConfig(num_classes=num_classes, server_anchor=0.0, classwise_blend=1.0)
    stats = [
        ClientClassStats("a", [100, 1], [0.9, 0.4]),
        ClientClassStats("b", [1, 100], [0.4, 0.9]),
    ]
    state, alpha, _, active = compute_reliability(stats, {"clients": {}, "alpha": {}}, cfg)
    assert alpha[0, 0] > alpha[1, 0], alpha
    assert alpha[1, 1] > alpha[0, 1], alpha
    assert active.tolist() == [True, True]

    out_channels = 3 * (5 + num_classes)
    base = {"head.m.0.weight": torch.zeros(out_channels, 1, 1, 1), "head.m.0.bias": torch.zeros(out_channels)}
    c0 = {k: v.clone() for k, v in base.items()}
    c1 = {k: v.clone() for k, v in base.items()}
    server = {k: v.clone() for k, v in base.items()}
    rows = _yolov5_class_rows(out_channels, num_classes)
    assert rows is not None
    for row in rows[0]:
        c0["head.m.0.bias"][row] = 10
        c1["head.m.0.bias"][row] = 0
        server["head.m.0.bias"][row] = 5
    for row in rows[1]:
        c0["head.m.0.bias"][row] = 0
        c1["head.m.0.bias"][row] = 20
        server["head.m.0.bias"][row] = 5

    fedavg = _fedavg_state_dicts([c0, c1, server], server)
    dynamic = apply_dynamic_classwise_head(fedavg, [c0, c1], alpha, active, cfg)
    assert dynamic["head.m.0.bias"][rows[0][0]] > fedavg["head.m.0.bias"][rows[0][0]]
    assert dynamic["head.m.0.bias"][rows[1][0]] > fedavg["head.m.0.bias"][rows[1][0]]
    assert dynamic["head.m.0.bias"][0] == fedavg["head.m.0.bias"][0]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        stats_path = tmp_path / "stats.json"
        stats_path.write_text(json.dumps({"clients": [asdict(x) for x in stats]}), encoding="utf-8")
        assert len(load_round_stats(stats_path, num_classes)) == 2
    _ = state


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    aggregate = subparsers.add_parser("aggregate", help="aggregate FedSTO checkpoints with DQA-CWA")
    aggregate.add_argument("--client-checkpoints", nargs="+", type=Path, required=True)
    aggregate.add_argument("--server-checkpoint", type=Path, required=True)
    aggregate.add_argument("--stats", type=Path, required=True)
    aggregate.add_argument("--state", type=Path, required=True)
    aggregate.add_argument("--out", type=Path, required=True)
    aggregate.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    aggregate.add_argument("--num-classes", type=int, default=10)
    aggregate.add_argument("--classwise-blend", type=float, default=0.75)
    aggregate.add_argument("--temperature", type=float, default=1.50)
    aggregate.add_argument("--uniform-mix", type=float, default=0.05)
    aggregate.add_argument("--server-anchor", type=float, default=0.50)
    aggregate.add_argument("--stability-lambda", type=float, default=0.25)

    subparsers.add_parser("self-test", help="run synthetic aggregation checks")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "self-test":
        _self_test()
        print("DQA-CWA self-test passed.")
        return 0

    config = AggregationConfig(
        num_classes=args.num_classes,
        classwise_blend=args.classwise_blend,
        temperature=args.temperature,
        uniform_mix=args.uniform_mix,
        server_anchor=args.server_anchor,
        stability_lambda=args.stability_lambda,
    )
    stats = load_round_stats(args.stats, config.num_classes)
    out, state = aggregate_checkpoints(
        client_checkpoints=args.client_checkpoints,
        server_checkpoint=args.server_checkpoint,
        output_checkpoint=args.out,
        stats=stats,
        state_path=args.state,
        config=config,
        repo_root=args.repo_root,
    )
    active = sum(bool(x) for x in state["last_active_classes"])
    print(json.dumps({"output": str(out), "active_classes": active, "sources": state["last_sources"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
