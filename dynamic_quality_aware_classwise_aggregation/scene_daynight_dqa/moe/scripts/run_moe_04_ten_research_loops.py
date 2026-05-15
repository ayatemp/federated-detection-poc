#!/usr/bin/env python3
"""Run ten follow-up MoE x DQA research loops.

The previous five-loop sprint found the strongest signal so far:
phase1 day-client expert checkpoints outperformed the single DQA aggregate.
This runner therefore focuses on expert selection and expert parameter mixing.

The ten loops are checkpoint-level probes:

1. Confirm day expert strength.
2. Average day expert residuals.
3. Soft-mix averaged day experts with the previous server.
4. Soft-mix the best expert with server anchors.
5. Transplant the best expert's detection head.
6. Average day expert class/head rows.
7. Apply day residuals only to neck/head.
8. Reaggregate while suppressing night clients.
9. Compute a split-router oracle from evaluated experts.
10. Blend the best client expert with the best DQA policy from the previous loop.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch


MOE_ROOT = Path(__file__).resolve().parents[1]
SCENE_ROOT = MOE_ROOT.parent
REPO_ROOT = SCENE_ROOT.parents[1]
SCENE_SCRIPTS = SCENE_ROOT / "scripts"
DQA_ROOT = SCENE_ROOT.parent
PROTOCOL_VERSION = "scene_daynight_dqa_moe_04_ten_research_loops_v1"

for path in (SCENE_SCRIPTS, DQA_ROOT, MOE_ROOT / "scripts", REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dqa_cwa_aggregation as dqa_v1  # noqa: E402
import dqa_cwa_aggregation_v2 as dqa_v2  # noqa: E402
import run_moe_02_fedmox_posthoc_five_loop as loop1  # noqa: E402
import run_scene_daynight_dqa_02_head_to_full as htf  # noqa: E402


SOURCE_WORKSPACE = SCENE_ROOT / "output" / "02_head_to_full_long_dqa"
PREV_LOOP_WORKSPACE = MOE_ROOT / "output" / "03_five_research_loops"
DEFAULT_WORKSPACE = MOE_ROOT / "output" / "04_ten_research_loops"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: Any) -> float | None:
    return htf.as_float(value)


def source_records(source_workspace: Path) -> dict[str, dict[str, str]]:
    return loop1.source_records(source_workspace)


def require_record(records: Mapping[str, dict[str, str]], label: str) -> Path:
    return loop1.require_record(records, label)


def _load(path: Path) -> dict[str, Any]:
    return dqa_v1._load_checkpoint(path, REPO_ROOT)


def _state_dict(ckpt: Mapping[str, Any], key: str) -> dict[str, torch.Tensor] | None:
    if ckpt.get(key) is None:
        return None
    return dqa_v1._model_state_dict(ckpt, key)


def _replace(base: dict[str, Any], state: Mapping[str, torch.Tensor], key: str) -> None:
    dqa_v1._replace_model_state(base, dict(state), key)


def save_checkpoint(base: dict[str, Any], output: Path) -> Path:
    base["epoch"] = -1
    base["optimizer"] = None
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, output)
    return output


def save_record(
    rows: list[dict[str, str]],
    label: str,
    path: Path,
    *,
    loop_id: str,
    hypothesis: str,
    implementation: str,
    kind: str = "aggregate",
) -> None:
    rows.append(
        {
            "label": label,
            "kind": kind,
            "phase": "ten_loop",
            "phase_round": "",
            "global_round": "",
            "client": "",
            "variant": implementation,
            "loop_id": loop_id,
            "hypothesis": hypothesis,
            "path": str(path.resolve()),
        }
    )


def client_ids() -> list[str]:
    return [
        "client0_highway_day",
        "client1_highway_night",
        "client2_citystreet_day",
        "client3_citystreet_night",
        "client4_residential_day",
        "client5_residential_night",
    ]


def day_ids() -> list[str]:
    return ["client0_highway_day", "client2_citystreet_day", "client4_residential_day"]


def night_ids() -> list[str]:
    return ["client1_highway_night", "client3_citystreet_night", "client5_residential_night"]


def phase1_client_paths(records: Mapping[str, dict[str, str]]) -> dict[str, Path]:
    return {
        client_id: require_record(records, label)
        for client_id, label in zip(client_ids(), loop1.client_labels("phase1_head", 30))
    }


def weighted_residual_state(
    base_state: Mapping[str, torch.Tensor],
    source_states: Sequence[Mapping[str, torch.Tensor]],
    anchor_state: Mapping[str, torch.Tensor],
    *,
    weights: Sequence[float],
    beta: float,
    key_filter: Callable[[str], bool] | None = None,
    localize_bn: bool = True,
) -> dict[str, torch.Tensor]:
    total = float(sum(weights)) or 1.0
    norm = [float(weight) / total for weight in weights]
    result: dict[str, torch.Tensor] = {}
    for key, base_value in base_state.items():
        if localize_bn and dqa_v1._is_batchnorm_key(key):
            result[key] = base_value
        elif key_filter is not None and not key_filter(key):
            result[key] = base_value
        elif torch.is_tensor(base_value) and base_value.dtype.is_floating_point:
            residual = torch.zeros_like(base_value.float())
            for weight, state in zip(norm, source_states):
                residual = residual + weight * (state[key].float() - anchor_state[key].float())
            result[key] = (base_value.float() + beta * residual).to(base_value.dtype)
        else:
            result[key] = base_value
    return result


def residual_mix_checkpoint(
    *,
    base: Path,
    sources: Sequence[Path],
    anchor: Path,
    output: Path,
    weights: Sequence[float] | None = None,
    beta: float,
    key_filter: Callable[[str], bool] | None = None,
) -> Path:
    base_ckpt = _load(base)
    source_ckpts = [_load(path) for path in sources]
    anchor_ckpt = _load(anchor)
    out = copy.deepcopy(base_ckpt)
    weights = list(weights or [1.0] * len(sources))
    model = weighted_residual_state(
        dqa_v1._model_state_dict(base_ckpt, "model"),
        [dqa_v1._model_state_dict(ckpt, "model") for ckpt in source_ckpts],
        dqa_v1._model_state_dict(anchor_ckpt, "model"),
        weights=weights,
        beta=beta,
        key_filter=key_filter,
    )
    _replace(out, model, "model")
    base_ema = _state_dict(base_ckpt, "ema")
    source_emas = [_state_dict(ckpt, "ema") for ckpt in source_ckpts]
    anchor_ema = _state_dict(anchor_ckpt, "ema")
    if base_ema is not None and anchor_ema is not None and all(item is not None for item in source_emas):
        ema = weighted_residual_state(
            base_ema,
            [item for item in source_emas if item is not None],
            anchor_ema,
            weights=weights,
            beta=beta,
            key_filter=key_filter,
        )
        _replace(out, ema, "ema")
    return save_checkpoint(out, output)


def softmix_checkpoint(server: Path, expert: Path, output: Path, *, server_alpha: float) -> Path:
    return loop1.softmix_checkpoint(server, expert, output, server_alpha=server_alpha)


def transplant_matching_keys_state(
    base_state: Mapping[str, torch.Tensor],
    source_state: Mapping[str, torch.Tensor],
    predicate: Callable[[str], bool],
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, base_value in base_state.items():
        if predicate(key) and torch.is_tensor(base_value) and base_value.dtype.is_floating_point:
            result[key] = source_state[key].to(base_value.dtype)
        else:
            result[key] = base_value
    return result


def transplant_checkpoint(base: Path, source: Path, output: Path, *, predicate: Callable[[str], bool]) -> Path:
    base_ckpt = _load(base)
    source_ckpt = _load(source)
    out = copy.deepcopy(base_ckpt)
    _replace(
        out,
        transplant_matching_keys_state(
            dqa_v1._model_state_dict(base_ckpt, "model"),
            dqa_v1._model_state_dict(source_ckpt, "model"),
            predicate,
        ),
        "model",
    )
    base_ema = _state_dict(base_ckpt, "ema")
    source_ema = _state_dict(source_ckpt, "ema")
    if base_ema is not None and source_ema is not None:
        _replace(out, transplant_matching_keys_state(base_ema, source_ema, predicate), "ema")
    return save_checkpoint(out, output)


def average_class_rows_state(
    base_state: Mapping[str, torch.Tensor],
    source_states: Sequence[Mapping[str, torch.Tensor]],
    *,
    num_classes: int,
    blend: float,
) -> dict[str, torch.Tensor]:
    result = {key: value for key, value in base_state.items()}
    for key, base_value in base_state.items():
        if not torch.is_tensor(base_value) or not base_value.dtype.is_floating_point:
            continue
        rows_by_class = dqa_v1._classification_rows(key, base_value, num_classes)
        if rows_by_class is None:
            continue
        updated = base_value.float().clone()
        mean_value = torch.stack([state[key].float() for state in source_states], dim=0).mean(dim=0)
        for rows in rows_by_class:
            for row in rows:
                updated[row] = (1.0 - blend) * updated[row] + blend * mean_value[row]
        result[key] = updated.to(base_value.dtype)
    return result


def average_class_rows_checkpoint(
    base: Path,
    sources: Sequence[Path],
    output: Path,
    *,
    num_classes: int,
    blend: float,
) -> Path:
    base_ckpt = _load(base)
    source_ckpts = [_load(path) for path in sources]
    out = copy.deepcopy(base_ckpt)
    _replace(
        out,
        average_class_rows_state(
            dqa_v1._model_state_dict(base_ckpt, "model"),
            [dqa_v1._model_state_dict(ckpt, "model") for ckpt in source_ckpts],
            num_classes=num_classes,
            blend=blend,
        ),
        "model",
    )
    base_ema = _state_dict(base_ckpt, "ema")
    source_emas = [_state_dict(ckpt, "ema") for ckpt in source_ckpts]
    if base_ema is not None and all(item is not None for item in source_emas):
        _replace(
            out,
            average_class_rows_state(
                base_ema,
                [item for item in source_emas if item is not None],
                num_classes=num_classes,
                blend=blend,
            ),
            "ema",
        )
    return save_checkpoint(out, output)


def phase1_stats(source_workspace: Path, setup) -> list[dqa_v1.ClientClassStats]:  # noqa: ANN001
    pseudo_json = source_workspace / "stats" / "03_round030_pseudo_label_stats.json"
    pseudo_stats = json.loads(pseudo_json.read_text(encoding="utf-8"))
    return htf.dqa01.pseudo_stats_to_dqa_stats(pseudo_stats, num_classes=len(setup.BDD_NAMES))


def subset_stats(stats: Sequence[dqa_v1.ClientClassStats], ids: Sequence[str]) -> list[dqa_v1.ClientClassStats]:
    by_id = {item.client_id: item for item in stats}
    return [by_id[item] for item in ids]


def dqa_reaggregate_subset(
    *,
    records: Mapping[str, dict[str, str]],
    setup,
    source_workspace: Path,
    selected_ids: Sequence[str],
    output: Path,
    state_path: Path,
    temperature: float,
    min_server_alpha: float,
    classwise_blend: float,
    residual_blend: float,
    server_anchor: float,
) -> Path:
    paths = phase1_client_paths(records)
    stats = phase1_stats(source_workspace, setup)
    config = dqa_v2.AggregationConfig(
        num_classes=len(setup.BDD_NAMES),
        count_ema=0.65,
        quality_ema=0.65,
        alpha_ema=0.40,
        temperature=temperature,
        uniform_mix=0.05,
        classwise_blend=classwise_blend,
        stability_lambda=0.70,
        min_effective_count=5.0,
        min_quality=0.10,
        max_quality=1.0,
        server_anchor=server_anchor,
        localize_bn=True,
        min_server_alpha=min_server_alpha,
        residual_blend=residual_blend,
    )
    dqa_v2.aggregate_checkpoints(
        client_checkpoints=[paths[item] for item in selected_ids],
        server_checkpoint=require_record(records, "phase1_head_round029_server_repair"),
        output_checkpoint=output,
        stats=subset_stats(stats, selected_ids),
        state_path=state_path,
        config=config,
        repo_root=REPO_ROOT,
    )
    return output


def is_head_key(key: str) -> bool:
    return key.startswith("head.")


def is_neck_or_head_key(key: str) -> bool:
    return key.startswith("neck.") or key.startswith("head.")


def is_head_weight_key(key: str) -> bool:
    return key.startswith("head.m.")


def previous_best_checkpoint(prev_workspace: Path, fallback: Path) -> Path:
    path = prev_workspace / "checkpoints" / "loop5_dqa_more_client_residual.pt"
    return path if path.exists() else fallback


def generate_candidates(args: argparse.Namespace, setup) -> list[dict[str, str]]:  # noqa: ANN001
    records = source_records(args.source_workspace)
    paths = phase1_client_paths(records)
    out_dir = args.workspace_root / "checkpoints"
    stats_dir = args.workspace_root / "stats"

    phase1_prev = require_record(records, "phase1_head_round029_server_repair")
    phase1_agg = require_record(records, "phase1_head_round030_dqa_aggregate")
    phase1_repair = require_record(records, "phase1_head_round030_server_repair")
    warmup = require_record(records, "warmup_global")
    best_expert = paths["client0_highway_day"]
    day_paths = [paths[item] for item in day_ids()]
    day_weights = [1.0, 1.0, 1.0]
    rows: list[dict[str, str]] = []

    # Loop 1: confirm and compare top day experts.
    for client_id in day_ids():
        save_record(
            rows,
            f"loop01_confirm_{client_id}",
            paths[client_id],
            loop_id="loop01_confirm_day_experts",
            hypothesis="day experts carry the best global target signal discovered in the previous sprint",
            implementation=f"direct evaluation of {client_id} phase1 expert",
        )

    # Loop 2: average day residuals.
    for beta in (0.50, 0.75, 1.00):
        label = f"loop02_day_residual_avg_b{int(beta * 100):03d}"
        path = residual_mix_checkpoint(
            base=phase1_prev,
            sources=day_paths,
            anchor=phase1_prev,
            output=out_dir / f"{label}.pt",
            weights=day_weights,
            beta=beta,
        )
        save_record(
            rows,
            label,
            path,
            loop_id="loop02_day_residual_average",
            hypothesis="averaging only useful day-expert residuals beats all-client aggregation",
            implementation=f"previous_server + {beta:.2f} * mean(day_client - previous_server)",
        )

    # Loop 3: server soft-mix with day residual average.
    day_avg = out_dir / "loop03_day_avg_full_residual_b100.pt"
    residual_mix_checkpoint(base=phase1_prev, sources=day_paths, anchor=phase1_prev, output=day_avg, weights=day_weights, beta=1.0)
    for alpha in (0.30, 0.50, 0.70):
        label = f"loop03_day_avg_softmix_a{int(alpha * 100):02d}"
        path = softmix_checkpoint(phase1_prev, day_avg, out_dir / f"{label}.pt", server_alpha=alpha)
        save_record(
            rows,
            label,
            path,
            loop_id="loop03_day_avg_softmix",
            hypothesis="FedMox-style soft mixture can stabilize day-expert average",
            implementation=f"softmix previous_server/day_avg with server_alpha={alpha:.2f}",
        )

    # Loop 4: best expert soft-mixed with server anchors.
    for base_label, base_path in (("prevserver", phase1_prev), ("warmup", warmup), ("repair", phase1_repair)):
        for alpha in (0.25, 0.50):
            label = f"loop04_bestexpert_{base_label}_softmix_a{int(alpha * 100):02d}"
            path = softmix_checkpoint(base_path, best_expert, out_dir / f"{label}.pt", server_alpha=alpha)
            save_record(
                rows,
                label,
                path,
                loop_id="loop04_best_expert_softmix",
                hypothesis="the best expert can be made safer by anchoring it to server/warmup weights",
                implementation=f"softmix {base_label}/client0_highway_day with server_alpha={alpha:.2f}",
            )

    # Loop 5: transplant best expert head.
    for base_label, base_path in (("prevserver", phase1_prev), ("aggregate", phase1_agg), ("warmup", warmup)):
        label = f"loop05_bestexpert_head_to_{base_label}"
        path = transplant_checkpoint(base_path, best_expert, out_dir / f"{label}.pt", predicate=is_head_weight_key)
        save_record(
            rows,
            label,
            path,
            loop_id="loop05_best_expert_head_transplant",
            hypothesis="the useful client signal lives mainly in YOLO head weights",
            implementation=f"transplant client0 head.m weights into {base_label}",
        )

    # Loop 6: average class rows from day experts.
    for base_label, base_path in (("prevserver", phase1_prev), ("aggregate", phase1_agg)):
        for blend in (0.50, 1.00):
            label = f"loop06_day_classrows_{base_label}_b{int(blend * 100):03d}"
            path = average_class_rows_checkpoint(
                base_path,
                day_paths,
                out_dir / f"{label}.pt",
                num_classes=len(setup.BDD_NAMES),
                blend=blend,
            )
            save_record(
                rows,
                label,
                path,
                loop_id="loop06_day_class_row_average",
                hypothesis="class-specific day expertise should be injected without moving bbox/objectness rows",
                implementation=f"average day expert classification rows into {base_label}, blend={blend:.2f}",
            )

    # Loop 7: apply day residuals only to neck/head.
    for beta in (0.50, 1.00):
        label = f"loop07_day_neck_head_residual_b{int(beta * 100):03d}"
        path = residual_mix_checkpoint(
            base=phase1_prev,
            sources=day_paths,
            anchor=phase1_prev,
            output=out_dir / f"{label}.pt",
            weights=day_weights,
            beta=beta,
            key_filter=is_neck_or_head_key,
        )
        save_record(
            rows,
            label,
            path,
            loop_id="loop07_day_neck_head_residual",
            hypothesis="preserving backbone while adapting neck/head is safer than full residual mixing",
            implementation=f"previous_server + {beta:.2f} * mean(day residual) on neck/head only",
        )

    # Loop 8: DQA reaggregation with night suppression.
    for blend, residual in ((0.28, 0.18), (0.40, 0.22)):
        label = f"loop08_day_only_dqa_b{int(blend * 100):02d}_r{int(residual * 100):02d}"
        path = dqa_reaggregate_subset(
            records=records,
            setup=setup,
            source_workspace=args.source_workspace,
            selected_ids=day_ids(),
            output=out_dir / f"{label}.pt",
            state_path=stats_dir / f"{label}_dqa_state.json",
            temperature=2.0,
            min_server_alpha=0.62,
            classwise_blend=blend,
            residual_blend=residual,
            server_anchor=8.0,
        )
        save_record(
            rows,
            label,
            path,
            loop_id="loop08_suppress_night_clients",
            hypothesis="night pseudoGT updates are harmful; day clients provide the transferable signal",
            implementation=f"day-only DQA temp=2.0 min_server=0.62 class_blend={blend:.2f} residual={residual:.2f}",
        )

    # Loop 9: evaluated checkpoints will be used for virtual router metrics.
    for client_id in night_ids():
        save_record(
            rows,
            f"loop09_router_pool_{client_id}",
            paths[client_id],
            loop_id="loop09_split_router_oracle",
            hypothesis="a router needs both positive and negative experts to estimate routing upper bound",
            implementation=f"include {client_id} in router oracle pool",
        )

    # Loop 10: blend previous best DQA reaggregate with best expert.
    prev_best = previous_best_checkpoint(args.prev_loop_workspace, phase1_agg)
    for alpha in (0.25, 0.50, 0.75):
        label = f"loop10_prevbest_bestexpert_softmix_a{int(alpha * 100):02d}"
        path = softmix_checkpoint(prev_best, best_expert, out_dir / f"{label}.pt", server_alpha=alpha)
        save_record(
            rows,
            label,
            path,
            loop_id="loop10_best_policy_plus_best_expert",
            hypothesis="the best DQA policy and best expert contain complementary signal",
            implementation=f"softmix previous-loop best policy/client0 expert with policy_alpha={alpha:.2f}",
        )

    return rows


def baseline_rows(source_workspace: Path, prev_workspace: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(source_workspace / "stats" / "02_head_to_full_final_metrics.csv"):
        if row.get("checkpoint_label") not in {"warmup_global", "phase1_final_aggregate", "phase1_final_repair", "phase2_final_repair"}:
            continue
        rows.append(
            {
                "loop_id": "baseline_02",
                "checkpoint_label": row.get("checkpoint_label", ""),
                "kind": row.get("kind", ""),
                "variant": "source 02 baseline",
                "precision": row.get("precision", ""),
                "recall": row.get("recall", ""),
                "map50": row.get("map50", ""),
                "map50_95": row.get("map50_95", ""),
                "gain_vs_warmup_map50_95": row.get("gain_vs_warmup_map50_95", ""),
                "worst_split": row.get("worst_split", ""),
                "worst_split_map50_95": row.get("worst_split_map50_95", ""),
                "day_avg_map50_95": row.get("day_avg_map50_95", ""),
                "night_avg_map50_95": row.get("night_avg_map50_95", ""),
                "day_night_gap_map50_95": row.get("day_night_gap_map50_95", ""),
            }
        )
    for row in read_csv(prev_workspace / "stats" / "03_five_research_loop_metrics.csv"):
        if row.get("checkpoint_label") in {"loop4_expert_client0_highway_day", "loop5_dqa_more_client_residual"}:
            copied = dict(row)
            copied["loop_id"] = "baseline_03_best"
            rows.append(copied)
    return rows


def new_metric_rows(args: argparse.Namespace, eval_records: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_path = args.workspace_root / "validation_reports" / "paper_protocol_eval_summary.csv"
    rows = [row for row in read_csv(summary_path) if row.get("status") == "ok"]
    by_label_split = {(row["checkpoint_label"], row["split"]): row for row in rows}
    totals = {row["checkpoint_label"]: row for row in rows if row.get("split") in {"scene_daynight_total", "total"}}
    warm = next((row for row in baseline_rows(args.source_workspace, args.prev_loop_workspace) if row["checkpoint_label"] == "warmup_global"), None)
    warm_m95 = as_float(warm.get("map50_95")) if warm else None
    meta = {row["label"]: row for row in eval_records}
    metric_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for label, total in totals.items():
        m95 = as_float(total.get("map50_95"))
        gap = htf.split_gap_metrics(by_label_split, label)
        metric_rows.append(
            {
                "loop_id": meta[label].get("loop_id", ""),
                "checkpoint_label": label,
                "kind": meta[label].get("kind", ""),
                "variant": meta[label].get("variant", ""),
                "precision": total.get("precision", ""),
                "recall": total.get("recall", ""),
                "map50": total.get("map50", ""),
                "map50_95": total.get("map50_95", ""),
                "gain_vs_warmup_map50_95": "" if m95 is None or warm_m95 is None else f"{m95 - warm_m95:.6f}",
                **gap,
            }
        )
    for row in rows:
        label = row["checkpoint_label"]
        if label not in meta:
            continue
        split_rows.append(
            {
                "checkpoint_label": label,
                "loop_id": meta[label].get("loop_id", ""),
                "split": row["split"],
                "map50": row.get("map50", ""),
                "map50_95": row.get("map50_95", ""),
            }
        )
    return metric_rows, split_rows


def router_oracle_rows(split_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pool_loops = {"loop01_confirm_day_experts", "loop09_split_router_oracle"}
    rows = [row for row in split_rows if row.get("loop_id") in pool_loops and row.get("split") not in {"scene_daynight_total", "total"}]
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_split.setdefault(str(row["split"]), []).append(row)
    oracle: list[dict[str, Any]] = []
    for split, items in sorted(by_split.items()):
        best = max(items, key=lambda row: as_float(row.get("map50_95")) or -1.0)
        oracle.append(
            {
                "loop_id": "loop09_split_router_oracle",
                "checkpoint_label": f"virtual_router_best_for_{split}",
                "kind": "virtual_oracle",
                "variant": f"best expert for {split}: {best['checkpoint_label']}",
                "precision": "",
                "recall": "",
                "map50": best.get("map50", ""),
                "map50_95": best.get("map50_95", ""),
                "gain_vs_warmup_map50_95": "",
                "worst_split": split,
                "worst_split_map50_95": best.get("map50_95", ""),
                "day_avg_map50_95": "",
                "night_avg_map50_95": "",
                "day_night_gap_map50_95": "",
            }
        )
    if oracle:
        values = [as_float(row.get("map50_95")) for row in oracle if as_float(row.get("map50_95")) is not None]
        if values:
            oracle.append(
                {
                    "loop_id": "loop09_split_router_oracle",
                    "checkpoint_label": "virtual_router_unweighted_split_avg",
                    "kind": "virtual_oracle",
                    "variant": "unweighted mean of best expert mAP50:95 across six splits",
                    "precision": "",
                    "recall": "",
                    "map50": "",
                    "map50_95": f"{sum(values) / len(values):.6f}",
                    "gain_vs_warmup_map50_95": "",
                    "worst_split": "",
                    "worst_split_map50_95": "",
                    "day_avg_map50_95": "",
                    "night_avg_map50_95": "",
                    "day_night_gap_map50_95": "",
                }
            )
    return oracle


def summarize_loop(rows: list[dict[str, Any]], loop_id: str) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("loop_id") == loop_id and as_float(row.get("map50_95")) is not None]
    if not candidates:
        return {"loop_id": loop_id, "best_checkpoint": "", "best_map50_95": "", "finding": "no metrics"}
    best = max(candidates, key=lambda row: as_float(row.get("map50_95")) or -1.0)
    return {
        "loop_id": loop_id,
        "best_checkpoint": best["checkpoint_label"],
        "best_map50": best.get("map50", ""),
        "best_map50_95": best.get("map50_95", ""),
        "best_night_avg_map50_95": best.get("night_avg_map50_95", ""),
        "finding": best.get("variant", ""),
    }


def loop_log(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    meta = [
        ("loop01_confirm_day_experts", "FedMix/FedJETs", "Confirm that client/day experts are real useful experts.", "Evaluate top day client experts.", "If confirmed, use them as expert pool."),
        ("loop02_day_residual_average", "FedAvg/FedMix", "Useful signal is shared across day experts and survives residual averaging.", "Average day residuals from previous server.", "If good, use day-only residual aggregation."),
        ("loop03_day_avg_softmix", "FedMox Soft-Mixture", "SoftMix stabilizes day expert average.", "Soft-mix day average with previous server.", "If good, add SoftMix to DQA aggregation."),
        ("loop04_best_expert_softmix", "Specialized FL / FedMix", "Best expert can be anchored without losing its gain.", "Soft-mix best expert with server/warmup/repair.", "If good, use best-expert anchor policy."),
        ("loop05_best_expert_head_transplant", "FedRep/FedMox head adaptation", "The useful part is mainly YOLO head weights.", "Transplant best expert head.m weights.", "If good, implement head expert modules."),
        ("loop06_day_class_row_average", "DQA-CWA", "Only class rows should move by expert information.", "Average day expert classification rows.", "If good, improve classwise aggregation."),
        ("loop07_day_neck_head_residual", "FedSTO selective training", "Neck/head residual is safer than full residual.", "Apply day residual on neck/head only.", "If good, selective expert update is the full-run candidate."),
        ("loop08_suppress_night_clients", "Domain-aware aggregation", "Night pseudoGT clients are harmful for global aggregation.", "DQA with only day clients.", "If good, learn a client suppression gate."),
        ("loop09_split_router_oracle", "FedJETs/pFedMoE router", "Split-wise routing has higher upper bound than one global model.", "Virtual best expert per split.", "If large, build real router."),
        ("loop10_best_policy_plus_best_expert", "PM-MoE/FedMix", "Best policy and best expert are complementary.", "Soft-mix previous best DQA policy with best expert.", "If good, promote hybrid policy."),
    ]
    summaries = {item[0]: summarize_loop(all_rows, item[0]) for item in meta}
    out = []
    for loop_id, papers, hypothesis, implementation, next_policy in meta:
        out.append(
            {
                "loop_id": loop_id,
                "papers": papers,
                "hypothesis": hypothesis,
                "implementation": implementation,
                **summaries[loop_id],
                "next_policy": next_policy,
            }
        )
    return out


def write_report(args: argparse.Namespace, all_rows: list[dict[str, Any]], logs: list[dict[str, Any]]) -> None:
    ranked = sorted(
        [row for row in all_rows if as_float(row.get("map50_95")) is not None],
        key=lambda row: as_float(row.get("map50_95")) or -1.0,
        reverse=True,
    )
    lines = [
        "# MoE x DQA Ten Research Loops",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- protocol: {PROTOCOL_VERSION}",
        "",
        "## Top Checkpoints",
        "",
        "| rank | loop | checkpoint | mAP50 | mAP50:95 | night avg | worst split | variant |",
        "|---:|---|---|---:|---:|---:|---|---|",
    ]
    for idx, row in enumerate(ranked[:25], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row.get("loop_id", "")),
                    str(row.get("checkpoint_label", "")),
                    str(row.get("map50", "")),
                    str(row.get("map50_95", "")),
                    str(row.get("night_avg_map50_95", "")),
                    str(row.get("worst_split", "")),
                    str(row.get("variant", "")).replace("|", "/"),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Loop Log", ""])
    for item in logs:
        lines.extend(
            [
                f"### {item['loop_id']}",
                "",
                f"- papers: {item['papers']}",
                f"- hypothesis: {item['hypothesis']}",
                f"- implementation: {item['implementation']}",
                f"- best: `{item.get('best_checkpoint', '')}` mAP50:95={item.get('best_map50_95', '')}",
                f"- finding: {item.get('finding', '')}",
                f"- next_policy: {item['next_policy']}",
                "",
            ]
        )
    (args.workspace_root / "04_ten_research_loop_report.md").write_text("\n".join(lines), encoding="utf-8")


def notify(args: argparse.Namespace, message: str, *, title: str, status: str | None = None, error: str | None = None) -> None:
    try:
        from notebook_notify import notify_discord

        context: dict[str, Any] = {
            "workspace": str(args.workspace_root.resolve()),
            "status": status or "",
        }
        metrics_path = args.workspace_root / "stats" / "04_ten_research_loop_metrics.csv"
        if metrics_path.exists():
            rows = read_csv(metrics_path)
            ranked = sorted(
                [row for row in rows if row.get("map50_95")],
                key=lambda row: float(row["map50_95"]),
                reverse=True,
            )
            context["metrics_csv"] = str(metrics_path)
            if ranked:
                context["best"] = ranked[0]["checkpoint_label"]
                context["best_map50_95"] = ranked[0]["map50_95"]
        if error:
            context["error"] = error[:500]
        print(notify_discord(message, title=title, context=context, fail_silently=True))
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-workspace", type=Path, default=SOURCE_WORKSPACE)
    parser.add_argument("--prev-loop-workspace", type=Path, default=PREV_LOOP_WORKSPACE)
    parser.add_argument("--warmup-checkpoint", type=Path, default=REPO_ROOT / "pseudogt_learnability" / "checkpoints" / "round000_warmup.pt")
    parser.add_argument("--client-limit", type=int, default=1500)
    parser.add_argument("--clients", default="all")
    parser.add_argument("--device", default="")
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument(
        "--eval-splits",
        default="highway_day,highway_night,citystreet_day,citystreet_night,residential_day,residential_night,total",
    )
    parser.add_argument("--eval-clients", action="store_true")
    parser.add_argument("--classwise", action="store_true")
    parser.add_argument("--no-eval-plots", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--notify-start", action="store_true")
    parser.add_argument("--notify-end", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.source_workspace = args.source_workspace.expanduser().resolve()
    args.prev_loop_workspace = args.prev_loop_workspace.expanduser().resolve()
    htf.pl03.ensure_dirs(args.workspace_root)
    setup, _fedsto, _manifest, _clients, _warmup = htf.prepare(args)
    if args.setup_only:
        print("Setup complete.")
        return []

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "source_workspace": str(args.source_workspace.resolve()),
        "prev_loop_workspace": str(args.prev_loop_workspace.resolve()),
        "workspace": str(args.workspace_root.resolve()),
        "sources": [
            "https://arxiv.org/abs/2508.16568",
            "https://arxiv.org/abs/2107.06724",
            "https://openreview.net/forum?id=hEl2HpiH3g",
            "https://arxiv.org/abs/2402.01350",
            "https://openaccess.thecvf.com/content/CVPR2025W/FedVision/html/Radwan_FedDG-MoE_Test-Time_Mixture-of-Experts_Fusion_for_Federated_Domain_Generalization_CVPRW_2025_paper.html",
            "https://arxiv.org/abs/2102.07623",
        ],
    }
    (args.workspace_root / "stats" / "04_ten_research_loop_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    eval_records = generate_candidates(args, setup)
    write_csv(
        args.workspace_root / "stats" / "04_ten_research_loop_checkpoints.csv",
        eval_records,
        ["label", "kind", "phase", "phase_round", "global_round", "client", "variant", "loop_id", "hypothesis", "path"],
    )

    metric_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    if args.evaluate:
        htf.base01_0.run_evaluation(args, eval_records)
        metric_rows, split_rows = new_metric_rows(args, eval_records)
        metric_rows.extend(router_oracle_rows(split_rows))

    all_rows = baseline_rows(args.source_workspace, args.prev_loop_workspace) + metric_rows
    metric_fields = [
        "loop_id",
        "checkpoint_label",
        "kind",
        "variant",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "gain_vs_warmup_map50_95",
        "worst_split",
        "worst_split_map50_95",
        "day_avg_map50_95",
        "night_avg_map50_95",
        "day_night_gap_map50_95",
    ]
    write_csv(args.workspace_root / "stats" / "04_ten_research_loop_metrics.csv", all_rows, metric_fields)
    logs = loop_log(all_rows)
    write_csv(
        args.workspace_root / "stats" / "04_ten_research_loop_log.csv",
        logs,
        [
            "loop_id",
            "papers",
            "hypothesis",
            "implementation",
            "best_checkpoint",
            "best_map50",
            "best_map50_95",
            "best_night_avg_map50_95",
            "finding",
            "next_policy",
        ],
    )
    write_report(args, all_rows, logs)
    return all_rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    do_start_notify = args.notify or args.notify_start
    do_end_notify = args.notify or args.notify_end
    if do_start_notify:
        notify(args, "MoE x DQA ten research-loop sprint started.", title="DQA MoE 04 start")
    status = "success"
    error: str | None = None
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = str(exc)
        raise
    finally:
        if do_end_notify:
            notify(
                args,
                f"MoE x DQA ten research-loop sprint finished with status={status}.",
                title="DQA MoE 04 finish",
                status=status,
                error=error,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
