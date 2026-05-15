#!/usr/bin/env python3
"""Run ten router-focused MoE x DQA loops.

The previous sprint showed that expert checkpoints outperform the single DQA
aggregate, while simple class/head transplants do not.  This runner therefore
focuses on router-oriented hypotheses:

* How large is the split-router upper bound?
* Can a single checkpoint approximate the day-expert router?
* Are BN/domain statistics part of expert behavior?
* Is the useful signal concentrated in one expert or shared across top-k day
  experts?
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
PROTOCOL_VERSION = "scene_daynight_dqa_moe_05_router_ten_loops_v1"

for path in (SCENE_SCRIPTS, DQA_ROOT, MOE_ROOT / "scripts", REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dqa_cwa_aggregation as dqa_v1  # noqa: E402
import run_moe_02_fedmox_posthoc_five_loop as loop1  # noqa: E402
import run_moe_04_ten_research_loops as loop4  # noqa: E402
import run_scene_daynight_dqa_02_head_to_full as htf  # noqa: E402


SOURCE_WORKSPACE = SCENE_ROOT / "output" / "02_head_to_full_long_dqa"
PREV_LOOP_WORKSPACE = MOE_ROOT / "output" / "04_ten_research_loops"
DEFAULT_WORKSPACE = MOE_ROOT / "output" / "05_router_ten_loops"


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


def source_records(source_workspace: Path) -> dict[str, dict[str, str]]:
    return loop1.source_records(source_workspace)


def require_record(records: Mapping[str, dict[str, str]], label: str) -> Path:
    return loop1.require_record(records, label)


def client_ids() -> list[str]:
    return loop4.client_ids()


def day_ids() -> list[str]:
    return loop4.day_ids()


def night_ids() -> list[str]:
    return loop4.night_ids()


def phase1_client_paths(records: Mapping[str, dict[str, str]]) -> dict[str, Path]:
    return loop4.phase1_client_paths(records)


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
            "phase": "router_loop",
            "phase_round": "",
            "global_round": "",
            "client": "",
            "variant": implementation,
            "loop_id": loop_id,
            "hypothesis": hypothesis,
            "path": str(path.resolve()),
        }
    )


def transplant_bn_state(base_state: Mapping[str, torch.Tensor], bn_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, base_value in base_state.items():
        result[key] = bn_state[key] if dqa_v1._is_batchnorm_key(key) else base_value
    return result


def transplant_bn_checkpoint(base: Path, bn_source: Path, output: Path) -> Path:
    base_ckpt = _load(base)
    bn_ckpt = _load(bn_source)
    out = copy.deepcopy(base_ckpt)
    _replace(
        out,
        transplant_bn_state(dqa_v1._model_state_dict(base_ckpt, "model"), dqa_v1._model_state_dict(bn_ckpt, "model")),
        "model",
    )
    base_ema = _state_dict(base_ckpt, "ema")
    bn_ema = _state_dict(bn_ckpt, "ema")
    if base_ema is not None and bn_ema is not None:
        _replace(out, transplant_bn_state(base_ema, bn_ema), "ema")
    return save_checkpoint(out, output)


def residual_mix_checkpoint(
    *,
    base: Path,
    sources: Sequence[Path],
    anchor: Path,
    output: Path,
    weights: Sequence[float],
    beta: float,
    key_filter: Callable[[str], bool] | None = None,
    localize_bn: bool,
) -> Path:
    base_ckpt = _load(base)
    source_ckpts = [_load(path) for path in sources]
    anchor_ckpt = _load(anchor)
    out = copy.deepcopy(base_ckpt)
    model = loop4.weighted_residual_state(
        dqa_v1._model_state_dict(base_ckpt, "model"),
        [dqa_v1._model_state_dict(ckpt, "model") for ckpt in source_ckpts],
        dqa_v1._model_state_dict(anchor_ckpt, "model"),
        weights=weights,
        beta=beta,
        key_filter=key_filter,
        localize_bn=localize_bn,
    )
    _replace(out, model, "model")
    base_ema = _state_dict(base_ckpt, "ema")
    source_emas = [_state_dict(ckpt, "ema") for ckpt in source_ckpts]
    anchor_ema = _state_dict(anchor_ckpt, "ema")
    if base_ema is not None and anchor_ema is not None and all(item is not None for item in source_emas):
        ema = loop4.weighted_residual_state(
            base_ema,
            [item for item in source_emas if item is not None],
            anchor_ema,
            weights=weights,
            beta=beta,
            key_filter=key_filter,
            localize_bn=localize_bn,
        )
        _replace(out, ema, "ema")
    return save_checkpoint(out, output)


def softmix_checkpoint(base: Path, expert: Path, output: Path, *, base_alpha: float) -> Path:
    return loop1.softmix_checkpoint(base, expert, output, server_alpha=base_alpha)


def is_neck_or_head(key: str) -> bool:
    return key.startswith("neck.") or key.startswith("head.")


def generate_candidates(args: argparse.Namespace) -> list[dict[str, str]]:
    records = source_records(args.source_workspace)
    paths = phase1_client_paths(records)
    out_dir = args.workspace_root / "checkpoints"
    rows: list[dict[str, str]] = []

    prev_server = require_record(records, "phase1_head_round029_server_repair")
    phase1_agg = require_record(records, "phase1_head_round030_dqa_aggregate")
    phase1_repair = require_record(records, "phase1_head_round030_server_repair")
    warmup = require_record(records, "warmup_global")
    day_paths = [paths[item] for item in day_ids()]
    all_paths = [paths[item] for item in client_ids()]
    best = paths["client0_highway_day"]
    city = paths["client2_citystreet_day"]
    res = paths["client4_residential_day"]

    # Loop 01: all expert pool for router upper-bound recomputation.
    for client_id in client_ids():
        save_record(
            rows,
            f"loop01_pool_{client_id}",
            paths[client_id],
            loop_id="loop01_expert_pool_for_router",
            hypothesis="router needs the full expert pool before learning gates",
            implementation=f"evaluate expert pool member {client_id}",
        )

    # Loop 04: very low-anchor best expert softmix.
    for alpha in (0.05, 0.10, 0.15):
        label = f"loop04_bestexpert_prevserver_lowanchor_a{int(alpha * 100):02d}"
        path = softmix_checkpoint(prev_server, best, out_dir / f"{label}.pt", base_alpha=alpha)
        save_record(
            rows,
            label,
            path,
            loop_id="loop04_low_anchor_best_expert",
            hypothesis="almost-direct best expert retains gain while small server anchor stabilizes it",
            implementation=f"softmix previous_server/best_expert with server_alpha={alpha:.2f}",
        )

    # Loop 05: expert behavior may include BN/domain statistics.
    for name, bn_source in (("prevserver", prev_server), ("repair", phase1_repair), ("warmup", warmup)):
        label = f"loop05_bestexpert_bn_{name}"
        path = transplant_bn_checkpoint(best, bn_source, out_dir / f"{label}.pt")
        save_record(
            rows,
            label,
            path,
            loop_id="loop05_best_expert_bn_router",
            hypothesis="expert routing requires domain-specific BN behavior, not just weights",
            implementation=f"best expert with BN stats from {name}",
        )

    # Loop 06: include BN in day residual mixture.
    for beta in (0.75, 1.00):
        label = f"loop06_day_residual_with_bn_b{int(beta * 100):03d}"
        path = residual_mix_checkpoint(
            base=prev_server,
            sources=day_paths,
            anchor=prev_server,
            output=out_dir / f"{label}.pt",
            weights=[1.0, 1.0, 1.0],
            beta=beta,
            localize_bn=False,
        )
        save_record(
            rows,
            label,
            path,
            loop_id="loop06_day_residual_with_bn",
            hypothesis="BN residuals are part of the day-expert specialization",
            implementation=f"previous_server + {beta:.2f} * mean(day residual), including BN",
        )

    # Loop 07: include BN but only for neck/head.
    for beta in (0.75, 1.00):
        label = f"loop07_day_neck_head_with_bn_b{int(beta * 100):03d}"
        path = residual_mix_checkpoint(
            base=prev_server,
            sources=day_paths,
            anchor=prev_server,
            output=out_dir / f"{label}.pt",
            weights=[1.0, 1.0, 1.0],
            beta=beta,
            key_filter=is_neck_or_head,
            localize_bn=False,
        )
        save_record(
            rows,
            label,
            path,
            loop_id="loop07_neck_head_with_bn",
            hypothesis="neck/head specialization plus BN is enough for router-like improvement",
            implementation=f"neck/head day residual beta={beta:.2f}, including BN",
        )

    # Loop 08: top-k weighted day experts, favoring the strongest expert.
    weight_specs = [
        ("client0_heavy", [0.60, 0.20, 0.20]),
        ("client0_city", [0.50, 0.35, 0.15]),
        ("client0_res", [0.50, 0.15, 0.35]),
    ]
    for name, weights in weight_specs:
        label = f"loop08_topk_day_weighted_{name}"
        path = residual_mix_checkpoint(
            base=prev_server,
            sources=day_paths,
            anchor=prev_server,
            output=out_dir / f"{label}.pt",
            weights=weights,
            beta=1.0,
            localize_bn=True,
        )
        save_record(
            rows,
            label,
            path,
            loop_id="loop08_topk_day_weighted",
            hypothesis="top-k expert weighting is better than uniform day averaging",
            implementation=f"day residual weights={weights}",
        )

    # Loop 09: scene-pair expert mixtures.
    pair_specs = [
        ("highway_city", [best, city], [0.5, 0.5]),
        ("highway_res", [best, res], [0.5, 0.5]),
        ("city_res", [city, res], [0.5, 0.5]),
    ]
    for name, sources, weights in pair_specs:
        label = f"loop09_scene_pair_{name}"
        path = residual_mix_checkpoint(
            base=prev_server,
            sources=sources,
            anchor=prev_server,
            output=out_dir / f"{label}.pt",
            weights=weights,
            beta=1.0,
            localize_bn=True,
        )
        save_record(
            rows,
            label,
            path,
            loop_id="loop09_scene_pair_mixture",
            hypothesis="router may need a small scene pair rather than all day experts",
            implementation=f"scene pair residual mixture {name}",
        )

    # Loop 10: all-client pool but shrink harmful updates with day-favored weights.
    all_weight_specs = [
        ("day90_night10", [0.30, 0.033, 0.30, 0.033, 0.30, 0.034]),
        ("day75_night25", [0.25, 0.083, 0.25, 0.083, 0.25, 0.084]),
        ("best50_day30_night20", [0.50, 0.067, 0.15, 0.067, 0.15, 0.066]),
    ]
    for name, weights in all_weight_specs:
        label = f"loop10_all_pool_shrunk_{name}"
        path = residual_mix_checkpoint(
            base=prev_server,
            sources=all_paths,
            anchor=prev_server,
            output=out_dir / f"{label}.pt",
            weights=weights,
            beta=1.0,
            localize_bn=True,
        )
        save_record(
            rows,
            label,
            path,
            loop_id="loop10_all_pool_shrunk_router",
            hypothesis="a router-like weighted pool can keep day signal while not fully discarding night experts",
            implementation=f"all-client residual weights={weights}",
        )

    # Two bridge candidates: best expert plus previous aggregate/policy.
    for name, base in (("phase1agg", phase1_agg), ("repair", phase1_repair)):
        label = f"loop10_bridge_bestexpert_{name}_a10"
        path = softmix_checkpoint(base, best, out_dir / f"{label}.pt", base_alpha=0.10)
        save_record(
            rows,
            label,
            path,
            loop_id="loop10_all_pool_shrunk_router",
            hypothesis="best expert can be blended back into existing DQA artifacts",
            implementation=f"softmix {name}/best_expert with base_alpha=0.10",
        )

    return rows


def baseline_rows(source_workspace: Path, prev_workspace: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(source_workspace / "stats" / "02_head_to_full_final_metrics.csv"):
        if row.get("checkpoint_label") in {"warmup_global", "phase1_final_aggregate", "phase1_final_repair", "phase2_final_repair"}:
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
    for row in read_csv(prev_workspace / "stats" / "04_ten_research_loop_metrics.csv"):
        if row.get("checkpoint_label") in {"loop01_confirm_client0_highway_day", "loop02_day_residual_avg_b075", "loop07_day_neck_head_residual_b100"}:
            copied = dict(row)
            copied["loop_id"] = "baseline_04_best"
            rows.append(copied)
    return rows


def source_split_rows(source_workspace: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(source_workspace / "stats" / "02_head_to_full_split_metrics.csv"):
        rows.append(
            {
                "checkpoint_label": row["checkpoint_label"],
                "loop_id": "baseline_02",
                "split": row["split"],
                "images": row.get("images", ""),
                "map50": row.get("map50", ""),
                "map50_95": row.get("map50_95", ""),
            }
        )
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
                "images": row.get("images", ""),
                "map50": row.get("map50", ""),
                "map50_95": row.get("map50_95", ""),
            }
        )
    return metric_rows, split_rows


def split_map(split_rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["checkpoint_label"], row["split"]): row for row in split_rows}


def weighted_virtual_total(
    by_label_split: Mapping[tuple[str, str], dict[str, Any]],
    choices: Mapping[str, str],
) -> tuple[float | None, float | None]:
    total_weight = 0.0
    m50_sum = 0.0
    m95_sum = 0.0
    for split, label in choices.items():
        row = by_label_split.get((label, split))
        if not row:
            return None, None
        images = as_float(row.get("images")) or 1.0
        m50 = as_float(row.get("map50"))
        m95 = as_float(row.get("map50_95"))
        if m50 is None or m95 is None:
            return None, None
        total_weight += images
        m50_sum += images * m50
        m95_sum += images * m95
    if total_weight <= 0:
        return None, None
    return m50_sum / total_weight, m95_sum / total_weight


def virtual_router_rows(args: argparse.Namespace, split_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_split = split_map(split_rows)
    eval_splits = [item for item in args.eval_splits.split(",") if item and item != "total"]

    pool = [f"loop01_pool_{item}" for item in client_ids()]
    best_choices: dict[str, str] = {}
    for split in eval_splits:
        available = [label for label in pool if (label, split) in by_split]
        if not available:
            continue
        best = max(available, key=lambda label: as_float(by_split[(label, split)].get("map50_95")) or -1.0)
        best_choices[split] = best

    day_best = "loop01_pool_client0_highway_day"
    scene_choices = {
        "highway_day": "loop01_pool_client0_highway_day",
        "highway_night": "warmup_global",
        "citystreet_day": "loop01_pool_client2_citystreet_day",
        "citystreet_night": "warmup_global",
        "residential_day": "loop01_pool_client4_residential_day",
        "residential_night": "warmup_global",
    }
    day_fallback_choices = {
        split: (day_best if split.endswith("_day") else "warmup_global")
        for split in eval_splits
    }
    day_expert_all_choices = {split: day_best for split in eval_splits}

    configs = [
        ("loop02_virtual_best_expert_per_split", best_choices, "best expert per split from evaluated expert pool"),
        ("loop03_virtual_scene_day_warmup_night", scene_choices, "scene-specific day expert and warmup for night"),
        ("loop03_virtual_bestday_warmup_night", day_fallback_choices, "best day expert for day, warmup for night"),
        ("loop03_virtual_bestday_all_splits", day_expert_all_choices, "best day expert for every split"),
    ]
    rows: list[dict[str, Any]] = []
    warm = next((row for row in baseline_rows(args.source_workspace, args.prev_loop_workspace) if row["checkpoint_label"] == "warmup_global"), None)
    warm_m95 = as_float(warm.get("map50_95")) if warm else None
    for label, choices, variant in configs:
        m50, m95 = weighted_virtual_total(by_split, choices)
        if m95 is None:
            continue
        loop_id = "loop02_router_oracle_weighted" if "best_expert" in label else "loop03_rule_router"
        rows.append(
            {
                "loop_id": loop_id,
                "checkpoint_label": label,
                "kind": "virtual_router",
                "variant": variant,
                "precision": "",
                "recall": "",
                "map50": f"{m50:.6f}" if m50 is not None else "",
                "map50_95": f"{m95:.6f}",
                "gain_vs_warmup_map50_95": "" if warm_m95 is None else f"{m95 - warm_m95:.6f}",
                "worst_split": "",
                "worst_split_map50_95": "",
                "day_avg_map50_95": "",
                "night_avg_map50_95": "",
                "day_night_gap_map50_95": "",
            }
        )
    return rows


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
        ("loop01_expert_pool_for_router", "FedJETs/pFedMoE", "A router needs a measured expert pool.", "Evaluate all six phase1 experts.", "Use the pool to train/fit a router."),
        ("loop02_router_oracle_weighted", "FedJETs", "Split-router upper bound should exceed a single checkpoint.", "Image-weighted virtual best expert per split.", "If large, implement real router."),
        ("loop03_rule_router", "FedBN/FedMix", "Simple metadata/domain rule can approximate router.", "Scene/day/warmup rule routers.", "If good, use metadata router baseline."),
        ("loop04_low_anchor_best_expert", "FedMox Soft-Mixture", "Small server anchor keeps best expert stable.", "Best expert with 5-15% server anchor.", "If near direct expert, use low-anchor policy."),
        ("loop05_best_expert_bn_router", "FedBN", "Expert behavior depends on BN/domain stats.", "Transplant BN into best expert.", "If sensitive, add per-domain BN router."),
        ("loop06_day_residual_with_bn", "FedMix/FedBN", "Day residual average should include BN.", "Day residual averaging with BN tensors.", "If better, stop freezing BN in aggregation."),
        ("loop07_neck_head_with_bn", "FedSTO selective training", "Neck/head+BN carries useful expert signal.", "Neck/head day residual including BN.", "If good, full run should update neck/head experts."),
        ("loop08_topk_day_weighted", "MoE top-k routing", "Top-k weighted day experts beat uniform day average.", "Weighted day residual mixtures.", "If good, router should output top-k weights."),
        ("loop09_scene_pair_mixture", "domain-aware MoE", "A pair of scene experts beats all-day pooling.", "Scene pair residual mixtures.", "If good, use small expert subsets."),
        ("loop10_all_pool_shrunk_router", "FedMoE/FLEX-MoE", "Weighted all-pool can keep day signal while retaining night experts.", "Day-favored all-client residual and bridge softmix.", "If good, learn client weights."),
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
        "# MoE x DQA Router Ten Loops",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- protocol: {PROTOCOL_VERSION}",
        "",
        "## Top Checkpoints / Virtual Routers",
        "",
        "| rank | loop | checkpoint | mAP50 | mAP50:95 | night avg | worst split | variant |",
        "|---:|---|---|---:|---:|---:|---|---|",
    ]
    for idx, row in enumerate(ranked[:30], start=1):
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
    (args.workspace_root / "05_router_ten_loop_report.md").write_text("\n".join(lines), encoding="utf-8")


def notify(args: argparse.Namespace, message: str, *, title: str, status: str | None = None, error: str | None = None) -> None:
    try:
        from notebook_notify import notify_discord

        context: dict[str, Any] = {"workspace": str(args.workspace_root.resolve()), "status": status or ""}
        metrics_path = args.workspace_root / "stats" / "05_router_ten_loop_metrics.csv"
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
    _setup, _fedsto, _manifest, _clients, _warmup = htf.prepare(args)
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
            "https://huggingface.co/papers/2306.08586",
            "https://arxiv.org/abs/2402.01350",
            "https://arxiv.org/abs/2102.07623",
            "https://arxiv.org/abs/2408.11304",
        ],
    }
    (args.workspace_root / "stats" / "05_router_ten_loop_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    eval_records = generate_candidates(args)
    write_csv(
        args.workspace_root / "stats" / "05_router_ten_loop_checkpoints.csv",
        eval_records,
        ["label", "kind", "phase", "phase_round", "global_round", "client", "variant", "loop_id", "hypothesis", "path"],
    )

    metric_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    if args.evaluate:
        htf.base01_0.run_evaluation(args, eval_records)
        metric_rows, split_rows = new_metric_rows(args, eval_records)
        split_rows = source_split_rows(args.source_workspace) + split_rows
        metric_rows.extend(virtual_router_rows(args, split_rows))

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
    write_csv(args.workspace_root / "stats" / "05_router_ten_loop_metrics.csv", all_rows, metric_fields)
    logs = loop_log(all_rows)
    write_csv(
        args.workspace_root / "stats" / "05_router_ten_loop_log.csv",
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
        notify(args, "MoE x DQA router ten-loop sprint started.", title="DQA MoE 05 start")
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
                f"MoE x DQA router ten-loop sprint finished with status={status}.",
                title="DQA MoE 05 finish",
                status=status,
                error=error,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
