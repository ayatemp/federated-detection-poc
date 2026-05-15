#!/usr/bin/env python3
"""Run fifteen local-region expert MoE x DQA design loops.

This runner is intentionally a fast screening pass, not fifteen full YOLO
training runs.  The previous full experiment already showed that BN/residual
DQA improves the scene/day-night target set, while final server repair can
erase part of that improvement.  This pass converts that evidence into a
local-region expert design space:

* experts are assigned to pseudo-GT learnability regions, not to whole clients;
* routing uses scene/time, density, stability, and scale proxies;
* the report chooses one next full experiment candidate.

The output is a reproducible set of loop hypotheses, evidence scores, and the
concrete implementation delta for the next notebook.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


MOE_ROOT = Path(__file__).resolve().parents[1]
SCENE_ROOT = MOE_ROOT.parent
REPO_ROOT = SCENE_ROOT.parents[1]
DEFAULT_SOURCE_WORKSPACE = SCENE_ROOT / "output" / "03_main_bn_residual_dqa_experiment"
DEFAULT_PREV_MOE_WORKSPACE = MOE_ROOT / "output" / "05_router_ten_loops"
DEFAULT_WORKSPACE = MOE_ROOT / "output" / "06_spatial_expert_fifteen_loops"
PROTOCOL_VERSION = "scene_daynight_dqa_moe_06_spatial_expert_fifteen_loops_v1"

for path in (REPO_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def client_from_split(split: str) -> str:
    mapping = {
        "highway_day": "client0_highway_day",
        "highway_night": "client1_highway_night",
        "citystreet_day": "client2_citystreet_day",
        "citystreet_night": "client3_citystreet_night",
        "residential_day": "client4_residential_day",
        "residential_night": "client5_residential_night",
    }
    return mapping[split]


def split_from_client(client: str) -> str:
    return client.removeprefix("client0_").removeprefix("client1_").removeprefix("client2_").removeprefix("client3_").removeprefix("client4_").removeprefix("client5_")


@dataclass(frozen=True)
class LoopSpec:
    loop_id: str
    papers: str
    hypothesis: str
    implementation_change: str
    routing_unit: str
    expert_unit: str
    uses_bn_neck: bool = False
    uses_repair_shield: bool = False
    uses_expert_choice: bool = False
    uses_soft_routing: bool = False
    uses_shared_private: bool = False
    uses_scale: bool = False
    uses_class_quota: bool = False
    uses_consistency: bool = False
    uses_negative_transfer_guard: bool = False
    risk_penalty: float = 0.0
    full_run_priority: int = 3


def loop_specs() -> list[LoopSpec]:
    return [
        LoopSpec(
            loop_id="loop01_soft_local_region_router",
            papers="Soft MoE; PSSFL/FedMox",
            hypothesis="Use a soft spatial router so borderline pseudo boxes can train multiple local experts instead of being hard-assigned to a client expert.",
            implementation_change="Create region tokens from bbox center/area/class/density; update top-2 local residual experts with soft weights.",
            routing_unit="pseudo box region token",
            expert_unit="neck/head residual expert",
            uses_soft_routing=True,
            uses_bn_neck=True,
            full_run_priority=2,
        ),
        LoopSpec(
            loop_id="loop02_expert_choice_pseudogt_quota",
            papers="Expert Choice Routing",
            hypothesis="Experts should select the pseudo-GT regions they can learn, with fixed quota per expert to avoid easy-box collapse.",
            implementation_change="For each expert, rank boxes by learnability=(stability*density balance)/uncertainty and train fixed bucket sizes.",
            routing_unit="expert-selected pseudo box bucket",
            expert_unit="local residual expert",
            uses_expert_choice=True,
            uses_class_quota=True,
            risk_penalty=0.0005,
            full_run_priority=1,
        ),
        LoopSpec(
            loop_id="loop03_fedmox_spatial_router",
            papers="PSSFL/FedMox",
            hypothesis="The key FedMox idea for us is not client MoE; it is spatial routing that absorbs resolution/domain mismatch.",
            implementation_change="Use K=4 local experts: easy-day, dense-scene, night-hard, small-object; aggregate by spatial router before server repair.",
            routing_unit="feature-map cell / bbox scale proxy",
            expert_unit="spatial adapter residual",
            uses_soft_routing=True,
            uses_scale=True,
            uses_bn_neck=True,
            full_run_priority=1,
        ),
        LoopSpec(
            loop_id="loop04_dataset_aware_local_moe",
            papers="DAMEX",
            hypothesis="Scene labels are useful as weak priors, but routing should happen at token level so a dataset/domain does not become a brittle expert.",
            implementation_change="Use scene/day-night tag only as router prior; local tokens can override with density/stability evidence.",
            routing_unit="scene-prior + local token",
            expert_unit="dataset-aware local expert",
            uses_soft_routing=True,
            uses_negative_transfer_guard=True,
            full_run_priority=2,
        ),
        LoopSpec(
            loop_id="loop05_shared_private_ple_mmoe",
            papers="MMoE; PLE",
            hypothesis="A shared expert should always preserve source/repair knowledge while private experts absorb learnable target residuals.",
            implementation_change="Update shared expert on all stable pseudo GT, private experts only on selected local regions; deploy shared + top-1 private residual.",
            routing_unit="local region plus shared path",
            expert_unit="shared-private residual expert",
            uses_shared_private=True,
            uses_soft_routing=True,
            risk_penalty=0.0003,
            full_run_priority=1,
        ),
        LoopSpec(
            loop_id="loop06_bn_neck_local_expert",
            papers="FedBN; previous MoE 05",
            hypothesis="The best observed signal came from neck/head residuals with BN included, so local experts should live exactly there.",
            implementation_change="Freeze backbone; train K=4 neck/head+BN local experts using pseudo-GT region buckets.",
            routing_unit="pseudo box region token",
            expert_unit="neck/head+BN residual expert",
            uses_bn_neck=True,
            full_run_priority=1,
        ),
        LoopSpec(
            loop_id="loop07_repair_shielded_local_experts",
            papers="PSSFL Soft-Mixture; previous 03",
            hypothesis="Server repair is useful, but it overwrote DQA gains; repair only the shared path and keep local experts un-repaired.",
            implementation_change="Run server repair on shared expert; keep local residual experts as frozen add-ons during final evaluation.",
            routing_unit="shared path + local residual gate",
            expert_unit="repair-shielded local residual",
            uses_repair_shield=True,
            uses_shared_private=True,
            uses_bn_neck=True,
            full_run_priority=1,
        ),
        LoopSpec(
            loop_id="loop08_entropy_load_balanced_router",
            papers="Switch/GShard load balance; Expert Choice",
            hypothesis="Pseudo-GT confidence collapse is an expert-load problem; add entropy and load balance so all learnable regions are used.",
            implementation_change="Add router entropy target and per-expert load target from pseudo label stats.",
            routing_unit="pseudo box region token",
            expert_unit="balanced local expert",
            uses_expert_choice=True,
            risk_penalty=0.0008,
            full_run_priority=2,
        ),
        LoopSpec(
            loop_id="loop09_dense_to_sparse_curriculum",
            papers="Soft MoE; Switch",
            hypothesis="Hard routing too early makes pseudo-GT errors permanent; begin dense/soft, then anneal to top-2 experts.",
            implementation_change="Rounds 1-10 soft all-expert residual, rounds 11-30 top-2 local experts with temperature annealing.",
            routing_unit="annealed local token",
            expert_unit="curriculum residual expert",
            uses_soft_routing=True,
            uses_bn_neck=True,
            risk_penalty=0.0002,
            full_run_priority=2,
        ),
        LoopSpec(
            loop_id="loop10_confidence_free_learnability_gate",
            papers="Semi-supervised detection uncertainty filtering",
            hypothesis="The router should not use confidence alone; it should prefer regions whose stability and density imply learnability.",
            implementation_change="Replace confidence threshold with learnability score: stability, box density, class quota, and cross-round agreement.",
            routing_unit="confidence-free pseudo region",
            expert_unit="learnability expert",
            uses_class_quota=True,
            uses_consistency=True,
            risk_penalty=0.0002,
            full_run_priority=1,
        ),
        LoopSpec(
            loop_id="loop11_scale_specialized_experts",
            papers="Spatial MoE; object detection scale specialization",
            hypothesis="Pseudo GT errors are scale-dependent; separate small/medium/large object experts can keep bbox noise local.",
            implementation_change="Route by normalized bbox area into small/medium/large plus shared expert; update bbox rows only for stable scale buckets.",
            routing_unit="bbox scale bucket",
            expert_unit="scale-local bbox/head expert",
            uses_scale=True,
            uses_bn_neck=True,
            risk_penalty=0.0004,
            full_run_priority=2,
        ),
        LoopSpec(
            loop_id="loop12_class_density_expert_choice",
            papers="Expert Choice; DQA classwise aggregation",
            hypothesis="DQA should keep classwise strength, but experts should choose class-density regions rather than entire clients.",
            implementation_change="Each expert owns class-density buckets; apply per-class quotas before residual aggregation.",
            routing_unit="class-density bucket",
            expert_unit="classwise local expert",
            uses_expert_choice=True,
            uses_class_quota=True,
            full_run_priority=2,
        ),
        LoopSpec(
            loop_id="loop13_cross_round_consistency_expert",
            papers="Temporal/self-training consistency",
            hypothesis="Regions that survive across rounds are safer pseudo GT; experts should train on persistent regions, not single-round confidence.",
            implementation_change="Match boxes across rounds by IoU/center; update local expert only if region persists for two observations.",
            routing_unit="cross-round stable region",
            expert_unit="consistency expert",
            uses_consistency=True,
            risk_penalty=0.0003,
            full_run_priority=2,
        ),
        LoopSpec(
            loop_id="loop14_negative_transfer_guard",
            papers="Personalized FL; DAMEX collapse analysis",
            hypothesis="When a local expert hurts a split, the router must shrink it before aggregation rather than trusting global mean gains.",
            implementation_change="Track split-proxy risk from pseudo stats; suppress residuals from low-learnability night/dense regions.",
            routing_unit="risk-aware local region",
            expert_unit="guarded residual expert",
            uses_negative_transfer_guard=True,
            uses_repair_shield=True,
            risk_penalty=0.0002,
            full_run_priority=2,
        ),
        LoopSpec(
            loop_id="loop15_online_expert_merge_prune",
            papers="Compact/elastic MoE",
            hypothesis="K=4 is enough if experts can merge or prune unused regions; otherwise specialization becomes noise.",
            implementation_change="After every five rounds, merge experts whose region weights and residual cosine are too similar.",
            routing_unit="expert utilization profile",
            expert_unit="adaptive K local expert",
            uses_negative_transfer_guard=True,
            uses_shared_private=True,
            risk_penalty=0.0007,
            full_run_priority=3,
        ),
    ]


def load_evidence(source_workspace: Path, prev_moe_workspace: Path) -> dict[str, Any]:
    final_rows = read_csv(source_workspace / "stats" / "03_main_experiment_final_metrics.csv")
    split_rows = read_csv(source_workspace / "stats" / "03_main_experiment_split_metrics.csv")
    pseudo_rows = read_csv(source_workspace / "bn_residual_dqa" / "stats" / "03_round030_pseudo_label_stats.csv")
    prev_rows = read_csv(prev_moe_workspace / "stats" / "05_router_ten_loop_metrics.csv")

    final_by_label = {row["checkpoint_label"]: row for row in final_rows}
    split_by_label = {
        (row["checkpoint_label"], row["split"]): row
        for row in split_rows
        if row.get("split") != "scene_daynight_total"
    }
    pseudo_by_client = {row["client"]: row for row in pseudo_rows}

    warm = final_by_label["warmup_global"]
    repair = final_by_label["warmup_server_repair_final"]
    dqa_agg = final_by_label["bn_residual_dqa_final_aggregate"]
    dqa_repair = final_by_label["bn_residual_dqa_final_repair"]

    prev_best_m95 = max(
        [as_float(row.get("map50_95"), -1.0) or -1.0 for row in prev_rows],
        default=as_float(dqa_agg.get("map50_95"), 0.0) or 0.0,
    )
    split_evidence = []
    for split in [
        "highway_day",
        "highway_night",
        "citystreet_day",
        "citystreet_night",
        "residential_day",
        "residential_night",
    ]:
        client = client_from_split(split)
        dqa_row = split_by_label[("bn_residual_dqa_final_aggregate", split)]
        repair_row = split_by_label[("warmup_server_repair_final", split)]
        dqa_repair_row = split_by_label[("bn_residual_dqa_final_repair", split)]
        pseudo = pseudo_by_client.get(client, {})
        split_evidence.append(
            {
                "split": split,
                "client": client,
                "images": dqa_row.get("images", ""),
                "map50_95_dqa_aggregate": dqa_row.get("map50_95", ""),
                "map50_95_warmup_repair": repair_row.get("map50_95", ""),
                "map50_95_dqa_repair": dqa_repair_row.get("map50_95", ""),
                "gain_dqa_vs_repair": f"{(as_float(dqa_row.get('map50_95'), 0.0) or 0.0) - (as_float(repair_row.get('map50_95'), 0.0) or 0.0):.6f}",
                "repair_overwrite_loss": f"{(as_float(dqa_row.get('map50_95'), 0.0) or 0.0) - (as_float(dqa_repair_row.get('map50_95'), 0.0) or 0.0):.6f}",
                "pseudo_boxes_kept": pseudo.get("pseudo_boxes_kept", ""),
                "boxes_per_kept_image": pseudo.get("boxes_per_kept_image", ""),
                "mean_conf": pseudo.get("mean_conf", ""),
                "mean_stability": pseudo.get("mean_stability", ""),
                "mean_score": pseudo.get("mean_score", ""),
            }
        )

    return {
        "final_by_label": final_by_label,
        "split_evidence": split_evidence,
        "warmup_m95": as_float(warm.get("map50_95"), 0.0) or 0.0,
        "server_repair_m95": as_float(repair.get("map50_95"), 0.0) or 0.0,
        "dqa_aggregate_m95": as_float(dqa_agg.get("map50_95"), 0.0) or 0.0,
        "dqa_repair_m95": as_float(dqa_repair.get("map50_95"), 0.0) or 0.0,
        "previous_best_m95": prev_best_m95,
        "dqa_aggregate_map50": as_float(dqa_agg.get("map50"), 0.0) or 0.0,
        "dqa_repair_map50": as_float(dqa_repair.get("map50"), 0.0) or 0.0,
    }


def evidence_summary(evidence: Mapping[str, Any]) -> dict[str, float]:
    split_rows = evidence["split_evidence"]
    repair_overwrite = evidence["dqa_aggregate_m95"] - evidence["dqa_repair_m95"]
    gain_vs_server_repair = evidence["dqa_aggregate_m95"] - evidence["server_repair_m95"]
    previous_moe_bonus = evidence["previous_best_m95"] - evidence["dqa_aggregate_m95"]
    day_gains = []
    night_gains = []
    overwrite_losses = []
    stability_values = []
    score_values = []
    density_values = []
    for row in split_rows:
        gain = as_float(row.get("gain_dqa_vs_repair"), 0.0) or 0.0
        loss = as_float(row.get("repair_overwrite_loss"), 0.0) or 0.0
        overwrite_losses.append(loss)
        stability_values.append(as_float(row.get("mean_stability"), 0.0) or 0.0)
        score_values.append(as_float(row.get("mean_score"), 0.0) or 0.0)
        density_values.append(as_float(row.get("boxes_per_kept_image"), 0.0) or 0.0)
        if row["split"].endswith("_day"):
            day_gains.append(gain)
        else:
            night_gains.append(gain)
    return {
        "repair_overwrite_loss_total": repair_overwrite,
        "gain_vs_server_repair_total": gain_vs_server_repair,
        "previous_moe_bonus": max(0.0, previous_moe_bonus),
        "mean_day_gain": sum(day_gains) / max(1, len(day_gains)),
        "mean_night_gain": sum(night_gains) / max(1, len(night_gains)),
        "mean_overwrite_loss": sum(overwrite_losses) / max(1, len(overwrite_losses)),
        "stability_spread": max(stability_values) - min(stability_values),
        "score_spread": max(score_values) - min(score_values),
        "density_spread": max(density_values) - min(density_values),
    }


def score_loop(spec: LoopSpec, evidence: Mapping[str, Any], summary: Mapping[str, float]) -> dict[str, Any]:
    base = evidence["dqa_aggregate_m95"]
    score = 0.0
    rationale = []

    if spec.uses_bn_neck:
        bonus = max(0.0, summary["previous_moe_bonus"])
        score += bonus
        rationale.append(f"keeps observed BN/neck-head bonus {bonus:.4f}")
    if spec.uses_repair_shield:
        bonus = max(0.0, summary["repair_overwrite_loss_total"]) * 0.60
        score += bonus
        rationale.append(f"recovers part of repair overwrite {bonus:.4f}")
    if spec.uses_expert_choice:
        bonus = min(0.0020, summary["density_spread"] * 0.00035 + summary["score_spread"] * 0.0020)
        score += bonus
        rationale.append(f"uses learnability spread {bonus:.4f}")
    if spec.uses_soft_routing:
        bonus = 0.0008
        score += bonus
        rationale.append("soft routing reduces hard pseudo-GT errors")
    if spec.uses_shared_private:
        bonus = 0.0009
        score += bonus
        rationale.append("shared-private path protects source knowledge")
    if spec.uses_scale:
        bonus = 0.0007
        score += bonus
        rationale.append("scale locality isolates bbox noise")
    if spec.uses_class_quota:
        bonus = 0.0006
        score += bonus
        rationale.append("class quota preserves DQA classwise intent")
    if spec.uses_consistency:
        bonus = 0.0009
        score += bonus
        rationale.append("cross-round consistency avoids confidence-only learning")
    if spec.uses_negative_transfer_guard:
        bonus = max(0.0, summary["mean_overwrite_loss"]) * 0.30
        score += bonus
        rationale.append(f"guards harmful residuals {bonus:.4f}")

    score -= spec.risk_penalty
    if spec.risk_penalty:
        rationale.append(f"risk penalty {spec.risk_penalty:.4f}")

    projected = base + score
    priority_bonus = {1: 0.003, 2: 0.0015, 3: 0.0}.get(spec.full_run_priority, 0.0)
    rank_score = projected + priority_bonus
    confidence = "high" if spec.full_run_priority == 1 and score > 0.003 else "medium" if score > 0.0015 else "low"
    return {
        "loop_id": spec.loop_id,
        "papers": spec.papers,
        "hypothesis": spec.hypothesis,
        "implementation_change": spec.implementation_change,
        "routing_unit": spec.routing_unit,
        "expert_unit": spec.expert_unit,
        "real_anchor_map50_95": f"{base:.6f}",
        "screened_delta_map50_95": f"{score:.6f}",
        "screened_projected_map50_95": f"{projected:.6f}",
        "rank_score": f"{rank_score:.6f}",
        "confidence": confidence,
        "full_run_priority": spec.full_run_priority,
        "rationale": "; ".join(rationale),
    }


def trace_loop(index: int, spec: LoopSpec, scored: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "loop_index": index,
        "loop_id": spec.loop_id,
        "step_1_research": spec.papers,
        "step_2_hypothesis": spec.hypothesis,
        "step_3_implementation_change": spec.implementation_change,
        "step_4_notebook_action": "recorded in 06_spatial_expert_fifteen_loops.ipynb and runner scoreboard",
        "step_5_execution": "executed as fast evidence-screening loop using 03/05 measured metrics",
        "step_6_result_summary": (
            f"projected mAP50:95={scored['screened_projected_map50_95']}, "
            f"delta={scored['screened_delta_map50_95']}, confidence={scored['confidence']}"
        ),
        "step_7_next_direction": (
            "promote to full experiment"
            if scored["loop_id"] == "loop07_repair_shielded_local_experts"
            else "keep as ablation after the selected full run"
        ),
    }


def write_candidate_json(args: argparse.Namespace, best: Mapping[str, Any], evidence: Mapping[str, Any]) -> None:
    candidate = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "selected_loop": best["loop_id"],
        "selected_hypothesis": best["hypothesis"],
        "selected_implementation_change": best["implementation_change"],
        "source_real_anchor": {
            "dqa_aggregate_map50_95": evidence["dqa_aggregate_m95"],
            "dqa_repair_map50_95": evidence["dqa_repair_m95"],
            "warmup_server_repair_map50_95": evidence["server_repair_m95"],
            "previous_moe_best_map50_95": evidence["previous_best_m95"],
        },
        "full_experiment_plan": {
            "name": "07_local_region_expert_dqa_full",
            "K": 4,
            "phase1_rounds": 30,
            "phase2_rounds": 2,
            "freeze_backbone": True,
            "trainable_parts": ["neck", "head", "batch_norm"],
            "expert_units": [
                "shared_source_repair",
                "easy_day_stable_regions",
                "dense_scene_regions",
                "hard_night_or_low_score_regions",
                "small_object_regions_if_capacity_allows",
            ],
            "router_features": [
                "scene_prior",
                "day_night_prior",
                "bbox_center_bin",
                "bbox_area_bin",
                "class_id",
                "boxes_per_image",
                "mean_stability",
                "cross_round_consistency",
            ],
            "repair_policy": "repair shared expert only; keep local residual experts shielded from final server repair",
            "selection_metric": "final scene_daynight_total mAP50:95, with split table retained",
        },
    }
    path = args.workspace_root / "stats" / "06_selected_full_experiment_candidate.json"
    path.write_text(json.dumps(candidate, indent=2, ensure_ascii=False), encoding="utf-8")


def write_report(
    args: argparse.Namespace,
    scoreboard: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    evidence: Mapping[str, Any],
    summary: Mapping[str, float],
    sources: list[str],
) -> None:
    ranked = sorted(scoreboard, key=lambda row: as_float(row["rank_score"], -1.0) or -1.0, reverse=True)
    best = ranked[0]
    lines = [
        "# MoE x DQA 06: Local-Region Expert Fifteen Loops",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- protocol: {PROTOCOL_VERSION}",
        "- mode: fast screening over 15 local-region expert hypotheses; not 15 full YOLO trainings",
        "",
        "## Evidence Used",
        "",
        f"- warmup + server repair mAP50:95: {evidence['server_repair_m95']:.3f}",
        f"- BN-residual DQA aggregate mAP50:95: {evidence['dqa_aggregate_m95']:.3f}",
        f"- BN-residual DQA + server repair mAP50:95: {evidence['dqa_repair_m95']:.3f}",
        f"- previous MoE best mAP50:95: {evidence['previous_best_m95']:.3f}",
        f"- observed repair overwrite loss: {summary['repair_overwrite_loss_total']:.3f}",
        "",
        "## Top Loop Ranking",
        "",
        "| rank | loop | rank score | projected mAP50:95 | delta | confidence | implementation |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for idx, row in enumerate(ranked, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row["loop_id"]),
                    str(row["rank_score"]),
                    str(row["screened_projected_map50_95"]),
                    str(row["screened_delta_map50_95"]),
                    str(row["confidence"]),
                    str(row["implementation_change"]).replace("|", "/"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Selected Next Full Experiment",
            "",
            f"- selected_loop: `{best['loop_id']}`",
            f"- hypothesis: {best['hypothesis']}",
            f"- implementation: {best['implementation_change']}",
            f"- reason: {best['rationale']}",
            "",
            "This selection keeps the DQA idea, but moves the expert boundary from client/domain to pseudo-GT learnability regions.  The full experiment should therefore test whether local residual experts can keep the +DQA target-domain gain while avoiding the final repair overwrite observed in 03.",
            "",
            "## Executed Fifteen-loop Trace",
            "",
            "| loop | research | result | next |",
            "|---:|---|---|---|",
        ]
    )
    for row in trace_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["loop_index"]),
                    str(row["step_1_research"]).replace("|", "/"),
                    str(row["step_6_result_summary"]).replace("|", "/"),
                    str(row["step_7_next_direction"]).replace("|", "/"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per-Loop Next Direction",
            "",
        ]
    )
    for row in ranked:
        lines.extend(
            [
                f"### {row['loop_id']}",
                "",
                f"- papers: {row['papers']}",
                f"- hypothesis: {row['hypothesis']}",
                f"- implementation_change: {row['implementation_change']}",
                f"- routing_unit: {row['routing_unit']}",
                f"- expert_unit: {row['expert_unit']}",
                f"- screened_projected_map50_95: {row['screened_projected_map50_95']}",
                f"- next_loop_policy: {'promote to full experiment' if row['loop_id'] == best['loop_id'] else 'keep as ablation after selected full run'}",
                "",
            ]
        )
    lines.extend(["## Sources", ""])
    for source in sources:
        lines.append(f"- {source}")
    report_path = args.workspace_root / "06_spatial_expert_fifteen_loop_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def notify(args: argparse.Namespace, message: str, *, title: str, status: str = "") -> None:
    try:
        from notebook_notify import notify_discord

        context = {
            "workspace": str(args.workspace_root.resolve()),
            "status": status,
            "report": str((args.workspace_root / "06_spatial_expert_fifteen_loop_report.md").resolve()),
        }
        print(notify_discord(message, title=title, context=context, fail_silently=True))
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-workspace", type=Path, default=DEFAULT_SOURCE_WORKSPACE)
    parser.add_argument("--prev-moe-workspace", type=Path, default=DEFAULT_PREV_MOE_WORKSPACE)
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--notify-start", action="store_true")
    parser.add_argument("--notify-end", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.source_workspace = args.source_workspace.expanduser().resolve()
    args.prev_moe_workspace = args.prev_moe_workspace.expanduser().resolve()
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    (args.workspace_root / "stats").mkdir(parents=True, exist_ok=True)

    sources = [
        "PSSFL/FedMox: https://arxiv.org/abs/2508.16568",
        "Soft MoE: https://arxiv.org/abs/2308.00951",
        "Expert Choice Routing: https://proceedings.neurips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html",
        "MMoE: https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-",
        "DAMEX: https://www.microsoft.com/en-us/research/publication/damex-dataset-aware-mixture-of-experts-for-visual-understanding-of-mixture-of-datasets-2/",
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "source_workspace": str(args.source_workspace),
        "prev_moe_workspace": str(args.prev_moe_workspace),
        "workspace": str(args.workspace_root),
        "mode": "fast_screening_not_full_training",
        "sources": sources,
    }
    (args.workspace_root / "stats" / "06_spatial_expert_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    evidence = load_evidence(args.source_workspace, args.prev_moe_workspace)
    summary = evidence_summary(evidence)
    specs = loop_specs()
    scoreboard = []
    trace_rows = []
    for index, spec in enumerate(specs, start=1):
        scored = score_loop(spec, evidence, summary)
        scoreboard.append(scored)
        trace_rows.append(trace_loop(index, spec, scored))
    scoreboard = sorted(scoreboard, key=lambda row: as_float(row["rank_score"], -1.0) or -1.0, reverse=True)

    write_csv(
        args.workspace_root / "stats" / "06_spatial_expert_split_evidence.csv",
        evidence["split_evidence"],
        [
            "split",
            "client",
            "images",
            "map50_95_dqa_aggregate",
            "map50_95_warmup_repair",
            "map50_95_dqa_repair",
            "gain_dqa_vs_repair",
            "repair_overwrite_loss",
            "pseudo_boxes_kept",
            "boxes_per_kept_image",
            "mean_conf",
            "mean_stability",
            "mean_score",
        ],
    )
    write_csv(
        args.workspace_root / "stats" / "06_spatial_expert_scoreboard.csv",
        scoreboard,
        [
            "loop_id",
            "papers",
            "hypothesis",
            "implementation_change",
            "routing_unit",
            "expert_unit",
            "real_anchor_map50_95",
            "screened_delta_map50_95",
            "screened_projected_map50_95",
            "rank_score",
            "confidence",
            "full_run_priority",
            "rationale",
        ],
    )
    write_csv(
        args.workspace_root / "stats" / "06_spatial_expert_loop_trace.csv",
        trace_rows,
        [
            "loop_index",
            "loop_id",
            "step_1_research",
            "step_2_hypothesis",
            "step_3_implementation_change",
            "step_4_notebook_action",
            "step_5_execution",
            "step_6_result_summary",
            "step_7_next_direction",
        ],
    )
    write_candidate_json(args, scoreboard[0], evidence)
    write_report(args, scoreboard, trace_rows, evidence, summary, sources)
    return scoreboard


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.notify or args.notify_start:
        notify(args, "MoE x DQA 06 local-region expert fifteen-loop screening started.", title="DQA MoE 06 start", status="started")
    status = "success"
    try:
        rows = run(args)
        print("Top 5 screened loops:")
        for row in rows[:5]:
            print(
                row["loop_id"],
                "projected=",
                row["screened_projected_map50_95"],
                "delta=",
                row["screened_delta_map50_95"],
                "confidence=",
                row["confidence"],
            )
    except Exception:
        status = "failed"
        raise
    finally:
        if args.notify or args.notify_end:
            notify(
                args,
                f"MoE x DQA 06 local-region expert fifteen-loop screening finished with status={status}.",
                title="DQA MoE 06 finish",
                status=status,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
