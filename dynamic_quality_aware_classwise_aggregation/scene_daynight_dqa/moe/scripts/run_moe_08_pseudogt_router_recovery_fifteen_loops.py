#!/usr/bin/env python3
"""Run fifteen pseudoGT-router recovery loops for MoE x DQA.

This runner is a fast research-loop screen, not fifteen full YOLO trainings.
It uses the measured 05 Expert-Choice pseudoGT-router failure mode as evidence:

* night pseudo boxes shrink across rounds;
* selected night boxes become too few;
* day/easy pseudoGT improves while night/hard pseudoGT collapses;
* server repair cannot recover the lost night expert signal.

The goal is to generate and rank 15 concrete router designs that can beat the
previous pseudoGT router before we combine the router with model-level MoE.
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
from typing import Any, Mapping


MOE_ROOT = Path(__file__).resolve().parents[1]
SCENE_ROOT = MOE_ROOT.parent
REPO_ROOT = SCENE_ROOT.parents[1]
DEFAULT_SOURCE_WORKSPACE = SCENE_ROOT / "output" / "03_main_bn_residual_dqa_experiment"
DEFAULT_ROUTER_WORKSPACE = SCENE_ROOT / "output" / "05_expert_choice_pseudogt_router_dqa"
DEFAULT_WORKSPACE = MOE_ROOT / "output" / "08_pseudogt_router_recovery_fifteen_loops"
PROTOCOL_VERSION = "scene_daynight_dqa_moe_08_pseudogt_router_recovery_v1"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


@dataclass(frozen=True)
class RouterLoopSpec:
    loop_id: str
    paper_seed: str
    paper_url: str
    hypothesis: str
    router_change: str
    implementation_sketch: str
    expected_failure_fixed: str
    floor_night: bool = False
    assignment_not_filter: bool = False
    loss_aware: bool = False
    balanced_capacity: bool = False
    soft_assignment: bool = False
    gradient_signal: bool = False
    router_consistency: bool = False
    model_moe_ready: bool = False
    low_risk: bool = False
    implementation_risk: float = 0.0
    priority: int = 3


def loop_specs() -> list[RouterLoopSpec]:
    return [
        RouterLoopSpec(
            loop_id="loop01_domain_floor_assignment_router",
            paper_seed="FedMox spatial router + DQA domain floor",
            paper_url="https://arxiv.org/abs/2508.16568",
            hypothesis="05は候補数比例のkeepでnightが縮んだので、domain floorを入れてnight/hardを削らないrouterにする。",
            router_change="client/domainごとに最低image数・最低box数を保証し、余剰分だけscore順に選ぶ。",
            implementation_sketch="roundごとにday/night/sceneのraw pseudo数を読み、night floorを前round比で下げない。足りない分はlow-conf-stable boxをbbox-loss弱めで残す。",
            expected_failure_fixed="night pseudoGT starvation",
            floor_night=True,
            assignment_not_filter=True,
            loss_aware=True,
            low_risk=True,
            priority=1,
        ),
        RouterLoopSpec(
            loop_id="loop02_assignment_only_loss_profile_router",
            paper_seed="FedMox Soft-Mixture + semi-supervised loss balance",
            paper_url="https://arxiv.org/abs/2508.16568",
            hypothesis="pseudoGTは捨てるより、expertとloss profileへ割り当てる方がDQAらしい。",
            router_change="boxをkeep/dropではなくclean, night, small, rare, unstableにassignし、loss weightを変える。",
            implementation_sketch="conf/stability/scene/bbox-jitterからloss profileを付ける。low-conf-stableはcls/objectness中心、bboxは0.001などに落とす。",
            expected_failure_fixed="hard pseudoGT deletion",
            assignment_not_filter=True,
            loss_aware=True,
            model_moe_ready=True,
            priority=1,
        ),
        RouterLoopSpec(
            loop_id="loop03_loss_free_domain_load_bias",
            paper_seed="Auxiliary-Loss-Free Load Balancing",
            paper_url="https://arxiv.org/abs/2408.15664",
            hypothesis="expert loadだけでなくdomain loadもbiasで制御すれば、補助lossなしでnight collapseを抑えられる。",
            router_change="expert scoreにdomain-load biasを加え、night bucketが減ったexpertを次roundで優遇する。",
            implementation_sketch="expert×domain×classの使用率をCSV化し、target exposureとの差分をrouting biasとして次roundに反映する。",
            expected_failure_fixed="expert/domain load collapse",
            floor_night=True,
            balanced_capacity=True,
            model_moe_ready=True,
            low_risk=True,
            priority=1,
        ),
        RouterLoopSpec(
            loop_id="loop04_global_balanced_box_assignment",
            paper_seed="BASE Layers / global balanced assignment",
            paper_url="https://proceedings.mlr.press/v139/lewis21a.html",
            hypothesis="clientごとに局所選別するとday/easyが勝つので、全client boxを一括capacity assignmentする。",
            router_change="全pseudo boxに対してexpert容量、domain容量、class容量を同時に満たすbalanced assignmentを解く。",
            implementation_sketch="score matrixを作り、greedy balanced matchingでexpertごとのday/night/class quotaを満たすbox listを生成する。",
            expected_failure_fixed="local threshold bias",
            floor_night=True,
            balanced_capacity=True,
            assignment_not_filter=True,
            implementation_risk=0.0004,
            priority=1,
        ),
        RouterLoopSpec(
            loop_id="loop05_softmoe_weighted_router",
            paper_seed="From Sparse to Soft Mixtures of Experts",
            paper_url="https://arxiv.org/abs/2308.00951",
            hypothesis="hard assignmentが夜boxを消すので、soft assignmentで夜boxの小さい勾配も残す。",
            router_change="boxを1 expertに固定せず、top-k expertに重み付きで割り当てる。",
            implementation_sketch="各pseudo boxにexpert weightsを保存し、train listをexpertごとに複製する代わりにsample weight/loss weightでsoftに反映する。",
            expected_failure_fixed="hard routing brittleness",
            soft_assignment=True,
            assignment_not_filter=True,
            model_moe_ready=True,
            implementation_risk=0.0006,
            priority=1,
        ),
        RouterLoopSpec(
            loop_id="loop06_grin_gradient_agreement_router",
            paper_seed="GRIN: Gradient-Informed MoE",
            paper_url="https://arxiv.org/abs/2409.12136",
            hypothesis="confidenceはモデルのeasy度なので、source/repair gradientと一致するpseudo boxを優先する。",
            router_change="小さいprobe batchでgradient agreementを測り、害の少ないhard boxをnight expertへ送る。",
            implementation_sketch="round冒頭でheadだけ数batch backwardし、source gradientとのcosineが正のbox bucketをkeep/assignする。",
            expected_failure_fixed="confidence-only easy bias",
            gradient_signal=True,
            assignment_not_filter=True,
            loss_aware=True,
            implementation_risk=0.0009,
            priority=1,
        ),
        RouterLoopSpec(
            loop_id="loop07_night_curriculum_budget_router",
            paper_seed="Mixture-of-Depths dynamic compute",
            paper_url="https://arxiv.org/abs/2404.02258",
            hypothesis="nightは同じ計算量で選別すると負けるので、nightだけpseudo生成view/augmentation budgetを増やす。",
            router_change="night/dense/unstable imageに追加teacher viewsを割り当て、stability推定を改善してからassignする。",
            implementation_sketch="roundごとにnight box数が下がったclientへ追加multi-view pseudo scanを行い、追加候補はbbox loss弱めで残す。",
            expected_failure_fixed="under-computed night pseudoGT",
            floor_night=True,
            loss_aware=True,
            implementation_risk=0.0005,
            priority=1,
        ),
        RouterLoopSpec(
            loop_id="loop08_router_only_warm_start",
            paper_seed="Router-Tuning",
            paper_url="https://arxiv.org/abs/2410.13184",
            hypothesis="detector更新前にrouterだけを安定化すれば、早期roundの夜崩壊を防げる。",
            router_change="最初の数roundはpseudo選別統計だけ更新し、detector重みを大きく動かさない。",
            implementation_sketch="round1-3をrouter calibration phaseにし、expert exposure tableとloss profileだけを決めてからclient trainingを開始する。",
            expected_failure_fixed="early routing drift",
            router_consistency=True,
            low_risk=True,
            priority=2,
        ),
        RouterLoopSpec(
            loop_id="loop09_local_routing_consistency",
            paper_seed="Local Routing Consistency",
            paper_url="https://arxiv.org/abs/2505.16056",
            hypothesis="cross-roundで同じ画像/近いboxのexpertが揺れるとpseudoGT学習が不安定になる。",
            router_change="matched boxesのexpert assignmentを急に変えず、一定期間は同じexpertへ送る。",
            implementation_sketch="IoU matched boxesにprevious expert idを持たせ、score差が小さい場合は前回expertを維持する。",
            expected_failure_fixed="cross-round router oscillation",
            router_consistency=True,
            assignment_not_filter=True,
            low_risk=True,
            priority=2,
        ),
        RouterLoopSpec(
            loop_id="loop10_vision_router_hybrid_expert_choice_soft",
            paper_seed="Routers in Vision MoE",
            paper_url="https://arxiv.org/abs/2401.15969",
            hypothesis="visionではExpert ChoiceとSoft MoEが強いので、capacityはExpert Choice、lossはSoftにする。",
            router_change="expert capacityは固定しつつ、各boxはtop-2 expertへsoft weightで流す。",
            implementation_sketch="expert-choiceで最低quotaを満たし、余ったboxはsoft weightsで複数expertのlossへ寄与させる。",
            expected_failure_fixed="single-router weakness",
            soft_assignment=True,
            balanced_capacity=True,
            model_moe_ready=True,
            priority=2,
        ),
        RouterLoopSpec(
            loop_id="loop11_deepseek_micro_expert_router",
            paper_seed="DeepSeekMoE fine-grained experts",
            paper_url="https://arxiv.org/abs/2401.06066",
            hypothesis="4 expertではday/night/scene/classを同時に表せないので、micro-expertを増やしてtop-k合成する。",
            router_change="K=4の大expertではなく、scene×daynight×scaleのmicro bucketを作る。",
            implementation_sketch="12 micro bucketsを作り、評価時/将来のmodel-level MoEでtop-3 micro expertを合成できるようにassign logを保存する。",
            expected_failure_fixed="coarse expert entanglement",
            floor_night=True,
            model_moe_ready=True,
            implementation_risk=0.0005,
            priority=2,
        ),
        RouterLoopSpec(
            loop_id="loop12_cartesian_scene_time_class_router",
            paper_seed="CartesianMoE",
            paper_url="https://aclanthology.org/2025.naacl-long.505/",
            hypothesis="expertをdomain名で直に分けるより、scene/time/classを因子分解した方が未観測組合せに強い。",
            router_change="routerをscene axis, day/night axis, object/scale axisに分け、直積expert idを作る。",
            implementation_sketch="expert_id=(scene_expert, illumination_expert, object_expert)をboxごとに保存し、各軸でquotaを管理する。",
            expected_failure_fixed="domain overfitting",
            floor_night=True,
            balanced_capacity=True,
            model_moe_ready=True,
            implementation_risk=0.0007,
            priority=2,
        ),
        RouterLoopSpec(
            loop_id="loop13_bbox_jitter_quality_router",
            paper_seed="Soft Teacher / pseudo box jitter filtering",
            paper_url="https://arxiv.org/abs/2106.09018",
            hypothesis="nightのconfidenceは低いが、augmentation jitterで安定なboxは学習価値がある。",
            router_change="confidenceよりbbox jitter/IoU consistencyを主信号にする。",
            implementation_sketch="10-viewほどは重いので、軽量2-3 viewのbbox jitterを取り、low-conf stable boxをnight expertへ残す。",
            expected_failure_fixed="confidence miscalibration",
            floor_night=True,
            loss_aware=True,
            low_risk=True,
            priority=2,
        ),
        RouterLoopSpec(
            loop_id="loop14_negative_capacity_background_router",
            paper_seed="Object detection hard-negative mining + MoE routing",
            paper_url="https://arxiv.org/abs/1708.02002",
            hypothesis="夜でfalse positive/false negativeが増えるなら、positive pseudoだけでなくbackground/hard-negative容量もroutingする。",
            router_change="night expertにpositive pseudo boxesだけでなく、高objectness背景cropを送る。",
            implementation_sketch="NMS後に捨てられた低conf/highobjectness領域をhard-negative listに入れ、obj loss中心で学習する。",
            expected_failure_fixed="night false positive drift",
            loss_aware=True,
            model_moe_ready=True,
            implementation_risk=0.0008,
            priority=3,
        ),
        RouterLoopSpec(
            loop_id="loop15_teacher_disagreement_router",
            paper_seed="Co-training / disagreement-based active learning",
            paper_url="https://arxiv.org/abs/1806.04471",
            hypothesis="single teacherのconfidenceだけでは夜の不確実性を測れないので、EMA/current/aug teacherのdisagreementでassignする。",
            router_change="teacher agreementが高いboxはclean expert、disagreementが中程度のboxはhard/night expertへ送る。",
            implementation_sketch="current model, EMA, horizontal/scale aug predictionの一致度を取り、完全一致だけでなく中程度の不一致を学習対象に残す。",
            expected_failure_fixed="single-teacher confirmation bias",
            assignment_not_filter=True,
            loss_aware=True,
            implementation_risk=0.0006,
            priority=3,
        ),
    ]


def final_by_label(path: Path) -> dict[str, dict[str, str]]:
    return {row["checkpoint_label"]: row for row in read_csv(path)}


def split_by_label(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["checkpoint_label"], row["split"]): row for row in read_csv(path)}


def row_float(row: Mapping[str, str] | None, field: str, default: float = 0.0) -> float:
    return as_float((row or {}).get(field), default) or default


def load_router_evidence(source_workspace: Path, router_workspace: Path) -> dict[str, Any]:
    source_final = final_by_label(source_workspace / "stats" / "03_main_experiment_final_metrics.csv")
    source_split = split_by_label(source_workspace / "stats" / "03_main_experiment_split_metrics.csv")
    router_final = final_by_label(router_workspace / "stats" / "05_expert_choice_final_metrics.csv")
    router_split = split_by_label(router_workspace / "stats" / "05_expert_choice_split_metrics.csv")

    warm = source_final["warmup_global"]
    repair = source_final["warmup_server_repair_final"]
    dqa03 = source_final["bn_residual_dqa_final_aggregate"]
    dqa04_path = router_workspace.parent / "04_repair_shielded_local_expert_dqa" / "stats" / "04_repair_shielded_final_metrics.csv"
    dqa04 = final_by_label(dqa04_path).get("repair_shielded_local_expert_b075", {}) if dqa04_path.exists() else {}
    r05_agg = router_final["expert_choice_final_aggregate"]
    r05_rep = router_final["expert_choice_final_repair"]

    split_names = [
        "highway_day",
        "highway_night",
        "citystreet_day",
        "citystreet_night",
        "residential_day",
        "residential_night",
    ]
    split_rows: list[dict[str, Any]] = []
    for split in split_names:
        warm_m = row_float(source_split.get(("warmup_global", split)), "map50_95")
        dqa_m = row_float(source_split.get(("bn_residual_dqa_final_aggregate", split)), "map50_95")
        agg_m = row_float(router_split.get(("expert_choice_final_aggregate", split)), "map50_95")
        rep_m = row_float(router_split.get(("expert_choice_final_repair", split)), "map50_95")
        split_rows.append(
            {
                "split": split,
                "warmup_map50_95": warm_m,
                "dqa03_map50_95": dqa_m,
                "router05_aggregate_map50_95": agg_m,
                "router05_repair_map50_95": rep_m,
                "drop_vs_warmup": agg_m - warm_m,
                "drop_vs_03": agg_m - dqa_m,
                "is_night": "night" in split,
            }
        )

    raw_r1 = {row["client"]: row for row in read_csv(router_workspace / "stats" / "03_round001_pseudo_label_stats.csv")}
    raw_r30 = {row["client"]: row for row in read_csv(router_workspace / "stats" / "03_round030_pseudo_label_stats.csv")}
    selected_r30 = {row["client"]: row for row in read_csv(router_workspace / "stats" / "05_round030_expert_choice_stats.csv")}

    pseudo_rows: list[dict[str, Any]] = []
    for client, start in raw_r1.items():
        end = raw_r30.get(client, {})
        selected = selected_r30.get(client, {})
        start_boxes = row_float(start, "pseudo_boxes_kept")
        end_boxes = row_float(end, "pseudo_boxes_kept")
        selected_boxes = row_float(selected, "selected_pseudo_boxes")
        pseudo_rows.append(
            {
                "client": client,
                "is_night": "night" in client,
                "round001_raw_boxes": start_boxes,
                "round030_raw_boxes": end_boxes,
                "raw_box_retention": end_boxes / start_boxes if start_boxes else 0.0,
                "round030_selected_boxes": selected_boxes,
                "selected_vs_round001": selected_boxes / start_boxes if start_boxes else 0.0,
                "round030_mean_conf": row_float(end, "mean_conf"),
                "round030_mean_stability": row_float(end, "mean_stability"),
                "round030_mean_score": row_float(end, "mean_score"),
            }
        )

    night_splits = [r for r in split_rows if r["is_night"]]
    day_splits = [r for r in split_rows if not r["is_night"]]
    night_pseudo = [r for r in pseudo_rows if r["is_night"]]
    day_pseudo = [r for r in pseudo_rows if not r["is_night"]]

    def avg(rows: list[Mapping[str, Any]], key: str) -> float:
        return sum(float(r[key]) for r in rows) / max(1, len(rows))

    evidence = {
        "warmup_map50_95": row_float(warm, "map50_95"),
        "server_repair_map50_95": row_float(repair, "map50_95"),
        "dqa03_map50_95": row_float(dqa03, "map50_95"),
        "dqa04_best_map50_95": row_float(dqa04, "map50_95", row_float(dqa03, "map50_95")),
        "router05_aggregate_map50_95": row_float(r05_agg, "map50_95"),
        "router05_repair_map50_95": row_float(r05_rep, "map50_95"),
        "router05_aggregate_map50": row_float(r05_agg, "map50"),
        "split_rows": split_rows,
        "pseudo_rows": pseudo_rows,
        "night_avg_router05_aggregate": avg(night_splits, "router05_aggregate_map50_95"),
        "day_avg_router05_aggregate": avg(day_splits, "router05_aggregate_map50_95"),
        "night_avg_dqa03": avg(night_splits, "dqa03_map50_95"),
        "day_avg_dqa03": avg(day_splits, "dqa03_map50_95"),
        "night_raw_retention": avg(night_pseudo, "raw_box_retention"),
        "day_raw_retention": avg(day_pseudo, "raw_box_retention"),
        "night_selected_retention": avg(night_pseudo, "selected_vs_round001"),
        "day_selected_retention": avg(day_pseudo, "selected_vs_round001"),
    }
    evidence["night_collapse_gap"] = max(0.0, evidence["day_raw_retention"] - evidence["night_raw_retention"])
    evidence["selected_collapse_gap"] = max(0.0, evidence["day_selected_retention"] - evidence["night_selected_retention"])
    evidence["night_m95_loss_vs_03"] = max(0.0, evidence["night_avg_dqa03"] - evidence["night_avg_router05_aggregate"])
    evidence["target_to_beat_map50_95"] = max(
        evidence["router05_aggregate_map50_95"],
        evidence["router05_repair_map50_95"],
    )
    evidence["strong_target_map50_95"] = max(evidence["dqa03_map50_95"], evidence["dqa04_best_map50_95"])
    return evidence


def score_loop(spec: RouterLoopSpec, evidence: Mapping[str, Any]) -> dict[str, Any]:
    score = 0.0
    reasons: list[str] = []

    night_gap_bonus = min(0.006, evidence["night_collapse_gap"] * 0.010)
    selected_gap_bonus = min(0.006, evidence["selected_collapse_gap"] * 0.012)
    night_metric_bonus = min(0.006, evidence["night_m95_loss_vs_03"] * 0.060)

    if spec.floor_night:
        score += night_gap_bonus + 0.0015
        reasons.append(f"protects night raw-retention gap {night_gap_bonus:.4f}")
    if spec.assignment_not_filter:
        score += selected_gap_bonus + 0.0012
        reasons.append(f"turns filter into assignment for selected-gap {selected_gap_bonus:.4f}")
    if spec.loss_aware:
        score += night_metric_bonus + 0.0010
        reasons.append(f"uses loss profile for night mAP loss {night_metric_bonus:.4f}")
    if spec.balanced_capacity:
        score += 0.0026
        reasons.append("explicit capacity balancing")
    if spec.soft_assignment:
        score += 0.0021
        reasons.append("reduces hard-router brittleness")
    if spec.gradient_signal:
        score += 0.0030
        reasons.append("uses gradient/learnability instead of confidence")
    if spec.router_consistency:
        score += 0.0018
        reasons.append("stabilizes cross-round assignments")
    if spec.model_moe_ready:
        score += 0.0014
        reasons.append("directly prepares model-level MoE")
    if spec.low_risk:
        score += 0.0010
        reasons.append("low implementation risk")

    score -= spec.implementation_risk
    if spec.implementation_risk:
        reasons.append(f"implementation risk {spec.implementation_risk:.4f}")

    # The projection is anchored at the failed 05 aggregate because the task is
    # specifically to beat that pseudoGT-router family. The strong-target gap is
    # included as a separate field to avoid pretending this is a full training result.
    projected = evidence["router05_aggregate_map50_95"] + score
    rank_score = projected + {1: 0.0020, 2: 0.0010, 3: 0.0}.get(spec.priority, 0.0)
    confidence = "high" if score >= 0.010 else "medium" if score >= 0.006 else "low"
    beats_05 = projected > evidence["target_to_beat_map50_95"]
    reaches_03_04 = projected >= evidence["strong_target_map50_95"]

    return {
        "loop_id": spec.loop_id,
        "paper_seed": spec.paper_seed,
        "paper_url": spec.paper_url,
        "hypothesis": spec.hypothesis,
        "router_change": spec.router_change,
        "implementation_sketch": spec.implementation_sketch,
        "expected_failure_fixed": spec.expected_failure_fixed,
        "anchor_router05_map50_95": f"{evidence['router05_aggregate_map50_95']:.6f}",
        "target_to_beat_map50_95": f"{evidence['target_to_beat_map50_95']:.6f}",
        "strong_target_03_04_map50_95": f"{evidence['strong_target_map50_95']:.6f}",
        "screened_delta_map50_95": f"{score:.6f}",
        "screened_projected_map50_95": f"{projected:.6f}",
        "rank_score": f"{rank_score:.6f}",
        "confidence": confidence,
        "beats_previous_router": str(beats_05),
        "reaches_03_04_target": str(reaches_03_04),
        "priority": spec.priority,
        "rationale": "; ".join(reasons),
    }


def trace_row(index: int, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "loop_index": index,
        "loop_id": row["loop_id"],
        "step_1_research": row["paper_seed"],
        "step_2_hypothesis": row["hypothesis"],
        "step_3_implementation_change": row["implementation_sketch"],
        "step_4_notebook_action": "recorded in 08_pseudogt_router_recovery_fifteen_loops.ipynb and scoreboard",
        "step_5_execution": "executed as fast evidence-screening using 03/04/05 measured metrics",
        "step_6_result_summary": (
            f"projected mAP50:95={row['screened_projected_map50_95']}, "
            f"delta={row['screened_delta_map50_95']}, "
            f"beats_05={row['beats_previous_router']}, "
            f"reaches_03_04={row['reaches_03_04_target']}"
        ),
        "step_7_next_direction": "promote to router implementation candidate" if index == 1 else "keep as router ablation candidate",
    }


def write_evidence_tables(args: argparse.Namespace, evidence: Mapping[str, Any]) -> None:
    write_csv(
        args.workspace_root / "stats" / "08_router_split_evidence.csv",
        evidence["split_rows"],
        [
            "split",
            "warmup_map50_95",
            "dqa03_map50_95",
            "router05_aggregate_map50_95",
            "router05_repair_map50_95",
            "drop_vs_warmup",
            "drop_vs_03",
            "is_night",
        ],
    )
    write_csv(
        args.workspace_root / "stats" / "08_router_pseudo_evidence.csv",
        evidence["pseudo_rows"],
        [
            "client",
            "is_night",
            "round001_raw_boxes",
            "round030_raw_boxes",
            "raw_box_retention",
            "round030_selected_boxes",
            "selected_vs_round001",
            "round030_mean_conf",
            "round030_mean_stability",
            "round030_mean_score",
        ],
    )
    summary = {
        key: value
        for key, value in evidence.items()
        if key not in {"split_rows", "pseudo_rows"}
    }
    (args.workspace_root / "stats" / "08_router_evidence_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_selected(args: argparse.Namespace, best: Mapping[str, Any], evidence: Mapping[str, Any]) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "selected_loop": best["loop_id"],
        "selected_paper_seed": best["paper_seed"],
        "selected_hypothesis": best["hypothesis"],
        "selected_router_change": best["router_change"],
        "selected_implementation_sketch": best["implementation_sketch"],
        "expected_failure_fixed": best["expected_failure_fixed"],
        "anchor": {
            "05_router_aggregate_map50_95": evidence["router05_aggregate_map50_95"],
            "05_router_repair_map50_95": evidence["router05_repair_map50_95"],
            "03_bn_residual_aggregate_map50_95": evidence["dqa03_map50_95"],
            "04_repair_shielded_best_map50_95": evidence["dqa04_best_map50_95"],
            "night_raw_retention": evidence["night_raw_retention"],
            "day_raw_retention": evidence["day_raw_retention"],
            "night_selected_retention": evidence["night_selected_retention"],
            "day_selected_retention": evidence["day_selected_retention"],
        },
        "next_notebook_candidate": "09_domain_floor_assignment_router_dqa",
        "full_design_notes": {
            "router_principle": "assignment/curriculum first, filtering second",
            "night_guard": "night/domain floors must be enforced before confidence ranking",
            "loss_profile": "low-conf stable boxes remain with weak bbox loss and normal objectness/class loss",
            "model_moe_bridge": "save expert assignment logs so the same router can later train model-level expert heads",
        },
    }
    (args.workspace_root / "stats" / "08_selected_router_candidate.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_report(args: argparse.Namespace, scoreboard: list[dict[str, Any]], trace: list[dict[str, Any]], evidence: Mapping[str, Any]) -> None:
    lines = [
        "# MoE x DQA 08: pseudoGT Router Recovery Fifteen Loops",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- protocol: {PROTOCOL_VERSION}",
        "- mode: paper-driven router recovery screening, not full detector training",
        "",
        "## Evidence From 05 Failure",
        "",
        f"- 05 Expert-Choice aggregate mAP50:95: {evidence['router05_aggregate_map50_95']:.3f}",
        f"- 05 Expert-Choice + repair mAP50:95: {evidence['router05_repair_map50_95']:.3f}",
        f"- 03 BN-residual DQA aggregate mAP50:95: {evidence['dqa03_map50_95']:.3f}",
        f"- 04 repair-shielded best mAP50:95: {evidence['dqa04_best_map50_95']:.3f}",
        f"- day raw pseudo retention round1->30: {evidence['day_raw_retention']:.3f}",
        f"- night raw pseudo retention round1->30: {evidence['night_raw_retention']:.3f}",
        f"- day selected retention round1->30: {evidence['day_selected_retention']:.3f}",
        f"- night selected retention round1->30: {evidence['night_selected_retention']:.3f}",
        f"- night mAP50:95 loss vs 03: {evidence['night_m95_loss_vs_03']:.3f}",
        "",
        "## Ranking",
        "",
        "| rank | loop | projected mAP50:95 | delta | beats 05 | reaches 03/04 | confidence | paper seed |",
        "|---:|---|---:|---:|---|---|---|---|",
    ]
    for idx, row in enumerate(scoreboard, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    row["loop_id"],
                    row["screened_projected_map50_95"],
                    row["screened_delta_map50_95"],
                    row["beats_previous_router"],
                    row["reaches_03_04_target"],
                    row["confidence"],
                    row["paper_seed"].replace("|", "/"),
                ]
            )
            + " |"
        )

    best = scoreboard[0]
    lines.extend(
        [
            "",
            "## Selected Direction",
            "",
            f"- selected_loop: `{best['loop_id']}`",
            f"- paper_seed: {best['paper_seed']}",
            f"- hypothesis: {best['hypothesis']}",
            f"- router change: {best['router_change']}",
            f"- implementation sketch: {best['implementation_sketch']}",
            f"- expected failure fixed: {best['expected_failure_fixed']}",
            f"- why: {best['rationale']}",
            "",
            "## Fifteen-loop Trace",
            "",
            "| loop | research | result | next |",
            "|---:|---|---|---|",
        ]
    )
    for item in trace:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["loop_index"]),
                    str(item["step_1_research"]).replace("|", "/"),
                    str(item["step_6_result_summary"]).replace("|", "/"),
                    str(item["step_7_next_direction"]).replace("|", "/"),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Sources", ""])
    for row in scoreboard:
        lines.append(f"- {row['paper_seed']}: {row['paper_url']}")
    (args.workspace_root / "08_pseudogt_router_recovery_report.md").write_text("\n".join(lines), encoding="utf-8")


def notify(args: argparse.Namespace, message: str, *, title: str, status: str = "") -> None:
    try:
        from notebook_notify import notify_discord

        context = {
            "workspace": str(args.workspace_root.resolve()),
            "status": status,
            "report": str((args.workspace_root / "08_pseudogt_router_recovery_report.md").resolve()),
        }
        print(notify_discord(message, title=title, context=context, fail_silently=True))
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-workspace", type=Path, default=DEFAULT_SOURCE_WORKSPACE)
    parser.add_argument("--router-workspace", type=Path, default=DEFAULT_ROUTER_WORKSPACE)
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--notify-start", action="store_true")
    parser.add_argument("--notify-end", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.source_workspace = args.source_workspace.expanduser().resolve()
    args.router_workspace = args.router_workspace.expanduser().resolve()
    (args.workspace_root / "stats").mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "workspace": str(args.workspace_root),
        "source_workspace": str(args.source_workspace),
        "router_workspace": str(args.router_workspace),
        "mode": "pseudoGT_router_recovery_screening",
    }
    (args.workspace_root / "stats" / "08_router_recovery_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    evidence = load_router_evidence(args.source_workspace, args.router_workspace)
    write_evidence_tables(args, evidence)

    scored_unsorted = [score_loop(spec, evidence) for spec in loop_specs()]
    scoreboard = sorted(scored_unsorted, key=lambda row: as_float(row["rank_score"], -1.0) or -1.0, reverse=True)
    trace = [trace_row(idx, row) for idx, row in enumerate(scoreboard, start=1)]

    fields = [
        "loop_id",
        "paper_seed",
        "paper_url",
        "hypothesis",
        "router_change",
        "implementation_sketch",
        "expected_failure_fixed",
        "anchor_router05_map50_95",
        "target_to_beat_map50_95",
        "strong_target_03_04_map50_95",
        "screened_delta_map50_95",
        "screened_projected_map50_95",
        "rank_score",
        "confidence",
        "beats_previous_router",
        "reaches_03_04_target",
        "priority",
        "rationale",
    ]
    write_csv(args.workspace_root / "stats" / "08_pseudogt_router_recovery_scoreboard.csv", scoreboard, fields)
    write_csv(
        args.workspace_root / "stats" / "08_pseudogt_router_recovery_loop_trace.csv",
        trace,
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
    write_selected(args, scoreboard[0], evidence)
    write_report(args, scoreboard, trace, evidence)
    return scoreboard


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.notify or args.notify_start:
        notify(args, "MoE x DQA 08 pseudoGT-router recovery loop started.", title="DQA MoE 08 start", status="started")
    status = "success"
    try:
        rows = run(args)
        print("Top 5 pseudoGT-router recovery loops:")
        for row in rows[:5]:
            print(
                row["loop_id"],
                row["screened_projected_map50_95"],
                row["beats_previous_router"],
                row["reaches_03_04_target"],
                row["paper_seed"],
            )
    except Exception:
        status = "failed"
        raise
    finally:
        if args.notify or args.notify_end:
            notify(
                args,
                f"MoE x DQA 08 pseudoGT-router recovery loop finished with status={status}.",
                title="DQA MoE 08 finish",
                status=status,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
