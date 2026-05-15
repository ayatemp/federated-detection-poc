#!/usr/bin/env python3
"""Run fifteen non-residual MoE theory loops for DQA.

The previous MoE path mainly tested checkpoint/residual composition.  This
runner deliberately avoids that family and screens MoE ideas from LLM/vision
research that affect routing, allocation, compute, and specialization:

* loss-free load balancing,
* fine-grained expert segmentation,
* dynamic-depth/dynamic-compute routing,
* gradient-informed router training,
* balanced assignment / expert-choice routing,
* factorized/cartesian routing.

The output is a hypothesis-screening report.  It is not a full training run.
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
DEFAULT_WORKSPACE = MOE_ROOT / "output" / "07_non_residual_moe_theory_loops"
PROTOCOL_VERSION = "scene_daynight_dqa_moe_07_non_residual_theory_loops_v1"

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
class LoopSpec:
    loop_id: str
    paper_seed: str
    paper_url: str
    hypothesis: str
    dqa_translation: str
    implementation_sketch: str
    novelty: str
    addresses_easy_box_collapse: bool = False
    balances_expert_load: bool = False
    avoids_aux_gradient: bool = False
    uses_gradient_signal: bool = False
    uses_dynamic_compute: bool = False
    uses_fine_grained_experts: bool = False
    uses_factorized_routing: bool = False
    uses_router_only_tuning: bool = False
    uses_test_time_routing: bool = False
    uses_global_assignment: bool = False
    uses_routing_consistency: bool = False
    implementation_risk: float = 0.0
    priority: int = 3


def loop_specs() -> list[LoopSpec]:
    return [
        LoopSpec(
            loop_id="loop01_loss_free_balanced_pseudogt_router",
            paper_seed="Auxiliary-Loss-Free Load Balancing / DeepSeek-V3",
            paper_url="https://arxiv.org/abs/2408.15664",
            hypothesis="pseudoGT劣化はconfidence collapseだけでなくrouting collapseなので、補助lossではなく動的biasでexpert負荷を均衡させる。",
            dqa_translation="各pseudo boxをexpertに割り当てる前に、直近のexpert利用率からrouting scoreへbiasを加える。",
            implementation_sketch="confidence/stability/class/scale score + expert-wise load biasでtop-k box assignmentを作り、負荷が偏ったexpertのbiasを次roundで下げる。",
            novelty="LLM MoEのloss-free load balancingをpseudoGT selectionに移植する。",
            addresses_easy_box_collapse=True,
            balances_expert_load=True,
            avoids_aux_gradient=True,
            priority=1,
        ),
        LoopSpec(
            loop_id="loop02_base_global_balanced_assignment",
            paper_seed="BASE Layers",
            paper_url="https://proceedings.mlr.press/v139/lewis21a.html",
            hypothesis="tokenがexpertを選ぶより、全pseudo boxを一括でbalanced assignmentした方がeasy box過多を防げる。",
            dqa_translation="roundごとにpseudo boxを全体で見て、expert容量制約つきの割当問題として選別する。",
            implementation_sketch="box featureとexpert prototypeのscore行列を作り、各expert capacityを固定してglobal top assignmentを保存する。",
            novelty="検出pseudoGTを個別thresholdではなく、global assignment問題として扱う。",
            addresses_easy_box_collapse=True,
            balances_expert_load=True,
            uses_global_assignment=True,
            implementation_risk=0.0003,
            priority=1,
        ),
        LoopSpec(
            loop_id="loop03_expert_choice_box_buckets",
            paper_seed="Expert Choice Routing",
            paper_url="https://arxiv.org/abs/2202.09368",
            hypothesis="expert側が学習可能なboxを選ぶと、未学習expertと過学習expertの両方を避けられる。",
            dqa_translation="各expertがclass/scale/density bucketから固定数のpseudo boxを選ぶ。",
            implementation_sketch="expertごとにlearnability scoreを計算し、capacity factorつきでbox listを生成する。",
            novelty="DQAのclasswise思想を、client単位ではなくexpert-choice box単位にする。",
            addresses_easy_box_collapse=True,
            balances_expert_load=True,
            uses_global_assignment=True,
            priority=1,
        ),
        LoopSpec(
            loop_id="loop04_grin_gradient_informed_router",
            paper_seed="GRIN / SparseMixer-v2",
            paper_url="https://arxiv.org/abs/2409.12136",
            hypothesis="routingはconfidenceではなく、そのpseudo boxで学習した時のgradient方向が有益かで決めるべき。",
            dqa_translation="小さいprobe batchでhead/neckのgradient agreementを測り、衝突が少ないboxをexpertへ送る。",
            implementation_sketch="各round冒頭に少数pseudo boxでgradient sketchを取り、source/repair gradientとcosine一致するboxを優先する。",
            novelty="LLM MoEのgradient-informed routingをpseudoGT learnability判定に使う。",
            addresses_easy_box_collapse=True,
            uses_gradient_signal=True,
            implementation_risk=0.0007,
            priority=1,
        ),
        LoopSpec(
            loop_id="loop05_mixture_of_depths_pseudogt_compute",
            paper_seed="Mixture-of-Depths",
            paper_url="https://arxiv.org/abs/2404.02258",
            hypothesis="全pseudoGTを同じepoch/augmentationで学習するのが危険なので、box/画像ごとに計算量を変える。",
            dqa_translation="easy/stable boxは軽く、境界/夜/小物体は追加augmentationや追加teacher viewを使う。",
            implementation_sketch="imageごとにcompute budget kを割り当て、hard regionのみmulti-view pseudoGT生成と追加学習を行う。",
            novelty="MoEをexpert数ではなく、pseudoGTへの計算配分問題として扱う。",
            uses_dynamic_compute=True,
            addresses_easy_box_collapse=True,
            priority=1,
        ),
        LoopSpec(
            loop_id="loop06_router_tuning_only",
            paper_seed="Router-Tuning / MindSkip",
            paper_url="https://arxiv.org/abs/2410.13184",
            hypothesis="検出器本体を動かす前に、routerだけをtarget consistencyで合わせればpseudoGT汚染が減る。",
            dqa_translation="YOLO重みは固定し、pseudo box assignment/router parametersだけをtarget consistencyで更新する。",
            implementation_sketch="round 0でrouter tableを学習し、その後のclient trainingはrouterが選んだpseudoGTだけに制限する。",
            novelty="DQAでまずrouter-only adaptationを行い、detector更新を遅らせる。",
            uses_router_only_tuning=True,
            addresses_easy_box_collapse=True,
            implementation_risk=0.0004,
            priority=1,
        ),
        LoopSpec(
            loop_id="loop07_deepseek_fine_grained_expert_segmentation",
            paper_seed="DeepSeekMoE",
            paper_url="https://arxiv.org/abs/2401.06066",
            hypothesis="K=4の大きいexpertより、多数の小さいmicro-expertをtop-k合成した方がscene/class/scaleの重なりを扱える。",
            dqa_translation="scene×time×class-density×scaleを小さいmicro-expertに分解し、1画像あたり複数expertを起動する。",
            implementation_sketch="K=12 micro policiesを作り、shared detector + top-3 micro expert training listでroundを回す。",
            novelty="DeepSeekMoEのfine-grained expert segmentationを検出pseudoGT routingに移す。",
            uses_fine_grained_experts=True,
            balances_expert_load=True,
            priority=2,
        ),
        LoopSpec(
            loop_id="loop08_shared_expert_isolation_without_residual",
            paper_seed="DeepSeekMoE shared expert isolation",
            paper_url="https://arxiv.org/abs/2401.06066",
            hypothesis="source/repair知識を共有expertとして隔離し、target expertはpseudoGTだけの専門性に集中させる。",
            dqa_translation="server repair branchをshared expert、target pseudoGT branchesをrouted expertとして訓練スケジュールを分ける。",
            implementation_sketch="shared pathはsource/cloudyのみ、routed expertはtarget bucketのみで訓練し、評価時にrouterで選ぶ。",
            novelty="checkpoint residualではなく訓練データとrouterでshared/routed役割を分離する。",
            uses_fine_grained_experts=True,
            priority=2,
        ),
        LoopSpec(
            loop_id="loop09_cartesian_scene_time_class_routing",
            paper_seed="CartesianMoE",
            paper_url="https://aclanthology.org/2025.naacl-long.505/",
            hypothesis="scene expertとtime/class expertを掛け合わせれば、少ないexpertで組合せ一般化できる。",
            dqa_translation="routerをscene軸・day/night軸・class/scale軸に分解し、直積でbox bucketを決める。",
            implementation_sketch="expert_id=(scene_axis, condition_axis, object_axis)としてpseudo listを生成し、未観測組合せにも共有を効かせる。",
            novelty="client expertではなく、因子分解されたrouting空間を作る。",
            uses_factorized_routing=True,
            balances_expert_load=True,
            priority=2,
        ),
        LoopSpec(
            loop_id="loop10_softmoe_slot_pseudogt_pooling",
            paper_seed="Soft MoE",
            paper_url="https://arxiv.org/abs/2308.00951",
            hypothesis="hard top-k assignmentがpseudoGT誤差を固定化するので、boxをslotへsoft poolingしてexpert入力を作る。",
            dqa_translation="box単位ではなく、複数boxのweighted slotをpseudoGT training curriculumとして作る。",
            implementation_sketch="画像内boxをsoft slotsへ集約し、slotごとの代表pseudoGTだけをexpertに供給する。",
            novelty="検出でhard pseudo box選別を避け、soft slot単位にする。",
            addresses_easy_box_collapse=True,
            balances_expert_load=True,
            implementation_risk=0.0005,
            priority=2,
        ),
        LoopSpec(
            loop_id="loop11_vmoe_adaptive_per_image_compute",
            paper_seed="V-MoE adaptive compute",
            paper_url="https://arxiv.org/abs/2106.05974",
            hypothesis="画像全体の難しさでexpert数を変えると、night/denseだけに容量を足せる。",
            dqa_translation="画像ごとにtop-k expert数を変え、day/easyはtop1、night/denseはtop3で処理する。",
            implementation_sketch="boxes_per_image, mean_stability, scene priorからper-image expert budgetを決める。",
            novelty="expert選択だけでなく、画像ごとの有効expert数を変える。",
            uses_dynamic_compute=True,
            balances_expert_load=True,
            priority=2,
        ),
        LoopSpec(
            loop_id="loop12_local_routing_consistency_regularizer",
            paper_seed="Local Routing Consistency",
            paper_url="https://huggingface.co/papers/2505.16056",
            hypothesis="近い画像/近いboxでexpertが揺れるとpseudoGTが不安定になるので、局所routingの一貫性を正則化する。",
            dqa_translation="同一scene近傍やcross-round matched boxesのrouter assignmentを滑らかにする。",
            implementation_sketch="matched boxesのrouter KLを小さくし、急なexpert切替をペナルティ化する。",
            novelty="MoE offloading研究のrouting localityをpseudoGT安定性へ転用する。",
            uses_routing_consistency=True,
            uses_router_only_tuning=True,
            priority=2,
        ),
        LoopSpec(
            loop_id="loop13_retrieval_router_prototype_bank",
            paper_seed="RouterRetriever",
            paper_url="https://ojs.aaai.org/index.php/AAAI/article/view/33306",
            hypothesis="routerを小さいMLPで学習するより、source/target region prototype検索でexpertを選ぶ方が少データで安定する。",
            dqa_translation="各expertにregion feature prototype bankを持たせ、pseudo box featureから最近傍expertを選ぶ。",
            implementation_sketch="YOLO中間特徴からbox ROI prototypeを作り、roundごとにprototype bankを更新する。",
            novelty="LLM/IRのrouter retrieverを検出expert選択に使う。",
            uses_test_time_routing=True,
            implementation_risk=0.0004,
            priority=2,
        ),
        LoopSpec(
            loop_id="loop14_da_moe_dynamic_expert_allocation",
            paper_seed="DA-MoE",
            paper_url="https://arxiv.org/abs/2409.06669",
            hypothesis="K固定ではなく、pseudoGTの分布崩壊が見えた時だけexpertを増やす方が自然。",
            dqa_translation="night/dense/low-stability bucketの負荷が閾値を超えたら新expertを生成し、過疎expertは停止する。",
            implementation_sketch="roundごとのload, score, split proxyからexpert birth/pruneを行う。",
            novelty="固定Kではなく、target分布に応じてexpert数を動的に変える。",
            uses_fine_grained_experts=True,
            balances_expert_load=True,
            implementation_risk=0.0008,
            priority=3,
        ),
        LoopSpec(
            loop_id="loop15_causal_factor_router",
            paper_seed="Mixture of Causal Experts / domain causal MoE",
            paper_url="https://www.sciencedirect.com/science/article/abs/pii/S0957417425010449",
            hypothesis="confidenceはdomain交絡を含むので、weather/time/scene/label-densityを因果因子として分けてroutingする。",
            dqa_translation="sceneやday/nightを直接expert名にせず、介入可能な因子としてrouter特徴に入れる。",
            implementation_sketch="causal factor tableを作り、confidenceを使わずstabilityとfactor coverageでpseudoGTを選ぶ。",
            novelty="DQAの得意/不得意を因果的な分布因子として定義し直す。",
            uses_factorized_routing=True,
            addresses_easy_box_collapse=True,
            implementation_risk=0.0006,
            priority=3,
        ),
    ]


def load_evidence(source_workspace: Path) -> dict[str, Any]:
    final_rows = read_csv(source_workspace / "stats" / "03_main_experiment_final_metrics.csv")
    split_rows = read_csv(source_workspace / "stats" / "03_main_experiment_split_metrics.csv")
    pseudo_rows = read_csv(source_workspace / "bn_residual_dqa" / "stats" / "03_round030_pseudo_label_stats.csv")
    final = {row["checkpoint_label"]: row for row in final_rows}
    dqa = final["bn_residual_dqa_final_aggregate"]
    repair = final["warmup_server_repair_final"]
    dqa_repair = final["bn_residual_dqa_final_repair"]
    pseudo_scores = [as_float(row.get("mean_score"), 0.0) or 0.0 for row in pseudo_rows]
    pseudo_density = [as_float(row.get("boxes_per_kept_image"), 0.0) or 0.0 for row in pseudo_rows]
    pseudo_conf = [as_float(row.get("mean_conf"), 0.0) or 0.0 for row in pseudo_rows]
    split_m95 = {
        (row["checkpoint_label"], row["split"]): as_float(row.get("map50_95"), 0.0) or 0.0
        for row in split_rows
        if row.get("split") != "scene_daynight_total"
    }
    day = [
        split_m95[("bn_residual_dqa_final_aggregate", split)]
        for split in ("highway_day", "citystreet_day", "residential_day")
    ]
    night = [
        split_m95[("bn_residual_dqa_final_aggregate", split)]
        for split in ("highway_night", "citystreet_night", "residential_night")
    ]
    return {
        "anchor_map50_95": as_float(dqa.get("map50_95"), 0.0) or 0.0,
        "anchor_map50": as_float(dqa.get("map50"), 0.0) or 0.0,
        "server_repair_map50_95": as_float(repair.get("map50_95"), 0.0) or 0.0,
        "dqa_repair_map50_95": as_float(dqa_repair.get("map50_95"), 0.0) or 0.0,
        "repair_overwrite_loss": (as_float(dqa.get("map50_95"), 0.0) or 0.0) - (as_float(dqa_repair.get("map50_95"), 0.0) or 0.0),
        "dqa_gain_vs_repair": (as_float(dqa.get("map50_95"), 0.0) or 0.0) - (as_float(repair.get("map50_95"), 0.0) or 0.0),
        "day_night_gap": (sum(day) / len(day)) - (sum(night) / len(night)),
        "score_spread": max(pseudo_scores) - min(pseudo_scores),
        "density_spread": max(pseudo_density) - min(pseudo_density),
        "confidence_spread": max(pseudo_conf) - min(pseudo_conf),
        "pseudo_rows": pseudo_rows,
    }


def score_loop(spec: LoopSpec, evidence: Mapping[str, Any]) -> dict[str, Any]:
    score = 0.0
    reasons: list[str] = []
    collapse_signal = min(0.004, evidence["score_spread"] * 0.020 + evidence["density_spread"] * 0.00025)
    gap_signal = min(0.003, evidence["day_night_gap"] * 0.025)
    overwrite_signal = min(0.002, max(0.0, evidence["repair_overwrite_loss"]) * 0.25)

    if spec.addresses_easy_box_collapse:
        score += collapse_signal
        reasons.append(f"targets pseudoGT collapse signal {collapse_signal:.4f}")
    if spec.balances_expert_load:
        bonus = min(0.002, evidence["density_spread"] * 0.00035)
        score += bonus
        reasons.append(f"uses load/density imbalance {bonus:.4f}")
    if spec.avoids_aux_gradient:
        score += 0.0012
        reasons.append("avoids auxiliary-loss interference")
    if spec.uses_gradient_signal:
        score += 0.0024
        reasons.append("uses gradient agreement instead of confidence")
    if spec.uses_dynamic_compute:
        score += 0.0018 + gap_signal * 0.25
        reasons.append("allocates compute to hard regions")
    if spec.uses_fine_grained_experts:
        score += 0.0017
        reasons.append("adds fine-grained specialization capacity")
    if spec.uses_factorized_routing:
        score += 0.0016
        reasons.append("factorizes scene/time/object routing")
    if spec.uses_router_only_tuning:
        score += 0.0013
        reasons.append("delays risky detector updates")
    if spec.uses_test_time_routing:
        score += 0.0011
        reasons.append("can improve routing without retraining detector")
    if spec.uses_global_assignment:
        score += 0.0018
        reasons.append("global assignment prevents per-image threshold bias")
    if spec.uses_routing_consistency:
        score += 0.0010
        reasons.append("stabilizes cross-round expert choices")

    # Non-residual methods do not directly recover the observed final-repair
    # overwrite, but the best router designs should reduce the need for repair.
    if not (spec.uses_dynamic_compute or spec.uses_gradient_signal):
        score += overwrite_signal * 0.25
    score -= spec.implementation_risk
    if spec.implementation_risk:
        reasons.append(f"implementation risk {spec.implementation_risk:.4f}")

    projected = evidence["anchor_map50_95"] + score
    rank_score = projected + {1: 0.0025, 2: 0.0012, 3: 0.0}.get(spec.priority, 0.0)
    confidence = "high" if score >= 0.006 else "medium" if score >= 0.0035 else "low"
    return {
        "loop_id": spec.loop_id,
        "paper_seed": spec.paper_seed,
        "paper_url": spec.paper_url,
        "hypothesis": spec.hypothesis,
        "dqa_translation": spec.dqa_translation,
        "implementation_sketch": spec.implementation_sketch,
        "novelty": spec.novelty,
        "anchor_map50_95": f"{evidence['anchor_map50_95']:.6f}",
        "screened_delta_map50_95": f"{score:.6f}",
        "screened_projected_map50_95": f"{projected:.6f}",
        "rank_score": f"{rank_score:.6f}",
        "confidence": confidence,
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
        "step_4_notebook_action": "recorded in 07_non_residual_moe_theory_loops.ipynb and scoreboard",
        "step_5_execution": "executed as fast non-residual MoE theory screening using 03 measured metrics",
        "step_6_result_summary": (
            f"projected mAP50:95={row['screened_projected_map50_95']}, "
            f"delta={row['screened_delta_map50_95']}, confidence={row['confidence']}"
        ),
        "step_7_next_direction": "promote to full design notebook" if index == 1 else "keep as ablation candidate",
    }


def write_report(args: argparse.Namespace, scoreboard: list[dict[str, Any]], trace: list[dict[str, Any]], evidence: Mapping[str, Any]) -> None:
    lines = [
        "# MoE x DQA 07: Non-residual MoE Theory Fifteen Loops",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- protocol: {PROTOCOL_VERSION}",
        "- mode: paper-driven hypothesis screening, not full detector training",
        "- excluded family: checkpoint/residual composition",
        "",
        "## Evidence Used",
        "",
        f"- 03 DQA aggregate mAP50:95: {evidence['anchor_map50_95']:.3f}",
        f"- 03 DQA + repair mAP50:95: {evidence['dqa_repair_map50_95']:.3f}",
        f"- warmup + repair mAP50:95: {evidence['server_repair_map50_95']:.3f}",
        f"- pseudo score spread: {evidence['score_spread']:.3f}",
        f"- pseudo density spread: {evidence['density_spread']:.3f}",
        f"- day-night gap: {evidence['day_night_gap']:.3f}",
        "",
        "## Ranking",
        "",
        "| rank | loop | rank score | projected mAP50:95 | delta | confidence | paper seed |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for idx, row in enumerate(scoreboard, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    row["loop_id"],
                    row["rank_score"],
                    row["screened_projected_map50_95"],
                    row["screened_delta_map50_95"],
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
            f"- DQA translation: {best['dqa_translation']}",
            f"- implementation sketch: {best['implementation_sketch']}",
            f"- why: {best['rationale']}",
            "",
            "## Executed Fifteen-loop Trace",
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
    (args.workspace_root / "07_non_residual_moe_theory_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_selected(args: argparse.Namespace, best: Mapping[str, Any], evidence: Mapping[str, Any]) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "selected_loop": best["loop_id"],
        "selected_paper_seed": best["paper_seed"],
        "selected_hypothesis": best["hypothesis"],
        "selected_dqa_translation": best["dqa_translation"],
        "selected_implementation_sketch": best["implementation_sketch"],
        "anchor": {
            "03_dqa_aggregate_map50_95": evidence["anchor_map50_95"],
            "03_dqa_repair_map50_95": evidence["dqa_repair_map50_95"],
            "warmup_repair_map50_95": evidence["server_repair_map50_95"],
        },
        "next_full_notebook_name": "05_loss_free_balanced_pseudogt_router_dqa",
        "non_residual_full_design": {
            "detector_update_policy": "do not compose checkpoint residuals; train/update via routed pseudoGT selection only",
            "router_state": "expert load bias updated each round from recent bucket usage",
            "pseudoGT_selection": "balanced assignment over class, scale, scene, density, stability",
            "main_metric": "scene_daynight_total mAP50:95 plus split metrics",
        },
    }
    (args.workspace_root / "stats" / "07_selected_non_residual_candidate.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def notify(args: argparse.Namespace, message: str, *, title: str, status: str = "") -> None:
    try:
        from notebook_notify import notify_discord

        context = {
            "workspace": str(args.workspace_root.resolve()),
            "status": status,
            "report": str((args.workspace_root / "07_non_residual_moe_theory_report.md").resolve()),
        }
        print(notify_discord(message, title=title, context=context, fail_silently=True))
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-workspace", type=Path, default=DEFAULT_SOURCE_WORKSPACE)
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--notify-start", action="store_true")
    parser.add_argument("--notify-end", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.source_workspace = args.source_workspace.expanduser().resolve()
    (args.workspace_root / "stats").mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "workspace": str(args.workspace_root),
        "source_workspace": str(args.source_workspace),
        "mode": "non_residual_moe_theory_screening",
    }
    (args.workspace_root / "stats" / "07_non_residual_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    evidence = load_evidence(args.source_workspace)
    scored_unsorted = [score_loop(spec, evidence) for spec in loop_specs()]
    scoreboard = sorted(scored_unsorted, key=lambda row: as_float(row["rank_score"], -1.0) or -1.0, reverse=True)
    trace = [trace_row(idx, row) for idx, row in enumerate(scoreboard, start=1)]

    fields = [
        "loop_id",
        "paper_seed",
        "paper_url",
        "hypothesis",
        "dqa_translation",
        "implementation_sketch",
        "novelty",
        "anchor_map50_95",
        "screened_delta_map50_95",
        "screened_projected_map50_95",
        "rank_score",
        "confidence",
        "priority",
        "rationale",
    ]
    write_csv(args.workspace_root / "stats" / "07_non_residual_moe_theory_scoreboard.csv", scoreboard, fields)
    write_csv(
        args.workspace_root / "stats" / "07_non_residual_moe_theory_loop_trace.csv",
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
        notify(args, "MoE x DQA 07 non-residual theory loop started.", title="DQA MoE 07 start", status="started")
    status = "success"
    try:
        rows = run(args)
        print("Top 5 non-residual MoE theory loops:")
        for row in rows[:5]:
            print(row["loop_id"], row["screened_projected_map50_95"], row["confidence"], row["paper_seed"])
    except Exception:
        status = "failed"
        raise
    finally:
        if args.notify or args.notify_end:
            notify(
                args,
                f"MoE x DQA 07 non-residual theory loop finished with status={status}.",
                title="DQA MoE 07 finish",
                status=status,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
