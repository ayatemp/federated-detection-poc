#!/usr/bin/env python3
"""Research-notebook loop for DQA-MoX until mAP50 target is reached.

Loop shape:
1. wait for / observe an active run
2. summarize mAP
3. refresh paper notes
4. create a notebook for the next hypothesis
5. execute that notebook
6. parse mAP and notify
7. repeat until target
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import textwrap
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_paper_round_dqamox_until_target import FINAL_METRICS_NAME, PROJECT_ROOT, REPO_ROOT, RUNNER


AGG_ROOT = PROJECT_ROOT / "aggressive_dqamox"
OUTPUT_ROOT = AGG_ROOT / "output" / "27_research_notebook_until_060"
NOTEBOOK_ROOT = AGG_ROOT / "notebooks" / "research_loop_until_060"
REPORT_ROOT = AGG_ROOT / "reports"
LOG_ROOT = AGG_ROOT / "logs"
STATE_PATH = REPORT_ROOT / "27_research_loop_state.json"
SUMMARY_PATH = REPORT_ROOT / "27_research_loop_mAP_summary.csv"
DISK_GUARD = AGG_ROOT / "scripts" / "disk_capacity_guard.py"


@dataclass(frozen=True)
class Strategy:
    name: str
    title: str
    rationale: str
    paper_basis: list[str]
    args: list[str]
    num_experts: int = 4
    top_k: int = 2
    router_temperature: float = 1.2
    router_balance_weight: float = 0.02
    router_entropy_weight: float = 0.0005


PAPER_BANK = [
    {
        "key": "FedMoX/PSSFL",
        "url": "https://arxiv.org/abs/2508.16568",
        "note": (
            "FedMoX treats the practical setting as server labeled high-resolution data plus "
            "client unlabeled low-resolution data, and uses sparse MoE with a spatial router "
            "and Soft-Mixture to stabilize semi-supervised FL."
        ),
    },
    {
        "key": "FedSTO",
        "url": "https://arxiv.org/abs/2310.17097",
        "note": (
            "FedSTO uses server-only labels and client-only unlabeled non-IID data; its two-stage "
            "training selectively refines detector parts first, then moves to full-parameter training."
        ),
    },
    {
        "key": "PseCo",
        "url": "https://arxiv.org/abs/2203.16317",
        "note": (
            "PseCo argues that classification score alone does not guarantee localization precision; "
            "prediction-guided assignment and consistency voting make learning robust to coarse boxes."
        ),
    },
    {
        "key": "Rethinking Pseudo Labels",
        "url": "https://arxiv.org/abs/2106.00168",
        "note": (
            "Certainty-aware pseudo labels combine classification and localization quality, dynamically "
            "adjust thresholds, and reweight category losses to reduce class imbalance."
        ),
    },
    {
        "key": "Mixed Pseudo Labels",
        "url": "https://arxiv.org/abs/2312.07006",
        "note": (
            "MixPL shows that pseudo labels can amplify both a detector's strengths and weaknesses, "
            "especially missed detections for small and tail-category objects; this supports keeping "
            "pseudoGT as a weak residual/domain signal when recent probes drift below warmup."
        ),
    },
    {
        "key": "Object-wise Contrastive + Regression Uncertainty",
        "url": "https://arxiv.org/abs/2212.02747",
        "note": (
            "RUPL-style regression uncertainty separates classification confidence from localization "
            "reliability, which matches the observed pseudoGT failure mode in DQA."
        ),
    },
    {
        "key": "FedLGMatch",
        "url": "https://www.sciencedirect.com/science/article/pii/S0950705125006884",
        "note": (
            "FedLGMatch emphasizes joint local/global pseudo labeling in federated SSL. For DQA-MoX, "
            "this motivates using clean local clients as expert residuals while keeping the global "
            "teacher as the dominant fallback for noisy domains."
        ),
    },
    {
        "key": "FedDG-MoE",
        "url": "https://openaccess.thecvf.com/content/CVPR2025W/FedVision/papers/Radwan_FedDG-MoE_Test-Time_Mixture-of-Experts_Fusion_for_Federated_Domain_Generalization_CVPRW_2025_paper.pdf",
        "note": (
            "FedDG-MoE stores client-specific MoE adapters and fuses them using domain similarity at "
            "test time. For our YOLO latent MoE, this argues against repeatedly averaging all expert "
            "residuals into one bland head; the next aggressive probe should preserve client residual "
            "experts as separate slots in a single checkpoint."
        ),
    },
    {
        "key": "Uncertainty-aware Long-tailed Weights",
        "url": "https://arxiv.org/abs/2503.09974",
        "note": (
            "This work points out that confidence thresholds are brittle under over-confidence and "
            "long-tail scarcity. DQA should downweight uncertain/tail pseudo labels rather than only "
            "drop them, which supports residual expert grafting with small or blended tail adapters."
        ),
    },
    {
        "key": "TMLR 2025 SSOD Building Blocks",
        "url": "https://openreview.net/forum?id=vRYt8QLKqK",
        "note": (
            "The paper analyzes real-world SSOD failures from class imbalance, label noise, and missing "
            "detections. This matches our observation that adding pseudo boxes improves source-val a "
            "little but hurts some night splits unless local experts are preserved."
        ),
    },
    {
        "key": "URSD",
        "url": "https://www.sciencedirect.com/science/article/pii/S0957417425038965",
        "note": (
            "URSD argues that sample mining uncertainty and anchor assignment uncertainty can block "
            "SSOD gains; this supports pruning noisy night pseudoGT instead of forcing every client "
            "into every short probe."
        ),
    },
    {
        "key": "CascadeMatch",
        "url": "https://machinelearning.apple.com/research/semi-supervised-long-tailed",
        "note": (
            "CascadeMatch uses progressive heads and data-driven pseudo-label mining for long-tailed SSOD; "
            "this is relevant because DQA-MoX repeatedly struggles with rare/night/client-specific slices."
        ),
    },
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def notify(message: str, *, title: str = "DQA-MoX research loop") -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from notebook_notify import notify_discord

        print(notify_discord(message, title=title, fail_silently=True))
    except Exception as exc:  # noqa: BLE001
        print(f"notify skipped: {exc}", flush=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "trial",
        "status",
        "best_map50",
        "best_map50_95",
        "warmup_map50",
        "repair_map50",
        "dqa_aggregate_map50",
        "dqa_repair_map50",
        "workspace",
        "notebook",
        "log",
        "finished_utc",
        "rationale",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(text: str | None) -> float | None:
    try:
        value = float(text or "nan")
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def summarize_metrics(metrics_path: Path) -> dict[str, Any]:
    rows = read_csv(metrics_path)
    by_kind = {row.get("kind", ""): row for row in rows}
    warmup = by_kind.get("warmup", {})
    repair = by_kind.get("server_repair", {})
    aggregate = by_kind.get("aggregate", {})
    dqa_repair = rows[-1] if rows else {}
    candidates = [row for row in rows if row.get("condition", "").startswith("warmup +")]
    values = [(as_float(row.get("map50")), as_float(row.get("map50_95"))) for row in candidates]
    values = [(m50, m95) for m50, m95 in values if m50 is not None]
    best_m50, best_m95 = max(values, key=lambda item: item[0]) if values else (None, None)
    return {
        "best_map50": best_m50,
        "best_map50_95": best_m95,
        "warmup_map50": as_float(warmup.get("map50")),
        "repair_map50": as_float(repair.get("map50")),
        "dqa_aggregate_map50": as_float(aggregate.get("map50")),
        "dqa_repair_map50": as_float(dqa_repair.get("map50")),
    }


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"notified_metric_paths": [], "completed_notebooks": []}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"notified_metric_paths": [], "completed_notebooks": []}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def mAP_message(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(row.get("trial", "")),
            f"best_mAP50={row.get('best_map50', '')}",
            f"best_mAP50_95={row.get('best_map50_95', '')}",
            f"warmup_mAP50={row.get('warmup_map50', '')}",
            f"repair_mAP50={row.get('repair_map50', '')}",
            f"dqa_agg_mAP50={row.get('dqa_aggregate_map50', '')}",
            f"dqa_repair_mAP50={row.get('dqa_repair_map50', '')}",
        ]
    )


def discover_and_notify_existing(state: dict[str, Any], *, notify_new: bool) -> tuple[float, list[dict[str, Any]]]:
    rows = read_csv(SUMMARY_PATH)
    known_trials = {row.get("trial", "") for row in rows}
    notified = set(state.get("notified_metric_paths", []))
    best = -1.0
    row_by_trial: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_by_trial[row.get("trial", "")] = row
        value = as_float(row.get("best_map50"))
        if value is not None:
            best = max(best, value)

    for metrics_path in sorted((AGG_ROOT / "output").rglob(FINAL_METRICS_NAME)):
        key = str(metrics_path.resolve())
        workspace = metrics_path.parents[1]
        trial = workspace.name
        summary = summarize_metrics(metrics_path)
        m50 = summary.get("best_map50")
        if m50 is not None:
            best = max(best, m50)
        if trial not in known_trials:
            row = {
                "trial": trial,
                "status": "discovered",
                "best_map50": fmt(summary.get("best_map50")),
                "best_map50_95": fmt(summary.get("best_map50_95")),
                "warmup_map50": fmt(summary.get("warmup_map50")),
                "repair_map50": fmt(summary.get("repair_map50")),
                "dqa_aggregate_map50": fmt(summary.get("dqa_aggregate_map50")),
                "dqa_repair_map50": fmt(summary.get("dqa_repair_map50")),
                "workspace": str(workspace.resolve()),
                "notebook": "",
                "log": "",
                "finished_utc": now(),
                "rationale": "discovered final metrics from an already completed run",
            }
            rows.append(row)
            row_by_trial[trial] = row
            known_trials.add(trial)
        if notify_new and key not in notified and m50 is not None:
            notify(mAP_message(row_by_trial.get(trial, {"trial": trial, "best_map50": fmt(m50)})), title="DQA-MoX mAP result")
            notified.add(key)
    state["notified_metric_paths"] = sorted(notified)
    write_csv(SUMMARY_PATH, rows)
    save_state(state)
    return best, rows


def wait_for_pid(pid: int, poll_seconds: int) -> None:
    if pid <= 0:
        return
    while Path(f"/proc/{pid}").exists():
        time.sleep(poll_seconds)


def fetch_arxiv_notes(iteration: int) -> list[str]:
    queries = [
        "mixture of experts routing load balancing soft mixture federated learning",
        "semi supervised object detection pseudo label localization uncertainty",
        "federated semi supervised object detection non IID unlabeled clients",
        "long tailed semi supervised object detection pseudo label adaptive threshold",
    ]
    query = queries[iteration % len(queries)]
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": f"all:{query}", "start": 0, "max_results": 3, "sortBy": "submittedDate", "sortOrder": "descending"}
    )
    notes: list[str] = []
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            data = response.read()
        root = ET.fromstring(data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            title = " ".join((entry.findtext("atom:title", default="", namespaces=ns) or "").split())
            link = entry.findtext("atom:id", default="", namespaces=ns) or ""
            summary = " ".join((entry.findtext("atom:summary", default="", namespaces=ns) or "").split())
            notes.append(f"- {title}: {link}\n  {summary[:350]}")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"- arXiv refresh failed for query `{query}`: {exc}")
    return notes


def write_research_note(iteration: int, rows: list[dict[str, Any]], strategy: Strategy) -> Path:
    path = REPORT_ROOT / f"27_research_note_iter_{iteration:03d}_{strategy.name}.md"
    recent = rows[-6:]
    lines = [
        f"# Research Note {iteration:03d}: {strategy.name}",
        "",
        f"- created_utc: {now()}",
        f"- strategy: {strategy.title}",
        f"- rationale: {strategy.rationale}",
        "",
        "## Recent mAP",
        "",
        "| trial | best mAP50 | best mAP50:95 | warmup | repair | DQA repair |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in recent:
        lines.append(
            f"| {row.get('trial', '')} | {row.get('best_map50', '')} | {row.get('best_map50_95', '')} | "
            f"{row.get('warmup_map50', '')} | {row.get('repair_map50', '')} | {row.get('dqa_repair_map50', '')} |"
        )
    lines += ["", "## Paper Basis", ""]
    for item in PAPER_BANK:
        if item["key"] in strategy.paper_basis:
            lines.append(f"- [{item['key']}]({item['url']}): {item['note']}")
    lines += ["", "## Fresh arXiv Check", "", *fetch_arxiv_notes(iteration), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def strategy_bank() -> list[Strategy]:
    return [
        Strategy(
            name="27a_soft_mixture_head_first_40_10",
            title="Soft-Mixture head-first DQA-MoX",
            rationale=(
                "25aのbackbone-firstが途中mAPを押し上げ切れていないので、FedMoXのSoft-Mixture思想を優先し、"
                "長いneck/head専門化でpseudoGTをhead側の専門家に吸わせてから短くfull更新する。"
            ),
            paper_basis=["FedMoX/PSSFL", "FedSTO", "PseCo"],
            num_experts=4,
            top_k=2,
            router_temperature=1.25,
            args=[
                "--warmup-epochs", "50", "--client-limit", "3000", "--client-sampling-ratio", "0.333", "--client-sampling-seed", "270001",
                "--phase1-rounds", "40", "--phase2-rounds", "10",
                "--phase1-train-scope", "neck_head", "--phase1-repair-train-scope", "neck_head",
                "--phase1-client-epochs", "1", "--phase1-client-lr", "0.00034", "--phase1-source-repeat", "1", "--phase1-pseudo-repeat", "3", "--phase1-loss-box", "0.0007",
                "--phase2-train-scope", "all", "--phase2-repair-train-scope", "all",
                "--phase2-client-epochs", "1", "--phase2-client-lr", "0.000045", "--phase2-source-repeat", "1", "--phase2-pseudo-repeat", "1", "--phase2-loss-box", "0.00008",
                "--server-repair-epochs", "1", "--server-repair-lr", "0.00020", "--server-repair-loss-box", "0.012",
                "--dqa-server-anchor", "0.18", "--dqa-min-server-alpha", "0.12", "--dqa-residual-blend", "0.00",
                "--late-dqa-server-anchor", "0.09", "--late-dqa-min-server-alpha", "0.04", "--late-dqa-residual-blend", "0.00",
                "--curriculum-start-round", "41", "--expert-keep-fraction", "0.88", "--expert-max-class-fraction", "0.36", "--actual-max-class-fraction", "0.52",
                "--late-expert-keep-fraction", "0.95", "--late-expert-max-class-fraction", "0.42", "--late-actual-max-class-fraction", "0.60",
                "--min-score", "0.17", "--min-stability", "0.52", "--late-min-score", "0.12", "--late-min-stability", "0.42", "--max-boxes-per-image", "14",
            ],
        ),
        Strategy(
            name="27b_localization_uncertainty_strict_then_open",
            title="Localization-quality strict-to-open DQA-MoX",
            rationale=(
                "Rethinking/PseCo/RUPL系の示唆に合わせ、前半はlocalization安定度を強く要求し、後半で少し開く。"
                "confidenceだけではなくstability側に寄せることでpseudo boxの局所誤差の蓄積を抑える。"
            ),
            paper_basis=["Rethinking Pseudo Labels", "PseCo", "Object-wise Contrastive + Regression Uncertainty"],
            num_experts=4,
            top_k=2,
            router_temperature=1.15,
            args=[
                "--warmup-epochs", "50", "--client-limit", "3000", "--client-sampling-ratio", "0.333", "--client-sampling-seed", "270002",
                "--phase1-rounds", "35", "--phase2-rounds", "15",
                "--phase1-train-scope", "neck_head", "--phase1-repair-train-scope", "neck_head",
                "--phase1-client-epochs", "1", "--phase1-client-lr", "0.00028", "--phase1-source-repeat", "1", "--phase1-pseudo-repeat", "2", "--phase1-loss-box", "0.00045",
                "--phase2-train-scope", "all", "--phase2-repair-train-scope", "all",
                "--phase2-client-epochs", "1", "--phase2-client-lr", "0.00004", "--phase2-source-repeat", "1", "--phase2-pseudo-repeat", "1", "--phase2-loss-box", "0.00005",
                "--server-repair-epochs", "1", "--server-repair-lr", "0.00018", "--server-repair-loss-box", "0.010",
                "--dqa-server-anchor", "0.16", "--dqa-min-server-alpha", "0.10", "--dqa-residual-blend", "0.00",
                "--late-dqa-server-anchor", "0.08", "--late-dqa-min-server-alpha", "0.03", "--late-dqa-residual-blend", "0.00",
                "--curriculum-start-round", "36", "--expert-keep-fraction", "0.78", "--expert-max-class-fraction", "0.30", "--actual-max-class-fraction", "0.44",
                "--late-expert-keep-fraction", "0.92", "--late-expert-max-class-fraction", "0.40", "--late-actual-max-class-fraction", "0.58",
                "--min-score", "0.20", "--min-stability", "0.68", "--late-min-score", "0.14", "--late-min-stability", "0.48", "--max-boxes-per-image", "10",
            ],
        ),
        Strategy(
            name="27b_probe_localization_uncertainty_r2",
            title="Probe: localization-quality strict DQA-MoX",
            rationale=(
                "27b本走行は中間target mAPなしで重すぎるため、同じlocalization-quality仮説を2 roundだけ評価する。"
                "ここでwarmupを超えないなら、長い35+15 scheduleには進めずMoE設計を変える。"
            ),
            paper_basis=["Rethinking Pseudo Labels", "PseCo", "Object-wise Contrastive + Regression Uncertainty"],
            num_experts=4,
            top_k=2,
            router_temperature=1.15,
            args=[
                "--warmup-epochs", "50", "--client-limit", "1200", "--client-sampling-ratio", "0.333", "--client-sampling-seed", "270102",
                "--phase1-rounds", "2", "--phase2-rounds", "0",
                "--phase1-train-scope", "neck_head", "--phase1-repair-train-scope", "neck_head",
                "--phase1-client-epochs", "1", "--phase1-client-lr", "0.00028", "--phase1-source-repeat", "1", "--phase1-pseudo-repeat", "2", "--phase1-loss-box", "0.00045",
                "--phase2-train-scope", "all", "--phase2-repair-train-scope", "all",
                "--phase2-client-epochs", "1", "--phase2-client-lr", "0.00004", "--phase2-source-repeat", "1", "--phase2-pseudo-repeat", "1", "--phase2-loss-box", "0.00005",
                "--server-repair-epochs", "1", "--server-repair-lr", "0.00018", "--server-repair-loss-box", "0.010",
                "--dqa-server-anchor", "0.16", "--dqa-min-server-alpha", "0.10", "--dqa-residual-blend", "0.00",
                "--late-dqa-server-anchor", "0.08", "--late-dqa-min-server-alpha", "0.03", "--late-dqa-residual-blend", "0.00",
                "--curriculum-start-round", "3", "--expert-keep-fraction", "0.78", "--expert-max-class-fraction", "0.30", "--actual-max-class-fraction", "0.44",
                "--late-expert-keep-fraction", "0.92", "--late-expert-max-class-fraction", "0.40", "--late-actual-max-class-fraction", "0.58",
                "--min-score", "0.20", "--min-stability", "0.68", "--late-min-score", "0.14", "--late-min-stability", "0.48", "--max-boxes-per-image", "8",
            ],
        ),
        Strategy(
            name="27c_probe_k6_night_tail_r2",
            title="Probe: K6 night-tail expert DQA-MoX",
            rationale=(
                "27b_probeはwarmup比+0.002 mAP50で止まり、改善はhighway_nightの+0.004程度に留まった。"
                "次は全体平均を少し触るのではなく、最悪splitのhighway_nightを明示的に踏むseedで、"
                "K=6の追加expertをnight/long-tailの受け皿にする。2 roundでwarmupを大きく超えないなら長いK6本走行はしない。"
            ),
            paper_basis=["CascadeMatch", "Rethinking Pseudo Labels", "FedMoX/PSSFL"],
            num_experts=6,
            top_k=2,
            router_temperature=1.40,
            router_balance_weight=0.035,
            router_entropy_weight=0.001,
            args=[
                "--warmup-epochs", "50", "--client-limit", "1200", "--client-sampling-ratio", "0.333", "--client-sampling-seed", "270206",
                "--phase1-rounds", "2", "--phase2-rounds", "0",
                "--phase1-train-scope", "neck_head", "--phase1-repair-train-scope", "neck_head",
                "--phase1-client-epochs", "1", "--phase1-client-lr", "0.00030", "--phase1-source-repeat", "1", "--phase1-pseudo-repeat", "3", "--phase1-loss-box", "0.00040",
                "--phase2-train-scope", "all", "--phase2-repair-train-scope", "all",
                "--phase2-client-epochs", "1", "--phase2-client-lr", "0.00004", "--phase2-source-repeat", "1", "--phase2-pseudo-repeat", "1", "--phase2-loss-box", "0.00004",
                "--server-repair-epochs", "1", "--server-repair-lr", "0.00018", "--server-repair-loss-box", "0.008",
                "--dqa-server-anchor", "0.14", "--dqa-min-server-alpha", "0.08", "--dqa-residual-blend", "0.00",
                "--late-dqa-server-anchor", "0.06", "--late-dqa-min-server-alpha", "0.02", "--late-dqa-residual-blend", "0.00",
                "--curriculum-start-round", "3", "--expert-keep-fraction", "0.86", "--expert-max-class-fraction", "0.38", "--actual-max-class-fraction", "0.56",
                "--late-expert-keep-fraction", "0.95", "--late-expert-max-class-fraction", "0.46", "--late-actual-max-class-fraction", "0.64",
                "--min-score", "0.16", "--min-stability", "0.50", "--late-min-score", "0.10", "--late-min-stability", "0.36", "--max-boxes-per-image", "14",
            ],
        ),
        Strategy(
            name="27d_probe_teacher_residual_mixpl_r2",
            title="Probe: teacher-anchored residual MixPL-style DQA-MoX",
            rationale=(
                "27cのK=6 night-tail化はhighway_nightとtotalを同時に悪化させた。MixPLはpseudo-labelが検出器の弱点を"
                "増幅し、tail/small objectのmissを悪化させると指摘している。そこでMoE容量を増やすのではなく、"
                "warmup teacherを強くanchorし、client更新を小さいresidual/adaptorとしてだけ混ぜる。sourceを多めに反復し、"
                "pseudo box lossを極小にして、pseudoGTは主にobjectness/router/domain信号として使う2-round probeにする。"
            ),
            paper_basis=["Mixed Pseudo Labels", "FedMoX/PSSFL", "Rethinking Pseudo Labels", "PseCo"],
            num_experts=4,
            top_k=2,
            router_temperature=1.10,
            router_balance_weight=0.018,
            router_entropy_weight=0.0004,
            args=[
                "--warmup-epochs", "50", "--client-limit", "1200", "--client-sampling-ratio", "0.500", "--client-sampling-seed", "270309",
                "--phase1-rounds", "2", "--phase2-rounds", "0",
                "--phase1-train-scope", "neck_head", "--phase1-repair-train-scope", "neck_head",
                "--phase1-client-epochs", "1", "--phase1-client-lr", "0.00016", "--phase1-source-repeat", "3", "--phase1-pseudo-repeat", "2", "--phase1-loss-box", "0.00012",
                "--phase2-train-scope", "all", "--phase2-repair-train-scope", "all",
                "--phase2-client-epochs", "1", "--phase2-client-lr", "0.00003", "--phase2-source-repeat", "2", "--phase2-pseudo-repeat", "1", "--phase2-loss-box", "0.00003",
                "--server-repair-epochs", "1", "--server-repair-lr", "0.00010", "--server-repair-loss-box", "0.003",
                "--dqa-temperature", "0.95", "--dqa-uniform-mix", "0.18", "--dqa-classwise-blend", "0.18", "--dqa-stability-lambda", "0.45",
                "--dqa-server-anchor", "0.72", "--dqa-min-server-alpha", "0.68", "--dqa-residual-blend", "0.04",
                "--late-dqa-server-anchor", "0.66", "--late-dqa-min-server-alpha", "0.62", "--late-dqa-residual-blend", "0.04",
                "--curriculum-start-round", "3", "--expert-keep-fraction", "0.88", "--expert-max-class-fraction", "0.34", "--actual-max-class-fraction", "0.46",
                "--late-expert-keep-fraction", "0.92", "--late-expert-max-class-fraction", "0.38", "--late-actual-max-class-fraction", "0.50",
                "--min-score", "0.14", "--min-stability", "0.42", "--late-min-score", "0.12", "--late-min-stability", "0.36", "--max-boxes-per-image", "16",
            ],
        ),
        Strategy(
            name="27e_probe_clean_day_expert_anchor_r2",
            title="Probe: clean-day expert residual DQA-MoX",
            rationale=(
                "27dはtotal mAP50を0.462まで戻したが、residential_night mAP50:95を0.236→0.214に落とした。"
                "一方でday側とhighway_nightは微増し、pseudoGTの悪さはclient/domain依存に見える。"
                "FedLGMatch/URSDの示唆に寄せ、夜clientを短期probeから外してday clientだけをclean local expertとして使い、"
                "nightは強いserver teacher anchorで守る。平均化ではなく、clean domain residualをMoE headに吸わせる2-round probe。"
            ),
            paper_basis=["FedLGMatch", "URSD", "FedMoX/PSSFL", "Mixed Pseudo Labels"],
            num_experts=4,
            top_k=2,
            router_temperature=1.05,
            router_balance_weight=0.020,
            router_entropy_weight=0.0004,
            args=[
                "--clients", "0,2,4",
                "--warmup-epochs", "50", "--client-limit", "1600", "--client-sampling-ratio", "1.000", "--client-sampling-seed", "270411",
                "--phase1-rounds", "2", "--phase2-rounds", "0",
                "--phase1-train-scope", "neck_head", "--phase1-repair-train-scope", "neck_head",
                "--phase1-client-epochs", "1", "--phase1-client-lr", "0.00018", "--phase1-source-repeat", "2", "--phase1-pseudo-repeat", "2", "--phase1-loss-box", "0.00016",
                "--phase2-train-scope", "all", "--phase2-repair-train-scope", "all",
                "--phase2-client-epochs", "1", "--phase2-client-lr", "0.00003", "--phase2-source-repeat", "2", "--phase2-pseudo-repeat", "1", "--phase2-loss-box", "0.00003",
                "--server-repair-epochs", "1", "--server-repair-lr", "0.00012", "--server-repair-loss-box", "0.004",
                "--dqa-temperature", "0.85", "--dqa-uniform-mix", "0.12", "--dqa-classwise-blend", "0.25", "--dqa-stability-lambda", "0.55",
                "--dqa-server-anchor", "0.55", "--dqa-min-server-alpha", "0.50", "--dqa-residual-blend", "0.06",
                "--late-dqa-server-anchor", "0.50", "--late-dqa-min-server-alpha", "0.46", "--late-dqa-residual-blend", "0.06",
                "--curriculum-start-round", "3", "--expert-keep-fraction", "0.84", "--expert-max-class-fraction", "0.34", "--actual-max-class-fraction", "0.44",
                "--late-expert-keep-fraction", "0.90", "--late-expert-max-class-fraction", "0.38", "--late-actual-max-class-fraction", "0.50",
                "--min-score", "0.18", "--min-stability", "0.55", "--late-min-score", "0.14", "--late-min-stability", "0.44", "--max-boxes-per-image", "12",
            ],
        ),
        Strategy(
            name="27g_probe_moe_head_only_router_r1",
            title="Probe: MoE-head-only router/adaptor DQA-MoX",
            rationale=(
                "27fでposthoc expert graftは0.462の壁を破れず、MoE residualを強くするとmAP50:95が崩れた。"
                "つまりexpert slot自体より、router/adaptorをbase detectorから分離して訓練時に作る必要がある。"
                "ここではbackbone/neck/shared headを固定し、head.routerとhead.expert_mだけを更新するmoe_head scopeを追加し、"
                "sourceを多めにしてpseudoGTをdomain/router信号として使う1-round probeにする。"
            ),
            paper_basis=["FedDG-MoE", "FedMoX/PSSFL", "Uncertainty-aware Long-tailed Weights", "TMLR 2025 SSOD Building Blocks"],
            num_experts=4,
            top_k=1,
            router_temperature=0.80,
            router_balance_weight=0.040,
            router_entropy_weight=0.0000,
            args=[
                "--warmup-epochs", "50", "--client-limit", "1400", "--client-sampling-ratio", "1.000", "--client-sampling-seed", "270512",
                "--phase1-rounds", "1", "--phase2-rounds", "0",
                "--phase1-train-scope", "moe_head", "--phase1-repair-train-scope", "moe_head",
                "--phase1-client-epochs", "1", "--phase1-client-lr", "0.00042", "--phase1-source-repeat", "4", "--phase1-pseudo-repeat", "1", "--phase1-loss-box", "0.00008",
                "--phase2-train-scope", "moe_head", "--phase2-repair-train-scope", "moe_head",
                "--phase2-client-epochs", "1", "--phase2-client-lr", "0.00004", "--phase2-source-repeat", "3", "--phase2-pseudo-repeat", "1", "--phase2-loss-box", "0.00003",
                "--server-repair-epochs", "1", "--server-repair-lr", "0.00030", "--server-repair-loss-box", "0.0006",
                "--dqa-temperature", "0.75", "--dqa-uniform-mix", "0.10", "--dqa-classwise-blend", "0.30", "--dqa-stability-lambda", "0.50",
                "--dqa-server-anchor", "0.64", "--dqa-min-server-alpha", "0.58", "--dqa-residual-blend", "0.12",
                "--late-dqa-server-anchor", "0.58", "--late-dqa-min-server-alpha", "0.52", "--late-dqa-residual-blend", "0.10",
                "--curriculum-start-round", "2", "--expert-keep-fraction", "0.82", "--expert-max-class-fraction", "0.32", "--actual-max-class-fraction", "0.42",
                "--late-expert-keep-fraction", "0.90", "--late-expert-max-class-fraction", "0.38", "--late-actual-max-class-fraction", "0.48",
                "--min-score", "0.20", "--min-stability", "0.58", "--late-min-score", "0.16", "--late-min-stability", "0.48", "--max-boxes-per-image", "10",
            ],
        ),
        Strategy(
            name="27c_k6_longtail_progressive_experts",
            title="K6 long-tail progressive expert DQA-MoX",
            rationale=(
                "CascadeMatchとclass imbalance対策を意識し、expert容量をK=6へ増やしてrare/client-specific領域の受け皿を増やす。"
                "後半は閾値を下げてrare scene/classを拾うがbox lossは小さく保つ。"
            ),
            paper_basis=["CascadeMatch", "Rethinking Pseudo Labels", "FedMoX/PSSFL"],
            num_experts=6,
            top_k=2,
            router_temperature=1.35,
            router_balance_weight=0.03,
            router_entropy_weight=0.0008,
            args=[
                "--warmup-epochs", "50", "--client-limit", "3000", "--client-sampling-ratio", "0.333", "--client-sampling-seed", "270003",
                "--phase1-rounds", "42", "--phase2-rounds", "8",
                "--phase1-train-scope", "neck_head", "--phase1-repair-train-scope", "neck_head",
                "--phase1-client-epochs", "1", "--phase1-client-lr", "0.00032", "--phase1-source-repeat", "1", "--phase1-pseudo-repeat", "4", "--phase1-loss-box", "0.00045",
                "--phase2-train-scope", "all", "--phase2-repair-train-scope", "all",
                "--phase2-client-epochs", "1", "--phase2-client-lr", "0.00004", "--phase2-source-repeat", "1", "--phase2-pseudo-repeat", "1", "--phase2-loss-box", "0.00004",
                "--server-repair-epochs", "1", "--server-repair-lr", "0.00018", "--server-repair-loss-box", "0.008",
                "--dqa-server-anchor", "0.12", "--dqa-min-server-alpha", "0.06", "--dqa-residual-blend", "0.00",
                "--late-dqa-server-anchor", "0.05", "--late-dqa-min-server-alpha", "0.02", "--late-dqa-residual-blend", "0.00",
                "--curriculum-start-round", "43", "--expert-keep-fraction", "0.90", "--expert-max-class-fraction", "0.40", "--actual-max-class-fraction", "0.58",
                "--late-expert-keep-fraction", "0.97", "--late-expert-max-class-fraction", "0.48", "--late-actual-max-class-fraction", "0.66",
                "--min-score", "0.15", "--min-stability", "0.46", "--late-min-score", "0.10", "--late-min-stability", "0.36", "--max-boxes-per-image", "16",
            ],
        ),
        Strategy(
            name="27d_k8_top3_high_capacity_soft_router",
            title="K8 top-3 high-capacity soft-router DQA-MoX",
            rationale=(
                "MoEのexpert collapseを疑い、K=8/top3/高temperatureで局所領域を複数expertに共有させる。"
                "client単位ではなくpseudoGT分布単位の専門化を広く探索する。"
            ),
            paper_basis=["FedMoX/PSSFL", "CascadeMatch"],
            num_experts=8,
            top_k=3,
            router_temperature=1.55,
            router_balance_weight=0.04,
            router_entropy_weight=0.001,
            args=[
                "--warmup-epochs", "50", "--client-limit", "3000", "--client-sampling-ratio", "0.333", "--client-sampling-seed", "270004",
                "--phase1-rounds", "38", "--phase2-rounds", "12",
                "--phase1-train-scope", "neck_head", "--phase1-repair-train-scope", "neck_head",
                "--phase1-client-epochs", "1", "--phase1-client-lr", "0.00030", "--phase1-source-repeat", "1", "--phase1-pseudo-repeat", "4", "--phase1-loss-box", "0.00035",
                "--phase2-train-scope", "all", "--phase2-repair-train-scope", "all",
                "--phase2-client-epochs", "1", "--phase2-client-lr", "0.000035", "--phase2-source-repeat", "1", "--phase2-pseudo-repeat", "1", "--phase2-loss-box", "0.00004",
                "--server-repair-epochs", "1", "--server-repair-lr", "0.00016", "--server-repair-loss-box", "0.008",
                "--dqa-server-anchor", "0.10", "--dqa-min-server-alpha", "0.05", "--dqa-residual-blend", "0.00",
                "--late-dqa-server-anchor", "0.04", "--late-dqa-min-server-alpha", "0.02", "--late-dqa-residual-blend", "0.00",
                "--curriculum-start-round", "39", "--expert-keep-fraction", "0.92", "--expert-max-class-fraction", "0.42", "--actual-max-class-fraction", "0.62",
                "--late-expert-keep-fraction", "0.98", "--late-expert-max-class-fraction", "0.50", "--late-actual-max-class-fraction", "0.70",
                "--min-score", "0.14", "--min-stability", "0.42", "--late-min-score", "0.10", "--late-min-stability", "0.34", "--max-boxes-per-image", "18",
            ],
        ),
    ]


def synthesize_strategy(iteration: int, best: float) -> Strategy:
    num_experts = [4, 6, 8][iteration % 3]
    top_k = 2 if num_experts < 8 else 3
    phase1 = [35, 40, 45, 30][iteration % 4]
    phase2 = 50 - phase1
    seed = 279000 + iteration
    anchor = max(0.03, 0.12 - 0.015 * (iteration % 5))
    stability = "0.44" if best < 0.52 else "0.56"
    return Strategy(
        name=f"27z_auto_{iteration:03d}_k{num_experts}_p{phase1}_{phase2}",
        title="Auto-synthesized research-loop DQA-MoX",
        rationale=(
            "既存候補を使い切った後、直近best mAPを見てMoE容量、selective/full比、server anchor、"
            "pseudoGT安定度を自動調整する。"
        ),
        paper_basis=["FedMoX/PSSFL", "FedSTO", "PseCo", "Rethinking Pseudo Labels"],
        num_experts=num_experts,
        top_k=top_k,
        router_temperature=1.2 + 0.1 * (iteration % 4),
        router_balance_weight=0.03,
        router_entropy_weight=0.0008,
        args=[
            "--warmup-epochs", "50", "--client-limit", "3000", "--client-sampling-ratio", "0.333", "--client-sampling-seed", str(seed),
            "--phase1-rounds", str(phase1), "--phase2-rounds", str(phase2),
            "--phase1-train-scope", "neck_head", "--phase1-repair-train-scope", "neck_head",
            "--phase1-client-epochs", "1", "--phase1-client-lr", "0.00030", "--phase1-source-repeat", "1", "--phase1-pseudo-repeat", "4", "--phase1-loss-box", "0.00040",
            "--phase2-train-scope", "all", "--phase2-repair-train-scope", "all",
            "--phase2-client-epochs", "1", "--phase2-client-lr", "0.000035", "--phase2-source-repeat", "1", "--phase2-pseudo-repeat", "1", "--phase2-loss-box", "0.00004",
            "--server-repair-epochs", "1", "--server-repair-lr", "0.00016", "--server-repair-loss-box", "0.008",
            "--dqa-server-anchor", f"{anchor:.3f}", "--dqa-min-server-alpha", f"{max(0.01, anchor - 0.05):.3f}", "--dqa-residual-blend", "0.00",
            "--late-dqa-server-anchor", f"{max(0.02, anchor - 0.06):.3f}", "--late-dqa-min-server-alpha", "0.01", "--late-dqa-residual-blend", "0.00",
            "--curriculum-start-round", str(phase1 + 1), "--expert-keep-fraction", "0.92", "--expert-max-class-fraction", "0.42", "--actual-max-class-fraction", "0.62",
            "--late-expert-keep-fraction", "0.98", "--late-expert-max-class-fraction", "0.50", "--late-actual-max-class-fraction", "0.70",
            "--min-score", "0.14", "--min-stability", stability, "--late-min-score", "0.10", "--late-min-stability", "0.34", "--max-boxes-per-image", "18",
        ],
    )


def strategy_is_blocked(strategy: Strategy, rows: list[dict[str, Any]]) -> bool:
    by_trial = {row.get("trial", ""): row for row in rows}
    if strategy.name == "27c_k6_longtail_progressive_experts":
        probe = by_trial.get("27c_probe_k6_night_tail_r2")
        if not probe:
            return False
        probe_best = as_float(probe.get("best_map50"))
        probe_warmup = as_float(probe.get("warmup_map50"))
        if probe_best is None or probe_warmup is None:
            return False
        return probe_best <= probe_warmup + 0.005
    return False


def choose_strategy(rows: list[dict[str, Any]], best: float) -> Strategy:
    completed = {row.get("trial", "") for row in rows}
    for strategy in strategy_bank():
        if strategy.name not in completed and not strategy_is_blocked(strategy, rows):
            return strategy
    count = len([row for row in rows if str(row.get("trial", "")).startswith("27")])
    return synthesize_strategy(count, best)


def notebook_json(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md_cell(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text: str) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def create_notebook(iteration: int, strategy: Strategy, research_note: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    notebook_path = NOTEBOOK_ROOT / f"{iteration:03d}_{strategy.name}.ipynb"
    workspace = OUTPUT_ROOT / strategy.name
    log_path = workspace / "logs" / f"{strategy.name}_train.log"
    metric_path = workspace / "stats" / FINAL_METRICS_NAME
    recent_table = "\n".join(
        [
            "| trial | best mAP50 | mAP50:95 |",
            "|---|---:|---:|",
            *[f"| {r.get('trial','')} | {r.get('best_map50','')} | {r.get('best_map50_95','')} |" for r in rows[-6:]],
        ]
    )
    paper_lines = "\n".join(
        f"- {p['key']}: {p['url']}\n  {p['note']}" for p in PAPER_BANK if p["key"] in strategy.paper_basis
    )
    cmd = [
        sys.executable,
        str(RUNNER),
        "--workspace-root", str(workspace),
        "--repair-baseline-rounds", "0",
        "--source-workspace", str(args.source_workspace),
        "--source-repair-baseline-rounds", str(args.source_repair_baseline_rounds),
        "--target-map50", str(args.target_map50),
        "--num-experts", str(strategy.num_experts),
        "--top-k", str(strategy.top_k),
        "--router-temperature", str(strategy.router_temperature),
        "--router-balance-weight", str(strategy.router_balance_weight),
        "--router-entropy-weight", str(strategy.router_entropy_weight),
        "--dqa-client-balance-stats",
        "--dqa-client-balance-target", "median",
        "--dqa-client-balance-max-scale", "4.0",
        "--load-bias-strength", "0.25",
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--gpus", str(args.gpus),
        "--max-images-per-client", "0",
        "--master-port", str(args.master_port + iteration * 40),
        "--evaluate",
        "--classwise",
        "--no-eval-plots",
        "--force",
        *strategy.args,
    ]
    if args.skip_warmup_training:
        cmd.extend(["--skip-warmup-training", "--warmup-checkpoint", str(args.warmup_checkpoint)])
    train_code = f"""
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path({str(REPO_ROOT)!r})
WORKSPACE = Path({str(workspace)!r})
LOG_PATH = Path({str(log_path)!r})
METRICS_PATH = Path({str(metric_path)!r})
CMD = {cmd!r}

WORKSPACE.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
(WORKSPACE / "stats").mkdir(parents=True, exist_ok=True)
(WORKSPACE / "stats" / "27_notebook_command.json").write_text(
    json.dumps({{"command": CMD}}, indent=2, ensure_ascii=False) + "\\n",
    encoding="utf-8",
)
print(" ".join(CMD))
with LOG_PATH.open("w", encoding="utf-8") as log:
    proc = subprocess.run(CMD, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
print("returncode", proc.returncode)
print("log", LOG_PATH)
if proc.returncode != 0:
    raise SystemExit(proc.returncode)
"""
    result_code = f"""
import csv
from pathlib import Path

METRICS_PATH = Path({str(metric_path)!r})
rows = list(csv.DictReader(METRICS_PATH.open(encoding="utf-8"))) if METRICS_PATH.exists() else []
for row in rows:
    print(row)
"""
    cells = [
        md_cell(
            f"""# {strategy.title}

- created_utc: {now()}
- target_mAP50: {args.target_map50:.3f}
- workspace: `{workspace}`
- log: `{log_path}`
- research_note: `{research_note}`

## Current Results

{recent_table}

## Hypothesis

{strategy.rationale}

## Paper Basis

{paper_lines}
"""
        ),
        code_cell(train_code.strip() + "\n"),
        code_cell(result_code.strip() + "\n"),
    ]
    notebook_path.write_text(json.dumps(notebook_json(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return notebook_path


def start_disk_guard(active_workspace: Path) -> subprocess.Popen[str]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"27_disk_guard_{active_workspace.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.Popen(
            [
                sys.executable,
                str(DISK_GUARD),
                "--active-workspace",
                str(active_workspace),
                "--min-free-gib",
                "80",
                "--critical-free-gib",
                "40",
                "--interval-seconds",
                "600",
            ],
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )


def stop_process(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_notebook(notebook_path: Path, workspace: Path) -> tuple[int, Path]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    exec_log = LOG_ROOT / f"27_execute_{notebook_path.stem}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    guard = start_disk_guard(workspace)
    cmd = [
        "jupyter",
        "nbconvert",
        "--execute",
        "--to",
        "notebook",
        "--inplace",
        "--ExecutePreprocessor.timeout=-1",
        str(notebook_path),
    ]
    try:
        with exec_log.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        return proc.returncode, exec_log
    finally:
        stop_process(guard)


def build_summary_row(strategy: Strategy, notebook_path: Path, exec_log: Path, status: str) -> dict[str, Any]:
    workspace = OUTPUT_ROOT / strategy.name
    metrics_path = workspace / "stats" / FINAL_METRICS_NAME
    summary = summarize_metrics(metrics_path) if metrics_path.exists() else {
        "best_map50": None,
        "best_map50_95": None,
        "warmup_map50": None,
        "repair_map50": None,
        "dqa_aggregate_map50": None,
        "dqa_repair_map50": None,
    }
    return {
        "trial": strategy.name,
        "status": status,
        "best_map50": fmt(summary.get("best_map50")),
        "best_map50_95": fmt(summary.get("best_map50_95")),
        "warmup_map50": fmt(summary.get("warmup_map50")),
        "repair_map50": fmt(summary.get("repair_map50")),
        "dqa_aggregate_map50": fmt(summary.get("dqa_aggregate_map50")),
        "dqa_repair_map50": fmt(summary.get("dqa_repair_map50")),
        "workspace": str(workspace.resolve()),
        "notebook": str(notebook_path.resolve()),
        "log": str(exec_log.resolve()),
        "finished_utc": now(),
        "rationale": strategy.rationale,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--target-map50", type=float, default=0.60)
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--max-iterations", type=int, default=0, help="0 means no fixed iteration cap.")
    parser.add_argument("--source-workspace", type=Path, default=PROJECT_ROOT / "output" / "08_full_latent_dqamox_from_warmup")
    parser.add_argument("--source-repair-baseline-rounds", type=int, default=30)
    parser.add_argument(
        "--warmup-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "output"
        / "08_full_latent_dqamox_from_warmup"
        / "checkpoints"
        / "round000_latent_dqamox_warmup.pt",
    )
    parser.add_argument("--skip-warmup-training", action="store_true")
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=39000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.skip_warmup_training and not args.warmup_checkpoint.exists():
        raise FileNotFoundError(f"--warmup-checkpoint does not exist: {args.warmup_checkpoint}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    state = load_state()

    wait_for_pid(args.wait_pid, args.poll_seconds)
    iteration = 0
    while True:
        best, rows = discover_and_notify_existing(state, notify_new=True)
        if best >= args.target_map50:
            notify(f"target reached: best_mAP50={best:.6f}", title="DQA-MoX target reached")
            return 0
        if args.max_iterations and iteration >= args.max_iterations:
            notify(f"stopped before target: best_mAP50={best:.6f}", title="DQA-MoX loop stopped")
            return 2
        strategy = choose_strategy(rows, best)
        research_note = write_research_note(iteration, rows, strategy)
        notebook_path = create_notebook(iteration, strategy, research_note, rows, args)
        notify(
            f"next notebook: {notebook_path.name}\ncurrent_best_mAP50={best:.6f}\nstrategy={strategy.title}",
            title="DQA-MoX notebook created",
        )
        returncode, exec_log = run_notebook(notebook_path, OUTPUT_ROOT / strategy.name)
        status = "completed" if returncode == 0 else f"failed_rc_{returncode}"
        row = build_summary_row(strategy, notebook_path, exec_log, status)
        rows = read_csv(SUMMARY_PATH)
        rows.append(row)
        write_csv(SUMMARY_PATH, rows)
        notify(mAP_message(row), title="DQA-MoX mAP result")
        try:
            value = float(row.get("best_map50") or "nan")
        except ValueError:
            value = -1.0
        if math.isfinite(value) and value >= args.target_map50:
            notify(f"target reached: best_mAP50={value:.6f}", title="DQA-MoX target reached")
            return 0
        iteration += 1


if __name__ == "__main__":
    raise SystemExit(main())
