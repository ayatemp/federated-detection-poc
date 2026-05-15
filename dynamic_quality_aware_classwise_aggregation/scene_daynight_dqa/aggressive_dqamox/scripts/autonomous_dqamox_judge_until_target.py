#!/usr/bin/env python3
"""Autonomous DQA-MoX judge loop until target mAP50 is reached.

This is intentionally different from a fixed sweep. It waits for an already
running controller if requested, reports only mAP summaries after each completed
trial, then chooses the next attacking configuration from the observed results.
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
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_paper_round_dqamox_until_target import (
    FINAL_METRICS_NAME,
    PROJECT_ROOT,
    REPO_ROOT,
    RUNNER,
)


AGG_ROOT = PROJECT_ROOT / "aggressive_dqamox"
OUTPUT_ROOT = AGG_ROOT / "output" / "26_autonomous_until_060"
REPORT_ROOT = AGG_ROOT / "reports"
STATE_PATH = REPORT_ROOT / "26_autonomous_judge_state.json"
SUMMARY_PATH = REPORT_ROOT / "26_autonomous_judge_mAP_summary.csv"
DISK_GUARD = AGG_ROOT / "scripts" / "disk_capacity_guard.py"


@dataclass(frozen=True)
class Strategy:
    name: str
    rationale: str
    args: list[str]
    num_experts: int = 4
    top_k: int = 2
    router_temperature: float = 1.15
    router_balance_weight: float = 0.02
    router_entropy_weight: float = 0.0005


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def notify(message: str, *, title: str = "DQA-MoX mAP") -> None:
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
        "log",
        "finished_utc",
        "rationale",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_value(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key) or "nan")
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def summarize_metrics(metrics_path: Path) -> dict[str, Any]:
    rows = read_csv(metrics_path)
    by_kind = {row.get("kind", ""): row for row in rows}
    warmup = by_kind.get("warmup", {})
    repair = by_kind.get("server_repair", {})
    aggregate = by_kind.get("aggregate", {})
    dqa_repair = rows[-1] if rows else {}
    candidates = [row for row in rows if row.get("condition", "").startswith("warmup +")]
    values = [(metric_value(row, "map50"), metric_value(row, "map50_95"), row) for row in candidates]
    values = [(m50, m95, row) for m50, m95, row in values if m50 is not None]
    best_m50, best_m95, _ = max(values, key=lambda item: item[0]) if values else (None, None, {})
    return {
        "best_map50": best_m50,
        "best_map50_95": best_m95,
        "warmup_map50": metric_value(warmup, "map50"),
        "repair_map50": metric_value(repair, "map50"),
        "dqa_aggregate_map50": metric_value(aggregate, "map50"),
        "dqa_repair_map50": metric_value(dqa_repair, "map50"),
    }


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"notified_metrics": [], "completed_trials": []}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"notified_metrics": [], "completed_trials": []}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def discover_existing_mAP(state: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    rows = read_csv(SUMMARY_PATH)
    known = {row.get("trial", "") for row in rows}
    notified = set(state.get("notified_metrics", []))
    best = -1.0
    for row in rows:
        value = metric_value(row, "best_map50")
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
        if trial not in known:
            rows.append(
                {
                    "trial": trial,
                    "status": "discovered",
                    "best_map50": format_float(summary.get("best_map50")),
                    "best_map50_95": format_float(summary.get("best_map50_95")),
                    "warmup_map50": format_float(summary.get("warmup_map50")),
                    "repair_map50": format_float(summary.get("repair_map50")),
                    "dqa_aggregate_map50": format_float(summary.get("dqa_aggregate_map50")),
                    "dqa_repair_map50": format_float(summary.get("dqa_repair_map50")),
                    "workspace": str(workspace.resolve()),
                    "log": "",
                    "finished_utc": now(),
                    "rationale": "discovered existing final metrics",
                }
            )
            known.add(trial)
        if key not in notified and m50 is not None:
            notify(
                "\n".join(
                    [
                        f"{trial}",
                        f"best_mAP50={format_float(summary.get('best_map50'))}",
                        f"best_mAP50_95={format_float(summary.get('best_map50_95'))}",
                        f"warmup_mAP50={format_float(summary.get('warmup_map50'))}",
                        f"repair_mAP50={format_float(summary.get('repair_map50'))}",
                        f"dqa_agg_mAP50={format_float(summary.get('dqa_aggregate_map50'))}",
                        f"dqa_repair_mAP50={format_float(summary.get('dqa_repair_map50'))}",
                    ]
                )
            )
            notified.add(key)
    state["notified_metrics"] = sorted(notified)
    write_csv(SUMMARY_PATH, rows)
    save_state(state)
    return best, rows


def wait_for_pid(pid: int, poll_seconds: int) -> None:
    if pid <= 0:
        return
    while Path(f"/proc/{pid}").exists():
        time.sleep(poll_seconds)


def start_disk_guard(active_workspace: Path) -> subprocess.Popen[str]:
    log_dir = AGG_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"26_disk_guard_{active_workspace.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
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


def cleanup_once(active_workspace: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(DISK_GUARD),
            "--once",
            "--active-workspace",
            str(active_workspace),
            "--min-free-gib",
            "80",
            "--critical-free-gib",
            "40",
        ],
        cwd=REPO_ROOT,
        check=False,
    )


def strategy_bank() -> list[Strategy]:
    return [
        Strategy(
            name="26a_judged_neckhead35_full15_top2",
            rationale="25-seriesが守りすぎる場合に、Phase1をneck/head主体で長くしてMoE headの専門性を強める。",
            args=[
                "--warmup-epochs", "50", "--client-limit", "3000", "--client-sampling-ratio", "0.333", "--client-sampling-seed", "260001",
                "--phase1-rounds", "35", "--phase2-rounds", "15",
                "--phase1-train-scope", "neck_head", "--phase1-repair-train-scope", "neck_head",
                "--phase1-client-epochs", "1", "--phase1-client-lr", "0.00032", "--phase1-source-repeat", "1", "--phase1-pseudo-repeat", "3", "--phase1-loss-box", "0.0008",
                "--phase2-train-scope", "all", "--phase2-repair-train-scope", "all",
                "--phase2-client-epochs", "1", "--phase2-client-lr", "0.000055", "--phase2-source-repeat", "1", "--phase2-pseudo-repeat", "1", "--phase2-loss-box", "0.00015",
                "--server-repair-epochs", "1", "--server-repair-lr", "0.00022", "--server-repair-loss-box", "0.015",
                "--dqa-server-anchor", "0.28", "--dqa-min-server-alpha", "0.22", "--dqa-residual-blend", "0.00",
                "--late-dqa-server-anchor", "0.18", "--late-dqa-min-server-alpha", "0.12", "--late-dqa-residual-blend", "0.00",
                "--curriculum-start-round", "36", "--expert-keep-fraction", "0.82", "--expert-max-class-fraction", "0.34", "--actual-max-class-fraction", "0.46",
                "--late-expert-keep-fraction", "0.92", "--late-expert-max-class-fraction", "0.38", "--late-actual-max-class-fraction", "0.52",
                "--min-score", "0.20", "--min-stability", "0.58", "--late-min-score", "0.16", "--late-min-stability", "0.52", "--max-boxes-per-image", "12",
            ],
            top_k=2,
            router_temperature=1.20,
        ),
        Strategy(
            name="26b_judged_full20_full30_top2_lowbox",
            rationale="pseudo boxの局在誤差を抑えつつ、backboneまでclient domainへ動かす攻めのfull-model設定。",
            args=[
                "--warmup-epochs", "50", "--client-limit", "3000", "--client-sampling-ratio", "0.333", "--client-sampling-seed", "260002",
                "--phase1-rounds", "20", "--phase2-rounds", "30",
                "--phase1-train-scope", "all", "--phase1-repair-train-scope", "all",
                "--phase1-client-epochs", "1", "--phase1-client-lr", "0.00008", "--phase1-source-repeat", "1", "--phase1-pseudo-repeat", "2", "--phase1-loss-box", "0.0002",
                "--phase2-train-scope", "all", "--phase2-repair-train-scope", "all",
                "--phase2-client-epochs", "1", "--phase2-client-lr", "0.000045", "--phase2-source-repeat", "1", "--phase2-pseudo-repeat", "1", "--phase2-loss-box", "0.0001",
                "--server-repair-epochs", "1", "--server-repair-lr", "0.00018", "--server-repair-loss-box", "0.012",
                "--dqa-server-anchor", "0.22", "--dqa-min-server-alpha", "0.16", "--dqa-residual-blend", "0.00",
                "--late-dqa-server-anchor", "0.14", "--late-dqa-min-server-alpha", "0.08", "--late-dqa-residual-blend", "0.00",
                "--curriculum-start-round", "21", "--expert-keep-fraction", "0.88", "--expert-max-class-fraction", "0.36", "--actual-max-class-fraction", "0.50",
                "--late-expert-keep-fraction", "0.94", "--late-expert-max-class-fraction", "0.40", "--late-actual-max-class-fraction", "0.55",
                "--min-score", "0.18", "--min-stability", "0.54", "--late-min-score", "0.14", "--late-min-stability", "0.48", "--max-boxes-per-image", "12",
            ],
            top_k=2,
            router_temperature=1.30,
        ),
        Strategy(
            name="26c_judged_k6_top2_head45_full5",
            rationale="expert数を増やし、Phase1をほぼhead専門化に寄せてclient/class/scene差の受け皿を増やす。",
            args=[
                "--warmup-epochs", "50", "--client-limit", "3000", "--client-sampling-ratio", "0.333", "--client-sampling-seed", "260003",
                "--phase1-rounds", "45", "--phase2-rounds", "5",
                "--phase1-train-scope", "neck_head", "--phase1-repair-train-scope", "neck_head",
                "--phase1-client-epochs", "1", "--phase1-client-lr", "0.00036", "--phase1-source-repeat", "1", "--phase1-pseudo-repeat", "3", "--phase1-loss-box", "0.0006",
                "--phase2-train-scope", "all", "--phase2-repair-train-scope", "all",
                "--phase2-client-epochs", "1", "--phase2-client-lr", "0.00004", "--phase2-source-repeat", "1", "--phase2-pseudo-repeat", "1", "--phase2-loss-box", "0.00008",
                "--server-repair-epochs", "1", "--server-repair-lr", "0.0002", "--server-repair-loss-box", "0.01",
                "--dqa-server-anchor", "0.18", "--dqa-min-server-alpha", "0.12", "--dqa-residual-blend", "0.00",
                "--late-dqa-server-anchor", "0.10", "--late-dqa-min-server-alpha", "0.05", "--late-dqa-residual-blend", "0.00",
                "--curriculum-start-round", "46", "--expert-keep-fraction", "0.88", "--expert-max-class-fraction", "0.36", "--actual-max-class-fraction", "0.52",
                "--late-expert-keep-fraction", "0.95", "--late-expert-max-class-fraction", "0.42", "--late-actual-max-class-fraction", "0.60",
                "--min-score", "0.17", "--min-stability", "0.50", "--late-min-score", "0.13", "--late-min-stability", "0.45", "--max-boxes-per-image", "14",
            ],
            num_experts=6,
            top_k=2,
            router_temperature=1.35,
            router_balance_weight=0.03,
            router_entropy_weight=0.0008,
        ),
        Strategy(
            name="26d_judged_k8_top3_rare_class_attack",
            rationale="rare class/sceneの受け皿不足を疑い、K=8 top3で過度なwinner-take-allを避ける。",
            args=[
                "--warmup-epochs", "50", "--client-limit", "3000", "--client-sampling-ratio", "0.333", "--client-sampling-seed", "260004",
                "--phase1-rounds", "35", "--phase2-rounds", "15",
                "--phase1-train-scope", "neck_head", "--phase1-repair-train-scope", "neck_head",
                "--phase1-client-epochs", "1", "--phase1-client-lr", "0.00030", "--phase1-source-repeat", "1", "--phase1-pseudo-repeat", "4", "--phase1-loss-box", "0.0005",
                "--phase2-train-scope", "all", "--phase2-repair-train-scope", "all",
                "--phase2-client-epochs", "1", "--phase2-client-lr", "0.00004", "--phase2-source-repeat", "1", "--phase2-pseudo-repeat", "1", "--phase2-loss-box", "0.00007",
                "--server-repair-epochs", "1", "--server-repair-lr", "0.00018", "--server-repair-loss-box", "0.010",
                "--dqa-server-anchor", "0.16", "--dqa-min-server-alpha", "0.10", "--dqa-residual-blend", "0.00",
                "--late-dqa-server-anchor", "0.08", "--late-dqa-min-server-alpha", "0.04", "--late-dqa-residual-blend", "0.00",
                "--curriculum-start-round", "36", "--expert-keep-fraction", "0.90", "--expert-max-class-fraction", "0.40", "--actual-max-class-fraction", "0.58",
                "--late-expert-keep-fraction", "0.96", "--late-expert-max-class-fraction", "0.46", "--late-actual-max-class-fraction", "0.64",
                "--min-score", "0.15", "--min-stability", "0.45", "--late-min-score", "0.12", "--late-min-stability", "0.40", "--max-boxes-per-image", "16",
            ],
            num_experts=8,
            top_k=3,
            router_temperature=1.50,
            router_balance_weight=0.04,
            router_entropy_weight=0.0010,
        ),
    ]


def synthesize_strategy(iteration: int, best: float) -> Strategy:
    phase1 = 30 + 5 * (iteration % 4)
    phase2 = 50 - phase1
    num_experts = [4, 6, 8][iteration % 3]
    top_k = 2 if num_experts < 8 else 3
    seed = 261000 + iteration
    anchor = max(0.04, 0.18 - 0.02 * (iteration % 5))
    min_alpha = max(0.02, anchor - 0.06)
    lr_head = 0.00028 + 0.00003 * (iteration % 4)
    full_lr = 0.000035 + 0.000005 * (iteration % 3)
    if best < 0.48:
        pseudo_repeat = "4"
        min_score = "0.14"
        min_stability = "0.42"
    else:
        pseudo_repeat = "3"
        min_score = "0.17"
        min_stability = "0.50"
    return Strategy(
        name=f"26z_auto_{iteration:03d}_k{num_experts}_p{phase1}_{phase2}",
        rationale="直近mAPに応じて、server anchorを下げ、MoE容量とpseudoGT露出を増減する自動生成trial。",
        args=[
            "--warmup-epochs", "50", "--client-limit", "3000", "--client-sampling-ratio", "0.333", "--client-sampling-seed", str(seed),
            "--phase1-rounds", str(phase1), "--phase2-rounds", str(phase2),
            "--phase1-train-scope", "neck_head", "--phase1-repair-train-scope", "neck_head",
            "--phase1-client-epochs", "1", "--phase1-client-lr", f"{lr_head:.8f}", "--phase1-source-repeat", "1", "--phase1-pseudo-repeat", pseudo_repeat, "--phase1-loss-box", "0.0005",
            "--phase2-train-scope", "all", "--phase2-repair-train-scope", "all",
            "--phase2-client-epochs", "1", "--phase2-client-lr", f"{full_lr:.8f}", "--phase2-source-repeat", "1", "--phase2-pseudo-repeat", "1", "--phase2-loss-box", "0.00007",
            "--server-repair-epochs", "1", "--server-repair-lr", "0.00018", "--server-repair-loss-box", "0.010",
            "--dqa-server-anchor", f"{anchor:.2f}", "--dqa-min-server-alpha", f"{min_alpha:.2f}", "--dqa-residual-blend", "0.00",
            "--late-dqa-server-anchor", f"{max(0.02, anchor - 0.08):.2f}", "--late-dqa-min-server-alpha", f"{max(0.01, min_alpha - 0.04):.2f}", "--late-dqa-residual-blend", "0.00",
            "--curriculum-start-round", str(phase1 + 1),
            "--expert-keep-fraction", "0.90", "--expert-max-class-fraction", "0.40", "--actual-max-class-fraction", "0.58",
            "--late-expert-keep-fraction", "0.96", "--late-expert-max-class-fraction", "0.46", "--late-actual-max-class-fraction", "0.64",
            "--min-score", min_score, "--min-stability", min_stability, "--late-min-score", "0.12", "--late-min-stability", "0.40", "--max-boxes-per-image", "16",
        ],
        num_experts=num_experts,
        top_k=top_k,
        router_temperature=1.25 + 0.1 * (iteration % 4),
        router_balance_weight=0.03,
        router_entropy_weight=0.0010,
    )


def already_finished(trial_name: str) -> bool:
    metrics = OUTPUT_ROOT / trial_name / "stats" / FINAL_METRICS_NAME
    return metrics.exists()


def choose_strategy(rows: list[dict[str, Any]], best: float) -> Strategy:
    completed = {row.get("trial", "") for row in rows}
    for strategy in strategy_bank():
        if strategy.name not in completed and not already_finished(strategy.name):
            return strategy
    iteration = len([row for row in rows if str(row.get("trial", "")).startswith("26")])
    return synthesize_strategy(iteration, best)


def run_strategy(args: argparse.Namespace, strategy: Strategy) -> dict[str, Any]:
    workspace = OUTPUT_ROOT / strategy.name
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{strategy.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    metrics_path = workspace / "stats" / FINAL_METRICS_NAME
    cleanup_once(workspace)
    guard = start_disk_guard(workspace)
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
        "--master-port", str(args.master_port),
        "--evaluate",
        "--classwise",
        "--no-eval-plots",
        "--force",
        *strategy.args,
    ]
    (workspace / "stats").mkdir(parents=True, exist_ok=True)
    (workspace / "stats" / "26_autonomous_judge_manifest.json").write_text(
        json.dumps({"created_utc": now(), "strategy": strategy.__dict__, "command": cmd}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    finally:
        stop_process(guard)
        cleanup_once(workspace)
    summary = summarize_metrics(metrics_path) if metrics_path.exists() else {
        "best_map50": None,
        "best_map50_95": None,
        "warmup_map50": None,
        "repair_map50": None,
        "dqa_aggregate_map50": None,
        "dqa_repair_map50": None,
    }
    status = "completed" if proc.returncode == 0 else f"failed_rc_{proc.returncode}"
    return {
        "trial": strategy.name,
        "status": status,
        "best_map50": format_float(summary.get("best_map50")),
        "best_map50_95": format_float(summary.get("best_map50_95")),
        "warmup_map50": format_float(summary.get("warmup_map50")),
        "repair_map50": format_float(summary.get("repair_map50")),
        "dqa_aggregate_map50": format_float(summary.get("dqa_aggregate_map50")),
        "dqa_repair_map50": format_float(summary.get("dqa_repair_map50")),
        "workspace": str(workspace.resolve()),
        "log": str(log_path.resolve()),
        "finished_utc": now(),
        "rationale": strategy.rationale,
    }


def notify_trial_mAP(row: dict[str, Any]) -> None:
    notify(
        "\n".join(
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
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--target-map50", type=float, default=0.60)
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--max-iterations", type=int, default=0, help="0 means no fixed iteration cap.")
    parser.add_argument("--source-workspace", type=Path, default=PROJECT_ROOT / "output" / "08_full_latent_dqamox_from_warmup")
    parser.add_argument("--source-repair-baseline-rounds", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=38600)
    parser.add_argument("--notify-first-progress-hours", type=float, default=1.0)
    parser.add_argument("--notify-progress-interval-hours", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    state = load_state()
    wait_for_pid(args.wait_pid, args.poll_seconds)

    iteration = 0
    while True:
        best, rows = discover_existing_mAP(state)
        if best >= args.target_map50:
            notify(f"target_reached_mAP50={best:.6f}")
            return 0
        if args.max_iterations and iteration >= args.max_iterations:
            notify(f"stopped_before_target_best_mAP50={best:.6f}")
            return 2
        strategy = choose_strategy(rows, best)
        row = run_strategy(args, strategy)
        rows = read_csv(SUMMARY_PATH)
        rows.append(row)
        write_csv(SUMMARY_PATH, rows)
        notify_trial_mAP(row)
        try:
            value = float(row.get("best_map50") or "nan")
        except ValueError:
            value = -1.0
        if math.isfinite(value) and value >= args.target_map50:
            notify(f"target_reached_mAP50={value:.6f}")
            return 0
        iteration += 1


if __name__ == "__main__":
    raise SystemExit(main())
