#!/usr/bin/env python3
"""Paper-round DQA-MoX continuation.

This controller is separate from the short aggressive 24-series loop.  It uses
round counts and client participation rules taken from FedMoX/FedSTO: warmup 50,
1 epoch per round, 33% online clients, and 50 total FL rounds with a FedSTO-like
20/30 selective-to-full split.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AGG_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AGG_ROOT.parent
REPO_ROOT = PROJECT_ROOT.parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_scene_daynight_dqa_18_client_balanced_single_injection_dqamox.py"
OUTPUT_ROOT = AGG_ROOT / "output" / "25_paper_round_until_target"
REPORT_ROOT = AGG_ROOT / "reports"
FINAL_METRICS_NAME = "18_client_balanced_single_injection_dqamox_final_metrics.csv"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class Trial:
    name: str
    hypothesis: str
    args: list[str] = field(default_factory=list)


TRIALS: list[Trial] = [
    Trial(
        name="25a_fedmox50_sto20_30_top1",
        hypothesis=(
            "FedMoX-style 50 FL rounds with 33% client sampling, mapped through "
            "FedSTO's long selective-to-full 20/30 split. Sparse top-1 routing "
            "tests whether actual communication length, not extra heuristics, is "
            "the missing condition for DQA-MoX."
        ),
        args=[
            "--warmup-epochs", "50",
            "--client-limit", "3000",
            "--client-sampling-ratio", "0.333",
            "--client-sampling-seed", "250816568",
            "--phase1-rounds", "20",
            "--phase2-rounds", "30",
            "--phase1-train-scope", "backbone",
            "--phase1-repair-train-scope", "backbone",
            "--phase1-client-epochs", "1",
            "--phase1-client-lr", "0.00025",
            "--phase1-source-repeat", "1",
            "--phase1-pseudo-repeat", "2",
            "--phase1-loss-box", "0.001",
            "--phase2-train-scope", "all",
            "--phase2-repair-train-scope", "all",
            "--phase2-client-epochs", "1",
            "--phase2-client-lr", "0.00005",
            "--phase2-source-repeat", "1",
            "--phase2-pseudo-repeat", "1",
            "--phase2-loss-box", "0.0002",
            "--server-repair-epochs", "1",
            "--server-repair-lr", "0.0002",
            "--server-repair-loss-box", "0.02",
            "--dqa-server-anchor", "0.40",
            "--dqa-min-server-alpha", "0.35",
            "--dqa-residual-blend", "0.01",
            "--late-dqa-server-anchor", "0.25",
            "--late-dqa-min-server-alpha", "0.20",
            "--late-dqa-residual-blend", "0.00",
            "--curriculum-start-round", "21",
            "--expert-keep-fraction", "0.75",
            "--expert-max-class-fraction", "0.30",
            "--actual-max-class-fraction", "0.40",
            "--late-expert-keep-fraction", "0.85",
            "--late-expert-max-class-fraction", "0.34",
            "--late-actual-max-class-fraction", "0.45",
            "--min-score", "0.25",
            "--min-stability", "0.65",
            "--late-min-score", "0.20",
            "--late-min-stability", "0.58",
            "--max-boxes-per-image", "10",
        ],
    ),
    Trial(
        name="25b_fedmox50_sto20_30_top2_soft",
        hypothesis=(
            "Same paper-round schedule as 25a, but uses soft top-2 routing. If "
            "top-1 is too brittle for YOLO pseudoGT, this keeps FedMoX-like sparse "
            "expert specialization while allowing smoother gradients."
        ),
        args=[
            "--top-k", "2",
            "--router-temperature", "1.15",
            "--warmup-epochs", "50",
            "--client-limit", "3000",
            "--client-sampling-ratio", "0.333",
            "--client-sampling-seed", "250816569",
            "--phase1-rounds", "20",
            "--phase2-rounds", "30",
            "--phase1-train-scope", "backbone",
            "--phase1-repair-train-scope", "backbone",
            "--phase1-client-epochs", "1",
            "--phase1-client-lr", "0.00022",
            "--phase1-source-repeat", "1",
            "--phase1-pseudo-repeat", "2",
            "--phase1-loss-box", "0.001",
            "--phase2-train-scope", "all",
            "--phase2-repair-train-scope", "all",
            "--phase2-client-epochs", "1",
            "--phase2-client-lr", "0.00004",
            "--phase2-source-repeat", "1",
            "--phase2-pseudo-repeat", "1",
            "--phase2-loss-box", "0.00015",
            "--server-repair-epochs", "1",
            "--server-repair-lr", "0.0002",
            "--server-repair-loss-box", "0.02",
            "--dqa-server-anchor", "0.35",
            "--dqa-min-server-alpha", "0.30",
            "--dqa-residual-blend", "0.01",
            "--late-dqa-server-anchor", "0.22",
            "--late-dqa-min-server-alpha", "0.18",
            "--late-dqa-residual-blend", "0.00",
            "--curriculum-start-round", "21",
            "--expert-keep-fraction", "0.78",
            "--expert-max-class-fraction", "0.32",
            "--actual-max-class-fraction", "0.42",
            "--late-expert-keep-fraction", "0.88",
            "--late-expert-max-class-fraction", "0.35",
            "--late-actual-max-class-fraction", "0.48",
            "--min-score", "0.23",
            "--min-stability", "0.62",
            "--late-min-score", "0.18",
            "--late-min-stability", "0.56",
            "--max-boxes-per-image", "10",
        ],
    ),
    Trial(
        name="25c_fedmox50_neckhead35_full15_top2_attack",
        hypothesis=(
            "FedMoX-length 50 FL rounds with a longer DQA selective phase: 35 "
            "neck/head rounds followed by 15 full rounds. This tests the user's "
            "preferred FedSTO-like idea that pseudoGT should first specialize the "
            "MoE detector heads for client/domain/class gaps, then use a short "
            "full-model phase only after the routing has stabilized."
        ),
        args=[
            "--top-k", "2",
            "--router-temperature", "1.20",
            "--warmup-epochs", "50",
            "--client-limit", "3000",
            "--client-sampling-ratio", "0.333",
            "--client-sampling-seed", "250816570",
            "--phase1-rounds", "35",
            "--phase2-rounds", "15",
            "--phase1-train-scope", "neck_head",
            "--phase1-repair-train-scope", "neck_head",
            "--phase1-client-epochs", "1",
            "--phase1-client-lr", "0.00032",
            "--phase1-source-repeat", "1",
            "--phase1-pseudo-repeat", "3",
            "--phase1-loss-box", "0.0008",
            "--phase2-train-scope", "all",
            "--phase2-repair-train-scope", "all",
            "--phase2-client-epochs", "1",
            "--phase2-client-lr", "0.000055",
            "--phase2-source-repeat", "1",
            "--phase2-pseudo-repeat", "1",
            "--phase2-loss-box", "0.00015",
            "--server-repair-epochs", "1",
            "--server-repair-lr", "0.00022",
            "--server-repair-loss-box", "0.015",
            "--dqa-server-anchor", "0.28",
            "--dqa-min-server-alpha", "0.22",
            "--dqa-residual-blend", "0.00",
            "--late-dqa-server-anchor", "0.18",
            "--late-dqa-min-server-alpha", "0.12",
            "--late-dqa-residual-blend", "0.00",
            "--curriculum-start-round", "36",
            "--expert-keep-fraction", "0.82",
            "--expert-max-class-fraction", "0.34",
            "--actual-max-class-fraction", "0.46",
            "--late-expert-keep-fraction", "0.92",
            "--late-expert-max-class-fraction", "0.38",
            "--late-actual-max-class-fraction", "0.52",
            "--min-score", "0.20",
            "--min-stability", "0.58",
            "--late-min-score", "0.16",
            "--late-min-stability", "0.52",
            "--max-boxes-per-image", "12",
        ],
    ),
    Trial(
        name="25d_fedmox50_full20_full30_top2_pseudogt_attack",
        hypothesis=(
            "A deliberately aggressive FedMoX-length run: full-model DQA in both "
            "phases, top-2 routing, weaker server anchoring, and very small box "
            "loss on pseudoGT. If 25a/25b are too defensive to move beyond warmup, "
            "this checks whether the missing ingredient is stronger client-side "
            "feature adaptation while keeping pseudo boxes from dominating."
        ),
        args=[
            "--top-k", "2",
            "--router-temperature", "1.30",
            "--warmup-epochs", "50",
            "--client-limit", "3000",
            "--client-sampling-ratio", "0.333",
            "--client-sampling-seed", "250816571",
            "--phase1-rounds", "20",
            "--phase2-rounds", "30",
            "--phase1-train-scope", "all",
            "--phase1-repair-train-scope", "all",
            "--phase1-client-epochs", "1",
            "--phase1-client-lr", "0.00008",
            "--phase1-source-repeat", "1",
            "--phase1-pseudo-repeat", "2",
            "--phase1-loss-box", "0.0002",
            "--phase2-train-scope", "all",
            "--phase2-repair-train-scope", "all",
            "--phase2-client-epochs", "1",
            "--phase2-client-lr", "0.000045",
            "--phase2-source-repeat", "1",
            "--phase2-pseudo-repeat", "1",
            "--phase2-loss-box", "0.0001",
            "--server-repair-epochs", "1",
            "--server-repair-lr", "0.00018",
            "--server-repair-loss-box", "0.012",
            "--dqa-server-anchor", "0.22",
            "--dqa-min-server-alpha", "0.16",
            "--dqa-residual-blend", "0.00",
            "--late-dqa-server-anchor", "0.14",
            "--late-dqa-min-server-alpha", "0.08",
            "--late-dqa-residual-blend", "0.00",
            "--curriculum-start-round", "21",
            "--expert-keep-fraction", "0.88",
            "--expert-max-class-fraction", "0.36",
            "--actual-max-class-fraction", "0.50",
            "--late-expert-keep-fraction", "0.94",
            "--late-expert-max-class-fraction", "0.40",
            "--late-actual-max-class-fraction", "0.55",
            "--min-score", "0.18",
            "--min-stability", "0.54",
            "--late-min-score", "0.14",
            "--late-min-stability", "0.48",
            "--max-boxes-per-image", "12",
        ],
    ),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def best_map50(path: Path) -> float | None:
    values: list[float] = []
    for row in read_rows(path):
        try:
            values.append(float(row.get("map50") or "nan"))
        except ValueError:
            pass
    values = [value for value in values if value == value]
    return max(values) if values else None


def notify(message: str, *, title: str, context: dict[str, Any] | None = None) -> None:
    try:
        from notebook_notify import notify_discord

        print(notify_discord(message, title=title, context=context or {}, fail_silently=True))
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "trial",
        "status",
        "best_map50",
        "workspace",
        "log",
        "hypothesis",
        "started_utc",
        "finished_utc",
        "runtime_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], target: float) -> None:
    lines = [
        "# Paper-Round DQA-MoX Until-Target Report",
        "",
        f"- updated_utc: {now()}",
        f"- target_map50: {target:.3f}",
        "- basis: FedMoX warmup 50 + FL 50 rounds; FedSTO two-stage selective/full ratio.",
        "",
        "| trial | status | best mAP50 | hypothesis |",
        "|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('trial', '')} | {row.get('status', '')} | {row.get('best_map50', '')} | "
            f"{str(row.get('hypothesis', '')).replace('|', '/')} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_trial(args: argparse.Namespace, trial: Trial, idx: int) -> dict[str, Any]:
    workspace = args.output_root / trial.name
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{trial.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    final_metrics = workspace / "stats" / FINAL_METRICS_NAME
    started = now()
    start_time = time.monotonic()

    existing = best_map50(final_metrics)
    if existing is not None and not args.force:
        status = "target_reached" if existing >= args.target_map50 else "completed_below_target"
        return {
            "trial": trial.name,
            "status": status,
            "best_map50": f"{existing:.6f}",
            "workspace": str(workspace.resolve()),
            "log": str(log_path.resolve()),
            "hypothesis": trial.hypothesis,
            "started_utc": started,
            "finished_utc": now(),
            "runtime_seconds": "0.0",
        }

    cmd = [
        sys.executable,
        str(RUNNER),
        "--workspace-root", str(workspace),
        "--repair-baseline-rounds", "0",
        "--source-workspace", str(args.source_workspace),
        "--source-repair-baseline-rounds", str(args.source_repair_baseline_rounds),
        "--target-map50", str(args.target_map50),
        "--num-experts", str(args.num_experts),
        "--top-k", str(args.top_k),
        "--router-temperature", str(args.router_temperature),
        "--router-balance-weight", str(args.router_balance_weight),
        "--router-entropy-weight", str(args.router_entropy_weight),
        "--dqa-client-balance-stats",
        "--dqa-client-balance-target", "median",
        "--dqa-client-balance-max-scale", "4.0",
        "--load-bias-strength", "0.25",
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--gpus", str(args.gpus),
        "--max-images-per-client", "0",
        "--master-port", str(args.master_port + idx * 40),
        "--evaluate",
        "--classwise",
        "--no-eval-plots",
        "--notify",
        "--notify-progress",
        "--notify-first-progress-hours", str(args.notify_first_progress_hours),
        "--notify-progress-interval-hours", str(args.notify_progress_interval_hours),
        *trial.args,
    ]
    if args.force:
        cmd.append("--force")

    (workspace / "stats").mkdir(parents=True, exist_ok=True)
    (workspace / "stats" / "25_paper_round_trial_manifest.json").write_text(
        json.dumps(
            {
                "created_utc": started,
                "trial": trial.name,
                "hypothesis": trial.hypothesis,
                "target_map50": args.target_map50,
                "paper_round_basis": {
                    "fedmox": "warmup 50 epochs, 50 FL rounds, 33% clients, 1 epoch/round",
                    "fedsto": "two-stage selective/full training, main ratio 100/150",
                },
                "command": cmd,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    notify(
        f"論文ラウンド準拠 DQA-MoX trial を開始します: {trial.name}",
        title="Paper-round DQA-MoX start",
        context={"workspace": str(workspace.resolve()), "hypothesis": trial.hypothesis},
    )
    print(f"[{started}] Starting {trial.name}")
    print(" ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)

    found = best_map50(final_metrics)
    status = "target_reached" if found is not None and found >= args.target_map50 else "completed_below_target"
    if proc.returncode != 0:
        status = f"failed_rc_{proc.returncode}"
    row = {
        "trial": trial.name,
        "status": status,
        "best_map50": "" if found is None else f"{found:.6f}",
        "workspace": str(workspace.resolve()),
        "log": str(log_path.resolve()),
        "hypothesis": trial.hypothesis,
        "started_utc": started,
        "finished_utc": now(),
        "runtime_seconds": f"{time.monotonic() - start_time:.1f}",
    }
    notify(
        f"論文ラウンド準拠 DQA-MoX trial 完了: {trial.name}: {status}, best_map50={row['best_map50']}",
        title="Paper-round DQA-MoX finish",
        context=row,
    )
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--source-workspace", type=Path, default=PROJECT_ROOT / "output" / "08_full_latent_dqamox_from_warmup")
    parser.add_argument("--source-repair-baseline-rounds", type=int, default=30)
    parser.add_argument("--target-map50", type=float, default=0.60)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-trials", type=int, default=1, help="0 means all remaining trials.")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--router-balance-weight", type=float, default=0.02)
    parser.add_argument("--router-entropy-weight", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=38200)
    parser.add_argument("--notify-first-progress-hours", type=float, default=1.0)
    parser.add_argument("--notify-progress-interval-hours", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    args.report_root = args.report_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)

    selected = TRIALS[args.start_index :]
    if args.max_trials > 0:
        selected = selected[: args.max_trials]
    summary_path = args.report_root / "25_paper_round_until_target_summary.csv"
    report_path = args.report_root / "25_paper_round_until_target_report.md"
    rows: list[dict[str, Any]] = read_rows(summary_path)

    notify(
        "論文ラウンド準拠 DQA-MoX loop を開始します。",
        title="Paper-round DQA-MoX loop",
        context={"target_map50": args.target_map50, "trials": [trial.name for trial in selected]},
    )
    for offset, trial in enumerate(selected, start=args.start_index):
        row = run_trial(args, trial, offset)
        rows.append(row)
        write_csv(summary_path, rows)
        write_report(report_path, rows, args.target_map50)
        try:
            value = float(row["best_map50"])
        except (TypeError, ValueError):
            value = -1.0
        if value >= args.target_map50 or row["status"] == "target_reached":
            notify(
                f"論文ラウンド準拠 DQA-MoX が target mAP50={value:.6f} に到達しました。",
                title="Paper-round target reached",
                context={"summary_csv": str(summary_path), **row},
            )
            return 0
        if row["status"].startswith("failed_rc_"):
            return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
