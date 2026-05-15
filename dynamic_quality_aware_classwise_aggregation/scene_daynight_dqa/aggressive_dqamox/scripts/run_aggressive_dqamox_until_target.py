#!/usr/bin/env python3
"""Aggressive full-from-warmup DQA-MoX loop.

The previous best run (22) proved that DQA-MoX protects against the server
repair-only collapse, but its server-anchored schedule was too defensive to beat
warmup by much.  This controller keeps the FedMoX-like full-from-warmup shape
while testing more client-dominant schedules.  Each trial is isolated under the
aggressive_dqamox output tree, and the controller stops only when a trial reaches
the configured total mAP50 target or the configured trial budget is exhausted.
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
OUTPUT_ROOT = AGG_ROOT / "output" / "24_aggressive_until_target"
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
        name="24a_client_dominant_soft_expand",
        hypothesis=(
            "Stop treating pseudoGT as a tiny correction. Use low server anchor, "
            "high pseudo ratio, and one low-LR full phase so client domains can "
            "move the MoE detector."
        ),
        args=[
            "--warmup-epochs", "50",
            "--client-limit", "3000",
            "--phase1-rounds", "2",
            "--phase2-rounds", "1",
            "--phase1-train-scope", "neck_head",
            "--phase1-repair-train-scope", "neck_head",
            "--phase1-client-epochs", "2",
            "--phase1-client-lr", "0.0006",
            "--phase1-source-repeat", "1",
            "--phase1-pseudo-repeat", "3",
            "--phase1-loss-box", "0.004",
            "--phase2-train-scope", "all",
            "--phase2-repair-train-scope", "neck_head",
            "--phase2-client-epochs", "1",
            "--phase2-client-lr", "0.00012",
            "--phase2-source-repeat", "1",
            "--phase2-pseudo-repeat", "3",
            "--phase2-loss-box", "0.001",
            "--server-repair-lr", "0.00045",
            "--server-repair-loss-box", "0.025",
            "--dqa-server-anchor", "0.30",
            "--dqa-min-server-alpha", "0.25",
            "--dqa-residual-blend", "0.02",
            "--late-dqa-server-anchor", "0.15",
            "--late-dqa-min-server-alpha", "0.10",
            "--late-dqa-residual-blend", "0.00",
            "--curriculum-start-round", "2",
            "--expert-keep-fraction", "0.80",
            "--expert-max-class-fraction", "0.30",
            "--actual-max-class-fraction", "0.40",
            "--late-expert-keep-fraction", "0.90",
            "--late-expert-max-class-fraction", "0.35",
            "--late-actual-max-class-fraction", "0.50",
            "--min-score", "0.24",
            "--min-stability", "0.62",
            "--late-min-score", "0.18",
            "--late-min-stability", "0.55",
        ],
    ),
    Trial(
        name="24b_client_heavy_neckhead_many_pseudo",
        hypothesis=(
            "If full-detector Phase2 causes drift, keep updates in neck/head but "
            "let pseudoGT dominate enough to create actual client-domain experts."
        ),
        args=[
            "--warmup-epochs", "50",
            "--client-limit", "3000",
            "--phase1-rounds", "3",
            "--phase2-rounds", "0",
            "--phase1-train-scope", "neck_head",
            "--phase1-repair-train-scope", "neck_head",
            "--phase1-client-epochs", "2",
            "--phase1-client-lr", "0.0007",
            "--phase1-source-repeat", "1",
            "--phase1-pseudo-repeat", "4",
            "--phase1-loss-box", "0.002",
            "--server-repair-lr", "0.0004",
            "--server-repair-loss-box", "0.02",
            "--dqa-server-anchor", "0.20",
            "--dqa-min-server-alpha", "0.15",
            "--dqa-residual-blend", "0.01",
            "--expert-keep-fraction", "0.85",
            "--expert-max-class-fraction", "0.34",
            "--actual-max-class-fraction", "0.50",
            "--min-score", "0.20",
            "--min-stability", "0.58",
        ],
    ),
    Trial(
        name="24c_k6_client_dominant_moe",
        hypothesis=(
            "Increase latent expert capacity while keeping top-k sparse routing; "
            "more experts may absorb scene/day-night differences without forcing "
            "one expert to cover too many modes."
        ),
        args=[
            "--num-experts", "6",
            "--top-k", "2",
            "--warmup-epochs", "50",
            "--client-limit", "3000",
            "--phase1-rounds", "2",
            "--phase2-rounds", "1",
            "--phase1-train-scope", "neck_head",
            "--phase1-repair-train-scope", "neck_head",
            "--phase1-client-epochs", "2",
            "--phase1-client-lr", "0.00055",
            "--phase1-source-repeat", "1",
            "--phase1-pseudo-repeat", "3",
            "--phase1-loss-box", "0.003",
            "--phase2-train-scope", "all",
            "--phase2-repair-train-scope", "neck_head",
            "--phase2-client-epochs", "1",
            "--phase2-client-lr", "0.00008",
            "--phase2-source-repeat", "1",
            "--phase2-pseudo-repeat", "3",
            "--phase2-loss-box", "0.0008",
            "--server-repair-lr", "0.0004",
            "--server-repair-loss-box", "0.02",
            "--dqa-server-anchor", "0.25",
            "--dqa-min-server-alpha", "0.20",
            "--dqa-residual-blend", "0.01",
            "--expert-keep-fraction", "0.80",
            "--expert-max-class-fraction", "0.32",
            "--actual-max-class-fraction", "0.45",
            "--min-score", "0.22",
            "--min-stability", "0.60",
        ],
    ),
    Trial(
        name="24d_pseudocls_router_attack_lowbox",
        hypothesis=(
            "Use many pseudo boxes but make localization almost harmless; let "
            "pseudoGT mostly train class/object/router behavior."
        ),
        args=[
            "--warmup-epochs", "50",
            "--client-limit", "3000",
            "--phase1-rounds", "3",
            "--phase2-rounds", "1",
            "--phase1-train-scope", "neck_head",
            "--phase1-repair-train-scope", "neck_head",
            "--phase1-client-epochs", "2",
            "--phase1-client-lr", "0.0008",
            "--phase1-source-repeat", "1",
            "--phase1-pseudo-repeat", "5",
            "--phase1-loss-box", "0.0003",
            "--phase2-train-scope", "neck_head",
            "--phase2-repair-train-scope", "neck_head",
            "--phase2-client-epochs", "1",
            "--phase2-client-lr", "0.0006",
            "--phase2-source-repeat", "1",
            "--phase2-pseudo-repeat", "5",
            "--phase2-loss-box", "0.0002",
            "--server-repair-lr", "0.00035",
            "--server-repair-loss-box", "0.018",
            "--dqa-server-anchor", "0.15",
            "--dqa-min-server-alpha", "0.10",
            "--dqa-residual-blend", "0.00",
            "--expert-keep-fraction", "0.95",
            "--expert-max-class-fraction", "0.38",
            "--actual-max-class-fraction", "0.55",
            "--min-score", "0.16",
            "--min-stability", "0.50",
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


def write_markdown_report(path: Path, rows: list[dict[str, Any]], target: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Aggressive DQA-MoX Loop Report",
        "",
        f"- updated_utc: {now()}",
        f"- target_map50: {target:.3f}",
        "",
        "| trial | status | best mAP50 | hypothesis |",
        "|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('trial','')} | {row.get('status','')} | {row.get('best_map50','')} | "
            f"{str(row.get('hypothesis','')).replace('|', '/')} |"
        )
    lines.append("")
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
        "--workspace-root",
        str(workspace),
        "--repair-baseline-rounds",
        "0",
        "--source-workspace",
        str(args.source_workspace),
        "--source-repair-baseline-rounds",
        str(args.source_repair_baseline_rounds),
        "--target-map50",
        str(args.target_map50),
        "--num-experts",
        str(args.num_experts),
        "--top-k",
        str(args.top_k),
        "--router-temperature",
        str(args.router_temperature),
        "--router-balance-weight",
        str(args.router_balance_weight),
        "--router-entropy-weight",
        str(args.router_entropy_weight),
        "--dqa-client-balance-stats",
        "--dqa-client-balance-target",
        "median",
        "--dqa-client-balance-max-scale",
        "4.0",
        "--max-boxes-per-image",
        "10",
        "--load-bias-strength",
        "0.25",
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--gpus",
        str(args.gpus),
        "--max-images-per-client",
        "0",
        "--master-port",
        str(args.master_port + idx * 20),
        "--evaluate",
        "--classwise",
        "--no-eval-plots",
        "--notify",
        "--notify-progress",
        "--notify-first-progress-hours",
        str(args.notify_first_progress_hours),
        "--notify-progress-interval-hours",
        str(args.notify_progress_interval_hours),
        *trial.args,
    ]
    if args.force:
        cmd.append("--force")

    manifest = {
        "created_utc": started,
        "trial": trial.name,
        "hypothesis": trial.hypothesis,
        "target_map50": args.target_map50,
        "command": cmd,
    }
    (workspace / "stats").mkdir(parents=True, exist_ok=True)
    (workspace / "stats" / "24_aggressive_trial_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    notify(
        f"Starting aggressive DQA-MoX trial {trial.name}",
        title="Aggressive DQA-MoX start",
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
        f"Finished aggressive DQA-MoX trial {trial.name}: {status}, best_map50={row['best_map50']}",
        title="Aggressive DQA-MoX finish",
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
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--router-temperature", type=float, default=1.15)
    parser.add_argument("--router-balance-weight", type=float, default=0.01)
    parser.add_argument("--router-entropy-weight", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=37200)
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
    summary_path = args.report_root / "24_aggressive_until_target_summary.csv"
    report_path = args.report_root / "24_aggressive_until_target_report.md"
    rows: list[dict[str, Any]] = read_rows(summary_path)

    notify(
        "Aggressive DQA-MoX until-target loop started.",
        title="Aggressive DQA-MoX loop",
        context={"target_map50": args.target_map50, "trials": [trial.name for trial in selected]},
    )
    for offset, trial in enumerate(selected, start=args.start_index):
        row = run_trial(args, trial, offset)
        rows.append(row)
        write_csv(summary_path, rows)
        write_markdown_report(report_path, rows, args.target_map50)
        try:
            value = float(row["best_map50"])
        except (TypeError, ValueError):
            value = -1.0
        if value >= args.target_map50 or row["status"] == "target_reached":
            notify(
                f"Aggressive DQA-MoX reached target mAP50={value:.6f}",
                title="Aggressive target reached",
                context={"summary_csv": str(summary_path), **row},
            )
            return 0
        if row["status"].startswith("failed_rc_"):
            return 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
