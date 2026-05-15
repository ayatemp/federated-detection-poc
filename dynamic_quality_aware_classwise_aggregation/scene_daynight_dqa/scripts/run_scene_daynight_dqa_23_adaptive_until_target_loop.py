#!/usr/bin/env python3
"""Run full-from-warmup DQA-MoX trials until the target mAP50 is reached.

This controller is intentionally conservative: every trial calls the existing
full protocol runner and starts from pretrained warmup, so the comparison stays
close to the FedMoX-style "warmup -> client/domain adaptation -> server repair"
story.  It does not reuse a DQA checkpoint from a previous failed trial.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_scene_daynight_dqa_18_client_balanced_single_injection_dqamox.py"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "23_adaptive_until_target_dqamox"
FINAL_METRICS_NAME = "18_client_balanced_single_injection_dqamox_final_metrics.csv"


@dataclass(frozen=True)
class Trial:
    name: str
    hypothesis: str
    args: list[str] = field(default_factory=list)


TRIALS: list[Trial] = [
    Trial(
        name="23a_75ep_twoepoch_neckhead_lowbox",
        hypothesis=(
            "Warmup ceiling may be the bottleneck, so extend warmup while keeping "
            "pseudoGT as a weak neck/head signal."
        ),
        args=[
            "--warmup-epochs", "75",
            "--client-limit", "3000",
            "--phase1-rounds", "1",
            "--phase2-rounds", "0",
            "--post-dqa-repair-rounds", "0",
            "--phase1-train-scope", "neck_head",
            "--phase1-repair-train-scope", "neck_head",
            "--phase1-client-epochs", "2",
            "--phase1-client-lr", "0.00025",
            "--phase1-source-repeat", "4",
            "--phase1-pseudo-repeat", "1",
            "--phase1-loss-box", "0.0015",
            "--server-repair-lr", "0.0006",
            "--server-repair-loss-box", "0.04",
            "--dqa-server-anchor", "0.70",
            "--dqa-min-server-alpha", "0.65",
            "--dqa-residual-blend", "0.08",
            "--expert-keep-fraction", "0.40",
            "--actual-max-class-fraction", "0.20",
            "--min-score", "0.40",
            "--min-stability", "0.82",
        ],
    ),
    Trial(
        name="23b_75ep_two_round_strict_neckhead",
        hypothesis=(
            "If one pseudoGT injection is too small, use two short strict "
            "neck/head rounds while anchoring strongly to the server."
        ),
        args=[
            "--warmup-epochs", "75",
            "--client-limit", "3000",
            "--phase1-rounds", "2",
            "--phase2-rounds", "0",
            "--post-dqa-repair-rounds", "0",
            "--phase1-train-scope", "neck_head",
            "--phase1-repair-train-scope", "neck_head",
            "--phase1-client-epochs", "1",
            "--phase1-client-lr", "0.00035",
            "--phase1-source-repeat", "4",
            "--phase1-pseudo-repeat", "1",
            "--phase1-loss-box", "0.001",
            "--server-repair-lr", "0.0006",
            "--server-repair-loss-box", "0.04",
            "--dqa-server-anchor", "0.75",
            "--dqa-min-server-alpha", "0.70",
            "--dqa-residual-blend", "0.06",
            "--expert-keep-fraction", "0.35",
            "--actual-max-class-fraction", "0.18",
            "--min-score", "0.42",
            "--min-stability", "0.84",
        ],
    ),
    Trial(
        name="23c_75ep_class_router_lowbox",
        hypothesis=(
            "PseudoGT localization is the risky part, so almost remove box loss "
            "and let pseudoGT mainly tune class/object/router behavior."
        ),
        args=[
            "--warmup-epochs", "75",
            "--client-limit", "3000",
            "--phase1-rounds", "1",
            "--phase2-rounds", "0",
            "--post-dqa-repair-rounds", "1",
            "--post-dqa-repair-train-scope", "neck_head",
            "--phase1-train-scope", "neck_head",
            "--phase1-repair-train-scope", "neck_head",
            "--phase1-client-epochs", "2",
            "--phase1-client-lr", "0.00035",
            "--phase1-source-repeat", "3",
            "--phase1-pseudo-repeat", "2",
            "--phase1-loss-box", "0.0003",
            "--server-repair-lr", "0.0006",
            "--server-repair-loss-box", "0.04",
            "--dqa-server-anchor", "0.65",
            "--dqa-min-server-alpha", "0.60",
            "--dqa-residual-blend", "0.08",
            "--expert-keep-fraction", "0.55",
            "--actual-max-class-fraction", "0.24",
            "--min-score", "0.32",
            "--min-stability", "0.76",
        ],
    ),
    Trial(
        name="23d_75ep_neckhead_then_ultralow_full",
        hypothesis=(
            "FedMoX-style final adaptation may need a tiny full-detector step, "
            "but only after a protected neck/head pseudoGT round."
        ),
        args=[
            "--warmup-epochs", "75",
            "--client-limit", "3000",
            "--phase1-rounds", "1",
            "--phase2-rounds", "1",
            "--post-dqa-repair-rounds", "0",
            "--phase1-train-scope", "neck_head",
            "--phase1-repair-train-scope", "neck_head",
            "--phase1-client-epochs", "1",
            "--phase1-client-lr", "0.00035",
            "--phase1-source-repeat", "4",
            "--phase1-pseudo-repeat", "1",
            "--phase1-loss-box", "0.001",
            "--phase2-train-scope", "all",
            "--phase2-repair-train-scope", "neck_head",
            "--phase2-client-epochs", "1",
            "--phase2-client-lr", "0.00005",
            "--phase2-source-repeat", "5",
            "--phase2-pseudo-repeat", "1",
            "--phase2-loss-box", "0.0003",
            "--server-repair-lr", "0.0005",
            "--server-repair-loss-box", "0.035",
            "--dqa-server-anchor", "0.70",
            "--dqa-min-server-alpha", "0.65",
            "--dqa-residual-blend", "0.06",
            "--expert-keep-fraction", "0.40",
            "--actual-max-class-fraction", "0.20",
            "--min-score", "0.40",
            "--min-stability", "0.82",
        ],
    ),
    Trial(
        name="23e_100ep_strict_single_injection",
        hypothesis=(
            "If the 50/75 epoch warmup ceiling is too low, first raise the base "
            "detector ceiling and then inject only the cleanest pseudoGT."
        ),
        args=[
            "--warmup-epochs", "100",
            "--client-limit", "3000",
            "--phase1-rounds", "1",
            "--phase2-rounds", "0",
            "--post-dqa-repair-rounds", "0",
            "--phase1-train-scope", "neck_head",
            "--phase1-repair-train-scope", "neck_head",
            "--phase1-client-epochs", "1",
            "--phase1-client-lr", "0.00025",
            "--phase1-source-repeat", "5",
            "--phase1-pseudo-repeat", "1",
            "--phase1-loss-box", "0.0008",
            "--server-repair-lr", "0.0005",
            "--server-repair-loss-box", "0.035",
            "--dqa-server-anchor", "0.78",
            "--dqa-min-server-alpha", "0.72",
            "--dqa-residual-blend", "0.05",
            "--expert-keep-fraction", "0.30",
            "--actual-max-class-fraction", "0.18",
            "--min-score", "0.45",
            "--min-stability", "0.86",
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


def best_map50(final_metrics: Path) -> float | None:
    best: float | None = None
    for row in read_rows(final_metrics):
        try:
            value = float(row.get("map50") or "nan")
        except ValueError:
            continue
        if value == value:
            best = value if best is None else max(best, value)
    return best


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
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


def notify(message: str, *, title: str, context: dict[str, Any] | None = None) -> None:
    try:
        from notebook_notify import notify_discord

        print(notify_discord(message, title=title, context=context or {}, fail_silently=True))
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}")


def run_trial(args: argparse.Namespace, trial: Trial) -> dict[str, Any]:
    workspace = args.output_root / trial.name
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{trial.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    final = workspace / "stats" / FINAL_METRICS_NAME
    started = now()
    start_time = time.monotonic()

    existing_best = best_map50(final)
    if existing_best is not None and not args.force:
        status = "target_reached" if existing_best >= args.target_map50 else "completed_below_target"
        return {
            "trial": trial.name,
            "status": status,
            "best_map50": f"{existing_best:.6f}",
            "workspace": str(workspace.resolve()),
            "log": str(log_path.resolve()),
            "hypothesis": trial.hypothesis,
            "started_utc": started,
            "finished_utc": now(),
            "runtime_seconds": "0",
        }

    cmd = [
        sys.executable,
        str(RUNNER),
        "--workspace-root",
        str(workspace),
        "--num-experts",
        str(args.num_experts),
        "--top-k",
        str(args.top_k),
        "--repair-baseline-rounds",
        "0",
        "--target-map50",
        str(args.target_map50),
        "--router-temperature",
        "1.3",
        "--router-balance-weight",
        "0.03",
        "--router-entropy-weight",
        "0.002",
        "--dqa-client-balance-stats",
        "--dqa-client-balance-target",
        "median",
        "--dqa-client-balance-max-scale",
        "2.0",
        "--expert-max-class-fraction",
        "0.18",
        "--max-boxes-per-image",
        "8",
        "--load-bias-strength",
        "0.45",
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--gpus",
        str(args.gpus),
        "--max-images-per-client",
        "0",
        "--master-port",
        str(args.master_port),
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
    (workspace / "stats" / "23_controller_trial_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    notify(
        f"Starting {trial.name}: {trial.hypothesis}",
        title="DQA23 adaptive trial start",
        context={"workspace": str(workspace.resolve()), "target_map50": args.target_map50},
    )
    print(f"[{started}] starting {trial.name}")
    print(" ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)

    finished = now()
    found = best_map50(final)
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
        "finished_utc": finished,
        "runtime_seconds": f"{time.monotonic() - start_time:.1f}",
    }
    notify(
        f"Finished {trial.name}: status={status}, best_map50={row['best_map50']}",
        title="DQA23 adaptive trial finish",
        context=row,
    )
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--target-map50", type=float, default=0.60)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-trials", type=int, default=0, help="0 means run all remaining trials.")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=37100)
    parser.add_argument("--notify-first-progress-hours", type=float, default=1.0)
    parser.add_argument("--notify-progress-interval-hours", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    selected = TRIALS[args.start_index :]
    if args.max_trials > 0:
        selected = selected[: args.max_trials]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "23_adaptive_until_target_summary.csv"

    rows: list[dict[str, Any]] = read_rows(summary_path)
    notify(
        "DQA23 adaptive full-from-warmup loop started.",
        title="DQA23 adaptive loop start",
        context={
            "output_root": str(args.output_root),
            "target_map50": args.target_map50,
            "trial_count": len(selected),
        },
    )

    for idx, trial in enumerate(selected, start=args.start_index):
        trial_args = argparse.Namespace(**vars(args))
        trial_args.master_port = args.master_port + idx * 20
        row = run_trial(trial_args, trial)
        rows.append(row)
        write_summary(summary_path, rows)
        try:
            value = float(row["best_map50"])
        except (TypeError, ValueError):
            value = -1.0
        if row["status"] == "target_reached" or value >= args.target_map50:
            notify(
                f"DQA23 adaptive loop reached target mAP50={value:.6f} at {trial.name}.",
                title="DQA23 target reached",
                context={"summary_csv": str(summary_path), **row},
            )
            return 0
        if row["status"].startswith("failed_rc_"):
            notify(
                f"DQA23 adaptive loop stopped after failed trial {trial.name}.",
                title="DQA23 trial failed",
                context={"summary_csv": str(summary_path), **row},
            )
            return 1

    notify(
        "DQA23 adaptive loop exhausted configured trials without reaching target.",
        title="DQA23 target not reached",
        context={"summary_csv": str(summary_path), "target_map50": args.target_map50},
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
