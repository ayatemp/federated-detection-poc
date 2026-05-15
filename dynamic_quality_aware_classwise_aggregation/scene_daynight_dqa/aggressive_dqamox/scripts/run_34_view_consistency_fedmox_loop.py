#!/usr/bin/env python3
"""Run staged FedMoX-style learning loops with hflip-consistent pseudoGT.

The controller keeps the client model footprint essentially unchanged: one
LatentMoE checkpoint per client, four light head experts, and sparse routing.
The new signal is in pseudo-label generation: `min_views=2` forces the existing
identity/hflip stable-augmentation verifier to accept boxes that survive both
counterfactual views.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
AGG_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = AGG_ROOT.parent
REPO_ROOT = PROJECT_ROOT.parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_scene_daynight_dqa_18_client_balanced_single_injection_dqamox.py"
DEFAULT_WARMUP = PROJECT_ROOT / "output" / "08_full_latent_dqamox_from_warmup" / "checkpoints" / "round000_latent_dqamox_warmup.pt"
DEFAULT_OUTPUT = AGG_ROOT / "output" / "34_view_consistency_fedmox_loop"
REPORTS_ROOT = AGG_ROOT / "reports"
SUMMARY_CSV = REPORTS_ROOT / "34_view_consistency_fedmox_loop_summary.csv"
FINAL_METRICS = "18_client_balanced_single_injection_dqamox_final_metrics.csv"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "34_view_consistency_fedmox_loop.ipynb"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class Stage:
    name: str
    phase1_rounds: int
    phase2_rounds: int
    min_best_map50: float
    min_gain_vs_warmup: float


@dataclass(frozen=True)
class Trial:
    name: str
    hypothesis: str
    args: list[str]


STAGES = [
    Stage("probe2", phase1_rounds=2, phase2_rounds=0, min_best_map50=0.465, min_gain_vs_warmup=0.004),
    Stage("probe5", phase1_rounds=5, phase2_rounds=0, min_best_map50=0.472, min_gain_vs_warmup=0.008),
    Stage("probe10", phase1_rounds=10, phase2_rounds=0, min_best_map50=0.485, min_gain_vs_warmup=0.018),
    Stage("phase1_20", phase1_rounds=20, phase2_rounds=0, min_best_map50=0.500, min_gain_vs_warmup=0.030),
    Stage("full20_30", phase1_rounds=20, phase2_rounds=30, min_best_map50=0.600, min_gain_vs_warmup=0.130),
]


TRIALS = [
    Trial(
        name="34a_sparse_moehead20_full30_hflip_consistency",
        hypothesis=(
            "FedMoX-like sparse top-1 routing with the lightest client footprint. "
            "Phase 1 trains only MoE-head/router slots using identity+hflip consistent pseudoGT; "
            "phase 2 opens full-model adaptation at very low LR."
        ),
        args=[
            "--num-experts", "4",
            "--top-k", "1",
            "--router-temperature", "1.00",
            "--router-balance-weight", "0.045",
            "--router-entropy-weight", "0.001",
            "--phase1-train-scope", "moe_head",
            "--phase1-repair-train-scope", "moe_head",
            "--phase1-client-lr", "0.00055",
            "--phase1-source-repeat", "1",
            "--phase1-pseudo-repeat", "2",
            "--phase1-loss-box", "0.00015",
            "--phase2-train-scope", "all",
            "--phase2-repair-train-scope", "all",
            "--phase2-client-lr", "0.00004",
            "--phase2-source-repeat", "1",
            "--phase2-pseudo-repeat", "1",
            "--phase2-loss-box", "0.00010",
            "--server-repair-lr", "0.00020",
            "--server-repair-loss-box", "0.015",
            "--dqa-server-anchor", "0.55",
            "--dqa-min-server-alpha", "0.50",
            "--dqa-residual-blend", "0.00",
            "--late-dqa-server-anchor", "0.35",
            "--late-dqa-min-server-alpha", "0.28",
            "--late-dqa-residual-blend", "0.00",
            "--curriculum-start-round", "21",
            "--expert-keep-fraction", "0.65",
            "--expert-max-class-fraction", "0.26",
            "--actual-max-class-fraction", "0.36",
            "--late-expert-keep-fraction", "0.82",
            "--late-expert-max-class-fraction", "0.32",
            "--late-actual-max-class-fraction", "0.44",
            "--min-views", "2",
            "--min-models", "0",
            "--min-score", "0.30",
            "--min-stability", "0.70",
            "--late-min-score", "0.24",
            "--late-min-stability", "0.64",
            "--max-boxes-per-image", "6",
        ],
    ),
    Trial(
        name="34b_soft_neckhead20_full30_hflip_consistency",
        hypothesis=(
            "Backup if pure MoE-head specialization is too weak: still self-only and hflip-consistent, "
            "but top-2 and neck/head phase-1 updates allow more representation movement before low-LR full adaptation."
        ),
        args=[
            "--num-experts", "4",
            "--top-k", "2",
            "--router-temperature", "1.15",
            "--router-balance-weight", "0.035",
            "--router-entropy-weight", "0.002",
            "--phase1-train-scope", "neck_head",
            "--phase1-repair-train-scope", "neck_head",
            "--phase1-client-lr", "0.00024",
            "--phase1-source-repeat", "2",
            "--phase1-pseudo-repeat", "1",
            "--phase1-loss-box", "0.00020",
            "--phase2-train-scope", "all",
            "--phase2-repair-train-scope", "all",
            "--phase2-client-lr", "0.000035",
            "--phase2-source-repeat", "1",
            "--phase2-pseudo-repeat", "1",
            "--phase2-loss-box", "0.00008",
            "--server-repair-lr", "0.00018",
            "--server-repair-loss-box", "0.012",
            "--dqa-server-anchor", "0.60",
            "--dqa-min-server-alpha", "0.55",
            "--dqa-residual-blend", "0.00",
            "--late-dqa-server-anchor", "0.38",
            "--late-dqa-min-server-alpha", "0.32",
            "--late-dqa-residual-blend", "0.00",
            "--curriculum-start-round", "21",
            "--expert-keep-fraction", "0.58",
            "--expert-max-class-fraction", "0.24",
            "--actual-max-class-fraction", "0.34",
            "--late-expert-keep-fraction", "0.78",
            "--late-expert-max-class-fraction", "0.30",
            "--late-actual-max-class-fraction", "0.42",
            "--min-views", "2",
            "--min-models", "0",
            "--min-score", "0.32",
            "--min-stability", "0.72",
            "--late-min-score", "0.25",
            "--late-min-stability", "0.66",
            "--max-boxes-per-image", "6",
        ],
    ),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup-checkpoint", type=Path, default=DEFAULT_WARMUP)
    parser.add_argument("--target-map50", type=float, default=0.60)
    parser.add_argument("--client-limit", type=int, default=3000)
    parser.add_argument("--client-sampling-ratio", type=float, default=0.333)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--master-port", type=int, default=37601)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--no-discord", action="store_true")
    parser.add_argument("--max-trials", type=int, default=0, help="0 means all configured trials.")
    return parser.parse_args(argv)


def notify(message: str, title: str) -> None:
    try:
        from notebook_notify import notify_discord

        print(notify_discord(message, title=title, fail_silently=True))
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}")


def write_summary_row(row: dict[str, Any]) -> None:
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial",
        "stage",
        "status",
        "best_label",
        "best_map50",
        "best_map50_95",
        "warmup_map50",
        "gain_vs_warmup",
        "threshold_best_map50",
        "threshold_gain_vs_warmup",
        "target_map50",
        "workspace",
        "metrics_csv",
        "log",
        "started_utc",
        "finished_utc",
        "runtime_seconds",
        "hypothesis",
        "decision",
    ]
    exists = SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_metrics(workspace: Path) -> tuple[list[dict[str, str]], dict[str, str] | None, dict[str, str] | None]:
    metrics_csv = workspace / "stats" / FINAL_METRICS
    if not metrics_csv.exists():
        return [], None, None
    with metrics_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    warmup = next((row for row in rows if row.get("checkpoint_label") == "warmup_global"), None)
    scored = []
    for row in rows:
        try:
            scored.append((float(row.get("map50") or 0.0), float(row.get("map50_95") or 0.0), row))
        except ValueError:
            continue
    best = max(scored, key=lambda item: (item[0], item[1]))[2] if scored else None
    return rows, warmup, best


def run_stage(args: argparse.Namespace, trial: Trial, stage: Stage, trial_index: int) -> tuple[str, dict[str, Any]]:
    workspace = (args.output_root / trial.name).resolve()
    log_dir = AGG_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    log_path = log_dir / f"34_{trial.name}_{stage.name}_{started.strftime('%Y%m%d_%H%M%S')}.log"
    cmd = [
        str(args.python_executable),
        str(RUNNER),
        "--workspace-root", str(workspace),
        "--source-workspace", str(PROJECT_ROOT / "output" / "08_full_latent_dqamox_from_warmup"),
        "--skip-warmup-training",
        "--warmup-checkpoint", str(args.warmup_checkpoint),
        "--warmup-epochs", "50",
        "--client-limit", str(args.client_limit),
        "--client-sampling-ratio", str(args.client_sampling_ratio),
        "--client-sampling-seed", str(340000 + trial_index),
        "--phase1-rounds", str(stage.phase1_rounds),
        "--phase2-rounds", str(stage.phase2_rounds),
        "--phase1-client-epochs", "1",
        "--phase2-client-epochs", "1",
        "--server-repair-epochs", "1",
        "--repair-baseline-rounds", "0",
        "--post-dqa-repair-rounds", "0",
        "--gpus", str(args.gpus),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--master-port", str(args.master_port + trial_index * 100 + stage.phase1_rounds + stage.phase2_rounds),
        "--target-map50", str(args.target_map50),
        "--evaluate",
        "--no-progress",
        *trial.args,
    ]
    print(" ".join(cmd))
    print("log:", log_path)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    finished = datetime.now(timezone.utc)
    rows, warmup, best = read_metrics(workspace)
    warmup_map50 = float(warmup.get("map50") or 0.0) if warmup else 0.0
    best_map50 = float(best.get("map50") or 0.0) if best else 0.0
    best_map95 = float(best.get("map50_95") or 0.0) if best else 0.0
    gain = best_map50 - warmup_map50
    if proc.returncode != 0:
        status = f"failed_rc_{proc.returncode}"
        decision = "try_next_trial"
    elif best_map50 >= args.target_map50:
        status = "target_reached"
        decision = "stop_success"
    elif stage.name != STAGES[-1].name and (best_map50 < stage.min_best_map50 or gain < stage.min_gain_vs_warmup):
        status = "stopped_weak_stage"
        decision = "try_next_trial"
    elif stage.name == STAGES[-1].name:
        status = "completed_below_target"
        decision = "try_next_trial"
    else:
        status = "stage_passed"
        decision = "continue_trial"

    row = {
        "trial": trial.name,
        "stage": stage.name,
        "status": status,
        "best_label": best.get("checkpoint_label", "") if best else "",
        "best_map50": f"{best_map50:.6f}" if best else "",
        "best_map50_95": f"{best_map95:.6f}" if best else "",
        "warmup_map50": f"{warmup_map50:.6f}" if warmup else "",
        "gain_vs_warmup": f"{gain:+.6f}" if warmup and best else "",
        "threshold_best_map50": stage.min_best_map50,
        "threshold_gain_vs_warmup": stage.min_gain_vs_warmup,
        "target_map50": args.target_map50,
        "workspace": str(workspace),
        "metrics_csv": str(workspace / "stats" / FINAL_METRICS),
        "log": str(log_path),
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "runtime_seconds": round((finished - started).total_seconds(), 1),
        "hypothesis": trial.hypothesis,
        "decision": decision,
    }
    write_summary_row(row)
    return decision, row


def format_stage_message(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"34 stage finished: {row['trial']} / {row['stage']}",
            f"- status={row['status']} decision={row['decision']}",
            f"- best={row['best_label']} mAP50={row['best_map50']} / mAP50:95={row['best_map50_95']}",
            f"- warmup={row['warmup_map50']} gain={row['gain_vs_warmup']}",
            f"- workspace={row['workspace']}",
            f"- metrics={row['metrics_csv']}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.warmup_checkpoint.exists():
        raise FileNotFoundError(args.warmup_checkpoint)
    args.output_root.mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    trials = TRIALS if args.max_trials <= 0 else TRIALS[: args.max_trials]

    intro = "\n".join(
        [
            "34 started: staged hflip-consistency FedMoX-DQA loop.",
            f"- target mAP50={args.target_map50}",
            "- setting: 50 FL rounds if promising, 33.3% clients, one checkpoint per client, self-only identity+hflip pseudoGT verifier.",
            f"- trials={', '.join(t.name for t in trials)}",
        ]
    )
    print(intro)
    if not args.no_discord:
        notify(intro, "DQA-MoX 34 started")

    final_rows: list[dict[str, Any]] = []
    for trial_index, trial in enumerate(trials):
        print(f"\n=== trial {trial.name} ===")
        print(trial.hypothesis)
        for stage in STAGES:
            decision, row = run_stage(args, trial, stage, trial_index)
            final_rows.append(row)
            message = format_stage_message(row)
            print(message)
            if not args.no_discord:
                notify(message, "DQA-MoX 34 stage")
            if decision == "stop_success":
                return 0
            if decision == "try_next_trial":
                break

    best_rows = [row for row in final_rows if row.get("best_map50")]
    best = max(best_rows, key=lambda row: float(row["best_map50"])) if best_rows else {}
    outro = "\n".join(
        [
            "34 loop finished without reaching target." if best else "34 loop finished without valid metrics.",
            f"- best trial={best.get('trial', '')} stage={best.get('stage', '')}",
            f"- best mAP50={best.get('best_map50', '')} / mAP50:95={best.get('best_map50_95', '')}",
            f"- summary={SUMMARY_CSV}",
        ]
    )
    print(outro)
    if not args.no_discord:
        notify(outro, "DQA-MoX 34 result")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
