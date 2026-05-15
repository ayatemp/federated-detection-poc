#!/usr/bin/env python3
"""Run the 12-hour DQA-FedSTO-MoX long training trial.

This is the first serious long-run after the short DQA/MoE/judger probes.
The design keeps the parts that consistently helped:

* local EMA pseudo labels;
* DQA quality/class-coverage filtering;
* conservative source/server anchoring;
* MoE router/expert specialization;
* BN/MoE soft mixing as the stabilizer discovered in the judger runs.

The run is intentionally not a diagnostic 1-2 round probe.  It skips only
warmup by loading the existing 50-epoch warmup checkpoint, then spends the
budget on a 30-round two-stage learning schedule that should fit roughly in
one overnight / 12-hour window on the current machine.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
AGG_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = AGG_ROOT.parent
REPO_ROOT = PROJECT_ROOT.parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_scene_daynight_dqa_18_client_balanced_single_injection_dqamox.py"
SOURCE_WORKSPACE = PROJECT_ROOT / "output" / "08_full_latent_dqamox_from_warmup"
DEFAULT_WARMUP = SOURCE_WORKSPACE / "checkpoints" / "round000_latent_dqamox_warmup.pt"
DEFAULT_WORKSPACE = AGG_ROOT / "output" / "45_dqa_fedsto_mox_long_12h"
FINAL_METRICS = "18_client_balanced_single_injection_dqamox_final_metrics.csv"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def notify(message: str, title: str, enabled: bool) -> None:
    if not enabled:
        return
    try:
        from notebook_notify import notify_discord

        notify_discord(message, title=title, fail_silently=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}")


def read_metrics(workspace: Path) -> tuple[list[dict[str, str]], dict[str, str] | None, dict[str, str] | None]:
    metrics_csv = workspace / "stats" / FINAL_METRICS
    if not metrics_csv.exists():
        return [], None, None
    with metrics_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    warmup = next((row for row in rows if row.get("checkpoint_label") == "warmup_global"), None)
    scored: list[tuple[float, float, dict[str, str]]] = []
    for row in rows:
        try:
            scored.append((float(row.get("map50") or 0.0), float(row.get("map50_95") or 0.0), row))
        except ValueError:
            continue
    best = max(scored, key=lambda item: (item[0], item[1]))[2] if scored else None
    return rows, warmup, best


def as_float(row: dict[str, str] | None, key: str) -> float:
    if not row:
        return 0.0
    try:
        return float(row.get(key) or 0.0)
    except ValueError:
        return 0.0


def build_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python_executable),
        str(RUNNER),
        "--workspace-root",
        str(args.workspace_root),
        "--source-workspace",
        str(SOURCE_WORKSPACE),
        "--source-repair-baseline-rounds",
        "30",
        "--skip-warmup-training",
        "--warmup-checkpoint",
        str(args.warmup_checkpoint),
        "--warmup-epochs",
        "50",
        "--repair-baseline-rounds",
        "0",
        "--phase1-rounds",
        str(args.phase1_rounds),
        "--phase2-rounds",
        str(args.phase2_rounds),
        "--post-dqa-repair-rounds",
        "0",
        "--clients",
        "all",
        "--client-sampling-ratio",
        "0.333",
        "--client-sampling-seed",
        "450069",
        "--use-local-ema-teacher",
        "--client-limit",
        str(args.client_limit),
        "--max-images-per-client",
        "0",
        "--num-experts",
        "4",
        "--expert-count",
        "4",
        "--top-k",
        "2",
        "--router-temperature",
        "1.15",
        "--router-balance-weight",
        "0.035",
        "--router-entropy-weight",
        "0.0015",
        "--router-specialization-map",
        "hybrid_dqa4",
        "--router-specialization-weight",
        "0.105",
        "--router-specialization-max-weight",
        "0.075",
        "--router-specialization-min-quality",
        "0.58",
        "--router-specialization-min-boxes",
        "350",
        "--router-specialization-class-threshold",
        "0.26",
        "--phase1-train-scope",
        "backbone_moe_head",
        "--phase1-repair-train-scope",
        "backbone_moe_head",
        "--phase1-client-epochs",
        "1",
        "--phase1-client-lr",
        "0.00018",
        "--phase1-source-repeat",
        "2",
        "--phase1-pseudo-repeat",
        "2",
        "--phase1-loss-box",
        "0.00035",
        "--late-phase1-client-lr",
        "0.00014",
        "--late-phase1-source-repeat",
        "1",
        "--late-phase1-pseudo-repeat",
        "2",
        "--late-phase1-loss-box",
        "0.00025",
        "--phase2-train-scope",
        "all",
        "--phase2-repair-train-scope",
        "all",
        "--phase2-client-epochs",
        "1",
        "--phase2-client-lr",
        "0.000035",
        "--phase2-source-repeat",
        "1",
        "--phase2-pseudo-repeat",
        "1",
        "--phase2-loss-box",
        "0.00008",
        "--orthogonal-weight",
        "0.00012",
        "--class-skew-residual",
        "--class-skew-orthogonal-weight",
        "0.00008",
        "--class-skew-srip-weight",
        "0.000015",
        "--class-skew-residual-weight",
        "0.025",
        "--server-repair-epochs",
        "1",
        "--server-repair-lr",
        "0.00012",
        "--server-repair-loss-box",
        "0.004",
        "--client-loss-cls",
        "0.36",
        "--client-loss-obj",
        "0.74",
        "--dqa-temperature",
        "0.85",
        "--dqa-uniform-mix",
        "0.08",
        "--dqa-classwise-blend",
        "0.25",
        "--dqa-stability-lambda",
        "0.55",
        "--dqa-server-anchor",
        "0.70",
        "--dqa-min-server-alpha",
        "0.62",
        "--dqa-residual-blend",
        "0.04",
        "--dqa-bn-blend",
        "0.14",
        "--dqa-moe-expert-blend",
        "0.12",
        "--dqa-moe-router-blend",
        "0.18",
        "--late-dqa-server-anchor",
        "0.52",
        "--late-dqa-min-server-alpha",
        "0.46",
        "--late-dqa-residual-blend",
        "0.06",
        "--curriculum-start-round",
        "13",
        "--expert-keep-fraction",
        "0.58",
        "--expert-max-class-fraction",
        "0.24",
        "--actual-max-class-fraction",
        "0.34",
        "--late-expert-keep-fraction",
        "0.82",
        "--late-expert-max-class-fraction",
        "0.32",
        "--late-actual-max-class-fraction",
        "0.46",
        "--load-bias-strength",
        "0.28",
        "--min-views",
        "2",
        "--min-models",
        "0",
        "--min-score",
        "0.30",
        "--min-stability",
        "0.68",
        "--late-min-score",
        "0.22",
        "--late-min-stability",
        "0.58",
        "--max-boxes-per-image",
        "8",
        "--max-class-fraction",
        "0.38",
        "--min-class-keep",
        "80",
        "--client-mixup",
        "0.05",
        "--client-mosaic",
        "0.40",
        "--client-scale",
        "0.12",
        "--client-hsv-s",
        "0.08",
        "--client-hsv-v",
        "0.08",
        "--style-source-repeat",
        "1",
        "--style-source-limit",
        "1200",
        "--style-beta",
        "0.0035",
        "--style-imgsz",
        "640",
        "--style-seed",
        "450201",
        "--imgsz",
        str(args.imgsz),
        "--pseudo-imgsz",
        str(args.pseudo_imgsz),
        "--batch-size",
        str(args.batch_size),
        "--val-batch-size",
        str(args.val_batch_size),
        "--workers",
        str(args.workers),
        "--gpus",
        str(args.gpus),
        "--master-port",
        str(args.master_port),
        "--target-map50",
        str(args.target_map50),
        "--estimated-phase1-round-minutes",
        "14",
        "--estimated-phase2-round-minutes",
        "17",
        "--estimated-eval-minutes",
        "60",
        "--evaluate",
        "--classwise",
        "--no-eval-plots",
        "--no-progress",
        "--notify",
        "--notify-start",
        "--notify-end",
        "--notify-progress",
        "--notify-first-progress-hours",
        "0.5",
        "--notify-progress-interval-hours",
        "3.0",
    ]


def write_report(args: argparse.Namespace, command: list[str], returncode: int, started: datetime, finished: datetime, log_path: Path) -> Path:
    rows, warmup, best = read_metrics(args.workspace_root)
    warm50 = as_float(warmup, "map50")
    best50 = as_float(best, "map50")
    best95 = as_float(best, "map50_95")
    gain = best50 - warm50 if warmup and best else 0.0
    report_path = args.workspace_root / "45_dqa_fedsto_mox_long_12h_report.md"
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": finished.isoformat(),
        "returncode": returncode,
        "runtime_seconds": round((finished - started).total_seconds(), 1),
        "workspace": str(args.workspace_root),
        "log": str(log_path),
        "metrics_csv": str(args.workspace_root / "stats" / FINAL_METRICS),
        "command": command,
        "warmup_map50": warm50,
        "best_label": best.get("checkpoint_label", "") if best else "",
        "best_map50": best50,
        "best_map50_95": best95,
        "gain_vs_warmup": gain,
    }
    (args.workspace_root / "45_dqa_fedsto_mox_long_12h_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# DQA-FedSTO-MoX Long 12h",
        "",
        f"- started_utc: {started.isoformat()}",
        f"- finished_utc: {finished.isoformat()}",
        f"- runtime_seconds: {manifest['runtime_seconds']}",
        f"- returncode: {returncode}",
        f"- workspace: {args.workspace_root}",
        f"- log: {log_path}",
        "",
        "## Design",
        "",
        "- 30 FL rounds: 18 selective rounds + 12 full/orthogonal rounds",
        "- local EMA pseudo labeler + high-resolution pseudo generation",
        "- DQA pseudo quality, class coverage, stability gates",
        "- hybrid domain/class router specialization",
        "- conservative DQA server anchoring with BN/MoE soft residuals",
        "- MixPL-inspired mild mosaic/mixup for pseudo-label robustness",
        "",
        "## Result",
        "",
        f"- warmup mAP50: {warm50:.6f}" if warmup else "- warmup mAP50: missing",
        f"- best: {best.get('checkpoint_label', '')} mAP50={best50:.6f}, mAP50:95={best95:.6f}" if best else "- best: missing",
        f"- gain_vs_warmup: {gain:+.6f}" if warmup and best else "- gain_vs_warmup: missing",
        "",
        "## Command",
        "",
        "```bash",
        " ".join(command),
        "```",
        "",
        "## Final Metrics",
        "",
        "| checkpoint | mAP50 | mAP50:95 | precision | recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('checkpoint_label', '')} | {row.get('map50', '')} | "
            f"{row.get('map50_95', '')} | {row.get('precision', '')} | {row.get('recall', '')} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--warmup-checkpoint", type=Path, default=DEFAULT_WARMUP)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--phase1-rounds", type=int, default=18)
    parser.add_argument("--phase2-rounds", type=int, default=12)
    parser.add_argument("--client-limit", type=int, default=2200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--pseudo-imgsz", type=int, default=1152)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=40145)
    parser.add_argument("--target-map50", type=float, default=0.60)
    parser.add_argument("--no-discord", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.warmup_checkpoint = args.warmup_checkpoint.expanduser().resolve()
    if not args.warmup_checkpoint.exists():
        raise FileNotFoundError(args.warmup_checkpoint)
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    log_dir = AGG_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    log_path = log_dir / f"45_dqa_fedsto_mox_long_12h_{started.strftime('%Y%m%d_%H%M%S')}.log"
    command = build_command(args)
    intro = "\n".join(
        [
            "45 DQA-FedSTO-MoX Long 12h started",
            f"- workspace={args.workspace_root}",
            f"- rounds=phase1:{args.phase1_rounds} phase2:{args.phase2_rounds}",
            f"- target mAP50={args.target_map50}",
            f"- log={log_path}",
        ]
    )
    print(intro)
    notify(intro, "DQA-MoX 45 started", enabled=not args.no_discord)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(command) + "\n\n")
        proc = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    finished = datetime.now(timezone.utc)
    report_path = write_report(args, command, proc.returncode, started, finished, log_path)
    _, warmup, best = read_metrics(args.workspace_root)
    warm50 = as_float(warmup, "map50")
    best50 = as_float(best, "map50")
    best95 = as_float(best, "map50_95")
    message = "\n".join(
        [
            "45 DQA-FedSTO-MoX Long 12h finished",
            f"- returncode={proc.returncode}",
            f"- best={best.get('checkpoint_label', '') if best else 'missing'} mAP50={best50:.6f} mAP50:95={best95:.6f}",
            f"- warmup={warm50:.6f} gain={best50 - warm50:+.6f}" if warmup and best else "- gain=missing",
            f"- report={report_path}",
        ]
    )
    print(message)
    notify(message, "DQA-MoX 45 result", enabled=not args.no_discord)
    return 0 if proc.returncode == 0 else proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
