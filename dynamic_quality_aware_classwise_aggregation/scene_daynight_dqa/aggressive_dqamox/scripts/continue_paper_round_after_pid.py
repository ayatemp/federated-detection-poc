#!/usr/bin/env python3
"""Continue paper-round DQA-MoX trials after an active controller exits."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

from run_paper_round_dqamox_until_target import AGG_ROOT, REPORT_ROOT, REPO_ROOT, TRIALS


SUMMARY_PATH = REPORT_ROOT / "25_paper_round_until_target_summary.csv"


def notify(message: str, *, title: str) -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from notebook_notify import notify_discord

        print(notify_discord(message, title=title, fail_silently=True))
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def best_map50(rows: list[dict[str, str]]) -> float:
    best = -1.0
    for row in rows:
        try:
            value = float(row.get("best_map50") or "nan")
        except ValueError:
            continue
        if value == value:
            best = max(best, value)
    return best


def next_index(rows: list[dict[str, str]]) -> int:
    status_by_trial = {row.get("trial", ""): row.get("status", "") for row in rows}
    for idx, trial in enumerate(TRIALS):
        status = status_by_trial.get(trial.name)
        if not status or status.startswith("failed_rc_"):
            return idx
    return len(TRIALS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--target-map50", type=float, default=0.60)
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    notify(
        f"paper-round controller PID {args.pid} の終了後に、mAP50={args.target_map50:.3f} 未達なら自動継続します。",
        title="DQA-MoX auto-continue armed",
    )
    while True:
        try:
            Path(f"/proc/{args.pid}").stat()
        except FileNotFoundError:
            break
        time.sleep(args.poll_seconds)

    rows = read_rows(SUMMARY_PATH)
    best = best_map50(rows)
    if best >= args.target_map50:
        notify(
            f"paper-round controller 終了後チェック: target 到達済みです。best mAP50={best:.6f}",
            title="DQA-MoX target already reached",
        )
        return 0

    start = next_index(rows)
    if start >= len(TRIALS):
        notify(
            f"定義済み trial は完了しましたが target 未達です。best mAP50={best:.6f}。次の仮説追加が必要です。",
            title="DQA-MoX needs new trials",
        )
        return 2

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_paper_round_dqamox_until_target.py"),
        "--start-index",
        str(start),
        "--max-trials",
        "0",
        "--target-map50",
        str(args.target_map50),
        "--notify-first-progress-hours",
        "1",
        "--notify-progress-interval-hours",
        "1",
    ]
    if args.force:
        cmd.append("--force")
    notify(
        f"target 未達のため paper-round DQA-MoX を自動継続します。best mAP50={best:.6f}, start_index={start}",
        title="DQA-MoX auto-continue start",
    )
    return subprocess.call(cmd, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
