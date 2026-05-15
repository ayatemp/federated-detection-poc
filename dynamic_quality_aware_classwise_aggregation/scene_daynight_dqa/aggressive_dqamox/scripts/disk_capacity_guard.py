#!/usr/bin/env python3
"""Disk capacity guard for long DQA-MoX loops.

The training scripts can easily write tens of GiB of checkpoints and temporary
visualization artifacts.  This guard keeps the current long run away from the
`PytorchStreamWriter failed writing file` failure mode by pruning artifacts that
are safe to regenerate: old `runs/`, old `pseudo_dataset/`, trash, and temporary
smoke-test folders.  It avoids deleting the active paper-round trial workspace.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = ROOT / "dynamic_quality_aware_classwise_aggregation" / "scene_daynight_dqa"
AGG_ROOT = PROJECT_ROOT / "aggressive_dqamox"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def free_gib(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 1024**3


def remove_path(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        before = path.stat().st_size if path.is_file() else 0
    except OSError:
        before = 0
    if path.is_dir():
        # Directory byte accounting is expensive; use a conservative return and
        # rely on fresh free-space measurements after deletion.
        shutil.rmtree(path, ignore_errors=True)
        return before
    path.unlink(missing_ok=True)
    return before


def notify(message: str, *, title: str = "DQA-MoX disk guard") -> None:
    try:
        from notebook_notify import notify_discord

        notify_discord(message, title=title, fail_silently=True, include_default_context=True)
    except Exception as exc:  # noqa: BLE001
        print(f"notify failed: {exc}", flush=True)


def prune_safe_artifacts(active_workspace: Path | None) -> list[str]:
    removed: list[str] = []

    trash = Path("/app/.Trash-0")
    if trash.exists():
        for child in trash.iterdir():
            remove_path(child)
            removed.append(str(child))

    tmp_patterns = ("dqa08*", "dqa083*", "dqa12_rebalance_smoke*", "dqa03_smoke*", "sdn010_*")
    for pattern in tmp_patterns:
        for child in Path("/tmp").glob(pattern):
            remove_path(child)
            removed.append(str(child))

    safe_roots = [
        PROJECT_ROOT / "output",
        PROJECT_ROOT / "moe" / "output",
    ]
    safe_names = {"runs", "pseudo_dataset"}
    for root in safe_roots:
        if not root.exists():
            continue
        for child in root.rglob("*"):
            if child.name in safe_names and child.is_dir():
                remove_path(child)
                removed.append(str(child))

    # Completed/abandoned aggressive trial artifacts. Keep the active workspace
    # intact because training may still need current run directories. Also keep
    # long-loop roots intact: another supervisor may be running the next trial
    # there while this guard process is still alive.
    aggressive_output = AGG_ROOT / "output"
    long_loop_roots = {
        aggressive_output / "25_paper_round_until_target",
        aggressive_output / "26_autonomous_until_060",
        aggressive_output / "27_research_notebook_until_060",
    }
    if aggressive_output.exists():
        for child in aggressive_output.rglob("*"):
            if not child.is_dir() or child.name not in safe_names:
                continue
            if active_workspace and active_workspace in child.parents:
                continue
            if any(root in child.parents for root in long_loop_roots):
                continue
            remove_path(child)
            removed.append(str(child))

    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("/app"))
    parser.add_argument("--active-workspace", type=Path, default=None)
    parser.add_argument("--min-free-gib", type=float, default=80.0)
    parser.add_argument("--critical-free-gib", type=float, default=40.0)
    parser.add_argument("--interval-seconds", type=float, default=600.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    active_workspace = args.active_workspace.resolve() if args.active_workspace else None
    print(f"[{now()}] disk guard started active_workspace={active_workspace}", flush=True)
    while True:
        free_before = free_gib(args.path)
        print(f"[{now()}] free_gib={free_before:.1f}", flush=True)
        if free_before < args.min_free_gib:
            removed = prune_safe_artifacts(active_workspace)
            free_after = free_gib(args.path)
            message = (
                f"Disk guard cleaned safe artifacts.\n"
                f"- free_before: {free_before:.1f} GiB\n"
                f"- free_after: {free_after:.1f} GiB\n"
                f"- removed_count: {len(removed)}\n"
                f"- active_workspace: {active_workspace}"
            )
            print(message, flush=True)
            notify(message)
            if free_after < args.critical_free_gib:
                notify(
                    f"CRITICAL: free disk remains low after cleanup: {free_after:.1f} GiB.",
                    title="DQA-MoX disk guard critical",
                )
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
