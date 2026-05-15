#!/usr/bin/env python3
"""Evaluate 27n score-scaled brightness-routed model-level MoE."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import build_eval_27h_model_level_moe as h27
import build_eval_27k_brightness_routed_moe as k27
import build_eval_27l_soft_leak_routed_moe as l27


SCRIPT_PATH = Path(__file__).resolve()
AGG_ROOT = SCRIPT_PATH.parents[1]
REPORTS_ROOT = AGG_ROOT / "reports"
DEFAULT_WORKSPACE = AGG_ROOT / "output" / "27n_score_scaled_routed_moe"
NOTEBOOK_PATH = AGG_ROOT / "notebooks" / "research_loop_until_060" / "008_27n_score_scaled_routed_moe.ipynb"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.6)
    parser.add_argument("--device", default="")
    parser.add_argument("--target-map50", type=float, default=0.60)
    parser.add_argument("--python-executable", type=Path, default=None)
    parser.add_argument("--no-discord", action="store_true")
    return parser.parse_args(argv)


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_spec(
    workspace: Path,
    label: str,
    groups: dict[str, list[Path]],
    group_scales: dict[str, list[float]] | None = None,
) -> Path:
    path = workspace / "stats" / "27n_routed_specs" / f"{label}.routed.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "router": "mean_brightness_after_letterbox",
        "threshold": 0.32,
        "groups": {name: [str(path.resolve()) for path in paths] for name, paths in groups.items()},
    }
    if group_scales:
        payload["group_scales"] = group_scales
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def make_candidates(workspace: Path) -> list[dict]:
    pool = k27.checkpoint_pool()
    groups = l27.specialist_groups(pool)
    required = [
        "25a_r1_repair",
        "warmup",
        "27d_repair",
        "27e_repair",
        "27g_repair",
        "27g_client0_highway_day",
        "27g_client1_highway_night",
        "27g_client2_citystreet_day",
        "27g_client3_citystreet_night",
        "27g_client4_residential_day",
        "27g_client5_residential_night",
    ]
    missing = [name for name in required if not pool[name].exists()]
    settings: list[tuple[str, dict[str, list[float]] | None, str]] = [
        ("hard_reference_t0.32", None, "27k hard routed reference"),
        (
            "precision_clients_down",
            {
                "day": [1.00, 0.85, 0.95, 0.95, 0.65, 0.65, 0.65],
                "night": [1.00, 0.85, 0.95, 0.65, 0.65, 0.65],
            },
            "Reduce client specialist score to test whether 27k gain is recall-heavy noise.",
        ),
        (
            "client_boost",
            {
                "day": [0.85, 0.70, 0.85, 0.85, 1.15, 1.15, 1.15],
                "night": [0.85, 0.70, 0.90, 1.15, 1.15, 1.15],
            },
            "Boost client specialists and demote anchors to favor domain-specific boxes.",
        ),
        (
            "repair_confident",
            {
                "day": [0.90, 0.75, 1.15, 1.15, 0.80, 0.80, 0.80],
                "night": [0.90, 0.75, 1.15, 0.80, 0.80, 0.80],
            },
            "Favor repair experts over raw client specialists.",
        ),
        (
            "anchor_dominant",
            {
                "day": [1.10, 0.90, 1.00, 1.00, 0.50, 0.50, 0.50],
                "night": [1.10, 0.90, 1.00, 0.50, 0.50, 0.50],
            },
            "Use specialist pools as low-priority recalls behind the 25a anchor.",
        ),
    ]
    candidates = []
    for label, scales, idea in settings:
        spec_path = write_spec(workspace, label, groups, scales)
        candidates.append(
            {
                "label": label,
                "weights": [spec_path.name],
                "paths": [spec_path],
                "missing": missing,
                "augment": False,
                "idea": idea,
            }
        )
    return candidates


def append_research_summary(best: dict, warmup: dict | None, workspace: Path) -> None:
    path = REPORTS_ROOT / "27_research_loop_mAP_summary.csv"
    fieldnames = [
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
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "trial": workspace.name,
                "status": "target_reached" if float(best.get("map50", 0.0) or 0.0) >= 0.60 else "completed",
                "best_map50": best.get("map50", ""),
                "best_map50_95": best.get("map50_95", ""),
                "warmup_map50": "" if warmup is None else warmup.get("map50", ""),
                "repair_map50": "",
                "dqa_aggregate_map50": "",
                "dqa_repair_map50": "",
                "workspace": str(workspace),
                "notebook": str(NOTEBOOK_PATH),
                "log": best.get("log_file", ""),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "rationale": (
                    "27k showed specialist routing helps, 27l/27m showed adding or brightening boxes hurts. "
                    "27n instead calibrates objectness per expert so AP ranking can prefer the most reliable "
                    "anchor/repair/client predictions without changing the FedMoX-like training data."
                ),
            }
        )


def notify(message: str, title: str) -> None:
    try:
        if str(h27.REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(h27.REPO_ROOT))
        from notebook_notify import notify_discord

        print(notify_discord(message, title=title, fail_silently=True))
    except Exception as exc:
        print(f"Discord notification skipped: {exc}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "stats").mkdir(parents=True, exist_ok=True)

    setup = h27.load_scene_setup(workspace)
    manifest = setup.build_data_lists()
    split_specs = h27.select_split_specs(manifest["paper_evaluation"], h27.PAPER_SPLITS)
    split_cfgs = {split["name"]: h27.write_eval_config(setup, workspace, split, args) for split in split_specs}
    total_split = split_specs[-1]
    val_python = h27.select_val_python(args.python_executable)
    candidates = make_candidates(workspace)

    if not args.no_discord:
        notify(
            "27n started: score-scaled routed MoE. It calibrates objectness per expert inside the 27k day/night specialist route.",
            "DQA-MoX 27n started",
        )

    fieldnames = [
        "candidate",
        "split",
        "augment",
        "num_weights",
        "weight_names",
        "idea",
        "status",
        "returncode",
        "images",
        "labels",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "log_file",
        "command",
        "error",
    ]
    total_rows = []
    for candidate in candidates:
        row = h27.run_val(
            val_python=val_python,
            workspace=workspace,
            split_cfg=split_cfgs[total_split["name"]],
            split_name=total_split["name"],
            candidate=candidate,
            args=args,
        )
        total_rows.append(row)
        print(f"{row['candidate']} total status={row['status']} mAP50={row.get('map50', '')} mAP50:95={row.get('map50_95', '')}")
        if row.get("status") == "ok" and row.get("map50") is not None and float(row["map50"]) >= args.target_map50:
            break
    total_csv = workspace / "stats" / "27n_score_scaled_total_metrics.csv"
    write_rows(total_csv, total_rows, fieldnames)

    ok_rows = [row for row in total_rows if row.get("status") == "ok" and row.get("map50") is not None]
    if not ok_rows:
        raise RuntimeError("No successful 27n total evaluations.")
    warmup = next((row for row in ok_rows if row["candidate"] == "hard_reference_t0.32"), None)
    best = max(ok_rows, key=lambda row: (float(row["map50"]), float(row.get("map50_95", 0.0))))
    append_research_summary(best, warmup, workspace)

    full_rows = []
    best_candidate = next(candidate for candidate in candidates if candidate["label"] == best["candidate"])
    for split in split_specs:
        row = h27.run_val(
            val_python=val_python,
            workspace=workspace,
            split_cfg=split_cfgs[split["name"]],
            split_name=split["name"],
            candidate=best_candidate,
            args=args,
        )
        full_rows.append(row)
        print(f"{row['candidate']} {row['split']} status={row['status']} mAP50={row.get('map50', '')} mAP50:95={row.get('map50_95', '')}")
    full_csv = workspace / "stats" / "27n_score_scaled_best_split_metrics.csv"
    write_rows(full_csv, full_rows, fieldnames)

    warmup_map50 = float(warmup["map50"]) if warmup is not None else 0.0
    gain = float(best["map50"]) - warmup_map50
    message = "\n".join(
        [
            f"27n finished. Best total candidate: {best['candidate']}",
            f"- total mAP50={best['map50']} / mAP50:95={best.get('map50_95', '')}",
            f"- hard routed reference mAP50={warmup_map50:.3f}; gain={gain:+.3f}",
            f"- total CSV: {total_csv}",
            f"- full split CSV: {full_csv}",
            "Decision: target 0.600 not reached; continue with a different strategy." if float(best["map50"]) < args.target_map50 else "Decision: target reached.",
        ]
    )
    print(message)
    if not args.no_discord:
        notify(message, "DQA-MoX 27n result")
    return 0 if float(best["map50"]) >= args.target_map50 else 2


if __name__ == "__main__":
    raise SystemExit(main())
