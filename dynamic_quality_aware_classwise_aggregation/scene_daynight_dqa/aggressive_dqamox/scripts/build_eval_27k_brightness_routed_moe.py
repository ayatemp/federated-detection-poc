#!/usr/bin/env python3
"""Evaluate 27k brightness-routed model-level MoE for scene-daynight DQA."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import build_eval_27h_model_level_moe as h27


SCRIPT_PATH = Path(__file__).resolve()
AGG_ROOT = SCRIPT_PATH.parents[1]
SCENE_ROOT = SCRIPT_PATH.parents[2]
REPORTS_ROOT = AGG_ROOT / "reports"
DEFAULT_WORKSPACE = AGG_ROOT / "output" / "27k_brightness_routed_moe"
NOTEBOOK_PATH = AGG_ROOT / "notebooks" / "research_loop_until_060" / "005_27k_brightness_routed_moe.ipynb"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--thresholds", default="0.32,0.34,0.36")
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


def checkpoint_pool() -> dict[str, Path]:
    pool = h27.checkpoint_pool()
    pool["25a_r1_repair"] = (
        AGG_ROOT
        / "output"
        / "25_paper_round_until_target"
        / "25a_fedmox50_sto20_30_top1"
        / "checkpoints"
        / "latent_dqamox_p1_round001_server_repair.pt"
    )
    return pool


def write_spec(workspace: Path, label: str, threshold: float, groups: dict[str, list[Path]]) -> Path:
    path = workspace / "stats" / "27k_routed_specs" / f"{label}_t{threshold:.2f}.routed.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "router": "mean_brightness_after_letterbox",
        "threshold": threshold,
        "groups": {name: [str(path.resolve()) for path in paths] for name, paths in groups.items()},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def make_routed_candidates(workspace: Path, pool: dict[str, Path], thresholds: list[float]) -> list[dict]:
    templates: list[tuple[str, str, dict[str, list[str]]]] = [
        (
            "day25a_night_broad25a",
            "day uses best single 25a; night uses broad model-level MoE plus 25a",
            {
                "day": ["25a_r1_repair"],
                "night": [
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
                ],
            },
        ),
        (
            "day25a_night_specialists",
            "day uses best single 25a; night uses only night specialists plus global anchors",
            {
                "day": ["25a_r1_repair"],
                "night": [
                    "25a_r1_repair",
                    "warmup",
                    "27g_repair",
                    "27g_client1_highway_night",
                    "27g_client3_citystreet_night",
                    "27g_client5_residential_night",
                ],
            },
        ),
        (
            "day_specialists_night_specialists",
            "day and night use disjoint client-specialist pools with 25a anchor",
            {
                "day": [
                    "25a_r1_repair",
                    "warmup",
                    "27d_repair",
                    "27e_repair",
                    "27g_client0_highway_day",
                    "27g_client2_citystreet_day",
                    "27g_client4_residential_day",
                ],
                "night": [
                    "25a_r1_repair",
                    "warmup",
                    "27g_repair",
                    "27g_client1_highway_night",
                    "27g_client3_citystreet_night",
                    "27g_client5_residential_night",
                ],
            },
        ),
    ]
    candidates = [
        {
            "label": "25a_single",
            "weights": ["25a_r1_repair"],
            "paths": [pool["25a_r1_repair"]],
            "missing": [] if pool["25a_r1_repair"].exists() else ["25a_r1_repair"],
            "augment": False,
            "idea": "best single checkpoint from 27i",
        }
    ]
    for threshold in thresholds:
        for label, idea, group_names in templates:
            missing = sorted({name for names in group_names.values() for name in names if not pool[name].exists()})
            groups = {name: [pool[item] for item in names] for name, names in group_names.items()}
            spec_path = write_spec(workspace, label, threshold, groups)
            candidates.append(
                {
                    "label": f"{label}_t{threshold:.2f}",
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
                    "SEAS-GMoE/HI-MoE motivate decoupling routing from noisy pseudo-label gradients. "
                    "27k evaluates a brightness-routed model-level MoE: day images use the strongest "
                    "single teacher, night images use specialist model pools."
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
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    pool = checkpoint_pool()
    candidates = make_routed_candidates(workspace, pool, thresholds)
    manifest_path = workspace / "stats" / "27k_brightness_routed_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "workspace": str(workspace),
                "val_python": str(val_python),
                "papers": [
                    "HI-MoE: Hierarchical Instance-Conditioned Mixture-of-Experts for Object Detection (arXiv:2604.04908)",
                    "Domain-Specialized Object Detection via Model-Level Mixtures of Experts (arXiv:2604.18256)",
                    "SEAS-GMoE: decoupling MoE routing from gradient noise (Expert Systems with Applications, 2026)",
                ],
                "candidates": [
                    {
                        "label": candidate["label"],
                        "paths": [str(path) for path in candidate["paths"]],
                        "missing": candidate["missing"],
                        "idea": candidate["idea"],
                    }
                    for candidate in candidates
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    if not args.no_discord:
        notify(
            "27k started: brightness-routed model-level MoE. Total split first; routes day-like images to 25a and night-like images to specialist ensembles.",
            "DQA-MoX 27k started",
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
    total_csv = workspace / "stats" / "27k_brightness_routed_total_metrics.csv"
    write_rows(total_csv, total_rows, fieldnames)

    ok_rows = [row for row in total_rows if row.get("status") == "ok" and row.get("map50") is not None]
    if not ok_rows:
        raise RuntimeError("No successful 27k total evaluations.")
    warmup = next((row for row in ok_rows if row["candidate"] == "25a_single"), None)
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
    full_csv = workspace / "stats" / "27k_brightness_routed_best_split_metrics.csv"
    write_rows(full_csv, full_rows, fieldnames)

    warmup_map50 = float(warmup["map50"]) if warmup is not None else 0.0
    gain = float(best["map50"]) - warmup_map50
    message = "\n".join(
        [
            f"27k finished. Best total candidate: {best['candidate']}",
            f"- total mAP50={best['map50']} / mAP50:95={best.get('map50_95', '')}",
            f"- 25a_single mAP50={warmup_map50:.3f}; gain={gain:+.3f}",
            f"- total CSV: {total_csv}",
            f"- full split CSV: {full_csv}",
            "Decision: target 0.600 not reached; continue with a different strategy." if float(best["map50"]) < args.target_map50 else "Decision: target reached.",
        ]
    )
    print(message)
    if not args.no_discord:
        notify(message, "DQA-MoX 27k result")
    return 0 if float(best["map50"]) >= args.target_map50 else 2


if __name__ == "__main__":
    raise SystemExit(main())
