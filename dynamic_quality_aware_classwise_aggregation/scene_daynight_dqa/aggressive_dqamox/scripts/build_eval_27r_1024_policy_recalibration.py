#!/usr/bin/env python3
"""Evaluate 27r 1024px routed-MoE policy recalibration."""

from __future__ import annotations

import argparse
import csv
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import build_eval_27h_model_level_moe as h27
import build_eval_27k_brightness_routed_moe as k27
import build_eval_27n_score_scaled_routed_moe as n27
import build_eval_27o_asymmetric_score_routed_moe as o27


SCRIPT_PATH = Path(__file__).resolve()
AGG_ROOT = SCRIPT_PATH.parents[1]
REPORTS_ROOT = AGG_ROOT / "reports"
DEFAULT_WORKSPACE = AGG_ROOT / "output" / "27r_1024_policy_recalibration"
NOTEBOOK_PATH = AGG_ROOT / "notebooks" / "research_loop_until_060" / "012_27r_1024_policy_recalibration.ipynb"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--device", default="")
    parser.add_argument("--target-map50", type=float, default=0.60)
    parser.add_argument("--previous-best-map50", type=float, default=0.522)
    parser.add_argument("--python-executable", type=Path, default=None)
    parser.add_argument("--no-discord", action="store_true")
    return parser.parse_args(argv)


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def notify(message: str, title: str) -> None:
    try:
        if str(h27.REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(h27.REPO_ROOT))
        from notebook_notify import notify_discord

        print(notify_discord(message, title=title, fail_silently=True))
    except Exception as exc:
        print(f"Discord notification skipped: {exc}")


def args_for_candidate(base_args: argparse.Namespace, candidate: dict) -> argparse.Namespace:
    args = argparse.Namespace(**vars(base_args))
    args.batch_size = int(candidate["batch_size"])
    args.imgsz = int(candidate["imgsz"])
    args.iou_thres = float(candidate["iou_thres"])
    return args


def make_variant(base: dict, label: str, iou: float, idea: str, augment: bool = False) -> dict:
    candidate = deepcopy(base)
    candidate["label"] = label
    candidate["imgsz"] = 1024
    candidate["batch_size"] = 8
    candidate["iou_thres"] = iou
    candidate["augment"] = augment
    candidate["idea"] = idea
    return candidate


def single_candidate(label: str, path: Path, iou: float, idea: str) -> dict:
    return {
        "label": label,
        "weights": [path.name],
        "paths": [path],
        "missing": [] if path.exists() else [str(path)],
        "augment": False,
        "imgsz": 1024,
        "batch_size": 8,
        "iou_thres": iou,
        "idea": idea,
    }


def make_candidates(workspace: Path) -> list[dict]:
    o_candidates = {candidate["label"]: candidate for candidate in o27.make_candidates(workspace)}
    n_candidates = {candidate["label"]: candidate for candidate in n27.make_candidates(workspace)}
    pool = k27.checkpoint_pool()
    best = o_candidates["day_light_night_hard"]
    return [
        make_variant(
            best,
            "best_day_light_1024_iou050",
            0.50,
            "Keep the 27q best routed policy but tighten NMS below 0.55 to reduce expert duplicates.",
        ),
        make_variant(
            best,
            "best_day_light_1024_iou045",
            0.45,
            "Aggressively suppress duplicate expert boxes at 1024px.",
        ),
        make_variant(
            best,
            "best_day_light_1024_iou055_tta",
            0.55,
            "Test-time augmentation on the current best 1024px routed MoE policy.",
            augment=True,
        ),
        make_variant(
            o_candidates["hard_reference_t0.32"],
            "hard_reference_1024_iou055",
            0.55,
            "Remove 27o asymmetric score scaling at high resolution to check whether scaling is now stale.",
        ),
        make_variant(
            n_candidates["anchor_dominant"],
            "anchor_dominant_1024_iou055",
            0.55,
            "Use the original 27n anchor-dominant scaling at 1024px.",
        ),
        make_variant(
            n_candidates["precision_clients_down"],
            "precision_clients_down_1024_iou055",
            0.55,
            "Demote client specialists more strongly to improve AP ranking at high resolution.",
        ),
        single_candidate(
            "single_25a_1024_iou055",
            pool["25a_r1_repair"],
            0.55,
            "Check whether the model-level MoE still beats the strongest single teacher at 1024px.",
        ),
    ]


def append_research_summary(
    *,
    workspace: Path,
    best: dict,
    status: str,
    args: argparse.Namespace,
) -> None:
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
                "status": status,
                "best_map50": best.get("map50", ""),
                "best_map50_95": best.get("map50_95", ""),
                "warmup_map50": args.previous_best_map50,
                "repair_map50": "",
                "dqa_aggregate_map50": "",
                "dqa_repair_map50": "",
                "workspace": str(workspace),
                "notebook": str(NOTEBOOK_PATH),
                "log": best.get("log_file", ""),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "rationale": (
                    "27q indicated a 1024px plateau around mAP50 0.522. 27r holds resolution fixed and recalibrates "
                    "the routed MoE policy: tighter NMS, TTA, hard routing, anchor-dominant scaling, and the strongest single teacher."
                ),
            }
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "stats").mkdir(parents=True, exist_ok=True)

    setup = h27.load_scene_setup(workspace)
    manifest = setup.build_data_lists()
    split_specs = h27.select_split_specs(manifest["paper_evaluation"], h27.PAPER_SPLITS)
    total_split = split_specs[-1]
    val_python = h27.select_val_python(args.python_executable)
    candidates = make_candidates(workspace)

    if not args.no_discord:
        notify(
            "27r started: 1024px routed-MoE policy recalibration. It tests tighter NMS, TTA, hard/anchor scaling, and the single 25a teacher.",
            "DQA-MoX 27r started",
        )

    fieldnames = [
        "candidate",
        "split",
        "augment",
        "num_weights",
        "weight_names",
        "idea",
        "imgsz",
        "nms_iou",
        "batch_size",
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
    split_cfg = h27.write_eval_config(setup, workspace, total_split, args_for_candidate(args, candidates[0]))
    total_csv = workspace / "stats" / "27r_1024_policy_recalibration_total_metrics.csv"
    rows: list[dict] = []
    status = "completed"
    for candidate in candidates:
        row = h27.run_val(
            val_python=val_python,
            workspace=workspace,
            split_cfg=split_cfg,
            split_name=total_split["name"],
            candidate=candidate,
            args=args_for_candidate(args, candidate),
        )
        row["imgsz"] = candidate["imgsz"]
        row["nms_iou"] = candidate["iou_thres"]
        row["batch_size"] = candidate["batch_size"]
        rows.append(row)
        write_rows(total_csv, rows, fieldnames)
        print(
            f"{row['candidate']} total status={row['status']} "
            f"augment={candidate['augment']} nms_iou={candidate['iou_thres']} "
            f"mAP50={row.get('map50', '')} mAP50:95={row.get('map50_95', '')}"
        )
        if row.get("status") == "ok" and row.get("map50") is not None and float(row["map50"]) >= args.target_map50:
            status = "target_reached"
            break

    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get("map50") is not None]
    if not ok_rows:
        raise RuntimeError("No successful 27r total evaluations.")
    best = max(ok_rows, key=lambda row: (float(row["map50"]), float(row.get("map50_95", 0.0))))
    append_research_summary(workspace=workspace, best=best, status=status, args=args)
    gain = float(best["map50"]) - args.previous_best_map50
    message = "\n".join(
        [
            f"27r finished. Best total candidate: {best['candidate']}",
            f"- total mAP50={best['map50']} / mAP50:95={best.get('map50_95', '')}",
            f"- previous best mAP50={args.previous_best_map50:.3f}; gain={gain:+.3f}",
            f"- status={status}",
            f"- total CSV: {total_csv}",
            "Decision: target 0.600 not reached; continue with a different strategy." if float(best["map50"]) < args.target_map50 else "Decision: target reached.",
        ]
    )
    print(message)
    if not args.no_discord:
        notify(message, "DQA-MoX 27r result")
    return 0 if float(best["map50"]) >= args.target_map50 else 2


if __name__ == "__main__":
    raise SystemExit(main())
