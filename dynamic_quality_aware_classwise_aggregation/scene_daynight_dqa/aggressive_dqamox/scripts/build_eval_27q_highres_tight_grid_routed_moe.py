#!/usr/bin/env python3
"""Evaluate 27q focused high-resolution tight-NMS routed MoE grid."""

from __future__ import annotations

import argparse
import csv
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import build_eval_27h_model_level_moe as h27
import build_eval_27o_asymmetric_score_routed_moe as o27
import build_eval_27p_highres_nms_routed_moe as p27


SCRIPT_PATH = Path(__file__).resolve()
AGG_ROOT = SCRIPT_PATH.parents[1]
REPORTS_ROOT = AGG_ROOT / "reports"
DEFAULT_WORKSPACE = AGG_ROOT / "output" / "27q_highres_tight_grid_routed_moe"
NOTEBOOK_PATH = AGG_ROOT / "notebooks" / "research_loop_until_060" / "011_27q_highres_tight_grid_routed_moe.ipynb"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--device", default="")
    parser.add_argument("--target-map50", type=float, default=0.60)
    parser.add_argument("--previous-best-map50", type=float, default=0.517)
    parser.add_argument("--min-continue-gain", type=float, default=0.006)
    parser.add_argument("--python-executable", type=Path, default=None)
    parser.add_argument("--full-splits", action="store_true")
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


def routed_base_candidate(workspace: Path) -> dict:
    candidates = o27.make_candidates(workspace)
    return next(candidate for candidate in candidates if candidate["label"] == "day_light_night_hard")


def make_variant(base: dict, label: str, imgsz: int, batch_size: int, iou: float, idea: str) -> dict:
    candidate = deepcopy(base)
    candidate["label"] = label
    candidate["imgsz"] = imgsz
    candidate["batch_size"] = batch_size
    candidate["iou_thres"] = iou
    candidate["idea"] = idea
    return candidate


def make_candidates(base: dict) -> list[dict]:
    return [
        make_variant(
            base,
            "highres_896_iou055",
            896,
            8,
            0.55,
            "Check whether the 640px tight-NMS gain transfers to the 896px routed MoE.",
        ),
        make_variant(
            base,
            "highres_1024_iou060",
            1024,
            8,
            0.60,
            "Use the best 896px NMS policy at the next resolution step.",
        ),
        make_variant(
            base,
            "highres_1024_iou055",
            1024,
            8,
            0.55,
            "Tighter NMS at 1024px to test whether duplicate expert boxes are limiting AP.",
        ),
        make_variant(
            base,
            "highres_1152_iou060",
            1152,
            6,
            0.60,
            "Push the high-resolution scaling curve while keeping NMS at the best 896px setting.",
        ),
        make_variant(
            base,
            "highres_1152_iou055",
            1152,
            6,
            0.55,
            "High-resolution plus tighter NMS for crowded scenes.",
        ),
        make_variant(
            base,
            "highres_1280_iou060",
            1280,
            4,
            0.60,
            "Final large-image probe if 1152px still gives useful gain.",
        ),
    ]


def args_for_candidate(base_args: argparse.Namespace, candidate: dict) -> argparse.Namespace:
    args = argparse.Namespace(**vars(base_args))
    args.batch_size = int(candidate["batch_size"])
    args.imgsz = int(candidate["imgsz"])
    args.iou_thres = float(candidate["iou_thres"])
    return args


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
                    "27p showed the biggest jump so far: high-resolution routed MoE improves total mAP50 from 0.472 to 0.517. "
                    "Loose NMS hurt at 640/896, so 27q focuses on 0.55/0.60 NMS at 896-1280px and skips split evaluation until a stronger total candidate is found."
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
    base_candidate = routed_base_candidate(workspace)
    candidates = make_candidates(base_candidate)

    if not args.no_discord:
        notify(
            "\n".join(
                [
                    "27q started: focused high-resolution/tight-NMS routed MoE grid.",
                    "27p found 1024px+IoU0.65 total mAP50=0.517; now probing 0.55/0.60 at 896-1280px.",
                ]
            ),
            "DQA-MoX 27q started",
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
    split_cfgs = {
        split["name"]: h27.write_eval_config(
            setup,
            workspace,
            split,
            args_for_candidate(args, candidates[0]),
        )
        for split in split_specs
    }
    total_csv = workspace / "stats" / "27q_highres_tight_grid_total_metrics.csv"
    rows: list[dict] = []
    status = "completed"
    best_score = args.previous_best_map50
    for candidate in candidates:
        row = h27.run_val(
            val_python=val_python,
            workspace=workspace,
            split_cfg=split_cfgs[total_split["name"]],
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
            f"imgsz={candidate['imgsz']} nms_iou={candidate['iou_thres']} "
            f"mAP50={row.get('map50', '')} mAP50:95={row.get('map50_95', '')}"
        )
        if row.get("status") == "ok" and row.get("map50") is not None:
            best_score = max(best_score, float(row["map50"]))
            if float(row["map50"]) >= args.target_map50:
                status = "target_reached"
                break
        completed_labels = {r["candidate"] for r in rows}
        if (
            candidate["label"] == "highres_1152_iou055"
            and best_score < args.previous_best_map50 + args.min_continue_gain
            and "highres_1152_iou060" in completed_labels
        ):
            status = "aborted_1152_no_gain"
            print(
                "Stopping 27q before 1280px: 1152px did not improve previous best by "
                f"{args.min_continue_gain:.3f} mAP50."
            )
            break

    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get("map50") is not None]
    if not ok_rows:
        raise RuntimeError("No successful 27q total evaluations.")
    best = max(ok_rows, key=lambda row: (float(row["map50"]), float(row.get("map50_95", 0.0))))
    append_research_summary(workspace=workspace, best=best, status=status, args=args)

    full_csv = workspace / "stats" / "27q_highres_tight_grid_best_split_metrics.csv"
    if args.full_splits or status == "target_reached":
        best_candidate = next(candidate for candidate in candidates if candidate["label"] == best["candidate"])
        full_rows = []
        for split in split_specs:
            row = h27.run_val(
                val_python=val_python,
                workspace=workspace,
                split_cfg=split_cfgs[split["name"]],
                split_name=split["name"],
                candidate=best_candidate,
                args=args_for_candidate(args, best_candidate),
            )
            row["imgsz"] = best_candidate["imgsz"]
            row["nms_iou"] = best_candidate["iou_thres"]
            row["batch_size"] = best_candidate["batch_size"]
            full_rows.append(row)
            print(f"{row['candidate']} {row['split']} status={row['status']} mAP50={row.get('map50', '')} mAP50:95={row.get('map50_95', '')}")
        write_rows(full_csv, full_rows, fieldnames)

    gain = float(best["map50"]) - args.previous_best_map50
    message = "\n".join(
        [
            f"27q finished. Best total candidate: {best['candidate']}",
            f"- total mAP50={best['map50']} / mAP50:95={best.get('map50_95', '')}",
            f"- previous best mAP50={args.previous_best_map50:.3f}; gain={gain:+.3f}",
            f"- status={status}",
            f"- total CSV: {total_csv}",
            f"- split CSV: {full_csv if full_csv.exists() else 'not generated in fast total-search mode'}",
            "Decision: target 0.600 not reached; continue with a different strategy." if float(best["map50"]) < args.target_map50 else "Decision: target reached.",
        ]
    )
    print(message)
    if not args.no_discord:
        notify(message, "DQA-MoX 27q result")
    return 0 if float(best["map50"]) >= args.target_map50 else 2


if __name__ == "__main__":
    raise SystemExit(main())
