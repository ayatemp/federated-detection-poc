#!/usr/bin/env python3
"""Evaluate 27p high-resolution and NMS-calibrated routed MoE."""

from __future__ import annotations

import argparse
import csv
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import build_eval_27h_model_level_moe as h27
import build_eval_27o_asymmetric_score_routed_moe as o27


SCRIPT_PATH = Path(__file__).resolve()
AGG_ROOT = SCRIPT_PATH.parents[1]
REPORTS_ROOT = AGG_ROOT / "reports"
DEFAULT_WORKSPACE = AGG_ROOT / "output" / "27p_highres_nms_routed_moe"
NOTEBOOK_PATH = AGG_ROOT / "notebooks" / "research_loop_until_060" / "010_27p_highres_nms_routed_moe.ipynb"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--device", default="")
    parser.add_argument("--target-map50", type=float, default=0.60)
    parser.add_argument("--min-highres-gain", type=float, default=0.004)
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
            "baseline_640_iou060",
            640,
            16,
            0.60,
            "27o best routed MoE, same evaluation policy as the paper protocol.",
        ),
        make_variant(
            base,
            "nms_tight_640_iou055",
            640,
            16,
            0.55,
            "Keep the routed experts but suppress dense duplicate boxes more aggressively.",
        ),
        make_variant(
            base,
            "nms_loose_640_iou065",
            640,
            16,
            0.65,
            "Keep more overlapping expert boxes for crowded small-object regions.",
        ),
        make_variant(
            base,
            "highres_768_iou060",
            768,
            10,
            0.60,
            "Increase inference resolution to recover small/distant objects without retraining.",
        ),
        make_variant(
            base,
            "highres_896_iou060",
            896,
            8,
            0.60,
            "Push resolution further while keeping the 27o routed expert policy fixed.",
        ),
        make_variant(
            base,
            "highres_896_iou065",
            896,
            8,
            0.65,
            "Combine high-resolution recall with looser NMS for dense traffic scenes.",
        ),
        make_variant(
            base,
            "highres_1024_iou065",
            1024,
            4,
            0.65,
            "Only run if high-resolution variants show useful gain; final aggressive test-time MoE policy.",
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
    baseline: dict | None,
    status: str,
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
                "warmup_map50": "" if baseline is None else baseline.get("map50", ""),
                "repair_map50": "",
                "dqa_aggregate_map50": "",
                "dqa_repair_map50": "",
                "workspace": str(workspace),
                "notebook": str(NOTEBOOK_PATH),
                "log": best.get("log_file", ""),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "rationale": (
                    "27k-27o showed that model-level routed MoE plus score calibration is the only path with positive gain, "
                    "while soft leak and image enhancement add boxes without improving AP. 27p keeps the best routed experts "
                    "fixed and searches high-resolution/NMS test-time policy for small-object recovery."
                ),
            }
        )


def run_total_candidates(
    *,
    workspace: Path,
    split_cfg: Path,
    total_split: dict,
    candidates: list[dict],
    args: argparse.Namespace,
    val_python: Path,
    fieldnames: list[str],
) -> tuple[list[dict], str]:
    rows: list[dict] = []
    total_csv = workspace / "stats" / "27p_highres_nms_total_metrics.csv"
    baseline_map50: float | None = None
    status = "completed"
    for index, candidate in enumerate(candidates):
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
            f"imgsz={candidate['imgsz']} nms_iou={candidate['iou_thres']} "
            f"mAP50={row.get('map50', '')} mAP50:95={row.get('map50_95', '')}"
        )
        if row.get("status") == "ok" and row.get("map50") is not None:
            score = float(row["map50"])
            if candidate["label"] == "baseline_640_iou060":
                baseline_map50 = score
            if score >= args.target_map50:
                break

        completed_labels = {r["candidate"] for r in rows}
        if (
            index >= 4
            and baseline_map50 is not None
            and "highres_768_iou060" in completed_labels
            and "highres_896_iou060" in completed_labels
        ):
            ok_scores = [float(r["map50"]) for r in rows if r.get("status") == "ok" and r.get("map50") is not None]
            if ok_scores and max(ok_scores) < baseline_map50 + args.min_highres_gain:
                status = "aborted_no_highres_gain"
                print(
                    "Stopping 27p early: high-resolution/NMS candidates did not exceed "
                    f"baseline by {args.min_highres_gain:.3f} mAP50."
                )
                break
    return rows, status


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
                    "27p started: high-resolution/NMS-calibrated routed MoE.",
                    "Learning so far: hard day/night routing + score calibration is the only positive path (0.464 -> 0.472),",
                    "while soft leak and night enhancement added recall but did not improve AP.",
                ]
            ),
            "DQA-MoX 27p started",
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
    total_rows, status = run_total_candidates(
        workspace=workspace,
        split_cfg=split_cfgs[total_split["name"]],
        total_split=total_split,
        candidates=candidates,
        args=args,
        val_python=val_python,
        fieldnames=fieldnames,
    )
    ok_rows = [row for row in total_rows if row.get("status") == "ok" and row.get("map50") is not None]
    if not ok_rows:
        raise RuntimeError("No successful 27p total evaluations.")
    baseline = next((row for row in ok_rows if row["candidate"] == "baseline_640_iou060"), None)
    best = max(ok_rows, key=lambda row: (float(row["map50"]), float(row.get("map50_95", 0.0))))
    if float(best["map50"]) >= args.target_map50:
        status = "target_reached"
    append_research_summary(workspace=workspace, best=best, baseline=baseline, status=status)

    best_candidate = next(candidate for candidate in candidates if candidate["label"] == best["candidate"])
    full_rows = []
    if status != "aborted_no_highres_gain" or best["candidate"] != "baseline_640_iou060":
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
            print(
                f"{row['candidate']} {row['split']} status={row['status']} "
                f"mAP50={row.get('map50', '')} mAP50:95={row.get('map50_95', '')}"
            )
        write_rows(workspace / "stats" / "27p_highres_nms_best_split_metrics.csv", full_rows, fieldnames)

    baseline_map50 = float(baseline["map50"]) if baseline is not None else 0.0
    gain = float(best["map50"]) - baseline_map50
    total_csv = workspace / "stats" / "27p_highres_nms_total_metrics.csv"
    split_csv = workspace / "stats" / "27p_highres_nms_best_split_metrics.csv"
    message = "\n".join(
        [
            f"27p finished. Best total candidate: {best['candidate']}",
            f"- total mAP50={best['map50']} / mAP50:95={best.get('map50_95', '')}",
            f"- 27o-policy baseline mAP50={baseline_map50:.3f}; gain={gain:+.3f}",
            f"- status={status}",
            f"- total CSV: {total_csv}",
            f"- split CSV: {split_csv if split_csv.exists() else 'not generated because early stop kept the baseline'}",
            "Decision: target 0.600 not reached; continue with a different strategy." if float(best["map50"]) < args.target_map50 else "Decision: target reached.",
        ]
    )
    print(message)
    if not args.no_discord:
        notify(message, "DQA-MoX 27p result")
    return 0 if float(best["map50"]) >= args.target_map50 else 2


if __name__ == "__main__":
    raise SystemExit(main())
