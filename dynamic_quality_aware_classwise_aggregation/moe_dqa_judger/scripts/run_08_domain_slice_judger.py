#!/usr/bin/env python3
"""Domain-slice judger for plateaued DQA-SoftMoX candidates.

07 found a useful anti-drift rule: keep the best incumbent and rebase later
deltas onto it.  Several candidates tie on total mAP, so this notebook asks a
more DQA-like question: which candidate is best across client/domain slices,
especially night slices that have been the weak point?

This is inspired by group-DRO/worst-group validation: choose not only by average
validation quality, but also by night and worst-domain quality.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "output" / "08_domain_slice_judger"
SOURCE_WORKSPACE = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation" / "moe_dqa" / "output" / "01_dqa_fedmox_yolo_full"
INCUMBENT = PROJECT_ROOT / "output" / "03_mix_judger_policy" / "candidates" / "r002_judger03_selected_r002.pt"

if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_02_mix_weight_optimizer as opt02  # noqa: E402


DOMAIN_LISTS = [
    "paper_eval_scene_daynight_highway_day_val.txt",
    "paper_eval_scene_daynight_highway_night_val.txt",
    "paper_eval_scene_daynight_citystreet_day_val.txt",
    "paper_eval_scene_daynight_citystreet_night_val.txt",
    "paper_eval_scene_daynight_residential_day_val.txt",
    "paper_eval_scene_daynight_residential_night_val.txt",
]


def notify(message: str, title: str, enabled: bool) -> None:
    if not enabled:
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from notebook_notify import notify_discord

        notify_discord(message, title=title, fail_silently=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notify skipped: {exc}")


def parse_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def domain_name(list_name: str) -> str:
    return list_name.replace("paper_eval_scene_daynight_", "").replace("_val.txt", "")


def domain_group(name: str) -> str:
    return "night" if name.endswith("_night") else "day"


def make_domain_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    source_cfg = args.source_workspace / "validation_reports" / "paper_protocol_configs" / "scene_daynight_total.yaml"
    cfg_template = yaml.safe_load(source_cfg.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for list_name in DOMAIN_LISTS:
        list_path = args.source_workspace / "data_lists" / list_name
        if not list_path.exists():
            continue
        name = domain_name(list_name)
        cfg = dict(cfg_template)
        cfg["Dataset"] = dict(cfg_template["Dataset"])
        cfg["Dataset"]["val"] = str(list_path.resolve())
        cfg["Dataset"]["test"] = str(list_path.resolve())
        cfg["Dataset"]["batch_size"] = int(args.val_batch_size)
        cfg["Dataset"]["workers"] = 0
        cfg["SSOD"] = {"train_domain": False}
        cfg_path = args.workspace_root / "configs" / f"{name}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        rows.append({"domain": name, "group": domain_group(name), "cfg": cfg_path, "list": list_path})
    return rows


def candidate_pool(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [
        {
            "label": "incumbent_r002",
            "source": "03_mix_judger_policy",
            "path": args.incumbent_path.expanduser().resolve(),
            "total_score": 0.57455,
            "total_map50": 0.462,
            "total_map50_95": 0.260,
        }
    ]
    full_path = PROJECT_ROOT / "output" / "07_incumbent_delta_judger" / "stats" / "07_full_eval.csv"
    for row in read_csv(full_path):
        path = Path(row.get("path", "")).resolve()
        if not path.exists():
            continue
        label = f"r{int(float(row.get('round', -1))):03d}_{row.get('candidate_id')}"
        rows.append(
            {
                "label": label,
                "source": "07_incumbent_delta",
                "path": path,
                "total_score": parse_float(row.get("score")),
                "total_map50": parse_float(row.get("map50")),
                "total_map50_95": parse_float(row.get("map50_95")),
            }
        )
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = str(row["path"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out[: args.max_candidates]


def existing_eval(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row.get("label", ""), row.get("domain", "")): row for row in read_csv(path)}


def evaluate(args: argparse.Namespace, candidates: list[dict[str, Any]], domains: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eval_path = args.workspace_root / "stats" / "08_domain_eval.csv"
    cache = existing_eval(eval_path) if args.resume and eval_path.exists() else {}
    rows: list[dict[str, Any]] = list(cache.values())
    for candidate in candidates:
        for domain in domains:
            key = (candidate["label"], domain["domain"])
            if key in cache:
                continue
            metrics = opt02.eval_checkpoint(Path(candidate["path"]), Path(domain["cfg"]), f"domain08_{candidate['label']}_{domain['domain']}", args)
            row = {**candidate, "domain": domain["domain"], "group": domain["group"], **metrics}
            row["domain_score"] = opt02.score_row(row)
            rows.append(row)
            write_csv(eval_path, rows)
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["label"]), []).append(row)
    out: list[dict[str, Any]] = []
    for label, group_rows in grouped.items():
        scores = [parse_float(row["domain_score"]) for row in group_rows]
        day = [parse_float(row["domain_score"]) for row in group_rows if row.get("group") == "day"]
        night = [parse_float(row["domain_score"]) for row in group_rows if row.get("group") == "night"]
        map50s = [parse_float(row["map50"]) for row in group_rows]
        night_map50s = [parse_float(row["map50"]) for row in group_rows if row.get("group") == "night"]
        first = group_rows[0]
        mean_score = sum(scores) / len(scores)
        night_mean = sum(night) / len(night) if night else math.nan
        day_mean = sum(day) / len(day) if day else math.nan
        worst = min(scores)
        row = {
            "label": label,
            "source": first.get("source", ""),
            "path": first.get("path", ""),
            "total_score": parse_float(first.get("total_score")),
            "total_map50": parse_float(first.get("total_map50")),
            "total_map50_95": parse_float(first.get("total_map50_95")),
            "domain_mean_score": mean_score,
            "day_mean_score": day_mean,
            "night_mean_score": night_mean,
            "worst_domain_score": worst,
            "domain_mean_map50": sum(map50s) / len(map50s),
            "night_mean_map50": sum(night_map50s) / len(night_map50s) if night_map50s else math.nan,
            "group_dro_score": 0.40 * mean_score + 0.40 * night_mean + 0.20 * worst,
        }
        out.append(row)
    out.sort(key=lambda row: parse_float(row["group_dro_score"]), reverse=True)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.source_workspace = args.source_workspace.expanduser().resolve()
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    (args.workspace_root / "stats").mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "dqa_softmox_domain_slice_judger_v1",
        "method": "group-DRO style domain/night slice selection over plateau candidates",
        "papers_used": ["Group DRO / worst-group validation", "Distributionally Robust Federated Averaging"],
    }
    (args.workspace_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    notify("Domain-slice judger started\nEvaluating plateau candidates on day/night scene slices.", "DQA-SoftMoX 08 started", args.notify_discord)
    domains = make_domain_configs(args)
    candidates = candidate_pool(args)
    write_csv(args.workspace_root / "stats" / "08_candidate_pool.csv", candidates)
    rows = evaluate(args, candidates, domains)
    summary = aggregate(rows)
    write_csv(args.workspace_root / "stats" / "08_domain_summary.csv", summary)

    report = [
        "# DQA-SoftMoX Domain-Slice Judger 08",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- candidates: {len(candidates)}",
        f"- domains: {len(domains)}",
        "",
        "| rank | candidate | total score | day score | night score | worst score | DRO score | night mAP50 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(summary, start=1):
        report.append(
            f"| {idx} | {row['label']} | {parse_float(row['total_score']):.4f} | {parse_float(row['day_mean_score']):.4f} | "
            f"{parse_float(row['night_mean_score']):.4f} | {parse_float(row['worst_domain_score']):.4f} | "
            f"{parse_float(row['group_dro_score']):.4f} | {parse_float(row['night_mean_map50']):.3f} |"
        )
    (args.workspace_root / "08_domain_slice_judger_report.md").write_text("\n".join(report), encoding="utf-8")
    notify(
        "Domain-slice judger finished\nTop candidates:\n"
        + "\n".join(
            f"- {row['label']}: DRO={parse_float(row['group_dro_score']):.4f}, night={parse_float(row['night_mean_score']):.4f}, worst={parse_float(row['worst_domain_score']):.4f}, total={parse_float(row['total_score']):.4f}"
            for row in summary[:5]
        ),
        "DQA-SoftMoX 08 finished",
        args.notify_discord,
    )
    result = {"manifest": manifest, "summary": summary, "rows": rows}
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-workspace", type=Path, default=SOURCE_WORKSPACE)
    parser.add_argument("--incumbent-path", type=Path, default=INCUMBENT)
    parser.add_argument("--max-candidates", type=int, default=9)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--notify-discord", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
