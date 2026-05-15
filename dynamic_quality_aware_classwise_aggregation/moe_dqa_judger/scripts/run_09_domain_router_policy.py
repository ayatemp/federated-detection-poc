#!/usr/bin/env python3
"""Build a domain-router policy from 08 domain-slice evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "output" / "09_domain_router_policy"
SOURCE_EVAL = PROJECT_ROOT / "output" / "08_domain_slice_judger" / "stats" / "08_domain_eval.csv"
SOURCE_SUMMARY = PROJECT_ROOT / "output" / "08_domain_slice_judger" / "stats" / "08_domain_summary.csv"


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


def aggregate_policy(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    scores = [parse_float(row["domain_score"]) for row in rows]
    night = [parse_float(row["domain_score"]) for row in rows if row["group"] == "night"]
    day = [parse_float(row["domain_score"]) for row in rows if row["group"] == "day"]
    map50 = [parse_float(row["map50"]) for row in rows]
    night_map50 = [parse_float(row["map50"]) for row in rows if row["group"] == "night"]
    return {
        "policy": label,
        "domain_mean_score": sum(scores) / len(scores),
        "day_mean_score": sum(day) / len(day),
        "night_mean_score": sum(night) / len(night),
        "worst_domain_score": min(scores),
        "domain_mean_map50": sum(map50) / len(map50),
        "night_mean_map50": sum(night_map50) / len(night_map50),
        "group_dro_score": 0.40 * (sum(scores) / len(scores)) + 0.40 * (sum(night) / len(night)) + 0.20 * min(scores),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    (args.workspace_root / "stats").mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.domain_eval)
    if not rows:
        raise FileNotFoundError(args.domain_eval)

    incumbent = [row for row in rows if row["label"] == "incumbent_r002"]
    domains = sorted({row["domain"] for row in rows})
    policy_rows: list[dict[str, Any]] = []
    for domain in domains:
        candidates = [row for row in rows if row["domain"] == domain and parse_float(row.get("total_score")) >= args.min_total_score]
        candidates.sort(key=lambda row: (parse_float(row["domain_score"]), parse_float(row.get("total_score"))), reverse=True)
        best = candidates[0]
        inc = next(row for row in incumbent if row["domain"] == domain)
        policy_rows.append(
            {
                "domain": domain,
                "group": best["group"],
                "selected_label": best["label"],
                "selected_score": parse_float(best["domain_score"]),
                "selected_map50": parse_float(best["map50"]),
                "selected_map50_95": parse_float(best["map50_95"]),
                "incumbent_score": parse_float(inc["domain_score"]),
                "incumbent_map50": parse_float(inc["map50"]),
                "incumbent_map50_95": parse_float(inc["map50_95"]),
                "delta_score": parse_float(best["domain_score"]) - parse_float(inc["domain_score"]),
                "delta_map50": parse_float(best["map50"]) - parse_float(inc["map50"]),
                "path": best["path"],
            }
        )
    write_csv(args.workspace_root / "stats" / "09_domain_router_policy.csv", policy_rows)
    selected_rows = []
    for item in policy_rows:
        selected_rows.append(next(row for row in rows if row["domain"] == item["domain"] and row["label"] == item["selected_label"]))
    summary_rows = [aggregate_policy(incumbent, "incumbent_r002"), aggregate_policy(selected_rows, "domain_router_oracle")]
    write_csv(args.workspace_root / "stats" / "09_domain_router_summary.csv", summary_rows)

    report = [
        "# DQA-SoftMoX Domain Router Policy 09",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- min_total_score: {args.min_total_score:.4f}",
        "",
        "## Policy",
        "",
        "| domain | selected | score | incumbent | delta | mAP50 | delta mAP50 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in policy_rows:
        report.append(
            f"| {row['domain']} | {row['selected_label']} | {row['selected_score']:.4f} | {row['incumbent_score']:.4f} | "
            f"{row['delta_score']:+.4f} | {row['selected_map50']:.3f} | {row['delta_map50']:+.3f} |"
        )
    report.extend(["", "## Summary", "", "| policy | mean | day | night | worst | DRO | night mAP50 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in summary_rows:
        report.append(
            f"| {row['policy']} | {row['domain_mean_score']:.4f} | {row['day_mean_score']:.4f} | {row['night_mean_score']:.4f} | "
            f"{row['worst_domain_score']:.4f} | {row['group_dro_score']:.4f} | {row['night_mean_map50']:.3f} |"
        )
    (args.workspace_root / "09_domain_router_policy_report.md").write_text("\n".join(report), encoding="utf-8")
    notify(
        "Domain-router policy finished\n"
        + "\n".join(
            f"- {row['domain']}: {row['selected_label']} delta_score={row['delta_score']:+.4f}, delta_mAP50={row['delta_map50']:+.3f}"
            for row in policy_rows
        )
        + "\nSummary: "
        + "; ".join(f"{row['policy']} DRO={row['group_dro_score']:.4f}, night={row['night_mean_score']:.4f}" for row in summary_rows),
        "DQA-SoftMoX 09 finished",
        args.notify_discord,
    )
    result = {"policy": policy_rows, "summary": summary_rows}
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--domain-eval", type=Path, default=SOURCE_EVAL)
    parser.add_argument("--min-total-score", type=float, default=0.5735)
    parser.add_argument("--notify-discord", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
