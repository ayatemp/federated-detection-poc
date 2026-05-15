#!/usr/bin/env python3
"""Build/evaluate repair-shielded local expert DQA candidates.

This is the full comparison follow-up to the 03 main experiment and the MoE 06
screening result.  It does not repeat warmup/server-repair/DQA training.
Instead it reuses the trained 03 checkpoints and builds deployable candidates
for the selected loop07 idea:

    repair the shared/source path, then add local pseudo-GT expert residuals
    after repair so the useful DQA signal is not overwritten.

The output table is directly comparable with the 03 final metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
PROTOCOL_VERSION = "scene_daynight_dqa_04_repair_shielded_local_expert_v1"

for path in (PROJECT_ROOT / "scripts", PROJECT_ROOT.parent, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_scene_daynight_dqa_01_0 as base01_0  # noqa: E402
import run_scene_daynight_dqa_03_main_experiment as main03  # noqa: E402


DEFAULT_SOURCE_WORKSPACE = PROJECT_ROOT / "output" / "03_main_bn_residual_dqa_experiment"
DEFAULT_WORKSPACE = PROJECT_ROOT / "output" / "04_repair_shielded_local_expert_dqa"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: Any) -> float | None:
    return main03.as_float(value)


def parse_betas(raw: str) -> list[float]:
    betas = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        betas.append(float(item))
    if not betas:
        raise ValueError("At least one candidate beta is required.")
    return betas


def round_tag(round_idx: int) -> str:
    return f"round{round_idx:03d}"


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def dqa_checkpoint_dir(args: argparse.Namespace) -> Path:
    return args.source_workspace / "bn_residual_dqa" / "checkpoints"


def source_checkpoint_paths(args: argparse.Namespace) -> dict[str, Path]:
    dqa_dir = dqa_checkpoint_dir(args)
    final_tag = round_tag(args.source_round)
    prev_tag = round_tag(args.source_round - 1)
    day_clients = [
        "client0_highway_day",
        "client2_citystreet_day",
        "client4_residential_day",
    ]
    return {
        "prev_repaired_shared": require_path(dqa_dir / f"{prev_tag}_bn_residual_dqa_server_repair.pt"),
        "final_aggregate": require_path(dqa_dir / f"{final_tag}_bn_residual_dqa_aggregate.pt"),
        "final_repaired_shared": require_path(dqa_dir / f"{final_tag}_bn_residual_dqa_server_repair.pt"),
        **{
            client: require_path(dqa_dir / f"{final_tag}_bn_residual_dqa_{client}.pt")
            for client in day_clients
        },
    }


def save_record(
    rows: list[dict[str, str]],
    label: str,
    path: Path,
    *,
    beta: float,
    variant: str,
) -> None:
    rows.append(
        {
            "condition": "repair_shielded_local_expert_dqa",
            "label": label,
            "kind": "aggregate",
            "round": "",
            "client": "",
            "variant": variant,
            "beta": f"{beta:.3f}",
            "path": str(path.resolve()),
        }
    )


def build_candidates(args: argparse.Namespace) -> list[dict[str, str]]:
    paths = source_checkpoint_paths(args)
    out_dir = args.workspace_root / "checkpoints"
    records: list[dict[str, str]] = []
    day_sources = [
        paths["client0_highway_day"],
        paths["client2_citystreet_day"],
        paths["client4_residential_day"],
    ]

    for beta in parse_betas(args.candidate_betas):
        label = f"repair_shielded_local_expert_b{int(round(beta * 100)):03d}"
        output = out_dir / f"{label}.pt"
        if args.force or not output.exists():
            main03.residual_dqa_checkpoint(
                base=paths["final_repaired_shared"],
                sources=day_sources,
                anchor=paths["prev_repaired_shared"],
                output=output,
                beta=beta,
                scope=args.residual_scope,
                include_bn=args.include_bn,
            )
        save_record(
            records,
            label,
            output,
            beta=beta,
            variant=(
                "final repaired shared + beta * mean(round30 day local expert - round29 repaired shared), "
                f"scope={args.residual_scope}, include_bn={args.include_bn}"
            ),
        )

    write_csv(
        args.workspace_root / "stats" / "04_repair_shielded_candidate_checkpoints.csv",
        records,
        ["condition", "label", "kind", "round", "client", "variant", "beta", "path"],
    )
    return records


def eval_args(args: argparse.Namespace) -> argparse.Namespace:
    copied = argparse.Namespace(**vars(args))
    copied.workspace_root = args.workspace_root
    copied.eval_clients = False
    copied.dry_run = args.dry_run
    copied.classwise = args.classwise
    copied.no_eval_plots = args.no_eval_plots
    return copied


def source_final_metrics(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_csv(args.source_workspace / "stats" / "03_main_experiment_final_metrics.csv")
    out = []
    for row in rows:
        copied: dict[str, Any] = dict(row)
        copied["experiment"] = "03_main"
        copied["delta_vs_03_dqa_aggregate_map50_95"] = (
            "0.000000"
            if row.get("checkpoint_label") == "bn_residual_dqa_final_aggregate"
            else ""
        )
        out.append(copied)
    return out


def source_split_metrics(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_csv(args.source_workspace / "stats" / "03_main_experiment_split_metrics.csv")
    return [{**row, "experiment": "03_main"} for row in rows]


def candidate_metric_rows(args: argparse.Namespace, records: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_path = args.workspace_root / "validation_reports" / "paper_protocol_eval_summary.csv"
    rows = [row for row in read_csv(summary_path) if row.get("status") == "ok"]
    totals = {row["checkpoint_label"]: row for row in rows if row.get("split") in {"scene_daynight_total", "total"}}
    by_label_split = {(row["checkpoint_label"], row["split"]): row for row in rows}
    meta = {row["label"]: row for row in records}

    source_final = {
        row["checkpoint_label"]: row
        for row in read_csv(args.source_workspace / "stats" / "03_main_experiment_final_metrics.csv")
    }
    warm_m95 = as_float(source_final.get("warmup_global", {}).get("map50_95"))
    server_repair_m95 = as_float(source_final.get("warmup_server_repair_final", {}).get("map50_95"))
    dqa_agg_m95 = as_float(source_final.get("bn_residual_dqa_final_aggregate", {}).get("map50_95"))

    metric_rows: list[dict[str, Any]] = []
    for label, total in totals.items():
        if label not in meta:
            continue
        m50 = as_float(total.get("map50"))
        m95 = as_float(total.get("map50_95"))
        gap = main03.split_gap_metrics(by_label_split, label)
        metric_rows.append(
            {
                "experiment": "04_repair_shielded",
                "checkpoint_label": label,
                "condition": "warmup + repair-shielded local expert DQA",
                "kind": meta[label].get("kind", ""),
                "source_condition": "repair_shielded_local_expert_dqa",
                "round": meta[label].get("round", ""),
                "precision": total.get("precision", ""),
                "recall": total.get("recall", ""),
                "map50": "" if m50 is None else f"{m50:.6f}",
                "map50_95": "" if m95 is None else f"{m95:.6f}",
                "gain_vs_warmup_map50_95": "" if m95 is None or warm_m95 is None else f"{m95 - warm_m95:.6f}",
                "delta_vs_server_repair_map50_95": "" if m95 is None or server_repair_m95 is None else f"{m95 - server_repair_m95:.6f}",
                "delta_vs_03_dqa_aggregate_map50_95": "" if m95 is None or dqa_agg_m95 is None else f"{m95 - dqa_agg_m95:.6f}",
                **gap,
            }
        )

    split_rows: list[dict[str, Any]] = []
    for row in rows:
        label = row["checkpoint_label"]
        if label not in meta:
            continue
        split_rows.append(
            {
                "experiment": "04_repair_shielded",
                "checkpoint_label": label,
                "condition": "warmup + repair-shielded local expert DQA",
                "split": row["split"],
                "images": row.get("images", ""),
                "labels": row.get("labels", ""),
                "precision": row.get("precision", ""),
                "recall": row.get("recall", ""),
                "map50": row.get("map50", ""),
                "map50_95": row.get("map50_95", ""),
            }
        )
    return metric_rows, split_rows


def write_combined_metrics(args: argparse.Namespace, records: list[dict[str, str]]) -> list[dict[str, Any]]:
    fields = [
        "experiment",
        "checkpoint_label",
        "condition",
        "kind",
        "source_condition",
        "round",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "gain_vs_warmup_map50_95",
        "delta_vs_server_repair_map50_95",
        "delta_vs_03_dqa_aggregate_map50_95",
        "worst_split",
        "worst_split_map50_95",
        "day_avg_map50_95",
        "night_avg_map50_95",
        "day_night_gap_map50_95",
    ]
    source_rows = source_final_metrics(args)
    source_fields = set().union(*(row.keys() for row in source_rows)) if source_rows else set()
    for row in source_rows:
        row.setdefault("delta_vs_03_dqa_aggregate_map50_95", "")
        for field in fields:
            row.setdefault(field, "")

    candidate_rows: list[dict[str, Any]] = []
    split_rows = source_split_metrics(args)
    if args.evaluate and not args.dry_run:
        candidate_rows, candidate_splits = candidate_metric_rows(args, records)
        split_rows.extend(candidate_splits)
    combined = source_rows + candidate_rows
    write_csv(args.workspace_root / "stats" / "04_repair_shielded_final_metrics.csv", combined, fields)
    write_csv(
        args.workspace_root / "stats" / "04_repair_shielded_split_metrics.csv",
        split_rows,
        ["experiment", "checkpoint_label", "condition", "split", "images", "labels", "precision", "recall", "map50", "map50_95"],
    )
    return combined


def write_report(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    ranked = sorted(
        [row for row in rows if as_float(row.get("map50_95")) is not None],
        key=lambda row: as_float(row.get("map50_95")) or -1.0,
        reverse=True,
    )
    lines = [
        "# 04 Repair-shielded Local Expert DQA",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- protocol: {PROTOCOL_VERSION}",
        f"- source_workspace: {args.source_workspace}",
        "",
        "## Ranking",
        "",
        "| rank | experiment | checkpoint | mAP50 | mAP50:95 | delta vs 03 DQA aggregate | condition |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(ranked, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row.get("experiment", "")),
                    str(row.get("checkpoint_label", "")),
                    str(row.get("map50", "")),
                    str(row.get("map50_95", "")),
                    str(row.get("delta_vs_03_dqa_aggregate_map50_95", "")),
                    str(row.get("condition", "")).replace("|", "/"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Design",
            "",
            "The 03 result showed `BN-residual DQA aggregate` > `BN-residual DQA + server repair`, so the selected 04 idea is to keep the repaired shared/source path but add local pseudo-GT expert residuals after repair.  This approximates a repair-shielded expert deployment as a single YOLO checkpoint.",
            "",
        ]
    )
    (args.workspace_root / "04_repair_shielded_local_expert_report.md").write_text("\n".join(lines), encoding="utf-8")


def notify(args: argparse.Namespace, message: str, *, title: str, status: str = "", error: str | None = None) -> None:
    try:
        from notebook_notify import notify_discord

        context = {
            "workspace": str(args.workspace_root.resolve()),
            "source_workspace": str(args.source_workspace.resolve()),
            "status": status,
        }
        metrics_path = args.workspace_root / "stats" / "04_repair_shielded_final_metrics.csv"
        if metrics_path.exists():
            context["metrics_csv"] = str(metrics_path)
        if error:
            context["error"] = error[:500]
        print(notify_discord(message, title=title, context=context, fail_silently=True))
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}")


def str2bool(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-workspace", type=Path, default=DEFAULT_SOURCE_WORKSPACE)
    parser.add_argument("--source-round", type=int, default=30)
    parser.add_argument("--candidate-betas", default="0.25,0.50,0.75,1.00")
    parser.add_argument("--residual-scope", choices=["neck_head", "all"], default="neck_head")
    parser.add_argument("--include-bn", type=str2bool, default=True)
    parser.add_argument("--client-limit", type=int, default=1500)
    parser.add_argument("--clients", default="all")
    parser.add_argument("--warmup-checkpoint", type=Path, default=REPO_ROOT / "pseudogt_learnability" / "checkpoints" / "round000_warmup.pt")
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=32941)
    parser.add_argument("--device", default="")
    parser.add_argument("--eval-splits", default="highway_day,highway_night,citystreet_day,citystreet_night,residential_day,residential_night,total")
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--classwise", action="store_true")
    parser.add_argument("--no-eval-plots", action="store_true")
    parser.add_argument("--eval-clients", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--notify-start", action="store_true")
    parser.add_argument("--notify-end", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.source_workspace = args.source_workspace.expanduser().resolve()
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    (args.workspace_root / "stats").mkdir(parents=True, exist_ok=True)

    # Build configs/data lists in the 04 workspace for paper-protocol evaluation.
    main03.prepare_workspace(args, args.workspace_root)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "workspace": str(args.workspace_root),
        "source_workspace": str(args.source_workspace),
        "source_round": args.source_round,
        "candidate_betas": parse_betas(args.candidate_betas),
        "residual_scope": args.residual_scope,
        "include_bn": args.include_bn,
        "mode": "reuse_03_training_build_04_repair_shielded_candidates",
    }
    (args.workspace_root / "stats" / "04_repair_shielded_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.setup_only:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        print("Setup complete.")
        return []

    records = build_candidates(args)
    if args.evaluate:
        base01_0.run_evaluation(eval_args(args), records)
    rows = write_combined_metrics(args, records)
    write_report(args, rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.notify or args.notify_start:
        notify(args, "Scene-Daynight DQA 04 repair-shielded local expert started.", title="DQA 04 start", status="started")
    status = "success"
    error: str | None = None
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = str(exc)
        raise
    finally:
        if args.notify or args.notify_end:
            notify(
                args,
                f"Scene-Daynight DQA 04 repair-shielded local expert finished with status={status}.",
                title="DQA 04 finish",
                status=status,
                error=error,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
