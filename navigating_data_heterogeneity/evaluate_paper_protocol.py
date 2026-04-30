#!/usr/bin/env python3
"""Evaluate FedSTO-style checkpoints on paper-style per-weather validation splits."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
ET_ROOT = PROJECT_ROOT / "vendor" / "efficientteacher"
DEFAULT_WORK_ROOT = PROJECT_ROOT / "efficientteacher_fedsto"
DEFAULT_VAL_BATCH_SIZE = 8
DEFAULT_VAL_IMGSZ = 640
DEFAULT_VAL_CONF_THRES = 0.001
DEFAULT_VAL_IOU_THRES = 0.6
PREFERRED_VAL_PYTHONS = [
    Path("/root/micromamba/envs/al_yolov8/bin/python"),
    Path(sys.executable),
    Path("/opt/venv/bin/python"),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Explicit checkpoint in the form label=/abs/or/rel/path.pt. Repeat as needed.",
    )
    parser.add_argument(
        "--splits",
        default="cloudy,overcast,rainy,snowy,total",
        help="Comma-separated split names from the paper eval protocol.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_VAL_BATCH_SIZE)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_VAL_IMGSZ)
    parser.add_argument("--conf-thres", type=float, default=DEFAULT_VAL_CONF_THRES)
    parser.add_argument("--iou-thres", type=float, default=DEFAULT_VAL_IOU_THRES)
    parser.add_argument("--device", default="")
    parser.add_argument("--python-executable", type=Path, default=None)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--best-basis",
        choices=("map50", "map50_95"),
        default="map50",
        help="Metric used when auto-selecting the best phase2 round from training_run_summary.csv.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def configure_setup(workspace: Path):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    setup = importlib.import_module("setup_fedsto_exact_reproduction")
    setup.WORK_ROOT = workspace
    setup.LIST_ROOT = workspace / "data_lists"
    setup.CONFIG_ROOT = workspace / "configs"
    setup.RUN_ROOT = workspace / "runs"
    return setup


def ensure_paper_eval_manifest(setup) -> dict:
    manifest = setup.build_data_lists()
    paper_eval = manifest.get("paper_evaluation")
    if not paper_eval:
        raise RuntimeError("setup_fedsto_exact_reproduction.py did not produce paper_evaluation metadata.")
    return paper_eval


def load_history(history_path: Path) -> list[dict]:
    if not history_path.exists():
        return []
    data = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Expected {history_path} to contain a list, got {type(data).__name__}")
    return data


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def best_phase2_round_from_summary(summary_csv: Path, basis: str) -> int | None:
    if not summary_csv.exists():
        return None
    metric_col = {
        "map50": "final_metrics/mAP_0.5",
        "map50_95": "final_metrics/mAP_0.5:0.95",
    }[basis]
    best_round = None
    best_metric = None
    with summary_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("phase") != "2" or row.get("role") != "server":
                continue
            metric = parse_float(row.get(metric_col))
            round_idx = parse_float(row.get("round"))
            if metric is None or round_idx is None:
                continue
            if best_metric is None or metric > best_metric:
                best_metric = metric
                best_round = int(round_idx)
    return best_round


def resolve_checkpoint_specs(workspace: Path, args: argparse.Namespace) -> list[tuple[str, Path]]:
    explicit = []
    for item in args.checkpoint:
        if "=" not in item:
            raise ValueError(f"Invalid --checkpoint value {item!r}; expected label=path.pt")
        label, raw_path = item.split("=", 1)
        explicit.append((label.strip(), Path(raw_path).expanduser().resolve()))
    if explicit:
        return explicit

    history = load_history(workspace / "history.json")
    global_dir = workspace / "global_checkpoints"
    selections: list[tuple[str, Path]] = []
    seen: set[Path] = set()

    def add(label: str, path: Path) -> None:
        resolved = path.resolve()
        if not path.exists() or resolved in seen:
            return
        seen.add(resolved)
        selections.append((label, resolved))

    add("warmup_global", global_dir / "round000_warmup.pt")

    phase1 = [entry for entry in history if int(entry.get("phase", 0)) == 1]
    if phase1:
        final_phase1 = Path(phase1[-1]["global"])
        add(f"phase1_round{int(phase1[-1]['round']):03d}_global", final_phase1)

    summary_csv = workspace / "validation_reports" / "tables" / "training_run_summary.csv"
    best_round = best_phase2_round_from_summary(summary_csv, args.best_basis)
    if best_round is not None:
        add(
            f"phase2_best_server_cloudy_round{best_round:03d}_global",
            global_dir / f"phase2_round{best_round:03d}_global.pt",
        )

    phase2 = [entry for entry in history if int(entry.get("phase", 0)) == 2]
    if phase2:
        final_phase2 = Path(phase2[-1]["global"])
        add(f"phase2_round{int(phase2[-1]['round']):03d}_global", final_phase2)

    return selections


def select_split_specs(paper_eval: dict, requested: str) -> list[dict]:
    by_name = {split["name"]: split for split in paper_eval["splits"]}
    by_name["total"] = paper_eval["total"]
    names = [name.strip() for name in requested.split(",") if name.strip()]
    selected = []
    for name in names:
        if name not in by_name:
            raise ValueError(f"Unknown paper-eval split {name!r}. Available: {', '.join(sorted(by_name))}")
        selected.append(by_name[name])
    return selected


def select_val_python(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit is not None else []
    for candidate in PREFERRED_VAL_PYTHONS:
        if candidate not in candidates:
            candidates.append(candidate)

    check_cmd = "import cv2, seaborn, torch, yaml"
    for candidate in candidates:
        if candidate is None or not candidate.exists():
            continue
        result = subprocess.run(
            [str(candidate), "-c", check_cmd],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates if candidate is not None)
    raise RuntimeError(
        "Could not find a Python interpreter with EfficientTeacher validation dependencies. "
        f"Tried: {tried}"
    )


def write_eval_config(setup, report_root: Path, split: dict, args: argparse.Namespace) -> Path:
    config_root = report_root / "paper_protocol_configs"
    config_root.mkdir(parents=True, exist_ok=True)
    cfg = setup.efficientteacher_config(
        name=f"paper_eval_{split['name']}",
        train=setup.LIST_ROOT / "server_cloudy_train.txt",
        val=Path(split["list"]),
        target=None,
        weights="",
        epochs=1,
        train_scope="all",
        batch_size=args.batch_size,
        workers=0,
        device=args.device,
    )
    cfg["Dataset"]["batch_size"] = args.batch_size
    cfg["Dataset"]["workers"] = 0
    cfg["SSOD"] = {"train_domain": False}
    out = config_root / f"{split['name']}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "run"


def parse_val_stdout(stdout: str) -> dict:
    parsed: dict[str, float] = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) >= 7 and parts[0] == "all":
            parsed = {
                "images": float(parts[1]),
                "labels": float(parts[2]),
                "precision": float(parts[3]),
                "recall": float(parts[4]),
                "map50": float(parts[5]),
                "map50_95": float(parts[6]),
            }
    return parsed


def parse_classwise_stdout(stdout: str, names: list[str] | tuple[str, ...] | None = None) -> list[dict]:
    rows: list[dict] = []
    name_to_idx = {name: idx for idx, name in enumerate(names or [])}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) < 7 or parts[0] == "all":
            continue
        try:
            images = float(parts[-6])
            labels = float(parts[-5])
            precision = float(parts[-4])
            recall = float(parts[-3])
            map50 = float(parts[-2])
            map50_95 = float(parts[-1])
        except ValueError:
            continue
        class_name = " ".join(parts[:-6])
        row = {
            "class": class_name,
            "class_index": name_to_idx.get(class_name, ""),
            "images": images,
            "labels": labels,
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map50_95,
        }
        rows.append(row)
    return rows


def write_csv_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary_markdown(
    workspace: Path,
    report_root: Path,
    rows: list[dict],
    split_specs: list[dict],
    checkpoints: list[tuple[str, Path]],
    val_python: Path,
) -> str:
    lines = [
        "# Paper Protocol Evaluation Summary",
        "",
        f"Created UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Workspace: `{workspace}`",
        f"Validation python: `{val_python}`",
        f"Report root: `{report_root}`",
        "",
        "## Splits",
        "",
        "| split | raw weather | images | boxes |",
        "| --- | --- | ---: | ---: |",
    ]
    for split in split_specs:
        lines.append(
            f"| {split['name']} | {split.get('raw_weather', 'union')} | "
            f"{split.get('images', 0)} | {split.get('boxes', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Checkpoints",
            "",
        ]
    )
    for label, path in checkpoints:
        lines.append(f"- `{label}`: `{path}`")

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['checkpoint_label']} | {row['split']} | "
            f"{row.get('precision', '')} | {row.get('recall', '')} | "
            f"{row.get('map50', '')} | {row.get('map50_95', '')} | {row['status']} |"
        )

    total_rows = [row for row in rows if row["split"] == "total" and row["status"] == "ok"]
    if total_rows:
        lines.extend(
            [
                "",
                "## Total Split",
                "",
                "| checkpoint | mAP@0.5 | mAP@0.5:0.95 |",
                "| --- | ---: | ---: |",
            ]
        )
        for row in total_rows:
            lines.append(f"| {row['checkpoint_label']} | {row['map50']} | {row['map50_95']} |")

    return "\n".join(lines) + "\n"


def run_evaluations(
    setup,
    workspace: Path,
    checkpoints: list[tuple[str, Path]],
    split_specs: list[dict],
    args: argparse.Namespace,
) -> tuple[Path, list[dict], list[dict], Path]:
    report_root = workspace / "validation_reports"
    report_root.mkdir(parents=True, exist_ok=True)
    log_root = report_root / "paper_protocol_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    run_root = report_root / "paper_protocol_val_runs"
    val_python = select_val_python(args.python_executable)

    rows: list[dict] = []
    class_rows: list[dict] = []
    split_cfgs = {split["name"]: write_eval_config(setup, report_root, split, args) for split in split_specs}
    class_names = list(getattr(setup, "BDD_NAMES", []))

    for checkpoint_label, checkpoint_path in checkpoints:
        for split in split_specs:
            safe_label = safe_name(f"{checkpoint_label}_{split['name']}")
            log_file = log_root / f"{safe_label}.log"
            row = {
                "checkpoint_label": checkpoint_label,
                "checkpoint_path": str(checkpoint_path),
                "split": split["name"],
                "split_list": split["list"],
                "log_file": str(log_file),
                "status": "skipped",
            }
            if not checkpoint_path.exists():
                row["status"] = "missing_checkpoint"
                rows.append(row)
                continue

            cmd = [
                str(val_python),
                "val.py",
                "--weights",
                str(checkpoint_path),
                "--cfg",
                str(split_cfgs[split["name"]]),
                "--batch-size",
                str(args.batch_size),
                "--imgsz",
                str(args.imgsz),
                "--conf-thres",
                str(args.conf_thres),
                "--iou-thres",
                str(args.iou_thres),
                "--project",
                str(run_root),
                "--name",
                safe_label,
                "--exist-ok",
            ]
            if args.plots:
                cmd.append("--plots")
            else:
                cmd.append("--no-plots")
            if args.verbose:
                cmd.append("--verbose")
            if args.device:
                cmd.extend(["--device", args.device])

            row["command"] = " ".join(cmd)
            if args.dry_run:
                row["status"] = "dry_run"
                rows.append(row)
                continue

            result = subprocess.run(
                cmd,
                cwd=ET_ROOT,
                capture_output=True,
                text=True,
            )
            log_file.write_text(result.stdout + "\nSTDERR\n" + result.stderr, encoding="utf-8")
            row["returncode"] = result.returncode
            if result.returncode == 0:
                row.update(parse_val_stdout(result.stdout))
                row["status"] = "ok"
                if args.verbose:
                    for class_row in parse_classwise_stdout(result.stdout, class_names):
                        class_rows.append(
                            {
                                "checkpoint_label": checkpoint_label,
                                "checkpoint_path": str(checkpoint_path),
                                "split": split["name"],
                                "split_list": split["list"],
                                "status": "ok",
                                "log_file": str(log_file),
                                **class_row,
                            }
                        )
            else:
                row["status"] = "failed"
                row["error"] = result.stderr[-1000:]
            rows.append(row)

    return report_root, rows, class_rows, val_python


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    setup = configure_setup(workspace)
    paper_eval = ensure_paper_eval_manifest(setup)
    split_specs = select_split_specs(paper_eval, args.splits)
    checkpoints = resolve_checkpoint_specs(workspace, args)
    if not checkpoints:
        raise RuntimeError(
            "No checkpoints selected. Provide --checkpoint label=path.pt or run a FedSTO/DQA experiment first."
        )

    report_root, rows, class_rows, val_python = run_evaluations(setup, workspace, checkpoints, split_specs, args)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "report_root": str(report_root),
        "validation_python": str(val_python),
        "dry_run": args.dry_run,
        "checkpoints": [{"label": label, "path": str(path)} for label, path in checkpoints],
        "splits": split_specs,
        "rows": rows,
        "class_rows": class_rows,
    }
    manifest_path = report_root / "paper_protocol_eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "checkpoint_label",
        "checkpoint_path",
        "split",
        "split_list",
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
    summary_csv = report_root / "paper_protocol_eval_summary.csv"
    write_csv_rows(summary_csv, rows, fieldnames)

    classwise_csv = report_root / "paper_protocol_classwise_summary.csv"
    if class_rows:
        write_csv_rows(
            classwise_csv,
            class_rows,
            [
                "checkpoint_label",
                "checkpoint_path",
                "split",
                "split_list",
                "status",
                "class_index",
                "class",
                "images",
                "labels",
                "precision",
                "recall",
                "map50",
                "map50_95",
                "log_file",
            ],
        )

    summary_md = report_root / "paper_protocol_eval_summary.md"
    summary_md.write_text(
        build_summary_markdown(workspace, report_root, rows, split_specs, checkpoints, val_python),
        encoding="utf-8",
    )

    print(f"Saved: {manifest_path}")
    print(f"Saved: {summary_csv}")
    if class_rows:
        print(f"Saved: {classwise_csv}")
    print(f"Saved: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
