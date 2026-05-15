#!/usr/bin/env python3
"""Run a small server-repair sanity grid from the latent DQA-MoX warmup.

The goal is deliberately narrow: check whether one server supervised repair
epoch is harmful because of the learning rate, the training scope, or both.
Each candidate starts from the same warmup checkpoint and is evaluated on
server_cloudy_val and the paper scene/day-night total split.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
NAV_ROOT = REPO_ROOT / "navigating_data_heterogeneity"
PSEUDOGT_SCRIPTS = REPO_ROOT / "pseudogt_learnability" / "scripts"

for path in (NAV_ROOT, PSEUDOGT_SCRIPTS, PROJECT_ROOT / "scripts", REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_paper_protocol as eval_protocol  # noqa: E402
import run_fedsto_efficientteacher_exact as fedsto  # noqa: E402
import run_pseudogt_learnability_03 as pl03  # noqa: E402
import run_scene_daynight_dqa_01 as dqa01  # noqa: E402
import run_scene_daynight_dqa_08_full_latent_dqamox as sdn08  # noqa: E402


PROTOCOL_VERSION = "scene_daynight_repair_sanity_grid_from_warmup_v1"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def patch_config(cfg: dict[str, Any], args: argparse.Namespace, *, lr: float, loss_box: float) -> dict[str, Any]:
    moe_args = argparse.Namespace(
        num_experts=args.num_experts,
        top_k=args.top_k,
        router_temperature=args.router_temperature,
        moe_scale=args.moe_scale,
        router_balance_weight=args.router_balance_weight,
        router_entropy_weight=args.router_entropy_weight,
    )
    sdn08.patch_latent_moe_config(cfg, moe_args)
    cfg["SSOD"] = {"train_domain": False}
    cfg["linear_lr"] = False
    cfg["hyp"]["lr0"] = float(lr)
    cfg["hyp"]["lrf"] = 1.0
    cfg["hyp"]["warmup_epochs"] = 0
    cfg["hyp"]["mosaic"] = float(args.mosaic)
    cfg["hyp"]["mixup"] = float(args.mixup)
    cfg["hyp"]["scale"] = float(args.scale)
    cfg["hyp"]["hsv_s"] = float(args.hsv_s)
    cfg["hyp"]["hsv_v"] = float(args.hsv_v)
    cfg.setdefault("Loss", {})
    cfg["Loss"]["box"] = float(loss_box)
    cfg["Loss"]["cls"] = float(args.loss_cls)
    cfg["Loss"]["obj"] = float(args.loss_obj)
    return cfg


def train_or_reuse(
    setup,
    args: argparse.Namespace,
    *,
    run_name: str,
    cfg_path: Path,
    compact_path: Path,
    stage: str,
    port: int,
) -> Path:
    if not args.force and fedsto.checkpoint_matches_protocol(compact_path, PROTOCOL_VERSION):
        print(f"Reusing checkpoint: {compact_path}")
        return compact_path

    raw = pl03.run_train(setup, fedsto, cfg_path, dry_run=args.dry_run, gpus=args.gpus, master_port=port)
    if args.dry_run:
        return raw
    fedsto.mark_checkpoint_protocol(raw, PROTOCOL_VERSION, f"{stage}_raw")
    fedsto.make_start_checkpoint(raw, compact_path, protocol=PROTOCOL_VERSION, stage=stage)
    pl03.cleanup_training_artifacts(raw, None)
    return compact_path


def eval_config(
    setup,
    args: argparse.Namespace,
    *,
    name: str,
    val_list: Path,
) -> Path:
    cfg = setup.efficientteacher_config(
        name=f"eval_{name}",
        train=setup.LIST_ROOT / "server_cloudy_train.txt",
        val=val_list,
        target=None,
        weights="",
        epochs=1,
        train_scope="all",
        batch_size=args.val_batch_size,
        workers=0,
        device="",
    )
    patch_config(cfg, args, lr=args.lrs[0], loss_box=args.loss_box)
    cfg["Dataset"]["batch_size"] = args.val_batch_size
    cfg["Dataset"]["workers"] = 0
    path = args.workspace_root / "eval_configs" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def evaluate_checkpoint(
    setup,
    args: argparse.Namespace,
    *,
    label: str,
    checkpoint: Path,
    split_name: str,
    split_list: Path,
    cfg_path: Path,
) -> dict[str, Any]:
    safe = eval_protocol.safe_name(f"{label}_{split_name}")
    log_file = args.workspace_root / "eval_logs" / f"{safe}.log"
    cmd = [
        str(args.python_executable),
        "val.py",
        "--weights",
        str(checkpoint.resolve()),
        "--cfg",
        str(cfg_path.resolve()),
        "--batch-size",
        str(args.val_batch_size),
        "--imgsz",
        str(args.imgsz),
        "--conf-thres",
        str(args.conf_thres),
        "--iou-thres",
        str(args.iou_thres),
        "--project",
        str((args.workspace_root / "eval_runs").resolve()),
        "--name",
        safe,
        "--exist-ok",
        "--no-plots",
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    row: dict[str, Any] = {
        "checkpoint_label": label,
        "checkpoint_path": str(checkpoint.resolve()),
        "split": split_name,
        "split_list": str(split_list.resolve()),
        "log_file": str(log_file.resolve()),
        "command": " ".join(cmd),
        "status": "dry_run" if args.dry_run else "pending",
    }
    if args.dry_run:
        return row
    log_file.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, cwd=setup.ET_ROOT, capture_output=True, text=True)
    log_file.write_text(result.stdout + "\nSTDERR\n" + result.stderr, encoding="utf-8")
    row["returncode"] = result.returncode
    if result.returncode == 0:
        row.update(eval_protocol.parse_val_stdout(result.stdout))
        row["status"] = "ok"
    else:
        row["status"] = "failed"
        row["error"] = (result.stderr or result.stdout)[-1000:]
    return row


def write_report(args: argparse.Namespace, rows: list[dict[str, Any]]) -> Path:
    report = args.workspace_root / "reports" / "repair_sanity_grid_summary.md"
    by_key = {
        (row.get("checkpoint_label"), row.get("split")): row
        for row in rows
        if row.get("status") == "ok"
    }
    labels = sorted({str(row.get("checkpoint_label")) for row in rows})
    lines = [
        "# Repair Sanity Grid From Warmup",
        "",
        f"Created UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Warmup checkpoint: `{args.warmup_checkpoint.resolve()}`",
        "",
        "| checkpoint | server_cloudy mAP50 | server_cloudy R | total mAP50 | total R |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in labels:
        cloudy = by_key.get((label, "server_cloudy_val"), {})
        total = by_key.get((label, "scene_daynight_total"), {})
        lines.append(
            "| {label} | {cloudy_map:.4f} | {cloudy_r:.4f} | {total_map:.4f} | {total_r:.4f} |".format(
                label=label,
                cloudy_map=float(cloudy.get("map50", 0.0) or 0.0),
                cloudy_r=float(cloudy.get("recall", 0.0) or 0.0),
                total_map=float(total.get("map50", 0.0) or 0.0),
                total_r=float(total.get("recall", 0.0) or 0.0),
            )
        )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=PROJECT_ROOT / "output" / "repair_sanity_grid_from_warmup",
    )
    parser.add_argument(
        "--warmup-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "output"
        / "08_full_latent_dqamox_from_warmup"
        / "checkpoints"
        / "round000_latent_dqamox_warmup.pt",
    )
    parser.add_argument("--scopes", nargs="+", default=["all", "neck_head", "moe_head"])
    parser.add_argument("--lrs", nargs="+", type=float, default=[0.0008, 0.00012, 0.00005])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--loss-box", type=float, default=0.05)
    parser.add_argument("--loss-cls", type=float, default=0.3)
    parser.add_argument("--loss-obj", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=29580)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--client-limit", type=int, default=1500)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--moe-scale", type=float, default=1.0)
    parser.add_argument("--router-balance-weight", type=float, default=0.01)
    parser.add_argument("--router-entropy-weight", type=float, default=0.001)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--hsv-s", type=float, default=0.35)
    parser.add_argument("--hsv-v", type=float, default=0.20)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.65)
    parser.add_argument("--device", default="")
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.warmup_checkpoint = args.warmup_checkpoint.expanduser().resolve()
    if not args.warmup_checkpoint.exists():
        raise FileNotFoundError(args.warmup_checkpoint)
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    for relative in ("checkpoints", "configs", "reports", "eval_configs", "eval_logs"):
        (args.workspace_root / relative).mkdir(parents=True, exist_ok=True)

    setup, configured_fedsto = dqa01.configure_modules(args.workspace_root, args.client_limit)
    globals()["fedsto"] = configured_fedsto
    manifest = setup.build_data_lists()
    split_specs = {
        "server_cloudy_val": setup.LIST_ROOT / "server_cloudy_val.txt",
        "scene_daynight_total": setup.LIST_ROOT / "paper_eval_scene_daynight_total_val.txt",
    }
    eval_cfgs = {
        name: eval_config(setup, args, name=name, val_list=path)
        for name, path in split_specs.items()
    }

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "warmup_checkpoint": str(args.warmup_checkpoint),
        "scopes": args.scopes,
        "lrs": args.lrs,
        "epochs": args.epochs,
        "loss_box": args.loss_box,
        "manifest_server": manifest.get("server", {}),
    }
    (args.workspace_root / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    checkpoint_specs: list[tuple[str, Path, str, str, float | None]] = [
        ("warmup", args.warmup_checkpoint, "warmup", "", None)
    ]
    for scope in args.scopes:
        for lr in args.lrs:
            lr_tag = f"{lr:.0e}".replace("+", "").replace("-", "m")
            run_name = f"repair_sanity_{scope}_lr{lr_tag}_box{args.loss_box:g}"
            cfg = setup.efficientteacher_config(
                name=run_name,
                train=setup.LIST_ROOT / "server_cloudy_train.txt",
                val=setup.LIST_ROOT / "server_cloudy_val.txt",
                target=None,
                weights=str(args.warmup_checkpoint.resolve()),
                epochs=args.epochs,
                train_scope=scope,
                orthogonal_weight=0.0,
                batch_size=args.batch_size,
                workers=args.workers,
                device="",
            )
            patch_config(cfg, args, lr=lr, loss_box=args.loss_box)
            cfg_path = setup.write_config(f"{run_name}.yaml", cfg)
            compact = args.workspace_root / "checkpoints" / f"{run_name}.pt"
            ckpt = train_or_reuse(
                setup,
                args,
                run_name=run_name,
                cfg_path=cfg_path,
                compact_path=compact,
                stage=run_name,
                port=args.master_port + len(checkpoint_specs),
            )
            checkpoint_specs.append((run_name, ckpt, "repair", scope, lr))

    rows: list[dict[str, Any]] = []
    for label, ckpt, kind, scope, lr in checkpoint_specs:
        for split_name, split_list in split_specs.items():
            row = evaluate_checkpoint(
                setup,
                args,
                label=label,
                checkpoint=ckpt,
                split_name=split_name,
                split_list=split_list,
                cfg_path=eval_cfgs[split_name],
            )
            row.update({"kind": kind, "scope": scope, "lr": "" if lr is None else lr, "loss_box": args.loss_box})
            rows.append(row)
            write_csv(
                args.workspace_root / "reports" / "repair_sanity_grid_metrics.csv",
                rows,
                [
                    "checkpoint_label",
                    "kind",
                    "scope",
                    "lr",
                    "loss_box",
                    "split",
                    "images",
                    "labels",
                    "precision",
                    "recall",
                    "map50",
                    "map50_95",
                    "status",
                    "returncode",
                    "checkpoint_path",
                    "split_list",
                    "log_file",
                    "command",
                    "error",
                ],
            )

    report = write_report(args, rows)
    print(f"Saved metrics: {args.workspace_root / 'reports' / 'repair_sanity_grid_metrics.csv'}")
    print(f"Saved report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
