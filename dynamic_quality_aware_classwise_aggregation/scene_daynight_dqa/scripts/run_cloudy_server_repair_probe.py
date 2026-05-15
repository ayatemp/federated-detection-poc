#!/usr/bin/env python3
"""Probe whether supervised server repair really improves cloudy data.

This script deliberately removes the FL/DQA/client pieces and tests only the
server-cloudy supervised update.  It compares two protocols from the same
warmup checkpoint:

1. continuous_N: one normal training run for N epochs from warmup.
2. relay_rN: N repetitions of 1-epoch repair, each using the previous repaired
   checkpoint as the next parent.

The important diagnostic split is server_cloudy_train as well as
server_cloudy_val.  If train does not improve, repair itself is broken.  If
train improves but val falls, repair is overfitting or shifting calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
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


PROTOCOL_VERSION = "scene_daynight_cloudy_server_repair_probe_v1"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def lr_tag(value: float) -> str:
    return f"{value:.0e}".replace("+", "").replace("-", "m")


def patch_config(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    moe_args = argparse.Namespace(
        num_experts=args.num_experts,
        top_k=args.top_k,
        router_temperature=args.router_temperature,
        moe_scale=args.moe_scale,
        router_balance_weight=args.router_balance_weight,
        router_entropy_weight=args.router_entropy_weight,
    )
    # The warmup checkpoint is a LatentMoE YOLO head, so eval/train configs
    # must instantiate the same head even though this probe is not testing MoE.
    sdn08.patch_latent_moe_config(cfg, moe_args)
    cfg["SSOD"] = {"train_domain": False}
    cfg["linear_lr"] = False
    cfg["find_unused_parameters"] = True
    cfg["hyp"]["lr0"] = float(args.lr)
    cfg["hyp"]["lrf"] = 1.0
    cfg["hyp"]["warmup_epochs"] = 0
    cfg["hyp"]["mosaic"] = float(args.mosaic)
    cfg["hyp"]["mixup"] = float(args.mixup)
    cfg["hyp"]["scale"] = float(args.scale)
    cfg["hyp"]["hsv_s"] = float(args.hsv_s)
    cfg["hyp"]["hsv_v"] = float(args.hsv_v)
    cfg.setdefault("Loss", {})
    cfg["Loss"]["box"] = float(args.loss_box)
    cfg["Loss"]["cls"] = float(args.loss_cls)
    cfg["Loss"]["obj"] = float(args.loss_obj)
    return cfg


def train_or_reuse(
    setup,
    args: argparse.Namespace,
    *,
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


def make_train_config(
    setup,
    args: argparse.Namespace,
    *,
    name: str,
    weights: Path,
    epochs: int,
) -> Path:
    cfg = setup.efficientteacher_config(
        name=name,
        train=setup.LIST_ROOT / "server_cloudy_train.txt",
        val=setup.LIST_ROOT / "server_cloudy_val.txt",
        target=None,
        weights=str(weights.resolve()),
        epochs=epochs,
        train_scope=args.train_scope,
        orthogonal_weight=args.orthogonal_weight,
        batch_size=args.batch_size,
        workers=args.workers,
        device="",
    )
    patch_config(cfg, args)
    return setup.write_config(f"{name}.yaml", cfg)


def make_eval_config(
    setup,
    args: argparse.Namespace,
    *,
    split_name: str,
    val_list: Path,
) -> Path:
    cfg = setup.efficientteacher_config(
        name=f"eval_{split_name}",
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
    patch_config(cfg, args)
    cfg["Dataset"]["batch_size"] = args.val_batch_size
    cfg["Dataset"]["workers"] = 0
    path = args.workspace_root / "eval_configs" / f"{split_name}.yaml"
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
    if args.verbose_eval:
        cmd.append("--verbose")
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
        row["error"] = (result.stderr or result.stdout)[-1500:]
    return row


def checkpoint_specs_to_eval(
    args: argparse.Namespace,
    specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if args.eval_all_checkpoints:
        return specs
    labels = {"warmup"}
    if args.continuous_epochs:
        labels.add(f"continuous_e{max(args.continuous_epochs):02d}")
    if args.relay_rounds > 0:
        labels.add(f"relay_r{args.relay_rounds:02d}")
    return [spec for spec in specs if spec["label"] in labels]


def write_report(args: argparse.Namespace, rows: list[dict[str, Any]], specs: list[dict[str, Any]]) -> Path:
    report = args.workspace_root / "reports" / "cloudy_server_repair_probe_summary.md"
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    by_key = {(row["checkpoint_label"], row["split"]): row for row in ok_rows}
    warmup_by_split = {row["split"]: row for row in ok_rows if row["checkpoint_label"] == "warmup"}
    labels = [spec["label"] for spec in specs]
    lines = [
        "# Cloudy Server Repair Probe",
        "",
        f"Created UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Workspace: `{args.workspace_root}`",
        f"Warmup checkpoint: `{args.warmup_checkpoint}`",
        f"Train scope: `{args.train_scope}`, lr: `{args.lr}`, loss_box: `{args.loss_box}`",
        "",
        "## Results",
        "",
        "| checkpoint | kind | cloudy train mAP50 | delta | cloudy val mAP50 | delta | total mAP50 | delta | cloudy val R |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in labels:
        spec = next((item for item in specs if item["label"] == label), {})
        train = by_key.get((label, "server_cloudy_train"), {})
        val = by_key.get((label, "server_cloudy_val"), {})
        total = by_key.get((label, "scene_daynight_total"), {})
        train_base = warmup_by_split.get("server_cloudy_train", {}).get("map50")
        val_base = warmup_by_split.get("server_cloudy_val", {}).get("map50")
        total_base = warmup_by_split.get("scene_daynight_total", {}).get("map50")
        train_map = train.get("map50")
        val_map = val.get("map50")
        total_map = total.get("map50")
        lines.append(
            "| {label} | {kind} | {train_map} | {train_delta} | {val_map} | {val_delta} | {total_map} | {total_delta} | {val_r} |".format(
                label=label,
                kind=spec.get("kind", ""),
                train_map=format_float(train_map),
                train_delta=format_delta(train_map, train_base),
                val_map=format_float(val_map),
                val_delta=format_delta(val_map, val_base),
                total_map=format_float(total_map),
                total_delta=format_delta(total_map, total_base),
                val_r=format_float(val.get("recall")),
            )
        )
    best_val = max(
        (row for row in ok_rows if row["split"] == "server_cloudy_val"),
        key=lambda row: float(row.get("map50") or -1.0),
        default=None,
    )
    if best_val is not None:
        lines.extend(
            [
                "",
                "## Best Cloudy Val",
                "",
                f"- checkpoint: `{best_val['checkpoint_label']}`",
                f"- mAP50: `{best_val.get('map50')}`",
                f"- recall: `{best_val.get('recall')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- train and val both up: supervised repair is healthy.",
            "- train up and val down: repair is overfitting or shifting calibration.",
            "- train down: checkpoint/config/EMA/BN/training protocol is broken.",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def format_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def format_delta(value: Any, base: Any) -> str:
    try:
        return f"{float(value) - float(base):+.4f}"
    except (TypeError, ValueError):
        return ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=PROJECT_ROOT / "server_repair_cloudy_probe" / "output" / "01_cloudy_repair_only",
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
    parser.add_argument("--continuous-epochs", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--relay-rounds", type=int, default=5)
    parser.add_argument("--train-scope", default="all")
    parser.add_argument("--orthogonal-weight", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--loss-box", type=float, default=0.05)
    parser.add_argument("--loss-cls", type=float, default=0.3)
    parser.add_argument("--loss-obj", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=29630)
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
    parser.add_argument("--eval-all-checkpoints", action="store_true", default=True)
    parser.add_argument("--verbose-eval", action="store_true")
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
    for relative in ("checkpoints", "configs", "reports", "eval_configs", "eval_logs", "starts"):
        (args.workspace_root / relative).mkdir(parents=True, exist_ok=True)

    setup, configured_fedsto = dqa01.configure_modules(args.workspace_root, args.client_limit)
    globals()["fedsto"] = configured_fedsto
    manifest = setup.build_data_lists()

    split_specs = {
        "server_cloudy_train": setup.LIST_ROOT / "server_cloudy_train.txt",
        "server_cloudy_val": setup.LIST_ROOT / "server_cloudy_val.txt",
        "scene_daynight_total": setup.LIST_ROOT / "paper_eval_scene_daynight_total_val.txt",
    }
    eval_cfgs = {
        name: make_eval_config(setup, args, split_name=name, val_list=path)
        for name, path in split_specs.items()
    }

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "warmup_checkpoint": str(args.warmup_checkpoint),
        "continuous_epochs": args.continuous_epochs,
        "relay_rounds": args.relay_rounds,
        "train_scope": args.train_scope,
        "lr": args.lr,
        "loss_box": args.loss_box,
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "manifest_server": manifest.get("server", {}),
    }
    (args.workspace_root / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    checkpoint_specs: list[dict[str, Any]] = [
        {
            "label": "warmup",
            "path": args.warmup_checkpoint,
            "kind": "warmup",
            "epochs": 0,
            "parent": "",
        }
    ]

    port_offset = 0
    for epochs in sorted(set(args.continuous_epochs)):
        label = f"continuous_e{epochs:02d}"
        name = f"cloudy_repair_{label}_lr{lr_tag(args.lr)}"
        cfg_path = make_train_config(setup, args, name=name, weights=args.warmup_checkpoint, epochs=epochs)
        compact = args.workspace_root / "checkpoints" / f"{label}.pt"
        ckpt = train_or_reuse(
            setup,
            args,
            cfg_path=cfg_path,
            compact_path=compact,
            stage=label,
            port=args.master_port + port_offset,
        )
        port_offset += 1
        checkpoint_specs.append(
            {
                "label": label,
                "path": ckpt,
                "kind": "continuous_from_warmup",
                "epochs": epochs,
                "parent": str(args.warmup_checkpoint),
            }
        )

    parent = args.warmup_checkpoint
    for round_idx in range(1, args.relay_rounds + 1):
        label = f"relay_r{round_idx:02d}"
        start = args.workspace_root / "starts" / f"{label}_start.pt"
        if args.force or not fedsto.checkpoint_matches_protocol(start, PROTOCOL_VERSION):
            if not args.dry_run:
                fedsto.make_start_checkpoint(
                    parent,
                    start,
                    protocol=PROTOCOL_VERSION,
                    stage=f"{label}_start",
                )
        name = f"cloudy_repair_{label}_lr{lr_tag(args.lr)}"
        cfg_path = make_train_config(setup, args, name=name, weights=start, epochs=1)
        compact = args.workspace_root / "checkpoints" / f"{label}.pt"
        ckpt = train_or_reuse(
            setup,
            args,
            cfg_path=cfg_path,
            compact_path=compact,
            stage=label,
            port=args.master_port + port_offset,
        )
        port_offset += 1
        checkpoint_specs.append(
            {
                "label": label,
                "path": ckpt,
                "kind": "one_epoch_checkpoint_relay",
                "epochs": round_idx,
                "parent": str(parent),
            }
        )
        parent = ckpt

    rows: list[dict[str, Any]] = []
    eval_specs = checkpoint_specs_to_eval(args, checkpoint_specs)
    fieldnames = [
        "checkpoint_label",
        "kind",
        "epochs",
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
    ]
    for spec in eval_specs:
        for split_name, split_list in split_specs.items():
            row = evaluate_checkpoint(
                setup,
                args,
                label=spec["label"],
                checkpoint=spec["path"],
                split_name=split_name,
                split_list=split_list,
                cfg_path=eval_cfgs[split_name],
            )
            row.update(
                {
                    "kind": spec["kind"],
                    "epochs": spec["epochs"],
                }
            )
            rows.append(row)
            write_csv(args.workspace_root / "reports" / "cloudy_server_repair_probe_metrics.csv", rows, fieldnames)

    spec_path = args.workspace_root / "reports" / "cloudy_server_repair_probe_checkpoints.json"
    spec_path.write_text(
        json.dumps(
            [
                {
                    **{key: value for key, value in spec.items() if key != "path"},
                    "path": str(Path(spec["path"]).resolve()),
                }
                for spec in checkpoint_specs
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report = write_report(args, rows, eval_specs)
    print(f"Saved metrics: {args.workspace_root / 'reports' / 'cloudy_server_repair_probe_metrics.csv'}")
    print(f"Saved checkpoint spec: {spec_path}")
    print(f"Saved report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
