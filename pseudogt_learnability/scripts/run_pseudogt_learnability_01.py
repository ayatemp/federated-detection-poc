#!/usr/bin/env python3
"""Run the first PseudoGT Learnability experiment.

The experiment isolates client pseudo-GT learning from DQA aggregation.  Every
client starts from the same copied warmup checkpoint, then candidate pseudo-GT
training profiles are compared with the same scene-wise evaluation protocol.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
NAV_ROOT = REPO_ROOT / "navigating_data_heterogeneity"
PROTOCOL_VERSION = "pseudogt_learnability_01_v1"


@dataclass(frozen=True)
class Variant:
    name: str
    train_scope: str
    aggregate_scope: str
    epochs: int
    lr0: float
    nms_conf: float
    low: float
    mid: float
    high: float
    teacher: float
    box: float
    obj: float
    cls: float
    low_mid_obj_weight: float
    mid_high_obj_weight: float
    pseudo_bbox_uncertain: bool
    pseudo_cls_uncertain: bool
    orthogonal_weight: float = 0.0
    note: str = ""


VARIANTS: dict[str, Variant] = {
    "backbone_obj_safe": Variant(
        name="backbone_obj_safe",
        train_scope="backbone",
        aggregate_scope="backbone",
        epochs=3,
        lr0=0.003,
        nms_conf=0.36,
        low=0.38,
        mid=0.62,
        high=0.92,
        teacher=0.30,
        box=0.005,
        obj=0.32,
        cls=0.02,
        low_mid_obj_weight=0.40,
        mid_high_obj_weight=0.85,
        pseudo_bbox_uncertain=False,
        pseudo_cls_uncertain=False,
        note="Phase-1-like feature adaptation with uncertain boxes used mainly as objectness signal.",
    ),
    "neck_head_high_precision": Variant(
        name="neck_head_high_precision",
        train_scope="neck_head",
        aggregate_scope="all",
        epochs=2,
        lr0=0.001,
        nms_conf=0.42,
        low=0.55,
        mid=0.74,
        high=0.90,
        teacher=0.22,
        box=0.015,
        obj=0.22,
        cls=0.06,
        low_mid_obj_weight=0.35,
        mid_high_obj_weight=0.80,
        pseudo_bbox_uncertain=False,
        pseudo_cls_uncertain=False,
        note="Higher precision pseudo boxes for head adaptation with backbone frozen.",
    ),
    "all_consistency_lowlr": Variant(
        name="all_consistency_lowlr",
        train_scope="all",
        aggregate_scope="all",
        epochs=2,
        lr0=0.0008,
        nms_conf=0.38,
        low=0.45,
        mid=0.68,
        high=0.88,
        teacher=0.24,
        box=0.012,
        obj=0.25,
        cls=0.05,
        low_mid_obj_weight=0.35,
        mid_high_obj_weight=0.80,
        pseudo_bbox_uncertain=False,
        pseudo_cls_uncertain=False,
        orthogonal_weight=1e-4,
        note="Conservative full-model adaptation with low LR and non-backbone orthogonal regularization.",
    ),
}

RESET_DQA_ENV = {
    "DQA0835_PSEUDO_MEMORY": "0",
    "DQA0836_SCOLQ_ENABLE": "0",
    "DQA0837_RSCOLQ_ENABLE": "0",
    "DQA0838_RSCOLQ_ENABLE": "0",
    "DQA0839_RSCOLQ_ENABLE": "0",
    "DQA08_TRI_STAGE_GATE": "1",
}


def _add_nav_path() -> None:
    if str(NAV_ROOT) not in sys.path:
        sys.path.insert(0, str(NAV_ROOT))


def configure_modules(workspace: Path):
    _add_nav_path()
    setup = importlib.import_module("setup_fedsto_scene_reproduction")
    setup.WORK_ROOT = workspace
    setup.LIST_ROOT = workspace / "data_lists"
    setup.CONFIG_ROOT = workspace / "configs"
    setup.RUN_ROOT = workspace / "runs"

    fedsto = importlib.import_module("run_fedsto_efficientteacher_exact")
    fedsto.setup = setup
    fedsto.PRETRAINED_PATH = workspace / "weights" / "efficient-yolov5l.pt"
    fedsto.GLOBAL_DIR = workspace / "global_checkpoints"
    fedsto.CLIENT_STATE_DIR = workspace / "client_states"
    fedsto.HISTORY_PATH = workspace / "history.json"
    return setup, fedsto


def ensure_dirs(workspace: Path) -> None:
    for relative in (
        "runs",
        "configs",
        "stats",
        "validation_reports",
        "logs",
        "checkpoints",
        "client_states",
        "global_checkpoints",
        "weights",
    ):
        (workspace / relative).mkdir(parents=True, exist_ok=True)


def resolve_variants(raw: str, epochs_override: int | None) -> list[Variant]:
    names = list(VARIANTS) if raw.strip().lower() == "all" else [item.strip() for item in raw.split(",") if item.strip()]
    variants: list[Variant] = []
    for name in names:
        if name not in VARIANTS:
            raise ValueError(f"Unknown variant {name!r}. Available: {', '.join(VARIANTS)}")
        variant = VARIANTS[name]
        if epochs_override is not None:
            variant = replace(variant, epochs=epochs_override)
        variants.append(variant)
    return variants


def copy_warmup_to_workspace(source: Path, workspace: Path, force: bool) -> Path:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Missing warmup checkpoint: {source}")
    dst = workspace / "global_checkpoints" / "round000_warmup.pt"
    if force or not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dst)
    return dst.resolve()


def config_device(args: argparse.Namespace) -> str:
    return "" if args.gpus > 1 else args.device


def apply_common_training_hyp(cfg: dict, lr0: float) -> None:
    cfg["linear_lr"] = False
    cfg["hyp"]["lr0"] = lr0
    cfg["hyp"]["lrf"] = 1.0
    cfg["hyp"]["warmup_epochs"] = 0
    cfg["hyp"]["mixup"] = 0.0
    cfg["hyp"]["scale"] = 0.5


def apply_safe_ssod_profile(cfg: dict, variant: Variant) -> None:
    ssod = cfg["SSOD"]
    ssod["nms_conf_thres"] = variant.nms_conf
    ssod["nms_iou_thres"] = 0.65
    ssod["teacher_loss_weight"] = variant.teacher
    ssod["box_loss_weight"] = variant.box
    ssod["obj_loss_weight"] = variant.obj
    ssod["cls_loss_weight"] = variant.cls
    ssod["ignore_thres_low"] = variant.low
    ssod["ignore_thres_high"] = variant.high
    ssod["uncertain_aug"] = True
    ssod["pseudo_label_with_obj"] = True
    ssod["pseudo_label_with_bbox"] = variant.pseudo_bbox_uncertain
    ssod["pseudo_label_with_cls"] = variant.pseudo_cls_uncertain
    ssod["ignore_obj"] = False
    ssod["use_ota"] = False
    ssod["resample_high_percent"] = 0.15
    ssod["resample_low_percent"] = 0.95
    ssod["ema_rate"] = 0.999
    ssod["cosine_ema"] = True
    ssod["imitate_teacher"] = False
    ssod["ssod_hyp"]["mosaic"] = 0.4
    ssod["ssod_hyp"]["cutout"] = 0.1
    ssod["ssod_hyp"]["autoaugment"] = 0.2
    ssod["ssod_hyp"]["scale"] = 0.5


def variant_env(variant: Variant) -> dict[str, str]:
    return {
        **RESET_DQA_ENV,
        "DQA08_IGNORE_THRES_MID": str(variant.mid),
        "DQA08_LOW_MID_OBJ_WEIGHT": str(variant.low_mid_obj_weight),
        "DQA08_MID_HIGH_OBJ_WEIGHT": str(variant.mid_high_obj_weight),
    }


def write_client_config(setup, variant: Variant, client: dict, start: Path, args: argparse.Namespace) -> Path:
    run_name = f"pl01_{variant.name}_client{client['id']}_{client['weather']}"
    cfg = setup.efficientteacher_config(
        name=run_name,
        train=setup.LIST_ROOT / "server_cloudy_train.txt",
        val=setup.LIST_ROOT / "server_cloudy_val.txt",
        target=setup.LIST_ROOT / f"client_{client['id']}_{client['weather']}_target.txt",
        weights=str(start.resolve()),
        epochs=variant.epochs,
        train_scope=variant.train_scope,
        orthogonal_weight=variant.orthogonal_weight,
        batch_size=args.batch_size,
        workers=args.workers,
        device=config_device(args),
    )
    apply_common_training_hyp(cfg, variant.lr0)
    apply_safe_ssod_profile(cfg, variant)
    return setup.write_config(f"{run_name}.yaml", cfg)


def write_server_repair_config(setup, variant: Variant, start: Path, args: argparse.Namespace) -> Path:
    run_name = f"pl01_{variant.name}_server_repair"
    cfg = setup.efficientteacher_config(
        name=run_name,
        train=setup.LIST_ROOT / "server_cloudy_train.txt",
        val=setup.LIST_ROOT / "server_cloudy_val.txt",
        target=None,
        weights=str(start.resolve()),
        epochs=args.server_repair_epochs,
        train_scope="all",
        orthogonal_weight=0.0,
        batch_size=args.batch_size,
        workers=args.workers,
        device=config_device(args),
    )
    apply_common_training_hyp(cfg, args.server_repair_lr)
    cfg["hyp"]["scale"] = 0.3
    cfg["SSOD"] = {"train_domain": False}
    return setup.write_config(f"{run_name}.yaml", cfg)


def run_train(setup, fedsto, config: Path, *, dry_run: bool, gpus: int, master_port: int, env: dict[str, str]) -> Path:
    if gpus > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(gpus),
            "--master_port",
            str(master_port),
            "train.py",
            "--cfg",
            str(config.resolve()),
        ]
    else:
        cmd = [sys.executable, "train.py", "--cfg", str(config.resolve())]
    print(" ".join(cmd))
    if not dry_run:
        merged_env = {**os.environ, **env}
        subprocess.run(cmd, cwd=setup.ET_ROOT, check=True, env=merged_env)
    with config.open(encoding="utf-8") as f:
        run_name = yaml.safe_load(f)["name"]
    return fedsto.checkpoint_path(run_name)


def reusable_checkpoint(fedsto, path: Path, force: bool) -> bool:
    if force or not fedsto.checkpoint_present(path):
        return False
    ok, reason = fedsto.validate_checkpoint(path)
    if ok:
        print(f"Reusing checkpoint: {path}")
        return True
    print(f"Ignoring invalid checkpoint ({reason}): {path}")
    return False


def save_checkpoint_record(records: list[dict], label: str, path: Path, kind: str, variant: str = "", client: str = "") -> None:
    records.append(
        {
            "label": label,
            "kind": kind,
            "variant": variant,
            "client": client,
            "path": str(path.resolve()),
        }
    )


def run_variant(setup, fedsto, variant: Variant, warmup: Path, args: argparse.Namespace, port_offset: int) -> tuple[list[dict], int]:
    records: list[dict] = []
    local_paths: list[Path] = []
    env = variant_env(variant)

    for client in setup.CLIENTS:
        client_tag = f"client{client['id']}_{client['weather']}"
        start = fedsto.CLIENT_STATE_DIR / f"pl01_{variant.name}_{client_tag}_start.pt"
        raw_run_name = f"pl01_{variant.name}_{client_tag}"
        raw_ckpt = fedsto.checkpoint_path(raw_run_name)
        final_ckpt = Path(args.workspace_root) / "checkpoints" / f"{variant.name}_{client_tag}.pt"

        if not args.dry_run and not fedsto.checkpoint_matches_protocol(start, PROTOCOL_VERSION):
            fedsto.make_start_checkpoint(
                warmup,
                start,
                protocol=PROTOCOL_VERSION,
                stage=f"{variant.name}_{client_tag}_start",
            )

        if not reusable_checkpoint(fedsto, final_ckpt, args.force):
            cfg = write_client_config(setup, variant, client, start, args)
            raw_ckpt = run_train(
                setup,
                fedsto,
                cfg,
                dry_run=args.dry_run,
                gpus=args.gpus,
                master_port=args.master_port + port_offset,
                env=env,
            )
            port_offset += 1
            if not args.dry_run:
                fedsto.mark_checkpoint_protocol(raw_ckpt, PROTOCOL_VERSION, f"{variant.name}_{client_tag}_raw")
                fedsto.make_start_checkpoint(
                    raw_ckpt,
                    final_ckpt,
                    protocol=PROTOCOL_VERSION,
                    stage=f"{variant.name}_{client_tag}",
                )

        local_paths.append(final_ckpt)
        save_checkpoint_record(records, f"{variant.name}_{client_tag}", final_ckpt, "client", variant.name, client_tag)

    aggregate_scope = variant.aggregate_scope
    aggregate = Path(args.workspace_root) / "checkpoints" / f"{variant.name}_aggregate_{aggregate_scope}.pt"
    if not args.dry_run and not reusable_checkpoint(fedsto, aggregate, args.force):
        fedsto.aggregate_checkpoints(local_paths, warmup, aggregate, backbone_only=(aggregate_scope == "backbone"))
        fedsto.mark_checkpoint_protocol(aggregate, PROTOCOL_VERSION, f"{variant.name}_aggregate_{aggregate_scope}")
    save_checkpoint_record(records, f"{variant.name}_aggregate_{aggregate_scope}", aggregate, "aggregate", variant.name)

    if args.server_repair_epochs > 0:
        repair_start = fedsto.GLOBAL_DIR / f"{variant.name}_server_repair_start.pt"
        repair_raw_name = f"pl01_{variant.name}_server_repair"
        repair_raw = fedsto.checkpoint_path(repair_raw_name)
        repair = Path(args.workspace_root) / "checkpoints" / f"{variant.name}_server_repair.pt"
        if not args.dry_run and not reusable_checkpoint(fedsto, repair, args.force):
            fedsto.make_start_checkpoint(
                aggregate,
                repair_start,
                protocol=PROTOCOL_VERSION,
                stage=f"{variant.name}_server_repair_start",
            )
            cfg = write_server_repair_config(setup, variant, repair_start, args)
            repair_raw = run_train(
                setup,
                fedsto,
                cfg,
                dry_run=args.dry_run,
                gpus=args.gpus,
                master_port=args.master_port + port_offset,
                env=RESET_DQA_ENV,
            )
            port_offset += 1
            if not args.dry_run:
                fedsto.mark_checkpoint_protocol(repair_raw, PROTOCOL_VERSION, f"{variant.name}_server_repair_raw")
                fedsto.make_start_checkpoint(
                    repair_raw,
                    repair,
                    protocol=PROTOCOL_VERSION,
                    stage=f"{variant.name}_server_repair",
                )
        save_checkpoint_record(records, f"{variant.name}_server_repair", repair, "server_repair", variant.name)

    return records, port_offset


def write_checkpoint_table(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label", "kind", "variant", "client", "path"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def run_evaluation(args: argparse.Namespace, records: list[dict]) -> None:
    checkpoints = [
        record
        for record in records
        if record["kind"] in {"warmup", "aggregate", "server_repair", "client"}
    ]
    cmd = [
        sys.executable,
        str((PROJECT_ROOT / "scripts" / "evaluate_scene_protocol.py").resolve()),
        "--workspace",
        str(Path(args.workspace_root).resolve()),
        "--splits",
        args.eval_splits,
        "--batch-size",
        str(args.val_batch_size),
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.no_eval_plots:
        cmd.append("--no-plots")
    if args.classwise:
        cmd.append("--verbose")
    if args.dry_run:
        cmd.append("--dry-run")
    for record in checkpoints:
        cmd.extend(["--checkpoint", f"{record['label']}={record['path']}"])
    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--warmup-checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "round000_warmup.pt")
    parser.add_argument("--variants", default="backbone_obj_safe,neck_head_high_precision,all_consistency_lowlr")
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=30341)
    parser.add_argument("--device", default="")
    parser.add_argument("--server-repair-epochs", type=int, default=1)
    parser.add_argument("--server-repair-lr", type=float, default=0.001)
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-splits", default="highway,citystreet,residential,total")
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--classwise", action="store_true")
    parser.add_argument("--no-eval-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.workspace_root = args.workspace_root.expanduser().resolve()
    ensure_dirs(args.workspace_root)
    setup, fedsto = configure_modules(args.workspace_root)
    setup_payload = setup.build_base_configs()
    manifest = setup_payload.get("manifest") if isinstance(setup_payload, dict) else None
    if manifest is None:
        manifest_path = args.workspace_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    warmup = copy_warmup_to_workspace(args.warmup_checkpoint, args.workspace_root, args.force)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Workspace: {args.workspace_root}")
    print(f"Warmup: {warmup}")
    print(json.dumps({"clients": manifest.get("clients"), "server": manifest.get("server")}, indent=2, ensure_ascii=False))

    if args.setup_only:
        print("Setup complete.")
        return 0

    args.gpus = fedsto.resolve_gpus(args.gpus)
    if not args.dry_run:
        fedsto.check_runtime_dependencies()

    variants = resolve_variants(args.variants, args.epochs_override)
    all_records: list[dict] = []
    save_checkpoint_record(all_records, "warmup_global", warmup, "warmup")
    port_offset = 0
    for variant in variants:
        print(f"\n=== Running variant: {variant.name} ===")
        print(json.dumps(asdict(variant), indent=2))
        records, port_offset = run_variant(setup, fedsto, variant, warmup, args, port_offset)
        all_records.extend(records)

    stats_root = args.workspace_root / "stats"
    stats_root.mkdir(parents=True, exist_ok=True)
    table_path = stats_root / "01_checkpoints.csv"
    write_checkpoint_table(table_path, all_records)

    manifest_path = stats_root / "01_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "protocol": PROTOCOL_VERSION,
                "project_root": str(PROJECT_ROOT),
                "workspace": str(args.workspace_root),
                "warmup_source": str(args.warmup_checkpoint.expanduser().resolve()),
                "warmup_workspace": str(warmup),
                "variants": [asdict(variant) for variant in variants],
                "checkpoints": all_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved: {table_path}")
    print(f"Saved: {manifest_path}")

    if args.evaluate:
        run_evaluation(args, all_records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
