from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

import run_fedsto_efficientteacher_exact as fedsto
import setup_fedsto_exact_reproduction as setup


DEFAULT_SEED_WEIGHTS = (
    Path(__file__).resolve().parents[1]
    / "dynamic_quality_aware_classwise_aggregation"
    / "scene_daynight_dqa"
    / "output"
    / "08_full_latent_dqamox_from_warmup"
    / "checkpoints"
    / "round000_latent_dqamox_warmup.pt"
)


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_subset(path: Path, lines: list[str], count: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = lines[:count]
    if not selected:
        raise RuntimeError(f"No entries available for {path}")
    path.write_text("\n".join(selected) + "\n", encoding="utf-8")
    cache_path = path.with_suffix(".cache")
    if cache_path.exists():
        cache_path.unlink()
    return path


def model_state(ckpt: dict, key: str) -> dict[str, torch.Tensor]:
    model = ckpt[key]
    return {name: tensor.detach().float().cpu() for name, tensor in model.float().state_dict().items()}


def max_abs_diff(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], limit: int | None = None) -> float:
    values = []
    for idx, (name, tensor) in enumerate(left.items()):
        if limit is not None and idx >= limit:
            break
        other = right.get(name)
        if other is None or not tensor.dtype.is_floating_point:
            continue
        values.append((tensor - other).abs().max().item())
    return max(values) if values else 0.0


def audit(path: Path) -> dict:
    ckpt = fedsto._load(path)
    row = {
        "path": str(path.resolve()),
        "stage": ckpt.get("fedsto_stage"),
        "ema_present": ckpt.get("ema") is not None,
        "model_present": ckpt.get("model") is not None,
        "optimizer_present": ckpt.get("optimizer") is not None,
        "updates": ckpt.get("updates"),
        "epoch": ckpt.get("epoch"),
    }
    print(json.dumps(row, ensure_ascii=False))
    return row


def write_config(name: str, train: Path, val: Path, target: Path | None, weights: Path, args: argparse.Namespace, *, train_scope: str) -> Path:
    cfg = setup.efficientteacher_config(
        name=name,
        train=train,
        val=val,
        target=target,
        weights=str(weights.resolve()),
        epochs=1,
        train_scope=train_scope,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
    )
    return setup.write_config(f"{name}.yaml", cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path("/app/Object_Detection/navigating_data_heterogeneity/efficientteacher_fedsto_ema_smoke"),
    )
    parser.add_argument("--seed-weights", type=Path, default=DEFAULT_SEED_WEIGHTS)
    parser.add_argument("--server-train-images", type=int, default=16)
    parser.add_argument("--server-val-images", type=int, default=8)
    parser.add_argument("--target-images", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=29571)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    if not args.seed_weights.exists():
        raise FileNotFoundError(f"Missing seed weights: {args.seed_weights}")

    os.environ.setdefault("ET_SKIP_AFTER_TRAIN_BEST_VAL", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")

    fedsto.apply_workspace_root(args.workspace_root)
    fedsto.ensure_efficientteacher_import_path()
    fedsto.check_runtime_dependencies()
    args.gpus = fedsto.resolve_gpus(args.gpus)

    source_list_root = Path(__file__).resolve().parent / "efficientteacher_fedsto" / "data_lists"
    server_train_full = read_lines(source_list_root / "server_cloudy_train.txt")
    server_val_full = read_lines(source_list_root / "server_cloudy_val.txt")
    client0_full = read_lines(source_list_root / "client_0_overcast_target.txt")

    train_list = write_subset(setup.LIST_ROOT / "server_cloudy_train_tiny.txt", server_train_full, args.server_train_images)
    val_list = write_subset(setup.LIST_ROOT / "server_cloudy_val_tiny.txt", server_val_full, args.server_val_images)
    target_list = write_subset(setup.LIST_ROOT / "client_0_overcast_target_tiny.txt", client0_full, args.target_images)

    summary: dict[str, object] = {
        "workspace": str(args.workspace_root.resolve()),
        "seed_weights": str(args.seed_weights.resolve()),
        "lists": {
            "server_train": str(train_list.resolve()),
            "server_val": str(val_list.resolve()),
            "client0_target": str(target_list.resolve()),
            "server_train_images": args.server_train_images,
            "server_val_images": args.server_val_images,
            "target_images": args.target_images,
        },
        "audits": [],
    }

    warmup_raw = fedsto.run_train(
        write_config("ema_smoke_warmup", train_list, val_list, None, args.seed_weights, args, train_scope="all"),
        False,
        gpus=args.gpus,
        master_port=args.master_port,
    )
    warmup = fedsto.GLOBAL_DIR / "round000_warmup.pt"
    fedsto.make_start_checkpoint(warmup_raw, warmup, protocol="ema_inheritance_smoke_v1", stage="warmup_after_strip")
    summary["audits"].append(audit(warmup))

    round1_start = fedsto.CLIENT_STATE_DIR / "client0_round001_start.pt"
    fedsto.make_start_checkpoint(warmup, round1_start, protocol="ema_inheritance_smoke_v1", stage="client0_round001_start")
    summary["audits"].append(audit(round1_start))

    round1_raw = fedsto.run_train(
        write_config("ema_smoke_client0_round001", train_list, val_list, target_list, round1_start, args, train_scope="all"),
        False,
        gpus=args.gpus,
        master_port=args.master_port + 1,
    )
    round1_final = fedsto.GLOBAL_DIR / "client0_round001_final.pt"
    local_teacher = fedsto.CLIENT_STATE_DIR / "client0_latest.pt"
    fedsto.make_start_checkpoint(round1_raw, round1_final, protocol="ema_inheritance_smoke_v1", stage="client0_round001_final")
    fedsto.make_start_checkpoint(round1_final, local_teacher, protocol="ema_inheritance_smoke_v1", stage="client0_latest_after_round001")
    summary["audits"].append(audit(round1_final))
    summary["audits"].append(audit(local_teacher))

    round2_start = fedsto.CLIENT_STATE_DIR / "client0_round002_start.pt"
    fedsto.make_start_checkpoint(
        warmup,
        round2_start,
        local_teacher,
        protocol="ema_inheritance_smoke_v1",
        stage="client0_round002_start_with_local_teacher",
    )
    summary["audits"].append(audit(round2_start))

    warmup_ckpt = fedsto._load(warmup)
    local_ckpt = fedsto._load(local_teacher)
    round2_ckpt = fedsto._load(round2_start)
    warmup_model = model_state(warmup_ckpt, "model")
    local_model = model_state(local_ckpt, "model")
    round2_model = model_state(round2_ckpt, "model")
    round2_ema = model_state(round2_ckpt, "ema")
    summary["inheritance_checks"] = {
        "round2_student_equals_global_max_abs": max_abs_diff(warmup_model, round2_model),
        "round2_teacher_equals_local_max_abs": max_abs_diff(local_model, round2_ema),
        "round2_teacher_differs_from_global_max_abs_first50": max_abs_diff(warmup_model, round2_ema, limit=50),
    }
    print(json.dumps(summary["inheritance_checks"], ensure_ascii=False))

    round2_raw = fedsto.run_train(
        write_config("ema_smoke_client0_round002", train_list, val_list, target_list, round2_start, args, train_scope="all"),
        False,
        gpus=args.gpus,
        master_port=args.master_port + 2,
    )
    round2_final = fedsto.GLOBAL_DIR / "client0_round002_final.pt"
    fedsto.make_start_checkpoint(round2_raw, round2_final, protocol="ema_inheritance_smoke_v1", stage="client0_round002_final")
    summary["audits"].append(audit(round2_final))

    summary_path = args.workspace_root / "ema_inheritance_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
