from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import torch

import setup_fedsto_exact_reproduction as setup


PRETRAINED_URL = "https://github.com/AlibabaResearch/efficientteacher/releases/download/1.0/efficient-yolov5l.pt"
PRETRAINED_PATH = setup.WORK_ROOT / "weights" / "efficient-yolov5l.pt"
GLOBAL_DIR = setup.WORK_ROOT / "global_checkpoints"
CLIENT_STATE_DIR = setup.WORK_ROOT / "client_states"


def ensure_efficientteacher_import_path() -> None:
    et_root = str(setup.ET_ROOT.resolve())
    if et_root not in sys.path:
        sys.path.insert(0, et_root)


def checkpoint_path(run_name: str) -> Path:
    return setup.RUN_ROOT / run_name / "weights" / "last.pt"


def validate_checkpoint(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if path.stat().st_size < 1024 * 1024:
        return False, f"too small ({path.stat().st_size} bytes)"
    try:
        ensure_efficientteacher_import_path()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return False, f"torch.load failed: {exc}"
    if not isinstance(checkpoint, dict):
        return False, f"unexpected checkpoint type: {type(checkpoint).__name__}"
    if "model" not in checkpoint:
        return False, "missing 'model' key"
    return True, "ok"


def download_pretrained(force: bool = False) -> Path:
    PRETRAINED_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PRETRAINED_PATH.exists() and not force:
        ok, reason = validate_checkpoint(PRETRAINED_PATH)
        if ok:
            return PRETRAINED_PATH
        print(f"Existing pretrained checkpoint is invalid: {PRETRAINED_PATH} ({reason})")
        PRETRAINED_PATH.unlink()

    if force and PRETRAINED_PATH.exists():
        PRETRAINED_PATH.unlink()

    if not PRETRAINED_PATH.exists():
        print(f"Downloading {PRETRAINED_URL} -> {PRETRAINED_PATH}")
        urllib.request.urlretrieve(PRETRAINED_URL, PRETRAINED_PATH)

    ok, reason = validate_checkpoint(PRETRAINED_PATH)
    if not ok:
        raise RuntimeError(
            f"Downloaded pretrained checkpoint is invalid: {PRETRAINED_PATH} ({reason}). "
            "Delete the file and retry, or manually place a valid efficient-yolov5l.pt there."
        )
    return PRETRAINED_PATH


def check_runtime_dependencies() -> None:
    missing = []
    if importlib.util.find_spec("pkg_resources") is None and not (setup.ET_ROOT / "pkg_resources.py").exists():
        missing.append("setuptools<81 or vendor/efficientteacher/pkg_resources.py")
    if importlib.util.find_spec("yaml") is None:
        missing.append("PyYAML")
    if importlib.util.find_spec("cv2") is None:
        missing.append("opencv-python")
    if importlib.util.find_spec("thop") is None:
        missing.append("thop")
    if importlib.util.find_spec("tensorboard") is None:
        missing.append("tensorboard")
    if importlib.util.find_spec("sklearn") is None:
        missing.append("scikit-learn")
    if missing:
        install_hint = f"{sys.executable} -m pip install 'setuptools<81' PyYAML opencv-python thop tensorboard scikit-learn"
        raise RuntimeError(
            "Missing EfficientTeacher runtime dependencies: "
            + ", ".join(missing)
            + "\nInstall them in this kernel/environment, for example:\n"
            + install_hint
        )


def write_runtime_config(
    name: str,
    *,
    target: Path | None,
    weights: Path,
    phase: int,
    role: str,
    round_idx: int,
    batch_size: int,
    workers: int,
    device: str,
) -> Path:
    train_scope = "backbone" if phase == 1 else "all"
    orthogonal_weight = 0.0 if phase == 1 else 1e-4
    cfg = setup.efficientteacher_config(
        name=name,
        train=setup.LIST_ROOT / "server_cloudy_train.txt",
        val=setup.LIST_ROOT / "server_cloudy_val.txt",
        target=target,
        weights=str(weights.resolve()),
        epochs=1,
        train_scope=train_scope,
        orthogonal_weight=orthogonal_weight,
        batch_size=batch_size,
        workers=workers,
        device=device,
    )
    if role == "server":
        cfg["SSOD"] = {"train_domain": False}
    return setup.write_config(f"runtime_phase{phase}_{role}_round{round_idx}_{name}.yaml", cfg)


def run_train(config: Path, dry_run: bool, *, gpus: int, master_port: int) -> Path:
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
        subprocess.run(cmd, cwd=setup.ET_ROOT, check=True)
    cfg_name = config.stem
    if cfg_name.startswith("runtime_phase"):
        with config.open() as f:
            import yaml
            run_name = yaml.safe_load(f)["name"]
    else:
        run_name = cfg_name
    return checkpoint_path(run_name)


def _load(path: Path) -> dict:
    ensure_efficientteacher_import_path()
    return torch.load(path, map_location="cpu", weights_only=False)


def _state_dict(ckpt: dict, key: str = "model") -> dict:
    return ckpt[key].float().state_dict()


def _is_backbone_key(key: str) -> bool:
    lowered = key.lower()
    return "backbone" in lowered


def make_start_checkpoint(global_ckpt: Path, out: Path, local_ema_ckpt: Path | None = None) -> Path:
    base = _load(global_ckpt)
    base = copy.deepcopy(base)
    base["epoch"] = -1
    base["optimizer"] = None
    if local_ema_ckpt and local_ema_ckpt.exists():
        local = _load(local_ema_ckpt)
        if local.get("ema") is not None:
            base["ema"] = copy.deepcopy(local["ema"])
            base["updates"] = local.get("updates", base.get("updates", 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, out)
    return out


def aggregate_checkpoints(paths: list[Path], base_path: Path, out: Path, *, backbone_only: bool) -> Path:
    base = _load(base_path)
    avg = {}
    state_dicts = [_state_dict(_load(path), "model") for path in paths]
    base_state = _state_dict(base, "model")
    for key, value in base_state.items():
        should_average = (not backbone_only) or _is_backbone_key(key)
        if should_average and value.dtype.is_floating_point:
            avg[key] = torch.stack([sd[key].float() for sd in state_dicts], dim=0).mean(dim=0).to(value.dtype)
        else:
            avg[key] = value

    model = base["model"].float()
    model.load_state_dict(avg, strict=False)
    base["model"] = model.half()
    if base.get("ema") is not None:
        ema = base["ema"].float()
        ema.load_state_dict(avg, strict=False)
        base["ema"] = ema.half()
    base["epoch"] = -1
    base["optimizer"] = None
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, out)
    return out


def run_protocol(args: argparse.Namespace) -> None:
    setup.build_base_configs()
    pretrained = PRETRAINED_PATH if args.dry_run else download_pretrained()
    if not args.dry_run:
        check_runtime_dependencies()
    GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    CLIENT_STATE_DIR.mkdir(parents=True, exist_ok=True)
    config_device = "" if args.gpus > 1 else args.device

    current_global = GLOBAL_DIR / "round000_warmup.pt"
    if not args.dry_run and current_global.exists() and not args.force_warmup:
        ok, reason = validate_checkpoint(current_global)
        if ok:
            print(f"Reusing completed warm-up checkpoint: {current_global}")
        else:
            print(f"Warm-up checkpoint exists but is invalid ({reason}); rerunning warm-up.")
            current_global.unlink()

    if args.dry_run or not current_global.exists():
        warmup_cfg = setup.write_config(
            "runtime_server_warmup.yaml",
            setup.efficientteacher_config(
                name="runtime_server_warmup",
                train=setup.LIST_ROOT / "server_cloudy_train.txt",
                val=setup.LIST_ROOT / "server_cloudy_val.txt",
                target=None,
                weights=str(pretrained.resolve()),
                epochs=args.warmup_epochs,
                train_scope="all",
                batch_size=args.batch_size,
                workers=args.workers,
                device=config_device,
            ),
        )
        global_ckpt = run_train(warmup_cfg, args.dry_run, gpus=args.gpus, master_port=args.master_port)
        if not args.dry_run:
            make_start_checkpoint(global_ckpt, current_global)
    if args.dry_run:
        print("Dry run complete. Commands/configs are generated; no training was executed.")
        return

    history = []
    for phase, rounds in [(1, args.phase1_rounds), (2, args.phase2_rounds)]:
        for round_idx in range(1, rounds + 1):
            local_paths = []
            for client in setup.CLIENTS:
                target = setup.LIST_ROOT / f"client_{client['id']}_{client['weather']}_target.txt"
                start = CLIENT_STATE_DIR / f"client_{client['id']}_phase{phase}_round{round_idx:03d}_start.pt"
                previous = CLIENT_STATE_DIR / f"client_{client['id']}_latest.pt"
                make_start_checkpoint(current_global, start, previous)
                run_name = f"phase{phase}_round{round_idx:03d}_client{client['id']}_{client['weather']}"
                cfg = write_runtime_config(
                    run_name,
                    target=target,
                    weights=start,
                    phase=phase,
                    role="client",
                    round_idx=round_idx,
                    batch_size=args.batch_size,
                    workers=args.workers,
                    device=config_device,
                )
                ckpt = run_train(cfg, args.dry_run, gpus=args.gpus, master_port=args.master_port)
                local_paths.append(ckpt)
                make_start_checkpoint(ckpt, previous)

            server_start = GLOBAL_DIR / f"phase{phase}_round{round_idx:03d}_server_start.pt"
            make_start_checkpoint(current_global, server_start)
            server_name = f"phase{phase}_round{round_idx:03d}_server"
            server_cfg = write_runtime_config(
                server_name,
                target=None,
                weights=server_start,
                phase=phase,
                role="server",
                round_idx=round_idx,
                batch_size=args.batch_size,
                workers=args.workers,
                device=config_device,
            )
            server_ckpt = run_train(server_cfg, args.dry_run, gpus=args.gpus, master_port=args.master_port)

            next_global = GLOBAL_DIR / f"phase{phase}_round{round_idx:03d}_global.pt"
            aggregate_checkpoints(local_paths + [server_ckpt], server_ckpt, next_global, backbone_only=(phase == 1))
            current_global = next_global
            history.append({"phase": phase, "round": round_idx, "global": str(current_global.resolve())})
            (setup.WORK_ROOT / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(f"Completed phase {phase} round {round_idx}: {current_global}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--warmup-epochs", type=int, default=50)
    parser.add_argument("--phase1-rounds", type=int, default=100)
    parser.add_argument("--phase2-rounds", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers. Use 0 in Docker/Colab when /dev/shm is small.")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs for torch.distributed.run. Use 2 on a 2x RTX 6000 Ada node.")
    parser.add_argument("--master-port", type=int, default=29500, help="DDP master port for torch.distributed.run.")
    parser.add_argument("--device", default="")
    parser.add_argument("--force-warmup", action="store_true", help="Rerun warm-up even if round000_warmup.pt already exists.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.setup_only:
        setup.build_base_configs()
    else:
        run_protocol(parsed)
