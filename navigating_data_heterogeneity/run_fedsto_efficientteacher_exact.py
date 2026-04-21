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
HISTORY_PATH = setup.WORK_ROOT / "history.json"


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


def checkpoint_present(path: Path) -> bool:
    return path.exists() and path.stat().st_size >= 1024 * 1024


def load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse {HISTORY_PATH}: {exc}") from exc
    if not isinstance(history, list):
        raise RuntimeError(f"Expected {HISTORY_PATH} to contain a list, got {type(history).__name__}")
    return history


def write_history(history: list[dict]) -> None:
    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


def completed_history_prefix(
    history: list[dict],
    *,
    phase1_rounds: int,
    phase2_rounds: int,
    warmup_checkpoint: Path,
) -> tuple[list[dict], Path, tuple[int, int] | None]:
    """Return the continuous completed prefix and the next phase/round to run."""
    by_round: dict[tuple[int, int], dict] = {}
    for entry in history:
        try:
            key = (int(entry["phase"]), int(entry["round"]))
        except (KeyError, TypeError, ValueError):
            continue
        by_round.setdefault(key, entry)

    normalized: list[dict] = []
    current_global = warmup_checkpoint
    for phase, rounds in [(1, phase1_rounds), (2, phase2_rounds)]:
        for round_idx in range(1, rounds + 1):
            entry = by_round.get((phase, round_idx))
            global_path = Path(entry.get("global", "")) if entry else None
            if global_path is not None and checkpoint_present(global_path):
                current_global = global_path
                normalized.append(
                    {
                        "phase": phase,
                        "round": round_idx,
                        "global": str(global_path.resolve()),
                    }
                )
                continue
            return normalized, current_global, (phase, round_idx)
    return normalized, current_global, None


def reuse_checkpoint_if_valid(path: Path, description: str, *, force_retrain: bool) -> Path | None:
    if force_retrain or not checkpoint_present(path):
        return None
    ok, reason = validate_checkpoint(path)
    if ok:
        print(f"Reusing completed {description}: {path}")
        return path
    print(f"Existing {description} checkpoint is invalid ({reason}); rerunning.")
    return None


def _remove_file(path: Path) -> tuple[int, int]:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return 0, 0
    path.unlink()
    return 1, size


def cleanup_round_intermediates(phase: int, round_idx: int) -> tuple[int, int]:
    removed = 0
    freed = 0
    for client in setup.CLIENTS:
        run_name = f"phase{phase}_round{round_idx:03d}_client{client['id']}_{client['weather']}"
        weights_dir = checkpoint_path(run_name).parent
        for filename in ("last.pt", "best.pt"):
            count, size = _remove_file(weights_dir / filename)
            removed += count
            freed += size
        count, size = _remove_file(CLIENT_STATE_DIR / f"client_{client['id']}_phase{phase}_round{round_idx:03d}_start.pt")
        removed += count
        freed += size

    server_name = f"phase{phase}_round{round_idx:03d}_server"
    server_weights_dir = checkpoint_path(server_name).parent
    for filename in ("last.pt", "best.pt"):
        count, size = _remove_file(server_weights_dir / filename)
        removed += count
        freed += size
    count, size = _remove_file(GLOBAL_DIR / f"phase{phase}_round{round_idx:03d}_server_start.pt")
    removed += count
    freed += size
    return removed, freed


def cleanup_completed_intermediates(history: list[dict]) -> None:
    removed = 0
    freed = 0
    for entry in history:
        try:
            phase = int(entry["phase"])
            round_idx = int(entry["round"])
        except (KeyError, TypeError, ValueError):
            continue
        count, size = cleanup_round_intermediates(phase, round_idx)
        removed += count
        freed += size
    if removed:
        print(f"Cleaned {removed} completed-round intermediate checkpoint files ({freed / 1024 ** 3:.2f} GiB).")


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

    if args.force_restart:
        history = []
        print("Ignoring existing history because --force-restart was set.")
    else:
        loaded_history = load_history()
        history, current_global, next_round = completed_history_prefix(
            loaded_history,
            phase1_rounds=args.phase1_rounds,
            phase2_rounds=args.phase2_rounds,
            warmup_checkpoint=current_global,
        )
        if loaded_history != history:
            write_history(history)
        if next_round is None:
            total_rounds = args.phase1_rounds + args.phase2_rounds
            print(f"All requested federated rounds are already complete ({total_rounds}/{total_rounds}).")
            print(f"Latest global checkpoint: {current_global}")
            return
        if history:
            phase, round_idx = next_round
            print(
                f"Resuming after {len(history)} completed federated rounds "
                f"from phase {phase} round {round_idx}."
            )
            print(f"Current global checkpoint: {current_global}")
        else:
            print("No completed federated rounds found; starting from phase 1 round 1.")

    if not args.keep_intermediate_checkpoints:
        cleanup_completed_intermediates(history)

    completed = {(int(entry["phase"]), int(entry["round"])) for entry in history}
    for phase, rounds in [(1, args.phase1_rounds), (2, args.phase2_rounds)]:
        for round_idx in range(1, rounds + 1):
            if (phase, round_idx) in completed:
                continue
            next_global = GLOBAL_DIR / f"phase{phase}_round{round_idx:03d}_global.pt"
            reused_global = reuse_checkpoint_if_valid(
                next_global,
                f"global checkpoint for phase {phase} round {round_idx}",
                force_retrain=args.force_retrain,
            )
            if reused_global is not None:
                current_global = reused_global
                history.append({"phase": phase, "round": round_idx, "global": str(current_global.resolve())})
                write_history(history)
                completed.add((phase, round_idx))
                print(f"Recovered phase {phase} round {round_idx} from existing global checkpoint.")
                if not args.keep_intermediate_checkpoints:
                    cleanup_round_intermediates(phase, round_idx)
                continue

            local_paths = []
            for client in setup.CLIENTS:
                target = setup.LIST_ROOT / f"client_{client['id']}_{client['weather']}_target.txt"
                start = CLIENT_STATE_DIR / f"client_{client['id']}_phase{phase}_round{round_idx:03d}_start.pt"
                previous = CLIENT_STATE_DIR / f"client_{client['id']}_latest.pt"
                run_name = f"phase{phase}_round{round_idx:03d}_client{client['id']}_{client['weather']}"
                ckpt = reuse_checkpoint_if_valid(
                    checkpoint_path(run_name),
                    f"client run {run_name}",
                    force_retrain=args.force_retrain,
                )
                if ckpt is not None:
                    local_paths.append(ckpt)
                    make_start_checkpoint(ckpt, previous)
                    continue

                if not checkpoint_present(start):
                    make_start_checkpoint(current_global, start, previous)
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
            server_name = f"phase{phase}_round{round_idx:03d}_server"
            server_ckpt = reuse_checkpoint_if_valid(
                checkpoint_path(server_name),
                f"server run {server_name}",
                force_retrain=args.force_retrain,
            )
            if server_ckpt is None:
                if not checkpoint_present(server_start):
                    make_start_checkpoint(current_global, server_start)
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

            aggregate_checkpoints(local_paths + [server_ckpt], server_ckpt, next_global, backbone_only=(phase == 1))
            current_global = next_global
            history.append({"phase": phase, "round": round_idx, "global": str(current_global.resolve())})
            write_history(history)
            completed.add((phase, round_idx))
            print(f"Completed phase {phase} round {round_idx}: {current_global}")
            if not args.keep_intermediate_checkpoints:
                cleanup_round_intermediates(phase, round_idx)


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
    parser.add_argument("--force-restart", action="store_true", help="Ignore history.json and start federated rounds from phase 1 round 1.")
    parser.add_argument("--force-retrain", action="store_true", help="Rerun train jobs even when their last.pt checkpoints already exist.")
    parser.add_argument(
        "--keep-intermediate-checkpoints",
        action="store_true",
        help="Keep per-run last/best weights and per-round start checkpoints after a global round is complete.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.setup_only:
        setup.build_base_configs()
    else:
        run_protocol(parsed)
