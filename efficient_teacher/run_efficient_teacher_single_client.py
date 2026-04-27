#!/usr/bin/env python3
"""Run a single-client EfficientTeacher baseline with DQA-matched conditions.

This runner intentionally reuses the FedSTO/DQA EfficientTeacher scaffold:

- BDD100K paper20k data split from ``navigating_data_heterogeneity``
- YOLOv5L EfficientTeacher checkpoint and config shape
- server warm-up on cloudy labeled GT
- one local client using online pseudo-labels
- one labeled server GT epoch after every pseudo-label client epoch

The only protocol change is that ``CLIENTS`` is reduced to one weather client.
With one client, aggregation is still routed through the FedSTO helper so phase-1
backbone-only behavior remains aligned with the DQA notebooks.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


RESEARCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RESEARCH_ROOT.parent
NAV_ROOT = REPO_ROOT / "navigating_data_heterogeneity"
DEFAULT_WORK_ROOT = RESEARCH_ROOT / "efficientteacher_single_client"
PROTOCOL_VERSION_ET = "efficientteacher_single_client_dqa_matched_phase2_orthogonal_v3"
PROTOCOL_VERSION_LOCALEMA = "efficientteacher_single_client_localema_phase2_orthogonal_v2"
LEGACY_PHASE1_PROTOCOLS = {
    PROTOCOL_VERSION_ET: {"efficientteacher_single_client_dqa_matched_no_localema_v2"},
    PROTOCOL_VERSION_LOCALEMA: {"efficientteacher_single_client_localema_dqa_matched_v1"},
}


def normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.workspace_root = Path(args.workspace_root).expanduser().resolve()
    if args.pseudo_stats_root is None:
        args.pseudo_stats_root = args.workspace_root / "pseudo_stats"
    else:
        args.pseudo_stats_root = Path(args.pseudo_stats_root).expanduser().resolve()
    if args.log_file is not None:
        args.log_file = Path(args.log_file).expanduser().resolve()
    if args.protocol_version is None:
        args.protocol_version = PROTOCOL_VERSION_LOCALEMA if args.local_ema else PROTOCOL_VERSION_ET
    return args


def _import_fedsto_modules():
    if str(NAV_ROOT) not in sys.path:
        sys.path.insert(0, str(NAV_ROOT))
    setup = importlib.import_module("setup_fedsto_exact_reproduction")
    fedsto = importlib.import_module("run_fedsto_efficientteacher_exact")
    return setup, fedsto


def _select_single_client(setup, client_id: int | None, weather: str | None) -> dict:
    original_clients = list(setup.CLIENTS)
    matches = []
    for client in original_clients:
        if client_id is not None and int(client["id"]) != int(client_id):
            continue
        if weather is not None and str(client["weather"]) != weather:
            continue
        matches.append(dict(client))
    if not matches:
        available = ", ".join(f"{client['id']}:{client['weather']}" for client in original_clients)
        raise ValueError(f"No matching client found. Requested id={client_id}, weather={weather}. Available: {available}")
    selected = matches[0]
    setup.CLIENTS = [selected]
    return selected


def configure_modules(args: argparse.Namespace):
    setup, fedsto = _import_fedsto_modules()
    client = _select_single_client(setup, args.client_id, args.client_weather)
    fedsto.apply_workspace_root(args.workspace_root)
    # ``fedsto.setup`` points at the same imported module, but assign explicitly
    # so future refactors cannot accidentally re-expand the client list.
    fedsto.setup.CLIENTS = [client]
    return setup, fedsto, client


def run_name(phase: int, round_idx: int, client: dict | None = None) -> str:
    prefix = f"et_phase{phase}_round{round_idx:03d}"
    if client is None:
        return f"{prefix}_server"
    return f"{prefix}_client{client['id']}_{client['weather']}"


def pseudo_stats_path(root: Path, phase: int, round_idx: int, client: dict) -> Path:
    return root / f"phase{phase}_round{round_idx:03d}_client{client['id']}_{client['weather']}.json"


def checkpoint_run_name(config: Path) -> str:
    cfg_name = config.stem
    if cfg_name.startswith("runtime_phase"):
        import yaml

        with config.open(encoding="utf-8") as f:
            return str(yaml.safe_load(f)["name"])
    return cfg_name


def protocol_compatible(actual: str | None, expected: str | None, phase: int) -> bool:
    if expected is None:
        return True
    if actual == expected:
        return True
    return phase == 1 and actual in LEGACY_PHASE1_PROTOCOLS.get(expected, set())


def completed_et_history_prefix(
    fedsto,
    history: list[dict],
    *,
    phase1_rounds: int,
    phase2_rounds: int,
    warmup_checkpoint: Path,
    expected_protocol: str | None,
) -> tuple[list[dict], Path, tuple[int, int] | None]:
    """Return completed ET rounds, preserving old phase-1 checkpoints after phase-2 fixes."""
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
            if entry and not protocol_compatible(entry.get("protocol"), expected_protocol, phase):
                return normalized, current_global, (phase, round_idx)

            global_path = Path(entry.get("global", "")) if entry else None
            if global_path is not None and fedsto.checkpoint_present(global_path):
                current_global = global_path
                normalized.append(
                    {
                        "phase": phase,
                        "round": round_idx,
                        "global": str(global_path.resolve()),
                        "protocol": entry.get("protocol") if entry else expected_protocol,
                    }
                )
                continue

            return normalized, current_global, (phase, round_idx)
    return normalized, current_global, None


def write_et_runtime_config(
    fedsto,
    *,
    name: str,
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
    cfg = fedsto.setup.efficientteacher_config(
        name=name,
        train=fedsto.setup.LIST_ROOT / "server_cloudy_train.txt",
        val=fedsto.setup.LIST_ROOT / "server_cloudy_val.txt",
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
    return fedsto.setup.write_config(f"runtime_phase{phase}_{role}_round{round_idx}_{name}.yaml", cfg)


def ensure_disk_space(
    path: Path,
    min_free_gib: float,
    *,
    setup=None,
    fedsto=None,
    client: dict | None = None,
    history: list[dict] | None = None,
    keep_intermediates: bool = False,
) -> None:
    if min_free_gib <= 0:
        return
    path.mkdir(parents=True, exist_ok=True)
    required = int(min_free_gib * 1024**3)
    free = shutil.disk_usage(path).free
    if free >= required:
        return

    if history and setup is not None and fedsto is not None and client is not None and not keep_intermediates:
        cleanup_completed_intermediates(setup, fedsto, client, history)
        free = shutil.disk_usage(path).free
        if free >= required:
            return

    raise RuntimeError(
        f"Only {free / 1024**3:.2f} GiB free under {path}; "
        f"EfficientTeacher single-client requires at least {min_free_gib:.2f} GiB. "
        "Free disk space or lower --min-free-gib for a tiny smoke test."
    )


def remove_file(path: Path) -> tuple[int, int]:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return 0, 0
    path.unlink()
    return 1, size


def cleanup_round_intermediates(setup, fedsto, client: dict, phase: int, round_idx: int) -> tuple[int, int]:
    removed = 0
    freed = 0

    client_weights_dir = fedsto.checkpoint_path(run_name(phase, round_idx, client)).parent
    for filename in ("last.pt", "best.pt"):
        count, size = remove_file(client_weights_dir / filename)
        removed += count
        freed += size
    count, size = remove_file(
        fedsto.CLIENT_STATE_DIR / f"client_{client['id']}_phase{phase}_round{round_idx:03d}_start.pt"
    )
    removed += count
    freed += size

    server_weights_dir = fedsto.checkpoint_path(run_name(phase, round_idx)).parent
    for filename in ("last.pt", "best.pt"):
        count, size = remove_file(server_weights_dir / filename)
        removed += count
        freed += size
    count, size = remove_file(fedsto.GLOBAL_DIR / f"phase{phase}_round{round_idx:03d}_server_start.pt")
    removed += count
    freed += size
    count, size = remove_file(fedsto.GLOBAL_DIR / f"phase{phase}_round{round_idx:03d}_client_aggregate.pt")
    removed += count
    freed += size
    return removed, freed


def cleanup_completed_intermediates(setup, fedsto, client: dict, history: list[dict]) -> None:
    removed = 0
    freed = 0
    for entry in history:
        try:
            phase = int(entry["phase"])
            round_idx = int(entry["round"])
        except (KeyError, TypeError, ValueError):
            continue
        count, size = cleanup_round_intermediates(setup, fedsto, client, phase, round_idx)
        removed += count
        freed += size
    if removed:
        print(f"Cleaned {removed} completed-round intermediate checkpoint files ({freed / 1024**3:.2f} GiB).")


def run_train(
    fedsto,
    config: Path,
    args: argparse.Namespace,
    *,
    extra_env: dict[str, str] | None = None,
) -> Path:
    if args.gpus > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(args.gpus),
            "--master_port",
            str(args.master_port),
            "train.py",
            "--cfg",
            str(config.resolve()),
        ]
    else:
        cmd = [sys.executable, "train.py", "--cfg", str(config.resolve())]

    print(" ".join(cmd))
    if args.dry_run:
        return fedsto.checkpoint_path(checkpoint_run_name(config))

    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.skip_final_best_val:
        env["ET_SKIP_AFTER_TRAIN_BEST_VAL"] = "1"
    else:
        env.pop("ET_SKIP_AFTER_TRAIN_BEST_VAL", None)
    if extra_env:
        env.update(extra_env)

    if args.stream_train_output:
        try:
            subprocess.run(cmd, cwd=fedsto.setup.ET_ROOT, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Training failed for {config}") from exc
    else:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        with args.log_file.open("a", encoding="utf-8") as log:
            log.write("\n" + "=" * 100 + "\n")
            log.write(" ".join(cmd) + "\n")
            try:
                subprocess.run(cmd, cwd=fedsto.setup.ET_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as exc:
                print(f"Training subprocess failed with exit code {exc.returncode}: {' '.join(cmd)}")
                print(f"Last 120 lines from {args.log_file}:")
                try:
                    with args.log_file.open(encoding="utf-8", errors="replace") as failed_log:
                        for line in deque(failed_log, maxlen=120):
                            print(line.rstrip())
                except OSError as log_exc:
                    print(f"Could not read failed training log: {log_exc}")
                raise RuntimeError(f"Training failed for {config}; see {args.log_file}") from exc
        print(f"Training output appended to {args.log_file}")

    return fedsto.checkpoint_path(checkpoint_run_name(config))


def append_round_summary(workspace: Path, payload: dict) -> None:
    path = workspace / "round_summaries.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def run_protocol(args: argparse.Namespace) -> None:
    setup, fedsto, client = configure_modules(args)
    setup.build_base_configs()
    mode = "LocalEMA" if args.local_ema else "EfficientTeacher"
    print(f"Running {mode} single-client protocol with client_{client['id']}_{client['weather']}.")
    pretrained = fedsto.PRETRAINED_PATH if args.dry_run else fedsto.download_pretrained()
    if not args.dry_run:
        fedsto.check_runtime_dependencies()

    fedsto.GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    fedsto.CLIENT_STATE_DIR.mkdir(parents=True, exist_ok=True)
    args.pseudo_stats_root.mkdir(parents=True, exist_ok=True)
    if args.log_file is None:
        args.log_file = args.workspace_root / "efficientteacher_latest.log"
    if not args.dry_run and not args.append_train_log and args.log_file.exists():
        args.log_file.unlink()
    args.gpus = fedsto.resolve_gpus(args.gpus)
    config_device = "" if args.gpus > 1 else args.device
    ensure_disk_space(args.workspace_root, args.min_free_gib)

    current_global = fedsto.GLOBAL_DIR / "round000_warmup.pt"
    if args.warmup_checkpoint is not None:
        source_warmup = Path(args.warmup_checkpoint).expanduser().resolve()
        ok, reason = fedsto.validate_checkpoint(source_warmup)
        if not ok:
            raise RuntimeError(f"Warm-up checkpoint is invalid: {source_warmup} ({reason})")
        if args.dry_run:
            print(f"Dry run: would seed warm-up from {source_warmup}")
        else:
            print(f"Seeding warm-up from external checkpoint: {source_warmup}")
            fedsto.make_start_checkpoint(
                source_warmup,
                current_global,
                protocol=args.protocol_version,
                stage="et_single_client_external_warmup_seed",
            )

    if not args.dry_run and current_global.exists() and not args.force_warmup:
        ok, reason = fedsto.validate_checkpoint(current_global)
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
        global_ckpt = run_train(fedsto, warmup_cfg, args)
        if not args.dry_run:
            fedsto.make_start_checkpoint(
                global_ckpt,
                current_global,
                protocol=args.protocol_version,
                stage="et_single_client_warmup_global",
            )

    if args.dry_run:
        print(f"Dry run complete. EfficientTeacher single-client workspace: {args.workspace_root}")
        print(f"Selected client: client_{client['id']}_{client['weather']}")
        return

    if args.force_restart:
        history = []
        print("Ignoring existing history because --force-restart was set.")
    else:
        loaded_history = fedsto.load_history()
        history, current_global, next_round = completed_et_history_prefix(
            fedsto,
            loaded_history,
            phase1_rounds=args.phase1_rounds,
            phase2_rounds=args.phase2_rounds,
            warmup_checkpoint=current_global,
            expected_protocol=args.protocol_version,
        )
        if loaded_history != history:
            fedsto.write_history(history)
        if next_round is None:
            total_rounds = args.phase1_rounds + args.phase2_rounds
            print(f"All requested EfficientTeacher rounds are already complete ({total_rounds}/{total_rounds}).")
            print(f"Latest global checkpoint: {current_global}")
            return
        if history:
            phase, round_idx = next_round
            print(
                f"Resuming EfficientTeacher single-client after {len(history)} completed rounds "
                f"from phase {phase} round {round_idx}."
            )
            print(f"Current global checkpoint: {current_global}")
        else:
            print("No completed EfficientTeacher single-client rounds found; starting from phase 1 round 1.")

    if not args.keep_intermediate_checkpoints:
        cleanup_completed_intermediates(setup, fedsto, client, history)

    completed = {(int(entry["phase"]), int(entry["round"])) for entry in history}
    for phase, rounds in [(1, args.phase1_rounds), (2, args.phase2_rounds)]:
        for round_idx in range(1, rounds + 1):
            if (phase, round_idx) in completed:
                continue
            ensure_disk_space(
                args.workspace_root,
                args.min_free_gib,
                setup=setup,
                fedsto=fedsto,
                client=client,
                history=history,
                keep_intermediates=args.keep_intermediate_checkpoints,
            )
            next_global = fedsto.GLOBAL_DIR / f"phase{phase}_round{round_idx:03d}_global.pt"
            reused_global = fedsto.reuse_checkpoint_if_valid(
                next_global,
                f"EfficientTeacher global checkpoint for phase {phase} round {round_idx}",
                force_retrain=args.force_retrain,
                expected_protocol=args.protocol_version,
            )
            if reused_global is not None:
                current_global = reused_global
                history.append(
                    {
                        "phase": phase,
                        "round": round_idx,
                        "global": str(current_global.resolve()),
                        "protocol": args.protocol_version,
                    }
                )
                fedsto.write_history(history)
                completed.add((phase, round_idx))
                print(f"Recovered EfficientTeacher phase {phase} round {round_idx} from existing global checkpoint.")
                if not args.keep_intermediate_checkpoints:
                    cleanup_round_intermediates(setup, fedsto, client, phase, round_idx)
                continue

            target = setup.LIST_ROOT / f"client_{client['id']}_{client['weather']}_target.txt"
            client_start = fedsto.CLIENT_STATE_DIR / f"client_{client['id']}_phase{phase}_round{round_idx:03d}_start.pt"
            latest_client = fedsto.CLIENT_STATE_DIR / f"client_{client['id']}_latest.pt"
            client_name = run_name(phase, round_idx, client)
            stats_path = pseudo_stats_path(args.pseudo_stats_root, phase, round_idx, client)

            client_ckpt = fedsto.reuse_checkpoint_if_valid(
                fedsto.checkpoint_path(client_name),
                f"EfficientTeacher client run {client_name}",
                force_retrain=args.force_retrain,
                expected_protocol=args.protocol_version,
            )
            if client_ckpt is not None and args.require_pseudo_stats and not stats_path.exists():
                print(
                    f"Existing client checkpoint is usable but pseudo-label stats are missing "
                    f"({stats_path}); rerunning this client."
                )
                client_ckpt = None
            if client_ckpt is None:
                if not fedsto.checkpoint_matches_protocol(client_start, args.protocol_version):
                    fedsto.make_start_checkpoint(
                        current_global,
                        client_start,
                        latest_client if args.local_ema else None,
                        protocol=args.protocol_version,
                        stage=f"et_phase{phase}_round{round_idx:03d}_client{client['id']}_start",
                    )
                client_cfg = write_et_runtime_config(
                    fedsto,
                    name=client_name,
                    target=target,
                    weights=client_start,
                    phase=phase,
                    role="client",
                    round_idx=round_idx,
                    batch_size=args.batch_size,
                    workers=args.workers,
                    device=config_device,
                )
                if stats_path.exists():
                    stats_path.unlink()
                client_ckpt = run_train(
                    fedsto,
                    client_cfg,
                    args,
                    extra_env={
                        "DQA_PSEUDO_STATS_OUT": str(stats_path.resolve()),
                        "DQA_CLIENT_ID": str(client["id"]),
                        "DQA_PHASE": str(phase),
                        "DQA_ROUND": str(round_idx),
                    },
                )
                fedsto.mark_checkpoint_protocol(
                    client_ckpt,
                    args.protocol_version,
                    f"et_phase{phase}_round{round_idx:03d}_client{client['id']}_pseudo_gt",
                )
            fedsto.make_start_checkpoint(
                client_ckpt,
                latest_client,
                protocol=args.protocol_version,
                stage=f"et_client_{client['id']}_latest_local_ema_source",
            )

            aggregate_start = fedsto.GLOBAL_DIR / f"phase{phase}_round{round_idx:03d}_client_aggregate.pt"
            if not fedsto.checkpoint_matches_protocol(aggregate_start, args.protocol_version):
                fedsto.aggregate_checkpoints([client_ckpt], current_global, aggregate_start, backbone_only=(phase == 1))
                fedsto.mark_checkpoint_protocol(
                    aggregate_start,
                    args.protocol_version,
                    f"et_phase{phase}_round{round_idx:03d}_single_client_aggregate",
                )

            server_start = fedsto.GLOBAL_DIR / f"phase{phase}_round{round_idx:03d}_server_start.pt"
            server_name = run_name(phase, round_idx)
            server_ckpt = fedsto.reuse_checkpoint_if_valid(
                fedsto.checkpoint_path(server_name),
                f"EfficientTeacher server GT run {server_name}",
                force_retrain=args.force_retrain,
                expected_protocol=args.protocol_version,
            )
            if server_ckpt is None:
                if not fedsto.checkpoint_matches_protocol(server_start, args.protocol_version):
                    fedsto.make_start_checkpoint(
                        aggregate_start,
                        server_start,
                        protocol=args.protocol_version,
                        stage=f"et_phase{phase}_round{round_idx:03d}_server_start",
                    )
                server_cfg = write_et_runtime_config(
                    fedsto,
                    name=server_name,
                    target=None,
                    weights=server_start,
                    phase=phase,
                    role="server",
                    round_idx=round_idx,
                    batch_size=args.batch_size,
                    workers=args.workers,
                    device=config_device,
                )
                server_ckpt = run_train(fedsto, server_cfg, args)
                fedsto.mark_checkpoint_protocol(
                    server_ckpt,
                    args.protocol_version,
                    f"et_phase{phase}_round{round_idx:03d}_server_gt_epoch",
                )

            ensure_disk_space(
                args.workspace_root,
                args.min_free_gib,
                setup=setup,
                fedsto=fedsto,
                client=client,
                history=history,
                keep_intermediates=args.keep_intermediate_checkpoints,
            )
            fedsto.make_start_checkpoint(
                server_ckpt,
                next_global,
                protocol=args.protocol_version,
                stage=f"et_phase{phase}_round{round_idx:03d}_global_after_server_gt",
            )
            current_global = next_global
            history.append(
                {
                    "phase": phase,
                    "round": round_idx,
                    "global": str(current_global.resolve()),
                    "protocol": args.protocol_version,
                }
            )
            fedsto.write_history(history)
            append_round_summary(
                args.workspace_root,
                {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "phase": phase,
                    "round": round_idx,
                    "client": client,
                    "client_checkpoint": str(client_ckpt.resolve()),
                    "client_aggregate": str(aggregate_start.resolve()),
                    "server_checkpoint": str(server_ckpt.resolve()),
                    "global": str(current_global.resolve()),
                    "pseudo_stats": str(stats_path.resolve()),
                    "protocol": args.protocol_version,
                    "local_ema": bool(args.local_ema),
                },
            )
            completed.add((phase, round_idx))
            print(f"Completed EfficientTeacher phase {phase} round {round_idx}: {current_global}")
            if not args.keep_intermediate_checkpoints:
                cleanup_round_intermediates(setup, fedsto, client, phase, round_idx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--pseudo-stats-root", type=Path, default=None)
    parser.add_argument("--client-id", type=int, default=0)
    parser.add_argument("--client-weather", default=None)
    parser.add_argument("--warmup-epochs", type=int, default=15)
    parser.add_argument("--phase1-rounds", type=int, default=14)
    parser.add_argument("--phase2-rounds", type=int, default=27)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=29520)
    parser.add_argument("--device", default="")
    parser.add_argument("--force-warmup", action="store_true")
    parser.add_argument("--force-restart", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--keep-intermediate-checkpoints", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=70.0)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--append-train-log", action="store_true")
    parser.add_argument("--stream-train-output", action="store_true")
    parser.add_argument("--require-pseudo-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--skip-final-best-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip EfficientTeacher's redundant post-train best.pt validation pass, which can fail on Ada multi-GPU runs.",
    )
    parser.add_argument(
        "--warmup-checkpoint",
        type=Path,
        default=None,
        help="Seed round000_warmup.pt from an existing checkpoint instead of training warm-up from the raw pretrained weights.",
    )
    parser.add_argument(
        "--local-ema",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Carry each client's previous EMA teacher into the next round. Disabled by default for plain ET.",
    )
    parser.add_argument("--protocol-version", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = normalize_paths(parse_args())
    if parsed.setup_only:
        configure_modules(parsed)[0].build_base_configs()
    else:
        run_protocol(parsed)
