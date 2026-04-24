#!/usr/bin/env python3
"""Run the FedSTO reproduction with DQA-CWA aggregation in phase 2.

The script mirrors navigating_data_heterogeneity/run_fedsto_efficientteacher_exact.py
but writes generated configs, runs, checkpoints, and aggregation state under this
research directory.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dqa_cwa_aggregation import (
    AggregationConfig,
    aggregate_checkpoints,
    compute_reliability,
    load_round_stats,
    save_state,
)


RESEARCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RESEARCH_ROOT.parent
NAV_ROOT = REPO_ROOT / "navigating_data_heterogeneity"
DEFAULT_DQA_WORK_ROOT = RESEARCH_ROOT / "efficientteacher_dqa_cwa"


def _prepare_fedsto_modules(work_root: Path):
    if str(NAV_ROOT) not in sys.path:
        sys.path.insert(0, str(NAV_ROOT))

    setup = importlib.import_module("setup_fedsto_exact_reproduction")
    setup.WORK_ROOT = work_root
    setup.LIST_ROOT = work_root / "data_lists"
    setup.CONFIG_ROOT = work_root / "configs"
    setup.RUN_ROOT = work_root / "runs"

    fedsto = importlib.import_module("run_fedsto_efficientteacher_exact")
    fedsto.PRETRAINED_PATH = work_root / "weights" / "efficient-yolov5l.pt"
    fedsto.GLOBAL_DIR = work_root / "global_checkpoints"
    fedsto.CLIENT_STATE_DIR = work_root / "client_states"
    fedsto.HISTORY_PATH = work_root / "history.json"
    return setup, fedsto


def _stats_path(root: Path, phase: int, round_idx: int) -> Path:
    return root / f"phase{phase}_round{round_idx:03d}.json"


def _client_stats_path(root: Path, phase: int, round_idx: int, client: dict) -> Path:
    return root / f"phase{phase}_round{round_idx:03d}_client{client['id']}.json"


def _dqa_config(args: argparse.Namespace) -> AggregationConfig:
    return AggregationConfig(
        num_classes=args.num_classes,
        count_ema=args.count_ema,
        quality_ema=args.quality_ema,
        alpha_ema=args.alpha_ema,
        temperature=args.temperature,
        uniform_mix=args.uniform_mix,
        classwise_blend=args.classwise_blend,
        stability_lambda=args.stability_lambda,
        min_effective_count=args.min_effective_count,
        server_anchor=args.server_anchor,
    )


def _dqa_run_name(phase: int, round_idx: int, client: dict | None = None) -> str:
    prefix = f"dqa_phase{phase}_round{round_idx:03d}"
    if client is None:
        return f"{prefix}_server"
    return f"{prefix}_client{client['id']}_{client['weather']}"


def _remove_file(path: Path) -> tuple[int, int]:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return 0, 0
    path.unlink()
    return 1, size


def cleanup_round_intermediates(setup, fedsto, phase: int, round_idx: int) -> tuple[int, int]:
    removed = 0
    freed = 0
    for client in setup.CLIENTS:
        weights_dir = fedsto.checkpoint_path(_dqa_run_name(phase, round_idx, client)).parent
        for filename in ("last.pt", "best.pt"):
            count, size = _remove_file(weights_dir / filename)
            removed += count
            freed += size
        count, size = _remove_file(
            fedsto.CLIENT_STATE_DIR / f"client_{client['id']}_phase{phase}_round{round_idx:03d}_start.pt"
        )
        removed += count
        freed += size

    server_weights_dir = fedsto.checkpoint_path(_dqa_run_name(phase, round_idx)).parent
    for filename in ("last.pt", "best.pt"):
        count, size = _remove_file(server_weights_dir / filename)
        removed += count
        freed += size
    count, size = _remove_file(fedsto.GLOBAL_DIR / f"phase{phase}_round{round_idx:03d}_server_start.pt")
    removed += count
    freed += size
    return removed, freed


def cleanup_completed_intermediates(setup, fedsto, history: list[dict]) -> None:
    removed = 0
    freed = 0
    for entry in history:
        try:
            phase = int(entry["phase"])
            round_idx = int(entry["round"])
        except (KeyError, TypeError, ValueError):
            continue
        count, size = cleanup_round_intermediates(setup, fedsto, phase, round_idx)
        removed += count
        freed += size
    if removed:
        print(f"Cleaned {removed} completed-round DQA intermediate checkpoint files ({freed / 1024 ** 3:.2f} GiB).")


def ensure_disk_space(
    path: Path,
    min_free_gib: float,
    *,
    setup=None,
    fedsto=None,
    history: list[dict] | None = None,
    keep_intermediates: bool = False,
) -> None:
    if min_free_gib <= 0:
        return
    path.mkdir(parents=True, exist_ok=True)
    required = int(min_free_gib * 1024 ** 3)
    free = shutil.disk_usage(path).free
    if free >= required:
        return

    if history and setup is not None and fedsto is not None and not keep_intermediates:
        cleanup_completed_intermediates(setup, fedsto, history)
        free = shutil.disk_usage(path).free
        if free >= required:
            return

    raise RuntimeError(
        f"Only {free / 1024 ** 3:.2f} GiB free under {path}; "
        f"DQA-CWA requires at least {min_free_gib:.2f} GiB. "
        "Free space or lower --min-free-gib before starting another train job."
    )


def run_train(
    fedsto,
    config: Path,
    args: argparse.Namespace,
    *,
    extra_env: dict[str, str] | None = None,
) -> Path:
    """Run EfficientTeacher while keeping notebook/stdout output compact by default."""
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
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    if args.dry_run:
        cfg_name = config.stem
        if cfg_name.startswith("runtime_phase"):
            with config.open(encoding="utf-8") as f:
                import yaml

                run_name = yaml.safe_load(f)["name"]
        else:
            run_name = cfg_name
        return fedsto.checkpoint_path(run_name)

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
                raise RuntimeError(f"Training failed for {config}; see {args.log_file}") from exc

        print(f"Training output appended to {args.log_file}")

    cfg_name = config.stem
    if cfg_name.startswith("runtime_phase"):
        with config.open(encoding="utf-8") as f:
            import yaml

            run_name = yaml.safe_load(f)["name"]
    else:
        run_name = cfg_name
    return fedsto.checkpoint_path(run_name)


def build_round_stats_from_clients(args: argparse.Namespace, phase: int, round_idx: int, setup) -> Path | None:
    round_stats = _stats_path(args.stats_root, phase, round_idx)
    client_entries = []
    for client in setup.CLIENTS:
        client_stats = _client_stats_path(args.stats_root, phase, round_idx, client)
        if not client_stats.exists():
            return None
        payload = json.loads(client_stats.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "clients" in payload:
            client_entries.extend(payload["clients"])
        else:
            client_entries.append(payload)

    round_stats.parent.mkdir(parents=True, exist_ok=True)
    round_stats.write_text(json.dumps({"clients": client_entries}, indent=2), encoding="utf-8")
    return round_stats


def rebuild_dqa_state_from_history(history: list[dict], args: argparse.Namespace, dqa_state: Path) -> None:
    dqa_entries = []
    for entry in history:
        try:
            phase = int(entry["phase"])
            round_idx = int(entry["round"])
        except (KeyError, TypeError, ValueError):
            continue
        if phase >= args.dqa_start_phase:
            dqa_entries.append((phase, round_idx))
    if not dqa_entries or dqa_state.exists():
        return

    state: dict = {"clients": {}, "alpha": {}}
    rebuilt = 0
    for phase, round_idx in dqa_entries:
        stats_file = _stats_path(args.stats_root, phase, round_idx)
        if not stats_file.exists():
            if args.fallback_fedavg_without_stats:
                continue
            raise FileNotFoundError(
                f"Cannot rebuild missing DQA-CWA EMA state because stats are missing: {stats_file}"
            )
        stats = load_round_stats(stats_file, args.num_classes)
        state, alpha, source_ids, active = compute_reliability(stats, state, _dqa_config(args))
        state["last_sources"] = source_ids
        state["last_alpha"] = alpha.tolist()
        state["last_active_classes"] = [bool(x) for x in active.tolist()]
        rebuilt += 1

    if rebuilt:
        save_state(dqa_state, state)
        print(f"Rebuilt DQA-CWA EMA state for {rebuilt} completed dynamic rounds: {dqa_state}")


def run_protocol(args: argparse.Namespace) -> None:
    setup, fedsto = _prepare_fedsto_modules(args.workspace_root)
    setup.build_base_configs()
    pretrained = fedsto.PRETRAINED_PATH if args.dry_run else fedsto.download_pretrained()
    if not args.dry_run:
        fedsto.check_runtime_dependencies()

    fedsto.GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    fedsto.CLIENT_STATE_DIR.mkdir(parents=True, exist_ok=True)
    args.stats_root.mkdir(parents=True, exist_ok=True)
    if args.log_file is None:
        args.log_file = args.workspace_root / "dqa_cwa_latest.log"
    if not args.dry_run and not args.append_train_log and args.log_file.exists():
        args.log_file.unlink()
    config_device = "" if args.gpus > 1 else args.device
    ensure_disk_space(args.workspace_root, args.min_free_gib)

    current_global = fedsto.GLOBAL_DIR / "round000_warmup.pt"
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
            fedsto.make_start_checkpoint(global_ckpt, current_global)

    if args.dry_run:
        print(f"Dry run complete. DQA-CWA workspace: {args.workspace_root}")
        return

    dqa_state = args.dqa_state or (args.workspace_root / "dqa_cwa_state.json")
    if args.force_restart:
        history = []
        print("Ignoring existing DQA-CWA history because --force-restart was set.")
    else:
        loaded_history = fedsto.load_history()
        history, current_global, next_round = fedsto.completed_history_prefix(
            loaded_history,
            phase1_rounds=args.phase1_rounds,
            phase2_rounds=args.phase2_rounds,
            warmup_checkpoint=current_global,
        )
        if loaded_history != history:
            fedsto.write_history(history)
        if next_round is None:
            total_rounds = args.phase1_rounds + args.phase2_rounds
            print(f"All requested DQA-CWA federated rounds are already complete ({total_rounds}/{total_rounds}).")
            print(f"Latest global checkpoint: {current_global}")
            return
        if history:
            phase, round_idx = next_round
            print(
                f"Resuming DQA-CWA after {len(history)} completed federated rounds "
                f"from phase {phase} round {round_idx}."
            )
            print(f"Current global checkpoint: {current_global}")
        else:
            print("No completed DQA-CWA federated rounds found; starting from phase 1 round 1.")

    rebuild_dqa_state_from_history(history, args, dqa_state)
    if not args.keep_intermediate_checkpoints:
        cleanup_completed_intermediates(setup, fedsto, history)

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
                history=history,
                keep_intermediates=args.keep_intermediate_checkpoints,
            )
            next_global = fedsto.GLOBAL_DIR / f"phase{phase}_round{round_idx:03d}_global.pt"
            reused_global = fedsto.reuse_checkpoint_if_valid(
                next_global,
                f"DQA-CWA global checkpoint for phase {phase} round {round_idx}",
                force_retrain=args.force_retrain,
            )
            if reused_global is not None:
                current_global = reused_global
                history.append({"phase": phase, "round": round_idx, "global": str(current_global.resolve())})
                fedsto.write_history(history)
                completed.add((phase, round_idx))
                print(f"Recovered DQA-CWA phase {phase} round {round_idx} from existing global checkpoint.")
                if not args.keep_intermediate_checkpoints:
                    cleanup_round_intermediates(setup, fedsto, phase, round_idx)
                continue

            local_paths = []
            for client in setup.CLIENTS:
                target = setup.LIST_ROOT / f"client_{client['id']}_{client['weather']}_target.txt"
                start = fedsto.CLIENT_STATE_DIR / f"client_{client['id']}_phase{phase}_round{round_idx:03d}_start.pt"
                previous = fedsto.CLIENT_STATE_DIR / f"client_{client['id']}_latest.pt"
                run_name = _dqa_run_name(phase, round_idx, client)
                client_stats = _client_stats_path(args.stats_root, phase, round_idx, client)
                stats_required = phase >= args.dqa_start_phase and not args.fallback_fedavg_without_stats
                ckpt = fedsto.reuse_checkpoint_if_valid(
                    fedsto.checkpoint_path(run_name),
                    f"DQA-CWA client run {run_name}",
                    force_retrain=args.force_retrain,
                )
                if ckpt is not None and stats_required and not client_stats.exists():
                    print(
                        f"Existing client run {run_name} checkpoint is usable but pseudo-label stats are missing "
                        f"({client_stats}); rerunning this client to produce DQA stats."
                    )
                    ckpt = None
                if ckpt is not None:
                    local_paths.append(ckpt)
                    fedsto.make_start_checkpoint(ckpt, previous)
                    continue

                if not fedsto.checkpoint_present(start):
                    fedsto.make_start_checkpoint(current_global, start, previous)
                cfg = fedsto.write_runtime_config(
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
                extra_env = None
                if phase >= args.dqa_start_phase:
                    extra_env = {
                        "DQA_PSEUDO_STATS_OUT": str(client_stats.resolve()),
                        "DQA_CLIENT_ID": str(client["id"]),
                        "DQA_PHASE": str(phase),
                        "DQA_ROUND": str(round_idx),
                    }
                    if client_stats.exists():
                        client_stats.unlink()
                ckpt = run_train(fedsto, cfg, args, extra_env=extra_env)
                local_paths.append(ckpt)
                fedsto.make_start_checkpoint(ckpt, previous)

            server_start = fedsto.GLOBAL_DIR / f"phase{phase}_round{round_idx:03d}_server_start.pt"
            server_name = _dqa_run_name(phase, round_idx)
            server_ckpt = fedsto.reuse_checkpoint_if_valid(
                fedsto.checkpoint_path(server_name),
                f"DQA-CWA server run {server_name}",
                force_retrain=args.force_retrain,
            )
            if server_ckpt is None:
                if not fedsto.checkpoint_present(server_start):
                    fedsto.make_start_checkpoint(current_global, server_start)
                server_cfg = fedsto.write_runtime_config(
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
                server_ckpt = run_train(fedsto, server_cfg, args)

            ensure_disk_space(
                args.workspace_root,
                args.min_free_gib,
                setup=setup,
                fedsto=fedsto,
                history=history,
                keep_intermediates=args.keep_intermediate_checkpoints,
            )
            if phase >= args.dqa_start_phase:
                stats_file = _stats_path(args.stats_root, phase, round_idx)
                if not stats_file.exists():
                    stats_file = build_round_stats_from_clients(args, phase, round_idx, setup) or stats_file
                if not stats_file.exists():
                    if not args.fallback_fedavg_without_stats:
                        raise FileNotFoundError(
                            f"Missing DQA-CWA stats file: {stats_file}. "
                            "Create it with collect_pseudo_stats.py or pass --fallback-fedavg-without-stats."
                        )
                    fedsto.aggregate_checkpoints(local_paths + [server_ckpt], server_ckpt, next_global, backbone_only=False)
                else:
                    stats = load_round_stats(stats_file, args.num_classes)
                    aggregate_checkpoints(
                        client_checkpoints=local_paths,
                        server_checkpoint=server_ckpt,
                        output_checkpoint=next_global,
                        stats=stats,
                        state_path=dqa_state,
                        config=_dqa_config(args),
                        repo_root=REPO_ROOT,
                    )
            else:
                fedsto.aggregate_checkpoints(local_paths + [server_ckpt], server_ckpt, next_global, backbone_only=(phase == 1))

            current_global = next_global
            history.append({"phase": phase, "round": round_idx, "global": str(current_global.resolve())})
            fedsto.write_history(history)
            completed.add((phase, round_idx))
            print(f"Completed DQA-CWA phase {phase} round {round_idx}: {current_global}")
            if not args.keep_intermediate_checkpoints:
                cleanup_round_intermediates(setup, fedsto, phase, round_idx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_DQA_WORK_ROOT)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--phase1-rounds", type=int, default=15)
    parser.add_argument("--phase2-rounds", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=29510)
    parser.add_argument("--device", default="")
    parser.add_argument("--force-warmup", action="store_true")
    parser.add_argument("--force-restart", action="store_true", help="Ignore DQA-CWA history.json and start federated rounds from phase 1 round 1.")
    parser.add_argument("--force-retrain", action="store_true", help="Rerun train jobs even when their last.pt checkpoints already exist.")
    parser.add_argument(
        "--keep-intermediate-checkpoints",
        action="store_true",
        help="Keep per-run last/best weights and per-round start checkpoints after a global round is complete.",
    )
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=80.0,
        help="Abort before starting another train/aggregate step if free disk space under the workspace is below this value.",
    )
    parser.add_argument("--log-file", type=Path, default=None, help="Append EfficientTeacher train output here instead of flooding notebook stdout.")
    parser.add_argument("--append-train-log", action="store_true", help="Append to --log-file instead of replacing it at run start.")
    parser.add_argument("--stream-train-output", action="store_true", help="Stream full EfficientTeacher output to stdout instead of compact logging.")
    parser.add_argument("--stats-root", type=Path, default=RESEARCH_ROOT / "stats")
    parser.add_argument("--dqa-state", type=Path, default=None)
    parser.add_argument("--dqa-start-phase", type=int, default=2)
    parser.add_argument("--fallback-fedavg-without-stats", action="store_true")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--count-ema", type=float, default=0.70)
    parser.add_argument("--quality-ema", type=float, default=0.70)
    parser.add_argument("--alpha-ema", type=float, default=0.50)
    parser.add_argument("--temperature", type=float, default=1.50)
    parser.add_argument("--uniform-mix", type=float, default=0.05)
    parser.add_argument("--classwise-blend", type=float, default=0.75)
    parser.add_argument("--stability-lambda", type=float, default=0.25)
    parser.add_argument("--min-effective-count", type=float, default=1.0)
    parser.add_argument("--server-anchor", type=float, default=0.50)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.setup_only:
        _prepare_fedsto_modules(parsed.workspace_root)[0].build_base_configs()
    else:
        run_protocol(parsed)
