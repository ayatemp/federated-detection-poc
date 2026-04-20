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
import sys
from pathlib import Path

from dqa_cwa_aggregation import AggregationConfig, aggregate_checkpoints, load_round_stats


RESEARCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RESEARCH_ROOT.parent
NAV_ROOT = REPO_ROOT / "navigating_data_heterogeneity"
DQA_WORK_ROOT = RESEARCH_ROOT / "efficientteacher_dqa_cwa"


def _prepare_fedsto_modules():
    if str(NAV_ROOT) not in sys.path:
        sys.path.insert(0, str(NAV_ROOT))

    setup = importlib.import_module("setup_fedsto_exact_reproduction")
    setup.WORK_ROOT = DQA_WORK_ROOT
    setup.LIST_ROOT = DQA_WORK_ROOT / "data_lists"
    setup.CONFIG_ROOT = DQA_WORK_ROOT / "configs"
    setup.RUN_ROOT = DQA_WORK_ROOT / "runs"

    fedsto = importlib.import_module("run_fedsto_efficientteacher_exact")
    fedsto.PRETRAINED_PATH = DQA_WORK_ROOT / "weights" / "efficient-yolov5l.pt"
    fedsto.GLOBAL_DIR = DQA_WORK_ROOT / "global_checkpoints"
    fedsto.CLIENT_STATE_DIR = DQA_WORK_ROOT / "client_states"
    return setup, fedsto


def _stats_path(root: Path, phase: int, round_idx: int) -> Path:
    return root / f"phase{phase}_round{round_idx:03d}.json"


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


def run_protocol(args: argparse.Namespace) -> None:
    setup, fedsto = _prepare_fedsto_modules()
    setup.build_base_configs()
    pretrained = fedsto.PRETRAINED_PATH if args.dry_run else fedsto.download_pretrained()
    if not args.dry_run:
        fedsto.check_runtime_dependencies()

    fedsto.GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    fedsto.CLIENT_STATE_DIR.mkdir(parents=True, exist_ok=True)
    args.stats_root.mkdir(parents=True, exist_ok=True)
    config_device = "" if args.gpus > 1 else args.device

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
        global_ckpt = fedsto.run_train(warmup_cfg, args.dry_run, gpus=args.gpus, master_port=args.master_port)
        if not args.dry_run:
            fedsto.make_start_checkpoint(global_ckpt, current_global)

    if args.dry_run:
        print(f"Dry run complete. DQA-CWA workspace: {DQA_WORK_ROOT}")
        return

    history = []
    dqa_state = args.dqa_state or (DQA_WORK_ROOT / "dqa_cwa_state.json")
    for phase, rounds in [(1, args.phase1_rounds), (2, args.phase2_rounds)]:
        for round_idx in range(1, rounds + 1):
            local_paths = []
            for client in setup.CLIENTS:
                target = setup.LIST_ROOT / f"client_{client['id']}_{client['weather']}_target.txt"
                start = fedsto.CLIENT_STATE_DIR / f"client_{client['id']}_phase{phase}_round{round_idx:03d}_start.pt"
                previous = fedsto.CLIENT_STATE_DIR / f"client_{client['id']}_latest.pt"
                fedsto.make_start_checkpoint(current_global, start, previous)
                run_name = f"dqa_phase{phase}_round{round_idx:03d}_client{client['id']}_{client['weather']}"
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
                ckpt = fedsto.run_train(cfg, args.dry_run, gpus=args.gpus, master_port=args.master_port)
                local_paths.append(ckpt)
                fedsto.make_start_checkpoint(ckpt, previous)

            server_start = fedsto.GLOBAL_DIR / f"phase{phase}_round{round_idx:03d}_server_start.pt"
            fedsto.make_start_checkpoint(current_global, server_start)
            server_name = f"dqa_phase{phase}_round{round_idx:03d}_server"
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
            server_ckpt = fedsto.run_train(server_cfg, args.dry_run, gpus=args.gpus, master_port=args.master_port)

            next_global = fedsto.GLOBAL_DIR / f"phase{phase}_round{round_idx:03d}_global.pt"
            if phase >= args.dqa_start_phase:
                stats_file = _stats_path(args.stats_root, phase, round_idx)
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
            (DQA_WORK_ROOT / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(f"Completed DQA-CWA phase {phase} round {round_idx}: {current_global}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--warmup-epochs", type=int, default=50)
    parser.add_argument("--phase1-rounds", type=int, default=100)
    parser.add_argument("--phase2-rounds", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=29510)
    parser.add_argument("--device", default="")
    parser.add_argument("--force-warmup", action="store_true")
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
        _prepare_fedsto_modules()[0].build_base_configs()
    else:
        run_protocol(parsed)
