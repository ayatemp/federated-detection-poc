#!/usr/bin/env python3
"""FedMox-inspired post-hoc five-loop sweep for scene/day-night DQA.

This is intentionally a fast checkpoint-level experiment.  It uses the already
trained 02 head-to-full DQA checkpoints and tests the pieces FedMox suggests
should matter before paying for another full multi-hour run:

* Soft-Mixture: preserve the previous server model and inject only part of the
  client aggregate.
* Head/class-only transfer: keep the detector body anchored to the server model
  and move only class-related rows.
* Hard-split probes: test whether day/night or night-only client signals are
  useful when they are not averaged away.

The output is a minimum of five new candidate checkpoints plus a shared
scene/day-night evaluation table against warmup and the normal 02 result.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


MOE_ROOT = Path(__file__).resolve().parents[1]
SCENE_ROOT = MOE_ROOT.parent
REPO_ROOT = SCENE_ROOT.parents[1]
SCENE_SCRIPTS = SCENE_ROOT / "scripts"
DQA_ROOT = SCENE_ROOT.parent
PROTOCOL_VERSION = "scene_daynight_dqa_moe_02_fedmox_posthoc_five_loop_v1"

for path in (SCENE_SCRIPTS, DQA_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dqa_cwa_aggregation as dqa_v1  # noqa: E402
import dqa_cwa_aggregation_v2 as dqa_v2  # noqa: E402
import run_scene_daynight_dqa_02_head_to_full as htf  # noqa: E402


SOURCE_WORKSPACE = SCENE_ROOT / "output" / "02_head_to_full_long_dqa"
DEFAULT_WORKSPACE = MOE_ROOT / "output" / "02_fedmox_posthoc_five_loop"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: Any) -> float | None:
    return htf.as_float(value)


def source_records(source_workspace: Path) -> dict[str, dict[str, str]]:
    path = source_workspace / "stats" / "02_head_to_full_checkpoints.csv"
    rows = read_csv(path)
    if not rows:
        raise FileNotFoundError(f"source checkpoint table is missing or empty: {path}")
    return {row["label"]: row for row in rows}


def require_record(records: Mapping[str, dict[str, str]], label: str) -> Path:
    row = records.get(label)
    if row is None:
        raise KeyError(f"missing source checkpoint label: {label}")
    path = Path(row["path"])
    if not path.exists():
        raise FileNotFoundError(f"checkpoint for {label} does not exist: {path}")
    return path


def _load(path: Path) -> dict[str, Any]:
    return dqa_v1._load_checkpoint(path, REPO_ROOT)


def _state_dict(ckpt: Mapping[str, Any], key: str) -> dict[str, torch.Tensor] | None:
    if ckpt.get(key) is None:
        return None
    return dqa_v1._model_state_dict(ckpt, key)


def _replace(base: dict[str, Any], state: Mapping[str, torch.Tensor], key: str) -> None:
    dqa_v1._replace_model_state(base, dict(state), key)


def softmix_state(
    server_state: Mapping[str, torch.Tensor],
    update_state: Mapping[str, torch.Tensor],
    *,
    server_alpha: float,
    localize_bn: bool = True,
) -> dict[str, torch.Tensor]:
    mixed: dict[str, torch.Tensor] = {}
    for key, server_value in server_state.items():
        update_value = update_state[key]
        if localize_bn and dqa_v1._is_batchnorm_key(key):
            mixed[key] = server_value
        elif torch.is_tensor(server_value) and server_value.dtype.is_floating_point:
            mixed[key] = (
                server_alpha * server_value.float()
                + (1.0 - server_alpha) * update_value.float()
            ).to(server_value.dtype)
        else:
            mixed[key] = server_value
    return mixed


def save_checkpoint(base: dict[str, Any], output: Path) -> Path:
    base["epoch"] = -1
    base["optimizer"] = None
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, output)
    return output


def softmix_checkpoint(server: Path, update: Path, output: Path, *, server_alpha: float) -> Path:
    server_ckpt = _load(server)
    update_ckpt = _load(update)
    base = copy.deepcopy(server_ckpt)

    model = softmix_state(
        dqa_v1._model_state_dict(server_ckpt, "model"),
        dqa_v1._model_state_dict(update_ckpt, "model"),
        server_alpha=server_alpha,
    )
    _replace(base, model, "model")

    server_ema = _state_dict(server_ckpt, "ema")
    update_ema = _state_dict(update_ckpt, "ema")
    if server_ema is not None and update_ema is not None:
        _replace(base, softmix_state(server_ema, update_ema, server_alpha=server_alpha), "ema")
    return save_checkpoint(base, output)


def class_only_dqa_checkpoint(
    *,
    server: Path,
    clients: Sequence[Path],
    stats: Sequence[dqa_v1.ClientClassStats],
    output: Path,
    config: dqa_v2.AggregationConfig,
    state_path: Path,
) -> Path:
    if len(clients) != len(stats):
        raise ValueError(f"clients/stat length mismatch: {len(clients)} vs {len(stats)}")
    state, alpha, _source_ids, active = dqa_v2.compute_reliability(stats, {}, config)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    client_ckpts = [_load(path) for path in clients]
    server_ckpt = _load(server)
    base = copy.deepcopy(server_ckpt)

    client_model = [dqa_v1._model_state_dict(ckpt, "model") for ckpt in client_ckpts]
    server_model = dqa_v1._model_state_dict(server_ckpt, "model")
    anchored_model = {key: value for key, value in server_model.items()}
    dynamic = dqa_v2.apply_dynamic_classwise_head(
        anchored_model,
        client_model,
        server_model,
        alpha,
        active,
        config,
    )
    _replace(base, dynamic, "model")

    server_ema = _state_dict(server_ckpt, "ema")
    client_emas = [_state_dict(ckpt, "ema") for ckpt in client_ckpts]
    if server_ema is not None and all(item is not None for item in client_emas):
        anchored_ema = {key: value for key, value in server_ema.items()}
        dynamic_ema = dqa_v2.apply_dynamic_classwise_head(
            anchored_ema,
            [item for item in client_emas if item is not None],
            server_ema,
            alpha,
            active,
            config,
        )
        _replace(base, dynamic_ema, "ema")
    return save_checkpoint(base, output)


def client_labels(prefix: str, round_idx: int) -> list[str]:
    tag = htf.round_tag(round_idx)
    return [
        f"{prefix}_{tag}_client0_highway_day",
        f"{prefix}_{tag}_client1_highway_night",
        f"{prefix}_{tag}_client2_citystreet_day",
        f"{prefix}_{tag}_client3_citystreet_night",
        f"{prefix}_{tag}_client4_residential_day",
        f"{prefix}_{tag}_client5_residential_night",
    ]


def subset_stats(
    stats: Sequence[dqa_v1.ClientClassStats],
    client_ids: Sequence[str],
) -> list[dqa_v1.ClientClassStats]:
    by_id = {item.client_id: item for item in stats}
    return [by_id[client_id] for client_id in client_ids]


def dqa_config(args: argparse.Namespace, num_classes: int, *, classwise_blend: float) -> dqa_v2.AggregationConfig:
    return dqa_v2.AggregationConfig(
        num_classes=num_classes,
        count_ema=args.dqa_count_ema,
        quality_ema=args.dqa_quality_ema,
        alpha_ema=args.dqa_alpha_ema,
        temperature=args.dqa_temperature,
        uniform_mix=args.dqa_uniform_mix,
        classwise_blend=classwise_blend,
        stability_lambda=args.dqa_stability_lambda,
        min_effective_count=args.dqa_min_effective_count,
        min_quality=args.dqa_min_quality,
        max_quality=1.0,
        server_anchor=args.dqa_server_anchor,
        localize_bn=True,
        min_server_alpha=args.dqa_min_server_alpha,
        residual_blend=0.0,
    )


def save_record(
    rows: list[dict[str, str]],
    label: str,
    path: Path,
    *,
    kind: str,
    note: str,
) -> None:
    rows.append(
        {
            "label": label,
            "kind": kind,
            "phase": "posthoc",
            "phase_round": "",
            "global_round": "",
            "client": "",
            "variant": note,
            "path": str(path.resolve()),
        }
    )


def generate_candidates(args: argparse.Namespace, setup) -> list[dict[str, str]]:  # noqa: ANN001
    records = source_records(args.source_workspace)
    out = args.workspace_root / "checkpoints"
    stats_dir = args.workspace_root / "stats"

    phase1_server_prev = require_record(records, "phase1_head_round029_server_repair")
    phase1_aggregate = require_record(records, "phase1_head_round030_dqa_aggregate")
    phase2_server_prev = require_record(records, "phase2_full_round031_server_repair")
    phase2_aggregate = require_record(records, "phase2_full_round032_dqa_aggregate")

    phase1_client_ids = [
        "client0_highway_day",
        "client1_highway_night",
        "client2_citystreet_day",
        "client3_citystreet_night",
        "client4_residential_day",
        "client5_residential_night",
    ]
    night_ids = [item for item in phase1_client_ids if item.endswith("_night")]
    day_ids = [item for item in phase1_client_ids if item.endswith("_day")]
    source_client_paths = {
        client_id: require_record(records, label)
        for client_id, label in zip(phase1_client_ids, client_labels("phase1_head", 30))
    }

    pseudo_json = args.source_workspace / "stats" / "03_round030_pseudo_label_stats.json"
    pseudo_stats = json.loads(pseudo_json.read_text(encoding="utf-8"))
    all_stats = htf.dqa01.pseudo_stats_to_dqa_stats(pseudo_stats, num_classes=len(setup.BDD_NAMES))

    rows: list[dict[str, str]] = []
    specs = [
        ("loop1_phase1_softmix_a70", "softmix", 0.70, 0.0, phase1_server_prev, phase1_aggregate, phase1_client_ids),
        ("loop2_phase1_softmix_a85", "softmix", 0.85, 0.0, phase1_server_prev, phase1_aggregate, phase1_client_ids),
        ("loop3_phase1_class_only_b25", "class_only", 0.0, 0.25, phase1_server_prev, phase1_aggregate, phase1_client_ids),
        ("loop4_phase1_class_only_b55", "class_only", 0.0, 0.55, phase1_server_prev, phase1_aggregate, phase1_client_ids),
        ("loop5_phase1_night_only_b55", "class_only", 0.0, 0.55, phase1_server_prev, phase1_aggregate, night_ids),
        ("loop6_phase1_day_only_b55", "class_only", 0.0, 0.55, phase1_server_prev, phase1_aggregate, day_ids),
        ("loop7_phase2_softmix_a90", "softmix", 0.90, 0.0, phase2_server_prev, phase2_aggregate, phase1_client_ids),
    ]

    for label, method, alpha, blend, server, aggregate, selected_ids in specs:
        output = out / f"{label}.pt"
        if method == "softmix":
            softmix_checkpoint(server, aggregate, output, server_alpha=alpha)
            note = f"FedMox Soft-Mixture server_alpha={alpha:.2f}"
        else:
            selected_paths = [source_client_paths[client_id] for client_id in selected_ids]
            selected_stats = subset_stats(all_stats, selected_ids)
            config = dqa_config(args, len(setup.BDD_NAMES), classwise_blend=blend)
            class_only_dqa_checkpoint(
                server=server,
                clients=selected_paths,
                stats=selected_stats,
                output=output,
                config=config,
                state_path=stats_dir / f"{label}_dqa_state.json",
            )
            note = f"class-only DQA blend={blend:.2f}; clients={','.join(selected_ids)}"
        save_record(rows, label, output, kind="aggregate", note=note)

    return rows


def baseline_records(args: argparse.Namespace) -> list[dict[str, str]]:
    records = source_records(args.source_workspace)
    wanted = [
        ("warmup_global", "warmup_global", "warmup", "02 warmup"),
        ("phase1_head_round030_dqa_aggregate", "normal02_phase1_aggregate", "aggregate", "normal 02 phase1 aggregate"),
        ("phase1_head_round030_server_repair", "normal02_phase1_repair", "server_repair", "normal 02 phase1 repair"),
        ("phase2_full_round032_dqa_aggregate", "normal02_phase2_aggregate", "aggregate", "normal 02 phase2 aggregate"),
        ("phase2_full_round032_server_repair", "normal02_phase2_repair", "server_repair", "normal 02 phase2 repair"),
    ]
    rows: list[dict[str, str]] = []
    for source_label, eval_label, kind, note in wanted:
        save_record(rows, eval_label, require_record(records, source_label), kind=kind, note=note)
    return rows


def split_metrics(by_label_split: Mapping[tuple[str, str], dict[str, str]], label: str) -> dict[str, Any]:
    return htf.split_gap_metrics(dict(by_label_split), label)


def write_metrics(args: argparse.Namespace, eval_records: list[dict[str, str]]) -> list[dict[str, Any]]:
    summary_path = args.workspace_root / "validation_reports" / "paper_protocol_eval_summary.csv"
    rows = [row for row in read_csv(summary_path) if row.get("status") == "ok"]
    by_label_split = {(row["checkpoint_label"], row["split"]): row for row in rows}
    total_rows = [row for row in rows if row.get("split") in {"scene_daynight_total", "total"}]
    by_label_total = {row["checkpoint_label"]: row for row in total_rows}
    warm = by_label_total.get("warmup_global")
    normal = by_label_total.get("normal02_phase2_repair")
    warm_m95 = as_float(warm.get("map50_95")) if warm else None
    normal_m95 = as_float(normal.get("map50_95")) if normal else None
    meta = {row["label"]: row for row in eval_records}

    metric_rows: list[dict[str, Any]] = []
    for label in [row["label"] for row in eval_records]:
        total = by_label_total.get(label)
        if not total:
            continue
        m50 = as_float(total.get("map50"))
        m95 = as_float(total.get("map50_95"))
        gap = split_metrics(by_label_split, label)
        metric_rows.append(
            {
                "checkpoint_label": label,
                "kind": meta[label]["kind"],
                "variant": meta[label]["variant"],
                "precision": total.get("precision", ""),
                "recall": total.get("recall", ""),
                "map50": "" if m50 is None else f"{m50:.6f}",
                "map50_95": "" if m95 is None else f"{m95:.6f}",
                "gain_vs_warmup_map50_95": "" if m95 is None or warm_m95 is None else f"{m95 - warm_m95:.6f}",
                "gain_vs_normal02_phase2_repair_map50_95": ""
                if m95 is None or normal_m95 is None
                else f"{m95 - normal_m95:.6f}",
                **gap,
            }
        )

    fields = [
        "checkpoint_label",
        "kind",
        "variant",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "gain_vs_warmup_map50_95",
        "gain_vs_normal02_phase2_repair_map50_95",
        "worst_split",
        "worst_split_map50_95",
        "day_avg_map50_95",
        "night_avg_map50_95",
        "day_night_gap_map50_95",
    ]
    write_csv(args.workspace_root / "stats" / "02_fedmox_posthoc_metrics.csv", metric_rows, fields)

    split_rows: list[dict[str, Any]] = []
    for row in rows:
        label = row["checkpoint_label"]
        if label not in meta:
            continue
        split_rows.append(
            {
                "checkpoint_label": label,
                "kind": meta[label]["kind"],
                "variant": meta[label]["variant"],
                "split": row["split"],
                "images": row.get("images", ""),
                "labels": row.get("labels", ""),
                "precision": row.get("precision", ""),
                "recall": row.get("recall", ""),
                "map50": row.get("map50", ""),
                "map50_95": row.get("map50_95", ""),
            }
        )
    write_csv(
        args.workspace_root / "stats" / "02_fedmox_posthoc_split_metrics.csv",
        split_rows,
        ["checkpoint_label", "kind", "variant", "split", "images", "labels", "precision", "recall", "map50", "map50_95"],
    )

    ranked = sorted(
        [row for row in metric_rows if row.get("map50_95") not in {"", None}],
        key=lambda row: float(row["map50_95"]),
        reverse=True,
    )
    (args.workspace_root / "stats" / "02_fedmox_posthoc_summary.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "protocol": PROTOCOL_VERSION,
                "source_workspace": str(args.source_workspace.resolve()),
                "workspace": str(args.workspace_root.resolve()),
                "best": ranked[:5],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return metric_rows


def notify(args: argparse.Namespace, message: str, *, title: str, status: str | None = None, error: str | None = None) -> None:
    try:
        from notebook_notify import notify_discord

        metrics_path = args.workspace_root / "stats" / "02_fedmox_posthoc_metrics.csv"
        context: dict[str, Any] = {
            "workspace": str(args.workspace_root.resolve()),
            "source_workspace": str(args.source_workspace.resolve()),
            "status": status or "",
        }
        if error:
            context["error"] = error[:500]
        if metrics_path.exists():
            context["metrics_csv"] = str(metrics_path)
            rows = read_csv(metrics_path)
            ranked = sorted(
                [row for row in rows if row.get("map50_95")],
                key=lambda row: float(row["map50_95"]),
                reverse=True,
            )
            if ranked:
                context["best_map50_95"] = ranked[0]["checkpoint_label"]
                context["best_map50_95_value"] = ranked[0]["map50_95"]
        print(notify_discord(message, title=title, context=context, fail_silently=True))
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-workspace", type=Path, default=SOURCE_WORKSPACE)
    parser.add_argument("--warmup-checkpoint", type=Path, default=REPO_ROOT / "pseudogt_learnability" / "checkpoints" / "round000_warmup.pt")
    parser.add_argument("--client-limit", type=int, default=1500)
    parser.add_argument("--clients", default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-clients", action="store_true")
    parser.add_argument(
        "--eval-splits",
        default="highway_day,highway_night,citystreet_day,citystreet_night,residential_day,residential_night,total",
    )
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--device", default="")
    parser.add_argument("--classwise", action="store_true")
    parser.add_argument("--no-eval-plots", action="store_true")
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--notify-start", action="store_true")
    parser.add_argument("--notify-end", action="store_true")
    parser.add_argument("--dqa-count-ema", type=float, default=0.65)
    parser.add_argument("--dqa-quality-ema", type=float, default=0.65)
    parser.add_argument("--dqa-alpha-ema", type=float, default=0.40)
    parser.add_argument("--dqa-temperature", type=float, default=2.50)
    parser.add_argument("--dqa-uniform-mix", type=float, default=0.05)
    parser.add_argument("--dqa-min-server-alpha", type=float, default=0.70)
    parser.add_argument("--dqa-server-anchor", type=float, default=10.0)
    parser.add_argument("--dqa-stability-lambda", type=float, default=0.70)
    parser.add_argument("--dqa-min-effective-count", type=float, default=5.0)
    parser.add_argument("--dqa-min-quality", type=float, default=0.10)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.source_workspace = args.source_workspace.expanduser().resolve()
    htf.pl03.ensure_dirs(args.workspace_root)
    setup, _fedsto, _manifest, _clients, _warmup = htf.prepare(args)
    if args.setup_only:
        print("Setup complete.")
        return []

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "paper_basis": {
            "fedmox_arxiv": "https://arxiv.org/abs/2508.16568",
            "implemented_here": "post-hoc Soft-Mixture and class-only DQA probes over existing 02 checkpoints",
            "not_implemented_here": "true in-model sparse spatial router / YOLO architectural MoE",
        },
        "source_workspace": str(args.source_workspace.resolve()),
        "workspace": str(args.workspace_root.resolve()),
        "dqa_config_base": asdict(dqa_config(args, len(setup.BDD_NAMES), classwise_blend=0.25)),
    }
    (args.workspace_root / "stats" / "02_fedmox_posthoc_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    eval_records = baseline_records(args) + generate_candidates(args, setup)
    write_csv(
        args.workspace_root / "stats" / "02_fedmox_posthoc_checkpoints.csv",
        eval_records,
        ["label", "kind", "phase", "phase_round", "global_round", "client", "variant", "path"],
    )

    if args.evaluate:
        htf.base01_0.run_evaluation(args, eval_records)
        return write_metrics(args, eval_records)
    return []


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    do_start_notify = args.notify or args.notify_start
    do_end_notify = args.notify or args.notify_end
    if do_start_notify:
        notify(args, "FedMox-inspired post-hoc five-loop sweep started.", title="DQA MoE 02 start")
    status = "success"
    error: str | None = None
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = str(exc)
        raise
    finally:
        if do_end_notify:
            notify(
                args,
                f"FedMox-inspired post-hoc five-loop sweep finished with status={status}.",
                title="DQA MoE 02 finish",
                status=status,
                error=error,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
