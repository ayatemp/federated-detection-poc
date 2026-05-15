#!/usr/bin/env python3
"""Run a five-loop MoE x DQA research sprint.

Each loop is represented explicitly:

1. FedMox post-hoc Soft-Mixture/class-only sweep (reads existing Loop 1 result).
2. Repair-residual reinjection: test whether server repair is erasing target signal.
3. FedBN-style BN transplant: test whether feature-shift BN statistics explain day/night collapse.
4. Client expert oracle probe: test whether a real router has useful experts to choose from.
5. DQA re-aggregation sweep: test whether aggregation hyperparameters, not training, are the blocker.

Loops 2-5 generate new checkpoint candidates from the already completed 02
head-to-full run, evaluate them, and write a combined report.
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
PROTOCOL_VERSION = "scene_daynight_dqa_moe_03_five_research_loops_v1"

for path in (SCENE_SCRIPTS, DQA_ROOT, MOE_ROOT / "scripts", REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dqa_cwa_aggregation as dqa_v1  # noqa: E402
import dqa_cwa_aggregation_v2 as dqa_v2  # noqa: E402
import run_moe_02_fedmox_posthoc_five_loop as loop1  # noqa: E402
import run_scene_daynight_dqa_02_head_to_full as htf  # noqa: E402


SOURCE_WORKSPACE = SCENE_ROOT / "output" / "02_head_to_full_long_dqa"
LOOP1_WORKSPACE = MOE_ROOT / "output" / "02_fedmox_posthoc_five_loop"
DEFAULT_WORKSPACE = MOE_ROOT / "output" / "03_five_research_loops"


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


def _load(path: Path) -> dict[str, Any]:
    return dqa_v1._load_checkpoint(path, REPO_ROOT)


def _state_dict(ckpt: Mapping[str, Any], key: str) -> dict[str, torch.Tensor] | None:
    if ckpt.get(key) is None:
        return None
    return dqa_v1._model_state_dict(ckpt, key)


def _replace(base: dict[str, Any], state: Mapping[str, torch.Tensor], key: str) -> None:
    dqa_v1._replace_model_state(base, dict(state), key)


def save_checkpoint(base: dict[str, Any], output: Path) -> Path:
    base["epoch"] = -1
    base["optimizer"] = None
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, output)
    return output


def source_records(source_workspace: Path) -> dict[str, dict[str, str]]:
    return loop1.source_records(source_workspace)


def require_record(records: Mapping[str, dict[str, str]], label: str) -> Path:
    return loop1.require_record(records, label)


def save_record(
    rows: list[dict[str, str]],
    label: str,
    path: Path,
    *,
    kind: str,
    loop_id: str,
    hypothesis: str,
    implementation: str,
) -> None:
    rows.append(
        {
            "label": label,
            "kind": kind,
            "phase": "research_loop",
            "phase_round": "",
            "global_round": "",
            "client": "",
            "variant": implementation,
            "loop_id": loop_id,
            "hypothesis": hypothesis,
            "path": str(path.resolve()),
        }
    )


def add_residual_state(
    base_state: Mapping[str, torch.Tensor],
    signal_state: Mapping[str, torch.Tensor],
    anchor_state: Mapping[str, torch.Tensor],
    *,
    beta: float,
    localize_bn: bool = True,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, base_value in base_state.items():
        if localize_bn and dqa_v1._is_batchnorm_key(key):
            result[key] = base_value
        elif torch.is_tensor(base_value) and base_value.dtype.is_floating_point:
            residual = signal_state[key].float() - anchor_state[key].float()
            result[key] = (base_value.float() + beta * residual).to(base_value.dtype)
        else:
            result[key] = base_value
    return result


def residual_reinject_checkpoint(base: Path, signal: Path, anchor: Path, output: Path, *, beta: float) -> Path:
    base_ckpt = _load(base)
    signal_ckpt = _load(signal)
    anchor_ckpt = _load(anchor)
    out = copy.deepcopy(base_ckpt)
    model = add_residual_state(
        dqa_v1._model_state_dict(base_ckpt, "model"),
        dqa_v1._model_state_dict(signal_ckpt, "model"),
        dqa_v1._model_state_dict(anchor_ckpt, "model"),
        beta=beta,
    )
    _replace(out, model, "model")
    base_ema = _state_dict(base_ckpt, "ema")
    signal_ema = _state_dict(signal_ckpt, "ema")
    anchor_ema = _state_dict(anchor_ckpt, "ema")
    if base_ema is not None and signal_ema is not None and anchor_ema is not None:
        _replace(out, add_residual_state(base_ema, signal_ema, anchor_ema, beta=beta), "ema")
    return save_checkpoint(out, output)


def transplant_bn_state(
    base_state: Mapping[str, torch.Tensor],
    bn_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, base_value in base_state.items():
        if dqa_v1._is_batchnorm_key(key):
            result[key] = bn_state[key]
        else:
            result[key] = base_value
    return result


def transplant_bn_checkpoint(base: Path, bn_source: Path, output: Path) -> Path:
    base_ckpt = _load(base)
    bn_ckpt = _load(bn_source)
    out = copy.deepcopy(base_ckpt)
    _replace(
        out,
        transplant_bn_state(dqa_v1._model_state_dict(base_ckpt, "model"), dqa_v1._model_state_dict(bn_ckpt, "model")),
        "model",
    )
    base_ema = _state_dict(base_ckpt, "ema")
    bn_ema = _state_dict(bn_ckpt, "ema")
    if base_ema is not None and bn_ema is not None:
        _replace(out, transplant_bn_state(base_ema, bn_ema), "ema")
    return save_checkpoint(out, output)


def client_ids() -> list[str]:
    return [
        "client0_highway_day",
        "client1_highway_night",
        "client2_citystreet_day",
        "client3_citystreet_night",
        "client4_residential_day",
        "client5_residential_night",
    ]


def phase1_client_labels() -> list[str]:
    return loop1.client_labels("phase1_head", 30)


def phase1_stats(source_workspace: Path, setup) -> list[dqa_v1.ClientClassStats]:  # noqa: ANN001
    pseudo_json = source_workspace / "stats" / "03_round030_pseudo_label_stats.json"
    pseudo_stats = json.loads(pseudo_json.read_text(encoding="utf-8"))
    return htf.dqa01.pseudo_stats_to_dqa_stats(pseudo_stats, num_classes=len(setup.BDD_NAMES))


def dqa_reaggregate(
    *,
    source_workspace: Path,
    records: Mapping[str, dict[str, str]],
    setup,
    output: Path,
    state_path: Path,
    temperature: float,
    min_server_alpha: float,
    classwise_blend: float,
    residual_blend: float,
    server_anchor: float,
) -> Path:
    clients = [require_record(records, label) for label in phase1_client_labels()]
    stats = phase1_stats(source_workspace, setup)
    server = require_record(records, "phase1_head_round029_server_repair")
    config = dqa_v2.AggregationConfig(
        num_classes=len(setup.BDD_NAMES),
        count_ema=0.65,
        quality_ema=0.65,
        alpha_ema=0.40,
        temperature=temperature,
        uniform_mix=0.05,
        classwise_blend=classwise_blend,
        stability_lambda=0.70,
        min_effective_count=5.0,
        min_quality=0.10,
        max_quality=1.0,
        server_anchor=server_anchor,
        localize_bn=True,
        min_server_alpha=min_server_alpha,
        residual_blend=residual_blend,
    )
    dqa_v2.aggregate_checkpoints(
        client_checkpoints=clients,
        server_checkpoint=server,
        output_checkpoint=output,
        stats=stats,
        state_path=state_path,
        config=config,
        repo_root=REPO_ROOT,
    )
    return output


def generate_loop_candidates(args: argparse.Namespace, setup) -> list[dict[str, str]]:  # noqa: ANN001
    records = source_records(args.source_workspace)
    ckpt_dir = args.workspace_root / "checkpoints"
    stats_dir = args.workspace_root / "stats"
    rows: list[dict[str, str]] = []

    phase1_prev = require_record(records, "phase1_head_round029_server_repair")
    phase1_agg = require_record(records, "phase1_head_round030_dqa_aggregate")
    phase1_repair = require_record(records, "phase1_head_round030_server_repair")
    phase2_prev = require_record(records, "phase2_full_round031_server_repair")
    phase2_agg = require_record(records, "phase2_full_round032_dqa_aggregate")
    phase2_repair = require_record(records, "phase2_full_round032_server_repair")
    warmup = require_record(records, "warmup_global")

    # Loop 2: target residual after repair.
    for beta in (0.20, 0.40, 0.60):
        label = f"loop2_repair_residual_phase1_b{int(beta * 100):02d}"
        path = residual_reinject_checkpoint(
            phase1_repair,
            phase1_agg,
            phase1_prev,
            ckpt_dir / f"{label}.pt",
            beta=beta,
        )
        save_record(
            rows,
            label,
            path,
            kind="aggregate",
            loop_id="loop2_repair_residual",
            hypothesis="server repair erases useful Phase1 target residuals",
            implementation=f"phase1_repair + {beta:.2f} * (phase1_aggregate - phase1_previous_server)",
        )
    label = "loop2_repair_residual_phase2_b40"
    path = residual_reinject_checkpoint(phase2_repair, phase2_agg, phase2_prev, ckpt_dir / f"{label}.pt", beta=0.40)
    save_record(
        rows,
        label,
        path,
        kind="aggregate",
        loop_id="loop2_repair_residual",
        hypothesis="server repair erases useful Phase2 target residuals",
        implementation="phase2_repair + 0.40 * (phase2_aggregate - phase2_previous_server)",
    )

    # Loop 3: feature-shift BN transplant inspired by FedBN.
    bn_specs = [
        ("loop3_phase1agg_bn_warmup", phase1_agg, warmup),
        ("loop3_phase1agg_bn_prevserver", phase1_agg, phase1_prev),
        ("loop3_phase1agg_bn_repair", phase1_agg, phase1_repair),
        ("loop3_phase2agg_bn_phase1repair", phase2_agg, phase1_repair),
    ]
    for label, base, bn_source in bn_specs:
        path = transplant_bn_checkpoint(base, bn_source, ckpt_dir / f"{label}.pt")
        save_record(
            rows,
            label,
            path,
            kind="aggregate",
            loop_id="loop3_fedbn_bn_transplant",
            hypothesis="day/night feature shift is partly BN-stat mismatch after aggregation",
            implementation=f"{base.name} with BN statistics from {bn_source.name}",
        )

    # Loop 4: existing client checkpoints as experts/oracle candidates.
    for client_id, source_label in zip(client_ids(), phase1_client_labels()):
        save_record(
            rows,
            f"loop4_expert_{client_id}",
            require_record(records, source_label),
            kind="aggregate",
            loop_id="loop4_client_expert_oracle",
            hypothesis="a real router can outperform one global model if client experts specialize by split",
            implementation=f"evaluate phase1 client expert checkpoint {client_id}",
        )

    # Loop 5: DQA aggregation hyperparameter alternatives.
    loop5_specs = [
        ("loop5_dqa_cool_strong", 1.2, 0.60, 0.35, 0.08, 6.0),
        ("loop5_dqa_warm_conservative", 4.0, 0.82, 0.18, 0.02, 20.0),
        ("loop5_dqa_class_only_mid", 2.5, 0.76, 0.35, 0.00, 14.0),
        ("loop5_dqa_more_client_residual", 2.0, 0.65, 0.28, 0.18, 8.0),
    ]
    for label, temp, min_server, blend, residual, anchor in loop5_specs:
        path = dqa_reaggregate(
            source_workspace=args.source_workspace,
            records=records,
            setup=setup,
            output=ckpt_dir / f"{label}.pt",
            state_path=stats_dir / f"{label}_dqa_state.json",
            temperature=temp,
            min_server_alpha=min_server,
            classwise_blend=blend,
            residual_blend=residual,
            server_anchor=anchor,
        )
        save_record(
            rows,
            label,
            path,
            kind="aggregate",
            loop_id="loop5_dqa_reaggregation",
            hypothesis="DQA aggregation policy, not local training, is the immediate bottleneck",
            implementation=(
                f"phase1 reaggregate temp={temp}, min_server={min_server}, "
                f"class_blend={blend}, residual={residual}, anchor={anchor}"
            ),
        )

    return rows


def source_baseline_rows(source_workspace: Path) -> list[dict[str, Any]]:
    path = source_workspace / "stats" / "02_head_to_full_final_metrics.csv"
    rows = read_csv(path)
    mapping = {
        "warmup_global": "normal02_baseline",
        "phase1_final_aggregate": "normal02_baseline",
        "phase1_final_repair": "normal02_baseline",
        "phase2_final_aggregate": "normal02_baseline",
        "phase2_final_repair": "normal02_baseline",
    }
    output = []
    for row in rows:
        label = row.get("checkpoint_label", "")
        if label not in mapping:
            continue
        output.append(
            {
                "loop_id": mapping[label],
                "checkpoint_label": label,
                "kind": row.get("kind", ""),
                "variant": "source 02 baseline",
                "precision": row.get("precision", ""),
                "recall": row.get("recall", ""),
                "map50": row.get("map50", ""),
                "map50_95": row.get("map50_95", ""),
                "gain_vs_warmup_map50_95": row.get("gain_vs_warmup_map50_95", ""),
                "worst_split": row.get("worst_split", ""),
                "worst_split_map50_95": row.get("worst_split_map50_95", ""),
                "day_avg_map50_95": row.get("day_avg_map50_95", ""),
                "night_avg_map50_95": row.get("night_avg_map50_95", ""),
                "day_night_gap_map50_95": row.get("day_night_gap_map50_95", ""),
            }
        )
    return output


def loop1_rows(loop1_workspace: Path) -> list[dict[str, Any]]:
    path = loop1_workspace / "stats" / "02_fedmox_posthoc_metrics.csv"
    rows = read_csv(path)
    output = []
    for row in rows:
        label = row.get("checkpoint_label", "")
        if not label.startswith("loop"):
            continue
        output.append(
            {
                "loop_id": "loop1_fedmox_posthoc",
                "checkpoint_label": label,
                "kind": row.get("kind", ""),
                "variant": row.get("variant", ""),
                "precision": row.get("precision", ""),
                "recall": row.get("recall", ""),
                "map50": row.get("map50", ""),
                "map50_95": row.get("map50_95", ""),
                "gain_vs_warmup_map50_95": row.get("gain_vs_warmup_map50_95", ""),
                "worst_split": row.get("worst_split", ""),
                "worst_split_map50_95": row.get("worst_split_map50_95", ""),
                "day_avg_map50_95": row.get("day_avg_map50_95", ""),
                "night_avg_map50_95": row.get("night_avg_map50_95", ""),
                "day_night_gap_map50_95": row.get("day_night_gap_map50_95", ""),
            }
        )
    return output


def split_gap(by_label_split: Mapping[tuple[str, str], dict[str, str]], label: str) -> dict[str, str]:
    return htf.split_gap_metrics(dict(by_label_split), label)


def new_metric_rows(args: argparse.Namespace, eval_records: list[dict[str, str]]) -> list[dict[str, Any]]:
    summary_path = args.workspace_root / "validation_reports" / "paper_protocol_eval_summary.csv"
    rows = [row for row in read_csv(summary_path) if row.get("status") == "ok"]
    by_label_split = {(row["checkpoint_label"], row["split"]): row for row in rows}
    totals = {
        row["checkpoint_label"]: row
        for row in rows
        if row.get("split") in {"scene_daynight_total", "total"}
    }
    source_warm = next(
        (row for row in source_baseline_rows(args.source_workspace) if row["checkpoint_label"] == "warmup_global"),
        None,
    )
    warm_m95 = as_float(source_warm.get("map50_95")) if source_warm else None
    meta = {row["label"]: row for row in eval_records}
    output = []
    for label, total in totals.items():
        m95 = as_float(total.get("map50_95"))
        gap = split_gap(by_label_split, label)
        output.append(
            {
                "loop_id": meta[label].get("loop_id", ""),
                "checkpoint_label": label,
                "kind": meta[label].get("kind", ""),
                "variant": meta[label].get("variant", ""),
                "precision": total.get("precision", ""),
                "recall": total.get("recall", ""),
                "map50": total.get("map50", ""),
                "map50_95": total.get("map50_95", ""),
                "gain_vs_warmup_map50_95": "" if m95 is None or warm_m95 is None else f"{m95 - warm_m95:.6f}",
                **gap,
            }
        )
    return output


def summarize_loop(rows: list[dict[str, Any]], loop_id: str) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("loop_id") == loop_id and as_float(row.get("map50_95")) is not None]
    if not candidates:
        return {"loop_id": loop_id, "best_checkpoint": "", "best_map50_95": "", "finding": "no metrics"}
    best = max(candidates, key=lambda row: as_float(row.get("map50_95")) or -1.0)
    return {
        "loop_id": loop_id,
        "best_checkpoint": best["checkpoint_label"],
        "best_map50_95": best["map50_95"],
        "best_map50": best.get("map50", ""),
        "best_night_avg_map50_95": best.get("night_avg_map50_95", ""),
        "finding": best.get("variant", ""),
    }


def loop_log_rows(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    loop_meta = [
        {
            "loop_id": "loop1_fedmox_posthoc",
            "papers": "FedMox; FedMox says Soft-Mixture and spatial routing stabilize PSSFL.",
            "hypothesis": "Soft-Mixture or class-only updates can recover target signal without true architecture MoE.",
            "implementation": "Reuses MoE/02 posthoc SoftMix/class-only candidates.",
            "next_policy": "If only slight recovery, test whether repair erases residuals directly.",
        },
        {
            "loop_id": "loop2_repair_residual",
            "papers": "FedMox Soft-Mixture; FedSTO two-stage repair/adaptation.",
            "hypothesis": "Server repair removes useful Phase1/Phase2 target residuals.",
            "implementation": "Add a scaled target residual back onto repaired checkpoints.",
            "next_policy": "If this helps, future DQA should use residual-preserving repair.",
        },
        {
            "loop_id": "loop3_fedbn_bn_transplant",
            "papers": "FedBN motivates local/feature-shift-sensitive BN handling.",
            "hypothesis": "Scene/day-night shift is partly BN-stat mismatch after aggregation.",
            "implementation": "Transplant BN statistics from warmup/server/repaired checkpoints.",
            "next_policy": "If this helps night, add per-domain BN or BN-router.",
        },
        {
            "loop_id": "loop4_client_expert_oracle",
            "papers": "FedJETs/pFedMoE emphasize gating/routing over static averaging.",
            "hypothesis": "Client experts contain split-specific improvements that one global model hides.",
            "implementation": "Evaluate phase1 client checkpoints as expert candidates.",
            "next_policy": "If oracle gap is large, implement learned router/head-MoE.",
        },
        {
            "loop_id": "loop5_dqa_reaggregation",
            "papers": "FedMoE-DA domain-aware fine-grained aggregation; DQA reliability aggregation.",
            "hypothesis": "Aggregation policy is the bottleneck before local training.",
            "implementation": "Regenerate phase1 DQA aggregates with four policy settings.",
            "next_policy": "If a policy wins, promote it to a full multi-round DQA run.",
        },
    ]
    summaries = {row["loop_id"]: row for row in [summarize_loop(all_rows, item["loop_id"]) for item in loop_meta]}
    return [{**item, **summaries.get(item["loop_id"], {})} for item in loop_meta]


def write_markdown_report(args: argparse.Namespace, all_rows: list[dict[str, Any]], logs: list[dict[str, Any]]) -> None:
    ranked = sorted(
        [row for row in all_rows if as_float(row.get("map50_95")) is not None],
        key=lambda row: as_float(row.get("map50_95")) or -1.0,
        reverse=True,
    )
    lines = [
        "# MoE x DQA Five Research Loops",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- protocol: {PROTOCOL_VERSION}",
        f"- source_workspace: `{args.source_workspace.resolve()}`",
        f"- loop1_workspace: `{args.loop1_workspace.resolve()}`",
        "",
        "## Top Checkpoints",
        "",
        "| rank | loop | checkpoint | mAP50 | mAP50:95 | night avg | worst split | variant |",
        "|---:|---|---|---:|---:|---:|---|---|",
    ]
    for idx, row in enumerate(ranked[:15], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row.get("loop_id", "")),
                    str(row.get("checkpoint_label", "")),
                    str(row.get("map50", "")),
                    str(row.get("map50_95", "")),
                    str(row.get("night_avg_map50_95", "")),
                    str(row.get("worst_split", "")),
                    str(row.get("variant", "")).replace("|", "/"),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Loop Log", ""])
    for item in logs:
        lines.extend(
            [
                f"### {item['loop_id']}",
                "",
                f"- papers: {item['papers']}",
                f"- hypothesis: {item['hypothesis']}",
                f"- implementation: {item['implementation']}",
                f"- best: `{item.get('best_checkpoint', '')}` mAP50:95={item.get('best_map50_95', '')}",
                f"- finding: {item.get('finding', '')}",
                f"- next_policy: {item['next_policy']}",
                "",
            ]
        )
    (args.workspace_root / "03_five_research_loop_report.md").write_text("\n".join(lines), encoding="utf-8")


def notify(args: argparse.Namespace, message: str, *, title: str, status: str | None = None, error: str | None = None) -> None:
    try:
        from notebook_notify import notify_discord

        context: dict[str, Any] = {
            "workspace": str(args.workspace_root.resolve()),
            "status": status or "",
        }
        metrics_path = args.workspace_root / "stats" / "03_five_research_loop_metrics.csv"
        if metrics_path.exists():
            context["metrics_csv"] = str(metrics_path)
            rows = read_csv(metrics_path)
            ranked = sorted(
                [row for row in rows if row.get("map50_95")],
                key=lambda row: float(row["map50_95"]),
                reverse=True,
            )
            if ranked:
                context["best"] = ranked[0]["checkpoint_label"]
                context["best_map50_95"] = ranked[0]["map50_95"]
        if error:
            context["error"] = error[:500]
        print(notify_discord(message, title=title, context=context, fail_silently=True))
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-workspace", type=Path, default=SOURCE_WORKSPACE)
    parser.add_argument("--loop1-workspace", type=Path, default=LOOP1_WORKSPACE)
    parser.add_argument("--warmup-checkpoint", type=Path, default=REPO_ROOT / "pseudogt_learnability" / "checkpoints" / "round000_warmup.pt")
    parser.add_argument("--client-limit", type=int, default=1500)
    parser.add_argument("--clients", default="all")
    parser.add_argument("--device", default="")
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument(
        "--eval-splits",
        default="highway_day,highway_night,citystreet_day,citystreet_night,residential_day,residential_night,total",
    )
    parser.add_argument("--eval-clients", action="store_true")
    parser.add_argument("--classwise", action="store_true")
    parser.add_argument("--no-eval-plots", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--notify-start", action="store_true")
    parser.add_argument("--notify-end", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.source_workspace = args.source_workspace.expanduser().resolve()
    args.loop1_workspace = args.loop1_workspace.expanduser().resolve()
    htf.pl03.ensure_dirs(args.workspace_root)
    setup, _fedsto, _manifest, _clients, _warmup = htf.prepare(args)
    if args.setup_only:
        print("Setup complete.")
        return []

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "source_workspace": str(args.source_workspace.resolve()),
        "loop1_workspace": str(args.loop1_workspace.resolve()),
        "workspace": str(args.workspace_root.resolve()),
        "sources": [
            "https://arxiv.org/abs/2508.16568",
            "https://arxiv.org/abs/2310.17097",
            "https://arxiv.org/abs/2102.07623",
            "https://arxiv.org/abs/2411.02115",
            "https://arxiv.org/abs/2402.01350",
            "https://huggingface.co/papers/2306.08586",
        ],
    }
    (args.workspace_root / "stats" / "03_five_research_loop_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    eval_records = generate_loop_candidates(args, setup)
    write_csv(
        args.workspace_root / "stats" / "03_five_research_loop_checkpoints.csv",
        eval_records,
        ["label", "kind", "phase", "phase_round", "global_round", "client", "variant", "loop_id", "hypothesis", "path"],
    )

    new_rows: list[dict[str, Any]] = []
    if args.evaluate:
        htf.base01_0.run_evaluation(args, eval_records)
        new_rows = new_metric_rows(args, eval_records)

    all_rows = source_baseline_rows(args.source_workspace) + loop1_rows(args.loop1_workspace) + new_rows
    metric_fields = [
        "loop_id",
        "checkpoint_label",
        "kind",
        "variant",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "gain_vs_warmup_map50_95",
        "worst_split",
        "worst_split_map50_95",
        "day_avg_map50_95",
        "night_avg_map50_95",
        "day_night_gap_map50_95",
    ]
    write_csv(args.workspace_root / "stats" / "03_five_research_loop_metrics.csv", all_rows, metric_fields)
    logs = loop_log_rows(all_rows)
    write_csv(
        args.workspace_root / "stats" / "03_five_research_loop_log.csv",
        logs,
        [
            "loop_id",
            "papers",
            "hypothesis",
            "implementation",
            "best_checkpoint",
            "best_map50",
            "best_map50_95",
            "best_night_avg_map50_95",
            "finding",
            "next_policy",
        ],
    )
    write_markdown_report(args, all_rows, logs)
    return all_rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    do_start_notify = args.notify or args.notify_start
    do_end_notify = args.notify or args.notify_end
    if do_start_notify:
        notify(args, "MoE x DQA five research-loop sprint started.", title="DQA MoE 03 start")
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
                f"MoE x DQA five research-loop sprint finished with status={status}.",
                title="DQA MoE 03 finish",
                status=status,
                error=error,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
