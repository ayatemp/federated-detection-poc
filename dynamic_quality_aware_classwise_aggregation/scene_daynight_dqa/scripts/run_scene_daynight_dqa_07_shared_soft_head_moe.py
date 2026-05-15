#!/usr/bin/env python3
"""Run 07 Shared-detector Soft Head-MoE DQA.

This runner keeps the useful part of 06, namely counterfactual pseudoGT routing,
but avoids the weak independent-detector/output-fusion failure mode.

The practical implementation is a deployable single-checkpoint approximation of
shared-trunk soft head-MoE:

* build clean/illumination/bridge pseudoGT buckets;
* train head/neck-only route experts from the same shared teacher checkpoint;
* keep the teacher trunk as the shared detector;
* compose final checkpoints as base + beta * sum(router_weight * head_delta);
* evaluate several soft-router mixtures with the scene-daynight protocol.

It is intentionally not output-space WBF and not residual deployment from local
client checkpoints.  The MoE signal enters before evaluation, as a head/neck
parameter mixture.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
NAV_ROOT = REPO_ROOT / "navigating_data_heterogeneity"
DQA_ROOT = PROJECT_ROOT.parent
PSEUDOGT_SCRIPTS = REPO_ROOT / "pseudogt_learnability" / "scripts"
PROTOCOL_VERSION = "scene_daynight_dqa_07_shared_soft_head_moe_v1"

for path in (PROJECT_ROOT / "scripts", PROJECT_ROOT.parent, DQA_ROOT, NAV_ROOT, PSEUDOGT_SCRIPTS, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dqa_cwa_aggregation as dqa_v1  # noqa: E402
import run_scene_daynight_dqa_01_0 as base01_0  # noqa: E402
import run_scene_daynight_dqa_03_main_experiment as main03  # noqa: E402
import run_scene_daynight_dqa_06_counterfactual_output_moe as cf06  # noqa: E402


DEFAULT_WORKSPACE = PROJECT_ROOT / "output" / "07_shared_soft_head_moe_dqa"
DEFAULT_SOURCE_WORKSPACE = PROJECT_ROOT / "output" / "03_main_bn_residual_dqa_experiment"
DEFAULT_SOURCE_06_WORKSPACE = PROJECT_ROOT / "output" / "06_counterfactual_output_moe_dqa"
DEFAULT_ROUTER_WORKSPACE = PROJECT_ROOT / "output" / "05_expert_choice_pseudogt_router_dqa"
DEFAULT_TEACHER = (
    DEFAULT_SOURCE_WORKSPACE
    / "bn_residual_dqa"
    / "checkpoints"
    / "round030_bn_residual_dqa_aggregate.pt"
)
DEFAULT_ROUTER_TEACHER = (
    DEFAULT_ROUTER_WORKSPACE
    / "checkpoints"
    / "round030_expert_choice_pseudogt_router_aggregate.pt"
)
EXPERT_ORDER = ("clean_original", "illumination_rescued", "cross_view_bridge")
DEFAULT_EVAL_SPLITS = "highway_day,highway_night,citystreet_day,citystreet_night,residential_day,residential_night,total"


@dataclass(frozen=True)
class MixtureSpec:
    name: str
    weights: dict[str, float]
    beta: float
    note: str


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


def seconds_to_hms(seconds: float | None) -> str:
    return cf06.seconds_to_hms(seconds)


def as_float(value: Any, default: float | None = None) -> float | None:
    return cf06.as_float(value, default)


def normalize_weights(weights: Mapping[str, float], enabled: Sequence[str]) -> dict[str, float]:
    selected = {expert: max(0.0, float(weights.get(expert, 0.0))) for expert in enabled}
    total = sum(selected.values())
    if total <= 0:
        return {expert: 1.0 / max(len(enabled), 1) for expert in enabled}
    return {expert: value / total for expert, value in selected.items()}


def default_mixture_specs(scan_summary: Mapping[str, Any] | None = None) -> list[MixtureSpec]:
    """Return conservative soft-head router candidates.

    The beta values are deliberately smaller than 1.0 because 06 showed each
    route expert is weaker as a standalone detector.  Here they should nudge the
    strong 03 teacher head, not replace it.
    """

    signal = (scan_summary or {}).get("day_night_signal", {})
    night_gap = as_float(signal.get("night_minus_day_rescued_ratio"), 0.0) or 0.0
    night_boost = min(0.08, max(0.0, night_gap))
    return [
        MixtureSpec(
            name="07_soft_head_conservative",
            weights={
                "clean_original": 0.68 - night_boost,
                "illumination_rescued": 0.24 + night_boost,
                "cross_view_bridge": 0.08,
            },
            beta=0.35,
            note="small expert nudge; safest test against 03",
        ),
        MixtureSpec(
            name="07_soft_head_balanced",
            weights={
                "clean_original": 0.52 - night_boost / 2.0,
                "illumination_rescued": 0.34 + night_boost,
                "cross_view_bridge": 0.14 - night_boost / 2.0,
            },
            beta=0.50,
            note="default shared-head soft MoE",
        ),
        MixtureSpec(
            name="07_soft_head_night_boost",
            weights={
                "clean_original": 0.42 - night_boost / 2.0,
                "illumination_rescued": 0.44 + night_boost,
                "cross_view_bridge": 0.14 - night_boost / 2.0,
            },
            beta=0.50,
            note="more illumination route for night-domain recovery",
        ),
        MixtureSpec(
            name="07_soft_head_bridge_boost",
            weights={
                "clean_original": 0.48,
                "illumination_rescued": 0.30,
                "cross_view_bridge": 0.22,
            },
            beta=0.50,
            note="tests whether cross-view route adds complementary geometry",
        ),
        MixtureSpec(
            name="07_soft_head_balanced_strong",
            weights={
                "clean_original": 0.52 - night_boost / 2.0,
                "illumination_rescued": 0.34 + night_boost,
                "cross_view_bridge": 0.14 - night_boost / 2.0,
            },
            beta=0.70,
            note="stronger version; useful if conservative under-moves",
        ),
    ]


def load_checkpoint(path: Path) -> dict[str, Any]:
    return dqa_v1._load_checkpoint(path, REPO_ROOT)


def state_dict(ckpt: Mapping[str, Any], key: str) -> dict[str, torch.Tensor] | None:
    if ckpt.get(key) is None:
        return None
    return dqa_v1._model_state_dict(ckpt, key)


def replace_state(base: dict[str, Any], state: Mapping[str, torch.Tensor], key: str) -> None:
    dqa_v1._replace_model_state(base, dict(state), key)


def key_filter_from_scope(scope: str) -> Callable[[str], bool] | None:
    return main03.key_filter_from_scope(scope)


def weighted_head_moe_state(
    base_state: Mapping[str, torch.Tensor],
    expert_states: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    weights: Mapping[str, float],
    beta: float,
    scope: str,
    include_bn: bool,
) -> dict[str, torch.Tensor]:
    key_filter = key_filter_from_scope(scope)
    result: dict[str, torch.Tensor] = {}
    for key, base_value in base_state.items():
        if not include_bn and dqa_v1._is_batchnorm_key(key):
            result[key] = base_value
        elif key_filter is not None and not key_filter(key):
            result[key] = base_value
        elif torch.is_tensor(base_value) and base_value.dtype.is_floating_point:
            residual = torch.zeros_like(base_value.float())
            for expert, weight in weights.items():
                state = expert_states.get(expert)
                if state is None or key not in state:
                    continue
                residual = residual + float(weight) * (state[key].float() - base_value.float())
            result[key] = (base_value.float() + float(beta) * residual).to(base_value.dtype)
        else:
            result[key] = base_value
    return result


def compose_soft_head_checkpoint(
    *,
    base_checkpoint: Path,
    expert_checkpoints: Mapping[str, Path],
    spec: MixtureSpec,
    output: Path,
    scope: str,
    include_bn: bool,
) -> Path:
    enabled = [expert for expert in EXPERT_ORDER if expert in expert_checkpoints]
    weights = normalize_weights(spec.weights, enabled)
    base_ckpt = load_checkpoint(base_checkpoint)
    expert_ckpts = {expert: load_checkpoint(path) for expert, path in expert_checkpoints.items() if expert in enabled}
    out = copy.deepcopy(base_ckpt)

    model = weighted_head_moe_state(
        dqa_v1._model_state_dict(base_ckpt, "model"),
        {expert: dqa_v1._model_state_dict(ckpt, "model") for expert, ckpt in expert_ckpts.items()},
        weights=weights,
        beta=spec.beta,
        scope=scope,
        include_bn=include_bn,
    )
    replace_state(out, model, "model")

    base_ema = state_dict(base_ckpt, "ema")
    expert_emas = {expert: state_dict(ckpt, "ema") for expert, ckpt in expert_ckpts.items()}
    if base_ema is not None and all(item is not None for item in expert_emas.values()):
        ema = weighted_head_moe_state(
            base_ema,
            {expert: item for expert, item in expert_emas.items() if item is not None},
            weights=weights,
            beta=spec.beta,
            scope=scope,
            include_bn=include_bn,
        )
        replace_state(out, ema, "ema")

    out.setdefault("meta", {})
    if isinstance(out["meta"], dict):
        out["meta"].update(
            {
                "protocol": PROTOCOL_VERSION,
                "stage": spec.name,
                "soft_head_moe_weights": weights,
                "soft_head_moe_beta": spec.beta,
                "soft_head_moe_scope": scope,
                "soft_head_moe_include_bn": include_bn,
            }
        )
    out["epoch"] = -1
    out["optimizer"] = None
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    return output


def prepare_counterfactual_assets(
    args: argparse.Namespace,
) -> tuple[Mapping[str, Any], list[dict[str, str]], dict[str, Path]]:
    setup, fedsto, _manifest, clients = cf06.prepare_workspace(args)
    selected_experts = cf06.resolve_experts(args.experts)

    if args.reuse_06_assets:
        scan_path = args.source_06_workspace / "stats" / "06_counterfactual_scan_summary.json"
        if not scan_path.exists():
            raise FileNotFoundError(f"--reuse-06-assets requested but scan summary is missing: {scan_path}")
        scan_summary = json.loads(scan_path.read_text(encoding="utf-8"))
        expert_checkpoints: dict[str, Path] = {}
        expert_records: list[dict[str, str]] = []
        for expert in selected_experts:
            ckpt = args.source_06_workspace / "checkpoints" / f"06_counterfactual_{expert}_expert.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"--reuse-06-assets requested but expert checkpoint is missing: {ckpt}")
            expert_checkpoints[expert] = ckpt
            expert_records.append(
                {
                    "condition": "07_reused_06_route_expert",
                    "label": f"06_{expert}_expert",
                    "kind": "route_expert",
                    "round": "",
                    "client": "",
                    "variant": "reused_from_06",
                    "path": str(ckpt.resolve()),
                }
            )
        return scan_summary, expert_records, expert_checkpoints

    scan_path = args.workspace_root / "stats" / "07_counterfactual_scan_summary.json"
    if scan_path.exists() and not args.force_pseudo:
        print(f"Reusing 07 counterfactual scan summary: {scan_path}")
        scan_summary = json.loads(scan_path.read_text(encoding="utf-8"))
    else:
        scan_summary = cf06.moe09.scan_counterfactual_views(args, setup, clients)
        scan_path.write_text(json.dumps(scan_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    expert_records, expert_checkpoints = cf06.train_experts(args, setup, fedsto, scan_summary, selected_experts)
    return scan_summary, expert_records, expert_checkpoints


def compose_mixtures(
    args: argparse.Namespace,
    scan_summary: Mapping[str, Any],
    expert_checkpoints: Mapping[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    specs = default_mixture_specs(scan_summary)
    records: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    for spec in specs:
        output = args.workspace_root / "checkpoints" / f"{spec.name}.pt"
        if output.exists() and not args.force:
            print(f"Reusing soft-head MoE checkpoint: {output}")
        else:
            compose_soft_head_checkpoint(
                base_checkpoint=args.teacher_checkpoint,
                expert_checkpoints=expert_checkpoints,
                spec=spec,
                output=output,
                scope=args.moe_scope,
                include_bn=args.include_bn,
            )
        weights = normalize_weights(spec.weights, [expert for expert in EXPERT_ORDER if expert in expert_checkpoints])
        rows.append(
            {
                "label": spec.name,
                "checkpoint": str(output),
                "beta": spec.beta,
                "scope": args.moe_scope,
                "include_bn": args.include_bn,
                "clean_original": weights.get("clean_original", 0.0),
                "illumination_rescued": weights.get("illumination_rescued", 0.0),
                "cross_view_bridge": weights.get("cross_view_bridge", 0.0),
                "note": spec.note,
            }
        )
        records.append(
            {
                "condition": "07_shared_soft_head_moe",
                "label": spec.name,
                "kind": "soft_head_moe",
                "round": "",
                "client": "",
                "variant": f"beta={spec.beta}:scope={args.moe_scope}",
                "path": str(output.resolve()),
            }
        )
    write_csv(
        args.workspace_root / "stats" / "07_soft_head_moe_variants.csv",
        rows,
        [
            "label",
            "checkpoint",
            "beta",
            "scope",
            "include_bn",
            "clean_original",
            "illumination_rescued",
            "cross_view_bridge",
            "note",
        ],
    )
    return rows, records


def build_eval_records(
    args: argparse.Namespace,
    expert_records: list[dict[str, str]],
    moe_records: list[dict[str, str]],
) -> list[dict[str, str]]:
    records = [
        {
            "condition": "03_reference",
            "label": "03_bn_residual_dqa_aggregate",
            "kind": "aggregate",
            "round": "30",
            "client": "",
            "variant": "reference",
            "path": str(args.teacher_checkpoint.resolve()),
        },
    ]
    if args.router_teacher_checkpoint.exists():
        records.append(
            {
                "condition": "05_reference",
                "label": "05_expert_choice_router_aggregate",
                "kind": "aggregate",
                "round": "30",
                "client": "",
                "variant": "reference",
                "path": str(args.router_teacher_checkpoint.resolve()),
            }
        )
    if args.evaluate_route_experts:
        records.extend(expert_records)
    records.extend(moe_records)
    write_csv(
        args.workspace_root / "stats" / "07_eval_checkpoints.csv",
        records,
        ["condition", "label", "kind", "round", "client", "variant", "path"],
    )
    return records


def summarize_metrics(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_csv(args.workspace_root / "validation_reports" / "paper_protocol_eval_summary.csv")
        if row.get("status") == "ok"
    ]
    total_rows = [row for row in rows if row.get("split") in {"scene_daynight_total", "total"}]
    by_label_total = {row["checkpoint_label"]: row for row in total_rows}
    by_label_split = {(row["checkpoint_label"], row["split"]): row for row in rows}
    ref = as_float(by_label_total.get("03_bn_residual_dqa_aggregate", {}).get("map50_95"), 0.0) or 0.0
    router = as_float(by_label_total.get("05_expert_choice_router_aggregate", {}).get("map50_95"), 0.0) or 0.0

    out: list[dict[str, Any]] = []
    for label, total in by_label_total.items():
        m95 = as_float(total.get("map50_95"), 0.0) or 0.0
        m50 = as_float(total.get("map50"), 0.0) or 0.0
        gap = base01_0.split_gap_metrics(by_label_split, label)
        out.append(
            {
                "checkpoint_label": label,
                "precision": total.get("precision", ""),
                "recall": total.get("recall", ""),
                "map50": f"{m50:.6f}",
                "map50_95": f"{m95:.6f}",
                "delta_vs_03_map50_95": f"{m95 - ref:.6f}",
                "delta_vs_05_router_map50_95": f"{m95 - router:.6f}",
                **gap,
            }
        )
    out.sort(key=lambda row: float(row.get("map50_95") or 0.0), reverse=True)
    write_csv(
        args.workspace_root / "stats" / "07_soft_head_moe_metrics.csv",
        out,
        [
            "checkpoint_label",
            "precision",
            "recall",
            "map50",
            "map50_95",
            "delta_vs_03_map50_95",
            "delta_vs_05_router_map50_95",
            "worst_split",
            "worst_split_map50_95",
            "day_avg_map50_95",
            "night_avg_map50_95",
            "day_night_gap_map50_95",
        ],
    )
    return out


def write_report(
    args: argparse.Namespace,
    scan_summary: Mapping[str, Any],
    variant_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    elapsed_seconds: float,
) -> None:
    totals = scan_summary.get("totals", {})
    signal = scan_summary.get("day_night_signal", {})
    lines = [
        "# Scene-Daynight DQA 07: Shared Soft Head-MoE",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- protocol: {PROTOCOL_VERSION}",
        f"- elapsed: {seconds_to_hms(elapsed_seconds)}",
        f"- workspace: `{args.workspace_root}`",
        f"- teacher: `{args.teacher_checkpoint}`",
        f"- reuse_06_assets: {args.reuse_06_assets}",
        "",
        "## Design",
        "",
        "07 keeps the 03 shared detector as the trunk and injects counterfactual pseudoGT route expertise only through head/neck deltas.  It is a single-checkpoint soft Head-MoE approximation, not output-space WBF.",
        "",
        "## Counterfactual pseudoGT signal",
        "",
        f"- clean_original boxes: {totals.get('clean_original_boxes', '')}",
        f"- illumination_rescued boxes: {totals.get('illumination_rescued_boxes', '')}",
        f"- cross_view_bridge boxes: {totals.get('cross_view_bridge_boxes', '')}",
        f"- rescued ratio: {as_float(totals.get('rescued_ratio'), 0.0):.3f}",
        f"- day rescued ratio: {as_float(signal.get('day_rescued_ratio'), 0.0):.3f}",
        f"- night rescued ratio: {as_float(signal.get('night_rescued_ratio'), 0.0):.3f}",
        f"- night-day rescued gap: {as_float(signal.get('night_minus_day_rescued_ratio'), 0.0):.3f}",
        "",
        "## Soft Head-MoE variants",
        "",
        "| variant | beta | clean | illum | bridge | note |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in variant_rows:
        lines.append(
            f"| {row['label']} | {float(row['beta']):.2f} | {float(row['clean_original']):.3f} | "
            f"{float(row['illumination_rescued']):.3f} | {float(row['cross_view_bridge']):.3f} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| checkpoint | mAP50 | mAP50:95 | delta vs 03 | delta vs 05 | day avg | night avg | worst split |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in metric_rows:
        lines.append(
            f"| {row['checkpoint_label']} | {row['map50']} | {row['map50_95']} | "
            f"{row['delta_vs_03_map50_95']} | {row['delta_vs_05_router_map50_95']} | "
            f"{row['day_avg_map50_95']} | {row['night_avg_map50_95']} | {row['worst_split']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation hook",
            "",
            "If 07 beats 03, the key claim is that pseudoGT routing can help when it is constrained to soft head deltas on top of a strong shared detector.  If 07 stays below 03 but above 06 output-MoE, the pseudoGT route signal is useful but still too noisy.  If it drops below the route experts, the mixture is over-regularized or the selected beta is too small.",
        ]
    )
    (args.workspace_root / "07_shared_soft_head_moe_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def notify(args: argparse.Namespace, message: str, *, title: str, status: str = "", error: str | None = None) -> None:
    try:
        from notebook_notify import notify_discord

        context: dict[str, Any] = {
            "workspace": str(args.workspace_root.expanduser().resolve()),
            "status": status,
            "report": str(args.workspace_root.expanduser().resolve() / "07_shared_soft_head_moe_report.md"),
        }
        metrics_path = args.workspace_root.expanduser().resolve() / "stats" / "07_soft_head_moe_metrics.csv"
        if metrics_path.exists():
            context["metrics_csv"] = str(metrics_path)
        if error:
            context["error"] = error[:500]
        print(notify_discord(message, title=title, context=context, fail_silently=True))
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-workspace", type=Path, default=DEFAULT_SOURCE_WORKSPACE)
    parser.add_argument("--source-06-workspace", type=Path, default=DEFAULT_SOURCE_06_WORKSPACE)
    parser.add_argument("--router-workspace", type=Path, default=DEFAULT_ROUTER_WORKSPACE)
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--router-teacher-checkpoint", type=Path, default=DEFAULT_ROUTER_TEACHER)
    parser.add_argument("--reuse-06-assets", action="store_true")
    parser.add_argument("--client-limit", type=int, default=1500)
    parser.add_argument("--clients", default="all")
    parser.add_argument("--max-images-per-client", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.20)
    parser.add_argument("--nms-iou-thres", type=float, default=0.65)
    parser.add_argument("--match-iou", type=float, default=0.55)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--min-stability", type=float, default=0.55)
    parser.add_argument("--min-score", type=float, default=0.10)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-boxes-per-image", type=int, default=24)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--experts", default="clean_original,illumination_rescued,cross_view_bridge")
    parser.add_argument("--expert-epochs", type=int, default=1)
    parser.add_argument("--expert-train-scope", choices=["neck_head", "all"], default="neck_head")
    parser.add_argument("--expert-orthogonal-weight", type=float, default=1e-4)
    parser.add_argument("--clean-pseudo-repeat", type=int, default=1)
    parser.add_argument("--illumination-pseudo-repeat", type=int, default=2)
    parser.add_argument("--bridge-pseudo-repeat", type=int, default=2)
    parser.add_argument("--hybrid-pseudo-repeat", type=int, default=1)
    parser.add_argument("--clean-lr", type=float, default=0.0007)
    parser.add_argument("--illumination-lr", type=float, default=0.0006)
    parser.add_argument("--bridge-lr", type=float, default=0.0005)
    parser.add_argument("--hybrid-lr", type=float, default=0.0006)
    parser.add_argument("--clean-loss-box", type=float, default=0.005)
    parser.add_argument("--illumination-loss-box", type=float, default=0.003)
    parser.add_argument("--bridge-loss-box", type=float, default=0.002)
    parser.add_argument("--hybrid-loss-box", type=float, default=0.003)
    parser.add_argument("--expert-scale-aug", type=float, default=0.25)
    parser.add_argument("--expert-hsv-s", type=float, default=0.35)
    parser.add_argument("--expert-hsv-v", type=float, default=0.20)
    parser.add_argument("--min-expert-images", type=int, default=20)
    parser.add_argument("--moe-scope", choices=["neck_head", "all"], default="neck_head")
    parser.add_argument("--include-bn", action="store_true", default=True)
    parser.add_argument("--no-include-bn", action="store_false", dest="include_bn")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=33271)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--evaluate-route-experts", action="store_true")
    parser.add_argument("--eval-splits", default=DEFAULT_EVAL_SPLITS)
    parser.add_argument("--eval-clients", action="store_true")
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--classwise", action="store_true")
    parser.add_argument("--no-eval-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-pseudo", action="store_true")
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--notify-start", action="store_true")
    parser.add_argument("--notify-end", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.monotonic()
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.source_workspace = args.source_workspace.expanduser().resolve()
    args.source_06_workspace = args.source_06_workspace.expanduser().resolve()
    args.router_workspace = args.router_workspace.expanduser().resolve()
    args.teacher_checkpoint = args.teacher_checkpoint.expanduser().resolve()
    args.router_teacher_checkpoint = args.router_teacher_checkpoint.expanduser().resolve()
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    (args.workspace_root / "stats").mkdir(parents=True, exist_ok=True)
    (args.workspace_root / "checkpoints").mkdir(parents=True, exist_ok=True)

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "workspace": str(args.workspace_root),
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "router_teacher_checkpoint": str(args.router_teacher_checkpoint),
        "reuse_06_assets": args.reuse_06_assets,
        "moe_scope": args.moe_scope,
        "include_bn": args.include_bn,
        "expert_train_scope": args.expert_train_scope,
        "experts": cf06.resolve_experts(args.experts),
    }
    (args.workspace_root / "stats" / "07_shared_soft_head_moe_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    scan_summary, expert_records, expert_checkpoints = prepare_counterfactual_assets(args)
    variant_rows, moe_records = compose_mixtures(args, scan_summary, expert_checkpoints)
    eval_records = build_eval_records(args, expert_records, moe_records)

    metric_rows: list[dict[str, Any]] = []
    if args.evaluate:
        base01_0.run_evaluation(args, eval_records)
        metric_rows = summarize_metrics(args)

    elapsed = time.monotonic() - start
    write_report(args, scan_summary, variant_rows, metric_rows, elapsed)
    result = {
        "status": "ok",
        "elapsed_seconds": elapsed,
        "elapsed_hms": seconds_to_hms(elapsed),
        "workspace": str(args.workspace_root),
        "variants": variant_rows,
        "metrics": metric_rows,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.notify or args.notify_start:
        notify(args, "Scene-Daynight DQA 07 shared soft Head-MoE started.", title="DQA 07 start", status="started")
    status = "success"
    error: str | None = None
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = str(exc)
        raise
    finally:
        if args.notify or args.notify_end:
            notify(
                args,
                f"Scene-Daynight DQA 07 shared soft Head-MoE finished with status={status}.",
                title="DQA 07 finish",
                status=status,
                error=error,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
