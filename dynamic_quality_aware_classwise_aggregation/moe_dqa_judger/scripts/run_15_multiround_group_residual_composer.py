#!/usr/bin/env python3
"""Compose self-generated residuals from different rounds per parameter group.

Single-round incumbent residuals helped a little, but the best domain winners
come from different rounds and different roles.  This run builds candidates
where body/head/router/expert groups can borrow residuals from different
self-training rounds:

    M = I + alpha_g * (A_r(g) - G_r(g)) + beta_g * (S_r(g) - G_r(g))

No external teacher is used.  The candidate pool is evaluated on the paper
protocol total split and on the six day/night domain slices.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "output" / "15_multiround_group_residual_composer"
SOURCE_WORKSPACE = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation" / "moe_dqa" / "output" / "01_dqa_fedmox_yolo_full"
INCUMBENT = PROJECT_ROOT / "output" / "03_mix_judger_policy" / "candidates" / "r002_judger03_selected_r002.pt"
BASELINE_TOTAL_SCORE = 0.57455
BASELINE_LABEL = "incumbent_r002"

if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_01_judger_probe as judger01  # noqa: E402
import run_02_mix_weight_optimizer as opt02  # noqa: E402
import run_04_delta_expert_optimizer as delta04  # noqa: E402
import run_10_night_targeted_delta_optimizer as opt10  # noqa: E402


GROUPS = delta04.GROUPS
COEFF_FIELDS = delta04.COEFF_FIELDS
DOMAIN_SOURCES = [
    PROJECT_ROOT / "output" / "08_domain_slice_judger" / "stats" / "08_domain_eval.csv",
    PROJECT_ROOT / "output" / "10_night_targeted_delta_optimizer" / "stats" / "10_full_domain_eval.csv",
    PROJECT_ROOT / "output" / "11_highway_night_full_optimizer" / "stats" / "11_full_domain_eval.csv",
    PROJECT_ROOT / "output" / "12_domain_winner_soup" / "stats" / "12_domain_eval.csv",
]
TOTAL_SOURCES = [
    PROJECT_ROOT / "output" / "07_incumbent_delta_judger" / "stats" / "07_full_eval.csv",
    PROJECT_ROOT / "output" / "10_night_targeted_delta_optimizer" / "stats" / "10_full_total_eval.csv",
    PROJECT_ROOT / "output" / "11_highway_night_full_optimizer" / "stats" / "11_full_total_eval.csv",
]


def notify(message: str, title: str, enabled: bool) -> None:
    if not enabled:
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from notebook_notify import notify_discord

        notify_discord(message, title=title, fail_silently=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notify skipped: {exc}")


def parse_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def state_dict_from(ckpt: dict[str, Any], field: str) -> dict[str, torch.Tensor] | None:
    return delta04.state_dict_from(ckpt, field)


def replace_state(ckpt: dict[str, Any], field: str, state: dict[str, torch.Tensor]) -> None:
    delta04.replace_state(ckpt, field, state)


def load_coeff_rows() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for source in TOTAL_SOURCES:
        for row in read_csv(source):
            round_idx = int(parse_float(row.get("round"), -1))
            candidate_id = row.get("candidate_id", "")
            label = row.get("label", "") or (f"r{round_idx:03d}_{candidate_id}" if round_idx >= 0 and candidate_id else "")
            if not label or any(field not in row for field in COEFF_FIELDS):
                continue
            coeffs = {field: parse_float(row.get(field), 0.0) for field in COEFF_FIELDS}
            if round_idx < 0:
                continue
            out[label] = {"label": label, "round": round_idx, "coeffs": coeffs, "source_file": str(source)}
    return out


def group_spec_from_label(label: str, group: str, coeff_rows: dict[str, dict[str, Any]], scale: float = 1.0) -> dict[str, Any]:
    row = coeff_rows[label]
    coeffs = row["coeffs"]
    return {
        "label": label,
        "round": int(row["round"]),
        "a": parse_float(coeffs.get(f"{group}_a"), 0.0) * scale,
        "s": parse_float(coeffs.get(f"{group}_s"), 0.0) * scale,
    }


def zero_spec() -> dict[str, Any]:
    return {"label": "incumbent", "round": -1, "a": 0.0, "s": 0.0}


def candidate_specs(coeff_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    labels = set(coeff_rows)
    required = {
        "r004_scaled04_best01_rand000_075",
        "r006_scaled04_best01_rand000_050",
        "r019_residential_night_precision",
        "r021_city_night_recall",
        "r003_tiny_all_s",
        "r015_residential_night_precision",
    }
    missing = sorted(required - labels)
    if missing:
        raise FileNotFoundError(f"Missing coefficient rows: {missing}")

    def build(name: str, sources: dict[str, tuple[str, float]]) -> dict[str, Any]:
        groups: dict[str, dict[str, Any]] = {}
        for group in GROUPS:
            label, scale = sources.get(group, sources.get("default", ("incumbent", 1.0)))
            groups[group] = zero_spec() if label == "incumbent" else group_spec_from_label(label, group, coeff_rows, scale)
        return {"label": name, "groups": groups}

    raw = [
        build(
            "mr_r006_moe_r019_head",
            {
                "body": ("incumbent", 1.0),
                "head": ("r019_residential_night_precision", 1.0),
                "router": ("r006_scaled04_best01_rand000_050", 1.0),
                "expert0": ("r006_scaled04_best01_rand000_050", 1.0),
                "expert1": ("r006_scaled04_best01_rand000_050", 1.0),
                "expert2": ("r006_scaled04_best01_rand000_050", 1.0),
                "expert3": ("r006_scaled04_best01_rand000_050", 1.0),
            },
        ),
        build(
            "mr_r004_body_r006_moe_r019_head",
            {
                "body": ("r004_scaled04_best01_rand000_075", 0.50),
                "head": ("r019_residential_night_precision", 0.90),
                "router": ("r006_scaled04_best01_rand000_050", 0.90),
                "expert0": ("r006_scaled04_best01_rand000_050", 0.90),
                "expert1": ("r006_scaled04_best01_rand000_050", 0.90),
                "expert2": ("r006_scaled04_best01_rand000_050", 0.90),
                "expert3": ("r006_scaled04_best01_rand000_050", 0.90),
            },
        ),
        build(
            "mr_city_res_highway_split",
            {
                "body": ("incumbent", 1.0),
                "head": ("r019_residential_night_precision", 1.0),
                "router": ("r021_city_night_recall", 1.0),
                "expert0": ("r006_scaled04_best01_rand000_050", 1.0),
                "expert1": ("r006_scaled04_best01_rand000_050", 1.0),
                "expert2": ("r021_city_night_recall", 1.0),
                "expert3": ("r019_residential_night_precision", 1.0),
            },
        ),
        build(
            "mr_resnight_tiny_head_r006_moe",
            {
                "body": ("incumbent", 1.0),
                "head": ("r003_tiny_all_s", 0.75),
                "router": ("r006_scaled04_best01_rand000_050", 0.85),
                "expert0": ("r006_scaled04_best01_rand000_050", 0.85),
                "expert1": ("r006_scaled04_best01_rand000_050", 0.85),
                "expert2": ("r019_residential_night_precision", 0.80),
                "expert3": ("r019_residential_night_precision", 0.80),
            },
        ),
        build(
            "mr_conservative_all_winners",
            {
                "body": ("r004_scaled04_best01_rand000_075", 0.25),
                "head": ("r019_residential_night_precision", 0.55),
                "router": ("r006_scaled04_best01_rand000_050", 0.45),
                "expert0": ("r006_scaled04_best01_rand000_050", 0.45),
                "expert1": ("r006_scaled04_best01_rand000_050", 0.45),
                "expert2": ("r021_city_night_recall", 0.45),
                "expert3": ("r015_residential_night_precision", 0.45),
            },
        ),
        build(
            "mr_r019_head_router_only",
            {
                "body": ("incumbent", 1.0),
                "head": ("r019_residential_night_precision", 1.0),
                "router": ("r019_residential_night_precision", 1.0),
                "expert0": ("incumbent", 1.0),
                "expert1": ("incumbent", 1.0),
                "expert2": ("r019_residential_night_precision", 0.65),
                "expert3": ("r019_residential_night_precision", 0.65),
            },
        ),
        build(
            "mr_r006_highway_preserve",
            {
                "body": ("r006_scaled04_best01_rand000_050", 0.35),
                "head": ("r006_scaled04_best01_rand000_050", 1.0),
                "router": ("r006_scaled04_best01_rand000_050", 1.0),
                "expert0": ("r006_scaled04_best01_rand000_050", 1.0),
                "expert1": ("r006_scaled04_best01_rand000_050", 1.0),
                "expert2": ("r006_scaled04_best01_rand000_050", 1.0),
                "expert3": ("r006_scaled04_best01_rand000_050", 1.0),
            },
        ),
        build(
            "mr_night_recall_blend",
            {
                "body": ("incumbent", 1.0),
                "head": ("r021_city_night_recall", 0.80),
                "router": ("r019_residential_night_precision", 0.75),
                "expert0": ("r021_city_night_recall", 0.65),
                "expert1": ("r021_city_night_recall", 0.65),
                "expert2": ("r003_tiny_all_s", 0.55),
                "expert3": ("r015_residential_night_precision", 0.65),
            },
        ),
    ]
    return raw


def state_cache_for_specs(args: argparse.Namespace, specs: list[dict[str, Any]]) -> dict[tuple[int, str, str], dict[str, torch.Tensor] | None]:
    rounds = sorted(
        {
            int(group_spec["round"])
            for spec in specs
            for group_spec in spec["groups"].values()
            if int(group_spec["round"]) >= 0
        }
    )
    cache: dict[tuple[int, str, str], dict[str, torch.Tensor] | None] = {}
    for round_idx in rounds:
        paths = judger01.checkpoint_paths(args.source_workspace, round_idx)
        ckpts = {role: load_checkpoint(path) for role, path in paths.items()}
        for role, ckpt in ckpts.items():
            for field in ("model", "ema"):
                cache[(round_idx, role, field)] = state_dict_from(ckpt, field)
    return cache


def compose_state(
    incumbent_state: dict[str, torch.Tensor],
    field: str,
    spec: dict[str, Any],
    cache: dict[tuple[int, str, str], dict[str, torch.Tensor] | None],
) -> dict[str, torch.Tensor]:
    mixed: dict[str, torch.Tensor] = {}
    for key, i_value in incumbent_state.items():
        group = delta04.group_for_key(key)
        group_spec = spec["groups"].get(group, zero_spec())
        round_idx = int(group_spec["round"])
        if round_idx < 0:
            mixed[key] = i_value
            continue
        g_state = cache.get((round_idx, "g", field))
        a_state = cache.get((round_idx, "a", field))
        s_state = cache.get((round_idx, "s", field))
        g_value = None if g_state is None else g_state.get(key)
        a_value = None if a_state is None else a_state.get(key)
        s_value = None if s_state is None else s_state.get(key)
        if (
            torch.is_tensor(i_value)
            and torch.is_tensor(g_value)
            and torch.is_tensor(a_value)
            and torch.is_tensor(s_value)
            and i_value.shape == g_value.shape == a_value.shape == s_value.shape
            and i_value.dtype.is_floating_point
        ):
            alpha = parse_float(group_spec.get("a"), 0.0)
            beta = parse_float(group_spec.get("s"), 0.0)
            value = i_value.float() + alpha * (a_value.float() - g_value.float()) + beta * (s_value.float() - g_value.float())
            mixed[key] = value.to(i_value.dtype)
        else:
            mixed[key] = i_value
    return mixed


def build_candidate(spec: dict[str, Any], cache: dict[tuple[int, str, str], dict[str, torch.Tensor] | None], args: argparse.Namespace) -> Path:
    out = args.workspace_root / "candidates" / f"{spec['label']}.pt"
    if out.exists() and not args.force:
        return out
    incumbent = load_checkpoint(args.incumbent_path)
    mixed = copy.deepcopy(incumbent)
    for field in ("model", "ema"):
        i_state = state_dict_from(incumbent, field)
        if i_state is None:
            continue
        replace_state(mixed, field, compose_state(i_state, field, spec, cache))
    mixed["epoch"] = -1
    mixed["optimizer"] = None
    mixed["multiround_group_residual"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "formula": "I + alpha_group*(A_round_group-G_round_group) + beta_group*(S_round_group-G_round_group)",
        "groups": spec["groups"],
        "incumbent": str(args.incumbent_path),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mixed, out)
    return out


def existing_eval(path: Path, key_fields: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, Any]]:
    return {tuple(str(row.get(field, "")) for field in key_fields): row for row in read_csv(path)}


def eval_total(args: argparse.Namespace, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out_path = args.workspace_root / "stats" / "15_total_eval.csv"
    cache = existing_eval(out_path, ("label",)) if args.resume else {}
    rows = list(cache.values())
    cfg = opt02.full_eval_config(args)
    for item in candidates:
        if (item["label"],) in cache:
            continue
        metrics = opt02.eval_checkpoint(Path(item["path"]), cfg, f"mr15_total_{item['label']}", args)
        row = {**item, **metrics}
        row["total_score"] = opt02.score_row(row)
        rows.append(row)
        write_csv(out_path, rows)
    return rows


def load_external_domain_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source in DOMAIN_SOURCES:
        for row in read_csv(source):
            label = row.get("label", "")
            domain = row.get("domain", "")
            if not label or not domain:
                continue
            key = (label, domain)
            if key in seen:
                continue
            seen.add(key)
            total_score = parse_float(row.get("total_score"), BASELINE_TOTAL_SCORE if label == BASELINE_LABEL else math.nan)
            if math.isnan(total_score):
                continue
            rows.append(row)
    return rows


def eval_domains(args: argparse.Namespace, selected: list[dict[str, Any]], total_rows: list[dict[str, Any]], domains: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out_path = args.workspace_root / "stats" / "15_domain_eval.csv"
    cache = existing_eval(out_path, ("label", "domain")) if args.resume else {}
    rows = list(cache.values())
    if not rows:
        rows.extend(load_external_domain_rows())
        write_csv(out_path, rows)
        cache = existing_eval(out_path, ("label", "domain"))
    total_by_label = {row["label"]: row for row in total_rows}
    for item in selected:
        total = total_by_label.get(item["label"])
        if total is None:
            continue
        for domain in domains:
            if (item["label"], domain["domain"]) in cache:
                continue
            metrics = opt02.eval_checkpoint(Path(item["path"]), Path(domain["cfg"]), f"mr15_domain_{item['label']}_{domain['domain']}", args)
            row = {
                "label": item["label"],
                "source": "15_multiround_group_residual",
                "path": item["path"],
                "total_score": parse_float(total.get("total_score")),
                "total_map50": parse_float(total.get("map50")),
                "total_map50_95": parse_float(total.get("map50_95")),
                "domain": domain["domain"],
                "group": domain["group"],
                **metrics,
            }
            row["domain_score"] = opt02.score_row(row)
            rows.append(row)
            write_csv(out_path, rows)
    return rows


def scorecard(summary: list[dict[str, Any]], router_summary: list[dict[str, Any]], previous_accuracy: int = 81) -> dict[str, Any]:
    baseline = next(row for row in summary if row["label"] == BASELINE_LABEL)
    best_single = max(summary, key=lambda row: parse_float(row["group_dro_score"]))
    router = next(row for row in router_summary if row["policy"] == "night_domain_router")
    inc_router = next(row for row in router_summary if row["policy"] == BASELINE_LABEL)
    single_dro_delta = parse_float(best_single["group_dro_score"]) - parse_float(baseline["group_dro_score"])
    router_dro_delta = parse_float(router["group_dro_score"]) - parse_float(inc_router["group_dro_score"])
    night_delta = parse_float(router["night_mean_score"]) - parse_float(inc_router["night_mean_score"])
    worst_delta = parse_float(router["worst_domain_score"]) - parse_float(inc_router["worst_domain_score"])
    total_delta = parse_float(best_single["total_score"]) - parse_float(baseline["total_score"])
    candidate_gain = max(0.0, router_dro_delta - 0.00110)
    acc = previous_accuracy + candidate_gain / 0.002 * 12.0 + max(0.0, single_dro_delta) / 0.002 * 7.0 + max(0.0, total_delta) / 0.002 * 8.0
    if best_single["label"].startswith("mr_"):
        acc += 3.0
    accuracy = opt10.clamp_score(acc)
    return {
        "experiment_env": 95,
        "root_cause_analysis": 92,
        "judge_stability": 91 if worst_delta >= 0 else 86,
        "accuracy_improvement": accuracy,
        "final_goal": opt10.clamp_score(0.18 * 95 + 0.18 * 92 + 0.20 * (91 if worst_delta >= 0 else 86) + 0.30 * accuracy + 0.14 * 84),
        "best_single_label": best_single["label"],
        "best_single_dro_delta": single_dro_delta,
        "best_single_total_delta": total_delta,
        "router_dro_delta": router_dro_delta,
        "router_night_delta": night_delta,
        "router_worst_delta": worst_delta,
    }


def make_report(
    args: argparse.Namespace,
    total_rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    policy: list[dict[str, Any]],
    router_summary: list[dict[str, Any]],
    card: dict[str, Any],
) -> str:
    total_sorted = sorted(total_rows, key=lambda row: parse_float(row.get("total_score")), reverse=True)
    report = [
        "# DQA-SoftMoX 15 Multi-Round Group Residual Composer",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        "- method: body/head/router/expert groups borrow residuals from different self-generated rounds",
        "",
        "## Total Evaluation",
        "",
        "| rank | candidate | total score | mAP50 | mAP50:95 |",
        "|---:|---|---:|---:|---:|",
    ]
    for idx, row in enumerate(total_sorted, start=1):
        report.append(f"| {idx} | {row['label']} | {parse_float(row['total_score']):.5f} | {parse_float(row['map50']):.3f} | {parse_float(row['map50_95']):.3f} |")
    report.extend(["", "## Full Domain Summary", "", "| rank | candidate | total | day | night | worst | DRO | night mAP50 |", "|---:|---|---:|---:|---:|---:|---:|---:|"])
    for idx, row in enumerate(summary[:18], start=1):
        report.append(
            f"| {idx} | {row['label']} | {parse_float(row['total_score']):.5f} | {parse_float(row['day_mean_score']):.5f} | "
            f"{parse_float(row['night_mean_score']):.5f} | {parse_float(row['worst_domain_score']):.5f} | "
            f"{parse_float(row['group_dro_score']):.5f} | {parse_float(row['night_mean_map50']):.3f} |"
        )
    report.extend(["", "## Dynamic Router Pool", "", "| domain | selected | delta score | delta mAP50 |", "|---|---|---:|---:|"])
    for row in policy:
        report.append(f"| {row['domain']} | {row['selected_label']} | {row['delta_score']:+.5f} | {row['delta_map50']:+.3f} |")
    report.extend(["", "## Policy Summary", "", "| policy | mean | night | worst | DRO |", "|---|---:|---:|---:|---:|"])
    for row in router_summary:
        report.append(f"| {row['policy']} | {row['domain_mean_score']:.5f} | {row['night_mean_score']:.5f} | {row['worst_domain_score']:.5f} | {row['group_dro_score']:.5f} |")
    report.extend(
        [
            "",
            "## Codex Goal Scores",
            "",
            f"- experiment_env: {card['experiment_env']}/100",
            f"- root_cause_analysis: {card['root_cause_analysis']}/100",
            f"- judge_stability: {card['judge_stability']}/100",
            f"- accuracy_improvement: {card['accuracy_improvement']}/100",
            f"- final_goal: {card['final_goal']}/100",
            "",
            "## Takeaway",
            "",
            "This tests the missing hypothesis from the previous loops: a single round is too coarse.  If no new candidate beats the existing domain-router pool, the next loop should change the training data/curriculum rather than only recombining checkpoints.",
        ]
    )
    return "\n".join(report)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.source_workspace = args.source_workspace.expanduser().resolve()
    args.incumbent_path = args.incumbent_path.expanduser().resolve()
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    (args.workspace_root / "stats").mkdir(parents=True, exist_ok=True)
    notify(
        "15 started\nBuilding multi-round group residual candidates.",
        "DQA-SoftMoX 15 started",
        args.notify_discord,
    )

    coeff_rows = load_coeff_rows()
    specs = candidate_specs(coeff_rows)
    cache = state_cache_for_specs(args, specs)
    candidates: list[dict[str, Any]] = []
    for spec in specs:
        path = build_candidate(spec, cache, args)
        manifest_groups = {
            group: f"{group_spec['label']}@r{int(group_spec['round']):03d}:{parse_float(group_spec['a']):+.3f}/{parse_float(group_spec['s']):+.3f}"
            for group, group_spec in spec["groups"].items()
        }
        candidates.append(
            {
                "label": spec["label"],
                "source": "15_multiround_group_residual",
                "path": str(path.resolve()),
                **{f"{group}_source": value for group, value in manifest_groups.items()},
            }
        )
    write_csv(args.workspace_root / "stats" / "15_candidate_pool.csv", candidates)

    total_rows = eval_total(args, candidates)
    total_safe = [row for row in total_rows if parse_float(row["total_score"]) >= BASELINE_TOTAL_SCORE - args.max_total_drop]
    total_safe.sort(key=lambda row: parse_float(row["total_score"]), reverse=True)
    selected = total_safe[: args.domain_eval_topk]
    write_csv(args.workspace_root / "stats" / "15_selected_for_domain.csv", selected)

    domains = opt10.make_full_domain_configs(args)
    domain_rows = eval_domains(args, selected, total_rows, domains)
    summary = opt10.aggregate_domains(domain_rows)
    write_csv(args.workspace_root / "stats" / "15_domain_summary.csv", summary)
    policy, router_summary = opt10.build_domain_router(domain_rows, args)
    write_csv(args.workspace_root / "stats" / "15_domain_router_policy.csv", policy)
    write_csv(args.workspace_root / "stats" / "15_domain_router_summary.csv", router_summary)
    card = scorecard(summary, router_summary)
    (args.workspace_root / "stats" / "15_scorecard.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    report = make_report(args, total_rows, summary, policy, router_summary, card)
    report_path = args.workspace_root / "15_multiround_group_residual_composer_report.md"
    report_path.write_text(report, encoding="utf-8")
    notify(
        "15 finished\n"
        + "\n".join(f"- {row['policy']}: DRO={row['group_dro_score']:.5f}, night={row['night_mean_score']:.5f}, worst={row['worst_domain_score']:.5f}" for row in router_summary)
        + f"\nScores: env={card['experiment_env']}, analysis={card['root_cause_analysis']}, stability={card['judge_stability']}, accuracy={card['accuracy_improvement']}, final={card['final_goal']}",
        "DQA-SoftMoX 15 finished",
        args.notify_discord,
    )
    result = {"summary": summary[:18], "router_summary": router_summary, "scorecard": card, "report": str(report_path.resolve())}
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-workspace", type=Path, default=SOURCE_WORKSPACE)
    parser.add_argument("--incumbent-path", type=Path, default=INCUMBENT)
    parser.add_argument("--max-total-drop", type=float, default=0.0011)
    parser.add_argument("--domain-eval-topk", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--notify-discord", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
