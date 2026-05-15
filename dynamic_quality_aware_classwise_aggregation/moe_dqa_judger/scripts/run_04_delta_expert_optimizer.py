#!/usr/bin/env python3
"""Optimize anchored delta mixtures for DQA-SoftMoX.

Notebook 03 learned a selector over convex G/A/S mixtures.  This loop explores a
more expressive but still safe family inspired by FedLAW/AdaMerging/FedMoE:

    M = G + alpha * (A - G) + beta * (S - G)

where alpha and beta are learned separately for:

    body, head_core, router, expert0, expert1, expert2, expert3

The previous global is always the anchor.  This lets the judge insert target
aggregate and server-repair deltas without blindly replacing the parent model.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "output" / "04_delta_expert_optimizer"
SOURCE_WORKSPACE = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation" / "moe_dqa" / "output" / "01_dqa_fedmox_yolo_full"

if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_01_judger_probe as judger01  # noqa: E402
import run_02_mix_weight_optimizer as opt02  # noqa: E402


GROUPS = ["body", "head", "router", "expert0", "expert1", "expert2", "expert3"]
COEFF_FIELDS = [f"{group}_{src}" for group in GROUPS for src in ("a", "s")]
EXPERT_RE = re.compile(r"^head\.expert_m\.\d+\.(\d+)\.")


def notify(message: str, title: str, enabled: bool) -> None:
    if not enabled:
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from notebook_notify import notify_discord

        notify_discord(message, title=title, fail_silently=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notify skipped: {exc}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_rounds(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def group_for_key(key: str) -> str:
    if "head.router" in key:
        return "router"
    match = EXPERT_RE.match(key)
    if match:
        return f"expert{int(match.group(1))}"
    if key.startswith("head."):
        return "head"
    return "body"


def coeff_vector(coeffs: dict[str, float]) -> list[float]:
    return [float(coeffs[name]) for name in COEFF_FIELDS]


def clean_coeffs(raw: dict[str, float], low: float = -0.25, high: float = 1.25) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in COEFF_FIELDS:
        out[name] = min(high, max(low, float(raw.get(name, 0.0))))
    return out


def coeff_from_group_values(values: dict[str, tuple[float, float]]) -> dict[str, float]:
    coeffs: dict[str, float] = {}
    for group in GROUPS:
        alpha, beta = values.get(group, values.get("default", (0.0, 0.0)))
        coeffs[f"{group}_a"] = alpha
        coeffs[f"{group}_s"] = beta
    return clean_coeffs(coeffs)


def fixed_delta_priors() -> list[tuple[str, dict[str, float]]]:
    specs = [
        ("freeze_g", {"default": (0.0, 0.0)}),
        ("pure_a_delta", {"default": (1.0, 0.0)}),
        ("pure_s_delta", {"default": (0.0, 1.0)}),
        (
            "role_split_v1",
            {
                "body": (0.25, 0.10),
                "head": (0.10, 0.70),
                "router": (0.75, 0.10),
                "expert0": (0.75, 0.10),
                "expert1": (0.75, 0.10),
                "expert2": (0.75, 0.10),
                "expert3": (0.75, 0.10),
            },
        ),
        (
            "repair_head_aggregate_moe",
            {
                "body": (0.10, 0.05),
                "head": (0.05, 0.85),
                "router": (0.80, 0.05),
                "expert0": (0.80, 0.05),
                "expert1": (0.80, 0.05),
                "expert2": (0.80, 0.05),
                "expert3": (0.80, 0.05),
            },
        ),
        (
            "conservative_delta",
            {
                "body": (0.15, 0.00),
                "head": (0.05, 0.35),
                "router": (0.35, 0.00),
                "expert0": (0.35, 0.00),
                "expert1": (0.35, 0.00),
                "expert2": (0.35, 0.00),
                "expert3": (0.35, 0.00),
            },
        ),
        (
            "expert_repair_head",
            {
                "body": (0.00, 0.00),
                "head": (0.10, 0.70),
                "router": (0.40, 0.20),
                "expert0": (0.65, 0.10),
                "expert1": (0.65, 0.10),
                "expert2": (0.25, 0.40),
                "expert3": (0.25, 0.40),
            },
        ),
        (
            "anti_source_drift",
            {
                "body": (0.00, -0.05),
                "head": (0.00, 0.15),
                "router": (0.20, 0.00),
                "expert0": (0.20, 0.00),
                "expert1": (0.20, 0.00),
                "expert2": (0.20, 0.00),
                "expert3": (0.20, 0.00),
            },
        ),
        (
            "slight_extrapolate_target",
            {
                "body": (1.05, -0.05),
                "head": (0.30, 0.55),
                "router": (1.10, -0.05),
                "expert0": (1.10, -0.05),
                "expert1": (1.10, -0.05),
                "expert2": (0.90, 0.05),
                "expert3": (0.90, 0.05),
            },
        ),
    ]
    return [(name, coeff_from_group_values(spec)) for name, spec in specs]


def sample_around(base: dict[str, float], rng: random.Random, sigma: float) -> dict[str, float]:
    return clean_coeffs({name: float(base[name]) + rng.gauss(0.0, sigma) for name in COEFF_FIELDS})


def state_dict_from(ckpt: dict[str, Any], field: str) -> dict[str, torch.Tensor] | None:
    obj = ckpt.get(field)
    if obj is None:
        return None
    if hasattr(obj, "float"):
        obj = obj.float()
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if isinstance(obj, dict):
        return obj
    return None


def replace_state(ckpt: dict[str, Any], field: str, state: dict[str, torch.Tensor]) -> None:
    obj = ckpt.get(field)
    if obj is None:
        return
    if hasattr(obj, "float"):
        obj = obj.float()
    if hasattr(obj, "load_state_dict"):
        obj.load_state_dict(state, strict=False)
        ckpt[field] = obj.half() if hasattr(obj, "half") else obj
    else:
        ckpt[field] = state


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def mix_delta_state(
    g_state: dict[str, torch.Tensor],
    a_state: dict[str, torch.Tensor],
    s_state: dict[str, torch.Tensor],
    coeffs: dict[str, float],
) -> dict[str, torch.Tensor]:
    mixed: dict[str, torch.Tensor] = {}
    for key, g_value in g_state.items():
        a_value = a_state.get(key)
        s_value = s_state.get(key)
        if (
            torch.is_tensor(g_value)
            and torch.is_tensor(a_value)
            and torch.is_tensor(s_value)
            and g_value.shape == a_value.shape == s_value.shape
            and g_value.dtype.is_floating_point
        ):
            group = group_for_key(key)
            alpha = float(coeffs[f"{group}_a"])
            beta = float(coeffs[f"{group}_s"])
            value = g_value.float() + alpha * (a_value.float() - g_value.float()) + beta * (s_value.float() - g_value.float())
            mixed[key] = value.to(g_value.dtype)
        else:
            mixed[key] = g_value
    return mixed


def build_delta_checkpoint(round_idx: int, coeffs: dict[str, float], candidate_id: str, args: argparse.Namespace) -> Path:
    paths = judger01.checkpoint_paths(args.source_workspace, round_idx)
    out = args.workspace_root / "candidates" / f"r{round_idx:03d}_{candidate_id}.pt"
    if out.exists() and not args.force:
        return out
    g_ckpt = load_checkpoint(paths["g"])
    a_ckpt = load_checkpoint(paths["a"])
    s_ckpt = load_checkpoint(paths["s"])
    mixed = copy.deepcopy(g_ckpt)
    for field in ("model", "ema"):
        g_state = state_dict_from(g_ckpt, field)
        a_state = state_dict_from(a_ckpt, field)
        s_state = state_dict_from(s_ckpt, field)
        if g_state is None or a_state is None or s_state is None:
            continue
        replace_state(mixed, field, mix_delta_state(g_state, a_state, s_state, coeffs))
    mixed["epoch"] = -1
    mixed["optimizer"] = None
    mixed["judger_delta_mix"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "round": round_idx,
        "coefficients": coeffs,
        "formula": "G + alpha*(A-G) + beta*(S-G)",
        "groups": GROUPS,
        "sources": {key: str(value) for key, value in paths.items()},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mixed, out)
    return out


def make_candidates(round_idx: int, args: argparse.Namespace, observed: list[dict[str, Any]]) -> list[tuple[str, dict[str, float], str]]:
    rng = random.Random(args.seed + round_idx * 37)
    priors = fixed_delta_priors()
    candidates = [(name, coeffs, "prior") for name, coeffs in priors]
    best_rows = sorted(observed, key=lambda row: float(row.get("score", 0.0)), reverse=True)[: args.observed_templates]
    templates = [coeff_from_group_values({"default": (0.0, 0.0)})] + [coeffs for _name, coeffs in priors]
    templates.extend({name: float(row[name]) for name in COEFF_FIELDS} for row in best_rows if all(name in row for name in COEFF_FIELDS))
    for idx in range(args.random_candidates):
        base = templates[idx % len(templates)]
        sigma = args.sample_sigma * (1.5 if idx % 4 == 0 else 1.0)
        candidates.append((f"rand{idx:03d}", sample_around(base, rng, sigma), "random"))
    return candidates


def evaluate_candidate(round_idx: int, candidate_id: str, coeffs: dict[str, float], phase: str, cfg: Path, scope: str, args: argparse.Namespace) -> dict[str, Any]:
    ckpt = build_delta_checkpoint(round_idx, coeffs, candidate_id, args)
    metrics = opt02.eval_checkpoint(ckpt, cfg, f"r{round_idx:03d}_{candidate_id}_{scope}", args)
    features = judger01.round_features(args.source_workspace, round_idx)
    row = {
        "round": round_idx,
        "candidate_id": candidate_id,
        "phase": phase,
        "eval_scope": scope,
        "path": str(ckpt.resolve()),
        **features,
        **coeffs,
        **metrics,
    }
    row["score"] = opt02.score_row(row)
    if not args.keep_all_candidates and scope == "mini":
        try:
            ckpt.unlink()
        except OSError:
            pass
    return row


def optimise_round(round_idx: int, args: argparse.Namespace, mini_cfg: Path, full_cfg: Path, all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    round_rows: list[dict[str, Any]] = []
    candidates = make_candidates(round_idx, args, all_rows)
    seen: set[tuple[float, ...]] = set()
    unique: list[tuple[str, dict[str, float], str]] = []
    for candidate_id, coeffs, phase in candidates:
        key = tuple(round(float(coeffs[name]), 4) for name in COEFF_FIELDS)
        if key in seen:
            continue
        seen.add(key)
        unique.append((candidate_id, coeffs, phase))

    for candidate_id, coeffs, phase in unique:
        row = evaluate_candidate(round_idx, candidate_id, coeffs, phase, mini_cfg, "mini", args)
        round_rows.append(row)
        all_rows.append(row)

    for iteration in range(args.surrogate_iterations):
        train = [row for row in round_rows if row.get("eval_scope") == "mini"]
        x = np.asarray([coeff_vector({name: float(row[name]) for name in COEFF_FIELDS}) for row in train], dtype=np.float64)
        y = np.asarray([float(row["score"]) for row in train], dtype=np.float64)
        model = RandomForestRegressor(n_estimators=220, max_depth=7, random_state=args.seed + round_idx * 100 + iteration, n_jobs=-1)
        model.fit(x, y)
        rng = random.Random(args.seed + round_idx * 1000 + iteration)
        base_rows = sorted(train, key=lambda row: float(row["score"]), reverse=True)[: max(2, args.surrogate_base_topk)]
        pool: list[tuple[float, dict[str, float]]] = []
        for idx in range(args.surrogate_pool):
            base = {name: float(base_rows[idx % len(base_rows)][name]) for name in COEFF_FIELDS}
            coeffs = sample_around(base, rng, args.sample_sigma * 0.7)
            pred = float(model.predict(np.asarray([coeff_vector(coeffs)], dtype=np.float64))[0])
            pool.append((pred, coeffs))
        pool.sort(key=lambda item: item[0], reverse=True)
        for idx, (_pred, coeffs) in enumerate(pool[: args.surrogate_evals]):
            row = evaluate_candidate(round_idx, f"sur{iteration:02d}_{idx:02d}", coeffs, f"surrogate{iteration}", mini_cfg, "mini", args)
            round_rows.append(row)
            all_rows.append(row)

    ranked = sorted([row for row in round_rows if row.get("eval_scope") == "mini"], key=lambda row: float(row["score"]), reverse=True)
    full_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ranked[: args.full_eval_topk]):
        coeffs = {name: float(row[name]) for name in COEFF_FIELDS}
        full = evaluate_candidate(round_idx, f"best{idx:02d}_{row['candidate_id']}", coeffs, "full", full_cfg, "full_total", args)
        full_rows.append(full)
        all_rows.append(full)

    best = max(full_rows, key=lambda row: float(row["score"])) if full_rows else ranked[0]
    notify(
        "Delta expert optimizer round {round} done\nbest={candidate_id}\nmAP50={map50:.3f}, mAP50:95={map50_95:.3f}, score={score:.4f}\n"
        "body a/s={body_a:.2f}/{body_s:.2f}, head={head_a:.2f}/{head_s:.2f}, router={router_a:.2f}/{router_s:.2f}".format(**best),
        "DQA-SoftMoX 04 round result",
        args.notify_discord,
    )
    return round_rows + full_rows


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.source_workspace = args.source_workspace.expanduser().resolve()
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    (args.workspace_root / "stats").mkdir(parents=True, exist_ok=True)
    mini_list, mini_cfg = opt02.ensure_mini_eval_config(args)
    full_cfg = opt02.full_eval_config(args)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "dqa_softmox_delta_expert_optimizer_v0",
        "method": "anchored delta mix by body/head/router/expert groups",
        "rounds": parse_rounds(args.rounds),
        "mini_images": args.mini_images,
        "source_workspace": str(args.source_workspace),
        "workspace": str(args.workspace_root),
    }
    (args.workspace_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    notify(
        "Delta expert optimizer started\nrounds={rounds}\nmini_images={mini_images}\nrandom={random_candidates}, surrogate={surrogate_iterations}x{surrogate_evals}".format(**manifest, random_candidates=args.random_candidates, surrogate_iterations=args.surrogate_iterations, surrogate_evals=args.surrogate_evals),
        "DQA-SoftMoX 04 started",
        args.notify_discord,
    )

    rows: list[dict[str, Any]] = []
    for round_idx in parse_rounds(args.rounds):
        paths = judger01.checkpoint_paths(args.source_workspace, round_idx)
        if not all(path.exists() for path in paths.values()):
            print(f"skip round {round_idx}; missing checkpoint")
            continue
        optimise_round(round_idx, args, mini_cfg, full_cfg, rows)
        write_csv(args.workspace_root / "stats" / "04_delta_expert_trials.csv", rows)

    full_rows = [row for row in rows if row.get("eval_scope") == "full_total"]
    ranked = sorted(full_rows, key=lambda row: float(row.get("score", 0.0)), reverse=True)
    write_csv(args.workspace_root / "stats" / "04_delta_expert_best_full.csv", ranked)
    report = [
        "# DQA-SoftMoX Delta Expert Optimizer 04",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- mini_images: {args.mini_images}",
        "",
        "| rank | round | candidate | mAP50 | mAP50:95 | score | body a/s | head a/s | router a/s | expert0 a/s | expert1 a/s | expert2 a/s | expert3 a/s |",
        "|---:|---:|---|---:|---:|---:|---|---|---|---|---|---|---|",
    ]
    for idx, row in enumerate(ranked[:30], start=1):
        report.append(
            "| {idx} | {round} | {candidate_id} | {map50:.3f} | {map50_95:.3f} | {score:.4f} | "
            "{body_a:.2f}/{body_s:.2f} | {head_a:.2f}/{head_s:.2f} | {router_a:.2f}/{router_s:.2f} | "
            "{expert0_a:.2f}/{expert0_s:.2f} | {expert1_a:.2f}/{expert1_s:.2f} | {expert2_a:.2f}/{expert2_s:.2f} | {expert3_a:.2f}/{expert3_s:.2f} |".format(idx=idx, **row)
        )
    (args.workspace_root / "04_delta_expert_optimizer_report.md").write_text("\n".join(report), encoding="utf-8")
    notify(
        "Delta expert optimizer finished\nbest full candidates:\n" + "\n".join(
            f"- r{int(row['round'])}: {row['candidate_id']} mAP50={float(row['map50']):.3f}, mAP50:95={float(row['map50_95']):.3f}, score={float(row['score']):.4f}"
            for row in ranked[:5]
        ),
        "DQA-SoftMoX 04 finished",
        args.notify_discord,
    )
    print(json.dumps(ranked[:10], indent=2, ensure_ascii=False))
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-workspace", type=Path, default=SOURCE_WORKSPACE)
    parser.add_argument("--rounds", default="1,2,3,4,5,6")
    parser.add_argument("--mini-images", type=int, default=512)
    parser.add_argument("--random-candidates", type=int, default=10)
    parser.add_argument("--observed-templates", type=int, default=10)
    parser.add_argument("--surrogate-iterations", type=int, default=1)
    parser.add_argument("--surrogate-pool", type=int, default=72)
    parser.add_argument("--surrogate-evals", type=int, default=3)
    parser.add_argument("--surrogate-base-topk", type=int, default=4)
    parser.add_argument("--full-eval-topk", type=int, default=2)
    parser.add_argument("--sample-sigma", type=float, default=0.18)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-all-candidates", action="store_true")
    parser.add_argument("--notify-discord", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
