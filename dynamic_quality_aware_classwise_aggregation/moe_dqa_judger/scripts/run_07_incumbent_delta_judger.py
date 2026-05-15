#!/usr/bin/env python3
"""Incumbent-rebased delta judger for DQA-SoftMoX.

06 showed that the judge can recognize the best historical candidates, but the
available round sequence still drifts after round 2.  This experiment keeps the
best known global checkpoint as an incumbent and applies only small late-round
DQA/FedMoX deltas onto it:

    M_t = I_best + alpha * (A_t - G_t) + beta * (S_t - G_t)

The coefficients are group-wise (body/head/router/expert0..3).  This asks a
more surgical question: do later rounds contain any useful MoE/head/body update
direction once we remove the inherited drift from their parent checkpoint?
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "output" / "07_incumbent_delta_judger"
SOURCE_WORKSPACE = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation" / "moe_dqa" / "output" / "01_dqa_fedmox_yolo_full"
DEFAULT_INCUMBENT = PROJECT_ROOT / "output" / "03_mix_judger_policy" / "candidates" / "r002_judger03_selected_r002.pt"
DEFAULT_JUDGER = PROJECT_ROOT / "output" / "06_robust_proxy_judger" / "robust_proxy_judger_v1.joblib"

if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_01_judger_probe as judger01  # noqa: E402
import run_02_mix_weight_optimizer as opt02  # noqa: E402
import run_04_delta_expert_optimizer as delta04  # noqa: E402
import run_06_robust_proxy_judger as robust06  # noqa: E402


GROUPS = delta04.GROUPS
COEFF_FIELDS = delta04.COEFF_FIELDS


def notify(message: str, title: str, enabled: bool) -> None:
    if not enabled:
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from notebook_notify import notify_discord

        notify_discord(message, title=title, fail_silently=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notify skipped: {exc}")


def parse_rounds(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def coeff_from_pairs(pairs: dict[str, tuple[float, float]], scale: float = 1.0) -> dict[str, float]:
    raw: dict[str, float] = {}
    for group in GROUPS:
        a, s = pairs.get(group, pairs.get("default", (0.0, 0.0)))
        raw[f"{group}_a"] = a * scale
        raw[f"{group}_s"] = s * scale
    return delta04.clean_coeffs(raw, low=-0.20, high=0.55)


def scale_coeffs(coeffs: dict[str, float], scale: float) -> dict[str, float]:
    return delta04.clean_coeffs({name: parse_float(coeffs.get(name), 0.0) * scale for name in COEFF_FIELDS}, low=-0.20, high=0.55)


def fixed_priors() -> list[tuple[str, dict[str, float]]]:
    specs = [
        ("tiny_all_a", {"default": (0.06, 0.00)}),
        ("tiny_all_s", {"default": (0.00, 0.06)}),
        ("moe_a_only", {"body": (0.0, 0.0), "head": (0.0, 0.0), "router": (0.16, 0.0), "expert0": (0.18, 0.0), "expert1": (0.18, 0.0), "expert2": (0.14, 0.0), "expert3": (0.14, 0.0)}),
        ("head_s_moe_a", {"body": (0.0, 0.0), "head": (0.0, 0.14), "router": (0.14, 0.0), "expert0": (0.18, 0.0), "expert1": (0.18, 0.0), "expert2": (0.12, 0.02), "expert3": (0.12, 0.02)}),
        ("anti_drift_head", {"body": (0.0, -0.04), "head": (0.02, 0.14), "router": (0.10, -0.02), "expert0": (0.10, 0.0), "expert1": (0.10, 0.0), "expert2": (0.08, 0.02), "expert3": (0.08, 0.02)}),
        ("body_frozen_head_moe", {"body": (0.0, 0.0), "head": (0.04, 0.12), "router": (0.12, 0.02), "expert0": (0.14, 0.02), "expert1": (0.14, 0.02), "expert2": (0.10, 0.04), "expert3": (0.10, 0.04)}),
        ("source_repair_only_head", {"body": (0.0, 0.0), "head": (0.0, 0.20), "router": (0.0, 0.0), "expert0": (0.0, 0.0), "expert1": (0.0, 0.0), "expert2": (0.0, 0.0), "expert3": (0.0, 0.0)}),
        ("target_router_only", {"body": (0.0, 0.0), "head": (0.0, 0.0), "router": (0.25, 0.0), "expert0": (0.0, 0.0), "expert1": (0.0, 0.0), "expert2": (0.0, 0.0), "expert3": (0.0, 0.0)}),
    ]
    return [(name, coeff_from_pairs(spec)) for name, spec in specs]


def round04_templates(round_idx: int, topk: int = 2) -> list[tuple[str, dict[str, float]]]:
    path = PROJECT_ROOT / "output" / "04_delta_expert_optimizer" / "stats" / "04_delta_expert_best_full.csv"
    rows = [row for row in read_csv(path) if int(float(row.get("round", -1))) == round_idx]
    rows.sort(key=lambda row: parse_float(row.get("score"), -1.0), reverse=True)
    out: list[tuple[str, dict[str, float]]] = []
    for row in rows[:topk]:
        coeffs = {name: parse_float(row.get(name), 0.0) for name in COEFF_FIELDS}
        out.append((f"scaled04_{row.get('candidate_id', 'candidate')}_025", scale_coeffs(coeffs, 0.25)))
        out.append((f"scaled04_{row.get('candidate_id', 'candidate')}_050", scale_coeffs(coeffs, 0.50)))
    return out


def sample_around(base: dict[str, float], rng: random.Random, sigma: float) -> dict[str, float]:
    return delta04.clean_coeffs({name: parse_float(base.get(name), 0.0) + rng.gauss(0.0, sigma) for name in COEFF_FIELDS}, low=-0.20, high=0.55)


def state_dict_from(ckpt: dict[str, Any], field: str) -> dict[str, torch.Tensor] | None:
    return delta04.state_dict_from(ckpt, field)


def replace_state(ckpt: dict[str, Any], field: str, state: dict[str, torch.Tensor]) -> None:
    delta04.replace_state(ckpt, field, state)


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def rebase_delta_state(
    incumbent_state: dict[str, torch.Tensor],
    g_state: dict[str, torch.Tensor],
    a_state: dict[str, torch.Tensor],
    s_state: dict[str, torch.Tensor],
    coeffs: dict[str, float],
) -> dict[str, torch.Tensor]:
    mixed: dict[str, torch.Tensor] = {}
    for key, i_value in incumbent_state.items():
        g_value = g_state.get(key)
        a_value = a_state.get(key)
        s_value = s_state.get(key)
        if (
            torch.is_tensor(i_value)
            and torch.is_tensor(g_value)
            and torch.is_tensor(a_value)
            and torch.is_tensor(s_value)
            and i_value.shape == g_value.shape == a_value.shape == s_value.shape
            and i_value.dtype.is_floating_point
        ):
            group = delta04.group_for_key(key)
            alpha = parse_float(coeffs.get(f"{group}_a"), 0.0)
            beta = parse_float(coeffs.get(f"{group}_s"), 0.0)
            value = i_value.float() + alpha * (a_value.float() - g_value.float()) + beta * (s_value.float() - g_value.float())
            mixed[key] = value.to(i_value.dtype)
        else:
            mixed[key] = i_value
    return mixed


def build_rebased_checkpoint(round_idx: int, coeffs: dict[str, float], candidate_id: str, args: argparse.Namespace) -> Path:
    out = args.workspace_root / "candidates" / f"r{round_idx:03d}_{candidate_id}.pt"
    if out.exists() and not args.force:
        return out
    paths = judger01.checkpoint_paths(args.source_workspace, round_idx)
    incumbent = load_checkpoint(args.incumbent_path)
    g_ckpt = load_checkpoint(paths["g"])
    a_ckpt = load_checkpoint(paths["a"])
    s_ckpt = load_checkpoint(paths["s"])
    mixed = copy.deepcopy(incumbent)
    for field in ("model", "ema"):
        i_state = state_dict_from(incumbent, field)
        g_state = state_dict_from(g_ckpt, field)
        a_state = state_dict_from(a_ckpt, field)
        s_state = state_dict_from(s_ckpt, field)
        if i_state is None or g_state is None or a_state is None or s_state is None:
            continue
        replace_state(mixed, field, rebase_delta_state(i_state, g_state, a_state, s_state, coeffs))
    mixed["epoch"] = -1
    mixed["optimizer"] = None
    mixed["incumbent_rebased_delta_judger"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "round": round_idx,
        "incumbent": str(args.incumbent_path),
        "formula": "I_best + alpha*(A_t-G_t) + beta*(S_t-G_t)",
        "coefficients": coeffs,
        "sources": {key: str(value) for key, value in paths.items()},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mixed, out)
    return out


def make_candidates(round_idx: int, args: argparse.Namespace) -> list[tuple[str, dict[str, float], str]]:
    priors = fixed_priors() + round04_templates(round_idx)
    candidates = [(name, coeffs, "prior") for name, coeffs in priors]
    rng = random.Random(args.seed + round_idx * 101)
    templates = [coeffs for _name, coeffs in priors]
    for idx in range(args.random_candidates):
        base = templates[idx % len(templates)]
        candidates.append((f"rand{idx:03d}", sample_around(base, rng, args.sample_sigma), "random"))
    seen: set[tuple[float, ...]] = set()
    unique: list[tuple[str, dict[str, float], str]] = []
    for name, coeffs, phase in candidates:
        key = tuple(round(parse_float(coeffs[field]), 4) for field in COEFF_FIELDS)
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, coeffs, phase))
    return unique


def eval_checkpoint(path: Path, cfg: Path, name: str, args: argparse.Namespace) -> dict[str, Any]:
    return opt02.eval_checkpoint(path, cfg, name, args)


def score_row(row: dict[str, Any]) -> float:
    return opt02.score_row(row)


def load_robust_judger(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return joblib.load(path)


def predict_full(row: dict[str, Any], judger: dict[str, Any] | None, args: argparse.Namespace) -> float:
    lcb = parse_float(row.get("mean_score"), 0.0) - args.lcb_lambda * parse_float(row.get("std_score"), 0.0)
    if not judger:
        return lcb
    model = judger["model"]
    feature_names = judger["feature_names"]
    values: list[float] = []
    for name in feature_names:
        if name.startswith("source="):
            values.append(1.0 if name == "source=04_delta_expert_optimizer" else 0.0)
        elif name.startswith("role="):
            values.append(1.0 if name == "role=learned_mix" else 0.0)
        else:
            values.append(parse_float(row.get(name), 0.0))
    pred = float(model.predict(np.asarray([values], dtype=np.float64))[0])
    return min(pred, lcb + args.pred_lcb_slack)


def aggregate(rows: list[dict[str, Any]], args: argparse.Namespace, judger: dict[str, Any] | None) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["round"]), str(row["candidate_id"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (round_idx, candidate_id), group_rows in grouped.items():
        scores = np.asarray([parse_float(row["split_score"]) for row in group_rows], dtype=np.float64)
        map50s = np.asarray([parse_float(row["map50"]) for row in group_rows], dtype=np.float64)
        map95s = np.asarray([parse_float(row["map50_95"]) for row in group_rows], dtype=np.float64)
        recalls = np.asarray([parse_float(row["recall"]) for row in group_rows], dtype=np.float64)
        first = group_rows[0]
        row = {
            "round": round_idx,
            "candidate_id": candidate_id,
            "phase": first["phase"],
            "path": first["path"],
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "min_score": float(np.min(scores)),
            "mean_map50": float(np.mean(map50s)),
            "std_map50": float(np.std(map50s, ddof=1)) if len(map50s) > 1 else 0.0,
            "min_map50": float(np.min(map50s)),
            "mean_map50_95": float(np.mean(map95s)),
            "mean_recall": float(np.mean(recalls)),
        }
        for field in COEFF_FIELDS:
            row[field] = parse_float(first.get(field), 0.0)
        row["proxy_lcb_score"] = row["mean_score"] - args.lcb_lambda * row["std_score"]
        row["pred_full_score"] = predict_full(row, judger, args)
        row["judger_score"] = row["pred_full_score"]
        out.append(row)
    out.sort(key=lambda row: parse_float(row["judger_score"]), reverse=True)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.source_workspace = args.source_workspace.expanduser().resolve()
    args.incumbent_path = args.incumbent_path.expanduser().resolve()
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    (args.workspace_root / "stats").mkdir(parents=True, exist_ok=True)
    splits = robust06.make_split_configs(args)
    judger = load_robust_judger(args.robust_judger_path.expanduser().resolve())

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "dqa_softmox_incumbent_rebased_delta_v1",
        "incumbent": str(args.incumbent_path),
        "incumbent_score": args.incumbent_score,
        "rounds": parse_rounds(args.rounds),
        "mini_splits": args.mini_splits,
        "mini_images": args.mini_images,
        "formula": "I_best + alpha*(A_t-G_t) + beta*(S_t-G_t)",
    }
    (args.workspace_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    notify(
        "Incumbent-rebased delta judger started\nrounds={rounds}, incumbent_score={incumbent_score:.4f}, candidates/round~{cands}".format(
            rounds=args.rounds,
            incumbent_score=args.incumbent_score,
            cands=len(fixed_priors()) + args.random_candidates,
        ),
        "DQA-SoftMoX 07 started",
        args.notify_discord,
    )

    split_rows: list[dict[str, Any]] = []
    for round_idx in parse_rounds(args.rounds):
        for candidate_id, coeffs, phase in make_candidates(round_idx, args):
            ckpt = build_rebased_checkpoint(round_idx, coeffs, candidate_id, args)
            for split in splits:
                metrics = eval_checkpoint(ckpt, Path(split["cfg"]), f"rebased07_r{round_idx:03d}_{candidate_id}_s{int(split['split']):02d}", args)
                row = {
                    "round": round_idx,
                    "candidate_id": candidate_id,
                    "phase": phase,
                    "path": str(ckpt.resolve()),
                    "split": int(split["split"]),
                    **coeffs,
                    **metrics,
                }
                row["split_score"] = score_row(row)
                split_rows.append(row)
                write_csv(args.workspace_root / "stats" / "07_split_eval.csv", split_rows)
            if not args.keep_all_candidates:
                try:
                    ckpt.unlink()
                except OSError:
                    pass

    summary = aggregate(split_rows, args, judger)
    write_csv(args.workspace_root / "stats" / "07_summary.csv", summary)

    full_cfg = opt02.full_eval_config(args)
    full_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = [
        {
            "round": 2,
            "candidate_id": "incumbent_r002",
            "accepted": True,
            "map50": 0.462,
            "map50_95": 0.260,
            "score": args.incumbent_score,
            "path": str(args.incumbent_path),
            "reason": "initial incumbent",
        }
    ]
    current_score = args.incumbent_score
    for round_idx in parse_rounds(args.rounds):
        round_rows = [row for row in summary if int(row["round"]) == round_idx]
        round_rows.sort(key=lambda row: parse_float(row["judger_score"]), reverse=True)
        best_full: dict[str, Any] | None = None
        for row in round_rows[: args.full_eval_topk]:
            coeffs = {field: parse_float(row[field], 0.0) for field in COEFF_FIELDS}
            ckpt = build_rebased_checkpoint(round_idx, coeffs, f"full_{row['candidate_id']}", args)
            metrics = eval_checkpoint(ckpt, full_cfg, f"rebased07_full_r{round_idx:03d}_{row['candidate_id']}", args)
            full = {**row, "path": str(ckpt.resolve()), "eval_scope": "full_total", **metrics}
            full["score"] = score_row(full)
            full_rows.append(full)
            if best_full is None or parse_float(full["score"]) > parse_float(best_full["score"]):
                best_full = full
        if best_full is None:
            continue
        accepted = parse_float(best_full["score"]) > current_score + args.accept_margin
        if accepted:
            current_score = parse_float(best_full["score"])
        accepted_rows.append(
            {
                **best_full,
                "accepted": accepted,
                "incumbent_after_score": current_score,
                "reason": "improved incumbent" if accepted else "rejected; keep incumbent",
            }
        )
        notify(
            f"07 round {round_idx}: best {best_full['candidate_id']} full mAP50={parse_float(best_full['map50']):.3f}, "
            f"mAP50:95={parse_float(best_full['map50_95']):.3f}, score={parse_float(best_full['score']):.4f}, accepted={accepted}",
            "DQA-SoftMoX 07 round result",
            args.notify_discord,
        )

    write_csv(args.workspace_root / "stats" / "07_full_eval.csv", full_rows)
    write_csv(args.workspace_root / "stats" / "07_accepted_policy.csv", accepted_rows)

    report = [
        "# DQA-SoftMoX Incumbent-Rebased Delta Judger 07",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- incumbent_score: {args.incumbent_score:.4f}",
        f"- mini_splits: {args.mini_splits}",
        f"- mini_images: {args.mini_images}",
        "",
        "## Accepted Policy",
        "",
        "| round | candidate | accepted | mAP50 | mAP50:95 | score | incumbent_after | reason |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in accepted_rows:
        report.append(
            f"| {int(float(row.get('round', -1)))} | {row.get('candidate_id')} | {row.get('accepted')} | "
            f"{parse_float(row.get('map50')):.3f} | {parse_float(row.get('map50_95')):.3f} | {parse_float(row.get('score')):.4f} | "
            f"{parse_float(row.get('incumbent_after_score', row.get('score'))):.4f} | {row.get('reason')} |"
        )
    report.extend(["", "## Top Proxy Candidates", "", "| rank | round | candidate | proxy LCB | pred full | mean score | std |", "|---:|---:|---|---:|---:|---:|---:|"])
    for idx, row in enumerate(summary[:30], start=1):
        report.append(
            f"| {idx} | {int(row['round'])} | {row['candidate_id']} | {parse_float(row['proxy_lcb_score']):.4f} | "
            f"{parse_float(row['pred_full_score']):.4f} | {parse_float(row['mean_score']):.4f} | {parse_float(row['std_score']):.4f} |"
        )
    (args.workspace_root / "07_incumbent_delta_judger_report.md").write_text("\n".join(report), encoding="utf-8")
    notify(
        "Incumbent-rebased delta judger finished\n" + "\n".join(
            f"- r{int(float(row.get('round', -1)))} {row.get('candidate_id')}: accepted={row.get('accepted')}, score={parse_float(row.get('score')):.4f}, incumbent_after={parse_float(row.get('incumbent_after_score', row.get('score'))):.4f}"
            for row in accepted_rows
        ),
        "DQA-SoftMoX 07 finished",
        args.notify_discord,
    )
    result = {"manifest": manifest, "accepted": accepted_rows, "top": summary[:20], "full": full_rows}
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-workspace", type=Path, default=SOURCE_WORKSPACE)
    parser.add_argument("--incumbent-path", type=Path, default=DEFAULT_INCUMBENT)
    parser.add_argument("--incumbent-score", type=float, default=0.57455)
    parser.add_argument("--robust-judger-path", type=Path, default=DEFAULT_JUDGER)
    parser.add_argument("--rounds", default="3,4,5,6")
    parser.add_argument("--mini-splits", type=int, default=3)
    parser.add_argument("--mini-images", type=int, default=384)
    parser.add_argument("--random-candidates", type=int, default=4)
    parser.add_argument("--sample-sigma", type=float, default=0.055)
    parser.add_argument("--lcb-lambda", type=float, default=0.75)
    parser.add_argument("--pred-lcb-slack", type=float, default=0.020)
    parser.add_argument("--full-eval-topk", type=int, default=2)
    parser.add_argument("--accept-margin", type=float, default=0.0002)
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
