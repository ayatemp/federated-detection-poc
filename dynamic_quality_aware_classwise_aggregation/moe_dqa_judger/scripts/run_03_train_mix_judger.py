#!/usr/bin/env python3
"""Train a small learned judger for module-wise DQA-SoftMoX mixing.

This script turns the expensive coefficient-search trace from notebook 02 into
an explicit selector:

    f(round features, candidate G/A/S weights) -> expected full-total score

At inference time we generate many legal body/head/moe candidate mixtures, score
them with the judger, and keep the best one.  The selected checkpoint can then be
materialized and optionally evaluated on the full total split.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "output" / "03_mix_judger_policy"
DEFAULT_OPTIMIZER_WORKSPACES = [
    PROJECT_ROOT / "output" / "02_mix_weight_optimizer_expanded",
    PROJECT_ROOT / "output" / "02_mix_weight_optimizer",
]
SOURCE_WORKSPACE = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation" / "moe_dqa" / "output" / "01_dqa_fedmox_yolo_full"

if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_01_judger_probe as judger01  # noqa: E402
import run_02_mix_weight_optimizer as opt02  # noqa: E402


BASE_FEATURES = judger01.FEATURE_FIELDS
WEIGHT_FIELDS = judger01.WEIGHT_FIELDS


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_workspaces(raw: str | None) -> list[Path]:
    if not raw:
        return DEFAULT_OPTIMIZER_WORKSPACES
    return [Path(item.strip()) for item in raw.split(",") if item.strip()]


def normalise_triplet(values: list[float]) -> list[float]:
    clipped = [max(float(v), 1e-4) for v in values]
    total = sum(clipped)
    return [v / total for v in clipped]


def canonical_weights(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for offset, prefix in ((0, "body"), (3, "head"), (6, "moe")):
        names = WEIGHT_FIELDS[offset : offset + 3]
        vals = normalise_triplet([float(row[name]) for name in names])
        out[names[0]], out[names[1]], out[names[2]] = vals
    return out


def feature_record(row: dict[str, Any], *, eval_scope_full: float) -> dict[str, float]:
    rec = {name: float(row.get(name, 0.0)) for name in BASE_FEATURES}
    rec.update(canonical_weights(row))
    repair = rec["repair_gain_vs_g"]
    repair_a = rec["repair_gain_vs_a"]
    agg_gap = rec["a_proxy_map50"] - rec["g_map50"]
    best_proxy = max(rec["g_map50"], rec["a_proxy_map50"])
    collapse = (1.0 - rec["expert_entropy_norm"]) + rec["dead_expert_fraction"]
    quality = rec["pseudo_mean_score"] * rec["class_entropy_norm"]
    rec.update(
        {
            "eval_scope_full": float(eval_scope_full),
            "round_sqrt": float(np.sqrt(max(rec["round"], 0.0))),
            "repair_gain_pos": max(repair, 0.0),
            "repair_gain_neg": min(repair, 0.0),
            "repair_gain_vs_a_pos": max(repair_a, 0.0),
            "repair_gain_vs_a_neg": min(repair_a, 0.0),
            "agg_gap_vs_g": agg_gap,
            "repair_vs_best_proxy": rec["s_map50"] - best_proxy,
            "pseudo_quality_x_entropy": quality,
            "expert_collapse_score": collapse,
            "body_s_x_repair": rec["body_s"] * repair,
            "head_s_x_repair": rec["head_s"] * repair,
            "moe_a_x_expert_entropy": rec["moe_a"] * rec["expert_entropy_norm"],
            "body_g_x_repair_neg": rec["body_g"] * min(repair, 0.0),
            "head_g_x_repair_neg": rec["head_g"] * min(repair, 0.0),
            "moe_g_x_collapse": rec["moe_g"] * collapse,
        }
    )
    return rec


FEATURE_COLUMNS = [
    *BASE_FEATURES,
    *WEIGHT_FIELDS,
    "eval_scope_full",
    "round_sqrt",
    "repair_gain_pos",
    "repair_gain_neg",
    "repair_gain_vs_a_pos",
    "repair_gain_vs_a_neg",
    "agg_gap_vs_g",
    "repair_vs_best_proxy",
    "pseudo_quality_x_entropy",
    "expert_collapse_score",
    "body_s_x_repair",
    "head_s_x_repair",
    "moe_a_x_expert_entropy",
    "body_g_x_repair_neg",
    "head_g_x_repair_neg",
    "moe_g_x_collapse",
]


def load_trials(workspaces: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for workspace in workspaces:
        path = workspace / "stats" / "02_mix_weight_optimizer_trials.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["optimizer_workspace"] = str(workspace.resolve())
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No 02_mix_weight_optimizer_trials.csv found.")

    df = pd.concat(frames, ignore_index=True)
    needed = set(BASE_FEATURES + WEIGHT_FIELDS + ["score", "round", "eval_scope", "returncode"])
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in optimizer trials: {missing}")
    df = df[df["returncode"].fillna(0).astype(float) == 0].copy()
    df = df[df["eval_scope"].isin(["mini", "full_total"])].copy()
    df = df.dropna(subset=["score", *BASE_FEATURES, *WEIGHT_FIELDS]).copy()
    df["round"] = df["round"].astype(float)

    # Duplicate candidates can appear in medium and expanded runs.  Keep the
    # strongest observed target for each exact role/round/eval key.
    round_cols = ["round", "eval_scope", *WEIGHT_FIELDS]
    for col in WEIGHT_FIELDS:
        df[col] = df[col].astype(float).round(6)
    df = df.sort_values("score", ascending=False).drop_duplicates(round_cols, keep="first")
    return df.reset_index(drop=True)


def training_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    x_rows: list[list[float]] = []
    y: list[float] = []
    sample_weight: list[float] = []
    for raw in df.to_dict("records"):
        eval_full = 1.0 if raw["eval_scope"] == "full_total" else 0.0
        rec = feature_record(raw, eval_scope_full=eval_full)
        x_rows.append([rec[name] for name in FEATURE_COLUMNS])
        y.append(float(raw["score"]))
        sample_weight.append(8.0 if eval_full else 1.0)
        records.append({**raw, **rec})
    return (
        np.asarray(x_rows, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(sample_weight, dtype=np.float64),
        records,
    )


def make_model(seed: int, model_type: str) -> Any:
    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1,
        )
    return ExtraTreesRegressor(
        n_estimators=700,
        max_depth=10,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
    )


def fit_model(df: pd.DataFrame, seed: int, model_type: str) -> tuple[Any, list[dict[str, Any]]]:
    x, y, w, records = training_matrix(df)
    model = make_model(seed, model_type)
    model.fit(x, y, sample_weight=w)
    return model, records


def full_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["eval_scope"] == "full_total"].copy()


def cross_validate(df: pd.DataFrame, seed: int, model_type: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for round_idx in sorted(df["round"].unique()):
        train = df[df["round"] != round_idx].copy()
        holdout = full_rows(df[df["round"] == round_idx])
        if train.empty or holdout.empty:
            continue
        model, _records = fit_model(train, seed + int(round_idx), model_type)
        pred_rows: list[dict[str, Any]] = []
        for raw in holdout.to_dict("records"):
            rec = feature_record(raw, eval_scope_full=1.0)
            pred = float(model.predict(np.asarray([[rec[name] for name in FEATURE_COLUMNS]], dtype=np.float64))[0])
            pred_rows.append({**raw, "pred_score": pred})
        selected = max(pred_rows, key=lambda item: item["pred_score"])
        best = max(pred_rows, key=lambda item: float(item["score"]))
        rows.append(
            {
                "round": int(round_idx),
                "selected_candidate": selected.get("candidate_id", ""),
                "selected_score": float(selected["score"]),
                "selected_pred_score": float(selected["pred_score"]),
                "best_candidate": best.get("candidate_id", ""),
                "best_score": float(best["score"]),
                "regret": float(best["score"]) - float(selected["score"]),
                "selected_map50": float(selected.get("map50", 0.0)),
                "best_map50": float(best.get("map50", 0.0)),
            }
        )
    return rows


def fixed_and_observed_templates(df: pd.DataFrame, max_templates: int) -> list[tuple[str, dict[str, float]]]:
    templates: list[tuple[str, dict[str, float]]] = []
    for idx, prior in enumerate(opt02.fixed_priors()):
        templates.append((f"prior{idx:02d}", prior))
    full = full_rows(df).sort_values("score", ascending=False)
    for idx, raw in enumerate(full.head(max_templates).to_dict("records")):
        templates.append((f"observed{idx:02d}", canonical_weights(raw)))
    return templates


def bootstrap_weights(source_workspace: Path, workspace: Path, round_idx: int) -> dict[str, float] | None:
    try:
        model, _rows = judger01.train_judger(source_workspace, 21, workspace / "bootstrap")
        return model.predict(judger01.round_features(source_workspace, round_idx))
    except Exception as exc:  # noqa: BLE001 - bootstrap is optional.
        print(f"bootstrap skipped for round {round_idx}: {exc}")
        return None


def dedupe_key(weights: dict[str, float]) -> tuple[float, ...]:
    return tuple(round(float(weights[name]), 4) for name in WEIGHT_FIELDS)


def generate_candidate_pool(
    features: dict[str, float],
    round_idx: int,
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> list[tuple[str, dict[str, float]]]:
    rng = random.Random(args.seed + 7000 + round_idx)
    pool: list[tuple[str, dict[str, float]]] = []
    seen: set[tuple[float, ...]] = set()

    def add(name: str, weights: dict[str, float]) -> None:
        clean = canonical_weights(weights)
        key = dedupe_key(clean)
        if key in seen:
            return
        seen.add(key)
        pool.append((name, clean))

    templates = fixed_and_observed_templates(df, args.observed_templates)
    for name, weights in templates:
        add(name, weights)
    boot = bootstrap_weights(args.source_workspace, args.workspace_root, round_idx)
    if boot:
        add("bootstrap", boot)
        templates.append(("bootstrap", boot))

    # Add targeted smooth mutations.  This keeps the search continuous while
    # staying in the legal simplex for each body/head/moe block.
    for idx in range(args.pool_samples):
        base_name, base = templates[idx % len(templates)]
        concentration = args.dirichlet_concentration * (1.5 if idx % 3 else 0.8)
        weights = opt02.sample_weights(base, rng, concentration)
        add(f"sample{idx:04d}_{base_name}", weights)

    # Add a few feature-conditioned handholds for extrapolation.
    if features["repair_gain_vs_g"] < -0.003 and features["repair_gain_vs_a"] < 0.001:
        add("feature_freeze_g", opt02.unflatten_weights([1, 0, 0] * 3))
    if features["repair_gain_vs_a"] > 0 and features["expert_entropy_norm"] > 0.5:
        add("feature_head_repair_moe_aggregate", opt02.unflatten_weights([0.65, 0.25, 0.10, 0.20, 0.10, 0.70, 0.15, 0.75, 0.10]))
    return pool


def select_for_round(
    model: Any,
    df: pd.DataFrame,
    round_idx: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    features = judger01.round_features(args.source_workspace, round_idx)
    pool = generate_candidate_pool(features, round_idx, df, args)
    rows: list[dict[str, Any]] = []
    x_rows: list[list[float]] = []
    for candidate_id, weights in pool:
        rec = feature_record({**features, **weights}, eval_scope_full=1.0)
        rows.append({"candidate_id": candidate_id, **features, **weights})
        x_rows.append([rec[name] for name in FEATURE_COLUMNS])
    preds = model.predict(np.asarray(x_rows, dtype=np.float64))
    scored = [
        {**row, "pred_score": float(pred)}
        for row, pred in zip(rows, preds, strict=True)
    ]
    best = max(scored, key=lambda item: item["pred_score"])
    guard_reason = ""
    if not args.disable_drift_guard and round_idx >= args.drift_guard_after_round:
        g = float(features["g_map50"])
        a = float(features["a_proxy_map50"])
        s = float(features["s_map50"])
        g_beats_children = g - max(a, s) > args.drift_freeze_margin
        repair_hurts_both = (
            features["repair_gain_vs_a"] < -args.drift_freeze_margin
            and features["repair_gain_vs_g"] < -args.drift_freeze_margin
        )
        if g_beats_children or repair_hurts_both:
            # The learned regressor still proposes among candidates, but this
            # guard prevents the known repeated repair/self-training failure:
            # once both child branches drift on the source anchor, keep the
            # previous global as the next parent.
            best = max(
                scored,
                key=lambda item: (
                    float(item["body_g"]) + float(item["head_g"]) + float(item["moe_g"]),
                    item["pred_score"],
                ),
            )
            guard_reason = "g_beats_children" if g_beats_children else "repair_hurts_both"
    best["round"] = round_idx
    best["pool_size"] = len(pool)
    best["guard_reason"] = guard_reason
    return best


def materialize_and_eval(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    full_cfg = opt02.full_eval_config(args)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        round_idx = int(row["round"])
        weights = {name: float(row[name]) for name in WEIGHT_FIELDS}
        ckpt = opt02.build_candidate(round_idx, weights, f"judger03_selected_r{round_idx:03d}", args)
        metrics = opt02.eval_checkpoint(ckpt, full_cfg, f"judger03_selected_r{round_idx:03d}_full", args)
        out = {**row, "path": str(ckpt.resolve()), **metrics}
        out["score"] = opt02.score_row(out)
        out_rows.append(out)
    return out_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.source_workspace = args.source_workspace.expanduser().resolve()
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    (args.workspace_root / "stats").mkdir(parents=True, exist_ok=True)

    optimizer_workspaces = [path.expanduser().resolve() for path in parse_workspaces(args.optimizer_workspaces)]
    df = load_trials(optimizer_workspaces)
    model, records = fit_model(df, args.seed, args.model_type)
    cv_rows = cross_validate(df, args.seed, args.model_type)

    selected_rows = [
        select_for_round(model, df, round_idx, args)
        for round_idx in [int(item.strip()) for item in args.rounds.split(",") if item.strip()]
    ]
    eval_rows = materialize_and_eval(selected_rows, args) if args.evaluate_full else []

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "base_features": BASE_FEATURES,
        "weight_fields": WEIGHT_FIELDS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "optimizer_workspaces": [str(path) for path in optimizer_workspaces],
        "model_type": args.model_type,
    }
    joblib.dump(artifact, args.workspace_root / "mix_judger_v1.joblib")

    write_csv(args.workspace_root / "stats" / "03_training_rows.csv", records)
    write_csv(args.workspace_root / "stats" / "03_leave_one_round_cv.csv", cv_rows)
    write_csv(args.workspace_root / "stats" / "03_selected_weights.csv", selected_rows)
    if eval_rows:
        write_csv(args.workspace_root / "stats" / "03_selected_full_eval.csv", eval_rows)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "dqa_softmox_mix_judger_v1",
        "method": "learned score model over module-wise G/A/S candidate weights",
        "workspace": str(args.workspace_root),
        "source_workspace": str(args.source_workspace),
        "optimizer_workspaces": [str(path) for path in optimizer_workspaces],
        "training_rows": int(len(df)),
        "full_training_rows": int((df["eval_scope"] == "full_total").sum()),
        "rounds": args.rounds,
        "model_type": args.model_type,
    }
    (args.workspace_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    report = [
        "# DQA-SoftMoX Mix Judger Policy 03",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- training_rows: {len(df)}",
        f"- full_training_rows: {(df['eval_scope'] == 'full_total').sum()}",
        f"- model_type: {args.model_type}",
        "",
        "## Leave-One-Round Full-Candidate CV",
        "",
        "| round | selected | best | selected score | best score | regret |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in cv_rows:
        report.append(
            "| {round} | {selected_candidate} | {best_candidate} | {selected_score:.4f} | {best_score:.4f} | {regret:.4f} |".format(**row)
        )
    report.extend(
        [
            "",
            "## Selected Policy Weights",
            "",
            "| round | pred | body G/A/S | head G/A/S | moe G/A/S | pool | guard |",
            "|---:|---:|---|---|---|---:|---|",
        ]
    )
    for row in selected_rows:
        report.append(
            "| {round} | {pred_score:.4f} | {body_g:.2f}/{body_a:.2f}/{body_s:.2f} | "
            "{head_g:.2f}/{head_a:.2f}/{head_s:.2f} | {moe_g:.2f}/{moe_a:.2f}/{moe_s:.2f} | {pool_size} | {guard_reason} |".format(**row)
        )
    if eval_rows:
        report.extend(
            [
                "",
                "## Selected Full-Total Evaluation",
                "",
                "| round | mAP50 | mAP50:95 | precision | recall | score |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in eval_rows:
            report.append(
                "| {round} | {map50:.3f} | {map50_95:.3f} | {precision:.3f} | {recall:.3f} | {score:.4f} |".format(**row)
            )
    (args.workspace_root / "03_mix_judger_policy_report.md").write_text("\n".join(report), encoding="utf-8")

    result = {
        "manifest": manifest,
        "cv": cv_rows,
        "selected": selected_rows,
        "eval": eval_rows,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-workspace", type=Path, default=SOURCE_WORKSPACE)
    parser.add_argument("--optimizer-workspaces", default="")
    parser.add_argument("--rounds", default="1,2,3,4,5")
    parser.add_argument("--model-type", choices=["extratrees", "rf"], default="extratrees")
    parser.add_argument("--pool-samples", type=int, default=2200)
    parser.add_argument("--observed-templates", type=int, default=12)
    parser.add_argument("--dirichlet-concentration", type=float, default=20.0)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--disable-drift-guard", action="store_true")
    parser.add_argument("--drift-guard-after-round", type=int, default=3)
    parser.add_argument("--drift-freeze-margin", type=float, default=0.003)
    parser.add_argument("--evaluate-full", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-all-candidates", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
