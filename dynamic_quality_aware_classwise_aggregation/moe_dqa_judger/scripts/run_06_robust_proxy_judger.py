#!/usr/bin/env python3
"""Robust multi-split proxy judger for DQA-SoftMoX mixing.

The earlier optimizers found useful mixing patterns, but they also exposed a
failure mode: a single small validation subset can over-select a checkpoint that
does not hold up on the full paper protocol.  This script builds the next judge
around repeated small validation splits.  It evaluates self-generated candidate
checkpoints on several disjoint mini splits, summarizes mean/std/min/LCB proxy
quality, and trains a lightweight calibrator against already-known full scores.

The output is a policy table: for each round, which self-generated checkpoint or
mix should be trusted next, plus the robust evidence behind the choice.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "output" / "06_robust_proxy_judger"
SOURCE_WORKSPACE = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation" / "moe_dqa" / "output" / "01_dqa_fedmox_yolo_full"

if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_01_judger_probe as judger01  # noqa: E402
import run_02_mix_weight_optimizer as opt02  # noqa: E402


FULL_SCORE_FILES = [
    PROJECT_ROOT / "output" / "02_mix_weight_optimizer_expanded" / "stats" / "02_mix_weight_optimizer_best_full.csv",
    PROJECT_ROOT / "output" / "03_mix_judger_policy" / "stats" / "03_selected_full_eval.csv",
    PROJECT_ROOT / "output" / "04_delta_expert_optimizer" / "stats" / "04_delta_expert_best_full.csv",
    PROJECT_ROOT / "output" / "05_greedy_soup_judger" / "stats" / "05_greedy_soup_full_eval.csv",
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


def safe_name(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)[:160]


def parse_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_round(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def score_from_metrics(row: dict[str, Any]) -> float:
    if "score" in row and str(row["score"]) not in {"", "nan", "None"}:
        return parse_float(row["score"])
    return opt02.score_row(row)


def source_cfg_path(args: argparse.Namespace) -> Path:
    cfg = args.source_workspace / "validation_reports" / "paper_protocol_configs" / "scene_daynight_total.yaml"
    if not cfg.exists():
        raise FileNotFoundError(cfg)
    return cfg


def source_list_path(args: argparse.Namespace) -> Path:
    path = args.source_workspace / "data_lists" / "paper_eval_scene_daynight_total_val.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def make_split_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    source_list = source_list_path(args)
    lines = [line.strip() for line in source_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = np.random.default_rng(args.seed)
    order = np.asarray(lines, dtype=object)
    rng.shuffle(order)

    source_cfg = yaml.safe_load(source_cfg_path(args).read_text(encoding="utf-8"))
    split_rows: list[dict[str, Any]] = []
    disjoint_possible = args.mini_images * args.mini_splits <= len(order)

    for split_idx in range(args.mini_splits):
        if disjoint_possible:
            chosen = order[split_idx * args.mini_images : (split_idx + 1) * args.mini_images].tolist()
        else:
            split_rng = np.random.default_rng(args.seed + split_idx * 9973)
            chosen = split_rng.choice(order, size=min(args.mini_images, len(order)), replace=False).tolist()
        chosen = sorted(str(item) for item in chosen)
        list_path = args.workspace_root / "data_lists" / f"robust_split{split_idx:02d}_{len(chosen)}.txt"
        cfg_path = args.workspace_root / "configs" / f"robust_split{split_idx:02d}_{len(chosen)}.yaml"
        list_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        list_path.write_text("\n".join(chosen) + "\n", encoding="utf-8")
        cfg = dict(source_cfg)
        cfg["Dataset"] = dict(source_cfg["Dataset"])
        cfg["Dataset"]["val"] = str(list_path.resolve())
        cfg["Dataset"]["test"] = str(list_path.resolve())
        cfg["Dataset"]["batch_size"] = int(args.val_batch_size)
        cfg["Dataset"]["workers"] = 0
        cfg["SSOD"] = {"train_domain": False}
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        split_rows.append({"split": split_idx, "list": list_path, "cfg": cfg_path, "images": len(chosen)})
    return split_rows


def load_known_full_scores() -> dict[Path, dict[str, Any]]:
    known: dict[Path, dict[str, Any]] = {}
    for csv_path in FULL_SCORE_FILES:
        for row in read_csv(csv_path):
            raw_path = row.get("path", "")
            if not raw_path:
                continue
            path = Path(raw_path).resolve()
            if not path.exists():
                continue
            current = known.get(path)
            score = score_from_metrics(row)
            if current is None or score > parse_float(current.get("known_full_score"), -1.0):
                known[path] = {
                    "known_full_score": score,
                    "known_full_map50": parse_float(row.get("map50")),
                    "known_full_map50_95": parse_float(row.get("map50_95")),
                    "known_full_recall": parse_float(row.get("recall")),
                    "known_full_precision": parse_float(row.get("precision")),
                    "known_full_source_file": str(csv_path),
                    "known_full_candidate_id": row.get("candidate_id", row.get("candidate", "")),
                    "known_full_round": parse_round(row.get("round", row.get("step", -1))),
                }
    return known


def add_candidate(
    rows: list[dict[str, Any]],
    seen: set[tuple[str, int]],
    *,
    label: str,
    path: Path,
    source: str,
    round_idx: int,
    role: str,
    known: dict[Path, dict[str, Any]],
) -> None:
    path = path.resolve()
    if not path.exists():
        return
    key = (str(path), round_idx)
    if key in seen:
        return
    seen.add(key)
    full = known.get(path, {})
    rows.append(
        {
            "label": safe_name(label),
            "path": str(path),
            "source": source,
            "round": round_idx,
            "role": role,
            **full,
        }
    )


def candidate_pool(args: argparse.Namespace) -> list[dict[str, Any]]:
    known = load_known_full_scores()
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    ckpt_dir = args.source_workspace / "checkpoints"

    add_candidate(
        rows,
        seen,
        label="warmup_g0",
        path=ckpt_dir / "round000_latent_dqamox_warmup.pt",
        source="source",
        round_idx=0,
        role="warmup",
        known=known,
    )
    for round_idx in range(1, args.max_round + 1):
        try:
            paths = judger01.checkpoint_paths(args.source_workspace, round_idx)
        except FileNotFoundError:
            continue
        for role, path in paths.items():
            add_candidate(
                rows,
                seen,
                label=f"r{round_idx:03d}_{role}",
                path=path,
                source="source",
                round_idx=round_idx,
                role=role,
                known=known,
            )

    scored_rows: list[dict[str, Any]] = []
    for csv_path in FULL_SCORE_FILES:
        csv_rows = read_csv(csv_path)
        csv_rows.sort(key=score_from_metrics, reverse=True)
        for row in csv_rows[: args.per_result_file_topk]:
            raw_path = row.get("path", "")
            if not raw_path:
                continue
            path = Path(raw_path).resolve()
            round_idx = parse_round(row.get("round", row.get("step", -1)))
            source = csv_path.parent.parent.name
            label_bits = [source, f"r{round_idx:03d}" if round_idx >= 0 else "rNA", row.get("candidate_id", row.get("candidate", path.stem))]
            scored_rows.append(
                {
                    "label": safe_name("_".join(str(bit) for bit in label_bits if str(bit))),
                    "path": path,
                    "source": source,
                    "round": round_idx,
                    "role": "learned_mix",
                    "score": score_from_metrics(row),
                }
            )

    scored_rows.sort(key=lambda row: parse_float(row.get("score"), -1.0), reverse=True)
    for row in scored_rows:
        add_candidate(
            rows,
            seen,
            label=str(row["label"]),
            path=Path(row["path"]),
            source=str(row["source"]),
            round_idx=parse_round(row["round"]),
            role=str(row["role"]),
            known=known,
        )
        if len(rows) >= args.max_candidates:
            break
    return rows[: args.max_candidates]


def existing_split_eval(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    cache: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_csv(path):
        cache[(row.get("label", ""), parse_round(row.get("split")))] = row
    return cache


def evaluate_splits(args: argparse.Namespace, candidates: list[dict[str, Any]], splits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eval_path = args.workspace_root / "stats" / "06_split_eval.csv"
    cache = existing_split_eval(eval_path) if args.resume else {}
    rows: list[dict[str, Any]] = list(cache.values())
    for candidate in candidates:
        for split in splits:
            key = (candidate["label"], int(split["split"]))
            if key in cache:
                continue
            eval_name = safe_name(f"robust06_{candidate['label']}_split{int(split['split']):02d}")
            metrics = opt02.eval_checkpoint(Path(candidate["path"]), Path(split["cfg"]), eval_name, args)
            row = {
                **candidate,
                "split": int(split["split"]),
                "split_images": int(split["images"]),
                "eval_scope": "mini_split",
                **metrics,
            }
            row["split_score"] = opt02.score_row(row)
            rows.append(row)
            write_csv(eval_path, rows)
    return rows


def aggregate_split_rows(split_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in split_rows:
        grouped.setdefault(str(row["label"]), []).append(row)
    out: list[dict[str, Any]] = []
    for label, rows in grouped.items():
        rows = sorted(rows, key=lambda row: parse_round(row.get("split")))
        scores = np.asarray([parse_float(row.get("split_score", row.get("score"))) for row in rows], dtype=np.float64)
        map50s = np.asarray([parse_float(row.get("map50")) for row in rows], dtype=np.float64)
        map95s = np.asarray([parse_float(row.get("map50_95")) for row in rows], dtype=np.float64)
        recalls = np.asarray([parse_float(row.get("recall")) for row in rows], dtype=np.float64)
        first = rows[0]
        std_score = float(np.nanstd(scores, ddof=1)) if len(scores) > 1 else 0.0
        std_map50 = float(np.nanstd(map50s, ddof=1)) if len(map50s) > 1 else 0.0
        out.append(
            {
                "label": label,
                "path": first["path"],
                "source": first.get("source", ""),
                "round": parse_round(first.get("round")),
                "role": first.get("role", ""),
                "split_count": len(rows),
                "mean_score": float(np.nanmean(scores)),
                "std_score": std_score,
                "min_score": float(np.nanmin(scores)),
                "mean_map50": float(np.nanmean(map50s)),
                "std_map50": std_map50,
                "min_map50": float(np.nanmin(map50s)),
                "mean_map50_95": float(np.nanmean(map95s)),
                "std_map50_95": float(np.nanstd(map95s, ddof=1)) if len(map95s) > 1 else 0.0,
                "mean_recall": float(np.nanmean(recalls)),
                "known_full_score": parse_float(first.get("known_full_score")),
                "known_full_map50": parse_float(first.get("known_full_map50")),
                "known_full_map50_95": parse_float(first.get("known_full_map50_95")),
                "known_full_source_file": first.get("known_full_source_file", ""),
                "known_full_candidate_id": first.get("known_full_candidate_id", ""),
            }
        )
    return out


def feature_matrix(
    rows: list[dict[str, Any]],
    *,
    sources: list[str] | None = None,
    roles: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if sources is None:
        sources = sorted({str(row.get("source", "")) for row in rows})
    if roles is None:
        roles = sorted({str(row.get("role", "")) for row in rows})
    fields = ["round", "mean_score", "std_score", "min_score", "mean_map50", "std_map50", "min_map50", "mean_map50_95", "mean_recall"]
    names = fields + [f"source={name}" for name in sources] + [f"role={name}" for name in roles]
    x_rows: list[list[float]] = []
    for row in rows:
        values = [parse_float(row.get(field), 0.0) for field in fields]
        values.extend(1.0 if str(row.get("source", "")) == name else 0.0 for name in sources)
        values.extend(1.0 if str(row.get("role", "")) == name else 0.0 for name in roles)
        x_rows.append(values)
    return np.asarray(x_rows, dtype=np.float64), names


def train_calibrator(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    train_rows = [row for row in rows if math.isfinite(parse_float(row.get("known_full_score")))]
    model_info: dict[str, Any] = {
        "model": None,
        "feature_names": [],
        "train_count": len(train_rows),
        "cv_mae": math.nan,
    }
    if len(train_rows) < 6:
        for row in rows:
            row["pred_full_score"] = row["mean_score"] - args.lcb_lambda * row["std_score"]
            row["judger_score"] = row["pred_full_score"]
        return model_info

    sources = sorted({str(row.get("source", "")) for row in rows})
    roles = sorted({str(row.get("role", "")) for row in rows})
    x_train, feature_names = feature_matrix(train_rows, sources=sources, roles=roles)
    y_train = np.asarray([parse_float(row["known_full_score"]) for row in train_rows], dtype=np.float64)
    model = ExtraTreesRegressor(
        n_estimators=360,
        max_depth=6,
        min_samples_leaf=1,
        random_state=args.seed,
        n_jobs=-1,
    )
    if len(train_rows) >= 8:
        cv_pred = cross_val_predict(model, x_train, y_train, cv=LeaveOneOut(), n_jobs=None)
        model_info["cv_mae"] = float(np.mean(np.abs(cv_pred - y_train)))
        for row, pred in zip(train_rows, cv_pred, strict=True):
            row["cv_pred_full_score"] = float(pred)
            row["cv_abs_error"] = abs(float(pred) - parse_float(row["known_full_score"]))
    model.fit(x_train, y_train)

    x_all, _ = feature_matrix(rows, sources=sources, roles=roles)
    pred = model.predict(x_all)
    for row, pred_score in zip(rows, pred, strict=True):
        lcb = parse_float(row["mean_score"]) - args.lcb_lambda * parse_float(row["std_score"], 0.0)
        row["proxy_lcb_score"] = lcb
        row["pred_full_score"] = float(pred_score)
        row["judger_score"] = min(float(pred_score), lcb + args.pred_lcb_slack)

    model_info.update({"model": model, "feature_names": feature_names})
    return model_info


def full_eval_selected(args: argparse.Namespace, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full_cfg = opt02.full_eval_config(args)
    cache_path = args.workspace_root / "stats" / "06_selected_policy_full.csv"
    cached = {row.get("label", ""): row for row in read_csv(cache_path)} if args.resume and cache_path.exists() else {}
    rows: list[dict[str, Any]] = []
    for row in selected:
        if str(row.get("label", "")) in cached and not args.force_full_reeval:
            rows.append({**row, **cached[str(row["label"])]})
            continue
        if math.isfinite(parse_float(row.get("known_full_score"))) and not args.force_full_reeval:
            full = {
                **row,
                "eval_scope": "known_full",
                "map50": row.get("known_full_map50"),
                "map50_95": row.get("known_full_map50_95"),
                "score": row.get("known_full_score"),
                "returncode": 0,
            }
        else:
            metrics = opt02.eval_checkpoint(Path(row["path"]), full_cfg, safe_name(f"robust06_full_{row['label']}"), args)
            full = {**row, "eval_scope": "full_total", **metrics}
            full["score"] = opt02.score_row(full)
        rows.append(full)
    return rows


def select_policy_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    rounds = sorted({parse_round(row.get("round")) for row in rows if parse_round(row.get("round")) >= 0})
    for round_idx in rounds:
        candidates = [row for row in rows if parse_round(row.get("round")) == round_idx]
        candidates.sort(key=lambda row: (parse_float(row.get("judger_score")), parse_float(row.get("proxy_lcb_score"))), reverse=True)
        selected.extend(candidates[: args.select_topk_per_round])
    return selected


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.source_workspace = args.source_workspace.expanduser().resolve()
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    (args.workspace_root / "stats").mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "dqa_softmox_robust_proxy_judger_v1",
        "method": "multi-split validation proxy with full-score calibrator",
        "source_workspace": str(args.source_workspace),
        "workspace": str(args.workspace_root),
        "max_round": args.max_round,
        "mini_splits": args.mini_splits,
        "mini_images": args.mini_images,
        "lcb_lambda": args.lcb_lambda,
        "papers_used": [
            "FedLAW/Revisiting weighted aggregation: learned weights and shrinkage",
            "pFedLA/FedLAMA: layer-wise aggregation",
            "Model soups: validation-gated checkpoint mixing",
        ],
    }
    (args.workspace_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    notify(
        "Robust proxy judger started\nmini_splits={mini_splits}, mini_images={mini_images}, max_round={max_round}".format(**manifest),
        "DQA-SoftMoX 06 started",
        args.notify_discord,
    )

    splits = make_split_configs(args)
    candidates = candidate_pool(args)
    write_csv(args.workspace_root / "stats" / "06_candidate_pool.csv", candidates)
    split_rows = evaluate_splits(args, candidates, splits)
    aggregate_rows = aggregate_split_rows(split_rows)
    for row in aggregate_rows:
        row["proxy_lcb_score"] = parse_float(row["mean_score"]) - args.lcb_lambda * parse_float(row["std_score"], 0.0)
    model_info = train_calibrator(aggregate_rows, args)
    aggregate_rows.sort(key=lambda row: parse_float(row.get("judger_score", row.get("proxy_lcb_score"))), reverse=True)
    write_csv(args.workspace_root / "stats" / "06_robust_proxy_summary.csv", aggregate_rows)

    selected = select_policy_rows(aggregate_rows, args)
    full_rows = full_eval_selected(args, selected)
    full_rows.sort(key=lambda row: (parse_round(row.get("round")), -parse_float(row.get("score"))))
    write_csv(args.workspace_root / "stats" / "06_selected_policy_full.csv", full_rows)

    if model_info.get("model") is not None:
        joblib.dump(
            {
                "model": model_info["model"],
                "feature_names": model_info["feature_names"],
                "manifest": manifest,
            },
            args.workspace_root / "robust_proxy_judger_v1.joblib",
        )

    report = [
        "# DQA-SoftMoX Robust Proxy Judger 06",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- candidate_count: {len(candidates)}",
        f"- mini_splits: {args.mini_splits}",
        f"- mini_images: {args.mini_images}",
        f"- calibrator_train_count: {model_info['train_count']}",
        f"- calibrator_loo_mae: {model_info['cv_mae']:.5f}" if math.isfinite(parse_float(model_info["cv_mae"])) else "- calibrator_loo_mae: n/a",
        "",
        "## Selected Policy",
        "",
        "| round | label | source | role | mean proxy | std | LCB | pred full | full mAP50 | full mAP50:95 | full score |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in full_rows:
        report.append(
            f"| {parse_round(row.get('round'))} | {row.get('label')} | {row.get('source')} | {row.get('role')} | "
            f"{parse_float(row.get('mean_score')):.4f} | {parse_float(row.get('std_score')):.4f} | "
            f"{parse_float(row.get('proxy_lcb_score')):.4f} | {parse_float(row.get('pred_full_score')):.4f} | "
            f"{parse_float(row.get('map50')):.3f} | {parse_float(row.get('map50_95')):.3f} | {parse_float(row.get('score')):.4f} |"
        )
    report.extend(["", "## Top Robust Candidates", "", "| rank | round | label | source | role | LCB | pred full | known full |", "|---:|---:|---|---|---|---:|---:|---:|"])
    for idx, row in enumerate(aggregate_rows[:20], start=1):
        report.append(
            f"| {idx} | {parse_round(row.get('round'))} | {row.get('label')} | {row.get('source')} | {row.get('role')} | "
            f"{parse_float(row.get('proxy_lcb_score')):.4f} | {parse_float(row.get('pred_full_score')):.4f} | "
            f"{parse_float(row.get('known_full_score')):.4f} |"
        )
    (args.workspace_root / "06_robust_proxy_judger_report.md").write_text("\n".join(report), encoding="utf-8")

    notify(
        "Robust proxy judger finished\n"
        + "\n".join(
            f"- r{int(row['round'])} {row['label']}: full mAP50={parse_float(row['map50']):.3f}, mAP50:95={parse_float(row['map50_95']):.3f}, score={parse_float(row['score']):.4f}, LCB={parse_float(row['proxy_lcb_score']):.4f}"
            for row in full_rows[:12]
        ),
        "DQA-SoftMoX 06 finished",
        args.notify_discord,
    )
    result = {
        "manifest": manifest,
        "selected": full_rows,
        "top": aggregate_rows[:20],
        "calibrator": {key: value for key, value in model_info.items() if key != "model"},
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-workspace", type=Path, default=SOURCE_WORKSPACE)
    parser.add_argument("--max-round", type=int, default=6)
    parser.add_argument("--mini-splits", type=int, default=3)
    parser.add_argument("--mini-images", type=int, default=384)
    parser.add_argument("--max-candidates", type=int, default=44)
    parser.add_argument("--per-result-file-topk", type=int, default=12)
    parser.add_argument("--select-topk-per-round", type=int, default=1)
    parser.add_argument("--lcb-lambda", type=float, default=0.75)
    parser.add_argument("--pred-lcb-slack", type=float, default=0.020)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force-full-reeval", action="store_true")
    parser.add_argument("--notify-discord", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
