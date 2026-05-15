#!/usr/bin/env python3
"""Optimize DQA-SoftMoX module-wise G/A/S mixing weights.

This is the "learn the mixing" probe.  For each requested round it treats the
module-wise weights as black-box parameters, evaluates mixed checkpoints on a
small validation list, trains a surrogate regressor over the observed weights,
and evaluates the best candidates on the full total split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "output" / "02_mix_weight_optimizer"
SOURCE_WORKSPACE = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation" / "moe_dqa" / "output" / "01_dqa_fedmox_yolo_full"
RUN01 = PROJECT_ROOT / "scripts" / "run_01_judger_probe.py"

if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_01_judger_probe as judger01  # noqa: E402


VAL_PYTHON = Path("/root/micromamba/envs/al_yolov8/bin/python")
VAL_PY = REPO_ROOT / "navigating_data_heterogeneity" / "vendor" / "efficientteacher" / "val.py"


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


def parse_val_stdout(stdout: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) >= 7 and parts[0] == "all":
            parsed = {
                "images": float(parts[1]),
                "labels": float(parts[2]),
                "precision": float(parts[3]),
                "recall": float(parts[4]),
                "map50": float(parts[5]),
                "map50_95": float(parts[6]),
            }
    return parsed


def normalise_triplet(values: list[float]) -> list[float]:
    clipped = [max(float(v), 1e-4) for v in values]
    total = sum(clipped)
    return [v / total for v in clipped]


def flatten_weights(weights: dict[str, float]) -> list[float]:
    return [float(weights[name]) for name in judger01.WEIGHT_FIELDS]


def unflatten_weights(values: list[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for offset, prefix in ((0, "body"), (3, "head"), (6, "moe")):
        triplet = normalise_triplet(values[offset : offset + 3])
        out[f"{prefix}_g"], out[f"{prefix}_a"], out[f"{prefix}_s"] = triplet
    return out


def dirichlet_around(base: list[float], concentration: float, rng: random.Random) -> list[float]:
    alpha = [max(v * concentration, 0.05) for v in base]
    sample = np.random.default_rng(rng.randint(0, 2**32 - 1)).dirichlet(alpha)
    return sample.astype(float).tolist()


def sample_weights(base: dict[str, float], rng: random.Random, concentration: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for prefix in ("body", "head", "moe"):
        triplet = [base[f"{prefix}_g"], base[f"{prefix}_a"], base[f"{prefix}_s"]]
        sampled = dirichlet_around(triplet, concentration, rng)
        out[f"{prefix}_g"], out[f"{prefix}_a"], out[f"{prefix}_s"] = sampled
    return out


def fixed_priors() -> list[dict[str, float]]:
    specs = [
        # Identity baselines. These let the optimizer discover that "do not mix"
        # is the best answer for a round/module when the candidates disagree.
        ([1.00, 0.00, 0.00], [1.00, 0.00, 0.00], [1.00, 0.00, 0.00]),
        ([0.00, 1.00, 0.00], [0.00, 1.00, 0.00], [0.00, 1.00, 0.00]),
        ([0.00, 0.00, 1.00], [0.00, 0.00, 1.00], [0.00, 0.00, 1.00]),
        # Intuition prior: body/MoE keep target signal, head keeps source repair.
        ([0.10, 0.75, 0.15], [0.20, 0.30, 0.50], [0.10, 0.80, 0.10]),
        # Repair-heavy.
        ([0.15, 0.45, 0.40], [0.20, 0.20, 0.60], [0.15, 0.55, 0.30]),
        # Aggregate-heavy.
        ([0.10, 0.85, 0.05], [0.15, 0.55, 0.30], [0.10, 0.85, 0.05]),
        # Body stays target-adapted while the head is strongly source-repaired.
        ([0.05, 0.90, 0.05], [0.05, 0.15, 0.80], [0.05, 0.85, 0.10]),
        # Conservative repair: mostly keep previous/global body, repair only the
        # detection head, and let MoE follow the target aggregate.
        ([0.65, 0.25, 0.10], [0.20, 0.10, 0.70], [0.15, 0.75, 0.10]),
        # Anchor-protected.
        ([0.30, 0.60, 0.10], [0.35, 0.25, 0.40], [0.30, 0.60, 0.10]),
        # FedMoX-like smooth middle.
        ([0.20, 0.60, 0.20], [0.20, 0.40, 0.40], [0.20, 0.60, 0.20]),
    ]
    priors = []
    for body, head, moe in specs:
        values = normalise_triplet(body) + normalise_triplet(head) + normalise_triplet(moe)
        priors.append(unflatten_weights(values))
    return priors


def ensure_mini_eval_config(args: argparse.Namespace) -> tuple[Path, Path]:
    source_list = args.source_workspace / "data_lists" / "paper_eval_scene_daynight_total_val.txt"
    if not source_list.exists():
        raise FileNotFoundError(source_list)
    lines = [line.strip() for line in source_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(lines)
    mini_lines = sorted(lines[: min(args.mini_images, len(lines))])
    mini_list = args.workspace_root / "data_lists" / f"judger_mini_total_{len(mini_lines)}.txt"
    mini_list.parent.mkdir(parents=True, exist_ok=True)
    mini_list.write_text("\n".join(mini_lines) + "\n", encoding="utf-8")

    source_cfg = args.source_workspace / "validation_reports" / "paper_protocol_configs" / "scene_daynight_total.yaml"
    if not source_cfg.exists():
        raise FileNotFoundError(source_cfg)
    cfg = yaml.safe_load(source_cfg.read_text(encoding="utf-8"))
    cfg["Dataset"]["val"] = str(mini_list.resolve())
    cfg["Dataset"]["test"] = str(mini_list.resolve())
    cfg["Dataset"]["batch_size"] = int(args.val_batch_size)
    cfg["Dataset"]["workers"] = 0
    cfg["SSOD"] = {"train_domain": False}
    out_cfg = args.workspace_root / "configs" / f"judger_mini_total_{len(mini_lines)}.yaml"
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return mini_list, out_cfg


def full_eval_config(args: argparse.Namespace) -> Path:
    cfg = args.source_workspace / "validation_reports" / "paper_protocol_configs" / "scene_daynight_total.yaml"
    if not cfg.exists():
        raise FileNotFoundError(cfg)
    return cfg


def eval_checkpoint(weights_path: Path, cfg_path: Path, name: str, args: argparse.Namespace) -> dict[str, Any]:
    log_dir = args.workspace_root / "validation_logs"
    run_dir = args.workspace_root / "validation_runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(VAL_PYTHON if VAL_PYTHON.exists() else sys.executable),
        str(VAL_PY),
        "--weights",
        str(weights_path.resolve()),
        "--cfg",
        str(cfg_path.resolve()),
        "--batch-size",
        str(args.val_batch_size),
        "--imgsz",
        "640",
        "--conf-thres",
        "0.001",
        "--iou-thres",
        "0.6",
        "--project",
        str(run_dir.resolve()),
        "--name",
        name,
        "--exist-ok",
        "--no-plots",
    ]
    result = subprocess.run(cmd, cwd=VAL_PY.parent, capture_output=True, text=True)
    log_path = log_dir / f"{name}.log"
    log_path.write_text(result.stdout + "\nSTDERR:\n" + result.stderr, encoding="utf-8")
    metrics = parse_val_stdout(result.stdout)
    return {
        "returncode": result.returncode,
        "log_file": str(log_path.resolve()),
        "command": " ".join(cmd),
        **metrics,
    }


def score_row(row: dict[str, Any]) -> float:
    # mAP50 is the target, but mAP50:95 and recall keep geometry/coverage from
    # being sacrificed too casually.
    return float(row.get("map50", 0.0)) + 0.35 * float(row.get("map50_95", 0.0)) + 0.05 * float(row.get("recall", 0.0))


def build_candidate(
    round_idx: int,
    weights: dict[str, float],
    candidate_id: str,
    args: argparse.Namespace,
) -> Path:
    paths = judger01.checkpoint_paths(args.source_workspace, round_idx)
    out = args.workspace_root / "candidates" / f"r{round_idx:03d}_{candidate_id}.pt"
    judger01.softmix_checkpoint(paths["g"], paths["a"], paths["s"], out, weights, force=args.force)
    return out


def candidate_row(round_idx: int, candidate_id: str, weights: dict[str, float], phase: str) -> dict[str, Any]:
    return {"round": round_idx, "candidate_id": candidate_id, "phase": phase, **weights}


def optimise_round(round_idx: int, args: argparse.Namespace, mini_cfg: Path, full_cfg: Path) -> list[dict[str, Any]]:
    rng = random.Random(args.seed + round_idx)
    features = judger01.round_features(args.source_workspace, round_idx)
    bootstrap_weights = judger01.train_judger(args.source_workspace, args.history_rounds, args.workspace_root / "bootstrap")[0].predict(features)

    candidates: list[tuple[str, dict[str, float], str]] = [("bootstrap", bootstrap_weights, "init")]
    for idx, prior in enumerate(fixed_priors()):
        candidates.append((f"prior{idx:02d}", prior, "init"))
    for idx in range(args.random_candidates):
        base = bootstrap_weights if idx % 2 == 0 else fixed_priors()[idx % len(fixed_priors())]
        candidates.append((f"rand{idx:03d}", sample_weights(base, rng, args.dirichlet_concentration), "random"))

    observed: list[dict[str, Any]] = []
    evaluated_keys: set[str] = set()

    def evaluate_candidate(candidate_id: str, weights: dict[str, float], phase: str) -> dict[str, Any]:
        if candidate_id in evaluated_keys:
            raise ValueError(candidate_id)
        evaluated_keys.add(candidate_id)
        ckpt = build_candidate(round_idx, weights, candidate_id, args)
        metrics = eval_checkpoint(ckpt, mini_cfg, f"r{round_idx:03d}_{candidate_id}_mini", args)
        row = {
            **candidate_row(round_idx, candidate_id, weights, phase),
            **features,
            "path": str(ckpt.resolve()),
            "eval_scope": "mini",
            **metrics,
        }
        row["score"] = score_row(row)
        observed.append(row)
        if not args.keep_all_candidates and phase not in {"final", "best"}:
            # Keep only metadata and final candidates to avoid hundreds of MB.
            try:
                ckpt.unlink()
            except OSError:
                pass
        return row

    for candidate_id, weights, phase in candidates:
        evaluate_candidate(candidate_id, weights, phase)

    for iteration in range(args.surrogate_iterations):
        x = np.asarray([flatten_weights({name: float(row[name]) for name in judger01.WEIGHT_FIELDS}) for row in observed], dtype=np.float64)
        y = np.asarray([float(row["score"]) for row in observed], dtype=np.float64)
        model = RandomForestRegressor(
            n_estimators=180,
            max_depth=6,
            min_samples_leaf=1,
            random_state=args.seed + round_idx * 100 + iteration,
        )
        model.fit(x, y)

        pool: list[tuple[float, dict[str, float]]] = []
        best = max(observed, key=lambda row: float(row["score"]))
        best_weights = {name: float(best[name]) for name in judger01.WEIGHT_FIELDS}
        for _ in range(args.surrogate_pool):
            base = best_weights if rng.random() < 0.7 else bootstrap_weights
            weights = sample_weights(base, rng, max(args.dirichlet_concentration * 1.8, 8.0))
            pred = float(model.predict(np.asarray([flatten_weights(weights)], dtype=np.float64))[0])
            pool.append((pred, weights))
        pool.sort(key=lambda item: item[0], reverse=True)
        for idx, (_pred, weights) in enumerate(pool[: args.surrogate_evals]):
            evaluate_candidate(f"sur{iteration:02d}_{idx:02d}", weights, f"surrogate{iteration}")

    ranked = sorted(observed, key=lambda row: float(row["score"]), reverse=True)
    full_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ranked[: args.full_eval_topk]):
        weights = {name: float(row[name]) for name in judger01.WEIGHT_FIELDS}
        candidate_id = f"best{idx:02d}_{row['candidate_id']}"
        ckpt = build_candidate(round_idx, weights, candidate_id, args)
        metrics = eval_checkpoint(ckpt, full_cfg, f"r{round_idx:03d}_{candidate_id}_full", args)
        full = {
            **candidate_row(round_idx, candidate_id, weights, "full"),
            **features,
            "path": str(ckpt.resolve()),
            "eval_scope": "full_total",
            **metrics,
        }
        full["score"] = score_row(full)
        full_rows.append(full)
    return observed + full_rows


def parse_rounds(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.source_workspace = args.source_workspace.expanduser().resolve()
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    (args.workspace_root / "stats").mkdir(parents=True, exist_ok=True)
    mini_list, mini_cfg = ensure_mini_eval_config(args)
    full_cfg = full_eval_config(args)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "dqa_softmox_mix_weight_optimizer_v0",
        "method": "AdaMerging-style black-box coefficient learning with RF surrogate",
        "source_workspace": str(args.source_workspace),
        "workspace": str(args.workspace_root),
        "rounds": parse_rounds(args.rounds),
        "mini_list": str(mini_list),
        "mini_images": args.mini_images,
        "random_candidates": args.random_candidates,
        "surrogate_iterations": args.surrogate_iterations,
        "surrogate_pool": args.surrogate_pool,
        "surrogate_evals": args.surrogate_evals,
    }
    (args.workspace_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for round_idx in parse_rounds(args.rounds):
        rows.extend(optimise_round(round_idx, args, mini_cfg, full_cfg))
        fields = sorted({key for row in rows for key in row.keys()})
        write_csv(args.workspace_root / "stats" / "02_mix_weight_optimizer_trials.csv", rows, fields)

    full_rows = [row for row in rows if row.get("eval_scope") == "full_total"]
    ranked = sorted(full_rows, key=lambda row: float(row.get("score", 0.0)), reverse=True)
    write_csv(args.workspace_root / "stats" / "02_mix_weight_optimizer_best_full.csv", ranked, sorted({key for row in ranked for key in row.keys()}))
    report = [
        "# DQA-SoftMoX Mix Weight Optimizer",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- mini_images: {args.mini_images}",
        "",
        "## Best Full-Total Candidates",
        "",
        "| rank | round | candidate | mAP50 | mAP50:95 | score | body G/A/S | head G/A/S | moe G/A/S |",
        "|---:|---:|---|---:|---:|---:|---|---|---|",
    ]
    for idx, row in enumerate(ranked[:20], start=1):
        report.append(
            "| {idx} | {round} | {candidate_id} | {map50:.3f} | {map50_95:.3f} | {score:.4f} | "
            "{body_g:.2f}/{body_a:.2f}/{body_s:.2f} | {head_g:.2f}/{head_a:.2f}/{head_s:.2f} | "
            "{moe_g:.2f}/{moe_a:.2f}/{moe_s:.2f} |".format(idx=idx, **row)
        )
    (args.workspace_root / "02_mix_weight_optimizer_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(ranked[:10], indent=2, ensure_ascii=False))
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-workspace", type=Path, default=SOURCE_WORKSPACE)
    parser.add_argument("--rounds", default="1,2")
    parser.add_argument("--history-rounds", type=int, default=21)
    parser.add_argument("--mini-images", type=int, default=768)
    parser.add_argument("--random-candidates", type=int, default=10)
    parser.add_argument("--surrogate-iterations", type=int, default=2)
    parser.add_argument("--surrogate-pool", type=int, default=72)
    parser.add_argument("--surrogate-evals", type=int, default=4)
    parser.add_argument("--full-eval-topk", type=int, default=3)
    parser.add_argument("--dirichlet-concentration", type=float, default=14.0)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-all-candidates", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
