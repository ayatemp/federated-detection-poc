#!/usr/bin/env python3
"""Build a DQA-SoftMoX judger probe from existing G/A/S checkpoints.

This probe is deliberately separated from the expensive FL loop.  It reuses the
round artifacts produced by the MOE x DQA 01 run, learns a tiny bootstrap judger
from historical round features, and writes module-wise soft-mixed checkpoints:

    M_t = judger(G_t, A_t, S_t)

where `body`, `head`, and `moe` each get their own weights over `G/A/S`.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
SOURCE_WORKSPACE = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation" / "moe_dqa" / "output" / "01_dqa_fedmox_yolo_full"
DEFAULT_WORKSPACE = PROJECT_ROOT / "output" / "01_judger_probe"
EVAL_SCRIPT = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation" / "scene_daynight_dqa" / "scripts" / "evaluate_scene_daynight_protocol.py"

for path in (
    REPO_ROOT / "navigating_data_heterogeneity" / "vendor" / "efficientteacher",
    REPO_ROOT / "navigating_data_heterogeneity",
    REPO_ROOT / "dynamic_quality_aware_classwise_aggregation",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


FEATURE_FIELDS = [
    "round",
    "g_map50",
    "a_proxy_map50",
    "s_map50",
    "repair_gain_vs_a",
    "repair_gain_vs_g",
    "pseudo_mean_score",
    "pseudo_mean_conf",
    "pseudo_mean_stability",
    "pseudo_boxes",
    "pseudo_images",
    "class_entropy_norm",
    "rare_fraction",
    "vehicle_fraction",
    "expert_entropy_norm",
    "dead_expert_fraction",
]

WEIGHT_FIELDS = [
    "body_g",
    "body_a",
    "body_s",
    "head_g",
    "head_a",
    "head_s",
    "moe_g",
    "moe_a",
    "moe_s",
]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 60.0), -60.0)))


def normalise(weights: list[float]) -> list[float]:
    clipped = [max(float(x), 1e-4) for x in weights]
    total = sum(clipped)
    return [x / total for x in clipped]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def last_map50(results_csv: Path) -> float | None:
    rows = read_csv(results_csv)
    if not rows:
        return None
    row = {k.strip(): v.strip() for k, v in rows[-1].items()}
    try:
        return float(row["metrics/mAP_0.5"])
    except (KeyError, ValueError):
        return None


def run_map50(source_workspace: Path, run_name: str) -> float | None:
    return last_map50(source_workspace / "runs" / run_name / "results.csv")


def round_tag(round_idx: int) -> str:
    return f"round{round_idx:03d}"


def checkpoint_paths(source_workspace: Path, round_idx: int) -> dict[str, Path]:
    ckpt_dir = source_workspace / "checkpoints"
    tag = round_tag(round_idx)
    if round_idx <= 1:
        previous = ckpt_dir / "round000_latent_dqamox_warmup.pt"
    else:
        previous = ckpt_dir / f"latent_dqamox_p1_{round_tag(round_idx - 1)}_server_repair.pt"
    return {
        "g": previous,
        "a": ckpt_dir / f"latent_dqamox_p1_{tag}_dqa_aggregate.pt",
        "s": ckpt_dir / f"latent_dqamox_p1_{tag}_server_repair.pt",
    }


def client_round_map50(source_workspace: Path, round_idx: int) -> float | None:
    pattern = f"sdn18_latent_dqamox_p1_{round_tag(round_idx)}_client*"
    values = [
        value
        for run_dir in sorted((source_workspace / "runs").glob(pattern))
        if (value := last_map50(run_dir / "results.csv")) is not None
    ]
    if not values:
        return None
    return float(np.mean(values))


def checkpoint_source_metrics(source_workspace: Path, round_idx: int) -> dict[str, float]:
    if round_idx <= 1:
        g = run_map50(source_workspace, "sdn18_client_balanced_single_injection_dqamox_warmup")
    else:
        g = run_map50(source_workspace, f"sdn18_latent_dqamox_p1_{round_tag(round_idx - 1)}_server_repair")
    a_proxy = client_round_map50(source_workspace, round_idx)
    s = run_map50(source_workspace, f"sdn18_latent_dqamox_p1_{round_tag(round_idx)}_server_repair")
    return {
        "g_map50": float(g or 0.0),
        "a_proxy_map50": float(a_proxy or 0.0),
        "s_map50": float(s or 0.0),
    }


def pseudo_stats(source_workspace: Path, round_idx: int) -> dict[str, float]:
    stats_dir = source_workspace / "stats"
    boxes: list[dict[str, str]] = []
    for path in sorted(stats_dir.glob(f"03_{round_tag(round_idx)}_client*_stable_boxes.csv")):
        boxes.extend(read_csv(path))
    if not boxes:
        return {
            "pseudo_mean_score": 0.0,
            "pseudo_mean_conf": 0.0,
            "pseudo_mean_stability": 0.0,
            "pseudo_boxes": 0.0,
            "pseudo_images": 0.0,
            "class_entropy_norm": 0.0,
            "rare_fraction": 0.0,
            "vehicle_fraction": 0.0,
        }

    scores = [float(row.get("score") or 0.0) for row in boxes]
    confs = [float(row.get("conf") or 0.0) for row in boxes]
    stability = [float(row.get("stability") or 0.0) for row in boxes]
    classes = [int(row.get("class_id") or 0) for row in boxes]
    image_count = len({row.get("source_image") or row.get("image") for row in boxes})
    counts = np.bincount(classes, minlength=10).astype(np.float64)
    probs = counts / max(counts.sum(), 1.0)
    entropy = -float(np.sum([p * math.log(p + 1e-12) for p in probs])) / math.log(len(probs))
    rare = float(counts[[1, 5, 6, 9]].sum() / max(counts.sum(), 1.0))
    vehicle = float(counts[[2, 7, 8]].sum() / max(counts.sum(), 1.0))
    return {
        "pseudo_mean_score": float(np.mean(scores)),
        "pseudo_mean_conf": float(np.mean(confs)),
        "pseudo_mean_stability": float(np.mean(stability)),
        "pseudo_boxes": float(len(boxes)),
        "pseudo_images": float(image_count),
        "class_entropy_norm": entropy,
        "rare_fraction": rare,
        "vehicle_fraction": vehicle,
    }


def expert_stats(source_workspace: Path, round_idx: int) -> dict[str, float]:
    path = source_workspace / "stats" / f"05_{round_tag(round_idx)}_expert_choice_boxes.csv"
    rows = read_csv(path)
    if not rows:
        return {"expert_entropy_norm": 0.0, "dead_expert_fraction": 1.0}
    expert_ids = [int(row.get("expert_id") or 0) for row in rows]
    counts = np.bincount(expert_ids, minlength=4).astype(np.float64)
    probs = counts / max(counts.sum(), 1.0)
    entropy = -float(np.sum([p * math.log(p + 1e-12) for p in probs if p > 0.0])) / math.log(len(probs))
    dead = float((counts <= 0).sum() / len(counts))
    return {"expert_entropy_norm": entropy, "dead_expert_fraction": dead}


def round_features(source_workspace: Path, round_idx: int) -> dict[str, float]:
    metrics = checkpoint_source_metrics(source_workspace, round_idx)
    features = {
        "round": float(round_idx),
        **metrics,
        **pseudo_stats(source_workspace, round_idx),
        **expert_stats(source_workspace, round_idx),
    }
    features["repair_gain_vs_a"] = features["s_map50"] - features["a_proxy_map50"]
    features["repair_gain_vs_g"] = features["s_map50"] - features["g_map50"]
    return features


def bootstrap_target_weights(features: dict[str, float]) -> dict[str, float]:
    """Create conservative pseudo-labels for the initial judger.

    This is not the final scientific answer; it is a bootstrap target that turns
    historical traces into a trainable small model.  Later runs can replace this
    target with optimized coefficients.
    """

    repair_trust = sigmoid(35.0 * (features["repair_gain_vs_a"] + 0.003))
    client_quality = max(0.0, min(1.0, 0.55 * features["pseudo_mean_score"] + 0.25 * features["class_entropy_norm"] + 0.20 * features["expert_entropy_norm"]))
    rare_missing = max(0.0, min(1.0, (0.035 - features["rare_fraction"]) / 0.035))
    expert_collapse = max(0.0, min(1.0, 1.0 - features["expert_entropy_norm"]))

    body = normalise([
        0.10 + 0.15 * (1.0 - client_quality),
        0.70 + 0.20 * client_quality - 0.10 * repair_trust,
        0.20 * repair_trust,
    ])
    head = normalise([
        0.15 + 0.20 * rare_missing + 0.10 * (1.0 - repair_trust),
        0.25 + 0.25 * client_quality * (1.0 - rare_missing),
        0.30 + 0.45 * repair_trust + 0.20 * rare_missing,
    ])
    moe = normalise([
        0.10 + 0.25 * expert_collapse,
        0.75 + 0.20 * client_quality - 0.25 * expert_collapse,
        0.10 + 0.10 * repair_trust,
    ])
    values = body + head + moe
    return dict(zip(WEIGHT_FIELDS, values, strict=True))


@dataclass
class JudgerModel:
    model: Any

    def predict(self, features: dict[str, float]) -> dict[str, float]:
        x = np.asarray([[features[name] for name in FEATURE_FIELDS]], dtype=np.float64)
        pred = np.asarray(self.model.predict(x)[0], dtype=np.float64)
        out: dict[str, float] = {}
        for offset, prefix in ((0, "body"), (3, "head"), (6, "moe")):
            values = normalise(pred[offset : offset + 3].tolist())
            out[f"{prefix}_g"], out[f"{prefix}_a"], out[f"{prefix}_s"] = values
        return out


def train_judger(source_workspace: Path, history_rounds: int, workspace: Path) -> tuple[JudgerModel, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    xs: list[list[float]] = []
    ys: list[list[float]] = []
    for round_idx in range(1, history_rounds + 1):
        paths = checkpoint_paths(source_workspace, round_idx)
        if not all(path.exists() for path in paths.values()):
            continue
        features = round_features(source_workspace, round_idx)
        target = bootstrap_target_weights(features)
        xs.append([features[name] for name in FEATURE_FIELDS])
        ys.append([target[name] for name in WEIGHT_FIELDS])
        rows.append({**features, **target})

    if len(xs) < 2:
        raise RuntimeError("Need at least two historical rounds to train the bootstrap judger.")

    base = RandomForestRegressor(
        n_estimators=160,
        max_depth=5,
        min_samples_leaf=1,
        random_state=20260513,
    )
    model = make_pipeline(StandardScaler(), MultiOutputRegressor(base))
    model.fit(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64))
    workspace.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_fields": FEATURE_FIELDS, "weight_fields": WEIGHT_FIELDS}, workspace / "judger_v0.joblib")
    write_csv(workspace / "stats" / "01_judger_training_table.csv", rows, FEATURE_FIELDS + WEIGHT_FIELDS)
    return JudgerModel(model), rows


def module_for_key(key: str) -> str:
    if "head.router" in key or "head.expert_m" in key:
        return "moe"
    if key.startswith("head."):
        return "head"
    return "body"


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


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


def mix_state_dicts(
    g_state: dict[str, torch.Tensor],
    a_state: dict[str, torch.Tensor],
    s_state: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    mixed: dict[str, torch.Tensor] = {}
    for key, a_value in a_state.items():
        g_value = g_state.get(key)
        s_value = s_state.get(key)
        if (
            torch.is_tensor(a_value)
            and torch.is_tensor(g_value)
            and torch.is_tensor(s_value)
            and a_value.shape == g_value.shape == s_value.shape
            and a_value.dtype.is_floating_point
        ):
            module = module_for_key(key)
            wg = float(weights[f"{module}_g"])
            wa = float(weights[f"{module}_a"])
            ws = float(weights[f"{module}_s"])
            mixed[key] = (wg * g_value.float() + wa * a_value.float() + ws * s_value.float()).to(a_value.dtype)
        else:
            mixed[key] = a_value
    return mixed


def softmix_checkpoint(g_path: Path, a_path: Path, s_path: Path, output: Path, weights: dict[str, float], *, force: bool = False) -> None:
    if output.exists() and not force:
        return
    g_ckpt = load_checkpoint(g_path)
    a_ckpt = load_checkpoint(a_path)
    s_ckpt = load_checkpoint(s_path)
    out = copy.deepcopy(a_ckpt)
    for field in ("model", "ema"):
        g_state = state_dict_from(g_ckpt, field)
        a_state = state_dict_from(a_ckpt, field)
        s_state = state_dict_from(s_ckpt, field)
        if g_state is None or a_state is None or s_state is None:
            continue
        replace_state(out, field, mix_state_dicts(g_state, a_state, s_state, weights))
    out["epoch"] = -1
    out["optimizer"] = None
    out["judger_softmix"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "weights": weights,
        "sources": {"g": str(g_path), "a": str(a_path), "s": str(s_path)},
        "module_policy": "body/head/moe module-wise G-A-S mixture",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)


def parse_rounds(args: argparse.Namespace) -> list[int]:
    if args.rounds:
        return sorted({int(item.strip()) for item in args.rounds.split(",") if item.strip()})
    return list(range(1, args.max_round + 1))


def evaluate_checkpoints(args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--workspace",
        str(args.source_workspace),
        "--splits",
        args.eval_splits,
        "--batch-size",
        str(args.val_batch_size),
        "--no-plots",
    ]
    for row in records:
        cmd.extend(["--checkpoint", f"{row['label']}={row['path']}"])
    print("Running eval:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def collect_eval_rows(args: argparse.Namespace, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = args.source_workspace / "validation_reports" / "paper_protocol_eval_summary.csv"
    rows = read_csv(summary)
    by_label = {
        row["checkpoint_label"]: row
        for row in rows
        if row.get("split") in {"total", "scene_daynight_total"}
    }
    record_meta = {row["label"]: row for row in records}
    out: list[dict[str, Any]] = []
    for label, meta in record_meta.items():
        metric = by_label.get(label, {})
        out.append({**meta, **{f"eval_{k}": v for k, v in metric.items()}})
    return out


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.source_workspace = args.source_workspace.expanduser().resolve()
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    (args.workspace_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (args.workspace_root / "stats").mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "dqa_softmox_judger_probe_v0",
        "source_workspace": str(args.source_workspace),
        "workspace": str(args.workspace_root),
        "history_rounds": args.history_rounds,
        "rounds": parse_rounds(args),
        "design": "bootstrap ML judger predicts body/head/moe module-wise G/A/S softmix weights",
    }
    (args.workspace_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    judger, _ = train_judger(args.source_workspace, args.history_rounds, args.workspace_root)
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for round_idx in parse_rounds(args):
        paths = checkpoint_paths(args.source_workspace, round_idx)
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            print(f"Skipping round {round_idx}; missing {missing[0]}")
            continue
        features = round_features(args.source_workspace, round_idx)
        weights = judger.predict(features)
        label = f"judger_softmix_p1_{round_tag(round_idx)}"
        output = args.workspace_root / "checkpoints" / f"{label}.pt"
        if not args.setup_only:
            softmix_checkpoint(paths["g"], paths["a"], paths["s"], output, weights, force=args.force)
        row = {
            "label": label,
            "round": round_idx,
            "path": str(output.resolve()),
            **features,
            **weights,
        }
        rows.append(row)
        records.append({"label": label, "path": str(output.resolve())})

    write_csv(args.workspace_root / "stats" / "01_judger_softmix_rounds.csv", rows, ["label", "round", "path", *FEATURE_FIELDS, *WEIGHT_FIELDS])
    if args.evaluate and not args.setup_only:
        evaluate_checkpoints(args, records)
        eval_rows = collect_eval_rows(args, records)
        fields = sorted({key for row in eval_rows for key in row.keys()})
        write_csv(args.workspace_root / "stats" / "01_judger_softmix_eval.csv", eval_rows, fields)

    report_lines = [
        "# DQA-SoftMoX Judger Probe 01",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- source_workspace: `{args.source_workspace}`",
        f"- rounds: `{parse_rounds(args)}`",
        "",
        "## SoftMix Weights",
        "",
        "| round | body G/A/S | head G/A/S | moe G/A/S | repair_gain_vs_a | pseudo_score | expert_entropy |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            "| {round} | {body_g:.3f}/{body_a:.3f}/{body_s:.3f} | {head_g:.3f}/{head_a:.3f}/{head_s:.3f} | "
            "{moe_g:.3f}/{moe_a:.3f}/{moe_s:.3f} | {repair_gain_vs_a:.4f} | {pseudo_mean_score:.3f} | {expert_entropy_norm:.3f} |".format(**row)
        )
    (args.workspace_root / "01_judger_probe_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-workspace", type=Path, default=SOURCE_WORKSPACE)
    parser.add_argument("--history-rounds", type=int, default=21)
    parser.add_argument("--rounds", default="")
    parser.add_argument("--max-round", type=int, default=2)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-splits", default="total")
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
