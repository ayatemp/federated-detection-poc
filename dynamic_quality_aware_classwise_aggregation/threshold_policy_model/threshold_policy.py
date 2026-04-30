"""Train and run a lightweight learned pseudoGT threshold policy.

The policy is trained from DQA05 logs.  DQA05 only contains one fixed-threshold
trajectory, so the targets here are offline oracle targets derived from the next
round's validation drift plus per-client/class pseudo-label quality statistics.
That makes this a deployable learned policy prototype, not proof that the
targets are globally optimal without a threshold sweep.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold


CLASS_NAMES = [
    "person",
    "rider",
    "car",
    "bus",
    "truck",
    "bike",
    "motor",
    "traffic light",
    "traffic sign",
    "train",
]

TARGET_COLUMNS = ["target_low", "target_high", "target_nms"]
SERVER_RE = re.compile(r"dqa_phase2_round(?P<round>\d{3})_server")
CLIENT_STATS_RE = re.compile(r"phase2_round(?P<round>\d{3})_client(?P<client>\d+)\.json")


@dataclass(frozen=True)
class PolicyPaths:
    stats_root: Path
    run_root: Path
    output_dir: Path


def clamp_array(values: Any, low: float, high: float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), low, high)


def weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if values.size == 0:
        return 0.35
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    total = float(weights.sum())
    if total <= 0:
        return float(np.quantile(values, q))
    cutoff = total * q
    running = np.cumsum(weights)
    return float(values[np.searchsorted(running, cutoff, side="left")])


def read_server_metrics(run_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result_path in sorted((run_root / "runs").glob("dqa_phase2_round*_server/results.csv")):
        match = SERVER_RE.search(result_path.parent.name)
        if not match:
            continue
        frame = pd.read_csv(result_path)
        frame.columns = [column.strip() for column in frame.columns]
        last = frame.iloc[-1]
        rows.append(
            {
                "round": int(match.group("round")),
                "server_precision": float(last["metrics/precision"]),
                "server_recall": float(last["metrics/recall"]),
                "server_map50": float(last["metrics/mAP_0.5"]),
                "server_map95": float(last["metrics/mAP_0.5:0.95"]),
            }
        )
    metrics = pd.DataFrame(rows).sort_values("round").reset_index(drop=True)
    if metrics.empty:
        raise FileNotFoundError(f"No phase2 server results.csv files under {run_root / 'runs'}")
    metrics["server_map95_prev"] = metrics["server_map95"].shift(1).fillna(metrics["server_map95"])
    metrics["server_map95_next"] = metrics["server_map95"].shift(-1)
    metrics["server_map95_delta"] = metrics["server_map95"] - metrics["server_map95_prev"]
    metrics["server_map95_next_delta"] = metrics["server_map95_next"] - metrics["server_map95"]
    metrics["server_map95_best_so_far"] = metrics["server_map95"].cummax()
    metrics["server_gap_to_best"] = metrics["server_map95_best_so_far"] - metrics["server_map95"]
    return metrics


def _safe_mean(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def read_client_stats(stats_root: Path, class_names: list[str] | None = None) -> pd.DataFrame:
    class_names = class_names or CLASS_NAMES
    paths = sorted(stats_root.glob("phase2_round*_client*.json"))
    if not paths:
        raise FileNotFoundError(f"No phase2 client stats found under {stats_root}")

    max_round = max(int(CLIENT_STATS_RE.search(path.name).group("round")) for path in paths if CLIENT_STATS_RE.search(path.name))
    previous: dict[tuple[int, int], dict[str, float]] = {}
    rows: list[dict[str, Any]] = []

    for path in paths:
        match = CLIENT_STATS_RE.search(path.name)
        if not match:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        client_id = int(payload["id"])
        round_idx = int(payload["round"])
        counts = [float(value) for value in payload["counts"]]
        total_count = float(sum(counts))
        total_quality = float(sum(float(value) for value in payload["quality_sums"]))
        total_conf = float(sum(float(value) for value in payload["confidence_sums"]))
        client_mean_quality = _safe_mean(total_quality, total_count)
        client_mean_conf = _safe_mean(total_conf, total_count)
        apply_round = round_idx + 1

        for class_idx, class_name in enumerate(class_names):
            count = counts[class_idx]
            quality = float(payload["mean_quality_scores"][class_idx])
            confidence = float(payload["mean_confidences"][class_idx])
            objectness = float(payload["mean_objectness"][class_idx])
            class_confidence = float(payload["mean_class_confidences"][class_idx])
            localization = float(payload["mean_localization_qualities"][class_idx])
            prev = previous.get((client_id, class_idx), {})
            prev_count = float(prev.get("count", count))
            prev_quality = float(prev.get("quality", quality))
            prev_confidence = float(prev.get("confidence", confidence))
            prev_objectness = float(prev.get("objectness", objectness))

            row: dict[str, Any] = {
                "round": round_idx,
                "apply_round": apply_round,
                "round_norm": (round_idx - 1) / max(max_round - 1, 1),
                "apply_round_norm": (apply_round - 1) / max(max_round, 1),
                "client_id": client_id,
                "class_index": class_idx,
                "class_name": class_name,
                "count": count,
                "log_count": math.log1p(count),
                "class_share": count / max(total_count, 1.0),
                "client_total_count": total_count,
                "log_client_total_count": math.log1p(total_count),
                "client_mean_quality": client_mean_quality,
                "client_mean_confidence": client_mean_conf,
                "mean_confidence": confidence,
                "mean_objectness": objectness,
                "mean_class_confidence": class_confidence,
                "mean_localization_quality": localization,
                "mean_quality": quality,
                "prev_count": prev_count,
                "prev_log_count": math.log1p(prev_count),
                "count_ratio": count / max(prev_count, 1.0),
                "log_count_delta": math.log1p(count) - math.log1p(prev_count),
                "quality_delta": quality - prev_quality,
                "confidence_delta": confidence - prev_confidence,
                "objectness_delta": objectness - prev_objectness,
                "rare_relief": max(0.0, min(1.0, (250.0 - count) / 250.0)),
                "stats_path": str(path),
            }
            for cid in range(3):
                row[f"client_{cid}"] = 1.0 if client_id == cid else 0.0
            for cidx in range(len(class_names)):
                row[f"class_{cidx}"] = 1.0 if class_idx == cidx else 0.0
            rows.append(row)

            previous[(client_id, class_idx)] = {
                "count": count,
                "quality": quality,
                "confidence": confidence,
                "objectness": objectness,
            }

    return pd.DataFrame(rows).sort_values(["round", "client_id", "class_index"]).reset_index(drop=True)


def add_oracle_targets(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    future_gap = (data["server_map95_best_so_far"] - data["server_map95_next"]).clip(lower=0.0)
    future_drop = (data["server_map95"] - data["server_map95_next"]).clip(lower=0.0)
    future_risk = (0.60 * (future_gap / 0.012) + 0.40 * (future_drop / 0.006)).clip(0.0, 1.0)
    quality_gap = ((0.78 - data["mean_quality"]) / 0.22).clip(-0.25, 1.0)
    quality_drop = ((-data["quality_delta"]) / 0.12).clip(0.0, 1.0)
    count_pressure = ((data["log_count_delta"] - math.log1p(0.10)) / 0.75).clip(0.0, 1.0)
    abundance = (data["log_count"] / data["log_client_total_count"].clip(lower=1.0)).clip(0.0, 1.0)

    risk_score = (
        0.42 * future_risk
        + 0.22 * quality_gap
        + 0.18 * quality_drop
        + 0.12 * count_pressure
        + 0.06 * abundance
        - 0.08 * data["rare_relief"]
    ).clip(0.0, 1.0)

    low = 0.35 + 0.22 * risk_score
    high = low + 0.40 + 0.05 * future_risk + 0.03 * quality_gap.clip(0.0, 1.0)
    nms = low - 0.02 + 0.05 * future_risk

    data["future_risk"] = future_risk
    data["oracle_risk_score"] = risk_score
    data["target_low"] = clamp_array(low, 0.35, 0.58)
    data["target_high"] = clamp_array(high, 0.75, 0.92)
    data["target_nms"] = clamp_array(nms, 0.35, 0.48)
    return data


def build_feature_columns(num_classes: int = 10, num_clients: int = 3) -> list[str]:
    base = [
        "round_norm",
        "apply_round_norm",
        "client_id",
        "class_index",
        "count",
        "log_count",
        "class_share",
        "client_total_count",
        "log_client_total_count",
        "client_mean_quality",
        "client_mean_confidence",
        "mean_confidence",
        "mean_objectness",
        "mean_class_confidence",
        "mean_localization_quality",
        "mean_quality",
        "prev_count",
        "prev_log_count",
        "count_ratio",
        "log_count_delta",
        "quality_delta",
        "confidence_delta",
        "objectness_delta",
        "rare_relief",
    ]
    return base + [f"client_{idx}" for idx in range(num_clients)] + [f"class_{idx}" for idx in range(num_classes)]


def build_policy_dataset(stats_root: Path, run_root: Path, class_names: list[str] | None = None) -> pd.DataFrame:
    stats = read_client_stats(stats_root, class_names=class_names)
    metrics = read_server_metrics(run_root)
    data = stats.merge(metrics, on="round", how="left")
    data = add_oracle_targets(data)
    return data


def rule_policy_predictions(data: pd.DataFrame) -> pd.DataFrame:
    progress = ((data["apply_round"] - 3 + 1) / 10.0).clip(0.0, 1.0)
    quality_penalty = ((0.775 - data["mean_quality"]) * 0.80).clip(-0.03, 0.06)
    quality_drop_penalty = ((-data["quality_delta"]) * 1.50).clip(0.0, 0.04)
    count_pressure = ((np.log(np.maximum(data["count_ratio"], 1e-6)) - math.log(1.10)) * 0.05).clip(0.0, 0.04)
    low = 0.35 + 0.10 * progress + quality_penalty + quality_drop_penalty + count_pressure - 0.05 * data["rare_relief"]
    high = low + 0.40 + 0.05 * progress
    return pd.DataFrame(
        {
            "target_low": clamp_array(low, 0.35, 0.55),
            "target_high": clamp_array(high, 0.75, 0.90),
            "target_nms": clamp_array(low - 0.02, 0.35, 0.45),
        },
        index=data.index,
    )


def train_model(data: pd.DataFrame, feature_columns: list[str]) -> tuple[ExtraTreesRegressor, dict[str, Any], pd.DataFrame]:
    train = data.dropna(subset=TARGET_COLUMNS).copy()
    if train["round"].nunique() < 3:
        raise ValueError("Need at least 3 phase2 rounds with next-round metrics to train/evaluate the policy.")

    x = train[feature_columns]
    y = train[TARGET_COLUMNS]
    groups = train["round"].to_numpy()
    model = ExtraTreesRegressor(
        n_estimators=96,
        max_depth=6,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )

    n_splits = min(5, train["round"].nunique())
    cv_pred = np.zeros((len(train), len(TARGET_COLUMNS)), dtype=float)
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(x, y, groups=groups):
        fold_model = clone(model)
        fold_model.fit(x.iloc[train_idx], y.iloc[train_idx])
        cv_pred[test_idx] = fold_model.predict(x.iloc[test_idx])
    cv_pred = clip_predictions(cv_pred)

    fixed_pred = pd.DataFrame(
        {
            "target_low": 0.35,
            "target_high": 0.75,
            "target_nms": 0.35,
        },
        index=train.index,
    )
    rule_pred = rule_policy_predictions(train)

    report = {
        "num_rows": int(len(train)),
        "num_rounds": int(train["round"].nunique()),
        "feature_count": int(len(feature_columns)),
        "cv_splits": int(n_splits),
        "target_columns": TARGET_COLUMNS,
        "model": "ExtraTreesRegressor(n_estimators=96,max_depth=6,min_samples_leaf=4)",
        "cv_model_mae": metric_dict(y, cv_pred),
        "fixed_05_mae": metric_dict(y, fixed_pred[TARGET_COLUMNS].to_numpy()),
        "hand_rule_mae": metric_dict(y, rule_pred[TARGET_COLUMNS].to_numpy()),
        "cv_model_rmse": metric_dict(y, cv_pred, squared=True),
    }

    full_model = clone(model)
    full_model.fit(x, y)
    cv_frame = train[["round", "apply_round", "client_id", "class_index", "class_name", *TARGET_COLUMNS]].copy()
    for idx, column in enumerate(TARGET_COLUMNS):
        cv_frame[f"cv_pred_{column.replace('target_', '')}"] = cv_pred[:, idx]
        cv_frame[f"fixed_{column.replace('target_', '')}"] = fixed_pred[column].to_numpy()
        cv_frame[f"rule_{column.replace('target_', '')}"] = rule_pred[column].to_numpy()
    return full_model, report, cv_frame


def metric_dict(y_true: pd.DataFrame, y_pred: np.ndarray, squared: bool = False) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for idx, column in enumerate(TARGET_COLUMNS):
        if squared:
            value = mean_squared_error(y_true[column], y_pred[:, idx]) ** 0.5
        else:
            value = mean_absolute_error(y_true[column], y_pred[:, idx])
        metrics[column] = round(float(value), 6)
    metrics["mean"] = round(float(np.mean(list(metrics.values()))), 6)
    return metrics


def clip_predictions(predictions: np.ndarray) -> np.ndarray:
    clipped = np.asarray(predictions, dtype=float).copy()
    clipped[:, 0] = clamp_array(clipped[:, 0], 0.35, 0.58)
    clipped[:, 1] = clamp_array(clipped[:, 1], 0.75, 0.92)
    clipped[:, 1] = np.maximum(clipped[:, 1], clipped[:, 0] + 0.25)
    clipped[:, 1] = clamp_array(clipped[:, 1], 0.75, 0.92)
    clipped[:, 2] = clamp_array(clipped[:, 2], 0.35, 0.48)
    return clipped


def save_bundle(
    model: ExtraTreesRegressor,
    feature_columns: list[str],
    output_path: Path,
    report: dict[str, Any],
    class_names: list[str] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "target_columns": TARGET_COLUMNS,
        "class_names": class_names or CLASS_NAMES,
        "report": report,
    }
    joblib.dump(bundle, output_path)


def load_bundle(path: Path) -> dict[str, Any]:
    return joblib.load(path)


def predict_policy(bundle: dict[str, Any], data: pd.DataFrame) -> pd.DataFrame:
    feature_columns = bundle["feature_columns"]
    predictions = clip_predictions(bundle["model"].predict(data[feature_columns]))
    out = data[["round", "apply_round", "client_id", "class_index", "class_name", "count", "mean_quality"]].copy()
    out["pred_low"] = predictions[:, 0]
    out["pred_high"] = predictions[:, 1]
    out["pred_nms"] = predictions[:, 2]
    return out


def summarize_latest_decision(predictions: pd.DataFrame, class_names: list[str] | None = None) -> dict[str, Any]:
    class_names = class_names or CLASS_NAMES
    latest_round = int(predictions["round"].max())
    latest = predictions[predictions["round"] == latest_round].copy()
    decisions: list[dict[str, Any]] = []
    for client_id, group in latest.groupby("client_id"):
        group = group.sort_values("class_index")
        lows = [round(float(value), 4) for value in group["pred_low"]]
        highs = [round(float(value), 4) for value in group["pred_high"]]
        nms = weighted_percentile(group["pred_nms"].to_numpy(), group["count"].to_numpy(), 0.25)
        mean_low = float(np.mean(lows))
        risk = max(0.0, min(1.0, (mean_low - 0.35) / 0.22))
        decisions.append(
            {
                "client_id": int(client_id),
                "source_round": latest_round,
                "apply_round": int(group["apply_round"].iloc[0]),
                "nms_conf_thres": round(float(np.clip(nms, 0.35, 0.48)), 4),
                "ignore_thres_low": lows,
                "ignore_thres_high": highs,
                "teacher_loss_weight": round(float(np.clip(0.35 - 0.10 * risk, 0.22, 0.35)), 4),
                "class_names": class_names,
            }
        )
    return {"source": "dqa05_learned_threshold_policy", "latest_source_round": latest_round, "clients": decisions}


def run_training(paths: PolicyPaths, class_names: list[str] | None = None) -> dict[str, Any]:
    class_names = class_names or CLASS_NAMES
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_policy_dataset(paths.stats_root, paths.run_root, class_names=class_names)
    feature_columns = build_feature_columns(num_classes=len(class_names), num_clients=int(dataset["client_id"].max()) + 1)
    model, report, cv_predictions = train_model(dataset, feature_columns)

    model_path = paths.output_dir / "dqa05_threshold_policy.joblib"
    save_bundle(model, feature_columns, model_path, report, class_names=class_names)
    bundle = load_bundle(model_path)
    predictions = predict_policy(bundle, dataset)
    latest_decision = summarize_latest_decision(predictions, class_names=class_names)

    dataset.to_csv(paths.output_dir / "dqa05_policy_dataset.csv", index=False)
    cv_predictions.to_csv(paths.output_dir / "dqa05_policy_cv_predictions.csv", index=False)
    predictions.to_csv(paths.output_dir / "dqa05_threshold_predictions.csv", index=False)
    (paths.output_dir / "latest_policy_decision.json").write_text(
        json.dumps(latest_decision, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    final_report = {
        **report,
        "model_path": str(model_path),
        "dataset_path": str(paths.output_dir / "dqa05_policy_dataset.csv"),
        "predictions_path": str(paths.output_dir / "dqa05_threshold_predictions.csv"),
        "latest_decision_path": str(paths.output_dir / "latest_policy_decision.json"),
        "label_note": (
            "Targets are offline oracle thresholds derived from DQA05 next-round server mAP drift "
            "and pseudo-label quality/count statistics. They are not direct labels from a threshold sweep."
        ),
    }
    (paths.output_dir / "dqa05_threshold_policy_report.json").write_text(
        json.dumps(final_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {**final_report, "latest_decision": latest_decision}
