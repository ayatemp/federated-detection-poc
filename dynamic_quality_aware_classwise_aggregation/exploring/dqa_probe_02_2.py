from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def find_repo_root(start: Path | None = None) -> Path:
    root = (start or Path.cwd()).resolve()
    while root.name != "Object_Detection" and root.parent != root:
        root = root.parent
    if root.name != "Object_Detection":
        raise RuntimeError("Run from inside /app/Object_Detection")
    return root


@dataclass
class ProbeConfig:
    source_warmup_key: str = "dqa_v2_scene_12h"
    experiment_name: str = "02_2_dqa_functionality_probe_7h"
    server_train_limit: int = 2048
    server_val_limit: int = 1024
    client_target_limit: int = 2048
    seed: int = 20260429
    max_wall_hours: float = 7.0
    device_cli: str = "0"
    config_device: str = ""
    img_size: int = 640
    batch_size: int = 8
    val_batch_size: int = 1
    workers: int = 0
    force_rerun: bool = False
    run_client_local_eval: bool = True
    run_server_calibration: bool = True
    skip_after_train_best_val: bool = True
    python_executable: str | None = None


def default_client_variants() -> list[dict[str, Any]]:
    strict = {
        "nms_conf_thres": 0.35,
        "ignore_thres_low": 0.35,
        "ignore_thres_high": 0.75,
        "teacher_loss_weight": 0.5,
        "box_loss_weight": 0.03,
        "obj_loss_weight": 0.5,
    }
    very_strict = {
        "nms_conf_thres": 0.50,
        "ignore_thres_low": 0.50,
        "ignore_thres_high": 0.85,
        "teacher_loss_weight": 0.35,
        "box_loss_weight": 0.02,
        "obj_loss_weight": 0.35,
    }
    return [
        {
            "name": "ssod_default_all_lr1e-2",
            "epochs": 1,
            "lr0": 1e-2,
            "train_scope": "all",
            "ssod_overrides": {},
            "note": "Current phase-2-like stress test: permissive pseudo labels and high LR.",
        },
        {
            "name": "ssod_default_all_lr1e-3",
            "epochs": 1,
            "lr0": 1e-3,
            "train_scope": "all",
            "ssod_overrides": {},
            "note": "Same pseudo-label thresholds, but smaller client update magnitude.",
        },
        {
            "name": "ssod_strict_all_lr3e-4",
            "epochs": 1,
            "lr0": 3e-4,
            "train_scope": "all",
            "ssod_overrides": strict,
            "note": "Strict pseudo labels plus low LR; primary candidate for making DQA work.",
        },
        {
            "name": "ssod_strict_neck_head_lr1e-3",
            "epochs": 1,
            "lr0": 1e-3,
            "train_scope": "neck_head",
            "ssod_overrides": strict,
            "note": "Keep feature extractor steadier; test if head-side adaptation is enough.",
        },
        {
            "name": "ssod_very_strict_all_lr3e-4",
            "epochs": 1,
            "lr0": 3e-4,
            "train_scope": "all",
            "ssod_overrides": very_strict,
            "note": "Tests whether very sparse/high-confidence pseudo labels are safer.",
        },
    ]


def default_aggregation_variants() -> list[dict[str, Any]]:
    return [
        {"name": "fedavg", "kind": "fedavg", "note": "Plain BN-local FedAvg reference."},
        {
            "name": "dqa_v2_default",
            "kind": "dqa_v2",
            "config": {"min_server_alpha": 0.45},
            "note": "Current v2-style server residual anchor.",
        },
        {
            "name": "dqa_v2_conservative",
            "kind": "dqa_v2",
            "config": {
                "min_server_alpha": 0.70,
                "residual_blend": 0.12,
                "classwise_blend": 0.18,
                "temperature": 2.0,
                "uniform_mix": 0.10,
            },
            "note": "DQA with a strong server floor and small client residuals.",
        },
        {
            "name": "dqa_v2_tiny_residual",
            "kind": "dqa_v2",
            "config": {
                "min_server_alpha": 0.80,
                "residual_blend": 0.05,
                "classwise_blend": 0.10,
                "temperature": 2.5,
                "uniform_mix": 0.10,
            },
            "note": "Nearly guarded DQA; useful if client pseudo updates are mostly harmful.",
        },
    ]


def default_server_patterns() -> list[dict[str, Any]]:
    return [
        {
            "name": "server_oracle_cloudy_all_lr3e-4_ep3",
            "epochs": 3,
            "lr0": 3e-4,
            "train_scope": "all",
            "weight_decay": 5e-4,
            "note": "Supervised cloudy-only continuation upper-bound on the same lists.",
        }
    ]


def default_calibration_variants() -> list[dict[str, Any]]:
    return [
        {
            "name": "cal_all_lr3e-4_ep1",
            "epochs": 1,
            "lr0": 3e-4,
            "train_scope": "all",
            "weight_decay": 5e-4,
            "note": "Low-LR server repair after aggregation.",
        },
        {
            "name": "cal_all_lr1e-3_ep1",
            "epochs": 1,
            "lr0": 1e-3,
            "train_scope": "all",
            "weight_decay": 5e-4,
            "note": "Slightly stronger server repair after aggregation.",
        },
    ]


class DQAFunctionalityProbe:
    def __init__(
        self,
        cfg: ProbeConfig | None = None,
        *,
        client_variants: list[dict[str, Any]] | None = None,
        aggregation_variants: list[dict[str, Any]] | None = None,
        server_patterns: list[dict[str, Any]] | None = None,
        calibration_variants: list[dict[str, Any]] | None = None,
    ) -> None:
        self.cfg = cfg or ProbeConfig()
        self.root = find_repo_root()
        self.dqa_root = self.root / "dynamic_quality_aware_classwise_aggregation"
        self.nav_root = self.root / "navigating_data_heterogeneity"
        self.et_root = self.nav_root / "vendor" / "efficientteacher"
        self.exp_root = self.dqa_root / "exploring" / "runs" / self.cfg.experiment_name
        self.full_list_root = self.exp_root / "data_lists_full"
        self.mini_list_root = self.exp_root / "mini_lists"
        self.config_root = self.exp_root / "configs"
        self.run_root = self.exp_root / "runs"
        self.result_root = self.exp_root / "results"
        self.log_root = self.result_root / "logs"
        self.eval_root = self.result_root / "evals"
        self.stats_root = self.result_root / "pseudo_stats"
        self.aggregate_root = self.exp_root / "aggregates"
        self.state_root = self.exp_root / "dqa_states"
        self.deadline = time.monotonic() + self.cfg.max_wall_hours * 3600.0
        self.client_variants = client_variants or default_client_variants()
        self.aggregation_variants = aggregation_variants or default_aggregation_variants()
        self.server_patterns = server_patterns or default_server_patterns()
        self.calibration_variants = calibration_variants or default_calibration_variants()
        self.available_warmups = {
            "dqa_corrected_12h": self.dqa_root
            / "efficientteacher_dqa_cwa_corrected_12h"
            / "global_checkpoints"
            / "round000_warmup.pt",
            "dqa_v2_weather": self.dqa_root
            / "efficientteacher_dqa_ver2"
            / "global_checkpoints"
            / "round000_warmup.pt",
            "dqa_v2_scene_12h": self.dqa_root
            / "efficientteacher_dqa_ver2_scene_12h"
            / "global_checkpoints"
            / "round000_warmup.pt",
        }
        self.warmup_ckpt = self.available_warmups[self.cfg.source_warmup_key]
        preferred_python = Path("/root/micromamba/envs/al_yolov8/bin/python")
        self.python_executable = (
            self.cfg.python_executable
            or (str(preferred_python) if preferred_python.exists() else sys.executable)
        )
        self._imports_ready = False

    def describe(self) -> dict[str, Any]:
        return {
            "experiment_root": str(self.exp_root),
            "warmup": str(self.warmup_ckpt),
            "client_variants": [item["name"] for item in self.client_variants],
            "aggregation_variants": [item["name"] for item in self.aggregation_variants],
            "calibration_variants": [item["name"] for item in self.calibration_variants],
            "budget_hours": self.cfg.max_wall_hours,
        }

    def _ensure_dirs(self) -> None:
        for path in [
            self.exp_root,
            self.full_list_root,
            self.mini_list_root,
            self.config_root,
            self.run_root,
            self.result_root,
            self.log_root,
            self.eval_root,
            self.stats_root,
            self.aggregate_root,
            self.state_root,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _import_repo_modules(self) -> None:
        if self._imports_ready:
            return
        for path in [self.nav_root, self.dqa_root, self.et_root]:
            text = str(path.resolve())
            if text not in sys.path:
                sys.path.insert(0, text)
        import setup_fedsto_exact_reproduction as setup
        import run_fedsto_efficientteacher_exact as fedsto
        import dqa_cwa_aggregation as dqa_v1
        import dqa_cwa_aggregation_v2 as dqa_v2

        self.setup = setup
        self.fedsto = fedsto
        self.dqa_v1 = dqa_v1
        self.dqa_v2 = dqa_v2
        self.fedsto.apply_workspace_root(self.exp_root)
        self.setup.LIST_ROOT = self.full_list_root
        self.setup.CONFIG_ROOT = self.config_root
        self.setup.RUN_ROOT = self.run_root
        self.setup.ET_ROOT = self.et_root
        self._imports_ready = True

    def minutes_left(self) -> float:
        return max(0.0, (self.deadline - time.monotonic()) / 60.0)

    def _has_budget(self, required_minutes: float) -> bool:
        return self.minutes_left() >= required_minutes

    @staticmethod
    def read_lines(path: Path) -> list[str]:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def sample_list(self, src: Path, dst: Path, limit: int, *, seed: int) -> Path:
        lines = self.read_lines(src)
        rng = random.Random(seed)
        rng.shuffle(lines)
        keep = lines[: min(limit, len(lines))]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n".join(keep) + "\n", encoding="utf-8")
        cache = dst.with_suffix(".cache")
        if cache.exists():
            cache.unlink()
        return dst

    @staticmethod
    def label_path_for_image(image_path: Path) -> Path:
        parts = list(image_path.parts)
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")

    def count_yolo_classes(self, list_path: Path) -> dict[str, Any]:
        counts = [0] * len(self.setup.BDD_NAMES)
        images = self.read_lines(list_path)
        missing = 0
        empty = 0
        for image in images:
            label = self.label_path_for_image(Path(image))
            if not label.exists():
                missing += 1
                continue
            lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty += 1
                continue
            for line in lines:
                cls = int(float(line.split()[0]))
                if 0 <= cls < len(counts):
                    counts[cls] += 1
        row = {
            "list": str(list_path),
            "images": len(images),
            "missing_labels": missing,
            "empty_labels": empty,
            "objects": sum(counts),
        }
        for name, value in zip(self.setup.BDD_NAMES, counts):
            row[f"gt_{name}"] = value
        return row

    def prepare_data(self) -> pd.DataFrame:
        self._ensure_dirs()
        self._import_repo_modules()
        if not self.warmup_ckpt.exists():
            raise FileNotFoundError(self.warmup_ckpt)
        manifest = self.setup.build_data_lists()
        self.mini_server_train = self.sample_list(
            self.full_list_root / "server_cloudy_train.txt",
            self.mini_list_root / f"server_cloudy_train_{self.cfg.server_train_limit}.txt",
            self.cfg.server_train_limit,
            seed=self.cfg.seed,
        )
        self.mini_server_val = self.sample_list(
            self.full_list_root / "server_cloudy_val.txt",
            self.mini_list_root / f"server_cloudy_val_{self.cfg.server_val_limit}.txt",
            self.cfg.server_val_limit,
            seed=self.cfg.seed + 1,
        )
        self.mini_targets = {}
        for client in self.setup.CLIENTS:
            src = self.full_list_root / f"client_{client['id']}_{client['weather']}_target.txt"
            dst = (
                self.mini_list_root
                / f"client_{client['id']}_{client['weather']}_target_{self.cfg.client_target_limit}.txt"
            )
            self.mini_targets[client["id"]] = self.sample_list(
                src,
                dst,
                self.cfg.client_target_limit,
                seed=self.cfg.seed + 10 + client["id"],
            )

        rows = []
        for role, path, weather in [
            ("server_train", self.full_list_root / "server_cloudy_train.txt", "partly_cloudy"),
            ("server_val", self.full_list_root / "server_cloudy_val.txt", "partly_cloudy"),
            ("mini_server_train", self.mini_server_train, "partly_cloudy"),
            ("mini_server_val", self.mini_server_val, "partly_cloudy"),
        ]:
            row = self.count_yolo_classes(path)
            row.update({"role": role, "weather": weather, "has_gt": True})
            rows.append(row)
        for client in self.setup.CLIENTS:
            path = self.full_list_root / f"client_{client['id']}_{client['weather']}_target.txt"
            rows.append(
                {
                    "role": f"client_{client['id']}_target_full",
                    "weather": client["weather"],
                    "has_gt": False,
                    "list": str(path),
                    "images": len(self.read_lines(path)),
                }
            )
            mini = self.mini_targets[client["id"]]
            rows.append(
                {
                    "role": f"client_{client['id']}_target_mini",
                    "weather": client["weather"],
                    "has_gt": False,
                    "list": str(mini),
                    "images": len(self.read_lines(mini)),
                }
            )
        audit = pd.DataFrame(rows)
        audit_path = self.result_root / "data_audit.csv"
        audit.to_csv(audit_path, index=False)
        (self.result_root / "manifest_snapshot.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return audit

    def _write_yaml(self, path: Path, data: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return path

    def make_config(
        self,
        *,
        name: str,
        weights: Path,
        target: Path | None,
        epochs: int,
        lr0: float,
        train_scope: str,
        weight_decay: float = 5e-4,
        ssod_overrides: dict[str, Any] | None = None,
        batch_size: int | None = None,
    ) -> Path:
        self._import_repo_modules()
        cfg = self.setup.efficientteacher_config(
            name=name,
            train=self.mini_server_train,
            val=self.mini_server_val,
            target=target,
            weights=str(weights.resolve()),
            epochs=int(epochs),
            train_scope=train_scope,
            orthogonal_weight=0.0,
            batch_size=batch_size or self.cfg.batch_size,
            workers=self.cfg.workers,
            device=self.cfg.config_device,
        )
        cfg["project"] = str(self.run_root.resolve())
        cfg["exist_ok"] = True
        cfg["Dataset"]["img_size"] = int(self.cfg.img_size)
        cfg["Dataset"]["workers"] = int(self.cfg.workers)
        cfg["hyp"]["lr0"] = float(lr0)
        cfg["hyp"]["lrf"] = 1.0
        cfg["hyp"]["weight_decay"] = float(weight_decay)
        cfg["hyp"]["warmup_epochs"] = 0
        if target is None:
            cfg["SSOD"] = {"train_domain": False}
        elif ssod_overrides:
            cfg["SSOD"].update(ssod_overrides)
        return self._write_yaml(self.config_root / f"{name}.yaml", cfg)

    def run_command(
        self,
        cmd: list[str],
        *,
        cwd: Path,
        log_path: Path,
        extra_env: dict[str, str] | None = None,
        timeout_minutes: float | None = None,
    ) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env.setdefault("WANDB_DISABLED", "true")
        env.setdefault("MPLBACKEND", "Agg")
        if self.cfg.skip_after_train_best_val:
            env["ET_SKIP_AFTER_TRAIN_BEST_VAL"] = "1"
        if extra_env:
            env.update(extra_env)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=None if timeout_minutes is None else timeout_minutes * 60.0,
        )
        log_path.write_text(proc.stdout, encoding="utf-8")
        proc.check_returncode()
        return proc

    def checkpoint_for_run(self, name: str, *, prefer_best: bool = False) -> Path:
        weight_dir = self.run_root / name / "weights"
        candidates = [weight_dir / "best.pt", weight_dir / "last.pt"] if prefer_best else [weight_dir / "last.pt", weight_dir / "best.pt"]
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    def parse_results_csv(self, run_name: str) -> dict[str, Any]:
        path = self.run_root / run_name / "results.csv"
        row = {"run_name": run_name, "results_csv": str(path), "status": "missing_results"}
        if not path.exists():
            return row
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        map_col = "metrics/mAP_0.5:0.95"
        best_idx = df[map_col].astype(float).idxmax()
        best = df.loc[best_idx]
        final = df.iloc[-1]
        row.update(
            {
                "status": "ok",
                "best_epoch": int(best.get("epoch", best_idx)),
                "map50_best": float(best["metrics/mAP_0.5"]),
                "map50_95_best": float(best[map_col]),
                "precision_best": float(best["metrics/precision"]),
                "recall_best": float(best["metrics/recall"]),
                "map50_final": float(final["metrics/mAP_0.5"]),
                "map50_95_final": float(final[map_col]),
                "precision_final": float(final["metrics/precision"]),
                "recall_final": float(final["metrics/recall"]),
                "checkpoint": str(self.checkpoint_for_run(run_name)),
            }
        )
        return row

    @staticmethod
    def parse_val_stdout(stdout: str) -> dict[str, Any]:
        metric_line = None
        for line in stdout.splitlines():
            if line.strip().startswith("all"):
                metric_line = line.strip()
        if metric_line is None:
            raise RuntimeError("Could not find the 'all' metrics row in val.py output")
        parts = re.split(r"\s+", metric_line)
        return {
            "images": int(float(parts[1])),
            "labels": int(float(parts[2])),
            "precision_final": float(parts[-4]),
            "recall_final": float(parts[-3]),
            "map50_final": float(parts[-2]),
            "map50_95_final": float(parts[-1]),
        }

    def val_config_path(self) -> Path:
        path = self.config_root / "eval_same_mini.yaml"
        if path.exists() and not self.cfg.force_rerun:
            return path
        return self.make_config(
            name="eval_same_mini_cfg",
            weights=self.warmup_ckpt,
            target=None,
            epochs=1,
            lr0=3e-4,
            train_scope="all",
            batch_size=self.cfg.val_batch_size,
        )

    def evaluate_checkpoint(
        self,
        name: str,
        checkpoint: Path,
        *,
        kind: str,
        meta: dict[str, Any] | None = None,
        val_ssod: bool = False,
    ) -> dict[str, Any]:
        out_json = self.eval_root / f"{name}.json"
        if out_json.exists() and not self.cfg.force_rerun:
            return json.loads(out_json.read_text(encoding="utf-8"))
        if not self._has_budget(2.0):
            row = {"name": name, "kind": kind, "status": "skipped_budget", **(meta or {})}
            out_json.write_text(json.dumps(row, indent=2), encoding="utf-8")
            return row
        cmd = [
            self.python_executable,
            "val.py",
            "--weights",
            str(checkpoint.resolve()),
            "--batch-size",
            str(self.cfg.val_batch_size),
            "--imgsz",
            str(self.cfg.img_size),
            "--device",
            self.cfg.device_cli,
            "--project",
            str((self.eval_root / "val_runs").resolve()),
            "--name",
            name,
            "--exist-ok",
            "--no-plots",
            "--cfg",
            str(self.val_config_path().resolve()),
        ]
        if val_ssod:
            cmd.insert(-2, "--val-ssod")
        proc = self.run_command(cmd, cwd=self.et_root, log_path=self.log_root / f"val_{name}.txt")
        row = self.parse_val_stdout(proc.stdout)
        row.update({"name": name, "kind": kind, "status": "ok", "checkpoint": str(checkpoint), **(meta or {})})
        out_json.write_text(json.dumps(row, indent=2), encoding="utf-8")
        return row

    def run_train(self, *, run_name: str, config_path: Path, stats_path: Path | None = None, extra_env: dict[str, str] | None = None) -> dict[str, Any]:
        checkpoint = self.checkpoint_for_run(run_name)
        if checkpoint.exists() and (stats_path is None or stats_path.exists()) and not self.cfg.force_rerun:
            row = self.parse_results_csv(run_name)
            row.update({"name": run_name, "status": "ok_existing"})
            return row
        if self.cfg.force_rerun and (self.run_root / run_name).exists():
            shutil.rmtree(self.run_root / run_name)
        if stats_path is not None and stats_path.exists() and self.cfg.force_rerun:
            stats_path.unlink()
        if not self._has_budget(8.0):
            return {"name": run_name, "status": "skipped_budget", "config": str(config_path)}
        cmd = [self.python_executable, "train.py", "--cfg", str(config_path.resolve())]
        env = extra_env or {}
        self.run_command(cmd, cwd=self.et_root, log_path=self.log_root / f"train_{run_name}.txt", extra_env=env)
        row = self.parse_results_csv(run_name)
        row.update({"name": run_name, "status": "ok", "config": str(config_path)})
        return row

    def run_warmup_and_server_baselines(self) -> pd.DataFrame:
        self.prepare_data()
        rows: list[dict[str, Any]] = []
        rows.append(self.evaluate_checkpoint("warmup_same_mini", self.warmup_ckpt, kind="baseline_eval"))
        for pattern in self.server_patterns:
            cfg_path = self.make_config(
                name=pattern["name"],
                weights=self.warmup_ckpt,
                target=None,
                epochs=pattern["epochs"],
                lr0=pattern["lr0"],
                train_scope=pattern["train_scope"],
                weight_decay=pattern.get("weight_decay", 5e-4),
            )
            row = self.run_train(run_name=pattern["name"], config_path=cfg_path)
            row.update({"kind": "server_supervised", **pattern})
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(self.result_root / "server_baseline_summary.csv", index=False)
        return df

    def run_client_training(self) -> pd.DataFrame:
        self.prepare_data()
        rows: list[dict[str, Any]] = []
        for variant_idx, variant in enumerate(self.client_variants, start=1):
            for client in self.setup.CLIENTS:
                run_name = f"{variant['name']}_client{client['id']}_{client['weather']}"
                stats_path = self.stats_root / variant["name"] / f"client{client['id']}.json"
                cfg_path = self.make_config(
                    name=run_name,
                    weights=self.warmup_ckpt,
                    target=self.mini_targets[client["id"]],
                    epochs=variant["epochs"],
                    lr0=variant["lr0"],
                    train_scope=variant["train_scope"],
                    ssod_overrides=variant.get("ssod_overrides", {}),
                )
                env = {
                    "DQA_PSEUDO_STATS_OUT": str(stats_path.resolve()),
                    "DQA_CLIENT_ID": f"client{client['id']}",
                    "DQA_PHASE": "2",
                    "DQA_ROUND": str(variant_idx),
                }
                row = self.run_train(run_name=run_name, config_path=cfg_path, stats_path=stats_path, extra_env=env)
                row.update(
                    {
                        "kind": "client_ssod",
                        "variant": variant["name"],
                        "client_id": client["id"],
                        "weather": client["weather"],
                        "stats_path": str(stats_path),
                        "note": variant.get("note", ""),
                    }
                )
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(self.result_root / "client_training_summary.csv", index=False)
        self.summarize_pseudo_stats()
        return df

    def summarize_pseudo_stats(self) -> pd.DataFrame:
        rows = []
        for variant in self.client_variants:
            for client in self.setup.CLIENTS:
                path = self.stats_root / variant["name"] / f"client{client['id']}.json"
                row = {"variant": variant["name"], "client_id": client["id"], "weather": client["weather"], "stats_path": str(path)}
                if path.exists():
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    counts = [float(x) for x in payload.get("counts", [])]
                    qualities = [float(x) for x in payload.get("mean_quality_scores", [])]
                    total = sum(counts)
                    row.update(
                        {
                            "status": "ok",
                            "pseudo_total": total,
                            "active_classes": sum(1 for x in counts if x > 0),
                            "zero_classes": sum(1 for x in counts if x <= 0),
                            "top_class_share": (max(counts) / total) if total else 0.0,
                            "mean_quality_active": (
                                sum(q for q, c in zip(qualities, counts) if c > 0) / max(1, sum(1 for c in counts if c > 0))
                            ),
                        }
                    )
                    for name, value in zip(self.setup.BDD_NAMES, counts):
                        row[f"pseudo_{name}"] = value
                    for name, value in zip(self.setup.BDD_NAMES, qualities):
                        row[f"quality_{name}"] = value
                else:
                    row["status"] = "missing"
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(self.result_root / "pseudo_stats_summary.csv", index=False)
        return df

    def evaluate_client_locals(self) -> pd.DataFrame:
        if not self.cfg.run_client_local_eval:
            return pd.DataFrame()
        rows = []
        for variant in self.client_variants:
            for client in self.setup.CLIENTS:
                run_name = f"{variant['name']}_client{client['id']}_{client['weather']}"
                ckpt = self.checkpoint_for_run(run_name)
                if not ckpt.exists():
                    rows.append({"name": run_name, "kind": "client_local_eval", "status": "missing_checkpoint"})
                    continue
                rows.append(
                    self.evaluate_checkpoint(
                        f"client_local_{run_name}",
                        ckpt,
                        kind="client_local_eval",
                        meta={"variant": variant["name"], "client_id": client["id"], "weather": client["weather"]},
                        val_ssod=True,
                    )
                )
        df = pd.DataFrame(rows)
        df.to_csv(self.result_root / "client_local_eval_summary.csv", index=False)
        return df

    def build_round_stats(self, variant_name: str) -> Path | None:
        clients = []
        for client in self.setup.CLIENTS:
            path = self.stats_root / variant_name / f"client{client['id']}.json"
            if not path.exists():
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["id"] = f"client{client['id']}"
            payload["client_id"] = f"client{client['id']}"
            clients.append(payload)
        out = self.stats_root / f"{variant_name}_round_stats.json"
        out.write_text(json.dumps({"clients": clients}, indent=2), encoding="utf-8")
        return out

    def client_checkpoints_for_variant(self, variant_name: str) -> list[Path] | None:
        paths = []
        variant = next(item for item in self.client_variants if item["name"] == variant_name)
        for client in self.setup.CLIENTS:
            run_name = f"{variant['name']}_client{client['id']}_{client['weather']}"
            path = self.checkpoint_for_run(run_name)
            if not path.exists():
                return None
            paths.append(path)
        return paths

    def aggregate_and_evaluate(self) -> pd.DataFrame:
        self.prepare_data()
        rows = []
        for variant in self.client_variants:
            variant_name = variant["name"]
            client_paths = self.client_checkpoints_for_variant(variant_name)
            stats_path = self.build_round_stats(variant_name)
            if client_paths is None or stats_path is None:
                rows.append({"variant": variant_name, "status": "missing_clients_or_stats"})
                continue
            stats = self.dqa_v1.load_round_stats(stats_path, len(self.setup.BDD_NAMES))
            for agg in self.aggregation_variants:
                agg_name = f"{variant_name}__{agg['name']}"
                out_ckpt = self.aggregate_root / f"{agg_name}.pt"
                state_path = self.state_root / f"{agg_name}.json"
                if self.cfg.force_rerun:
                    for path in [out_ckpt, state_path]:
                        if path.exists():
                            path.unlink()
                if not out_ckpt.exists():
                    if agg["kind"] == "fedavg":
                        self.dqa_v1.aggregate_fedavg_checkpoints(
                            client_checkpoints=client_paths,
                            server_checkpoint=self.warmup_ckpt,
                            output_checkpoint=out_ckpt,
                            repo_root=self.root,
                            localize_bn=True,
                        )
                    elif agg["kind"] == "dqa_v2":
                        cfg_kwargs = {"num_classes": len(self.setup.BDD_NAMES)}
                        cfg_kwargs.update(agg.get("config", {}))
                        agg_cfg = self.dqa_v2.AggregationConfig(**cfg_kwargs)
                        self.dqa_v2.aggregate_checkpoints(
                            client_checkpoints=client_paths,
                            server_checkpoint=self.warmup_ckpt,
                            output_checkpoint=out_ckpt,
                            stats=stats,
                            state_path=state_path,
                            config=agg_cfg,
                            repo_root=self.root,
                        )
                    else:
                        raise ValueError(f"unknown aggregation kind: {agg['kind']}")
                eval_row = self.evaluate_checkpoint(
                    f"aggregate_{agg_name}",
                    out_ckpt,
                    kind="aggregate_eval",
                    meta={"variant": variant_name, "aggregation": agg["name"], "aggregation_note": agg.get("note", "")},
                )
                if state_path.exists():
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    eval_row["active_classes"] = sum(1 for x in state.get("last_active_classes", []) if x)
                    eval_row["state_path"] = str(state_path)
                rows.append(eval_row)
        df = pd.DataFrame(rows)
        df.to_csv(self.result_root / "aggregation_summary.csv", index=False)
        return df

    def run_server_calibrations(self) -> pd.DataFrame:
        if not self.cfg.run_server_calibration:
            return pd.DataFrame()
        aggregation_summary = self.result_root / "aggregation_summary.csv"
        if not aggregation_summary.exists():
            self.aggregate_and_evaluate()
        rows = []
        for variant in self.client_variants:
            for agg in self.aggregation_variants:
                agg_name = f"{variant['name']}__{agg['name']}"
                agg_ckpt = self.aggregate_root / f"{agg_name}.pt"
                if not agg_ckpt.exists():
                    continue
                for cal in self.calibration_variants:
                    run_name = f"{agg_name}__{cal['name']}"
                    cfg_path = self.make_config(
                        name=run_name,
                        weights=agg_ckpt,
                        target=None,
                        epochs=cal["epochs"],
                        lr0=cal["lr0"],
                        train_scope=cal["train_scope"],
                        weight_decay=cal.get("weight_decay", 5e-4),
                    )
                    row = self.run_train(run_name=run_name, config_path=cfg_path)
                    row.update(
                        {
                            "kind": "server_calibration",
                            "variant": variant["name"],
                            "aggregation": agg["name"],
                            "calibration": cal["name"],
                            "source_checkpoint": str(agg_ckpt),
                            "note": cal.get("note", ""),
                        }
                    )
                    rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(self.result_root / "server_calibration_summary.csv", index=False)
        return df

    def build_leaderboard(self) -> pd.DataFrame:
        frames = []
        for filename in [
            "server_baseline_summary.csv",
            "client_local_eval_summary.csv",
            "aggregation_summary.csv",
            "server_calibration_summary.csv",
        ]:
            path = self.result_root / filename
            if path.exists():
                frame = pd.read_csv(path)
                frame["source_table"] = filename
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True, sort=False)
        baseline = df.loc[df.get("name", pd.Series(dtype=str)).eq("warmup_same_mini"), "map50_95_final"].dropna()
        if len(baseline):
            base_map = float(baseline.iloc[0])
            df["delta_map50_95_vs_warmup"] = df["map50_95_final"] - base_map
        if "map50_95_final" in df.columns:
            df = df.sort_values("map50_95_final", ascending=False, na_position="last")
        df.to_csv(self.result_root / "overall_leaderboard.csv", index=False)
        return df

    def run_all(self) -> pd.DataFrame:
        self.deadline = time.monotonic() + self.cfg.max_wall_hours * 3600.0
        self.prepare_data()
        self.run_warmup_and_server_baselines()
        self.run_client_training()
        self.evaluate_client_locals()
        self.aggregate_and_evaluate()
        self.run_server_calibrations()
        return self.build_leaderboard()
