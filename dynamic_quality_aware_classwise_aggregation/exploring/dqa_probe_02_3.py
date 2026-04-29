from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from dqa_probe_02_2 import DQAFunctionalityProbe, ProbeConfig


@dataclass
class ControlSweepConfig:
    source_experiment_name: str = "02_2_dqa_functionality_probe_7h"
    experiment_name: str = "02_3_dqa_control_sweep_6h"
    source_warmup_key: str = "dqa_v2_scene_12h"
    max_wall_hours: float = 6.0
    server_train_limit: int = 2048
    server_val_limit: int = 1024
    client_target_limit: int = 2048
    device_cli: str = "0"
    batch_size: int = 8
    val_batch_size: int = 1
    workers: int = 0
    force_rerun: bool = False
    run_top_calibration: bool = True
    top_calibration_k: int = 8


def default_source_variants() -> list[str]:
    return [
        "ssod_default_all_lr1e-2",
        "ssod_default_all_lr1e-3",
        "ssod_strict_all_lr3e-4",
        "ssod_strict_neck_head_lr1e-3",
        "ssod_very_strict_all_lr3e-4",
    ]


def default_subsets() -> list[dict[str, Any]]:
    return [
        {"name": "all_clients", "clients": [0, 1, 2], "note": "All weather clients."},
        {"name": "overcast_only", "clients": [0], "note": "Client 0 only."},
        {"name": "rainy_only", "clients": [1], "note": "Client 1 only."},
        {"name": "snowy_only", "clients": [2], "note": "Client 2 only."},
        {"name": "drop_overcast", "clients": [1, 2], "note": "Leave client 0 out."},
        {"name": "drop_rainy", "clients": [0, 2], "note": "Leave client 1 out."},
        {"name": "drop_snowy", "clients": [0, 1], "note": "Leave client 2 out."},
    ]


def default_control_candidates() -> list[dict[str, Any]]:
    return [
        {
            "name": "fedavg",
            "kind": "fedavg",
            "config": {},
            "note": "Plain FedAvg reference for each client subset.",
        },
        {
            "name": "dqa_current",
            "kind": "dqa_v2",
            "config": {"min_server_alpha": 0.45},
            "note": "Current v2-style DQA anchor.",
        },
        {
            "name": "dqa_floor60_resid20",
            "kind": "dqa_v2",
            "config": {
                "min_server_alpha": 0.60,
                "residual_blend": 0.20,
                "classwise_blend": 0.25,
                "temperature": 2.0,
                "uniform_mix": 0.10,
            },
            "note": "Moderate server floor and smaller residuals.",
        },
        {
            "name": "dqa_floor70_resid12",
            "kind": "dqa_v2",
            "config": {
                "min_server_alpha": 0.70,
                "residual_blend": 0.12,
                "classwise_blend": 0.18,
                "temperature": 2.0,
                "uniform_mix": 0.10,
            },
            "note": "Conservative DQA, close to a guarded client residual.",
        },
        {
            "name": "dqa_floor80_resid05",
            "kind": "dqa_v2",
            "config": {
                "min_server_alpha": 0.80,
                "residual_blend": 0.05,
                "classwise_blend": 0.10,
                "temperature": 2.5,
                "uniform_mix": 0.10,
            },
            "note": "Tiny residual DQA. Tests whether almost all client movement should be suppressed.",
        },
        {
            "name": "dqa_floor90_resid02",
            "kind": "dqa_v2",
            "config": {
                "min_server_alpha": 0.90,
                "residual_blend": 0.02,
                "classwise_blend": 0.05,
                "temperature": 4.0,
                "uniform_mix": 0.05,
            },
            "note": "Near-rollback DQA. Useful as a lower-risk acceptance baseline.",
        },
        {
            "name": "dqa_count_gate100",
            "kind": "dqa_v2",
            "config": {
                "min_server_alpha": 0.70,
                "residual_blend": 0.12,
                "classwise_blend": 0.18,
                "temperature": 2.0,
                "uniform_mix": 0.10,
                "min_effective_count": 100.0,
            },
            "note": "Ignore classes with too little pseudo-label support.",
        },
        {
            "name": "dqa_quality_gate30",
            "kind": "dqa_v2",
            "config": {
                "min_server_alpha": 0.70,
                "residual_blend": 0.12,
                "classwise_blend": 0.18,
                "temperature": 2.0,
                "uniform_mix": 0.10,
                "min_quality": 0.30,
            },
            "note": "Ignore low-quality pseudo-label classes.",
        },
    ]


class DQAControlSweep:
    def __init__(
        self,
        cfg: ControlSweepConfig | None = None,
        *,
        source_variants: list[str] | None = None,
        subsets: list[dict[str, Any]] | None = None,
        candidates: list[dict[str, Any]] | None = None,
    ) -> None:
        self.cfg = cfg or ControlSweepConfig()
        self.source_variants = source_variants or default_source_variants()
        self.subsets = subsets or default_subsets()
        self.candidates = candidates or default_control_candidates()
        self.probe = DQAFunctionalityProbe(
            ProbeConfig(
                source_warmup_key=self.cfg.source_warmup_key,
                experiment_name=self.cfg.experiment_name,
                server_train_limit=self.cfg.server_train_limit,
                server_val_limit=self.cfg.server_val_limit,
                client_target_limit=self.cfg.client_target_limit,
                max_wall_hours=self.cfg.max_wall_hours,
                device_cli=self.cfg.device_cli,
                batch_size=self.cfg.batch_size,
                val_batch_size=self.cfg.val_batch_size,
                workers=self.cfg.workers,
                force_rerun=self.cfg.force_rerun,
                run_client_local_eval=False,
                run_server_calibration=False,
            )
        )
        self.source_probe = DQAFunctionalityProbe(
            ProbeConfig(
                source_warmup_key=self.cfg.source_warmup_key,
                experiment_name=self.cfg.source_experiment_name,
                server_train_limit=self.cfg.server_train_limit,
                server_val_limit=self.cfg.server_val_limit,
                client_target_limit=self.cfg.client_target_limit,
                max_wall_hours=self.cfg.max_wall_hours,
                device_cli=self.cfg.device_cli,
                batch_size=self.cfg.batch_size,
                val_batch_size=self.cfg.val_batch_size,
                workers=self.cfg.workers,
                force_rerun=False,
            )
        )
        self.deadline = time.monotonic() + self.cfg.max_wall_hours * 3600.0
        self.root = self.probe.exp_root
        self.aggregate_root = self.root / "aggregates"
        self.state_root = self.root / "dqa_states"
        self.result_root = self.root / "results"
        self.stats_root = self.root / "stats"
        self.log_root = self.result_root / "logs"
        for path in [self.aggregate_root, self.state_root, self.result_root, self.stats_root, self.log_root]:
            path.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict[str, Any]:
        return {
            "experiment_root": str(self.root),
            "source_experiment": str(self.source_probe.exp_root),
            "source_variants": self.source_variants,
            "subsets": [item["name"] for item in self.subsets],
            "candidates": [item["name"] for item in self.candidates],
            "max_wall_hours": self.cfg.max_wall_hours,
        }

    def minutes_left(self) -> float:
        return max(0.0, (self.deadline - time.monotonic()) / 60.0)

    @staticmethod
    def slug(text: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")

    def prepare(self) -> None:
        self.probe.prepare_data()
        self.source_probe.prepare_data()

    def source_status(self) -> pd.DataFrame:
        self.prepare()
        rows: list[dict[str, Any]] = []
        for variant in self.source_variants:
            for client in self.source_probe.setup.CLIENTS:
                run_name = f"{variant}_client{client['id']}_{client['weather']}"
                checkpoint = self.source_probe.checkpoint_for_run(run_name)
                stats_path = self.source_probe.stats_root / variant / f"client{client['id']}.json"
                rows.append(
                    {
                        "variant": variant,
                        "client_id": client["id"],
                        "weather": client["weather"],
                        "checkpoint_exists": checkpoint.exists(),
                        "checkpoint": str(checkpoint),
                        "stats_exists": stats_path.exists(),
                        "stats_path": str(stats_path),
                    }
                )
        df = pd.DataFrame(rows)
        df.to_csv(self.result_root / "source_status.csv", index=False)
        return df

    def pseudo_summary(self) -> pd.DataFrame:
        path = self.source_probe.result_root / "pseudo_stats_summary.csv"
        if path.exists():
            df = pd.read_csv(path)
        else:
            df = self.source_probe.summarize_pseudo_stats()
        df.to_csv(self.result_root / "source_pseudo_stats_summary.csv", index=False)
        return df

    def client_paths(self, variant: str, client_ids: list[int]) -> list[Path] | None:
        out = []
        by_id = {client["id"]: client for client in self.source_probe.setup.CLIENTS}
        for client_id in client_ids:
            client = by_id[client_id]
            run_name = f"{variant}_client{client_id}_{client['weather']}"
            checkpoint = self.source_probe.checkpoint_for_run(run_name)
            if not checkpoint.exists():
                return None
            out.append(checkpoint)
        return out

    def round_stats(self, variant: str, subset: dict[str, Any]) -> Path | None:
        clients = []
        by_id = {client["id"]: client for client in self.source_probe.setup.CLIENTS}
        for client_id in subset["clients"]:
            client = by_id[client_id]
            stats_path = self.source_probe.stats_root / variant / f"client{client_id}.json"
            if not stats_path.exists():
                return None
            payload = json.loads(stats_path.read_text(encoding="utf-8"))
            payload["id"] = f"client{client_id}"
            payload["client_id"] = f"client{client_id}"
            payload["weather"] = client["weather"]
            clients.append(payload)
        out = self.stats_root / f"{variant}__{subset['name']}_round_stats.json"
        out.write_text(json.dumps({"clients": clients}, indent=2), encoding="utf-8")
        return out

    def aggregate_one(self, variant: str, subset: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
        self.prepare()
        name = self.slug(f"{variant}__{subset['name']}__{candidate['name']}")
        out_ckpt = self.aggregate_root / f"{name}.pt"
        state_path = self.state_root / f"{name}.json"
        client_paths = self.client_paths(variant, subset["clients"])
        stats_path = self.round_stats(variant, subset)
        base_row = {
            "name": name,
            "variant": variant,
            "subset": subset["name"],
            "subset_clients": ",".join(str(x) for x in subset["clients"]),
            "candidate": candidate["name"],
            "candidate_kind": candidate["kind"],
            "candidate_note": candidate.get("note", ""),
            "subset_note": subset.get("note", ""),
        }
        if client_paths is None or stats_path is None:
            return {**base_row, "status": "missing_source"}
        if self.cfg.force_rerun:
            for path in [out_ckpt, state_path, self.probe.eval_root / f"control_{name}.json"]:
                if path.exists():
                    path.unlink()
        if not out_ckpt.exists():
            if candidate["kind"] == "fedavg":
                self.probe.dqa_v1.aggregate_fedavg_checkpoints(
                    client_checkpoints=client_paths,
                    server_checkpoint=self.probe.warmup_ckpt,
                    output_checkpoint=out_ckpt,
                    repo_root=self.probe.root,
                    localize_bn=True,
                )
            elif candidate["kind"] == "dqa_v2":
                stats = self.probe.dqa_v1.load_round_stats(stats_path, len(self.probe.setup.BDD_NAMES))
                kwargs = {"num_classes": len(self.probe.setup.BDD_NAMES)}
                kwargs.update(candidate.get("config", {}))
                config = self.probe.dqa_v2.AggregationConfig(**kwargs)
                self.probe.dqa_v2.aggregate_checkpoints(
                    client_checkpoints=client_paths,
                    server_checkpoint=self.probe.warmup_ckpt,
                    output_checkpoint=out_ckpt,
                    stats=stats,
                    state_path=state_path,
                    config=config,
                    repo_root=self.probe.root,
                )
            else:
                raise ValueError(f"Unknown candidate kind: {candidate['kind']}")
        row = self.probe.evaluate_checkpoint(f"control_{name}", out_ckpt, kind="control_aggregate_eval", meta=base_row)
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            row["active_classes"] = sum(1 for value in state.get("last_active_classes", []) if value)
            row["state_path"] = str(state_path)
        return row

    def run_sweep(self) -> pd.DataFrame:
        self.deadline = time.monotonic() + self.cfg.max_wall_hours * 3600.0
        self.prepare()
        rows = []
        for variant in self.source_variants:
            for subset in self.subsets:
                for candidate in self.candidates:
                    if self.minutes_left() < 2.0:
                        rows.append(
                            {
                                "variant": variant,
                                "subset": subset["name"],
                                "candidate": candidate["name"],
                                "status": "skipped_budget",
                            }
                        )
                        continue
                    rows.append(self.aggregate_one(variant, subset, candidate))
                    pd.DataFrame(rows).to_csv(self.result_root / "control_sweep_partial.csv", index=False)
        df = pd.DataFrame(rows)
        df.to_csv(self.result_root / "control_sweep_summary.csv", index=False)
        return df

    def build_decision_table(self) -> pd.DataFrame:
        sweep_path = self.result_root / "control_sweep_summary.csv"
        if not sweep_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(sweep_path)
        warmup_path = self.source_probe.result_root / "evals" / "warmup_same_mini.json"
        if warmup_path.exists():
            warmup = json.loads(warmup_path.read_text(encoding="utf-8"))
            df["delta_map50_95_vs_warmup"] = df["map50_95_final"] - float(warmup["map50_95_final"])
            df["delta_map50_vs_warmup"] = df["map50_final"] - float(warmup["map50_final"])
        keys = ["variant", "subset"]
        fedavg = (
            df[df["candidate"].eq("fedavg")][keys + ["map50_95_final"]]
            .rename(columns={"map50_95_final": "fedavg_map50_95"})
        )
        out = df.merge(fedavg, on=keys, how="left")
        out["delta_vs_fedavg"] = out["map50_95_final"] - out["fedavg_map50_95"]
        out["accept_over_warmup"] = out.get("delta_map50_95_vs_warmup", pd.Series(0, index=out.index)) > 0.0
        out["accept_over_fedavg"] = out["delta_vs_fedavg"] > 0.0
        out = out.sort_values("map50_95_final", ascending=False, na_position="last")
        out.to_csv(self.result_root / "decision_table.csv", index=False)
        return out

    def run_top_calibrations(self) -> pd.DataFrame:
        if not self.cfg.run_top_calibration:
            return pd.DataFrame()
        table = self.build_decision_table()
        if table.empty:
            return pd.DataFrame()
        ok = table[table["status"].eq("ok")].copy()
        top = ok.head(self.cfg.top_calibration_k)
        rows = []
        for _, item in top.iterrows():
            checkpoint = Path(item["checkpoint"])
            run_name = self.slug(f"cal_{item['name']}__lr3e-4_ep1")
            config_path = self.probe.make_config(
                name=run_name,
                weights=checkpoint,
                target=None,
                epochs=1,
                lr0=3e-4,
                train_scope="all",
                weight_decay=5e-4,
            )
            row = self.probe.run_train(run_name=run_name, config_path=config_path)
            row.update(
                {
                    "kind": "control_top_calibration",
                    "source_control_name": item["name"],
                    "variant": item["variant"],
                    "subset": item["subset"],
                    "candidate": item["candidate"],
                    "source_checkpoint": str(checkpoint),
                }
            )
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(self.result_root / "top_calibration_summary.csv", index=False)
        return df

    def leaderboard(self) -> pd.DataFrame:
        frames = []
        for filename in ["decision_table.csv", "top_calibration_summary.csv"]:
            path = self.result_root / filename
            if path.exists():
                frame = pd.read_csv(path)
                frame["source_table"] = filename
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True, sort=False)
        if "map50_95_final" in out.columns:
            out = out.sort_values("map50_95_final", ascending=False, na_position="last")
        out.to_csv(self.result_root / "overall_leaderboard.csv", index=False)
        return out

    def run_all(self) -> pd.DataFrame:
        self.source_status()
        self.pseudo_summary()
        self.run_sweep()
        self.build_decision_table()
        self.run_top_calibrations()
        return self.leaderboard()
