from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DQA_EXPLORING = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation" / "exploring"
if str(DQA_EXPLORING) not in sys.path:
    sys.path.insert(0, str(DQA_EXPLORING))

from dqa_probe_02_2 import (  # noqa: E402
    DQAFunctionalityProbe,
    ProbeConfig,
    default_aggregation_variants,
)


def et_probe_client_variants(*, include_labelmatch: bool = False) -> list[dict[str, Any]]:
    """Short ET pseudo-label/loss candidates for DQA-oriented probing."""
    variants: list[dict[str, Any]] = [
        {
            "name": "et_default_lr1e-3",
            "epochs": 1,
            "lr0": 1e-3,
            "train_scope": "all",
            "ssod_overrides": {},
            "extra_env": {},
            "note": "Current EfficientTeacher/FedSTO pseudo-label defaults with safer LR.",
        },
        {
            "name": "et_reliable_only_strict",
            "epochs": 1,
            "lr0": 3e-4,
            "train_scope": "all",
            "ssod_overrides": {
                "nms_conf_thres": 0.35,
                "ignore_thres_low": 0.35,
                "ignore_thres_high": 0.75,
                "teacher_loss_weight": 0.35,
                "box_loss_weight": 0.02,
                "obj_loss_weight": 0.35,
                "cls_loss_weight": 0.10,
                "pseudo_label_with_bbox": False,
                "pseudo_label_with_cls": False,
            },
            "extra_env": {},
            "note": "Strict reliable labels; uncertain labels affect objectness only.",
        },
        {
            "name": "et_capped_balanced_obj",
            "epochs": 1,
            "lr0": 3e-4,
            "train_scope": "all",
            "ssod_overrides": {
                "nms_conf_thres": 0.25,
                "ignore_thres_low": 0.30,
                "ignore_thres_high": 0.72,
                "teacher_loss_weight": 0.30,
                "box_loss_weight": 0.015,
                "obj_loss_weight": 0.30,
                "cls_loss_weight": 0.05,
                "pseudo_label_with_bbox": False,
                "pseudo_label_with_cls": False,
            },
            "extra_env": {
                "ET_MAX_PSEUDO_PER_IMAGE": "28",
                "ET_MAX_PSEUDO_PER_CLASS_IMAGE": "8",
            },
            "note": "Cap pseudo boxes near GT density; keep uncertain supervision weak.",
        },
        {
            "name": "et_high_precision_capped",
            "epochs": 1,
            "lr0": 3e-4,
            "train_scope": "neck_head",
            "ssod_overrides": {
                "nms_conf_thres": 0.50,
                "ignore_thres_low": 0.50,
                "ignore_thres_high": 0.85,
                "teacher_loss_weight": 0.25,
                "box_loss_weight": 0.015,
                "obj_loss_weight": 0.25,
                "cls_loss_weight": 0.05,
                "pseudo_label_with_bbox": False,
                "pseudo_label_with_cls": False,
            },
            "extra_env": {
                "ET_MAX_PSEUDO_PER_IMAGE": "24",
                "ET_MAX_PSEUDO_PER_CLASS_IMAGE": "6",
            },
            "note": "High-confidence capped pseudo labels; update neck/head only.",
        },
    ]
    if include_labelmatch:
        variants.append(
            {
                "name": "et_labelmatch_lowloss",
                "epochs": 1,
                "lr0": 3e-4,
                "train_scope": "all",
                "ssod_overrides": {
                    "pseudo_label_type": "LabelMatch",
                    "nms_conf_thres": 0.25,
                    "ignore_thres_low": 0.30,
                    "ignore_thres_high": 0.72,
                    "teacher_loss_weight": 0.25,
                    "box_loss_weight": 0.015,
                    "obj_loss_weight": 0.25,
                    "cls_loss_weight": 0.05,
                    "pseudo_label_with_bbox": False,
                    "pseudo_label_with_cls": False,
                },
                "extra_env": {},
                "note": "Optional ET LabelMatch path with low unsupervised loss.",
            }
        )
    return variants


def et_probe_aggregation_variants() -> list[dict[str, Any]]:
    items = default_aggregation_variants()
    return [
        item
        for item in items
        if item["name"] in {"fedavg", "dqa_v2_default", "dqa_v2_conservative"}
    ]


def et_probe_calibration_variants() -> list[dict[str, Any]]:
    return [
        {
            "name": "cal_all_lr1e-3_ep1",
            "epochs": 1,
            "lr0": 1e-3,
            "train_scope": "all",
            "weight_decay": 5e-4,
            "note": "One labeled-server repair epoch for the most promising ET settings.",
        }
    ]


@dataclass
class ETPseudoGTProbeConfig(ProbeConfig):
    experiment_name: str = "efficientteacher_pseudogt_dqa_probe_2h"
    server_train_limit: int = 1024
    server_val_limit: int = 512
    client_target_limit: int = 512
    max_wall_hours: float = 2.0
    batch_size: int = 8
    val_batch_size: int = 4
    run_server_calibration: bool = True
    top_calibration_k: int = 4


class ETPseudoGTDQAProbe(DQAFunctionalityProbe):
    def __init__(
        self,
        cfg: ETPseudoGTProbeConfig | None = None,
        *,
        include_labelmatch: bool = False,
        client_variants: list[dict[str, Any]] | None = None,
    ) -> None:
        cfg = cfg or ETPseudoGTProbeConfig()
        super().__init__(
            cfg,
            client_variants=client_variants or et_probe_client_variants(include_labelmatch=include_labelmatch),
            aggregation_variants=et_probe_aggregation_variants(),
            server_patterns=[],
            calibration_variants=et_probe_calibration_variants(),
        )
        # The base probe treats an empty list as "use defaults"; this ET probe
        # only wants the reused warm-up baseline, not an extra supervised oracle.
        self.server_patterns = []
        self.et_project_root = self.root / "efficient_teacher"
        self.exp_root = self.et_project_root / self.cfg.experiment_name
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

    def describe(self) -> dict[str, Any]:
        desc = super().describe()
        desc.update(
            {
                "purpose": "ET pseudo-label and unsupervised-loss probe for DQA input quality",
                "client_target_limit": self.cfg.client_target_limit,
                "server_val_limit": self.cfg.server_val_limit,
                "top_calibration_k": self.cfg.top_calibration_k,
            }
        )
        return desc

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
                env.update({str(k): str(v) for k, v in variant.get("extra_env", {}).items()})
                row = self.run_train(run_name=run_name, config_path=cfg_path, stats_path=stats_path, extra_env=env)
                row.update(
                    {
                        "kind": "client_ssod",
                        "variant": variant["name"],
                        "client_id": client["id"],
                        "weather": client["weather"],
                        "stats_path": str(stats_path),
                        "extra_env": json.dumps(variant.get("extra_env", {}), sort_keys=True),
                        "note": variant.get("note", ""),
                    }
                )
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(self.result_root / "client_training_summary.csv", index=False)
        self.summarize_pseudo_stats()
        self.summarize_pseudo_density()
        return df

    def summarize_pseudo_density(self) -> pd.DataFrame:
        if not hasattr(self, "setup"):
            self.prepare_data()
        pseudo_path = self.result_root / "pseudo_stats_summary.csv"
        if not pseudo_path.exists():
            self.summarize_pseudo_stats()
        df = pd.read_csv(pseudo_path)
        if "pseudo_total" not in df:
            df["pseudo_total"] = pd.NA
        if "top_class_share" not in df:
            df["top_class_share"] = pd.NA
        if "mean_quality_active" not in df:
            df["mean_quality_active"] = pd.NA
        target_images = float(self.cfg.client_target_limit)
        df["pseudo_per_image"] = pd.to_numeric(df["pseudo_total"], errors="coerce") / max(target_images, 1.0)

        audit_path = self.result_root / "data_audit.csv"
        gt_per_image = None
        if audit_path.exists():
            audit = pd.read_csv(audit_path)
            gt_rows = audit.loc[audit["role"].eq("mini_server_val")]
            if len(gt_rows) and "objects" in gt_rows:
                row = gt_rows.iloc[0]
                gt_per_image = float(row["objects"]) / max(float(row["images"]), 1.0)
        df["reference_gt_per_image"] = gt_per_image
        df["pseudo_to_gt_density"] = (
            df["pseudo_per_image"] / float(gt_per_image)
            if gt_per_image and gt_per_image > 0
            else pd.NA
        )
        out = self.result_root / "pseudo_density_summary.csv"
        df.to_csv(out, index=False)
        return df

    def run_top_server_calibrations(self) -> pd.DataFrame:
        if not self.cfg.run_server_calibration:
            return pd.DataFrame()
        aggregation_summary = self.result_root / "aggregation_summary.csv"
        if not aggregation_summary.exists():
            self.aggregate_and_evaluate()

        agg_df = pd.read_csv(aggregation_summary)
        metric = "map50_95_final"
        status = agg_df["status"] if "status" in agg_df else pd.Series(["ok"] * len(agg_df))
        eligible = agg_df.loc[status.eq("ok")].copy()
        if metric not in eligible:
            return pd.DataFrame()
        eligible = eligible.sort_values([metric, "map50_final"], ascending=False)
        top = eligible.head(int(self.cfg.top_calibration_k))

        rows = []
        for _, source in top.iterrows():
            variant = str(source["variant"])
            aggregation = str(source["aggregation"])
            agg_name = f"{variant}__{aggregation}"
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
                        "variant": variant,
                        "aggregation": aggregation,
                        "calibration": cal["name"],
                        "source_checkpoint": str(agg_ckpt),
                        "source_map50_95_final": float(source[metric]),
                        "note": cal.get("note", ""),
                    }
                )
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(self.result_root / "server_calibration_summary.csv", index=False)
        return df

    def build_recommendation_table(self) -> pd.DataFrame:
        self.summarize_pseudo_density()
        density = pd.read_csv(self.result_root / "pseudo_density_summary.csv")
        for column in ["pseudo_per_image", "pseudo_to_gt_density", "top_class_share", "mean_quality_active"]:
            if column not in density:
                density[column] = pd.NA
        density_grouped = density.groupby("variant", as_index=False).agg(
            pseudo_per_image_mean=("pseudo_per_image", "mean"),
            pseudo_per_image_max=("pseudo_per_image", "max"),
            pseudo_to_gt_density_mean=("pseudo_to_gt_density", "mean"),
            top_class_share_mean=("top_class_share", "mean"),
            mean_quality_active_mean=("mean_quality_active", "mean"),
        )

        frames = [density_grouped]
        client_eval_path = self.result_root / "client_local_eval_summary.csv"
        if client_eval_path.exists():
            client_eval = pd.read_csv(client_eval_path)
            status = client_eval["status"] if "status" in client_eval else pd.Series(["ok"] * len(client_eval))
            client_eval_ok = client_eval.loc[status.eq("ok")].copy()
            if len(client_eval_ok) and {"variant", "map50_95_final"}.issubset(client_eval_ok.columns):
                frames.append(
                    client_eval_ok.groupby("variant", as_index=False).agg(
                        client_map50_95_mean=("map50_95_final", "mean"),
                        client_map50_95_min=("map50_95_final", "min"),
                    )
                )

        agg_path = self.result_root / "aggregation_summary.csv"
        if agg_path.exists():
            agg = pd.read_csv(agg_path)
            status = agg["status"] if "status" in agg else pd.Series(["ok"] * len(agg))
            agg_ok = agg.loc[status.eq("ok")].copy()
            if len(agg_ok) and {"variant", "map50_95_final", "map50_final"}.issubset(agg_ok.columns):
                frames.append(
                    agg_ok.groupby("variant", as_index=False).agg(
                        aggregate_map50_95_max=("map50_95_final", "max"),
                        aggregate_map50_max=("map50_final", "max"),
                    )
                )
            if len(agg_ok) and {"aggregation", "variant", "map50_95_final", "map50_final"}.issubset(agg_ok.columns):
                fedavg = agg_ok.loc[agg_ok["aggregation"].eq("fedavg")]
                dqa = agg_ok.loc[agg_ok["aggregation"].astype(str).str.startswith("dqa")]
                if len(fedavg):
                    frames.append(
                        fedavg.groupby("variant", as_index=False).agg(
                            fedavg_map50_95=("map50_95_final", "max"),
                            fedavg_map50=("map50_final", "max"),
                        )
                    )
                if len(dqa):
                    frames.append(
                        dqa.groupby("variant", as_index=False).agg(
                            dqa_map50_95_best=("map50_95_final", "max"),
                            dqa_map50_best=("map50_final", "max"),
                        )
                    )

        cal_path = self.result_root / "server_calibration_summary.csv"
        if cal_path.exists():
            cal = pd.read_csv(cal_path)
            status = cal["status"] if "status" in cal else pd.Series(["ok"] * len(cal))
            cal_ok = cal.loc[status.eq("ok")].copy()
            if len(cal_ok) and {"variant", "map50_95_final", "map50_final"}.issubset(cal_ok.columns):
                frames.append(
                    cal_ok.groupby("variant", as_index=False).agg(
                        calibration_map50_95_max=("map50_95_final", "max"),
                        calibration_map50_max=("map50_final", "max"),
                    )
                )

        table = frames[0]
        for frame in frames[1:]:
            table = table.merge(frame, on="variant", how="left")
        if {"dqa_map50_95_best", "fedavg_map50_95"}.issubset(table.columns):
            table["delta_dqa_vs_fedavg_map50_95"] = table["dqa_map50_95_best"] - table["fedavg_map50_95"]
        if "aggregate_map50_95_max" in table:
            table = table.sort_values("aggregate_map50_95_max", ascending=False, na_position="last")
        table.to_csv(self.result_root / "recommendation_table.csv", index=False)
        return table

    def run_all(self) -> dict[str, pd.DataFrame]:
        outputs = {
            "server_baseline": self.run_warmup_and_server_baselines(),
            "client_training": self.run_client_training(),
            "pseudo_density": self.summarize_pseudo_density(),
            "client_local_eval": self.evaluate_client_locals(),
            "aggregation": self.aggregate_and_evaluate(),
            "server_calibration": self.run_top_server_calibrations(),
        }
        outputs["leaderboard"] = self.build_leaderboard()
        outputs["recommendation"] = self.build_recommendation_table()
        return outputs


def reset_probe_outputs(probe: ETPseudoGTDQAProbe) -> None:
    if probe.exp_root.exists():
        shutil.rmtree(probe.exp_root)
