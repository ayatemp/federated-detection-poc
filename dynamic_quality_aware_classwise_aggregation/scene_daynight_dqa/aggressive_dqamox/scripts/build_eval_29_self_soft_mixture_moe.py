#!/usr/bin/env python3
"""Evaluate self-only Soft-Mixture output MoE policies from notebook 28 checkpoints.

The experiment intentionally avoids external teachers. It reuses the warmup,
DQA aggregate, DQA repair, and per-client specialists produced by notebook 28,
then tests whether inference-time MoE can absorb noisy pseudoGT specialization
without destructive weight averaging.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

import build_eval_27h_model_level_moe as h27
import build_eval_27t_path_domain_routed_moe as t27
import build_eval_27v_consensus_wbf_moe as v27

from configs.defaults import get_cfg  # noqa: E402
from utils.datasets import LoadImagesAndLabels  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
AGG_ROOT = SCRIPT_PATH.parents[1]
SCENE_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = SCRIPT_PATH.parents[4]
REPORTS_ROOT = AGG_ROOT / "reports"
DEFAULT_INPUT_WORKSPACE = AGG_ROOT / "output" / "28_learned_quality_pseudogt_verifier_r1"
DEFAULT_WORKSPACE = AGG_ROOT / "output" / "29_self_soft_mixture_moe_from_28"
NOTEBOOK_PATH = SCENE_ROOT / "notebooks" / "29_self_soft_mixture_moe_from_28.ipynb"


SPLIT_CLIENTS = {
    "highway_day": "client0_highway_day_28r2",
    "highway_night": "client1_highway_night_28r2",
    "citystreet_day": "client2_citystreet_day_28r2",
    "citystreet_night": "client3_citystreet_night_28r2",
    "residential_day": "client4_residential_day_28r2",
    "residential_night": "client5_residential_night_28r2",
}
NIGHT_SPLITS = {split for split in SPLIT_CLIENTS if split.endswith("_night")}
RARE_CLASSES = [1, 3, 5, 6, 9]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-workspace", type=Path, default=DEFAULT_INPUT_WORKSPACE)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--target-map50", type=float, default=0.55)
    parser.add_argument("--previous-warmup-map50", type=float, default=0.460)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--merge-iou", type=float, default=0.50)
    parser.add_argument("--device", default="")
    parser.add_argument("--gate-images", type=int, default=180)
    parser.add_argument("--min-gate-gain", type=float, default=0.006)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--no-discord", action="store_true")
    return parser.parse_args(argv)


def checkpoint_specs(input_workspace: Path) -> tuple[dict[str, Path], list[str]]:
    ckpt = input_workspace / "checkpoints"
    specs: dict[str, Path] = {
        "warmup_28": ckpt / "round000_latent_dqamox_warmup.pt",
        "agg_28r2": ckpt / "latent_dqamox_p1_round002_dqa_aggregate.pt",
        "repair_28r2": ckpt / "latent_dqamox_p1_round002_server_repair.pt",
        "agg_28r1": ckpt / "latent_dqamox_p1_round001_dqa_aggregate.pt",
        "repair_28r1": ckpt / "latent_dqamox_p1_round001_server_repair.pt",
    }
    for idx, split in enumerate(t27.DOMAIN_SPLITS):
        specs[f"client{idx}_{split}_28r2"] = ckpt / f"latent_dqamox_p1_round002_client{idx}_{split}.pt"
    missing = sorted(label for label, path in specs.items() if not path.exists())
    return specs, missing


def all_routes(labels: list[str]) -> dict[str, list[str]]:
    return {"*": labels}


def split_client_routes(prefix: list[str], suffix: list[str] | None = None) -> dict[str, list[str]]:
    suffix = suffix or []
    return {split: [*prefix, SPLIT_CLIENTS[split], *suffix] for split in t27.DOMAIN_SPLITS}


def night_absorb_routes() -> dict[str, list[str]]:
    routes: dict[str, list[str]] = {}
    for split in t27.DOMAIN_SPLITS:
        if split in NIGHT_SPLITS:
            routes[split] = ["warmup_28", "agg_28r2", "repair_28r2", SPLIT_CLIENTS[split]]
        else:
            routes[split] = ["agg_28r2", "repair_28r2"]
    return routes


def day_guard_night_adapt_routes() -> dict[str, list[str]]:
    routes: dict[str, list[str]] = {}
    for split in t27.DOMAIN_SPLITS:
        if split.endswith("_day"):
            routes[split] = ["warmup_28", "repair_28r2", SPLIT_CLIENTS[split]]
        else:
            routes[split] = ["agg_28r2", "repair_28r2", SPLIT_CLIENTS[split]]
    return routes


def client_score_scales(value: float) -> dict[str, float]:
    return {label: value for label in SPLIT_CLIENTS.values()}


def client_class_filters() -> dict[str, list[int]]:
    return {label: RARE_CLASSES for label in SPLIT_CLIENTS.values()}


def make_candidates(args: argparse.Namespace) -> list[dict]:
    base = {"pre_iou": 0.50, "merge_iou": args.merge_iou}
    return [
        {
            **base,
            "label": "warmup_28_single",
            "routes": all_routes(["warmup_28"]),
            "route_summary": "all images -> notebook 28 warmup model",
            "idea": "Reference: supervised source signal with no pseudoGT adaptation.",
        },
        {
            **base,
            "label": "agg_28r2_single",
            "routes": all_routes(["agg_28r2"]),
            "route_summary": "all images -> notebook 28 DQA aggregate",
            "idea": "Single adapted model; checks whether DQA aggregation alone beats warmup.",
        },
        {
            **base,
            "label": "repair_28r2_single",
            "routes": all_routes(["repair_28r2"]),
            "route_summary": "all images -> notebook 28 server repair",
            "idea": "Single repaired adapted model; checks whether source repair preserves target gains.",
        },
        {
            **base,
            "label": "self_softmix_agg_repair_softnms",
            "routes": all_routes(["agg_28r2", "repair_28r2"]),
            "score_scales": {"agg_28r2": 0.98, "repair_28r2": 1.04},
            "fuser": "soft_nms",
            "soft_nms_sigma": 0.45,
            "soft_nms_score_thr": 0.0001,
            "allow_full": False,
            "route_summary": "all images -> aggregate + repair, Soft-NMS",
            "idea": "FedMoX-style Soft-Mixture at prediction level: mix generalization and perception without averaging weights.",
        },
        {
            **base,
            "label": "domain_client_absorb_s055",
            "routes": split_client_routes(["agg_28r2", "repair_28r2"]),
            "score_scales": {"agg_28r2": 0.98, "repair_28r2": 1.04, **client_score_scales(0.55)},
            "route_summary": "path split -> aggregate + repair + matching client specialist x0.55",
            "idea": "Noisy pseudoGT absorption: client experts are allowed to contribute, but damped behind source-repaired anchors.",
        },
        {
            **base,
            "label": "domain_client_wbf_consensus_s060_b004",
            "routes": split_client_routes(["agg_28r2", "repair_28r2"]),
            "score_scales": {"agg_28r2": 0.98, "repair_28r2": 1.04, **client_score_scales(0.60)},
            "fuser": "wbf",
            "wbf_iou": 0.55,
            "wbf_agreement_bonus": 0.04,
            "wbf_model_weights": {"agg_28r2": 1.00, "repair_28r2": 1.15, **client_score_scales(0.65)},
            "wbf_topk_per_class": 35,
            "allow_full": False,
            "route_summary": "path split -> aggregate + repair + matching client, WBF consensus",
            "idea": "Use agreement, not raw client confidence, as the verifier for pseudo-trained specialists.",
        },
        {
            **base,
            "label": "night_absorb_wbf_s070_b006",
            "routes": night_absorb_routes(),
            "score_scales": {"warmup_28": 0.95, "agg_28r2": 0.98, "repair_28r2": 1.04, **client_score_scales(0.70)},
            "fuser": "wbf",
            "wbf_iou": 0.55,
            "wbf_agreement_bonus": 0.06,
            "wbf_model_weights": {"warmup_28": 0.95, "agg_28r2": 1.00, "repair_28r2": 1.15, **client_score_scales(0.70)},
            "wbf_topk_per_class": 35,
            "allow_full": False,
            "route_summary": "day -> aggregate+repair; night -> warmup+aggregate+repair+night client, WBF",
            "idea": "28 mainly lost residential_night; this gives night specialists more room while preserving day domains.",
        },
        {
            **base,
            "label": "rare_client_channel_wbf_s080_b005",
            "routes": split_client_routes(["agg_28r2", "repair_28r2"]),
            "score_scales": {"agg_28r2": 0.98, "repair_28r2": 1.04, **client_score_scales(0.80)},
            "class_filters": client_class_filters(),
            "fuser": "wbf",
            "wbf_iou": 0.50,
            "wbf_agreement_bonus": 0.05,
            "wbf_model_weights": {"agg_28r2": 1.00, "repair_28r2": 1.15, **client_score_scales(0.70)},
            "wbf_topk_per_class": 40,
            "allow_full": False,
            "route_summary": "path split -> aggregate+repair all classes, client only rare classes, WBF",
            "idea": "Self-only class-channel MoE: let pseudo-trained clients speak only on classes that aggregation underfits.",
        },
        {
            **base,
            "label": "day_guard_night_adapt_s050",
            "routes": day_guard_night_adapt_routes(),
            "score_scales": {"warmup_28": 1.00, "agg_28r2": 0.96, "repair_28r2": 1.05, **client_score_scales(0.50)},
            "route_summary": "day -> warmup+repair+day client; night -> aggregate+repair+night client",
            "idea": "Asymmetric guard: day splits keep the source anchor, night splits get the adapted aggregate.",
        },
    ]


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def notify(message: str, title: str) -> None:
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from notebook_notify import notify_discord

        print(notify_discord(message, title=title, fail_silently=True))
    except Exception as exc:
        print(f"Discord notification skipped: {exc}")


def append_summary(*, workspace: Path, best: dict, status: str, args: argparse.Namespace, full_evaluated: bool) -> None:
    path = REPORTS_ROOT / "29_self_soft_mixture_moe_summary.csv"
    fieldnames = [
        "trial",
        "status",
        "best_candidate",
        "best_phase",
        "best_map50",
        "best_map50_95",
        "warmup_gate_map50",
        "previous_warmup_map50",
        "target_map50",
        "full_evaluated",
        "workspace",
        "notebook",
        "metrics_csv",
        "finished_utc",
        "rationale",
    ]
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "trial": workspace.name,
                "status": status,
                "best_candidate": best.get("candidate", ""),
                "best_phase": best.get("phase", ""),
                "best_map50": best.get("map50", ""),
                "best_map50_95": best.get("map50_95", ""),
                "warmup_gate_map50": best.get("warmup_gate_map50", ""),
                "previous_warmup_map50": args.previous_warmup_map50,
                "target_map50": args.target_map50,
                "full_evaluated": full_evaluated,
                "workspace": str(workspace),
                "notebook": str(NOTEBOOK_PATH),
                "metrics_csv": str(workspace / "stats" / "29_self_soft_mixture_moe_metrics.csv"),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "rationale": (
                    "FedMoX motivates balancing a supervised/perception-biased model and an unsupervised/"
                    "generalization-biased model. This notebook translates that into self-only prediction-level "
                    "MoE over notebook-28 warmup, aggregate, repair, and client experts. WBF/Soft-NMS are used as "
                    "agreement-based pseudoGT verifiers instead of an external teacher."
                ),
            }
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    input_workspace = args.input_workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    stats_dir = workspace / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    setup = h27.load_scene_setup(workspace)
    manifest = setup.build_data_lists()
    split_specs = h27.select_split_specs(manifest["paper_evaluation"], h27.PAPER_SPLITS)
    total_split = split_specs[-1]
    split_cfg = h27.write_eval_config(setup, workspace, total_split, args)
    val_cfg = get_cfg()
    val_cfg.merge_from_file(str(split_cfg))

    specs, missing = checkpoint_specs(input_workspace)
    if missing:
        raise RuntimeError(f"Missing checkpoint inputs from {input_workspace}: {missing}")

    device = select_device(args.device, batch_size=1)
    dataset = LoadImagesAndLabels(
        val_cfg.Dataset.val,
        img_size=args.imgsz,
        batch_size=1,
        rect=False,
        stride=32,
        pad=0.0,
        cfg=val_cfg,
        prefix="29: ",
    )
    gate_indices = t27.even_indices(len(dataset), args.gate_images)
    candidates = make_candidates(args)
    fieldnames = [
        "candidate",
        "phase",
        "split",
        "images",
        "labels",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "num_ap_classes",
        "imgsz",
        "pre_iou",
        "merge_iou",
        "route_summary",
        "idea",
    ]
    metrics_csv = stats_dir / "29_self_soft_mixture_moe_metrics.csv"
    manifest_path = stats_dir / "29_self_soft_mixture_moe_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "workspace": str(workspace),
                "input_workspace": str(input_workspace),
                "dataset_size": len(dataset),
                "gate_images": len(gate_indices),
                "target_map50": args.target_map50,
                "model_specs": {label: str(path) for label, path in specs.items()},
                "candidate_labels": [candidate["label"] for candidate in candidates],
                "full_eval_candidates": [candidate["label"] for candidate in candidates if candidate.get("allow_full", True)],
                "gate_only_candidates": [candidate["label"] for candidate in candidates if not candidate.get("allow_full", True)],
                "papers": [
                    "FedMoX (arXiv:2508.16568): spatial sparse MoE and Soft-Mixture for PSSFL stability.",
                    "Unbiased Teacher (arXiv:2102.09480): pseudo-label bias and class-balance risk in SSOD.",
                    "Unbiased Teacher v2 (arXiv:2206.09500): avoid misleading pseudo boxes for regression.",
                    "Weighted Boxes Fusion (arXiv:1910.13302) and Soft-NMS (arXiv:1704.04503): detection-level consensus fusers.",
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    if not args.no_discord:
        notify(
            "\n".join(
                [
                    "29 started: self-only Soft-Mixture output MoE from notebook 28 checkpoints.",
                    f"- target mAP50={args.target_map50:.3f}",
                    f"- gate images={len(gate_indices)} / dataset={len(dataset)}",
                    f"- input workspace={input_workspace}",
                    "- rule: full evaluation only if gate gain over warmup is large enough.",
                ]
            ),
            "DQA-MoX 29 started",
        )

    rows: list[dict] = []
    gate_total_rows: list[dict] = []
    warmup_gate_map50: float | None = None
    for candidate in candidates:
        labels = t27.required_labels(candidate)
        models, imgsz = v27.load_models(specs=specs, labels=labels, device=device, data=split_cfg, imgsz=args.imgsz)
        try:
            candidate_rows = v27.evaluate_candidate(
                models=models,
                dataset=dataset,
                candidate=candidate,
                indices=gate_indices,
                imgsz=imgsz,
                device=device,
                args=args,
                phase=f"gate_{candidate['label']}",
            )
        finally:
            t27.release_models(models)
        rows.extend(candidate_rows)
        write_rows(metrics_csv, rows, fieldnames)
        total_row = next(row for row in candidate_rows if row["split"] == "scene_daynight_total")
        gate_total_rows.append(total_row)
        if candidate["label"] == "warmup_28_single":
            warmup_gate_map50 = float(total_row["map50"])
            print(f"gate warmup mAP50={warmup_gate_map50:.6f} mAP50:95={total_row['map50_95']}")
        else:
            assert warmup_gate_map50 is not None
            gain = float(total_row["map50"]) - warmup_gate_map50
            print(f"gate {candidate['label']} mAP50={total_row['map50']:.6f} gain_vs_warmup={gain:+.6f}")

    assert warmup_gate_map50 is not None
    non_warmup = [row for row in gate_total_rows if row["candidate"] != "warmup_28_single"]
    candidate_by_label = {candidate["label"]: candidate for candidate in candidates}
    best_gate_overall = max(non_warmup, key=lambda row: (float(row["map50"]), float(row["map50_95"])))
    best_gate_overall["warmup_gate_map50"] = round(warmup_gate_map50, 6)
    full_allowed = [row for row in non_warmup if candidate_by_label[row["candidate"]].get("allow_full", True)]
    best_gate = max(full_allowed, key=lambda row: (float(row["map50"]), float(row["map50_95"])))
    best_gate["warmup_gate_map50"] = round(warmup_gate_map50, 6)
    best_gain = float(best_gate["map50"]) - warmup_gate_map50
    best = best_gate
    full_evaluated = False
    status = "aborted_gate_no_gain"

    if best_gain >= args.min_gate_gain:
        best_candidate = next(candidate for candidate in candidates if candidate["label"] == best_gate["candidate"])
        labels = t27.required_labels(best_candidate)
        models, imgsz = v27.load_models(specs=specs, labels=labels, device=device, data=split_cfg, imgsz=args.imgsz)
        try:
            full_rows = v27.evaluate_candidate(
                models=models,
                dataset=dataset,
                candidate=best_candidate,
                indices=list(range(len(dataset))),
                imgsz=imgsz,
                device=device,
                args=args,
                phase=f"full_{best_candidate['label']}",
            )
        finally:
            t27.release_models(models)
        rows.extend(full_rows)
        write_rows(metrics_csv, rows, fieldnames)
        best = next(row for row in full_rows if row["split"] == "scene_daynight_total")
        best["warmup_gate_map50"] = round(warmup_gate_map50, 6)
        full_evaluated = True
        status = "target_reached" if float(best["map50"]) >= args.target_map50 else "completed_below_target"

    append_summary(workspace=workspace, best=best, status=status, args=args, full_evaluated=full_evaluated)
    target_reached = full_evaluated and float(best["map50"]) >= args.target_map50
    message = "\n".join(
        [
            f"29 finished. Status={status}",
            f"- gate warmup mAP50={warmup_gate_map50:.6f}",
            f"- best gate overall={best_gate_overall['candidate']} mAP50={best_gate_overall['map50']:.6f}",
            f"- best gate={best_gate['candidate']} mAP50={best_gate['map50']:.6f}; gain={best_gain:+.6f}",
            f"- reported best={best['candidate']} phase={best['phase']} mAP50={best['map50']} / mAP50:95={best['map50_95']}",
            f"- full evaluated={full_evaluated}",
            f"- metrics CSV: {metrics_csv}",
            "Decision: target reached." if target_reached else f"Decision: target {args.target_map50:.3f} not reached; next loop should change training/pseudo absorption, not only fusion.",
        ]
    )
    print(message)
    if not args.no_discord:
        notify(message, "DQA-MoX 29 result")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0 if target_reached else 2


if __name__ == "__main__":
    raise SystemExit(main())
