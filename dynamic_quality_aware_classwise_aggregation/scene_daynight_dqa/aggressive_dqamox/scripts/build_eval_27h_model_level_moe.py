#!/usr/bin/env python3
"""Evaluate 27h prediction-level/model-level MoE ensembles for scene-daynight DQA."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


SCRIPT_PATH = Path(__file__).resolve()
AGG_ROOT = SCRIPT_PATH.parents[1]
SCENE_ROOT = SCRIPT_PATH.parents[2]
RESEARCH_ROOT = SCRIPT_PATH.parents[3]
REPO_ROOT = SCRIPT_PATH.parents[4]
NAV_ROOT = REPO_ROOT / "navigating_data_heterogeneity"
ET_ROOT = NAV_ROOT / "vendor" / "efficientteacher"
REPORTS_ROOT = AGG_ROOT / "reports"
DEFAULT_WORKSPACE = AGG_ROOT / "output" / "27h_model_level_moe_tta"
PREFERRED_VAL_PYTHONS = [
    Path("/root/micromamba/envs/al_yolov8/bin/python"),
    Path(sys.executable),
    Path("/opt/venv/bin/python"),
]
PAPER_SPLITS = [
    "highway_day",
    "highway_night",
    "citystreet_day",
    "citystreet_night",
    "residential_day",
    "residential_night",
    "total",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.6)
    parser.add_argument("--device", default="")
    parser.add_argument("--target-map50", type=float, default=0.60)
    parser.add_argument("--full-split-min-gain", type=float, default=0.003)
    parser.add_argument("--python-executable", type=Path, default=None)
    parser.add_argument("--no-discord", action="store_true")
    return parser.parse_args(argv)


def load_scene_setup(workspace: Path):
    scripts_root = SCENE_ROOT / "scripts"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    import setup_scene_daynight as setup  # type: ignore

    setup.WORK_ROOT = workspace
    setup.LIST_ROOT = workspace / "data_lists"
    setup.CONFIG_ROOT = workspace / "configs"
    setup.RUN_ROOT = workspace / "runs"
    return setup


def select_val_python(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit is not None else []
    for candidate in PREFERRED_VAL_PYTHONS:
        if candidate not in candidates:
            candidates.append(candidate)
    check_cmd = "import cv2, seaborn, torch, yaml"
    for candidate in candidates:
        if candidate is None or not candidate.exists():
            continue
        result = subprocess.run([str(candidate), "-c", check_cmd], capture_output=True, text=True)
        if result.returncode == 0:
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates if candidate is not None)
    raise RuntimeError(f"Could not find a validation Python with dependencies. Tried: {tried}")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "run"


def select_split_specs(paper_eval: dict, requested: list[str]) -> list[dict]:
    by_name = {split["name"]: split for split in paper_eval["splits"]}
    by_name["total"] = paper_eval["total"]
    return [by_name[name] for name in requested]


def write_eval_config(setup, workspace: Path, split: dict, args: argparse.Namespace) -> Path:
    config_root = workspace / "validation_reports" / "paper_protocol_configs"
    config_root.mkdir(parents=True, exist_ok=True)
    cfg = setup.efficientteacher_config(
        name=f"paper_eval_{split['name']}",
        train=setup.LIST_ROOT / "server_cloudy_train.txt",
        val=Path(split["list"]),
        target=None,
        weights="",
        epochs=1,
        train_scope="all",
        batch_size=args.batch_size,
        workers=0,
        device="",
    )
    cfg["Dataset"]["batch_size"] = args.batch_size
    cfg["Dataset"]["workers"] = 0
    cfg["SSOD"] = {"train_domain": False}
    out = config_root / f"{split['name']}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


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


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_pool() -> dict[str, Path]:
    base = SCENE_ROOT
    research = AGG_ROOT / "output" / "27_research_notebook_until_060"
    pool = {
        "warmup": base / "output" / "08_full_latent_dqamox_from_warmup" / "checkpoints" / "round000_latent_dqamox_warmup.pt",
        "27d_repair": research / "27d_probe_teacher_residual_mixpl_r2" / "checkpoints" / "latent_dqamox_p1_round002_server_repair.pt",
        "27e_repair": research / "27e_probe_clean_day_expert_anchor_r2" / "checkpoints" / "latent_dqamox_p1_round002_server_repair.pt",
        "27g_repair": research / "27g_probe_moe_head_only_router_r1" / "checkpoints" / "latent_dqamox_p1_round001_server_repair.pt",
        "27g_aggregate": research / "27g_probe_moe_head_only_router_r1" / "checkpoints" / "latent_dqamox_p1_round001_dqa_aggregate.pt",
    }
    client_root = research / "27g_probe_moe_head_only_router_r1" / "checkpoints"
    for idx, client in enumerate(
        [
            "highway_day",
            "highway_night",
            "citystreet_day",
            "citystreet_night",
            "residential_day",
            "residential_night",
        ]
    ):
        pool[f"27g_client{idx}_{client}"] = client_root / f"latent_dqamox_p1_round001_client{idx}_{client}.pt"
    return pool


def make_candidates(pool: dict[str, Path]) -> list[dict]:
    raw = [
        {
            "label": "warmup_single",
            "weights": ["warmup"],
            "augment": False,
            "idea": "single warmup baseline under this runner",
        },
        {
            "label": "warmup_tta",
            "weights": ["warmup"],
            "augment": True,
            "idea": "test-time augmentation baseline",
        },
        {
            "label": "repair_pool_warmup_27d_27e_27g",
            "weights": ["warmup", "27d_repair", "27e_repair", "27g_repair"],
            "augment": False,
            "idea": "model-level MoE over stable repair experts",
        },
        {
            "label": "repair_pool_warmup_27d_27e_27g_tta",
            "weights": ["warmup", "27d_repair", "27e_repair", "27g_repair"],
            "augment": True,
            "idea": "model-level MoE over repair experts plus TTA",
        },
        {
            "label": "client_pool_27g_all",
            "weights": [
                "27g_client0_highway_day",
                "27g_client1_highway_night",
                "27g_client2_citystreet_day",
                "27g_client3_citystreet_night",
                "27g_client4_residential_day",
                "27g_client5_residential_night",
            ],
            "augment": False,
            "idea": "pure client-specialist prediction MoE",
        },
        {
            "label": "client_pool_27g_all_plus_warmup",
            "weights": [
                "warmup",
                "27g_client0_highway_day",
                "27g_client1_highway_night",
                "27g_client2_citystreet_day",
                "27g_client3_citystreet_night",
                "27g_client4_residential_day",
                "27g_client5_residential_night",
            ],
            "augment": False,
            "idea": "client-specialist prediction MoE anchored by warmup",
        },
        {
            "label": "broad_pool_repairs_clients_warmup",
            "weights": [
                "warmup",
                "27d_repair",
                "27e_repair",
                "27g_repair",
                "27g_client0_highway_day",
                "27g_client1_highway_night",
                "27g_client2_citystreet_day",
                "27g_client3_citystreet_night",
                "27g_client4_residential_day",
                "27g_client5_residential_night",
            ],
            "augment": False,
            "idea": "wide prediction MoE over global and client experts",
        },
    ]
    candidates = []
    for spec in raw:
        missing = [name for name in spec["weights"] if not pool[name].exists()]
        if missing:
            spec = {**spec, "missing": missing, "paths": []}
        else:
            spec = {**spec, "missing": [], "paths": [pool[name] for name in spec["weights"]]}
        candidates.append(spec)
    return candidates


def run_val(
    *,
    val_python: Path,
    workspace: Path,
    split_cfg: Path,
    split_name: str,
    candidate: dict,
    args: argparse.Namespace,
) -> dict:
    report_root = workspace / "validation_reports"
    log_root = report_root / "27h_model_level_moe_logs"
    run_root = report_root / "27h_model_level_moe_val_runs"
    log_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    safe_label = safe_name(f"{candidate['label']}_{split_name}")
    log_file = log_root / f"{safe_label}.log"
    row = {
        "candidate": candidate["label"],
        "split": split_name,
        "augment": candidate["augment"],
        "num_weights": len(candidate["paths"]),
        "weight_names": " ".join(candidate["weights"]),
        "idea": candidate["idea"],
        "log_file": str(log_file),
        "status": "skipped",
    }
    if candidate["missing"]:
        row["status"] = "missing_checkpoint"
        row["error"] = " ".join(candidate["missing"])
        return row

    cmd = [
        str(val_python),
        "val.py",
        "--weights",
        *[str(path) for path in candidate["paths"]],
        "--cfg",
        str(split_cfg),
        "--batch-size",
        str(args.batch_size),
        "--imgsz",
        str(args.imgsz),
        "--conf-thres",
        str(args.conf_thres),
        "--iou-thres",
        str(args.iou_thres),
        "--project",
        str(run_root),
        "--name",
        safe_label,
        "--exist-ok",
        "--no-plots",
        "--verbose",
    ]
    if candidate["augment"]:
        cmd.append("--augment")
    if args.device:
        cmd.extend(["--device", args.device])
    row["command"] = " ".join(cmd)
    result = subprocess.run(cmd, cwd=ET_ROOT, capture_output=True, text=True)
    log_file.write_text(result.stdout + "\nSTDERR\n" + result.stderr, encoding="utf-8")
    row["returncode"] = result.returncode
    if result.returncode == 0:
        row.update(parse_val_stdout(result.stdout))
        row["status"] = "ok"
    else:
        row["status"] = "failed"
        row["error"] = (result.stderr or result.stdout)[-1200:]
    return row


def append_research_summary(row: dict, workspace: Path) -> None:
    path = REPORTS_ROOT / "27_research_loop_mAP_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial",
        "status",
        "best_map50",
        "best_map50_95",
        "warmup_map50",
        "repair_map50",
        "dqa_aggregate_map50",
        "dqa_repair_map50",
        "workspace",
        "notebook",
        "log",
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
                "trial": "27h_model_level_moe_tta",
                "status": "target_reached" if float(row.get("map50", 0.0) or 0.0) >= 0.60 else "completed",
                "best_map50": row.get("map50", ""),
                "best_map50_95": row.get("map50_95", ""),
                "warmup_map50": row.get("warmup_map50", ""),
                "repair_map50": "",
                "dqa_aggregate_map50": "",
                "dqa_repair_map50": "",
                "workspace": str(workspace),
                "notebook": str(AGG_ROOT / "notebooks" / "research_loop_until_060" / "002_27h_model_level_moe_tta.ipynb"),
                "log": row.get("log_file", ""),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "rationale": (
                    "Recent model-level detection MoE and FedDG-MoE papers motivate test-time "
                    "prediction fusion instead of destructive weight averaging. Probe ensembles "
                    "warmup, repair experts, and 27g client specialists on the official total split."
                ),
            }
        )


def notify_discord(message: str, title: str) -> None:
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from notebook_notify import notify_discord as send  # type: ignore

        print(send(message, title=title, fail_silently=True))
    except Exception as exc:  # pragma: no cover - notification must not break experiments
        print(f"Discord notification skipped: {exc}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    setup = load_scene_setup(workspace)
    manifest = setup.build_data_lists()
    split_specs = select_split_specs(manifest["paper_evaluation"], PAPER_SPLITS)
    split_cfgs = {split["name"]: write_eval_config(setup, workspace, split, args) for split in split_specs}
    total_split = split_specs[-1]
    val_python = select_val_python(args.python_executable)

    pool = checkpoint_pool()
    candidates = make_candidates(pool)
    candidate_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "val_python": str(val_python),
        "papers": [
            "Domain-Specialized Object Detection via Model-Level Mixtures of Experts (arXiv:2604.18256)",
            "FedDG-MoE: Test-Time Mixture-of-Experts Fusion for Federated Domain Generalization (CVPRW 2025)",
        ],
        "candidates": [
            {
                "label": c["label"],
                "weights": c["weights"],
                "paths": [str(path) for path in c["paths"]],
                "missing": c["missing"],
                "augment": c["augment"],
                "idea": c["idea"],
            }
            for c in candidates
        ],
    }
    (workspace / "stats").mkdir(parents=True, exist_ok=True)
    (workspace / "stats" / "27h_model_level_moe_manifest.json").write_text(
        json.dumps(candidate_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not args.no_discord:
        notify_discord(
            "27h started: model-level/test-time MoE probe. "
            "Total split first, using warmup, 27d/27e/27g repair experts, 27g client experts, and TTA variants.",
            "DQA-MoX 27h started",
        )

    total_rows = []
    for candidate in candidates:
        row = run_val(
            val_python=val_python,
            workspace=workspace,
            split_cfg=split_cfgs[total_split["name"]],
            split_name=total_split["name"],
            candidate=candidate,
            args=args,
        )
        total_rows.append(row)
        print(
            f"{row['candidate']} total status={row['status']} "
            f"mAP50={row.get('map50', '')} mAP50:95={row.get('map50_95', '')}"
        )

    fieldnames = [
        "candidate",
        "split",
        "augment",
        "num_weights",
        "weight_names",
        "idea",
        "status",
        "returncode",
        "images",
        "labels",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "log_file",
        "command",
        "error",
    ]
    total_csv = workspace / "stats" / "27h_model_level_moe_total_metrics.csv"
    write_rows(total_csv, total_rows, fieldnames)
    ok_rows = [row for row in total_rows if row["status"] == "ok" and row.get("map50") is not None]
    if not ok_rows:
        raise RuntimeError("No successful 27h total evaluations.")
    warmup_row = next((row for row in ok_rows if row["candidate"] == "warmup_single"), None)
    warmup_map50 = float(warmup_row["map50"]) if warmup_row else 0.0
    best = max(ok_rows, key=lambda row: (float(row["map50"]), float(row.get("map50_95", 0.0))))
    best["warmup_map50"] = warmup_map50
    append_research_summary(best, workspace)

    full_rows: list[dict] = []
    best_gain = float(best["map50"]) - warmup_map50
    if float(best["map50"]) >= args.target_map50 or best_gain >= args.full_split_min_gain:
        best_candidate = next(c for c in candidates if c["label"] == best["candidate"])
        for split in split_specs:
            row = run_val(
                val_python=val_python,
                workspace=workspace,
                split_cfg=split_cfgs[split["name"]],
                split_name=split["name"],
                candidate=best_candidate,
                args=args,
            )
            full_rows.append(row)
            print(
                f"{row['candidate']} {row['split']} status={row['status']} "
                f"mAP50={row.get('map50', '')} mAP50:95={row.get('map50_95', '')}"
            )
        write_rows(workspace / "stats" / "27h_model_level_moe_best_split_metrics.csv", full_rows, fieldnames)

    summary = [
        f"27h finished. Best total candidate: {best['candidate']}",
        f"- total mAP50={best['map50']} / mAP50:95={best.get('map50_95', '')}",
        f"- warmup total mAP50={warmup_map50:.3f}; gain={best_gain:+.3f}",
        f"- total CSV: {total_csv}",
    ]
    if full_rows:
        summary.append(f"- full split CSV: {workspace / 'stats' / '27h_model_level_moe_best_split_metrics.csv'}")
    if float(best["map50"]) < args.target_map50:
        summary.append("Decision: target 0.600 not reached; next loop should move beyond output-level fusion.")
    else:
        summary.append("Decision: target reached.")
    message = "\n".join(summary)
    print(message)
    if not args.no_discord:
        notify_discord(message, "DQA-MoX 27h result")
    return 0 if float(best["map50"]) >= args.target_map50 else 2


if __name__ == "__main__":
    raise SystemExit(main())
