#!/usr/bin/env python3
"""Build and evaluate post-hoc LatentMoE expert-graft checkpoints.

27b-27e all suggest the same failure mode: DQA residual averaging is stable but
it washes out local MoE specialization before the detector can move toward the
0.60 mAP50 target.  This probe keeps a single deployable checkpoint, but packs
client residual experts into separate MoE slots and evaluates the result before
spending more GPU time on training.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parents[1]
AGG_ROOT = PROJECT_ROOT / "aggressive_dqamox"
OUTPUT_ROOT = AGG_ROOT / "output" / "27_research_notebook_until_060"
WORKSPACE_DEFAULT = AGG_ROOT / "output" / "27f_posthoc_expert_graft"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_scene_daynight_protocol.py"
ET_VENDOR = REPO_ROOT / "navigating_data_heterogeneity" / "vendor" / "efficientteacher"
SUMMARY_NAME = "27f_posthoc_expert_graft_total_metrics.csv"
EXPERT_RE = re.compile(r"head\.expert_m\.(\d+)\.(\d+)\.(weight|bias)$")

for path in (REPO_ROOT, ET_VENDOR, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from notebook_notify import notify_discord
except Exception:  # pragma: no cover - notification is best effort in notebooks
    notify_discord = None


@dataclass(frozen=True)
class SourceExpert:
    path: Path
    expert_idx: int | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class Variant:
    label: str
    base: Path
    slots: tuple[tuple[SourceExpert, ...], ...]
    moe_scale: float
    top_k: int = 4
    temperature: float = 1.0


def notify(message: str, *, title: str) -> None:
    if notify_discord is None:
        return
    try:
        notify_discord(message, title=title, fail_silently=True)
    except Exception:
        pass


def ckpt(workspace: str, filename: str) -> Path:
    return OUTPUT_ROOT / workspace / "checkpoints" / filename


def source_expert(path: Path, weight: float = 1.0, expert_idx: int | None = None) -> SourceExpert:
    return SourceExpert(path=path, expert_idx=expert_idx, weight=weight)


def default_variants() -> list[Variant]:
    w27c = "27c_probe_k6_night_tail_r2"
    w27d = "27d_probe_teacher_residual_mixpl_r2"
    w27e = "27e_probe_clean_day_expert_anchor_r2"

    day_slots = (
        (source_expert(ckpt(w27e, "latent_dqamox_p1_round002_client0_highway_day.pt")),),
        (source_expert(ckpt(w27e, "latent_dqamox_p1_round002_client2_citystreet_day.pt")),),
        (source_expert(ckpt(w27e, "latent_dqamox_p1_round002_client4_residential_day.pt")),),
        (
            source_expert(ckpt(w27d, "latent_dqamox_p1_round002_client1_highway_night.pt")),
            source_expert(ckpt(w27d, "latent_dqamox_p1_round001_client3_citystreet_night.pt")),
            source_expert(ckpt(w27d, "latent_dqamox_p1_round002_client5_residential_night.pt")),
        ),
    )
    night_slots = (
        (source_expert(ckpt(w27c, "latent_dqamox_p1_round002_client1_highway_night.pt")),),
        (source_expert(ckpt(w27d, "latent_dqamox_p1_round001_client3_citystreet_night.pt")),),
        (source_expert(ckpt(w27c, "latent_dqamox_p1_round002_client5_residential_night.pt")),),
        (source_expert(ckpt(w27e, "latent_dqamox_p1_round002_client0_highway_day.pt"), weight=0.35),),
        (source_expert(ckpt(w27e, "latent_dqamox_p1_round002_client2_citystreet_day.pt"), weight=0.35),),
        (
            source_expert(ckpt(w27e, "latent_dqamox_p1_round002_client4_residential_day.pt"), weight=0.35),
        ),
    )
    mixed_slots = (
        (source_expert(ckpt(w27d, "latent_dqamox_p1_round001_client0_highway_day.pt")),),
        (source_expert(ckpt(w27d, "latent_dqamox_p1_round002_client2_citystreet_day.pt")),),
        (source_expert(ckpt(w27d, "latent_dqamox_p1_round002_client1_highway_night.pt")),),
        (source_expert(ckpt(w27d, "latent_dqamox_p1_round002_client5_residential_night.pt")),),
    )

    return [
        Variant(
            label="27f_day3_nightblend_s1p0",
            base=ckpt(w27e, "latent_dqamox_p1_round002_server_repair.pt"),
            slots=day_slots,
            moe_scale=1.0,
        ),
        Variant(
            label="27f_day3_nightblend_s1p8",
            base=ckpt(w27e, "latent_dqamox_p1_round002_server_repair.pt"),
            slots=day_slots,
            moe_scale=1.8,
        ),
        Variant(
            label="27f_night3_dayblend_s1p4",
            base=ckpt(w27c, "latent_dqamox_p1_round002_server_repair.pt"),
            slots=night_slots,
            moe_scale=1.4,
        ),
        Variant(
            label="27f_27d_tail_slots_s1p5",
            base=ckpt(w27d, "latent_dqamox_p1_round002_server_repair.pt"),
            slots=mixed_slots,
            moe_scale=1.5,
        ),
    ]


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def model_state(ckpt_obj: dict[str, Any], key: str = "model") -> dict[str, torch.Tensor]:
    model = ckpt_obj.get(key)
    if model is None or not hasattr(model, "state_dict"):
        raise ValueError(f"checkpoint has no loadable {key!r} model")
    return {k: v.detach().cpu() for k, v in model.float().state_dict().items()}


def num_experts_from_state(state: dict[str, torch.Tensor]) -> int:
    experts = {int(match.group(2)) for key in state for match in [EXPERT_RE.fullmatch(key)] if match}
    if not experts:
        raise ValueError("no LatentMoE expert_m keys found")
    return max(experts) + 1


def best_expert_index(state: dict[str, torch.Tensor]) -> int:
    norms: dict[int, float] = {}
    for key, tensor in state.items():
        match = EXPERT_RE.fullmatch(key)
        if not match or not torch.is_tensor(tensor):
            continue
        expert_idx = int(match.group(2))
        value = tensor.float()
        norms[expert_idx] = norms.get(expert_idx, 0.0) + float(torch.sum(value * value))
    if not norms:
        raise ValueError("no expert tensors for source checkpoint")
    return max(sorted(norms), key=lambda idx: norms[idx])


def source_tensor_for_slot(
    slot_sources: tuple[SourceExpert, ...],
    key_template: str,
    checkpoint_cache: dict[Path, dict[str, Any]],
    state_cache: dict[Path, dict[str, torch.Tensor]],
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    weights: list[float] = []
    for source in slot_sources:
        if source.path not in checkpoint_cache:
            checkpoint_cache[source.path] = load_checkpoint(source.path)
            state_cache[source.path] = model_state(checkpoint_cache[source.path])
        source_state = state_cache[source.path]
        expert_idx = best_expert_index(source_state) if source.expert_idx is None else source.expert_idx
        source_key = key_template.format(expert_idx=expert_idx)
        values.append(source_state[source_key].float())
        weights.append(float(source.weight))
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-12)
    stacked = torch.stack(values, dim=0)
    return torch.sum(stacked * weight_tensor.view(-1, *([1] * (stacked.ndim - 1))), dim=0)


def set_head_runtime(model: Any, *, top_k: int, temperature: float, moe_scale: float) -> None:
    head = getattr(model, "head", None)
    if head is None and hasattr(model, "module"):
        head = getattr(model.module, "head", None)
    if head is None:
        raise ValueError("model has no head attribute")
    head.top_k = int(top_k)
    head.temperature = float(temperature)
    head.moe_scale = float(moe_scale)


def build_variant(variant: Variant, output_path: Path) -> dict[str, Any]:
    base = copy.deepcopy(load_checkpoint(variant.base))
    state = model_state(base)
    num_experts = num_experts_from_state(state)
    if len(variant.slots) != num_experts:
        raise ValueError(f"{variant.label}: expected {num_experts} slots, got {len(variant.slots)}")

    checkpoint_cache: dict[Path, dict[str, Any]] = {}
    state_cache: dict[Path, dict[str, torch.Tensor]] = {}
    new_state = dict(state)

    for key, target in state.items():
        match = EXPERT_RE.fullmatch(key)
        if match:
            level_idx, slot_idx, param_name = int(match.group(1)), int(match.group(2)), match.group(3)
            key_template = f"head.expert_m.{level_idx}.{{expert_idx}}.{param_name}"
            grafted = source_tensor_for_slot(variant.slots[slot_idx], key_template, checkpoint_cache, state_cache)
            new_state[key] = grafted.to(dtype=target.dtype)
        elif key.startswith("head.router.") and torch.is_tensor(target):
            new_state[key] = torch.zeros_like(target)

    base["model"].float().load_state_dict(new_state, strict=True)
    set_head_runtime(base["model"], top_k=variant.top_k, temperature=variant.temperature, moe_scale=variant.moe_scale)
    base["ema"] = None
    base["optimizer"] = None
    base["epoch"] = -1
    base["fedsto_stage"] = f"{variant.label}_posthoc_expert_graft"
    base["codex_posthoc_expert_graft"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base": str(variant.base),
        "moe_scale": variant.moe_scale,
        "top_k": variant.top_k,
        "temperature": variant.temperature,
        "slots": [
            [{"path": str(src.path), "expert_idx": src.expert_idx, "weight": src.weight} for src in slot]
            for slot in variant.slots
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, output_path)
    return base["codex_posthoc_expert_graft"]


def read_eval_totals(workspace: Path) -> list[dict[str, str]]:
    path = workspace / "validation_reports" / "paper_protocol_eval_summary.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("split") in {"total", "scene_daynight_total"}
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(workspace: Path, checkpoint_specs: list[tuple[str, Path]], args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT.resolve()),
        "--workspace",
        str(workspace.resolve()),
        "--splits",
        args.splits,
        "--batch-size",
        str(args.batch_size),
        "--no-plots",
        "--verbose",
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    for label, path in checkpoint_specs:
        cmd.extend(["--checkpoint", f"{label}={path.resolve()}"])
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def summarize(workspace: Path, target_map50: float) -> list[dict[str, str]]:
    totals = read_eval_totals(workspace)
    rows: list[dict[str, str]] = []
    for row in totals:
        rows.append(
            {
                "checkpoint_label": row.get("checkpoint_label", ""),
                "map50": row.get("map50", ""),
                "map50_95": row.get("map50_95", ""),
                "precision": row.get("precision", ""),
                "recall": row.get("recall", ""),
                "checkpoint_path": row.get("checkpoint_path", ""),
            }
        )
    write_csv(
        workspace / "stats" / SUMMARY_NAME,
        rows,
        ["checkpoint_label", "map50", "map50_95", "precision", "recall", "checkpoint_path"],
    )
    best = max((float(row["map50"]) for row in rows if row.get("map50")), default=float("nan"))
    message_lines = [
        f"target_mAP50={target_map50:.3f}",
        f"best_total_mAP50={best:.6f}" if best == best else "best_total_mAP50=nan",
        "",
        *[f"{row['checkpoint_label']}: mAP50={row['map50']} / mAP50:95={row['map50_95']}" for row in rows],
    ]
    notify("\n".join(message_lines), title="DQA-MoX 27f posthoc expert-graft result")
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE_DEFAULT)
    parser.add_argument("--target-map50", type=float, default=0.60)
    parser.add_argument("--splits", default="total")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.resolve()
    checkpoint_root = workspace / "checkpoints"
    stats_root = workspace / "stats"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    stats_root.mkdir(parents=True, exist_ok=True)

    variants = default_variants()
    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "paper_basis": [
            "FedDG-MoE: test-time fusion keeps client-specific MoE adapters instead of averaging them away.",
            "Uncertainty-aware long-tailed pseudo-label weights: do not hard-drop noisy/tail signals; downweight them.",
            "TMLR 2025 SSOD building blocks: class imbalance and missing detections make pseudo-label quantity/quality trade-offs central.",
        ],
        "variants": [],
    }
    built_specs: list[tuple[str, Path]] = []
    for variant in variants:
        out = checkpoint_root / f"{variant.label}.pt"
        print(f"building {variant.label} -> {out}", flush=True)
        meta = build_variant(variant, out)
        manifest["variants"].append({"label": variant.label, "checkpoint": str(out), **meta})
        built_specs.append((variant.label, out))

    w27d = "27d_probe_teacher_residual_mixpl_r2"
    w27e = "27e_probe_clean_day_expert_anchor_r2"
    baseline_specs = [
        ("warmup_global", ckpt(w27e, "round000_latent_dqamox_warmup.pt")),
        ("27d_final_repair", ckpt(w27d, "latent_dqamox_p1_round002_server_repair.pt")),
        ("27e_final_repair", ckpt(w27e, "latent_dqamox_p1_round002_server_repair.pt")),
    ]
    manifest["baselines"] = [{"label": label, "checkpoint": str(path)} for label, path in baseline_specs]
    (stats_root / "27f_posthoc_expert_graft_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    notify(
        "27f posthoc expert-graftを開始: FedDG-MoEのadapter fusion風にclient residual expertsを単一checkpointへ詰め替え、まずtotalだけ評価します。",
        title="DQA-MoX 27f start",
    )

    if not args.skip_eval:
        evaluate(workspace, baseline_specs + built_specs, args)
    rows = summarize(workspace, args.target_map50)
    if rows:
        best = max(float(row["map50"]) for row in rows if row.get("map50"))
        print(f"best_total_mAP50={best:.6f}", flush=True)
    print(f"Saved: {stats_root / SUMMARY_NAME}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
