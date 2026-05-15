#!/usr/bin/env python3
"""Greedy-soup judger for monotonic DQA-SoftMoX model selection.

The delta optimizer showed that mini validation can over-select harmful deltas.
This loop borrows the greedy model-soup idea: only add a candidate checkpoint to
the global soup if the proxy score improves.  The judge is therefore monotonic on
the proxy by construction, while still using only self-generated checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "output" / "05_greedy_soup_judger"
SOURCE_WORKSPACE = REPO_ROOT / "dynamic_quality_aware_classwise_aggregation" / "moe_dqa" / "output" / "01_dqa_fedmox_yolo_full"

if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_01_judger_probe as judger01  # noqa: E402
import run_02_mix_weight_optimizer as opt02  # noqa: E402


def notify(message: str, title: str, enabled: bool) -> None:
    if not enabled:
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from notebook_notify import notify_discord

        notify_discord(message, title=title, fail_silently=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notify skipped: {exc}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_val_stdout(stdout: str) -> dict[str, float]:
    return opt02.parse_val_stdout(stdout)


def eval_checkpoint(path: Path, cfg: Path, name: str, args: argparse.Namespace) -> dict[str, Any]:
    return opt02.eval_checkpoint(path, cfg, name, args)


def score_row(row: dict[str, Any]) -> float:
    return opt02.score_row(row)


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


def average_state_dicts(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    base = states[0]
    out: dict[str, torch.Tensor] = {}
    for key, value in base.items():
        values = [state.get(key) for state in states]
        if (
            torch.is_tensor(value)
            and all(torch.is_tensor(v) and v.shape == value.shape for v in values)
            and value.dtype.is_floating_point
        ):
            avg = torch.stack([v.float() for v in values]).mean(dim=0)
            out[key] = avg.to(value.dtype)
        else:
            out[key] = value
    return out


def build_soup(paths: list[Path], output: Path, args: argparse.Namespace) -> Path:
    if output.exists() and not args.force:
        return output
    ckpts = [judger01.load_checkpoint(path) for path in paths]
    out = copy.deepcopy(ckpts[0])
    for field in ("model", "ema"):
        states = [state_dict_from(ckpt, field) for ckpt in ckpts]
        if any(state is None for state in states):
            continue
        replace_state(out, field, average_state_dicts(states))  # type: ignore[arg-type]
    out["epoch"] = -1
    out["optimizer"] = None
    out["greedy_soup_judger"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "members": [str(path) for path in paths],
        "method": "uniform average of greedily accepted self-generated checkpoints",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    return output


def candidate_pool(args: argparse.Namespace) -> list[dict[str, Any]]:
    ckpt_dir = args.source_workspace / "checkpoints"
    rows: list[dict[str, Any]] = [
        {"label": "warmup_g0", "source": "source", "round": 0, "path": ckpt_dir / "round000_latent_dqamox_warmup.pt"},
    ]
    for round_idx in range(1, args.max_round + 1):
        paths = judger01.checkpoint_paths(args.source_workspace, round_idx)
        for key, path in paths.items():
            if path.exists():
                rows.append({"label": f"r{round_idx:03d}_{key}", "source": "source", "round": round_idx, "path": path})

    extra_dirs = [
        PROJECT_ROOT / "output" / "02_mix_weight_optimizer_expanded" / "candidates",
        PROJECT_ROOT / "output" / "03_mix_judger_policy" / "candidates",
        PROJECT_ROOT / "output" / "04_delta_expert_optimizer" / "candidates",
    ]
    for directory in extra_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.pt")):
            rows.append({"label": path.stem, "source": directory.parent.name, "round": -1, "path": path})

    seen: set[Path] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        path = Path(row["path"]).resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        row["path"] = path
        out.append(row)
    return out


def mini_eval_config(args: argparse.Namespace) -> Path:
    _mini_list, mini_cfg = opt02.ensure_mini_eval_config(args)
    return mini_cfg


def full_eval_config(args: argparse.Namespace) -> Path:
    return opt02.full_eval_config(args)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.source_workspace = args.source_workspace.expanduser().resolve()
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    (args.workspace_root / "stats").mkdir(parents=True, exist_ok=True)
    mini_cfg = mini_eval_config(args)
    full_cfg = full_eval_config(args)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "dqa_softmox_greedy_soup_judger_v0",
        "method": "greedy proxy-monotonic model soup over self-generated checkpoints",
        "workspace": str(args.workspace_root),
        "source_workspace": str(args.source_workspace),
        "max_round": args.max_round,
        "mini_images": args.mini_images,
    }
    (args.workspace_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    notify(
        f"Greedy soup judger started\nmax_round={args.max_round}, mini_images={args.mini_images}",
        "DQA-SoftMoX 05 started",
        args.notify_discord,
    )

    pool = candidate_pool(args)
    individual_rows: list[dict[str, Any]] = []
    for idx, item in enumerate(pool[: args.max_candidates]):
        metrics = eval_checkpoint(Path(item["path"]), mini_cfg, f"ind_{idx:03d}_{item['label']}_mini", args)
        row = {**item, "eval_scope": "mini_individual", **metrics}
        row["path"] = str(item["path"])
        row["score"] = score_row(row)
        individual_rows.append(row)
    write_csv(args.workspace_root / "stats" / "05_individual_candidates.csv", individual_rows)

    ranked = sorted(individual_rows, key=lambda row: float(row["score"]), reverse=True)
    accepted: list[dict[str, Any]] = [ranked[0]]
    current_score = float(ranked[0]["score"])
    current_path = Path(ranked[0]["path"])
    soup_rows: list[dict[str, Any]] = [
        {
            "step": 1,
            "action": "seed",
            "candidate": ranked[0]["label"],
            "accepted": True,
            "member_count": 1,
            "score": current_score,
            "path": str(current_path),
            "map50": ranked[0]["map50"],
            "map50_95": ranked[0]["map50_95"],
        }
    ]

    for step, candidate in enumerate(ranked[1 : args.greedy_limit], start=2):
        proposal_members = [Path(row["path"]) for row in accepted] + [Path(candidate["path"])]
        proposal_path = args.workspace_root / "candidates" / f"greedy_soup_step{step:03d}_{candidate['label']}.pt"
        build_soup(proposal_members, proposal_path, args)
        metrics = eval_checkpoint(proposal_path, mini_cfg, f"soup_step{step:03d}_{candidate['label']}_mini", args)
        row = {
            "step": step,
            "action": "try_add",
            "candidate": candidate["label"],
            "candidate_path": candidate["path"],
            "member_count": len(proposal_members),
            "path": str(proposal_path),
            **metrics,
        }
        row["score"] = score_row(row)
        accepted_flag = float(row["score"]) >= current_score + args.accept_margin
        row["accepted"] = accepted_flag
        if accepted_flag:
            accepted.append(candidate)
            current_score = float(row["score"])
            current_path = proposal_path
        elif not args.keep_rejected:
            try:
                proposal_path.unlink()
            except OSError:
                pass
        soup_rows.append(row)
        if accepted_flag:
            notify(
                "Greedy soup accepted step {step}: {candidate}\nmini mAP50={map50:.3f}, mAP50:95={map50_95:.3f}, score={score:.4f}, members={member_count}".format(**row),
                "DQA-SoftMoX 05 accepted",
                args.notify_discord,
            )
    write_csv(args.workspace_root / "stats" / "05_greedy_soup_steps.csv", soup_rows)

    full_rows: list[dict[str, Any]] = []
    accepted_steps = [row for row in soup_rows if row.get("accepted") in {True, "True", "true"}]
    for row in accepted_steps[-args.full_eval_last :]:
        path = Path(row["path"])
        metrics = eval_checkpoint(path, full_cfg, f"full_{Path(path).stem}", args)
        full = {**row, "eval_scope": "full_total", **metrics}
        full["score"] = score_row(full)
        full_rows.append(full)
    write_csv(args.workspace_root / "stats" / "05_greedy_soup_full_eval.csv", full_rows)

    report = [
        "# DQA-SoftMoX Greedy Soup Judger 05",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- pool_size: {len(pool)}",
        f"- evaluated_individuals: {len(individual_rows)}",
        "",
        "## Accepted Mini Path",
        "",
        "| step | candidate | members | mini mAP50 | mini mAP50:95 | mini score |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in accepted_steps:
        report.append("| {step} | {candidate} | {member_count} | {map50:.3f} | {map50_95:.3f} | {score:.4f} |".format(**row))
    report.extend(["", "## Full Evaluation", "", "| step | candidate | members | mAP50 | mAP50:95 | score |", "|---:|---|---:|---:|---:|---:|"])
    for row in full_rows:
        report.append("| {step} | {candidate} | {member_count} | {map50:.3f} | {map50_95:.3f} | {score:.4f} |".format(**row))
    (args.workspace_root / "05_greedy_soup_judger_report.md").write_text("\n".join(report), encoding="utf-8")

    notify(
        "Greedy soup judger finished\n" + "\n".join(
            f"- step {int(row['step'])} {row['candidate']}: full mAP50={float(row['map50']):.3f}, mAP50:95={float(row['map50_95']):.3f}, score={float(row['score']):.4f}"
            for row in full_rows
        ),
        "DQA-SoftMoX 05 finished",
        args.notify_discord,
    )
    result = {"manifest": manifest, "accepted": accepted_steps, "full": full_rows}
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-workspace", type=Path, default=SOURCE_WORKSPACE)
    parser.add_argument("--max-round", type=int, default=6)
    parser.add_argument("--mini-images", type=int, default=512)
    parser.add_argument("--max-candidates", type=int, default=36)
    parser.add_argument("--greedy-limit", type=int, default=24)
    parser.add_argument("--accept-margin", type=float, default=0.0)
    parser.add_argument("--full-eval-last", type=int, default=6)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-rejected", action="store_true")
    parser.add_argument("--notify-discord", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
