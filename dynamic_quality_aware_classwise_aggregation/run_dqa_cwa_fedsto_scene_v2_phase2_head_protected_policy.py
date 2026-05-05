#!/usr/bin/env python3
"""Run DQA08_2: phase2-only adaptation from an existing DQA08 phase1 seed.

This runner is a comparison experiment for notebook 08_2.  It copies a selected
DQA08 phase1 global checkpoint into a fresh workspace as ``round000_warmup.pt``
and then runs only phase2.  The phase2 client update is intentionally
head-protected:

- target clients train the backbone only
- low/mid pseudo labels are treated as ignored regions, not positive objectness
- high pseudo labels remain normal bbox + obj + cls targets
- the source/server update keeps the full model trainable

The goal is to test whether phase2 was hurting precision mainly by letting noisy
target pseudo labels rewrite the detector head.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import run_dqa_cwa_fedsto as base
import run_dqa_cwa_fedsto_scene_v2_tri_stage_policy as tri_stage


DQA_PROTOCOL_VERSION = "dqa08_2_scene_phase2_head_protected_policy_v1"
DEFAULT_WORK_ROOT = base.RESEARCH_ROOT / "efficientteacher_dqa08_2_scene_phase2_head_protected"
DEFAULT_SOURCE_WORK_ROOT = base.RESEARCH_ROOT / "efficientteacher_dqa08_scene_tri_stage_policy_8h"
DEFAULT_SOURCE_PHASE1_ROUND = 12


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return float(raw)


def _source_workspace() -> Path:
    return Path(os.environ.get("DQA08_2_SOURCE_WORK_ROOT", str(DEFAULT_SOURCE_WORK_ROOT))).resolve()


def _source_phase1_round() -> int:
    return int(os.environ.get("DQA08_2_SOURCE_PHASE1_ROUND", str(DEFAULT_SOURCE_PHASE1_ROUND)))


def _seed_metadata_path(workspace_root: Path) -> Path:
    return workspace_root / "phase2_seed.json"


def _seed_phase1_checkpoint(workspace_root: Path) -> Path:
    """Copy DQA08 phase1 checkpoint as this experiment's initial global model."""
    workspace_root = workspace_root.resolve()
    source_root = _source_workspace()
    source_round = _source_phase1_round()
    source = source_root / "global_checkpoints" / f"phase1_round{source_round:03d}_global.pt"
    target = workspace_root / "global_checkpoints" / "round000_warmup.pt"
    metadata_path = _seed_metadata_path(workspace_root)
    force = _bool_env("DQA08_2_FORCE_SEED", False)

    if not source.exists():
        raise FileNotFoundError(
            f"Missing DQA08 phase1 seed checkpoint: {source}\n"
            "Set DQA08_2_SOURCE_WORK_ROOT / DQA08_2_SOURCE_PHASE1_ROUND if you want a different seed."
        )

    wanted: dict[str, Any] = {
        "protocol": DQA_PROTOCOL_VERSION,
        "seed_kind": "dqa08_phase1_global_as_round000_warmup",
        "source_work_root": str(source_root),
        "source_phase1_round": source_round,
        "source_checkpoint": str(source),
        "target_checkpoint": str(target.resolve()),
        "client_train_scope": os.environ.get("DQA08_2_CLIENT_TRAIN_SCOPE", "backbone"),
        "uncertain_ignore": _bool_env("DQA08_2_UNCERTAIN_IGNORE", True),
    }

    if target.exists() and not force:
        if metadata_path.exists():
            try:
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
            same_seed = (
                existing.get("protocol") == wanted["protocol"]
                and existing.get("source_checkpoint") == wanted["source_checkpoint"]
                and existing.get("client_train_scope") == wanted["client_train_scope"]
                and existing.get("uncertain_ignore") == wanted["uncertain_ignore"]
            )
            if same_seed:
                print(f"Reusing 08_2 phase2 seed checkpoint: {target}")
                return target
        raise RuntimeError(
            f"Seed checkpoint already exists with different or unknown metadata: {target}\n"
            f"Existing metadata: {metadata_path if metadata_path.exists() else '(missing)'}\n"
            "Use DQA08_2_FORCE_SEED=1 only if you intentionally want to overwrite the seed."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    metadata_path.write_text(json.dumps(wanted, indent=2), encoding="utf-8")
    print(f"Seeded 08_2 phase2 from DQA08 phase1 round {source_round}: {target}")
    return target


def _prepare_head_protected_scene_modules(work_root: Path):
    setup, fedsto = tri_stage._prepare_tri_stage_scene_modules(work_root)
    tri_stage_config = setup.efficientteacher_config

    client_scope = os.environ.get("DQA08_2_CLIENT_TRAIN_SCOPE", "backbone")
    server_scope = os.environ.get("DQA08_2_SERVER_TRAIN_SCOPE", "all")
    uncertain_ignore = _bool_env("DQA08_2_UNCERTAIN_IGNORE", True)
    client_lr0 = _float_env("DQA08_2_CLIENT_LR0", _float_env("DQA08_CLIENT_LR0", 3e-4))
    server_orthogonal = _float_env("DQA08_2_SERVER_ORTHOGONAL_WEIGHT", 1e-4)

    def head_protected_config(**kwargs):
        cfg = tri_stage_config(**kwargs)
        target = kwargs.get("target")
        name = str(kwargs.get("name", ""))

        if target is not None:
            cfg.setdefault("FedSTO", {})["train_scope"] = client_scope
            cfg.setdefault("hyp", {})["lr0"] = client_lr0
            cfg["hyp"]["lrf"] = 1.0
            cfg.setdefault("SSOD", {})["ignore_obj"] = uncertain_ignore
            cfg["SSOD"]["pseudo_label_with_obj"] = True
            cfg["SSOD"]["pseudo_label_with_bbox"] = False
            cfg["SSOD"]["pseudo_label_with_cls"] = False
        elif name != "runtime_server_warmup":
            cfg.setdefault("FedSTO", {})["train_scope"] = server_scope
            cfg["FedSTO"]["orthogonal_weight"] = server_orthogonal

        return cfg

    setup.efficientteacher_config = head_protected_config
    setup.base.efficientteacher_config = head_protected_config
    fedsto.setup.efficientteacher_config = head_protected_config
    return setup, fedsto


base.DQA_PROTOCOL_VERSION = DQA_PROTOCOL_VERSION
base.DEFAULT_DQA_WORK_ROOT = DEFAULT_WORK_ROOT
base._prepare_fedsto_modules = _prepare_head_protected_scene_modules
base.run_train = tri_stage._tri_stage_run_train


def main() -> None:
    parsed = base.parse_args()
    if parsed.protocol_version == tri_stage.DQA_PROTOCOL_VERSION:
        parsed.protocol_version = DQA_PROTOCOL_VERSION
    if parsed.workspace_root == tri_stage.base.RESEARCH_ROOT / "efficientteacher_dqa08_scene_tri_stage_policy_8h":
        parsed.workspace_root = DEFAULT_WORK_ROOT

    os.environ["DQA08_STATS_ROOT"] = str(parsed.stats_root.resolve())
    os.environ["DQA08_PHASE2_ROUNDS"] = str(parsed.phase2_rounds)
    os.environ["DQA08_NUM_CLASSES"] = str(parsed.num_classes)
    os.environ.setdefault("DQA08_POLICY_HORIZON_ROUNDS", str(parsed.phase2_rounds))
    os.environ.setdefault(
        "DQA08_THRESHOLD_LOG",
        str((parsed.stats_root / "head_protected_policy_schedule.jsonl").resolve()),
    )

    if parsed.setup_only:
        base._prepare_fedsto_modules(parsed.workspace_root)[0].build_base_configs()
        return

    if parsed.phase1_rounds != 0:
        raise ValueError(
            "08_2 is designed as a phase2-only comparison. "
            "Run with --phase1-rounds 0 and seed from DQA08 phase1."
        )

    if not parsed.dry_run:
        _seed_phase1_checkpoint(parsed.workspace_root)
        if not tri_stage._policy_model_path().exists():
            raise FileNotFoundError(
                f"Missing learned threshold policy model: {tri_stage._policy_model_path()}. "
                "Run threshold_policy_model/01_train_dqa05_threshold_policy.ipynb first."
            )

    print("DQA08_2 profile: phase2-only head-protected policy")
    print(f"DQA08_2 seed: {_source_workspace()} phase1 round {_source_phase1_round()}")
    print(f"DQA08_2 client train_scope: {os.environ.get('DQA08_2_CLIENT_TRAIN_SCOPE', 'backbone')}")
    print(f"DQA08_2 uncertain ignore_obj: {_bool_env('DQA08_2_UNCERTAIN_IGNORE', True)}")
    base.run_protocol(parsed)


if __name__ == "__main__":
    main()
