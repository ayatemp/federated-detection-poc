#!/usr/bin/env python3
"""Audit this FedSTO reproduction against the public paper specification."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
OFFICIAL_REPO = PROJECT_ROOT / "external" / "ssfod_official"
EFFICIENTTEACHER = PROJECT_ROOT / "external" / "efficientteacher"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def assert_true(checks: list[dict], name: str, ok: bool, detail: str = "") -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def run_setup(workspace: Path) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        str(SCRIPTS / "run_fedsto_paper_reproduction.py"),
        "--workspace-root",
        str(workspace),
        "--setup-only",
    ]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
    output = (proc.stdout + "\n" + proc.stderr).strip()
    return proc.returncode == 0, output[-4000:]


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT / "outputs" / "verification_probe")
    parser.add_argument("--json-out", type=Path, default=PROJECT_ROOT / "outputs" / "verification_report.json")
    args = parser.parse_args()

    checks: list[dict] = []

    assert_true(checks, "FedSTO directory exists", PROJECT_ROOT.exists(), str(PROJECT_ROOT))
    assert_true(checks, "official ssfod snapshot exists", OFFICIAL_REPO.exists(), str(OFFICIAL_REPO))
    official_readme = OFFICIAL_REPO / "README.md"
    assert_true(checks, "official README exists", official_readme.exists(), str(official_readme))
    if official_readme.exists():
        text = read_text(official_readme)
        assert_true(
            checks,
            "official repo confirms training code not fully published",
            "Still in progress" in text and "To be uploaded" in text,
            "The public repo is dataset/setup-focused, not a full training release.",
        )

    assert_true(checks, "EfficientTeacher vendor exists", EFFICIENTTEACHER.exists(), str(EFFICIENTTEACHER))
    fedsto_reg = EFFICIENTTEACHER / "utils" / "fedsto_regularization.py"
    assert_true(checks, "FedSTO regularization module exists", fedsto_reg.exists(), str(fedsto_reg))
    if fedsto_reg.exists():
        reg_text = read_text(fedsto_reg)
        assert_true(checks, "train_scope supports backbone selective training", "scope == \"backbone\"" in reg_text)
        assert_true(checks, "non-backbone orthogonal regularization exists", "spectral_orthogonal_regularization" in reg_text)

    trainer = EFFICIENTTEACHER / "trainer" / "trainer.py"
    ssod_loss = EFFICIENTTEACHER / "models" / "loss" / "ssod" / "ssod_loss.py"
    assert_true(checks, "EfficientTeacher trainer applies FedSTO train scope", "apply_fedsto_train_scope" in read_text(trainer) if trainer.exists() else False)
    assert_true(checks, "EfficientTeacher trainer adds orthogonal loss during backprop", "spectral_orthogonal_regularization" in read_text(trainer) if trainer.exists() else False)
    if ssod_loss.exists():
        loss_text = read_text(ssod_loss)
        assert_true(checks, "SSOD loss uses low/high pseudo thresholds", "ignore_thres_low" in loss_text and "ignore_thres_high" in loss_text)
        assert_true(checks, "SSOD loss supports soft objectness pseudo labels", "pseudo_label_with_obj" in loss_text and "uc_scores" in loss_text)
        assert_true(checks, "SSOD loss supports pseudo bbox and cls routing", "pseudo_label_with_bbox" in loss_text and "pseudo_label_with_cls" in loss_text)

    run_script = read_text(SCRIPTS / "run_fedsto_paper_reproduction.py")
    assert_true(checks, "default warmup is 50", re.search(r"--warmup-epochs\".*, default=50", run_script, re.S) is not None)
    assert_true(checks, "default phase1 is 100", re.search(r"--phase1-rounds\".*, default=100", run_script, re.S) is not None)
    assert_true(checks, "default phase2 is 150", re.search(r"--phase2-rounds\".*, default=150", run_script, re.S) is not None)
    assert_true(checks, "local EMA checkpoint loading is implemented", "_checkpoint_teacher_model" in run_script and "base[\"ema\"]" in run_script)
    assert_true(
        checks,
        "default local EMA matches paper broadcast-reset behavior",
        "local_ema_source = previous if args.persist_client_ema_across_rounds else None" in run_script
        and "--persist-client-ema-across-rounds" in run_script,
        "Appendix H.2 states local EMA is reinitialized with global weights after each server broadcast.",
    )
    assert_true(
        checks,
        "broadcast-reset EMA uses global model weights, not stale server EMA",
        "reset_ema_to_model=local_ema_source is None" in run_script,
        "The client start checkpoint should set EMA from the broadcast model for paper-default runs.",
    )
    assert_true(
        checks,
        "runtime server configs keep fixed paper EMA when SSOD is disabled",
        '"train_domain": False' in run_script and '"ema_rate": 0.999' in run_script and '"cosine_ema": False' in run_script,
        "Server-update configs should not fall back to EfficientTeacher's default cosine EMA/ramp settings.",
    )

    setup_ok, setup_tail = run_setup(args.workspace_root)
    assert_true(checks, "setup-only config generation succeeds", setup_ok, setup_tail)

    manifest_path = args.workspace_root / "manifest.json"
    assert_true(checks, "manifest generated", manifest_path.exists(), str(manifest_path))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert_true(checks, "server cloudy train count is paper-scale", manifest["server"]["train_images"] == 4881)
        assert_true(checks, "three unlabeled clients exist", len(manifest["clients"]) == 3)
        assert_true(
            checks,
            "client weather split is overcast/rainy/snowy",
            [c["weather"] for c in manifest["clients"]] == ["overcast", "rainy", "snowy"],
        )
        assert_true(checks, "client total is 15000 images", sum(c["images"] for c in manifest["clients"]) == 15000)

    cfg_dir = args.workspace_root / "configs"
    phase1_cfg = cfg_dir / "client_0_overcast_phase1.yaml"
    phase2_cfg = cfg_dir / "client_0_overcast_phase2.yaml"
    warmup_cfg = cfg_dir / "server_warmup_yolov5l_bdd100k.yaml"
    for cfg_path in (warmup_cfg, phase1_cfg, phase2_cfg):
        assert_true(checks, f"config exists: {cfg_path.name}", cfg_path.exists(), str(cfg_path))

    if warmup_cfg.exists():
        cfg = load_yaml(warmup_cfg)
        assert_true(checks, "warmup config epochs=50", cfg["epochs"] == 50)
        assert_true(checks, "warmup lr0=0.01", abs(float(cfg["hyp"]["lr0"]) - 0.01) < 1e-12)
        assert_true(checks, "loss class/object/anchor match paper", cfg["Loss"]["cls"] == 0.3 and cfg["Loss"]["obj"] == 0.7 and cfg["Loss"]["anchor_t"] == 4.0)
        assert_true(checks, "server EMA rate matches paper", cfg["SSOD"]["ema_rate"] == 0.999)
        assert_true(checks, "server disables non-paper cosine EMA", cfg["SSOD"]["cosine_ema"] is False)

    if phase1_cfg.exists():
        cfg = load_yaml(phase1_cfg)
        ssod = cfg["SSOD"]
        assert_true(checks, "phase1 uses backbone-only training", cfg["FedSTO"]["train_scope"] == "backbone")
        assert_true(checks, "phase1 is SSOD client training", ssod["train_domain"] is True)
        assert_true(checks, "phase1 NMS conf/iou match paper", ssod["nms_conf_thres"] == 0.1 and ssod["nms_iou_thres"] == 0.65)
        assert_true(checks, "phase1 ignore thresholds match paper", ssod["ignore_thres_low"] == 0.1 and ssod["ignore_thres_high"] == 0.6)
        assert_true(checks, "phase1 EMA rate matches paper", ssod["ema_rate"] == 0.999)
        assert_true(checks, "phase1 disables non-paper cosine EMA", ssod["cosine_ema"] is False)

    if phase2_cfg.exists():
        cfg = load_yaml(phase2_cfg)
        assert_true(checks, "phase2 uses full-parameter training", cfg["FedSTO"]["train_scope"] == "all")
        assert_true(checks, "phase2 enables non-backbone orthogonal regularization", cfg["FedSTO"]["orthogonal_weight"] > 0 and cfg["FedSTO"]["orthogonal_scope"] == "non_backbone")

    trainer_text = read_text(EFFICIENTTEACHER / "trainer" / "trainer.py") if trainer.exists() else ""
    ssod_trainer_path = EFFICIENTTEACHER / "trainer" / "ssod_trainer.py"
    ssod_trainer_text = read_text(ssod_trainer_path) if ssod_trainer_path.exists() else ""
    torch_utils_text = read_text(EFFICIENTTEACHER / "utils" / "torch_utils.py")
    assert_true(checks, "ModelEMA supports fixed paper alpha without ramp", "ramp=True" in torch_utils_text and "self.decay = lambda x: decay" in torch_utils_text)
    assert_true(checks, "supervised trainer EMA decay reads cfg.SSOD.ema_rate", "decay=ema_decay, ramp=ema_ramp" in trainer_text)
    assert_true(checks, "SSOD trainer EMA decay reads cfg.SSOD.ema_rate", "decay=float(self.cfg.SSOD.ema_rate)" in ssod_trainer_text)
    assert_true(checks, "SSOD trainer disables ModelEMA ramp for paper config", "ramp=bool(self.cfg.SSOD.cosine_ema)" in ssod_trainer_text)

    failed = [check for check in checks if not check["ok"]]
    public_blockers = [
        "Official FedSTO training code is not published in Kthyeon/ssfod.",
        "Exact author-selected BDD100K sample IDs are not published.",
        "Any unpublished local EfficientTeacher edits used by the authors cannot be proven.",
    ]
    fidelity_gaps = [
        "The paper specifies YOLOv5 Large, but not a downloadable exact author checkpoint or full training environment.",
        "The paper lists augmentations, but exact probabilities/order for graying, Gaussian blur, and color conversion are not fully specified.",
        "The paper describes orthogonal regularization but does not publish the exact regularization coefficient used by the authors.",
        "Algorithm 1 samples clients, but the BDD 1-server/3-client experiment does not explicitly publish a client-sampling ratio; this implementation uses all three clients per round.",
        "The main text says 50/100/150 rounds, while the communication-cost table discusses 50/150/150.",
        "The official repository contains dataset setup scripts but no FedSTO trainer, so implementation identity cannot be verified.",
    ]
    report = {
        "project_root": str(PROJECT_ROOT),
        "public_spec_status": "pass" if not failed else "fail",
        "strict_100_percent_reproduction": False,
        "author_code_identity_status": "blocked_by_unpublished_artifacts",
        "honest_assessment": (
            "This is a high-fidelity public-spec reproduction, not a provable 100% reproduction of the "
            "authors' unpublished implementation."
        ),
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "failed_checks": failed,
        "public_artifact_blockers": public_blockers,
        "fidelity_gaps": fidelity_gaps,
        "checks": checks,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
