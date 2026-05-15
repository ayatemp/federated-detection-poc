# FedSTO Paper Reproduction

This directory is a paper-faithful reproduction package for:

**Navigating Data Heterogeneity in Federated Learning: A Semi-Supervised Federated Object Detection**
NeurIPS 2023, FedSTO.

Public sources used:

- Paper: https://arxiv.org/abs/2310.17097
- NeurIPS PDF: https://papers.nips.cc/paper_files/paper/2023/file/066e4dbfeccb5dc2851acd5eca584937-Paper-Conference.pdf
- Supplemental: https://papers.neurips.cc/paper_files/paper/2023/file/066e4dbfeccb5dc2851acd5eca584937-Supplemental-Conference.pdf
- Official repository: https://github.com/Kthyeon/ssfod
- EfficientTeacher: https://github.com/AlibabaResearch/efficientteacher

The official FedSTO repository currently publishes dataset setup scripts, but not the training implementation. The README says the code implementation is still to be uploaded. This package therefore targets the maximum reproducibility possible from public artifacts: official data setup metadata, paper hyperparameters, YOLOv5L/EfficientTeacher SSOD training, local EMA pseudo labelers, FedSTO phase scheduling, selective backbone training, full-parameter Phase 2, and non-backbone orthogonal regularization.

## Layout

- `external/ssfod_official`: snapshot of the official public repository.
- `external/efficientteacher`: symlink to the local EfficientTeacher vendor used for the executable reproduction.
- `scripts/prepare_bdd100k_paper20k.py`: prepares the BDD100K paper-scale split.
- `scripts/setup_fedsto_paper_reproduction.py`: creates paper-style data lists and EfficientTeacher configs.
- `scripts/run_fedsto_paper_reproduction.py`: runs the FedSTO warmup, Phase 1, and Phase 2 protocol.
- `scripts/evaluate_paper_protocol.py`: evaluates checkpoints on cloudy/overcast/rainy/snowy/total splits.
- `scripts/verify_reproduction.py`: audits implementation fidelity against the public paper specification.
- `paper_specs/fedsto_public_spec.md`: exact public spec and gap audit.

## Paper Protocol

Default run:

```bash
python scripts/run_fedsto_paper_reproduction.py \
  --workspace-root /app/Object_Detection/FedSTO/outputs/efficientteacher_fedsto \
  --warmup-epochs 50 \
  --phase1-rounds 100 \
  --phase2-rounds 150 \
  --batch-size 32 \
  --workers 0 \
  --gpus 1
```

Evaluation:

```bash
python scripts/evaluate_paper_protocol.py \
  --workspace /app/Object_Detection/FedSTO/outputs/efficientteacher_fedsto \
  --batch-size 8
```

Verification:

```bash
python scripts/verify_reproduction.py \
  --workspace-root /app/Object_Detection/FedSTO/outputs/verification_probe
```

## Important Reproduction Boundary

This package cannot honestly claim bit-for-bit 100% identity with the authors' private implementation because the training code and exact selected sample IDs were not published. It enforces every public item I could verify from the paper, supplemental, official repo, and EfficientTeacher implementation.

The verifier reports two separate notions:

- `public_spec_status`: whether all publicly stated requirements are implemented.
- `author_code_identity_status`: whether this can be proven identical to the unpublished author code.

The second one is expected to remain blocked until the authors publish their training code or exact split IDs.

One correction from the first draft: FedSTO's Appendix H.2 says each local EMA is reinitialized with the global model after every server broadcast. The default runner now follows that broadcast-reset behavior. The older cross-round persistent client EMA behavior is available only as `--persist-client-ema-across-rounds` for ablation, not as the paper default. The FedSTO configs also use fixed `ema_rate: 0.999`; EfficientTeacher's cosine EMA and `ModelEMA` ramp are disabled for the paper reproduction path.
