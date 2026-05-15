# FedSTO Reproduction Audit

Created: 2026-05-15

## Result

`scripts/verify_reproduction.py` passes all public-spec checks after EMA corrections:

- The first draft carried client EMA teachers across rounds.
- Appendix H.2 says local EMA is reinitialized from the broadcast global model after each server broadcast.
- The runner now defaults to broadcast-reset EMA. Cross-round persistent EMA is only available through `--persist-client-ema-across-rounds` as a non-paper ablation.
- EfficientTeacher's default `ModelEMA` used a 0.9999-style decay ramp. FedSTO states fixed alpha = 0.999, so FedSTO configs now set `ema_rate: 0.999`, disable cosine EMA, and instantiate `ModelEMA(..., ramp=False)` through `cosine_ema: false`.

- Checks passed: 46 / 46 in `outputs/final_paper_audit_report_v4.json`
- Public specification status: `pass`
- Strict 100% reproduction: `false`
- Author-code identity status: `blocked_by_unpublished_artifacts`

Verification report:

- `/app/Object_Detection/FedSTO/outputs/final_paper_audit_report_v4.json`

## What Was Verified

- Official public `Kthyeon/ssfod` snapshot is included under `external/ssfod_official`.
- Official repo currently confirms the training code is not fully published.
- EfficientTeacher SSOD trainer is available under `external/efficientteacher`.
- FedSTO train scope is implemented and supports backbone-only selective training.
- Non-backbone orthogonal regularization is applied during backpropagation.
- SSOD pseudo label loss supports low/high thresholds, soft objectness, bbox routing, and cls routing.
- Default schedule matches the main paper text: 50 warmup, 100 Phase 1, 150 Phase 2.
- Local EMA checkpoint loading is implemented, but paper-default behavior is broadcast-reset after server broadcast.
- Setup generation succeeds.
- BDD non-IID split is paper-scale:
  - server cloudy train: 4,881 images
  - client overcast: 5,000 images
  - client rainy: 5,000 images
  - client snowy: 5,000 images
- Configs match public paper hyperparameters:
  - lr0 = 0.01
  - class/object balance = 0.3 / 0.7
  - anchor threshold = 4.0
  - NMS conf/IoU = 0.1 / 0.65
  - ignore low/high = 0.1 / 0.6
  - EMA = 0.999
  - cosine EMA and EfficientTeacher EMA ramp disabled because FedSTO states fixed alpha = 0.999
  - runtime server-update configs keep the same fixed EMA settings when SSOD is disabled
  - client broadcast reset sets EMA from the broadcast model weights, not a stale server EMA
  - Phase 1 = backbone-only
  - Phase 2 = full-parameter + non-backbone orthogonal regularization

## Why I Cannot Honestly Mark Bit-for-Bit 100%

The public paper can be reproduced, but the unpublished author code cannot be proven identical.

Blocked items:

1. The official FedSTO training implementation is not published.
2. Exact BDD100K sample IDs used by the authors are not published.
3. Any local EfficientTeacher edits used by the authors are not published.
4. The exact orthogonal regularization coefficient is not specified in the paper.
5. Algorithm 1 samples clients, but the 1-server/3-client BDD experiment does not explicitly publish the sampling ratio; this implementation uses all three clients per round.

The directory is therefore marked as **public-spec complete** rather than **bit-for-bit author-code identical**. The honest answer to "is this 100% FedSTO?" is **no** until the authors publish their trainer and exact split IDs.

## Note On Round Count

The main experimental text states 300 rounds: 50 warmup, 100 Phase 1, and 150 Phase 2. The communication-cost table later discusses 350 rounds with 150 Phase 1 and 150 Phase 2. This package defaults to the main experimental schedule and exposes `--phase1-rounds` so the communication-cost setting can also be reproduced.
