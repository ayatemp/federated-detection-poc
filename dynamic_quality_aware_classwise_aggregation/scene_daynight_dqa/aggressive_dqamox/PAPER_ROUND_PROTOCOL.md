# Paper-Round DQA-MoX Protocol

This note records the round-count choices for the next DQA-MoX continuation.

## Primary References

- FedMoX / PSSFL: arXiv:2508.16568, "Closer to Reality: Practical Semi-Supervised Federated Learning for Foundation Model Adaptation"
- FedSTO / SSFOD: arXiv:2310.17097, "Navigating Data Heterogeneity in Federated Learning: A Semi-Supervised Federated Object Detection"

## Round Design

FedMoX reports:

- warm-up training: 50 epochs
- federated rounds: 50
- client sampling: 33% clients online per round
- local training: 1 server epoch and 1 client epoch per round

FedSTO reports:

- warm-up: 50 rounds
- Phase 1 selective training: 100 rounds in the main setting
- Phase 2 full-parameter / orthogonal training: 150 rounds in the main setting
- local training: 1 server epoch and 1 client epoch per round
- the 100-client bandwidth table also discusses 150 Phase 1 + 150 Phase 2 communication rounds

## Mapping To The Current DQA-MoX Code

The immediate executable protocol uses the FedMoX total FL length and the FedSTO phase ratio:

- warmup: 50 epochs
- total DQA-MoX FL rounds: 50
- Phase 1: 20 rounds, roughly the FedSTO 100/(100+150) ratio
- Phase 2: 30 rounds, roughly the FedSTO 150/(100+150) ratio
- client sampling: 0.333, so the 6-client scene/day-night setup trains 2 clients per round
- local epochs: 1 per client/server round

This is intentionally not the short diagnostic 2/1-round setting.  It is the first practical full-run bridge between FedMoX's 50-round PSSFL protocol and FedSTO's two-stage SSFOD protocol.

## Current Hypothesis

Short DQA-MoX runs often show that pseudoGT can inject useful client/domain signal, but repeated full pseudo-box training drifts.  The new run tests whether paper-style communication rounds solve the missing ingredient:

- FedMoX-style sparse top-1 latent MoE head for client/domain specialization.
- FedSTO-style long selective Phase 1 before full updates.
- Round-level client sampling, so aggregation sees a stream of heterogeneous clients instead of the same full client set every round.
- Soft server/client mixture through DQA anchors, used as the FedMoX-style stabilization mechanism rather than as GT-selected checkpoint retention.
