# DQA x MOE Trust Region

This workspace contains a 12-hour-scale DQA x MOE experiment that keeps the
verified FedSTO reproduction as the main training path.

## Core Idea

The previous DQA variants often lost to warmup because DQA replaced the stable
FedSTO update too aggressively.  This version instead treats DQA x MOE as a
small training-time residual controller:

1. Run the same FedSTO schedule: warmup, phase1, phase2.
2. After each phase2 server repair, treat the client checkpoints as
   domain experts.
3. Build a lightweight DQA router from each client's own validation metrics.
4. Build several small candidates that vary the module scope and residual
   strength.
5. Evaluate the candidates on source/cloudy validation.
6. Select the best accepted candidate with a trust-region proxy score.
7. If every candidate regresses beyond the tolerance, broadcast the FedSTO
   server-repaired checkpoint instead.

This means DQA participates in the next round's training, but it cannot easily
destroy the FedSTO trajectory.

## Default 12h Setting

The default command is intentionally close to the successful FedSTO run:

- warmup: 50 epochs
- phase1: 20 rounds
- phase2: 20 rounds
- local epoch: 1
- server repair epoch: 1
- clients: FedSTO paper BDD split clients, all participating each round
- DQA starts at phase2 round 6
- DQA candidate search: `head`, `head_bn`, `neck_head`
- DQA residual multipliers: `0.75`, `1.00`
- final paper-style eval: cloudy, overcast, rainy, snowy, total

## Run

```bash
python dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/scripts/run_dqa_moe_trust_region.py \
  --workspace-root /app/Object_Detection/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/01_fedsto_dqa_moe_trust_region_12h \
  --warmup-epochs 50 \
  --phase1-rounds 20 \
  --phase2-rounds 20 \
  --batch-size 128 \
  --workers 32 \
  --gpus 2 \
  --master-port 29541 \
  --dqa-start-round 6 \
  --dqa-scope head_bn \
  --dqa-search-candidates \
  --dqa-candidate-scopes head,head_bn,neck_head \
  --dqa-candidate-lambda-multipliers 0.75,1.00 \
  --dqa-max-candidates 6 \
  --run-final-eval
```

Add `--discord` if `DISCORD_WEBHOOK_URL` is configured.

The runner refuses CPU-only execution by default.  The 12h setting requires
visible CUDA GPUs; use `--allow-cpu` only for tiny debugging runs.

## Shared-Routed Head MoE Variant

`notebooks/02_shared_routed_head_moe_12h.ipynb` enables the stronger variant
designed after the first 12h result:

- the server-repaired checkpoint is the shared expert;
- client checkpoints are routed weather/domain experts;
- each client writes pseudo-label class/quality stats from its own local EMA;
- YOLO final head channels are routed class-wise, while box/objectness channels
  use the global expert mixture;
- DQA searches `balanced`, `rainy`, and `snowy` routing focuses across
  `head` and `head_bn` candidates;
- source/cloudy trust-region acceptance is still the safety gate.

The corresponding command is:

```bash
python dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/scripts/run_dqa_moe_trust_region.py \
  --workspace-root /app/Object_Detection/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/02_shared_routed_head_moe_12h \
  --warmup-epochs 50 \
  --phase1-rounds 20 \
  --phase2-rounds 20 \
  --batch-size 128 \
  --workers 32 \
  --gpus 2 \
  --master-port 29543 \
  --dqa-start-round 6 \
  --dqa-router-mode shared_routed \
  --dqa-router-candidates balanced,rainy,snowy \
  --dqa-candidate-scopes head,head_bn \
  --dqa-candidate-lambda-multipliers 0.60 \
  --dqa-max-candidates 6 \
  --dqa-lambda-start 0.012 \
  --dqa-lambda-end 0.045 \
  --dqa-max-relative-update 0.008 \
  --dqa-router-proxy-weight 0.001 \
  --dqa-collect-pseudo-stats \
  --dqa-pseudo-quality-mode feature_balanced \
  --run-final-eval \
  --discord
```

## Outputs

- `history.json`: FedSTO-compatible global checkpoint history.
- `dqa_moe_history.json`: per-round DQA candidate, router, and acceptance log.
- `dqa_moe_round_summary.csv`: compact table for quick plotting.
- `validation_reports/paper_protocol_eval_summary.csv`: final paper-style eval.

## Why This Is Stronger Than The First Version

The first implementation had one fixed DQA candidate per round.  This version
turns DQA into a small online model-selection problem: each scheduled phase2
round tries multiple module-wise MOE residuals, scores them, and only broadcasts
the best safe candidate.  The dynamic part is now both "how much to mix" and
"where to mix" rather than a fixed hand-tuned path.
