#!/usr/bin/env bash
set -euo pipefail

REPO="/app/Object_Detection"
RUNNER="${REPO}/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/scripts/run_dqa_anonymous_backbone_moe.py"
WORKSPACE="${REPO}/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/05_dqa_expert_choice_upcycled_moe_full"
SEED_WORKSPACE="${REPO}/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/04_canonical_fixed_dqa_upcycled_moe_v2"
CANONICAL_WARMUP="${SEED_WORKSPACE}/global_checkpoints/round000_warmup.pt"
CANONICAL_PHASE1="${SEED_WORKSPACE}/global_checkpoints/phase1_round020_global.pt"

python "${RUNNER}" \
  --workspace-root "${WORKSPACE}" \
  --protocol-suffix expert_choice_st_full_v1 \
  --warmup-checkpoint "${CANONICAL_WARMUP}" \
  --phase1-checkpoint "${CANONICAL_PHASE1}" \
  --warmup-epochs 50 \
  --phase1-rounds 20 \
  --phase2-rounds 20 \
  --moe-start-phase 2 \
  --phase2-train-scope backbone_adapter_moe_head_moe \
  --phase2-late-train-scope backbone_adapter_moe_head \
  --phase2-head-unfreeze-after-round 12 \
  --orthogonal-weight 0.0001 \
  --batch-size 128 \
  --phase2-batch-size 64 \
  --phase2-server-lr0 0.001 \
  --workers 32 \
  --gpus 2 \
  --master-port 29571 \
  --moe-num-experts 4 \
  --moe-top-k 1 \
  --moe-temperature 0.85 \
  --moe-scale 0.25 \
  --moe-shared-scale 1.0 \
  --moe-adapter-ratio 0.125 \
  --moe-levels c3,c4,c5 \
  --moe-kernels 3,5,7 \
  --moe-context-dim 8 \
  --moe-quality-dim 4 \
  --moe-router-noise-std 0.005 \
  --moe-balance-weight 0.02 \
  --moe-entropy-weight 0.0 \
  --moe-sample-entropy-weight 0.006 \
  --moe-z-loss-weight 0.0001 \
  --moe-diversity-weight 0.002 \
  --moe-routing-mode expert_choice \
  --moe-straight-through \
  --moe-expert-choice-capacity-factor 1.25 \
  --moe-dqa-prior-strength 0.20 \
  --moe-router-init-std 0.01 \
  --enable-head-moe \
  --head-moe-start-phase 2 \
  --head-moe-num-experts 4 \
  --head-moe-top-k 1 \
  --head-moe-temperature 0.90 \
  --head-moe-scale 0.25 \
  --head-moe-balance-weight 0.006 \
  --head-moe-entropy-weight 0.0 \
  --head-moe-sample-entropy-weight 0.003 \
  --head-moe-routing-mode expert_choice \
  --head-moe-straight-through \
  --head-moe-expert-choice-capacity-factor 1.10 \
  --head-moe-router-init-std 0.005 \
  --enable-neck-moe \
  --neck-moe-start-phase 2 \
  --neck-moe-num-experts 4 \
  --neck-moe-top-k 1 \
  --neck-moe-temperature 0.90 \
  --neck-moe-scale 0.15 \
  --neck-moe-shared-scale 1.0 \
  --neck-moe-adapter-ratio 0.0625 \
  --neck-moe-levels p3,p4,p5 \
  --neck-moe-kernels 3,5 \
  --neck-moe-router-noise-std 0.005 \
  --neck-moe-balance-weight 0.015 \
  --neck-moe-entropy-weight 0.0 \
  --neck-moe-sample-entropy-weight 0.004 \
  --neck-moe-z-loss-weight 0.0001 \
  --neck-moe-diversity-weight 0.001 \
  --neck-moe-routing-mode expert_choice \
  --neck-moe-straight-through \
  --neck-moe-expert-choice-capacity-factor 1.20 \
  --neck-moe-dqa-prior-strength 0.15 \
  --neck-moe-router-init-std 0.01 \
  --phase2-aggregate-lambda 0.12 \
  --phase2-max-relative-update 0.006 \
  --phase2-max-absolute-update 0.0 \
  --phase2-aggregate-scope adapter_head \
  --router-diagnostic-split cloudy \
  --router-diagnostic-images 16 \
  --router-diagnostic-batch-size 4 \
  --run-final-eval \
  --final-eval-splits cloudy,overcast,rainy,snowy,total \
  --val-batch-size 32 \
  --discord
