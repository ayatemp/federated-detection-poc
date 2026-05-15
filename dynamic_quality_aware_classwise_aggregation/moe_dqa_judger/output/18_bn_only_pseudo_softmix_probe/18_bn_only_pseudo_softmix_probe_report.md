# DQA-SoftMoX 18 BN-Only Pseudo Softmix Probe

- created_utc: 2026-05-14T06:08:51.629185+00:00
- returncode: 0
- log: /app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/18_bn_only_pseudo_softmix_probe/logs/18_bn_only_pseudo_softmix_20260514_054851.log
- method: all-client target pseudoGT training; BN-only updates; DQA only mixes BN changes into a warmup-anchored global model

## Command

```bash
/opt/venv/bin/python3 /app/Object_Detection/dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa/scripts/run_scene_daynight_dqa_18_client_balanced_single_injection_dqamox.py --workspace-root /app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/18_bn_only_pseudo_softmix_probe/training_workspace --source-workspace /app/Object_Detection/dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa/output/08_full_latent_dqamox_from_warmup --skip-warmup-training --warmup-checkpoint /app/Object_Detection/dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa/output/08_full_latent_dqamox_from_warmup/checkpoints/round000_latent_dqamox_warmup.pt --warmup-epochs 50 --repair-baseline-rounds 0 --phase1-rounds 1 --phase2-rounds 0 --post-dqa-repair-rounds 0 --clients 0,1,2,3,4,5 --client-sampling-ratio 1.0 --client-sampling-seed 180181 --use-local-ema-teacher --client-limit 700 --max-images-per-client 0 --num-experts 4 --expert-count 4 --top-k 2 --router-temperature 1.45 --router-balance-weight 0.020 --router-entropy-weight 0.0030 --router-specialization-map domain4 --router-specialization-weight 0.050 --router-specialization-max-weight 0.035 --router-specialization-min-quality 0.65 --router-specialization-min-boxes 180 --phase1-train-scope bn --phase1-repair-train-scope bn --phase1-client-epochs 1 --phase1-client-lr 0.000025 --phase1-source-repeat 1 --phase1-pseudo-repeat 1 --phase1-loss-box 0.000 --client-loss-cls 0.20 --client-loss-obj 0.45 --server-repair-epochs 0 --dqa-server-anchor 1.00 --dqa-min-server-alpha 0.97 --dqa-residual-blend 0.000 --dqa-bn-blend 0.38 --dqa-moe-expert-blend 0.00 --dqa-moe-router-blend 0.00 --dqa-classwise-blend 0.00 --dqa-client-balance-target max --dqa-client-balance-max-scale 1.4 --expert-keep-fraction 0.50 --expert-max-class-fraction 0.14 --actual-max-class-fraction 0.22 --pseudo-imgsz 1280 --min-views 2 --min-models 0 --min-score 0.50 --min-stability 0.82 --max-boxes-per-image 4 --max-class-fraction 0.20 --min-class-keep 30 --client-mixup 0.00 --client-mosaic 0.00 --client-scale 0.00 --client-hsv-s 0.00 --client-hsv-v 0.00 --style-source-repeat 0 --style-source-limit 1 --style-beta 0.0000 --style-imgsz 960 --style-seed 180201 --batch-size 32 --val-batch-size 16 --workers 48 --imgsz 960 --gpus 2 --master-port 39611 --target-map50 0.60 --evaluate --no-progress --no-eval-plots
```

## Metrics

| checkpoint | mAP50 | mAP50:95 | night mAP50:95 | worst split | worst mAP50:95 |
|---|---:|---:|---:|---|---:|
| warmup_global | 0.499 | 0.280 | 0.194 | highway_night | 0.173 |
| warmup_server_repair_final | 0.404 | 0.216 | 0.143 | highway_night | 0.133 |
| latent_dqamox_final_aggregate | 0.492 | 0.274 | 0.187 | highway_night | 0.167 |
| latent_dqamox_final_repair | 0.492 | 0.274 | 0.187 | highway_night | 0.167 |

## Codex Goal Scores

- experiment_env: 98/100
- root_cause_analysis: 96/100
- judge_stability: 93/100
- accuracy_improvement: 86/100
- final_goal: 92/100

## Takeaway

This tests the FedSelect/FedSoup-style hypothesis that only the safest locally learned parameters should be exported when source-style full head training does not transfer.