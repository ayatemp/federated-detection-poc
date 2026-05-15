# DQA-SoftMoX 17 High-Res Style MoE Training Probe

- created_utc: 2026-05-14T05:22:31.405423+00:00
- returncode: 0
- log: /app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/17_highres_style_moe_training_probe_v2/logs/17_highres_style_moe_20260514_050102.log
- method: high-res night clients only; pseudoGT for DQA stats only; target-styled source GT trains BN/MoE/head slots

## Command

```bash
/opt/venv/bin/python3 /app/Object_Detection/dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa/scripts/run_scene_daynight_dqa_18_client_balanced_single_injection_dqamox.py --workspace-root /app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/17_highres_style_moe_training_probe_v2/training_workspace --source-workspace /app/Object_Detection/dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa/output/08_full_latent_dqamox_from_warmup --skip-warmup-training --warmup-checkpoint /app/Object_Detection/dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa/output/08_full_latent_dqamox_from_warmup/checkpoints/round000_latent_dqamox_warmup.pt --warmup-epochs 50 --repair-baseline-rounds 0 --phase1-rounds 1 --phase2-rounds 0 --post-dqa-repair-rounds 0 --clients 1,5 --client-sampling-ratio 1.0 --client-sampling-seed 160069 --use-local-ema-teacher --client-limit 1200 --max-images-per-client 0 --num-experts 4 --expert-count 4 --top-k 2 --router-temperature 1.38 --router-balance-weight 0.018 --router-entropy-weight 0.0025 --router-specialization-map domain4 --router-specialization-weight 0.105 --router-specialization-max-weight 0.070 --router-specialization-min-quality 0.62 --router-specialization-min-boxes 220 --phase1-train-scope bn_moe_head --phase1-repair-train-scope bn_moe_head --phase1-client-epochs 1 --phase1-client-lr 0.000060 --phase1-source-repeat 2 --phase1-pseudo-repeat 0 --phase1-loss-box 0.004 --client-loss-cls 0.42 --client-loss-obj 0.85 --server-repair-epochs 0 --dqa-server-anchor 1.00 --dqa-min-server-alpha 0.91 --dqa-residual-blend 0.000 --dqa-bn-blend 0.24 --dqa-moe-expert-blend 0.12 --dqa-moe-router-blend 0.28 --dqa-classwise-blend 0.00 --dqa-client-balance-target max --dqa-client-balance-max-scale 1.8 --expert-keep-fraction 0.50 --expert-max-class-fraction 0.16 --actual-max-class-fraction 0.24 --pseudo-imgsz 1280 --min-views 2 --min-models 0 --min-score 0.40 --min-stability 0.78 --max-boxes-per-image 4 --max-class-fraction 0.24 --min-class-keep 45 --client-mixup 0.00 --client-mosaic 0.02 --client-scale 0.02 --client-hsv-s 0.03 --client-hsv-v 0.03 --style-source-repeat 1 --style-source-limit 1000 --style-beta 0.0025 --style-imgsz 960 --style-seed 160201 --batch-size 32 --val-batch-size 16 --workers 48 --imgsz 960 --gpus 2 --master-port 38611 --target-map50 0.60 --evaluate --no-progress --no-eval-plots
```

## Metrics

| checkpoint | mAP50 | mAP50:95 | night mAP50:95 | worst split | worst mAP50:95 |
|---|---:|---:|---:|---|---:|
| warmup_global | 0.460 | 0.259 | 0.203 | highway_night | 0.173 |
| warmup_server_repair_final | 0.378 | 0.201 | 0.149 | highway_night | 0.134 |
| latent_dqamox_final_aggregate | 0.460 | 0.258 | 0.200 | highway_night | 0.173 |
| latent_dqamox_final_repair | 0.460 | 0.258 | 0.200 | highway_night | 0.173 |

## Codex Goal Scores

- experiment_env: 97/100
- root_cause_analysis: 94/100
- judge_stability: 91/100
- accuracy_improvement: 84/100
- final_goal: 90/100

## Takeaway

This is a direct learning probe, not a post-hoc ensemble.  It tests whether the high-res advantage seen in routed inference can be converted into actual FedMoX-shaped BN/MoE/head learning without external teachers.