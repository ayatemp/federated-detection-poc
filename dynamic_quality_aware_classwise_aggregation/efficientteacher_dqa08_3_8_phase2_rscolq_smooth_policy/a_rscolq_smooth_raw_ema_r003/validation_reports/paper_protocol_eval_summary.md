# Paper Protocol Evaluation Summary

Created UTC: 2026-05-04T10:14:44.250713+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_8_phase2_rscolq_smooth_policy/a_rscolq_smooth_raw_ema_r003`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_8_phase2_rscolq_smooth_policy/a_rscolq_smooth_raw_ema_r003/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| highway | union | 2499 | 36377 |
| citystreet | union | 6112 | 127178 |
| residential | union | 1253 | 20855 |
| scene_total | union | 9864 | 0 |

## Checkpoints

- `p1_r003`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_scene_tri_stage_policy_8h/global_checkpoints/phase1_round003_global.pt`
- `p1_r012`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_scene_tri_stage_policy_8h/global_checkpoints/phase1_round012_global.pt`
- `old08_3_4_c_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_4_phase2_feature_quality_sweep/c_feature_balanced_neck_head_r003/global_checkpoints/phase2_round002_global.pt`
- `old08_3_4_d_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_4_phase2_feature_quality_sweep/d_feature_conservative_min_gate_r003/global_checkpoints/phase2_round002_global.pt`
- `old08_3_6_a_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_6_phase2_scolq_policy/a_scolq_soft_bbox_r003/global_checkpoints/phase2_round001_global.pt`
- `old08_3_6_a_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_6_phase2_scolq_policy/a_scolq_soft_bbox_r003/global_checkpoints/phase2_round002_global.pt`
- `old08_3_6_a_r010`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_6_phase2_scolq_policy/a_scolq_soft_bbox_r003/global_checkpoints/phase2_round010_global.pt`
- `old08_3_7_a_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_7_phase2_rscolq_policy/a_rscolq_round_stable_r003/global_checkpoints/phase2_round001_global.pt`
- `old08_3_7_a_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_7_phase2_rscolq_policy/a_rscolq_round_stable_r003/global_checkpoints/phase2_round002_global.pt`
- `old08_3_7_a_r010`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_7_phase2_rscolq_policy/a_rscolq_round_stable_r003/global_checkpoints/phase2_round010_global.pt`
- `a_rscolq_smooth_raw_ema_r003_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_8_phase2_rscolq_smooth_policy/a_rscolq_smooth_raw_ema_r003/global_checkpoints/phase2_round001_global.pt`
- `a_rscolq_smooth_raw_ema_r003_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_8_phase2_rscolq_smooth_policy/a_rscolq_smooth_raw_ema_r003/global_checkpoints/phase2_round002_global.pt`
- `a_rscolq_smooth_raw_ema_r003_r010`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_8_phase2_rscolq_smooth_policy/a_rscolq_smooth_raw_ema_r003/global_checkpoints/phase2_round010_global.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| p1_r003 | highway | 0.533 | 0.329 | 0.326 | 0.182 | ok |
| p1_r003 | citystreet | 0.472 | 0.409 | 0.381 | 0.207 | ok |
| p1_r003 | residential | 0.725 | 0.36 | 0.414 | 0.232 | ok |
| p1_r003 | scene_total | 0.471 | 0.401 | 0.375 | 0.204 | ok |
| p1_r012 | highway | 0.505 | 0.34 | 0.32 | 0.175 | ok |
| p1_r012 | citystreet | 0.745 | 0.332 | 0.374 | 0.202 | ok |
| p1_r012 | residential | 0.369 | 0.473 | 0.407 | 0.225 | ok |
| p1_r012 | scene_total | 0.743 | 0.326 | 0.368 | 0.199 | ok |
| old08_3_4_c_r002 | highway | 0.477 | 0.368 | 0.329 | 0.187 | ok |
| old08_3_4_c_r002 | citystreet | 0.493 | 0.414 | 0.386 | 0.216 | ok |
| old08_3_4_c_r002 | residential | 0.532 | 0.415 | 0.419 | 0.237 | ok |
| old08_3_4_c_r002 | scene_total | 0.492 | 0.408 | 0.38 | 0.212 | ok |
| old08_3_4_d_r002 | highway | 0.465 | 0.375 | 0.332 | 0.187 | ok |
| old08_3_4_d_r002 | citystreet | 0.488 | 0.418 | 0.387 | 0.216 | ok |
| old08_3_4_d_r002 | residential | 0.508 | 0.419 | 0.42 | 0.238 | ok |
| old08_3_4_d_r002 | scene_total | 0.503 | 0.399 | 0.381 | 0.213 | ok |
| old08_3_6_a_r001 | highway | 0.461 | 0.376 | 0.331 | 0.187 | ok |
| old08_3_6_a_r001 | citystreet | 0.484 | 0.414 | 0.387 | 0.215 | ok |
| old08_3_6_a_r001 | residential | 0.44 | 0.443 | 0.424 | 0.239 | ok |
| old08_3_6_a_r001 | scene_total | 0.493 | 0.401 | 0.382 | 0.211 | ok |
| old08_3_6_a_r002 | highway | 0.478 | 0.368 | 0.333 | 0.188 | ok |
| old08_3_6_a_r002 | citystreet | 0.488 | 0.418 | 0.387 | 0.216 | ok |
| old08_3_6_a_r002 | residential | 0.539 | 0.412 | 0.418 | 0.237 | ok |
| old08_3_6_a_r002 | scene_total | 0.489 | 0.411 | 0.381 | 0.212 | ok |
| old08_3_6_a_r010 | highway | 0.479 | 0.343 | 0.312 | 0.171 | ok |
| old08_3_6_a_r010 | citystreet | 0.479 | 0.399 | 0.364 | 0.199 | ok |
| old08_3_6_a_r010 | residential | 0.439 | 0.418 | 0.387 | 0.218 | ok |
| old08_3_6_a_r010 | scene_total | 0.483 | 0.389 | 0.357 | 0.196 | ok |
| old08_3_7_a_r001 | highway | 0.46 | 0.376 | 0.331 | 0.187 | ok |
| old08_3_7_a_r001 | citystreet | 0.485 | 0.413 | 0.387 | 0.215 | ok |
| old08_3_7_a_r001 | residential | 0.442 | 0.442 | 0.423 | 0.239 | ok |
| old08_3_7_a_r001 | scene_total | 0.485 | 0.406 | 0.381 | 0.212 | ok |
| old08_3_7_a_r002 | highway | 0.475 | 0.371 | 0.334 | 0.188 | ok |
| old08_3_7_a_r002 | citystreet | 0.493 | 0.413 | 0.387 | 0.216 | ok |
| old08_3_7_a_r002 | residential | 0.534 | 0.413 | 0.418 | 0.236 | ok |
| old08_3_7_a_r002 | scene_total | 0.494 | 0.406 | 0.381 | 0.212 | ok |
| old08_3_7_a_r010 | highway | 0.465 | 0.36 | 0.314 | 0.172 | ok |
| old08_3_7_a_r010 | citystreet | 0.479 | 0.403 | 0.363 | 0.2 | ok |
| old08_3_7_a_r010 | residential | 0.431 | 0.418 | 0.387 | 0.219 | ok |
| old08_3_7_a_r010 | scene_total | 0.479 | 0.395 | 0.357 | 0.196 | ok |
| a_rscolq_smooth_raw_ema_r003_r001 | highway | 0.464 | 0.379 | 0.336 | 0.188 | ok |
| a_rscolq_smooth_raw_ema_r003_r001 | citystreet | 0.488 | 0.411 | 0.386 | 0.215 | ok |
| a_rscolq_smooth_raw_ema_r003_r001 | residential | 0.418 | 0.468 | 0.422 | 0.238 | ok |
| a_rscolq_smooth_raw_ema_r003_r001 | scene_total | 0.489 | 0.404 | 0.381 | 0.211 | ok |
| a_rscolq_smooth_raw_ema_r003_r002 | highway | 0.472 | 0.369 | 0.332 | 0.188 | ok |
| a_rscolq_smooth_raw_ema_r003_r002 | citystreet | 0.506 | 0.402 | 0.386 | 0.216 | ok |
| a_rscolq_smooth_raw_ema_r003_r002 | residential | 0.518 | 0.417 | 0.418 | 0.238 | ok |
| a_rscolq_smooth_raw_ema_r003_r002 | scene_total | 0.507 | 0.395 | 0.38 | 0.213 | ok |
| a_rscolq_smooth_raw_ema_r003_r010 | highway | 0.505 | 0.352 | 0.319 | 0.175 | ok |
| a_rscolq_smooth_raw_ema_r003_r010 | citystreet | 0.516 | 0.38 | 0.367 | 0.201 | ok |
| a_rscolq_smooth_raw_ema_r003_r010 | residential | 0.422 | 0.434 | 0.391 | 0.219 | ok |
| a_rscolq_smooth_raw_ema_r003_r010 | scene_total | 0.48 | 0.398 | 0.361 | 0.198 | ok |
