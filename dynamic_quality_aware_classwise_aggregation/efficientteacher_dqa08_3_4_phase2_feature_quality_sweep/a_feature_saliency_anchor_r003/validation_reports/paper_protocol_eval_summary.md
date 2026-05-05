# Paper Protocol Evaluation Summary

Created UTC: 2026-05-03T14:24:15.286626+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_4_phase2_feature_quality_sweep/a_feature_saliency_anchor_r003`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_4_phase2_feature_quality_sweep/a_feature_saliency_anchor_r003/validation_reports`

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
- `old08_3_e_r02`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_phase2_update_gating_sweep/e_ug_best_phase1_r003/global_checkpoints/phase2_round002_global.pt`
- `old08_3_2_c_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/c_dqa_high_light_head_r003/global_checkpoints/phase2_round001_global.pt`
- `old08_3_2_c_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/c_dqa_high_light_head_r003/global_checkpoints/phase2_round002_global.pt`
- `old08_3_3_a_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/a_c_impulse_decay_r003/global_checkpoints/phase2_round001_global.pt`
- `old08_3_3_a_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/a_c_impulse_decay_r003/global_checkpoints/phase2_round002_global.pt`
- `old08_3_2_c_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/c_dqa_high_light_head_r003/global_checkpoints/phase2_round001_global.pt`
- `old08_3_2_c_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/c_dqa_high_light_head_r003/global_checkpoints/phase2_round002_global.pt`
- `old08_3_3_a_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/a_c_impulse_decay_r003/global_checkpoints/phase2_round001_global.pt`
- `old08_3_3_a_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/a_c_impulse_decay_r003/global_checkpoints/phase2_round002_global.pt`
- `a_feature_saliency_anchor_r003_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_4_phase2_feature_quality_sweep/a_feature_saliency_anchor_r003/global_checkpoints/phase2_round001_global.pt`
- `a_feature_saliency_anchor_r003_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_4_phase2_feature_quality_sweep/a_feature_saliency_anchor_r003/global_checkpoints/phase2_round002_global.pt`
- `a_feature_saliency_anchor_r003_r010`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_4_phase2_feature_quality_sweep/a_feature_saliency_anchor_r003/global_checkpoints/phase2_round010_global.pt`
- `b_feature_contrast_strict_cap_r003_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_4_phase2_feature_quality_sweep/b_feature_contrast_strict_cap_r003/global_checkpoints/phase2_round001_global.pt`
- `b_feature_contrast_strict_cap_r003_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_4_phase2_feature_quality_sweep/b_feature_contrast_strict_cap_r003/global_checkpoints/phase2_round002_global.pt`
- `b_feature_contrast_strict_cap_r003_r010`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_4_phase2_feature_quality_sweep/b_feature_contrast_strict_cap_r003/global_checkpoints/phase2_round010_global.pt`
- `c_feature_balanced_neck_head_r003_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_4_phase2_feature_quality_sweep/c_feature_balanced_neck_head_r003/global_checkpoints/phase2_round001_global.pt`
- `c_feature_balanced_neck_head_r003_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_4_phase2_feature_quality_sweep/c_feature_balanced_neck_head_r003/global_checkpoints/phase2_round002_global.pt`
- `c_feature_balanced_neck_head_r003_r010`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_4_phase2_feature_quality_sweep/c_feature_balanced_neck_head_r003/global_checkpoints/phase2_round010_global.pt`
- `d_feature_conservative_min_gate_r003_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_4_phase2_feature_quality_sweep/d_feature_conservative_min_gate_r003/global_checkpoints/phase2_round001_global.pt`
- `d_feature_conservative_min_gate_r003_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_4_phase2_feature_quality_sweep/d_feature_conservative_min_gate_r003/global_checkpoints/phase2_round002_global.pt`
- `d_feature_conservative_min_gate_r003_r010`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_4_phase2_feature_quality_sweep/d_feature_conservative_min_gate_r003/global_checkpoints/phase2_round010_global.pt`
- `e_feature_no_conf_recall_repair_r003_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_4_phase2_feature_quality_sweep/e_feature_no_conf_recall_repair_r003/global_checkpoints/phase2_round001_global.pt`
- `e_feature_no_conf_recall_repair_r003_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_4_phase2_feature_quality_sweep/e_feature_no_conf_recall_repair_r003/global_checkpoints/phase2_round002_global.pt`
- `e_feature_no_conf_recall_repair_r003_r010`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_4_phase2_feature_quality_sweep/e_feature_no_conf_recall_repair_r003/global_checkpoints/phase2_round010_global.pt`

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
| old08_3_e_r02 | highway | 0.54 | 0.328 | 0.328 | 0.186 | ok |
| old08_3_e_r02 | citystreet | 0.469 | 0.422 | 0.386 | 0.215 | ok |
| old08_3_e_r02 | residential | 0.426 | 0.444 | 0.413 | 0.237 | ok |
| old08_3_e_r02 | scene_total | 0.469 | 0.414 | 0.379 | 0.211 | ok |
| old08_3_2_c_r001 | highway | 0.459 | 0.377 | 0.331 | 0.187 | ok |
| old08_3_2_c_r001 | citystreet | 0.486 | 0.413 | 0.387 | 0.215 | ok |
| old08_3_2_c_r001 | residential | 0.441 | 0.444 | 0.422 | 0.239 | ok |
| old08_3_2_c_r001 | scene_total | 0.487 | 0.405 | 0.381 | 0.212 | ok |
| old08_3_2_c_r002 | highway | 0.474 | 0.368 | 0.329 | 0.187 | ok |
| old08_3_2_c_r002 | citystreet | 0.489 | 0.416 | 0.385 | 0.215 | ok |
| old08_3_2_c_r002 | residential | 0.53 | 0.413 | 0.419 | 0.237 | ok |
| old08_3_2_c_r002 | scene_total | 0.489 | 0.409 | 0.38 | 0.212 | ok |
| old08_3_3_a_r001 | highway | 0.46 | 0.377 | 0.331 | 0.187 | ok |
| old08_3_3_a_r001 | citystreet | 0.487 | 0.412 | 0.387 | 0.215 | ok |
| old08_3_3_a_r001 | residential | 0.442 | 0.443 | 0.422 | 0.238 | ok |
| old08_3_3_a_r001 | scene_total | 0.487 | 0.405 | 0.381 | 0.212 | ok |
| old08_3_3_a_r002 | highway | 0.475 | 0.371 | 0.33 | 0.187 | ok |
| old08_3_3_a_r002 | citystreet | 0.492 | 0.414 | 0.386 | 0.215 | ok |
| old08_3_3_a_r002 | residential | 0.531 | 0.415 | 0.419 | 0.237 | ok |
| old08_3_3_a_r002 | scene_total | 0.493 | 0.406 | 0.38 | 0.212 | ok |
| old08_3_2_c_r001 | highway | 0.459 | 0.377 | 0.331 | 0.187 | ok |
| old08_3_2_c_r001 | citystreet | 0.486 | 0.413 | 0.387 | 0.215 | ok |
| old08_3_2_c_r001 | residential | 0.441 | 0.444 | 0.422 | 0.239 | ok |
| old08_3_2_c_r001 | scene_total | 0.487 | 0.405 | 0.381 | 0.212 | ok |
| old08_3_2_c_r002 | highway | 0.474 | 0.368 | 0.329 | 0.187 | ok |
| old08_3_2_c_r002 | citystreet | 0.489 | 0.416 | 0.385 | 0.215 | ok |
| old08_3_2_c_r002 | residential | 0.53 | 0.413 | 0.419 | 0.237 | ok |
| old08_3_2_c_r002 | scene_total | 0.489 | 0.409 | 0.38 | 0.212 | ok |
| old08_3_3_a_r001 | highway | 0.46 | 0.377 | 0.331 | 0.187 | ok |
| old08_3_3_a_r001 | citystreet | 0.487 | 0.412 | 0.387 | 0.215 | ok |
| old08_3_3_a_r001 | residential | 0.442 | 0.443 | 0.422 | 0.238 | ok |
| old08_3_3_a_r001 | scene_total | 0.487 | 0.405 | 0.381 | 0.212 | ok |
| old08_3_3_a_r002 | highway | 0.475 | 0.371 | 0.33 | 0.187 | ok |
| old08_3_3_a_r002 | citystreet | 0.492 | 0.414 | 0.386 | 0.215 | ok |
| old08_3_3_a_r002 | residential | 0.531 | 0.415 | 0.419 | 0.237 | ok |
| old08_3_3_a_r002 | scene_total | 0.493 | 0.406 | 0.38 | 0.212 | ok |
| a_feature_saliency_anchor_r003_r001 | highway | 0.46 | 0.376 | 0.331 | 0.187 | ok |
| a_feature_saliency_anchor_r003_r001 | citystreet | 0.487 | 0.412 | 0.387 | 0.215 | ok |
| a_feature_saliency_anchor_r003_r001 | residential | 0.441 | 0.443 | 0.422 | 0.239 | ok |
| a_feature_saliency_anchor_r003_r001 | scene_total | 0.487 | 0.405 | 0.381 | 0.211 | ok |
| a_feature_saliency_anchor_r003_r002 | highway | 0.478 | 0.368 | 0.334 | 0.188 | ok |
| a_feature_saliency_anchor_r003_r002 | citystreet | 0.527 | 0.39 | 0.387 | 0.216 | ok |
| a_feature_saliency_anchor_r003_r002 | residential | 0.534 | 0.413 | 0.418 | 0.236 | ok |
| a_feature_saliency_anchor_r003_r002 | scene_total | 0.489 | 0.41 | 0.381 | 0.212 | ok |
| a_feature_saliency_anchor_r003_r010 | highway | 0.478 | 0.346 | 0.314 | 0.171 | ok |
| a_feature_saliency_anchor_r003_r010 | citystreet | 0.494 | 0.388 | 0.363 | 0.199 | ok |
| a_feature_saliency_anchor_r003_r010 | residential | 0.429 | 0.423 | 0.389 | 0.218 | ok |
| a_feature_saliency_anchor_r003_r010 | scene_total | 0.494 | 0.381 | 0.357 | 0.195 | ok |
| b_feature_contrast_strict_cap_r003_r001 | highway | 0.46 | 0.376 | 0.331 | 0.188 | ok |
| b_feature_contrast_strict_cap_r003_r001 | citystreet | 0.487 | 0.412 | 0.387 | 0.215 | ok |
| b_feature_contrast_strict_cap_r003_r001 | residential | 0.444 | 0.443 | 0.423 | 0.239 | ok |
| b_feature_contrast_strict_cap_r003_r001 | scene_total | 0.487 | 0.405 | 0.381 | 0.212 | ok |
| b_feature_contrast_strict_cap_r003_r002 | highway | 0.481 | 0.37 | 0.333 | 0.188 | ok |
| b_feature_contrast_strict_cap_r003_r002 | citystreet | 0.491 | 0.416 | 0.386 | 0.215 | ok |
| b_feature_contrast_strict_cap_r003_r002 | residential | 0.532 | 0.415 | 0.419 | 0.237 | ok |
| b_feature_contrast_strict_cap_r003_r002 | scene_total | 0.491 | 0.409 | 0.38 | 0.212 | ok |
| b_feature_contrast_strict_cap_r003_r010 | highway | 0.472 | 0.357 | 0.315 | 0.173 | ok |
| b_feature_contrast_strict_cap_r003_r010 | citystreet | 0.49 | 0.396 | 0.366 | 0.201 | ok |
| b_feature_contrast_strict_cap_r003_r010 | residential | 0.426 | 0.425 | 0.388 | 0.218 | ok |
| b_feature_contrast_strict_cap_r003_r010 | scene_total | 0.491 | 0.388 | 0.36 | 0.197 | ok |
| c_feature_balanced_neck_head_r003_r001 | highway | 0.46 | 0.376 | 0.331 | 0.187 | ok |
| c_feature_balanced_neck_head_r003_r001 | citystreet | 0.488 | 0.412 | 0.387 | 0.215 | ok |
| c_feature_balanced_neck_head_r003_r001 | residential | 0.442 | 0.444 | 0.423 | 0.239 | ok |
| c_feature_balanced_neck_head_r003_r001 | scene_total | 0.488 | 0.405 | 0.382 | 0.212 | ok |
| c_feature_balanced_neck_head_r003_r002 | highway | 0.477 | 0.368 | 0.329 | 0.187 | ok |
| c_feature_balanced_neck_head_r003_r002 | citystreet | 0.493 | 0.414 | 0.386 | 0.216 | ok |
| c_feature_balanced_neck_head_r003_r002 | residential | 0.532 | 0.415 | 0.419 | 0.237 | ok |
| c_feature_balanced_neck_head_r003_r002 | scene_total | 0.492 | 0.408 | 0.38 | 0.212 | ok |
| c_feature_balanced_neck_head_r003_r010 | highway | 0.478 | 0.345 | 0.312 | 0.171 | ok |
| c_feature_balanced_neck_head_r003_r010 | citystreet | 0.484 | 0.395 | 0.363 | 0.199 | ok |
| c_feature_balanced_neck_head_r003_r010 | residential | 0.42 | 0.428 | 0.388 | 0.217 | ok |
| c_feature_balanced_neck_head_r003_r010 | scene_total | 0.48 | 0.392 | 0.357 | 0.196 | ok |
| d_feature_conservative_min_gate_r003_r001 | highway | 0.464 | 0.379 | 0.335 | 0.187 | ok |
| d_feature_conservative_min_gate_r003_r001 | citystreet | 0.49 | 0.41 | 0.386 | 0.214 | ok |
| d_feature_conservative_min_gate_r003_r001 | residential | 0.444 | 0.443 | 0.419 | 0.237 | ok |
| d_feature_conservative_min_gate_r003_r001 | scene_total | 0.49 | 0.403 | 0.381 | 0.211 | ok |
| d_feature_conservative_min_gate_r003_r002 | highway | 0.465 | 0.375 | 0.332 | 0.187 | ok |
| d_feature_conservative_min_gate_r003_r002 | citystreet | 0.488 | 0.418 | 0.387 | 0.216 | ok |
| d_feature_conservative_min_gate_r003_r002 | residential | 0.508 | 0.419 | 0.42 | 0.238 | ok |
| d_feature_conservative_min_gate_r003_r002 | scene_total | 0.503 | 0.399 | 0.381 | 0.213 | ok |
| d_feature_conservative_min_gate_r003_r010 | highway | 0.498 | 0.346 | 0.318 | 0.174 | ok |
| d_feature_conservative_min_gate_r003_r010 | citystreet | 0.483 | 0.397 | 0.365 | 0.2 | ok |
| d_feature_conservative_min_gate_r003_r010 | residential | 0.508 | 0.389 | 0.395 | 0.22 | ok |
| d_feature_conservative_min_gate_r003_r010 | scene_total | 0.485 | 0.389 | 0.359 | 0.197 | ok |
| e_feature_no_conf_recall_repair_r003_r001 | highway | 0.465 | 0.377 | 0.334 | 0.187 | ok |
| e_feature_no_conf_recall_repair_r003_r001 | citystreet | 0.488 | 0.411 | 0.386 | 0.214 | ok |
| e_feature_no_conf_recall_repair_r003_r001 | residential | 0.42 | 0.466 | 0.421 | 0.237 | ok |
| e_feature_no_conf_recall_repair_r003_r001 | scene_total | 0.488 | 0.405 | 0.381 | 0.211 | ok |
| e_feature_no_conf_recall_repair_r003_r002 | highway | 0.468 | 0.367 | 0.328 | 0.187 | ok |
| e_feature_no_conf_recall_repair_r003_r002 | citystreet | 0.489 | 0.417 | 0.386 | 0.216 | ok |
| e_feature_no_conf_recall_repair_r003_r002 | residential | 0.548 | 0.407 | 0.418 | 0.238 | ok |
| e_feature_no_conf_recall_repair_r003_r002 | scene_total | 0.489 | 0.409 | 0.38 | 0.212 | ok |
| e_feature_no_conf_recall_repair_r003_r010 | highway | 0.47 | 0.355 | 0.315 | 0.172 | ok |
| e_feature_no_conf_recall_repair_r003_r010 | citystreet | 0.481 | 0.399 | 0.364 | 0.2 | ok |
| e_feature_no_conf_recall_repair_r003_r010 | residential | 0.467 | 0.397 | 0.389 | 0.218 | ok |
| e_feature_no_conf_recall_repair_r003_r010 | scene_total | 0.481 | 0.391 | 0.358 | 0.197 | ok |
