# Paper Protocol Evaluation Summary

Created UTC: 2026-05-03T04:37:43.680275+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/a_c_impulse_decay_r003`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/a_c_impulse_decay_r003/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| scene_total | union | 9864 | 0 |

## Checkpoints

- `p1_r003`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_scene_tri_stage_policy_8h/global_checkpoints/phase1_round003_global.pt`
- `p1_r012`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_scene_tri_stage_policy_8h/global_checkpoints/phase1_round012_global.pt`
- `old08_3_e_r02`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_phase2_update_gating_sweep/e_ug_best_phase1_r003/global_checkpoints/phase2_round002_global.pt`
- `old08_3_2_c_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/c_dqa_high_light_head_r003/global_checkpoints/phase2_round001_global.pt`
- `old08_3_2_c_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/c_dqa_high_light_head_r003/global_checkpoints/phase2_round002_global.pt`
- `a_c_impulse_decay_r003_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/a_c_impulse_decay_r003/global_checkpoints/phase2_round001_global.pt`
- `a_c_impulse_decay_r003_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/a_c_impulse_decay_r003/global_checkpoints/phase2_round002_global.pt`
- `a_c_impulse_decay_r003_r003`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/a_c_impulse_decay_r003/global_checkpoints/phase2_round003_global.pt`
- `a_c_impulse_decay_r003_r010`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/a_c_impulse_decay_r003/global_checkpoints/phase2_round010_global.pt`
- `b_strict_cap_decay_r003_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/b_strict_cap_decay_r003/global_checkpoints/phase2_round001_global.pt`
- `b_strict_cap_decay_r003_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/b_strict_cap_decay_r003/global_checkpoints/phase2_round002_global.pt`
- `b_strict_cap_decay_r003_r003`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/b_strict_cap_decay_r003/global_checkpoints/phase2_round003_global.pt`
- `b_strict_cap_decay_r003_r010`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/b_strict_cap_decay_r003/global_checkpoints/phase2_round010_global.pt`
- `c_neck_head_safe_r003_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/c_neck_head_safe_r003/global_checkpoints/phase2_round001_global.pt`
- `c_neck_head_safe_r003_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/c_neck_head_safe_r003/global_checkpoints/phase2_round002_global.pt`
- `c_neck_head_safe_r003_r003`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/c_neck_head_safe_r003/global_checkpoints/phase2_round003_global.pt`
- `c_neck_head_safe_r003_r010`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/c_neck_head_safe_r003/global_checkpoints/phase2_round010_global.pt`
- `d_non_backbone_recall_repair_r003_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/d_non_backbone_recall_repair_r003/global_checkpoints/phase2_round001_global.pt`
- `d_non_backbone_recall_repair_r003_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/d_non_backbone_recall_repair_r003/global_checkpoints/phase2_round002_global.pt`
- `d_non_backbone_recall_repair_r003_r003`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/d_non_backbone_recall_repair_r003/global_checkpoints/phase2_round003_global.pt`
- `d_non_backbone_recall_repair_r003_r010`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/d_non_backbone_recall_repair_r003/global_checkpoints/phase2_round010_global.pt`
- `e_high_precision_seed_r012_recall_repair_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/e_high_precision_seed_r012_recall_repair/global_checkpoints/phase2_round001_global.pt`
- `e_high_precision_seed_r012_recall_repair_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_3_phase2_anti_drift_sweep/e_high_precision_seed_r012_recall_repair/global_checkpoints/phase2_round002_global.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| p1_r003 | scene_total | 0.471 | 0.401 | 0.375 | 0.204 | ok |
| p1_r012 | scene_total | 0.743 | 0.326 | 0.368 | 0.199 | ok |
| old08_3_e_r02 | scene_total | 0.469 | 0.414 | 0.379 | 0.211 | ok |
| old08_3_2_c_r001 | scene_total | 0.487 | 0.405 | 0.381 | 0.212 | ok |
| old08_3_2_c_r002 | scene_total | 0.489 | 0.409 | 0.38 | 0.212 | ok |
| a_c_impulse_decay_r003_r001 | scene_total | 0.487 | 0.405 | 0.381 | 0.212 | ok |
| a_c_impulse_decay_r003_r002 | scene_total | 0.493 | 0.406 | 0.38 | 0.212 | ok |
| a_c_impulse_decay_r003_r003 | scene_total | 0.503 | 0.398 | 0.376 | 0.211 | ok |
| a_c_impulse_decay_r003_r010 | scene_total | 0.481 | 0.394 | 0.359 | 0.196 | ok |
| b_strict_cap_decay_r003_r001 | scene_total | 0.493 | 0.4 | 0.38 | 0.211 | ok |
| b_strict_cap_decay_r003_r002 | scene_total | 0.49 | 0.408 | 0.38 | 0.212 | ok |
| b_strict_cap_decay_r003_r003 | scene_total | 0.525 | 0.383 | 0.376 | 0.21 | ok |
| b_strict_cap_decay_r003_r010 | scene_total | 0.474 | 0.399 | 0.358 | 0.196 | ok |
| c_neck_head_safe_r003_r001 | scene_total | 0.487 | 0.406 | 0.381 | 0.212 | ok |
| c_neck_head_safe_r003_r002 | scene_total | 0.494 | 0.406 | 0.38 | 0.212 | ok |
| c_neck_head_safe_r003_r003 | scene_total | 0.494 | 0.404 | 0.376 | 0.211 | ok |
| c_neck_head_safe_r003_r010 | scene_total | 0.487 | 0.387 | 0.358 | 0.196 | ok |
| d_non_backbone_recall_repair_r003_r001 | scene_total | 0.496 | 0.399 | 0.381 | 0.211 | ok |
| d_non_backbone_recall_repair_r003_r002 | scene_total | 0.493 | 0.408 | 0.38 | 0.212 | ok |
| d_non_backbone_recall_repair_r003_r003 | scene_total | 0.507 | 0.396 | 0.376 | 0.211 | ok |
| d_non_backbone_recall_repair_r003_r010 | scene_total | 0.477 | 0.398 | 0.36 | 0.197 | ok |
| e_high_precision_seed_r012_recall_repair_r001 | scene_total | 0.459 | 0.415 | 0.374 | 0.207 | ok |
| e_high_precision_seed_r012_recall_repair_r002 | scene_total | 0.489 | 0.4 | 0.373 | 0.207 | ok |
