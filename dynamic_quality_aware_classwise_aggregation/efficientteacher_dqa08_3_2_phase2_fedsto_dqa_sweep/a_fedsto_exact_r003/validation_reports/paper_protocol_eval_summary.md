# Paper Protocol Evaluation Summary

Created UTC: 2026-05-02T15:57:16.387448+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/a_fedsto_exact_r003`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/a_fedsto_exact_r003/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| scene_total | union | 9864 | 0 |

## Checkpoints

- `p1_r003`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_scene_tri_stage_policy_8h/global_checkpoints/phase1_round003_global.pt`
- `p1_r012`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_scene_tri_stage_policy_8h/global_checkpoints/phase1_round012_global.pt`
- `old08_3_e_r02`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_phase2_update_gating_sweep/e_ug_best_phase1_r003/global_checkpoints/phase2_round002_global.pt`
- `a_fedsto_exact_r003_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/a_fedsto_exact_r003/global_checkpoints/phase2_round001_global.pt`
- `a_fedsto_exact_r003_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/a_fedsto_exact_r003/global_checkpoints/phase2_round002_global.pt`
- `a_fedsto_exact_r003_r003`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/a_fedsto_exact_r003/global_checkpoints/phase2_round003_global.pt`
- `a_fedsto_exact_r003_r010`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/a_fedsto_exact_r003/global_checkpoints/phase2_round010_global.pt`
- `b_fedsto_dqa_residual_r003_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/b_fedsto_dqa_residual_r003/global_checkpoints/phase2_round001_global.pt`
- `b_fedsto_dqa_residual_r003_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/b_fedsto_dqa_residual_r003/global_checkpoints/phase2_round002_global.pt`
- `b_fedsto_dqa_residual_r003_r003`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/b_fedsto_dqa_residual_r003/global_checkpoints/phase2_round003_global.pt`
- `b_fedsto_dqa_residual_r003_r010`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/b_fedsto_dqa_residual_r003/global_checkpoints/phase2_round010_global.pt`
- `c_dqa_high_light_head_r003_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/c_dqa_high_light_head_r003/global_checkpoints/phase2_round001_global.pt`
- `c_dqa_high_light_head_r003_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/c_dqa_high_light_head_r003/global_checkpoints/phase2_round002_global.pt`
- `c_dqa_high_light_head_r003_r003`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/c_dqa_high_light_head_r003/global_checkpoints/phase2_round003_global.pt`
- `c_dqa_high_light_head_r003_r010`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/c_dqa_high_light_head_r003/global_checkpoints/phase2_round010_global.pt`
- `d_dqa_strict_localization_r003_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/d_dqa_strict_localization_r003/global_checkpoints/phase2_round001_global.pt`
- `d_dqa_strict_localization_r003_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/d_dqa_strict_localization_r003/global_checkpoints/phase2_round002_global.pt`
- `d_dqa_strict_localization_r003_r003`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/d_dqa_strict_localization_r003/global_checkpoints/phase2_round003_global.pt`
- `d_dqa_strict_localization_r003_r010`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/d_dqa_strict_localization_r003/global_checkpoints/phase2_round010_global.pt`
- `e_dqa_consensus_anchor_r003_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/e_dqa_consensus_anchor_r003/global_checkpoints/phase2_round001_global.pt`
- `e_dqa_consensus_anchor_r003_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/e_dqa_consensus_anchor_r003/global_checkpoints/phase2_round002_global.pt`
- `e_dqa_consensus_anchor_r003_r003`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/e_dqa_consensus_anchor_r003/global_checkpoints/phase2_round003_global.pt`
- `e_dqa_consensus_anchor_r003_r010`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_2_phase2_fedsto_dqa_sweep/e_dqa_consensus_anchor_r003/global_checkpoints/phase2_round010_global.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| p1_r003 | scene_total | 0.471 | 0.401 | 0.375 | 0.204 | ok |
| p1_r012 | scene_total | 0.743 | 0.326 | 0.368 | 0.199 | ok |
| old08_3_e_r02 | scene_total | 0.469 | 0.414 | 0.379 | 0.211 | ok |
| a_fedsto_exact_r003_r001 | scene_total | 0.585 | 0.34 | 0.364 | 0.194 | ok |
| a_fedsto_exact_r003_r002 | scene_total | 0.494 | 0.381 | 0.356 | 0.191 | ok |
| a_fedsto_exact_r003_r003 | scene_total | 0.496 | 0.366 | 0.353 | 0.183 | ok |
| a_fedsto_exact_r003_r010 | scene_total | 0.542 | 0.328 | 0.333 | 0.171 | ok |
| b_fedsto_dqa_residual_r003_r001 | scene_total | 0.467 | 0.391 | 0.362 | 0.192 | ok |
| b_fedsto_dqa_residual_r003_r002 | scene_total | 0.638 | 0.32 | 0.357 | 0.184 | ok |
| b_fedsto_dqa_residual_r003_r003 | scene_total | 0.501 | 0.374 | 0.358 | 0.192 | ok |
| b_fedsto_dqa_residual_r003_r010 | scene_total | 0.523 | 0.342 | 0.337 | 0.175 | ok |
| c_dqa_high_light_head_r003_r001 | scene_total | 0.487 | 0.405 | 0.381 | 0.212 | ok |
| c_dqa_high_light_head_r003_r002 | scene_total | 0.489 | 0.409 | 0.38 | 0.212 | ok |
| c_dqa_high_light_head_r003_r003 | scene_total | 0.511 | 0.396 | 0.377 | 0.211 | ok |
| c_dqa_high_light_head_r003_r010 | scene_total | 0.505 | 0.366 | 0.352 | 0.191 | ok |
| d_dqa_strict_localization_r003_r001 | scene_total | 0.488 | 0.404 | 0.381 | 0.211 | ok |
| d_dqa_strict_localization_r003_r002 | scene_total | 0.491 | 0.409 | 0.38 | 0.212 | ok |
| d_dqa_strict_localization_r003_r003 | scene_total | 0.513 | 0.394 | 0.377 | 0.211 | ok |
| d_dqa_strict_localization_r003_r010 | scene_total | 0.479 | 0.386 | 0.351 | 0.191 | ok |
| e_dqa_consensus_anchor_r003_r001 | scene_total | 0.494 | 0.401 | 0.381 | 0.21 | ok |
| e_dqa_consensus_anchor_r003_r002 | scene_total | 0.489 | 0.408 | 0.379 | 0.212 | ok |
| e_dqa_consensus_anchor_r003_r003 | scene_total | 0.516 | 0.391 | 0.377 | 0.21 | ok |
| e_dqa_consensus_anchor_r003_r010 | scene_total | 0.494 | 0.378 | 0.354 | 0.193 | ok |
