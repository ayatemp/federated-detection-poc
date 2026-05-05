# Paper Protocol Evaluation Summary

Created UTC: 2026-05-02T02:38:58.058998+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_phase2_update_gating_sweep/a_ug_decay_backbone_r012`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_phase2_update_gating_sweep/a_ug_decay_backbone_r012/validation_reports`

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
- `e_r02`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_phase2_update_gating_sweep/e_ug_best_phase1_r003/global_checkpoints/phase2_round002_global.pt`
- `e_r10`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_phase2_update_gating_sweep/e_ug_best_phase1_r003/global_checkpoints/phase2_round010_global.pt`
- `p2_08_r024`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_scene_tri_stage_policy_8h/global_checkpoints/phase2_round024_global.pt`
- `p2_082_r024`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_2_scene_phase2_head_protected/global_checkpoints/phase2_round024_global.pt`

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
| e_r02 | highway | 0.54 | 0.328 | 0.328 | 0.186 | ok |
| e_r02 | citystreet | 0.469 | 0.422 | 0.386 | 0.215 | ok |
| e_r02 | residential | 0.426 | 0.444 | 0.413 | 0.237 | ok |
| e_r02 | scene_total | 0.469 | 0.414 | 0.379 | 0.211 | ok |
| e_r10 | highway | 0.594 | 0.31 | 0.317 | 0.178 | ok |
| e_r10 | citystreet | 0.484 | 0.405 | 0.374 | 0.207 | ok |
| e_r10 | residential | 0.422 | 0.446 | 0.402 | 0.227 | ok |
| e_r10 | scene_total | 0.484 | 0.399 | 0.367 | 0.203 | ok |
| p2_08_r024 | highway | 0.581 | 0.288 | 0.29 | 0.158 | ok |
| p2_08_r024 | citystreet | 0.468 | 0.371 | 0.332 | 0.179 | ok |
| p2_08_r024 | residential | 0.392 | 0.415 | 0.361 | 0.198 | ok |
| p2_08_r024 | scene_total | 0.46 | 0.369 | 0.327 | 0.176 | ok |
| p2_082_r024 | highway | 0.523 | 0.299 | 0.287 | 0.156 | ok |
| p2_082_r024 | citystreet | 0.459 | 0.372 | 0.329 | 0.177 | ok |
| p2_082_r024 | residential | 0.386 | 0.409 | 0.354 | 0.196 | ok |
| p2_082_r024 | scene_total | 0.453 | 0.371 | 0.324 | 0.174 | ok |
