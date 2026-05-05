# Paper Protocol Evaluation Summary

Created UTC: 2026-05-01T17:07:43.068866+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_2_scene_phase2_head_protected`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_2_scene_phase2_head_protected/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| highway | union | 2499 | 36377 |
| citystreet | union | 6112 | 127178 |
| residential | union | 1253 | 20855 |
| scene_total | union | 9864 | 0 |

## Checkpoints

- `dqa08_phase1_seed_r012`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_2_scene_phase2_head_protected/global_checkpoints/round000_warmup.pt`
- `dqa08_phase2_r024`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_scene_tri_stage_policy_8h/global_checkpoints/phase2_round024_global.pt`
- `dqa08_2_phase2_r024`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_2_scene_phase2_head_protected/global_checkpoints/phase2_round024_global.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dqa08_phase1_seed_r012 | highway | 0.505 | 0.34 | 0.32 | 0.175 | ok |
| dqa08_phase1_seed_r012 | citystreet | 0.745 | 0.332 | 0.374 | 0.202 | ok |
| dqa08_phase1_seed_r012 | residential | 0.369 | 0.473 | 0.407 | 0.225 | ok |
| dqa08_phase1_seed_r012 | scene_total | 0.743 | 0.326 | 0.368 | 0.199 | ok |
| dqa08_phase2_r024 | highway | 0.581 | 0.288 | 0.29 | 0.158 | ok |
| dqa08_phase2_r024 | citystreet | 0.468 | 0.371 | 0.332 | 0.179 | ok |
| dqa08_phase2_r024 | residential | 0.392 | 0.415 | 0.361 | 0.198 | ok |
| dqa08_phase2_r024 | scene_total | 0.46 | 0.369 | 0.327 | 0.176 | ok |
| dqa08_2_phase2_r024 | highway | 0.523 | 0.299 | 0.287 | 0.156 | ok |
| dqa08_2_phase2_r024 | citystreet | 0.459 | 0.372 | 0.329 | 0.177 | ok |
| dqa08_2_phase2_r024 | residential | 0.386 | 0.409 | 0.354 | 0.196 | ok |
| dqa08_2_phase2_r024 | scene_total | 0.453 | 0.371 | 0.324 | 0.174 | ok |
