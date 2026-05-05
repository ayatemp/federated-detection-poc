# Paper Protocol Evaluation Summary

Created UTC: 2026-05-04T19:58:12.908208+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_2_scene_phase1_dqa_sba_stable_policy`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_2_scene_phase1_dqa_sba_stable_policy/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| highway | union | 2499 | 36377 |
| citystreet | union | 6112 | 127178 |
| residential | union | 1253 | 20855 |
| scene_total | union | 9864 | 0 |

## Checkpoints

- `warmup_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_2_scene_phase1_dqa_sba_stable_policy/global_checkpoints/round000_warmup.pt`
- `phase1_round003_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_2_scene_phase1_dqa_sba_stable_policy/global_checkpoints/phase1_round003_global.pt`
- `phase1_round006_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_2_scene_phase1_dqa_sba_stable_policy/global_checkpoints/phase1_round006_global.pt`
- `phase1_round012_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_2_scene_phase1_dqa_sba_stable_policy/global_checkpoints/phase1_round012_global.pt`
- `phase2_round001_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_2_scene_phase1_dqa_sba_stable_policy/global_checkpoints/phase2_round001_global.pt`
- `phase2_round004_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_2_scene_phase1_dqa_sba_stable_policy/global_checkpoints/phase2_round004_global.pt`
- `phase2_round008_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_2_scene_phase1_dqa_sba_stable_policy/global_checkpoints/phase2_round008_global.pt`
- `phase2_round012_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_2_scene_phase1_dqa_sba_stable_policy/global_checkpoints/phase2_round012_global.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| warmup_global | highway | 0.513 | 0.317 | 0.311 | 0.167 | ok |
| warmup_global | citystreet | 0.534 | 0.346 | 0.358 | 0.191 | ok |
| warmup_global | residential | 0.436 | 0.424 | 0.402 | 0.218 | ok |
| warmup_global | scene_total | 0.498 | 0.362 | 0.353 | 0.189 | ok |
| phase1_round003_global | highway | 0.557 | 0.329 | 0.336 | 0.183 | ok |
| phase1_round003_global | citystreet | 0.489 | 0.405 | 0.385 | 0.209 | ok |
| phase1_round003_global | residential | 0.444 | 0.436 | 0.422 | 0.234 | ok |
| phase1_round003_global | scene_total | 0.491 | 0.397 | 0.38 | 0.206 | ok |
| phase1_round006_global | highway | 0.532 | 0.338 | 0.335 | 0.181 | ok |
| phase1_round006_global | citystreet | 0.486 | 0.407 | 0.385 | 0.209 | ok |
| phase1_round006_global | residential | 0.653 | 0.394 | 0.426 | 0.234 | ok |
| phase1_round006_global | scene_total | 0.486 | 0.401 | 0.38 | 0.205 | ok |
| phase1_round012_global | highway | 0.524 | 0.335 | 0.331 | 0.177 | ok |
| phase1_round012_global | citystreet | 0.558 | 0.355 | 0.38 | 0.205 | ok |
| phase1_round012_global | residential | 0.443 | 0.438 | 0.417 | 0.229 | ok |
| phase1_round012_global | scene_total | 0.56 | 0.35 | 0.374 | 0.201 | ok |
| phase2_round001_global | highway | 0.568 | 0.334 | 0.336 | 0.184 | ok |
| phase2_round001_global | citystreet | 0.477 | 0.415 | 0.381 | 0.21 | ok |
| phase2_round001_global | residential | 0.461 | 0.436 | 0.42 | 0.234 | ok |
| phase2_round001_global | scene_total | 0.482 | 0.405 | 0.375 | 0.207 | ok |
| phase2_round004_global | highway | 0.612 | 0.324 | 0.329 | 0.182 | ok |
| phase2_round004_global | citystreet | 0.468 | 0.422 | 0.377 | 0.208 | ok |
| phase2_round004_global | residential | 0.433 | 0.449 | 0.411 | 0.231 | ok |
| phase2_round004_global | scene_total | 0.471 | 0.411 | 0.372 | 0.205 | ok |
| phase2_round008_global | highway | 0.579 | 0.321 | 0.318 | 0.176 | ok |
| phase2_round008_global | citystreet | 0.47 | 0.409 | 0.369 | 0.203 | ok |
| phase2_round008_global | residential | 0.439 | 0.433 | 0.4 | 0.223 | ok |
| phase2_round008_global | scene_total | 0.478 | 0.396 | 0.363 | 0.199 | ok |
| phase2_round012_global | highway | 0.504 | 0.337 | 0.313 | 0.172 | ok |
| phase2_round012_global | citystreet | 0.516 | 0.37 | 0.362 | 0.198 | ok |
| phase2_round012_global | residential | 0.455 | 0.418 | 0.392 | 0.218 | ok |
| phase2_round012_global | scene_total | 0.491 | 0.38 | 0.356 | 0.194 | ok |
