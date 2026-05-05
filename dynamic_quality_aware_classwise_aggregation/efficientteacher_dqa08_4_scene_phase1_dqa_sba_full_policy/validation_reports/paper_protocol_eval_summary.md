# Paper Protocol Evaluation Summary

Created UTC: 2026-05-04T16:30:51.745169+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_scene_phase1_dqa_sba_full_policy`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_scene_phase1_dqa_sba_full_policy/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| highway | union | 2499 | 36377 |
| citystreet | union | 6112 | 127178 |
| residential | union | 1253 | 20855 |
| scene_total | union | 9864 | 0 |

## Checkpoints

- `warmup_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_scene_phase1_dqa_sba_full_policy/global_checkpoints/round000_warmup.pt`
- `phase1_round003_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_scene_phase1_dqa_sba_full_policy/global_checkpoints/phase1_round003_global.pt`
- `phase1_round012_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_scene_phase1_dqa_sba_full_policy/global_checkpoints/phase1_round012_global.pt`
- `phase2_round001_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_scene_phase1_dqa_sba_full_policy/global_checkpoints/phase2_round001_global.pt`
- `phase2_round004_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_scene_phase1_dqa_sba_full_policy/global_checkpoints/phase2_round004_global.pt`
- `phase2_round008_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_scene_phase1_dqa_sba_full_policy/global_checkpoints/phase2_round008_global.pt`
- `phase2_round012_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_scene_phase1_dqa_sba_full_policy/global_checkpoints/phase2_round012_global.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| warmup_global | highway | 0.581 | 0.298 | 0.308 | 0.168 | ok |
| warmup_global | citystreet | 0.528 | 0.342 | 0.357 | 0.191 | ok |
| warmup_global | residential | 0.434 | 0.406 | 0.388 | 0.214 | ok |
| warmup_global | scene_total | 0.532 | 0.337 | 0.352 | 0.189 | ok |
| phase1_round003_global | highway | 0.491 | 0.345 | 0.334 | 0.18 | ok |
| phase1_round003_global | citystreet | 0.479 | 0.403 | 0.38 | 0.206 | ok |
| phase1_round003_global | residential | 0.439 | 0.429 | 0.412 | 0.228 | ok |
| phase1_round003_global | scene_total | 0.48 | 0.397 | 0.375 | 0.203 | ok |
| phase1_round012_global | highway | 0.474 | 0.354 | 0.327 | 0.176 | ok |
| phase1_round012_global | citystreet | 0.597 | 0.329 | 0.374 | 0.202 | ok |
| phase1_round012_global | residential | 0.4 | 0.449 | 0.407 | 0.226 | ok |
| phase1_round012_global | scene_total | 0.58 | 0.341 | 0.369 | 0.199 | ok |
| phase2_round001_global | highway | 0.511 | 0.339 | 0.33 | 0.181 | ok |
| phase2_round001_global | citystreet | 0.503 | 0.389 | 0.374 | 0.206 | ok |
| phase2_round001_global | residential | 0.455 | 0.427 | 0.408 | 0.23 | ok |
| phase2_round001_global | scene_total | 0.507 | 0.383 | 0.37 | 0.203 | ok |
| phase2_round004_global | highway | 0.499 | 0.34 | 0.322 | 0.179 | ok |
| phase2_round004_global | citystreet | 0.492 | 0.393 | 0.369 | 0.203 | ok |
| phase2_round004_global | residential | 0.458 | 0.43 | 0.403 | 0.227 | ok |
| phase2_round004_global | scene_total | 0.493 | 0.388 | 0.364 | 0.2 | ok |
| phase2_round008_global | highway | 0.501 | 0.331 | 0.314 | 0.173 | ok |
| phase2_round008_global | citystreet | 0.481 | 0.394 | 0.362 | 0.198 | ok |
| phase2_round008_global | residential | 0.449 | 0.434 | 0.394 | 0.22 | ok |
| phase2_round008_global | scene_total | 0.483 | 0.388 | 0.357 | 0.195 | ok |
| phase2_round012_global | highway | 0.521 | 0.315 | 0.305 | 0.167 | ok |
| phase2_round012_global | citystreet | 0.474 | 0.396 | 0.353 | 0.191 | ok |
| phase2_round012_global | residential | 0.424 | 0.427 | 0.388 | 0.218 | ok |
| phase2_round012_global | scene_total | 0.473 | 0.389 | 0.348 | 0.189 | ok |
