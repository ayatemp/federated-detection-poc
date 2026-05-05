# Paper Protocol Evaluation Summary

Created UTC: 2026-05-04T23:16:14.196924+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_3_scene_phase1_dqa_sba_short_plateau_policy`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_3_scene_phase1_dqa_sba_short_plateau_policy/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| highway | union | 2499 | 36377 |
| citystreet | union | 6112 | 127178 |
| residential | union | 1253 | 20855 |
| scene_total | union | 9864 | 0 |

## Checkpoints

- `warmup_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_3_scene_phase1_dqa_sba_short_plateau_policy/global_checkpoints/round000_warmup.pt`
- `phase1_round003_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_3_scene_phase1_dqa_sba_short_plateau_policy/global_checkpoints/phase1_round003_global.pt`
- `phase1_round006_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_3_scene_phase1_dqa_sba_short_plateau_policy/global_checkpoints/phase1_round006_global.pt`
- `phase2_round001_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_3_scene_phase1_dqa_sba_short_plateau_policy/global_checkpoints/phase2_round001_global.pt`
- `phase2_round002_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_3_scene_phase1_dqa_sba_short_plateau_policy/global_checkpoints/phase2_round002_global.pt`
- `phase2_round003_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_3_scene_phase1_dqa_sba_short_plateau_policy/global_checkpoints/phase2_round003_global.pt`
- `phase2_round004_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_3_scene_phase1_dqa_sba_short_plateau_policy/global_checkpoints/phase2_round004_global.pt`
- `phase2_round008_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_3_scene_phase1_dqa_sba_short_plateau_policy/global_checkpoints/phase2_round008_global.pt`
- `phase2_round012_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_3_scene_phase1_dqa_sba_short_plateau_policy/global_checkpoints/phase2_round012_global.pt`
- `phase2_round016_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_4_3_scene_phase1_dqa_sba_short_plateau_policy/global_checkpoints/phase2_round016_global.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| warmup_global | highway | 0.477 | 0.33 | 0.305 | 0.166 | ok |
| warmup_global | citystreet | 0.499 | 0.355 | 0.354 | 0.189 | ok |
| warmup_global | residential | 0.435 | 0.41 | 0.395 | 0.22 | ok |
| warmup_global | scene_total | 0.503 | 0.35 | 0.349 | 0.187 | ok |
| phase1_round003_global | highway | 0.462 | 0.366 | 0.333 | 0.182 | ok |
| phase1_round003_global | citystreet | 0.468 | 0.423 | 0.384 | 0.207 | ok |
| phase1_round003_global | residential | 0.472 | 0.416 | 0.421 | 0.234 | ok |
| phase1_round003_global | scene_total | 0.469 | 0.415 | 0.379 | 0.205 | ok |
| phase1_round006_global | highway | 0.52 | 0.345 | 0.331 | 0.182 | ok |
| phase1_round006_global | citystreet | 0.466 | 0.422 | 0.383 | 0.206 | ok |
| phase1_round006_global | residential | 0.522 | 0.385 | 0.417 | 0.231 | ok |
| phase1_round006_global | scene_total | 0.467 | 0.413 | 0.377 | 0.203 | ok |
| phase2_round001_global | highway | 0.502 | 0.338 | 0.327 | 0.187 | ok |
| phase2_round001_global | citystreet | 0.477 | 0.416 | 0.383 | 0.211 | ok |
| phase2_round001_global | residential | 0.42 | 0.444 | 0.418 | 0.235 | ok |
| phase2_round001_global | scene_total | 0.477 | 0.408 | 0.377 | 0.208 | ok |
| phase2_round002_global | highway | 0.533 | 0.333 | 0.328 | 0.187 | ok |
| phase2_round002_global | citystreet | 0.483 | 0.413 | 0.382 | 0.211 | ok |
| phase2_round002_global | residential | 0.455 | 0.427 | 0.419 | 0.236 | ok |
| phase2_round002_global | scene_total | 0.484 | 0.405 | 0.377 | 0.208 | ok |
| phase2_round003_global | highway | 0.555 | 0.332 | 0.329 | 0.187 | ok |
| phase2_round003_global | citystreet | 0.479 | 0.415 | 0.382 | 0.211 | ok |
| phase2_round003_global | residential | 0.428 | 0.443 | 0.418 | 0.236 | ok |
| phase2_round003_global | scene_total | 0.479 | 0.408 | 0.376 | 0.208 | ok |
| phase2_round004_global | highway | 0.532 | 0.332 | 0.328 | 0.187 | ok |
| phase2_round004_global | citystreet | 0.481 | 0.416 | 0.382 | 0.211 | ok |
| phase2_round004_global | residential | 0.436 | 0.442 | 0.418 | 0.236 | ok |
| phase2_round004_global | scene_total | 0.482 | 0.407 | 0.376 | 0.208 | ok |
| phase2_round008_global | highway | 0.534 | 0.332 | 0.328 | 0.187 | ok |
| phase2_round008_global | citystreet | 0.484 | 0.414 | 0.381 | 0.211 | ok |
| phase2_round008_global | residential | 0.429 | 0.448 | 0.416 | 0.235 | ok |
| phase2_round008_global | scene_total | 0.485 | 0.406 | 0.376 | 0.208 | ok |
| phase2_round012_global | highway | 0.533 | 0.332 | 0.328 | 0.187 | ok |
| phase2_round012_global | citystreet | 0.484 | 0.413 | 0.381 | 0.21 | ok |
| phase2_round012_global | residential | 0.431 | 0.453 | 0.416 | 0.234 | ok |
| phase2_round012_global | scene_total | 0.483 | 0.408 | 0.376 | 0.208 | ok |
| phase2_round016_global | highway | 0.553 | 0.332 | 0.328 | 0.186 | ok |
| phase2_round016_global | citystreet | 0.484 | 0.413 | 0.38 | 0.21 | ok |
| phase2_round016_global | residential | 0.442 | 0.448 | 0.415 | 0.233 | ok |
| phase2_round016_global | scene_total | 0.486 | 0.405 | 0.375 | 0.207 | ok |
