# Paper Protocol Evaluation Summary

Created UTC: 2026-04-30T12:38:27.737816+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa05_scene_class_profile_5h`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa05_scene_class_profile_5h/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| highway | union | 2499 | 36377 |
| citystreet | union | 6112 | 127178 |
| residential | union | 1253 | 20855 |
| scene_total | union | 9864 | 0 |

## Checkpoints

- `warmup`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa05_scene_class_profile_5h/global_checkpoints/round000_warmup.pt`
- `phase1_round008`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa05_scene_class_profile_5h/global_checkpoints/phase1_round008_global.pt`
- `phase2_round002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa05_scene_class_profile_5h/global_checkpoints/phase2_round002_global.pt`
- `phase2_round014`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa05_scene_class_profile_5h/global_checkpoints/phase2_round014_global.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| warmup | highway | 0.484 | 0.364 | 0.339 | 0.184 | ok |
| warmup | citystreet | 0.583 | 0.375 | 0.397 | 0.215 | ok |
| warmup | residential | 0.437 | 0.463 | 0.44 | 0.242 | ok |
| warmup | scene_total | 0.56 | 0.377 | 0.391 | 0.212 | ok |
| phase1_round008 | highway | 0.552 | 0.355 | 0.36 | 0.198 | ok |
| phase1_round008 | citystreet | 0.574 | 0.411 | 0.417 | 0.228 | ok |
| phase1_round008 | residential | 0.63 | 0.426 | 0.464 | 0.256 | ok |
| phase1_round008 | scene_total | 0.576 | 0.404 | 0.412 | 0.225 | ok |
| phase2_round002 | highway | 0.507 | 0.409 | 0.368 | 0.208 | ok |
| phase2_round002 | citystreet | 0.578 | 0.425 | 0.428 | 0.241 | ok |
| phase2_round002 | residential | 0.646 | 0.417 | 0.471 | 0.269 | ok |
| phase2_round002 | scene_total | 0.569 | 0.423 | 0.422 | 0.238 | ok |
| phase2_round014 | highway | 0.509 | 0.405 | 0.354 | 0.195 | ok |
| phase2_round014 | citystreet | 0.597 | 0.407 | 0.404 | 0.222 | ok |
| phase2_round014 | residential | 0.536 | 0.445 | 0.44 | 0.247 | ok |
| phase2_round014 | scene_total | 0.602 | 0.398 | 0.398 | 0.219 | ok |
