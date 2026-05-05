# Paper Protocol Evaluation Summary

Created UTC: 2026-04-30T21:18:51.294351+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa06_scene_adaptive_pseudogt_8h`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa06_scene_adaptive_pseudogt_8h/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| highway | union | 2499 | 36377 |
| citystreet | union | 6112 | 127178 |
| residential | union | 1253 | 20855 |
| scene_total | union | 9864 | 0 |

## Checkpoints

- `warmup`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa06_scene_adaptive_pseudogt_8h/global_checkpoints/round000_warmup.pt`
- `phase1_round004`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa06_scene_adaptive_pseudogt_8h/global_checkpoints/phase1_round004_global.pt`
- `phase1_round012`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa06_scene_adaptive_pseudogt_8h/global_checkpoints/phase1_round012_global.pt`
- `phase2_round002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa06_scene_adaptive_pseudogt_8h/global_checkpoints/phase2_round002_global.pt`
- `phase2_round024`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa06_scene_adaptive_pseudogt_8h/global_checkpoints/phase2_round024_global.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| warmup | highway | 0.441 | 0.342 | 0.307 | 0.167 | ok |
| warmup | citystreet | 0.741 | 0.31 | 0.355 | 0.188 | ok |
| warmup | residential | 0.694 | 0.348 | 0.387 | 0.214 | ok |
| warmup | scene_total | 0.744 | 0.305 | 0.35 | 0.186 | ok |
| phase1_round004 | highway | 0.445 | 0.365 | 0.325 | 0.18 | ok |
| phase1_round004 | citystreet | 0.658 | 0.326 | 0.373 | 0.203 | ok |
| phase1_round004 | residential | 0.715 | 0.359 | 0.403 | 0.225 | ok |
| phase1_round004 | scene_total | 0.645 | 0.326 | 0.367 | 0.2 | ok |
| phase1_round012 | highway | 0.751 | 0.268 | 0.313 | 0.174 | ok |
| phase1_round012 | citystreet | 0.745 | 0.327 | 0.366 | 0.198 | ok |
| phase1_round012 | residential | 0.691 | 0.364 | 0.393 | 0.216 | ok |
| phase1_round012 | scene_total | 0.744 | 0.321 | 0.36 | 0.195 | ok |
| phase2_round002 | highway | 0.667 | 0.298 | 0.311 | 0.176 | ok |
| phase2_round002 | citystreet | 0.535 | 0.33 | 0.364 | 0.199 | ok |
| phase2_round002 | residential | 0.574 | 0.358 | 0.39 | 0.218 | ok |
| phase2_round002 | scene_total | 0.534 | 0.324 | 0.358 | 0.196 | ok |
| phase2_round024 | highway | 0.694 | 0.266 | 0.284 | 0.155 | ok |
| phase2_round024 | citystreet | 0.619 | 0.316 | 0.331 | 0.175 | ok |
| phase2_round024 | residential | 0.45 | 0.357 | 0.354 | 0.19 | ok |
| phase2_round024 | scene_total | 0.529 | 0.311 | 0.325 | 0.172 | ok |
