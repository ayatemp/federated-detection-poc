# Paper Protocol Evaluation Summary

Created UTC: 2026-05-01T04:16:58.303627+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa07_scene_learned_adaptive_policy_8h`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa07_scene_learned_adaptive_policy_8h/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| scene_total | union | 9864 | 0 |

## Checkpoints

- `warmup_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa07_scene_learned_adaptive_policy_8h/global_checkpoints/round000_warmup.pt`
- `phase1_round002_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa07_scene_learned_adaptive_policy_8h/global_checkpoints/phase1_round002_global.pt`
- `phase1_round012_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa07_scene_learned_adaptive_policy_8h/global_checkpoints/phase1_round012_global.pt`
- `phase2_round001_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa07_scene_learned_adaptive_policy_8h/global_checkpoints/phase2_round001_global.pt`
- `phase2_round002_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa07_scene_learned_adaptive_policy_8h/global_checkpoints/phase2_round002_global.pt`
- `phase2_round024_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa07_scene_learned_adaptive_policy_8h/global_checkpoints/phase2_round024_global.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| warmup_global | scene_total | 0.468 | 0.359 | 0.34 | 0.18 | ok |
| phase1_round002_global | scene_total | 0.475 | 0.4 | 0.378 | 0.206 | ok |
| phase1_round012_global | scene_total | 0.57 | 0.336 | 0.371 | 0.2 | ok |
| phase2_round001_global | scene_total | 0.589 | 0.331 | 0.373 | 0.206 | ok |
| phase2_round002_global | scene_total | 0.464 | 0.411 | 0.372 | 0.205 | ok |
| phase2_round024_global | scene_total | 0.466 | 0.371 | 0.328 | 0.176 | ok |
