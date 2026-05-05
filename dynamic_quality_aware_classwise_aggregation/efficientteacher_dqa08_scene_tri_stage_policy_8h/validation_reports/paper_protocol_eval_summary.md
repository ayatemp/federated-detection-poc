# Paper Protocol Evaluation Summary

Created UTC: 2026-05-01T11:28:04.962228+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_scene_tri_stage_policy_8h`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_scene_tri_stage_policy_8h/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| scene_total | union | 9864 | 0 |

## Checkpoints

- `warmup_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_scene_tri_stage_policy_8h/global_checkpoints/round000_warmup.pt`
- `phase1_round003_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_scene_tri_stage_policy_8h/global_checkpoints/phase1_round003_global.pt`
- `phase1_round012_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_scene_tri_stage_policy_8h/global_checkpoints/phase1_round012_global.pt`
- `phase2_round001_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_scene_tri_stage_policy_8h/global_checkpoints/phase2_round001_global.pt`
- `phase2_round024_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_scene_tri_stage_policy_8h/global_checkpoints/phase2_round024_global.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| warmup_global | scene_total | 0.733 | 0.317 | 0.353 | 0.189 | ok |
| phase1_round003_global | scene_total | 0.471 | 0.401 | 0.375 | 0.204 | ok |
| phase1_round012_global | scene_total | 0.743 | 0.326 | 0.368 | 0.199 | ok |
| phase2_round001_global | scene_total | 0.469 | 0.401 | 0.368 | 0.203 | ok |
| phase2_round024_global | scene_total | 0.46 | 0.369 | 0.327 | 0.176 | ok |
