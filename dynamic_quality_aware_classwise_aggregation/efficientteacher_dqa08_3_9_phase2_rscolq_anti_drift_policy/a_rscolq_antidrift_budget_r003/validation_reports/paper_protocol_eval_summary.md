# Paper Protocol Evaluation Summary

Created UTC: 2026-05-04T11:49:17.497955+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_9_phase2_rscolq_anti_drift_policy/a_rscolq_antidrift_budget_r003`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_9_phase2_rscolq_anti_drift_policy/a_rscolq_antidrift_budget_r003/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| highway | union | 2499 | 36377 |
| citystreet | union | 6112 | 127178 |
| residential | union | 1253 | 20855 |
| scene_total | union | 9864 | 0 |

## Checkpoints

- `a_rscolq_antidrift_budget_r003_r001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_9_phase2_rscolq_anti_drift_policy/a_rscolq_antidrift_budget_r003/global_checkpoints/phase2_round001_global.pt`
- `a_rscolq_antidrift_budget_r003_r002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_9_phase2_rscolq_anti_drift_policy/a_rscolq_antidrift_budget_r003/global_checkpoints/phase2_round002_global.pt`
- `a_rscolq_antidrift_budget_r003_r010`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa08_3_9_phase2_rscolq_anti_drift_policy/a_rscolq_antidrift_budget_r003/global_checkpoints/phase2_round010_global.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| a_rscolq_antidrift_budget_r003_r001 | highway | 0.551 | 0.331 | 0.332 | 0.187 | ok |
| a_rscolq_antidrift_budget_r003_r001 | citystreet | 0.476 | 0.418 | 0.387 | 0.215 | ok |
| a_rscolq_antidrift_budget_r003_r001 | residential | 0.415 | 0.464 | 0.421 | 0.239 | ok |
| a_rscolq_antidrift_budget_r003_r001 | scene_total | 0.49 | 0.401 | 0.381 | 0.212 | ok |
| a_rscolq_antidrift_budget_r003_r002 | highway | 0.515 | 0.343 | 0.333 | 0.187 | ok |
| a_rscolq_antidrift_budget_r003_r002 | citystreet | 0.482 | 0.42 | 0.387 | 0.216 | ok |
| a_rscolq_antidrift_budget_r003_r002 | residential | 0.449 | 0.445 | 0.42 | 0.238 | ok |
| a_rscolq_antidrift_budget_r003_r002 | scene_total | 0.484 | 0.411 | 0.381 | 0.213 | ok |
| a_rscolq_antidrift_budget_r003_r010 | highway | 0.489 | 0.363 | 0.323 | 0.178 | ok |
| a_rscolq_antidrift_budget_r003_r010 | citystreet | 0.483 | 0.407 | 0.37 | 0.204 | ok |
| a_rscolq_antidrift_budget_r003_r010 | residential | 0.466 | 0.424 | 0.399 | 0.224 | ok |
| a_rscolq_antidrift_budget_r003_r010 | scene_total | 0.485 | 0.399 | 0.364 | 0.201 | ok |
