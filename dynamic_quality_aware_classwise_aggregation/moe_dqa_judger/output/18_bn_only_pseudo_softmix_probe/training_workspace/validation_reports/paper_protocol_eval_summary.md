# Paper Protocol Evaluation Summary

Created UTC: 2026-05-14T06:08:50.111735+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/18_bn_only_pseudo_softmix_probe/training_workspace`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/18_bn_only_pseudo_softmix_probe/training_workspace/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| highway_day | union | 1290 | 20532 |
| highway_night | union | 1007 | 12708 |
| citystreet_day | union | 3067 | 69855 |
| citystreet_night | union | 2582 | 47259 |
| residential_day | union | 862 | 14881 |
| residential_night | union | 279 | 4184 |
| scene_daynight_total | union | 9087 | 0 |

## Checkpoints

- `warmup_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/18_bn_only_pseudo_softmix_probe/training_workspace/checkpoints/round000_latent_dqamox_warmup.pt`
- `warmup_server_repair_final`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa/output/08_full_latent_dqamox_from_warmup/checkpoints/repair_baseline_p0_round030_server_repair.pt`
- `latent_dqamox_final_aggregate`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/18_bn_only_pseudo_softmix_probe/training_workspace/checkpoints/latent_dqamox_p1_round001_dqa_aggregate.pt`
- `latent_dqamox_final_repair`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/18_bn_only_pseudo_softmix_probe/training_workspace/checkpoints/latent_dqamox_p1_round001_dqa_aggregate.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| warmup_global | highway_day | 0.773 | 0.407 | 0.466 | 0.271 | ok |
| warmup_global | highway_night | 0.54 | 0.297 | 0.309 | 0.173 | ok |
| warmup_global | citystreet_day | 0.751 | 0.5 | 0.549 | 0.314 | ok |
| warmup_global | citystreet_night | 0.637 | 0.376 | 0.396 | 0.211 | ok |
| warmup_global | residential_day | 0.699 | 0.536 | 0.591 | 0.349 | ok |
| warmup_global | residential_night | 0.492 | 0.396 | 0.398 | 0.199 | ok |
| warmup_global | scene_daynight_total | 0.704 | 0.468 | 0.499 | 0.28 | ok |
| warmup_server_repair_final | highway_day | 0.66 | 0.392 | 0.39 | 0.22 | ok |
| warmup_server_repair_final | highway_night | 0.508 | 0.254 | 0.237 | 0.133 | ok |
| warmup_server_repair_final | citystreet_day | 0.698 | 0.443 | 0.455 | 0.247 | ok |
| warmup_server_repair_final | citystreet_night | 0.627 | 0.303 | 0.3 | 0.152 | ok |
| warmup_server_repair_final | residential_day | 0.622 | 0.495 | 0.493 | 0.275 | ok |
| warmup_server_repair_final | residential_night | 0.545 | 0.292 | 0.29 | 0.144 | ok |
| warmup_server_repair_final | scene_daynight_total | 0.669 | 0.402 | 0.404 | 0.216 | ok |
| latent_dqamox_final_aggregate | highway_day | 0.745 | 0.412 | 0.457 | 0.265 | ok |
| latent_dqamox_final_aggregate | highway_night | 0.537 | 0.297 | 0.312 | 0.167 | ok |
| latent_dqamox_final_aggregate | citystreet_day | 0.723 | 0.508 | 0.542 | 0.308 | ok |
| latent_dqamox_final_aggregate | citystreet_night | 0.591 | 0.4 | 0.389 | 0.205 | ok |
| latent_dqamox_final_aggregate | residential_day | 0.649 | 0.564 | 0.589 | 0.342 | ok |
| latent_dqamox_final_aggregate | residential_night | 0.518 | 0.4 | 0.393 | 0.19 | ok |
| latent_dqamox_final_aggregate | scene_daynight_total | 0.708 | 0.458 | 0.492 | 0.274 | ok |
| latent_dqamox_final_repair | highway_day | 0.745 | 0.412 | 0.457 | 0.265 | ok |
| latent_dqamox_final_repair | highway_night | 0.537 | 0.297 | 0.312 | 0.167 | ok |
| latent_dqamox_final_repair | citystreet_day | 0.723 | 0.508 | 0.542 | 0.308 | ok |
| latent_dqamox_final_repair | citystreet_night | 0.591 | 0.4 | 0.389 | 0.205 | ok |
| latent_dqamox_final_repair | residential_day | 0.649 | 0.564 | 0.589 | 0.342 | ok |
| latent_dqamox_final_repair | residential_night | 0.518 | 0.4 | 0.393 | 0.19 | ok |
| latent_dqamox_final_repair | scene_daynight_total | 0.708 | 0.458 | 0.492 | 0.274 | ok |
