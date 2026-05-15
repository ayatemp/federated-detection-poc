# Paper Protocol Evaluation Summary

Created UTC: 2026-05-14T05:22:30.143442+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/17_highres_style_moe_training_probe_v2/training_workspace`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/17_highres_style_moe_training_probe_v2/training_workspace/validation_reports`

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

- `warmup_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/17_highres_style_moe_training_probe_v2/training_workspace/checkpoints/round000_latent_dqamox_warmup.pt`
- `warmup_server_repair_final`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa/output/08_full_latent_dqamox_from_warmup/checkpoints/repair_baseline_p0_round030_server_repair.pt`
- `latent_dqamox_final_aggregate`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/17_highres_style_moe_training_probe_v2/training_workspace/checkpoints/latent_dqamox_p1_round001_dqa_aggregate.pt`
- `latent_dqamox_final_repair`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/17_highres_style_moe_training_probe_v2/training_workspace/checkpoints/latent_dqamox_p1_round001_dqa_aggregate.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| warmup_global | highway_day | 0.731 | 0.4 | 0.442 | 0.253 | ok |
| warmup_global | highway_night | 0.592 | 0.265 | 0.301 | 0.173 | ok |
| warmup_global | citystreet_day | 0.738 | 0.445 | 0.501 | 0.287 | ok |
| warmup_global | citystreet_night | 0.662 | 0.352 | 0.379 | 0.201 | ok |
| warmup_global | residential_day | 0.672 | 0.476 | 0.527 | 0.311 | ok |
| warmup_global | residential_night | 0.763 | 0.35 | 0.419 | 0.236 | ok |
| warmup_global | scene_daynight_total | 0.681 | 0.427 | 0.46 | 0.259 | ok |
| warmup_server_repair_final | highway_day | 0.597 | 0.371 | 0.36 | 0.202 | ok |
| warmup_server_repair_final | highway_night | 0.557 | 0.229 | 0.239 | 0.134 | ok |
| warmup_server_repair_final | citystreet_day | 0.703 | 0.395 | 0.421 | 0.227 | ok |
| warmup_server_repair_final | citystreet_night | 0.623 | 0.294 | 0.298 | 0.149 | ok |
| warmup_server_repair_final | residential_day | 0.641 | 0.402 | 0.437 | 0.249 | ok |
| warmup_server_repair_final | residential_night | 0.631 | 0.29 | 0.319 | 0.163 | ok |
| warmup_server_repair_final | scene_daynight_total | 0.662 | 0.366 | 0.378 | 0.201 | ok |
| latent_dqamox_final_aggregate | highway_day | 0.744 | 0.397 | 0.44 | 0.251 | ok |
| latent_dqamox_final_aggregate | highway_night | 0.569 | 0.271 | 0.305 | 0.173 | ok |
| latent_dqamox_final_aggregate | citystreet_day | 0.713 | 0.456 | 0.5 | 0.286 | ok |
| latent_dqamox_final_aggregate | citystreet_night | 0.686 | 0.344 | 0.381 | 0.201 | ok |
| latent_dqamox_final_aggregate | residential_day | 0.669 | 0.473 | 0.524 | 0.309 | ok |
| latent_dqamox_final_aggregate | residential_night | 0.637 | 0.373 | 0.411 | 0.225 | ok |
| latent_dqamox_final_aggregate | scene_daynight_total | 0.692 | 0.422 | 0.46 | 0.258 | ok |
| latent_dqamox_final_repair | highway_day | 0.744 | 0.397 | 0.44 | 0.251 | ok |
| latent_dqamox_final_repair | highway_night | 0.569 | 0.271 | 0.305 | 0.173 | ok |
| latent_dqamox_final_repair | citystreet_day | 0.713 | 0.456 | 0.5 | 0.286 | ok |
| latent_dqamox_final_repair | citystreet_night | 0.686 | 0.344 | 0.381 | 0.201 | ok |
| latent_dqamox_final_repair | residential_day | 0.669 | 0.473 | 0.524 | 0.309 | ok |
| latent_dqamox_final_repair | residential_night | 0.637 | 0.373 | 0.411 | 0.225 | ok |
| latent_dqamox_final_repair | scene_daynight_total | 0.692 | 0.422 | 0.46 | 0.258 | ok |
