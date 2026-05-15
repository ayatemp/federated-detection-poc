# Paper Protocol Evaluation Summary

Created UTC: 2026-05-15T10:06:51.619110+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa/output/02_ema_fixed_allclient_5p5_paper_eval`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa/output/02_ema_fixed_allclient_5p5_paper_eval/validation_reports`

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

- `warmup_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa/output/02_ema_fixed_allclient_5p5_paper_eval/checkpoints/round000_latent_dqamox_warmup.pt`
- `warmup_server_repair_final`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa/output/02_ema_fixed_allclient_5p5_paper_eval/checkpoints/repair_baseline_p0_round005_server_repair.pt`
- `latent_dqamox_final_aggregate`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa/output/02_ema_fixed_allclient_5p5_paper_eval/checkpoints/latent_dqamox_p2_round010_dqa_aggregate.pt`
- `latent_dqamox_final_repair`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa/output/02_ema_fixed_allclient_5p5_paper_eval/checkpoints/latent_dqamox_p2_round010_server_repair.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| warmup_global | highway_day | 0.766 | 0.405 | 0.445 | 0.246 | ok |
| warmup_global | highway_night | 0.588 | 0.295 | 0.313 | 0.178 | ok |
| warmup_global | citystreet_day | 0.742 | 0.453 | 0.5 | 0.284 | ok |
| warmup_global | citystreet_night | 0.655 | 0.358 | 0.373 | 0.197 | ok |
| warmup_global | residential_day | 0.668 | 0.487 | 0.531 | 0.308 | ok |
| warmup_global | residential_night | 0.67 | 0.327 | 0.393 | 0.199 | ok |
| warmup_global | scene_daynight_total | 0.712 | 0.42 | 0.458 | 0.255 | ok |
| warmup_server_repair_final | highway_day | 0.749 | 0.399 | 0.428 | 0.24 | ok |
| warmup_server_repair_final | highway_night | 0.523 | 0.282 | 0.295 | 0.169 | ok |
| warmup_server_repair_final | citystreet_day | 0.708 | 0.462 | 0.49 | 0.28 | ok |
| warmup_server_repair_final | citystreet_night | 0.611 | 0.36 | 0.363 | 0.193 | ok |
| warmup_server_repair_final | residential_day | 0.723 | 0.463 | 0.528 | 0.307 | ok |
| warmup_server_repair_final | residential_night | 0.594 | 0.336 | 0.362 | 0.199 | ok |
| warmup_server_repair_final | scene_daynight_total | 0.707 | 0.413 | 0.446 | 0.25 | ok |
| latent_dqamox_final_aggregate | highway_day | 0.673 | 0.403 | 0.405 | 0.227 | ok |
| latent_dqamox_final_aggregate | highway_night | 0.494 | 0.293 | 0.285 | 0.161 | ok |
| latent_dqamox_final_aggregate | citystreet_day | 0.714 | 0.447 | 0.476 | 0.269 | ok |
| latent_dqamox_final_aggregate | citystreet_night | 0.616 | 0.342 | 0.346 | 0.182 | ok |
| latent_dqamox_final_aggregate | residential_day | 0.715 | 0.463 | 0.512 | 0.296 | ok |
| latent_dqamox_final_aggregate | residential_night | 0.633 | 0.293 | 0.345 | 0.19 | ok |
| latent_dqamox_final_aggregate | scene_daynight_total | 0.698 | 0.404 | 0.431 | 0.239 | ok |
| latent_dqamox_final_repair | highway_day | 0.662 | 0.397 | 0.398 | 0.224 | ok |
| latent_dqamox_final_repair | highway_night | 0.56 | 0.263 | 0.278 | 0.155 | ok |
| latent_dqamox_final_repair | citystreet_day | 0.715 | 0.442 | 0.471 | 0.266 | ok |
| latent_dqamox_final_repair | citystreet_night | 0.607 | 0.345 | 0.345 | 0.179 | ok |
| latent_dqamox_final_repair | residential_day | 0.642 | 0.482 | 0.505 | 0.29 | ok |
| latent_dqamox_final_repair | residential_night | 0.638 | 0.289 | 0.342 | 0.186 | ok |
| latent_dqamox_final_repair | scene_daynight_total | 0.695 | 0.4 | 0.426 | 0.235 | ok |
