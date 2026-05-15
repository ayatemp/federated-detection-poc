# Paper Protocol Evaluation Summary

Created UTC: 2026-05-14T05:45:38.897864+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/17_highres_style_moe_training_probe_v2/training_workspace`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/17_highres_style_moe_training_probe_v2/training_workspace/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| highway_night | union | 1007 | 12708 |
| citystreet_night | union | 2582 | 47259 |
| residential_night | union | 279 | 4184 |
| scene_daynight_total | union | 9087 | 0 |

## Checkpoints

- `warmup_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/17_highres_style_moe_training_probe_v2/training_workspace/checkpoints/round000_latent_dqamox_warmup.pt`
- `client1_highway_night`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/17_highres_style_moe_training_probe_v2/training_workspace/checkpoints/latent_dqamox_p1_round001_client1_highway_night.pt`
- `client5_residential_night`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/17_highres_style_moe_training_probe_v2/training_workspace/checkpoints/latent_dqamox_p1_round001_client5_residential_night.pt`
- `latent_dqamox_final_aggregate`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/17_highres_style_moe_training_probe_v2/training_workspace/checkpoints/latent_dqamox_p1_round001_dqa_aggregate.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| warmup_global | highway_night | 0.54 | 0.297 | 0.309 | 0.173 | ok |
| warmup_global | citystreet_night | 0.637 | 0.376 | 0.396 | 0.211 | ok |
| warmup_global | residential_night | 0.492 | 0.396 | 0.398 | 0.199 | ok |
| warmup_global | scene_daynight_total | 0.704 | 0.468 | 0.499 | 0.28 | ok |
| client1_highway_night | highway_night | 0.529 | 0.279 | 0.283 | 0.149 | ok |
| client1_highway_night | citystreet_night | 0.572 | 0.336 | 0.334 | 0.169 | ok |
| client1_highway_night | residential_night | 0.587 | 0.305 | 0.317 | 0.151 | ok |
| client1_highway_night | scene_daynight_total | 0.667 | 0.436 | 0.453 | 0.247 | ok |
| client5_residential_night | highway_night | 0.518 | 0.281 | 0.282 | 0.148 | ok |
| client5_residential_night | citystreet_night | 0.571 | 0.335 | 0.333 | 0.168 | ok |
| client5_residential_night | residential_night | 0.601 | 0.3 | 0.316 | 0.151 | ok |
| client5_residential_night | scene_daynight_total | 0.664 | 0.436 | 0.452 | 0.246 | ok |
| latent_dqamox_final_aggregate | highway_night | 0.557 | 0.288 | 0.311 | 0.171 | ok |
| latent_dqamox_final_aggregate | citystreet_night | 0.639 | 0.377 | 0.394 | 0.209 | ok |
| latent_dqamox_final_aggregate | residential_night | 0.667 | 0.382 | 0.399 | 0.198 | ok |
| latent_dqamox_final_aggregate | scene_daynight_total | 0.707 | 0.463 | 0.496 | 0.278 | ok |
