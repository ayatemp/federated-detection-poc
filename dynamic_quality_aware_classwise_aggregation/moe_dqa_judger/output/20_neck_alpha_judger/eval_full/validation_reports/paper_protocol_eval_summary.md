# Paper Protocol Evaluation Summary

Created UTC: 2026-05-14T08:24:12.000340+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/eval_full`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/eval_full/validation_reports`

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

- `identity_warmup`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/identity_warmup.pt`
- `uniform_neck_140`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/uniform_neck_140.pt`
- `uniform_neck_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/uniform_neck_160.pt`
- `align_neck_140`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/align_neck_140.pt`
- `align_neck_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/align_neck_160.pt`
- `city_res_night_neck_120`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/city_res_night_neck_120.pt`
- `city_res_night_neck_140`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/city_res_night_neck_140.pt`
- `highway_neck_100`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/highway_neck_100.pt`
- `highway_neck_140`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/highway_neck_140.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| identity_warmup | highway_day | 0.773 | 0.407 | 0.466 | 0.271 | ok |
| identity_warmup | highway_night | 0.54 | 0.297 | 0.309 | 0.173 | ok |
| identity_warmup | citystreet_day | 0.751 | 0.5 | 0.549 | 0.314 | ok |
| identity_warmup | citystreet_night | 0.637 | 0.376 | 0.396 | 0.211 | ok |
| identity_warmup | residential_day | 0.699 | 0.536 | 0.591 | 0.349 | ok |
| identity_warmup | residential_night | 0.492 | 0.396 | 0.398 | 0.199 | ok |
| identity_warmup | scene_daynight_total | 0.704 | 0.468 | 0.499 | 0.28 | ok |
| uniform_neck_140 | highway_day | 0.772 | 0.407 | 0.466 | 0.27 | ok |
| uniform_neck_140 | highway_night | 0.541 | 0.296 | 0.31 | 0.173 | ok |
| uniform_neck_140 | citystreet_day | 0.756 | 0.499 | 0.549 | 0.314 | ok |
| uniform_neck_140 | citystreet_night | 0.625 | 0.385 | 0.397 | 0.21 | ok |
| uniform_neck_140 | residential_day | 0.681 | 0.547 | 0.593 | 0.35 | ok |
| uniform_neck_140 | residential_night | 0.494 | 0.393 | 0.395 | 0.197 | ok |
| uniform_neck_140 | scene_daynight_total | 0.709 | 0.466 | 0.5 | 0.281 | ok |
| uniform_neck_160 | highway_day | 0.757 | 0.413 | 0.466 | 0.27 | ok |
| uniform_neck_160 | highway_night | 0.541 | 0.297 | 0.311 | 0.173 | ok |
| uniform_neck_160 | citystreet_day | 0.728 | 0.516 | 0.549 | 0.314 | ok |
| uniform_neck_160 | citystreet_night | 0.627 | 0.384 | 0.398 | 0.211 | ok |
| uniform_neck_160 | residential_day | 0.681 | 0.547 | 0.593 | 0.35 | ok |
| uniform_neck_160 | residential_night | 0.607 | 0.393 | 0.395 | 0.197 | ok |
| uniform_neck_160 | scene_daynight_total | 0.707 | 0.468 | 0.5 | 0.281 | ok |
| align_neck_140 | highway_day | 0.772 | 0.407 | 0.466 | 0.27 | ok |
| align_neck_140 | highway_night | 0.541 | 0.296 | 0.31 | 0.173 | ok |
| align_neck_140 | citystreet_day | 0.756 | 0.499 | 0.549 | 0.314 | ok |
| align_neck_140 | citystreet_night | 0.625 | 0.385 | 0.397 | 0.211 | ok |
| align_neck_140 | residential_day | 0.681 | 0.547 | 0.593 | 0.35 | ok |
| align_neck_140 | residential_night | 0.494 | 0.393 | 0.395 | 0.199 | ok |
| align_neck_140 | scene_daynight_total | 0.704 | 0.469 | 0.5 | 0.281 | ok |
| align_neck_160 | highway_day | 0.771 | 0.408 | 0.466 | 0.27 | ok |
| align_neck_160 | highway_night | 0.54 | 0.297 | 0.311 | 0.173 | ok |
| align_neck_160 | citystreet_day | 0.728 | 0.516 | 0.549 | 0.314 | ok |
| align_neck_160 | citystreet_night | 0.627 | 0.384 | 0.398 | 0.211 | ok |
| align_neck_160 | residential_day | 0.651 | 0.564 | 0.593 | 0.35 | ok |
| align_neck_160 | residential_night | 0.607 | 0.393 | 0.395 | 0.197 | ok |
| align_neck_160 | scene_daynight_total | 0.707 | 0.468 | 0.5 | 0.281 | ok |
| city_res_night_neck_120 | highway_day | 0.772 | 0.407 | 0.466 | 0.27 | ok |
| city_res_night_neck_120 | highway_night | 0.536 | 0.299 | 0.311 | 0.173 | ok |
| city_res_night_neck_120 | citystreet_day | 0.756 | 0.499 | 0.549 | 0.314 | ok |
| city_res_night_neck_120 | citystreet_night | 0.637 | 0.378 | 0.397 | 0.211 | ok |
| city_res_night_neck_120 | residential_day | 0.68 | 0.547 | 0.593 | 0.35 | ok |
| city_res_night_neck_120 | residential_night | 0.493 | 0.393 | 0.397 | 0.2 | ok |
| city_res_night_neck_120 | scene_daynight_total | 0.707 | 0.467 | 0.5 | 0.281 | ok |
| city_res_night_neck_140 | highway_day | 0.771 | 0.407 | 0.466 | 0.27 | ok |
| city_res_night_neck_140 | highway_night | 0.541 | 0.296 | 0.311 | 0.173 | ok |
| city_res_night_neck_140 | citystreet_day | 0.725 | 0.518 | 0.549 | 0.314 | ok |
| city_res_night_neck_140 | citystreet_night | 0.625 | 0.385 | 0.397 | 0.21 | ok |
| city_res_night_neck_140 | residential_day | 0.681 | 0.547 | 0.593 | 0.35 | ok |
| city_res_night_neck_140 | residential_night | 0.495 | 0.393 | 0.395 | 0.197 | ok |
| city_res_night_neck_140 | scene_daynight_total | 0.706 | 0.468 | 0.5 | 0.281 | ok |
| highway_neck_100 | highway_day | 0.774 | 0.407 | 0.466 | 0.271 | ok |
| highway_neck_100 | highway_night | 0.537 | 0.298 | 0.31 | 0.173 | ok |
| highway_neck_100 | citystreet_day | 0.771 | 0.491 | 0.549 | 0.314 | ok |
| highway_neck_100 | citystreet_night | 0.637 | 0.378 | 0.397 | 0.211 | ok |
| highway_neck_100 | residential_day | 0.674 | 0.55 | 0.593 | 0.35 | ok |
| highway_neck_100 | residential_night | 0.491 | 0.393 | 0.397 | 0.2 | ok |
| highway_neck_100 | scene_daynight_total | 0.705 | 0.468 | 0.5 | 0.281 | ok |
| highway_neck_140 | highway_day | 0.771 | 0.408 | 0.466 | 0.27 | ok |
| highway_neck_140 | highway_night | 0.541 | 0.296 | 0.31 | 0.173 | ok |
| highway_neck_140 | citystreet_day | 0.725 | 0.518 | 0.549 | 0.314 | ok |
| highway_neck_140 | citystreet_night | 0.625 | 0.385 | 0.397 | 0.211 | ok |
| highway_neck_140 | residential_day | 0.681 | 0.547 | 0.593 | 0.35 | ok |
| highway_neck_140 | residential_night | 0.494 | 0.392 | 0.395 | 0.197 | ok |
| highway_neck_140 | scene_daynight_total | 0.706 | 0.468 | 0.5 | 0.281 | ok |
