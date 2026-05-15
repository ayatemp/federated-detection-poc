# Paper Protocol Evaluation Summary

Created UTC: 2026-05-14T06:58:57.467709+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/eval_full`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/eval_full/validation_reports`

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

- `identity_warmup`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/identity_warmup.pt`
- `uniform_neck_only_010`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/uniform_neck_only_010.pt`
- `align_neck_only_012`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/align_neck_only_012.pt`
- `city_res_night_neck`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/city_res_night_neck.pt`
- `highway_neck`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/highway_neck.pt`
- `reverse_uniform_tiny`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/reverse_uniform_tiny.pt`
- `bk_reverse_neck_pos`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/bk_reverse_neck_pos.pt`

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
| uniform_neck_only_010 | highway_day | 0.773 | 0.407 | 0.466 | 0.271 | ok |
| uniform_neck_only_010 | highway_night | 0.537 | 0.298 | 0.31 | 0.173 | ok |
| uniform_neck_only_010 | citystreet_day | 0.726 | 0.516 | 0.549 | 0.314 | ok |
| uniform_neck_only_010 | citystreet_night | 0.637 | 0.378 | 0.397 | 0.21 | ok |
| uniform_neck_only_010 | residential_day | 0.675 | 0.549 | 0.593 | 0.349 | ok |
| uniform_neck_only_010 | residential_night | 0.491 | 0.393 | 0.397 | 0.2 | ok |
| uniform_neck_only_010 | scene_daynight_total | 0.707 | 0.467 | 0.5 | 0.28 | ok |
| align_neck_only_012 | highway_day | 0.772 | 0.407 | 0.466 | 0.27 | ok |
| align_neck_only_012 | highway_night | 0.536 | 0.299 | 0.311 | 0.173 | ok |
| align_neck_only_012 | citystreet_day | 0.73 | 0.514 | 0.549 | 0.314 | ok |
| align_neck_only_012 | citystreet_night | 0.637 | 0.378 | 0.397 | 0.211 | ok |
| align_neck_only_012 | residential_day | 0.679 | 0.547 | 0.593 | 0.35 | ok |
| align_neck_only_012 | residential_night | 0.492 | 0.393 | 0.397 | 0.2 | ok |
| align_neck_only_012 | scene_daynight_total | 0.707 | 0.467 | 0.5 | 0.28 | ok |
| city_res_night_neck | highway_day | 0.772 | 0.407 | 0.466 | 0.27 | ok |
| city_res_night_neck | highway_night | 0.536 | 0.299 | 0.311 | 0.173 | ok |
| city_res_night_neck | citystreet_day | 0.756 | 0.499 | 0.549 | 0.314 | ok |
| city_res_night_neck | citystreet_night | 0.637 | 0.378 | 0.397 | 0.211 | ok |
| city_res_night_neck | residential_day | 0.68 | 0.547 | 0.593 | 0.35 | ok |
| city_res_night_neck | residential_night | 0.493 | 0.393 | 0.397 | 0.2 | ok |
| city_res_night_neck | scene_daynight_total | 0.707 | 0.467 | 0.5 | 0.281 | ok |
| highway_neck | highway_day | 0.774 | 0.407 | 0.466 | 0.271 | ok |
| highway_neck | highway_night | 0.537 | 0.298 | 0.31 | 0.173 | ok |
| highway_neck | citystreet_day | 0.771 | 0.491 | 0.549 | 0.314 | ok |
| highway_neck | citystreet_night | 0.637 | 0.378 | 0.397 | 0.211 | ok |
| highway_neck | residential_day | 0.674 | 0.55 | 0.593 | 0.35 | ok |
| highway_neck | residential_night | 0.491 | 0.393 | 0.397 | 0.2 | ok |
| highway_neck | scene_daynight_total | 0.705 | 0.468 | 0.5 | 0.281 | ok |
| reverse_uniform_tiny | highway_day | 0.763 | 0.41 | 0.466 | 0.271 | ok |
| reverse_uniform_tiny | highway_night | 0.557 | 0.29 | 0.309 | 0.173 | ok |
| reverse_uniform_tiny | citystreet_day | 0.762 | 0.495 | 0.549 | 0.314 | ok |
| reverse_uniform_tiny | citystreet_night | 0.639 | 0.375 | 0.396 | 0.211 | ok |
| reverse_uniform_tiny | residential_day | 0.682 | 0.546 | 0.592 | 0.35 | ok |
| reverse_uniform_tiny | residential_night | 0.486 | 0.396 | 0.398 | 0.2 | ok |
| reverse_uniform_tiny | scene_daynight_total | 0.705 | 0.468 | 0.499 | 0.281 | ok |
| bk_reverse_neck_pos | highway_day | 0.771 | 0.408 | 0.465 | 0.27 | ok |
| bk_reverse_neck_pos | highway_night | 0.542 | 0.297 | 0.31 | 0.174 | ok |
| bk_reverse_neck_pos | citystreet_day | 0.725 | 0.518 | 0.55 | 0.314 | ok |
| bk_reverse_neck_pos | citystreet_night | 0.634 | 0.38 | 0.397 | 0.211 | ok |
| bk_reverse_neck_pos | residential_day | 0.7 | 0.535 | 0.593 | 0.35 | ok |
| bk_reverse_neck_pos | residential_night | 0.49 | 0.395 | 0.397 | 0.2 | ok |
| bk_reverse_neck_pos | scene_daynight_total | 0.703 | 0.47 | 0.5 | 0.281 | ok |
