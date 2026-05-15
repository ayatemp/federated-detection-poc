# Paper Protocol Evaluation Summary

Created UTC: 2026-05-14T09:23:57.015194+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/eval_probe`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/eval_probe/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| highway_night | union | 1007 | 12708 |
| residential_night | union | 279 | 4184 |
| scene_daynight_total | union | 9087 | 0 |

## Checkpoints

- `identity_warmup`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/identity_warmup.pt`
- `uniform_all_100`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/uniform_all_100.pt`
- `uniform_all_120`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/uniform_all_120.pt`
- `uniform_all_140`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/uniform_all_140.pt`
- `uniform_shallow_080_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/uniform_shallow_080_160.pt`
- `uniform_middle_080_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/uniform_middle_080_160.pt`
- `uniform_deep_080_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/uniform_deep_080_160.pt`
- `uniform_ends_080_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/uniform_ends_080_160.pt`
- `align_all_100`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/align_all_100.pt`
- `align_all_120`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/align_all_120.pt`
- `align_all_140`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/align_all_140.pt`
- `align_shallow_080_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/align_shallow_080_160.pt`
- `align_middle_080_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/align_middle_080_160.pt`
- `align_deep_080_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/align_deep_080_160.pt`
- `align_ends_080_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/align_ends_080_160.pt`
- `highway_all_100`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/highway_all_100.pt`
- `highway_all_120`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/highway_all_120.pt`
- `highway_all_140`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/highway_all_140.pt`
- `highway_shallow_080_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/highway_shallow_080_160.pt`
- `highway_middle_080_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/highway_middle_080_160.pt`
- `highway_deep_080_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/highway_deep_080_160.pt`
- `highway_ends_080_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/highway_ends_080_160.pt`
- `city_res_night_all_100`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/city_res_night_all_100.pt`
- `city_res_night_all_120`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/city_res_night_all_120.pt`
- `city_res_night_all_140`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/city_res_night_all_140.pt`
- `city_res_night_shallow_080_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/city_res_night_shallow_080_160.pt`
- `city_res_night_middle_080_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/city_res_night_middle_080_160.pt`
- `city_res_night_deep_080_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/city_res_night_deep_080_160.pt`
- `city_res_night_ends_080_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/city_res_night_ends_080_160.pt`
- `uniform_bkrev_all_140`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/uniform_bkrev_all_140.pt`
- `uniform_bkrev_middle_080_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/uniform_bkrev_middle_080_160.pt`
- `align_bkrev_all_140`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/align_bkrev_all_140.pt`
- `align_bkrev_middle_080_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/align_bkrev_middle_080_160.pt`
- `highway_bkrev_all_140`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/highway_bkrev_all_140.pt`
- `highway_bkrev_middle_080_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/highway_bkrev_middle_080_160.pt`
- `city_res_night_bkrev_all_140`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/city_res_night_bkrev_all_140.pt`
- `city_res_night_bkrev_middle_080_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger/checkpoints/city_res_night_bkrev_middle_080_160.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| identity_warmup | highway_night | 0.54 | 0.297 | 0.309 | 0.173 | ok |
| identity_warmup | residential_night | 0.492 | 0.396 | 0.398 | 0.199 | ok |
| identity_warmup | scene_daynight_total | 0.704 | 0.468 | 0.499 | 0.28 | ok |
| uniform_all_100 | highway_night | 0.537 | 0.298 | 0.31 | 0.173 | ok |
| uniform_all_100 | residential_night | 0.491 | 0.393 | 0.397 | 0.2 | ok |
| uniform_all_100 | scene_daynight_total | 0.707 | 0.467 | 0.5 | 0.28 | ok |
| uniform_all_120 | highway_night | 0.536 | 0.299 | 0.311 | 0.173 | ok |
| uniform_all_120 | residential_night | 0.493 | 0.392 | 0.397 | 0.199 | ok |
| uniform_all_120 | scene_daynight_total | 0.709 | 0.466 | 0.5 | 0.28 | ok |
| uniform_all_140 | highway_night | 0.541 | 0.296 | 0.31 | 0.173 | ok |
| uniform_all_140 | residential_night | 0.494 | 0.393 | 0.395 | 0.197 | ok |
| uniform_all_140 | scene_daynight_total | 0.709 | 0.466 | 0.5 | 0.281 | ok |
| uniform_shallow_080_160 | highway_night | 0.535 | 0.299 | 0.311 | 0.173 | ok |
| uniform_shallow_080_160 | residential_night | 0.498 | 0.393 | 0.395 | 0.199 | ok |
| uniform_shallow_080_160 | scene_daynight_total | 0.705 | 0.469 | 0.5 | 0.281 | ok |
| uniform_middle_080_160 | highway_night | 0.536 | 0.298 | 0.31 | 0.173 | ok |
| uniform_middle_080_160 | residential_night | 0.61 | 0.392 | 0.397 | 0.2 | ok |
| uniform_middle_080_160 | scene_daynight_total | 0.708 | 0.467 | 0.5 | 0.28 | ok |
| uniform_deep_080_160 | highway_night | 0.548 | 0.293 | 0.31 | 0.173 | ok |
| uniform_deep_080_160 | residential_night | 0.489 | 0.392 | 0.396 | 0.2 | ok |
| uniform_deep_080_160 | scene_daynight_total | 0.708 | 0.467 | 0.499 | 0.28 | ok |
| uniform_ends_080_160 | highway_night | 0.535 | 0.299 | 0.311 | 0.173 | ok |
| uniform_ends_080_160 | residential_night | 0.487 | 0.394 | 0.397 | 0.198 | ok |
| uniform_ends_080_160 | scene_daynight_total | 0.708 | 0.467 | 0.5 | 0.281 | ok |
| align_all_100 | highway_night | 0.537 | 0.298 | 0.31 | 0.173 | ok |
| align_all_100 | residential_night | 0.491 | 0.393 | 0.397 | 0.2 | ok |
| align_all_100 | scene_daynight_total | 0.708 | 0.467 | 0.5 | 0.28 | ok |
| align_all_120 | highway_night | 0.536 | 0.299 | 0.311 | 0.173 | ok |
| align_all_120 | residential_night | 0.492 | 0.393 | 0.397 | 0.2 | ok |
| align_all_120 | scene_daynight_total | 0.707 | 0.467 | 0.5 | 0.28 | ok |
| align_all_140 | highway_night | 0.541 | 0.296 | 0.31 | 0.173 | ok |
| align_all_140 | residential_night | 0.494 | 0.393 | 0.395 | 0.199 | ok |
| align_all_140 | scene_daynight_total | 0.704 | 0.469 | 0.5 | 0.281 | ok |
| align_shallow_080_160 | highway_night | 0.542 | 0.296 | 0.311 | 0.173 | ok |
| align_shallow_080_160 | residential_night | 0.498 | 0.393 | 0.396 | 0.199 | ok |
| align_shallow_080_160 | scene_daynight_total | 0.705 | 0.469 | 0.5 | 0.281 | ok |
| align_middle_080_160 | highway_night | 0.536 | 0.298 | 0.31 | 0.173 | ok |
| align_middle_080_160 | residential_night | 0.61 | 0.392 | 0.397 | 0.2 | ok |
| align_middle_080_160 | scene_daynight_total | 0.708 | 0.467 | 0.5 | 0.28 | ok |
| align_deep_080_160 | highway_night | 0.548 | 0.293 | 0.31 | 0.173 | ok |
| align_deep_080_160 | residential_night | 0.489 | 0.392 | 0.396 | 0.2 | ok |
| align_deep_080_160 | scene_daynight_total | 0.708 | 0.467 | 0.499 | 0.28 | ok |
| align_ends_080_160 | highway_night | 0.535 | 0.299 | 0.311 | 0.173 | ok |
| align_ends_080_160 | residential_night | 0.487 | 0.394 | 0.397 | 0.198 | ok |
| align_ends_080_160 | scene_daynight_total | 0.708 | 0.467 | 0.5 | 0.281 | ok |
| highway_all_100 | highway_night | 0.537 | 0.298 | 0.31 | 0.173 | ok |
| highway_all_100 | residential_night | 0.491 | 0.393 | 0.397 | 0.2 | ok |
| highway_all_100 | scene_daynight_total | 0.705 | 0.468 | 0.5 | 0.281 | ok |
| highway_all_120 | highway_night | 0.536 | 0.299 | 0.311 | 0.173 | ok |
| highway_all_120 | residential_night | 0.492 | 0.393 | 0.397 | 0.199 | ok |
| highway_all_120 | scene_daynight_total | 0.705 | 0.468 | 0.5 | 0.28 | ok |
| highway_all_140 | highway_night | 0.541 | 0.296 | 0.31 | 0.173 | ok |
| highway_all_140 | residential_night | 0.494 | 0.392 | 0.395 | 0.197 | ok |
| highway_all_140 | scene_daynight_total | 0.706 | 0.468 | 0.5 | 0.281 | ok |
| highway_shallow_080_160 | highway_night | 0.535 | 0.299 | 0.311 | 0.173 | ok |
| highway_shallow_080_160 | residential_night | 0.498 | 0.393 | 0.395 | 0.199 | ok |
| highway_shallow_080_160 | scene_daynight_total | 0.705 | 0.469 | 0.5 | 0.281 | ok |
| highway_middle_080_160 | highway_night | 0.536 | 0.298 | 0.31 | 0.173 | ok |
| highway_middle_080_160 | residential_night | 0.609 | 0.392 | 0.397 | 0.2 | ok |
| highway_middle_080_160 | scene_daynight_total | 0.708 | 0.467 | 0.499 | 0.28 | ok |
| highway_deep_080_160 | highway_night | 0.548 | 0.293 | 0.31 | 0.173 | ok |
| highway_deep_080_160 | residential_night | 0.488 | 0.393 | 0.396 | 0.2 | ok |
| highway_deep_080_160 | scene_daynight_total | 0.704 | 0.469 | 0.499 | 0.28 | ok |
| highway_ends_080_160 | highway_night | 0.535 | 0.299 | 0.311 | 0.173 | ok |
| highway_ends_080_160 | residential_night | 0.487 | 0.394 | 0.397 | 0.198 | ok |
| highway_ends_080_160 | scene_daynight_total | 0.708 | 0.467 | 0.5 | 0.281 | ok |
| city_res_night_all_100 | highway_night | 0.537 | 0.298 | 0.31 | 0.173 | ok |
| city_res_night_all_100 | residential_night | 0.491 | 0.393 | 0.397 | 0.199 | ok |
| city_res_night_all_100 | scene_daynight_total | 0.705 | 0.468 | 0.5 | 0.28 | ok |
| city_res_night_all_120 | highway_night | 0.536 | 0.299 | 0.311 | 0.173 | ok |
| city_res_night_all_120 | residential_night | 0.493 | 0.393 | 0.397 | 0.2 | ok |
| city_res_night_all_120 | scene_daynight_total | 0.707 | 0.467 | 0.5 | 0.281 | ok |
| city_res_night_all_140 | highway_night | 0.541 | 0.296 | 0.311 | 0.173 | ok |
| city_res_night_all_140 | residential_night | 0.495 | 0.393 | 0.395 | 0.197 | ok |
| city_res_night_all_140 | scene_daynight_total | 0.706 | 0.468 | 0.5 | 0.281 | ok |
| city_res_night_shallow_080_160 | highway_night | 0.542 | 0.296 | 0.311 | 0.173 | ok |
| city_res_night_shallow_080_160 | residential_night | 0.498 | 0.393 | 0.396 | 0.2 | ok |
| city_res_night_shallow_080_160 | scene_daynight_total | 0.707 | 0.468 | 0.5 | 0.281 | ok |
| city_res_night_middle_080_160 | highway_night | 0.536 | 0.298 | 0.31 | 0.173 | ok |
| city_res_night_middle_080_160 | residential_night | 0.61 | 0.392 | 0.397 | 0.2 | ok |
| city_res_night_middle_080_160 | scene_daynight_total | 0.708 | 0.467 | 0.5 | 0.28 | ok |
| city_res_night_deep_080_160 | highway_night | 0.548 | 0.293 | 0.31 | 0.173 | ok |
| city_res_night_deep_080_160 | residential_night | 0.489 | 0.393 | 0.396 | 0.2 | ok |
| city_res_night_deep_080_160 | scene_daynight_total | 0.704 | 0.469 | 0.499 | 0.28 | ok |
| city_res_night_ends_080_160 | highway_night | 0.535 | 0.299 | 0.311 | 0.173 | ok |
| city_res_night_ends_080_160 | residential_night | 0.488 | 0.394 | 0.396 | 0.198 | ok |
| city_res_night_ends_080_160 | scene_daynight_total | 0.708 | 0.467 | 0.5 | 0.281 | ok |
| uniform_bkrev_all_140 | highway_night | 0.546 | 0.295 | 0.31 | 0.174 | ok |
| uniform_bkrev_all_140 | residential_night | 0.489 | 0.393 | 0.396 | 0.197 | ok |
| uniform_bkrev_all_140 | scene_daynight_total | 0.706 | 0.469 | 0.5 | 0.281 | ok |
| uniform_bkrev_middle_080_160 | highway_night | 0.547 | 0.294 | 0.31 | 0.174 | ok |
| uniform_bkrev_middle_080_160 | residential_night | 0.603 | 0.392 | 0.397 | 0.199 | ok |
| uniform_bkrev_middle_080_160 | scene_daynight_total | 0.705 | 0.469 | 0.499 | 0.28 | ok |
| align_bkrev_all_140 | highway_night | 0.546 | 0.295 | 0.31 | 0.173 | ok |
| align_bkrev_all_140 | residential_night | 0.488 | 0.393 | 0.396 | 0.197 | ok |
| align_bkrev_all_140 | scene_daynight_total | 0.706 | 0.469 | 0.5 | 0.281 | ok |
| align_bkrev_middle_080_160 | highway_night | 0.547 | 0.294 | 0.31 | 0.174 | ok |
| align_bkrev_middle_080_160 | residential_night | 0.603 | 0.392 | 0.397 | 0.199 | ok |
| align_bkrev_middle_080_160 | scene_daynight_total | 0.705 | 0.469 | 0.499 | 0.281 | ok |
| highway_bkrev_all_140 | highway_night | 0.547 | 0.294 | 0.31 | 0.173 | ok |
| highway_bkrev_all_140 | residential_night | 0.488 | 0.393 | 0.395 | 0.197 | ok |
| highway_bkrev_all_140 | scene_daynight_total | 0.706 | 0.469 | 0.5 | 0.281 | ok |
| highway_bkrev_middle_080_160 | highway_night | 0.547 | 0.294 | 0.31 | 0.174 | ok |
| highway_bkrev_middle_080_160 | residential_night | 0.603 | 0.392 | 0.397 | 0.199 | ok |
| highway_bkrev_middle_080_160 | scene_daynight_total | 0.706 | 0.468 | 0.499 | 0.28 | ok |
| city_res_night_bkrev_all_140 | highway_night | 0.546 | 0.295 | 0.31 | 0.173 | ok |
| city_res_night_bkrev_all_140 | residential_night | 0.49 | 0.393 | 0.396 | 0.197 | ok |
| city_res_night_bkrev_all_140 | scene_daynight_total | 0.706 | 0.469 | 0.5 | 0.281 | ok |
| city_res_night_bkrev_middle_080_160 | highway_night | 0.547 | 0.294 | 0.31 | 0.173 | ok |
| city_res_night_bkrev_middle_080_160 | residential_night | 0.603 | 0.392 | 0.397 | 0.199 | ok |
| city_res_night_bkrev_middle_080_160 | scene_daynight_total | 0.705 | 0.469 | 0.499 | 0.281 | ok |
