# Paper Protocol Evaluation Summary

Created UTC: 2026-05-14T07:58:35.414864+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/eval_total`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/eval_total/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| scene_daynight_total | union | 9087 | 0 |

## Checkpoints

- `identity_warmup`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/identity_warmup.pt`
- `uniform_neck_060`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/uniform_neck_060.pt`
- `uniform_neck_080`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/uniform_neck_080.pt`
- `uniform_neck_100`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/uniform_neck_100.pt`
- `uniform_neck_120`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/uniform_neck_120.pt`
- `uniform_neck_140`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/uniform_neck_140.pt`
- `uniform_neck_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/uniform_neck_160.pt`
- `uniform_neck_200`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/uniform_neck_200.pt`
- `uniform_neck_240`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/uniform_neck_240.pt`
- `align_neck_060`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/align_neck_060.pt`
- `align_neck_080`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/align_neck_080.pt`
- `align_neck_100`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/align_neck_100.pt`
- `align_neck_120`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/align_neck_120.pt`
- `align_neck_140`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/align_neck_140.pt`
- `align_neck_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/align_neck_160.pt`
- `align_neck_200`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/align_neck_200.pt`
- `align_neck_240`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/align_neck_240.pt`
- `city_res_night_neck_060`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/city_res_night_neck_060.pt`
- `city_res_night_neck_080`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/city_res_night_neck_080.pt`
- `city_res_night_neck_100`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/city_res_night_neck_100.pt`
- `city_res_night_neck_120`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/city_res_night_neck_120.pt`
- `city_res_night_neck_140`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/city_res_night_neck_140.pt`
- `city_res_night_neck_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/city_res_night_neck_160.pt`
- `city_res_night_neck_200`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/city_res_night_neck_200.pt`
- `city_res_night_neck_240`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/city_res_night_neck_240.pt`
- `highway_neck_060`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/highway_neck_060.pt`
- `highway_neck_080`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/highway_neck_080.pt`
- `highway_neck_100`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/highway_neck_100.pt`
- `highway_neck_120`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/highway_neck_120.pt`
- `highway_neck_140`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/highway_neck_140.pt`
- `highway_neck_160`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/highway_neck_160.pt`
- `highway_neck_200`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/highway_neck_200.pt`
- `highway_neck_240`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/highway_neck_240.pt`
- `align_bkm03_neck100`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/align_bkm03_neck100.pt`
- `align_bkm03_neck140`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/align_bkm03_neck140.pt`
- `align_bkm03_neck200`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/align_bkm03_neck200.pt`
- `align_bkp02_neck100`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/align_bkp02_neck100.pt`
- `align_bkp02_neck140`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/align_bkp02_neck140.pt`
- `align_bkp02_neck200`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/align_bkp02_neck200.pt`
- `city_res_night_bkm03_neck100`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/city_res_night_bkm03_neck100.pt`
- `city_res_night_bkm03_neck140`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/city_res_night_bkm03_neck140.pt`
- `city_res_night_bkm03_neck200`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/city_res_night_bkm03_neck200.pt`
- `city_res_night_bkp02_neck100`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/city_res_night_bkp02_neck100.pt`
- `city_res_night_bkp02_neck140`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/city_res_night_bkp02_neck140.pt`
- `city_res_night_bkp02_neck200`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/city_res_night_bkp02_neck200.pt`
- `highway_bkm03_neck100`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/highway_bkm03_neck100.pt`
- `highway_bkm03_neck140`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/highway_bkm03_neck140.pt`
- `highway_bkm03_neck200`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/highway_bkm03_neck200.pt`
- `highway_bkp02_neck100`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/highway_bkp02_neck100.pt`
- `highway_bkp02_neck140`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/highway_bkp02_neck140.pt`
- `highway_bkp02_neck200`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger/checkpoints/highway_bkp02_neck200.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| identity_warmup | scene_daynight_total | 0.704 | 0.468 | 0.499 | 0.28 | ok |
| uniform_neck_060 | scene_daynight_total | 0.71 | 0.465 | 0.499 | 0.28 | ok |
| uniform_neck_080 | scene_daynight_total | 0.709 | 0.466 | 0.5 | 0.28 | ok |
| uniform_neck_100 | scene_daynight_total | 0.707 | 0.467 | 0.5 | 0.28 | ok |
| uniform_neck_120 | scene_daynight_total | 0.709 | 0.466 | 0.5 | 0.28 | ok |
| uniform_neck_140 | scene_daynight_total | 0.709 | 0.466 | 0.5 | 0.281 | ok |
| uniform_neck_160 | scene_daynight_total | 0.707 | 0.468 | 0.5 | 0.281 | ok |
| uniform_neck_200 | scene_daynight_total | 0.706 | 0.468 | 0.5 | 0.281 | ok |
| uniform_neck_240 | scene_daynight_total | 0.705 | 0.469 | 0.5 | 0.28 | ok |
| align_neck_060 | scene_daynight_total | 0.71 | 0.465 | 0.499 | 0.28 | ok |
| align_neck_080 | scene_daynight_total | 0.71 | 0.466 | 0.5 | 0.28 | ok |
| align_neck_100 | scene_daynight_total | 0.708 | 0.467 | 0.5 | 0.28 | ok |
| align_neck_120 | scene_daynight_total | 0.707 | 0.467 | 0.5 | 0.28 | ok |
| align_neck_140 | scene_daynight_total | 0.704 | 0.469 | 0.5 | 0.281 | ok |
| align_neck_160 | scene_daynight_total | 0.707 | 0.468 | 0.5 | 0.281 | ok |
| align_neck_200 | scene_daynight_total | 0.706 | 0.468 | 0.5 | 0.281 | ok |
| align_neck_240 | scene_daynight_total | 0.705 | 0.469 | 0.5 | 0.28 | ok |
| city_res_night_neck_060 | scene_daynight_total | 0.71 | 0.465 | 0.499 | 0.28 | ok |
| city_res_night_neck_080 | scene_daynight_total | 0.71 | 0.466 | 0.5 | 0.28 | ok |
| city_res_night_neck_100 | scene_daynight_total | 0.705 | 0.468 | 0.5 | 0.28 | ok |
| city_res_night_neck_120 | scene_daynight_total | 0.707 | 0.467 | 0.5 | 0.281 | ok |
| city_res_night_neck_140 | scene_daynight_total | 0.706 | 0.468 | 0.5 | 0.281 | ok |
| city_res_night_neck_160 | scene_daynight_total | 0.707 | 0.468 | 0.5 | 0.281 | ok |
| city_res_night_neck_200 | scene_daynight_total | 0.706 | 0.468 | 0.5 | 0.281 | ok |
| city_res_night_neck_240 | scene_daynight_total | 0.705 | 0.469 | 0.5 | 0.28 | ok |
| highway_neck_060 | scene_daynight_total | 0.71 | 0.465 | 0.499 | 0.28 | ok |
| highway_neck_080 | scene_daynight_total | 0.709 | 0.466 | 0.5 | 0.28 | ok |
| highway_neck_100 | scene_daynight_total | 0.705 | 0.468 | 0.5 | 0.281 | ok |
| highway_neck_120 | scene_daynight_total | 0.705 | 0.468 | 0.5 | 0.28 | ok |
| highway_neck_140 | scene_daynight_total | 0.706 | 0.468 | 0.5 | 0.281 | ok |
| highway_neck_160 | scene_daynight_total | 0.707 | 0.468 | 0.5 | 0.281 | ok |
| highway_neck_200 | scene_daynight_total | 0.706 | 0.468 | 0.5 | 0.281 | ok |
| highway_neck_240 | scene_daynight_total | 0.705 | 0.469 | 0.5 | 0.28 | ok |
| align_bkm03_neck100 | scene_daynight_total | 0.704 | 0.47 | 0.5 | 0.281 | ok |
| align_bkm03_neck140 | scene_daynight_total | 0.706 | 0.469 | 0.5 | 0.281 | ok |
| align_bkm03_neck200 | scene_daynight_total | 0.705 | 0.469 | 0.5 | 0.281 | ok |
| align_bkp02_neck100 | scene_daynight_total | 0.707 | 0.467 | 0.499 | 0.28 | ok |
| align_bkp02_neck140 | scene_daynight_total | 0.708 | 0.467 | 0.5 | 0.28 | ok |
| align_bkp02_neck200 | scene_daynight_total | 0.706 | 0.468 | 0.5 | 0.28 | ok |
| city_res_night_bkm03_neck100 | scene_daynight_total | 0.705 | 0.469 | 0.5 | 0.281 | ok |
| city_res_night_bkm03_neck140 | scene_daynight_total | 0.706 | 0.469 | 0.5 | 0.281 | ok |
| city_res_night_bkm03_neck200 | scene_daynight_total | 0.705 | 0.469 | 0.5 | 0.281 | ok |
| city_res_night_bkp02_neck100 | scene_daynight_total | 0.709 | 0.466 | 0.499 | 0.28 | ok |
| city_res_night_bkp02_neck140 | scene_daynight_total | 0.709 | 0.466 | 0.5 | 0.28 | ok |
| city_res_night_bkp02_neck200 | scene_daynight_total | 0.706 | 0.469 | 0.5 | 0.28 | ok |
| highway_bkm03_neck100 | scene_daynight_total | 0.707 | 0.468 | 0.5 | 0.281 | ok |
| highway_bkm03_neck140 | scene_daynight_total | 0.706 | 0.469 | 0.5 | 0.281 | ok |
| highway_bkm03_neck200 | scene_daynight_total | 0.705 | 0.469 | 0.5 | 0.281 | ok |
| highway_bkp02_neck100 | scene_daynight_total | 0.707 | 0.467 | 0.499 | 0.28 | ok |
| highway_bkp02_neck140 | scene_daynight_total | 0.709 | 0.466 | 0.499 | 0.28 | ok |
| highway_bkp02_neck200 | scene_daynight_total | 0.704 | 0.469 | 0.5 | 0.28 | ok |
