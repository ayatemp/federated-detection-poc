# Paper Protocol Evaluation Summary

Created UTC: 2026-05-14T06:39:02.609712+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/eval_total`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/eval_total/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| scene_daynight_total | union | 9087 | 0 |

## Checkpoints

- `identity_warmup`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/identity_warmup.pt`
- `uniform_tiny_bn`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/uniform_tiny_bn.pt`
- `uniform_small_bn`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/uniform_small_bn.pt`
- `uniform_neck_only_010`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/uniform_neck_only_010.pt`
- `uniform_backbone_only_004`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/uniform_backbone_only_004.pt`
- `align_tiny_bn`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/align_tiny_bn.pt`
- `align_neck_only_012`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/align_neck_only_012.pt`
- `invdiv_tiny_bn`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/invdiv_tiny_bn.pt`
- `night_tiny_bn`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/night_tiny_bn.pt`
- `day_tiny_bn`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/day_tiny_bn.pt`
- `city_res_night_neck`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/city_res_night_neck.pt`
- `highway_neck`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/highway_neck.pt`
- `reverse_uniform_tiny`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/reverse_uniform_tiny.pt`
- `reverse_uniform_neck`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/reverse_uniform_neck.pt`
- `reverse_align_tiny`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/reverse_align_tiny.pt`
- `reverse_night_neck`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/reverse_night_neck.pt`
- `bkwarm_neck_align_pos`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/bkwarm_neck_align_pos.pt`
- `bkwarm_neck_align_neg`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/bkwarm_neck_align_neg.pt`
- `bk_reverse_neck_pos`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/bk_reverse_neck_pos.pt`
- `bk_pos_neck_reverse`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger/checkpoints/bk_pos_neck_reverse.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| identity_warmup | scene_daynight_total | 0.704 | 0.468 | 0.499 | 0.28 | ok |
| uniform_tiny_bn | scene_daynight_total | 0.708 | 0.466 | 0.499 | 0.28 | ok |
| uniform_small_bn | scene_daynight_total | 0.707 | 0.466 | 0.499 | 0.28 | ok |
| uniform_neck_only_010 | scene_daynight_total | 0.707 | 0.467 | 0.5 | 0.28 | ok |
| uniform_backbone_only_004 | scene_daynight_total | 0.711 | 0.464 | 0.499 | 0.28 | ok |
| align_tiny_bn | scene_daynight_total | 0.708 | 0.466 | 0.499 | 0.28 | ok |
| align_neck_only_012 | scene_daynight_total | 0.707 | 0.467 | 0.5 | 0.28 | ok |
| invdiv_tiny_bn | scene_daynight_total | 0.708 | 0.466 | 0.499 | 0.28 | ok |
| night_tiny_bn | scene_daynight_total | 0.71 | 0.465 | 0.499 | 0.28 | ok |
| day_tiny_bn | scene_daynight_total | 0.709 | 0.465 | 0.499 | 0.28 | ok |
| city_res_night_neck | scene_daynight_total | 0.707 | 0.467 | 0.5 | 0.281 | ok |
| highway_neck | scene_daynight_total | 0.705 | 0.468 | 0.5 | 0.281 | ok |
| reverse_uniform_tiny | scene_daynight_total | 0.705 | 0.468 | 0.499 | 0.281 | ok |
| reverse_uniform_neck | scene_daynight_total | 0.705 | 0.467 | 0.498 | 0.28 | ok |
| reverse_align_tiny | scene_daynight_total | 0.705 | 0.467 | 0.499 | 0.281 | ok |
| reverse_night_neck | scene_daynight_total | 0.704 | 0.467 | 0.498 | 0.28 | ok |
| bkwarm_neck_align_pos | scene_daynight_total | 0.71 | 0.465 | 0.499 | 0.28 | ok |
| bkwarm_neck_align_neg | scene_daynight_total | 0.704 | 0.467 | 0.499 | 0.28 | ok |
| bk_reverse_neck_pos | scene_daynight_total | 0.703 | 0.47 | 0.5 | 0.281 | ok |
| bk_pos_neck_reverse | scene_daynight_total | 0.71 | 0.464 | 0.499 | 0.28 | ok |
