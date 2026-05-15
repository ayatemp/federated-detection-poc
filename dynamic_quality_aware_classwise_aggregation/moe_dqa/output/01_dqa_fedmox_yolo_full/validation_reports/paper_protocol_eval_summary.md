# Paper Protocol Evaluation Summary

Created UTC: 2026-05-13T09:47:39.225122+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa/output/01_dqa_fedmox_yolo_full`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa/output/01_dqa_fedmox_yolo_full/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| scene_daynight_total | union | 9087 | 0 |

## Checkpoints

- `judger_softmix_p1_round001`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/01_judger_probe/checkpoints/judger_softmix_p1_round001.pt`
- `judger_softmix_p1_round002`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/01_judger_probe/checkpoints/judger_softmix_p1_round002.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| judger_softmix_p1_round001 | scene_daynight_total | 0.712 | 0.419 | 0.459 | 0.257 | ok |
| judger_softmix_p1_round002 | scene_daynight_total | 0.695 | 0.43 | 0.461 | 0.26 | ok |
