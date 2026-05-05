# Paper Protocol Evaluation Summary

Created UTC: 2026-04-24T01:00:35.590312+00:00
Workspace: `/app/Object_Detection/navigating_data_heterogeneity/efficientteacher_fedsto`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/navigating_data_heterogeneity/efficientteacher_fedsto/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| cloudy | partly cloudy | 738 | 14937 |

## Checkpoints

- `warmup`: `/app/Object_Detection/navigating_data_heterogeneity/efficientteacher_fedsto/global_checkpoints/round000_warmup.pt`
- `final`: `/app/Object_Detection/navigating_data_heterogeneity/efficientteacher_fedsto/global_checkpoints/phase2_round150_global.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| warmup | cloudy | 0.784 | 0.419 | 0.495 | 0.279 | ok |
| final | cloudy | 0.162 | 0.0431 | 0.105 | 0.0639 | ok |
