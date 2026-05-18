# Paper Protocol Evaluation Summary

Created UTC: 2026-05-17T01:30:21.333865+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/03_dqa_anonymous_backbone_moe`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/03_dqa_anonymous_backbone_moe/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| cloudy | partly cloudy | 738 | 14937 |
| overcast | overcast | 1239 | 25686 |
| rainy | rainy | 738 | 13160 |
| snowy | snowy | 769 | 14321 |
| total | union | 3484 | 0 |

## Checkpoints

- `warmup_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/03_dqa_anonymous_backbone_moe/global_checkpoints/round000_warmup.pt`
- `phase1_round020_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/03_dqa_anonymous_backbone_moe/global_checkpoints/phase1_round020_global.pt`
- `phase2_round020_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/03_dqa_anonymous_backbone_moe/global_checkpoints/phase2_round020_global.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| warmup_global | cloudy | 0.419 | 0.413 | 0.355 | 0.168 | ok |
| warmup_global | overcast | 0.44 | 0.4 | 0.347 | 0.162 | ok |
| warmup_global | rainy | 0.543 | 0.376 | 0.324 | 0.149 | ok |
| warmup_global | snowy | 0.424 | 0.385 | 0.323 | 0.149 | ok |
| warmup_global | total | 0.438 | 0.387 | 0.336 | 0.156 | ok |
| phase1_round020_global | cloudy | 0.42 | 0.437 | 0.365 | 0.177 | ok |
| phase1_round020_global | overcast | 0.419 | 0.429 | 0.359 | 0.171 | ok |
| phase1_round020_global | rainy | 0.6 | 0.366 | 0.33 | 0.156 | ok |
| phase1_round020_global | snowy | 0.453 | 0.382 | 0.327 | 0.153 | ok |
| phase1_round020_global | total | 0.415 | 0.415 | 0.346 | 0.163 | ok |
| phase2_round020_global | cloudy | 0.504 | 0.447 | 0.438 | 0.249 | ok |
| phase2_round020_global | overcast | 0.638 | 0.424 | 0.426 | 0.243 | ok |
| phase2_round020_global | rainy | 0.695 | 0.385 | 0.403 | 0.226 | ok |
| phase2_round020_global | snowy | 0.679 | 0.385 | 0.389 | 0.217 | ok |
| phase2_round020_global | total | 0.531 | 0.411 | 0.414 | 0.234 | ok |

## Total Split

| checkpoint | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: |
| warmup_global | 0.336 | 0.156 |
| phase1_round020_global | 0.346 | 0.163 |
| phase2_round020_global | 0.414 | 0.234 |
