# Paper Protocol Evaluation Summary

Created UTC: 2026-05-16T06:37:52.500012+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/01_fedsto_dqa_moe_trust_region_12h`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/01_fedsto_dqa_moe_trust_region_12h/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| cloudy | partly cloudy | 738 | 14937 |
| overcast | overcast | 1239 | 25686 |
| rainy | rainy | 738 | 13160 |
| snowy | snowy | 769 | 14321 |
| total | union | 3484 | 0 |

## Checkpoints

- `warmup_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/01_fedsto_dqa_moe_trust_region_12h/global_checkpoints/round000_warmup.pt`
- `phase1_round020_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/01_fedsto_dqa_moe_trust_region_12h/global_checkpoints/phase1_round020_global.pt`
- `phase2_round020_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/01_fedsto_dqa_moe_trust_region_12h/global_checkpoints/phase2_round020_global.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| warmup_global | cloudy | 0.561 | 0.437 | 0.439 | 0.233 | ok |
| warmup_global | overcast | 0.564 | 0.436 | 0.429 | 0.223 | ok |
| warmup_global | rainy | 0.589 | 0.397 | 0.397 | 0.206 | ok |
| warmup_global | snowy | 0.553 | 0.384 | 0.381 | 0.192 | ok |
| warmup_global | total | 0.551 | 0.421 | 0.413 | 0.213 | ok |
| phase1_round020_global | cloudy | 0.587 | 0.43 | 0.446 | 0.24 | ok |
| phase1_round020_global | overcast | 0.557 | 0.451 | 0.437 | 0.231 | ok |
| phase1_round020_global | rainy | 0.632 | 0.379 | 0.399 | 0.212 | ok |
| phase1_round020_global | snowy | 0.543 | 0.381 | 0.384 | 0.198 | ok |
| phase1_round020_global | total | 0.578 | 0.415 | 0.419 | 0.221 | ok |
| phase2_round020_global | cloudy | 0.644 | 0.423 | 0.457 | 0.261 | ok |
| phase2_round020_global | overcast | 0.64 | 0.432 | 0.453 | 0.252 | ok |
| phase2_round020_global | rainy | 0.589 | 0.409 | 0.414 | 0.232 | ok |
| phase2_round020_global | snowy | 0.583 | 0.373 | 0.393 | 0.217 | ok |
| phase2_round020_global | total | 0.629 | 0.409 | 0.432 | 0.241 | ok |

## Total Split

| checkpoint | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: |
| warmup_global | 0.413 | 0.213 |
| phase1_round020_global | 0.419 | 0.221 |
| phase2_round020_global | 0.432 | 0.241 |
