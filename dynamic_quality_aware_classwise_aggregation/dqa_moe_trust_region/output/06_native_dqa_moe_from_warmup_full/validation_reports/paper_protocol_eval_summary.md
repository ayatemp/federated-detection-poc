# Paper Protocol Evaluation Summary

Created UTC: 2026-05-18T00:12:32.569946+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/06_native_dqa_moe_from_warmup_full`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/06_native_dqa_moe_from_warmup_full/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| cloudy | partly cloudy | 738 | 14937 |
| overcast | overcast | 1239 | 25686 |
| rainy | rainy | 738 | 13160 |
| snowy | snowy | 769 | 14321 |
| total | union | 3484 | 0 |

## Checkpoints

- `warmup_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/06_native_dqa_moe_from_warmup_full/global_checkpoints/round000_warmup.pt`
- `phase1_round020_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/06_native_dqa_moe_from_warmup_full/global_checkpoints/phase1_round020_global.pt`
- `phase2_round020_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/06_native_dqa_moe_from_warmup_full/global_checkpoints/phase2_round020_global.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| warmup_global | cloudy | 0.733 | 0.423 | 0.48 | 0.247 | ok |
| warmup_global | overcast | 0.673 | 0.415 | 0.449 | 0.232 | ok |
| warmup_global | rainy | 0.656 | 0.425 | 0.44 | 0.23 | ok |
| warmup_global | snowy | 0.646 | 0.411 | 0.426 | 0.222 | ok |
| warmup_global | total | 0.689 | 0.41 | 0.447 | 0.229 | ok |
| phase1_round020_global | cloudy | 0.756 | 0.445 | 0.505 | 0.282 | ok |
| phase1_round020_global | overcast | 0.706 | 0.436 | 0.476 | 0.265 | ok |
| phase1_round020_global | rainy | 0.671 | 0.445 | 0.462 | 0.262 | ok |
| phase1_round020_global | snowy | 0.636 | 0.43 | 0.439 | 0.251 | ok |
| phase1_round020_global | total | 0.706 | 0.433 | 0.47 | 0.262 | ok |
| phase2_round020_global | cloudy | 0.74 | 0.453 | 0.505 | 0.287 | ok |
| phase2_round020_global | overcast | 0.699 | 0.441 | 0.481 | 0.271 | ok |
| phase2_round020_global | rainy | 0.673 | 0.451 | 0.466 | 0.266 | ok |
| phase2_round020_global | snowy | 0.66 | 0.42 | 0.441 | 0.253 | ok |
| phase2_round020_global | total | 0.706 | 0.435 | 0.474 | 0.266 | ok |

## Total Split

| checkpoint | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: |
| warmup_global | 0.447 | 0.229 |
| phase1_round020_global | 0.47 | 0.262 |
| phase2_round020_global | 0.474 | 0.266 |
