# Paper Protocol Evaluation Summary

Created UTC: 2026-05-16T14:07:00.377699+00:00
Workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/02_shared_routed_head_moe_12h`
Validation python: `/root/micromamba/envs/al_yolov8/bin/python`
Report root: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/02_shared_routed_head_moe_12h/validation_reports`

## Splits

| split | raw weather | images | boxes |
| --- | --- | ---: | ---: |
| cloudy | partly cloudy | 738 | 14937 |
| overcast | overcast | 1239 | 25686 |
| rainy | rainy | 738 | 13160 |
| snowy | snowy | 769 | 14321 |
| total | union | 3484 | 0 |

## Checkpoints

- `warmup_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/02_shared_routed_head_moe_12h/global_checkpoints/round000_warmup.pt`
- `phase1_round020_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/02_shared_routed_head_moe_12h/global_checkpoints/phase1_round020_global.pt`
- `phase2_round020_global`: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/02_shared_routed_head_moe_12h/global_checkpoints/phase2_round020_global.pt`

## Results

| checkpoint | split | P | R | mAP@0.5 | mAP@0.5:0.95 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| warmup_global | cloudy | 0.603 | 0.422 | 0.443 | 0.234 | ok |
| warmup_global | overcast | 0.575 | 0.428 | 0.427 | 0.222 | ok |
| warmup_global | rainy | 0.545 | 0.414 | 0.404 | 0.211 | ok |
| warmup_global | snowy | 0.546 | 0.398 | 0.387 | 0.199 | ok |
| warmup_global | total | 0.556 | 0.421 | 0.416 | 0.216 | ok |
| phase1_round020_global | cloudy | 0.573 | 0.448 | 0.453 | 0.24 | ok |
| phase1_round020_global | overcast | 0.596 | 0.436 | 0.438 | 0.23 | ok |
| phase1_round020_global | rainy | 0.712 | 0.379 | 0.408 | 0.215 | ok |
| phase1_round020_global | snowy | 0.521 | 0.418 | 0.391 | 0.205 | ok |
| phase1_round020_global | total | 0.579 | 0.421 | 0.424 | 0.222 | ok |
| phase2_round020_global | cloudy | 0.608 | 0.44 | 0.462 | 0.263 | ok |
| phase2_round020_global | overcast | 0.601 | 0.445 | 0.45 | 0.251 | ok |
| phase2_round020_global | rainy | 0.554 | 0.416 | 0.413 | 0.23 | ok |
| phase2_round020_global | snowy | 0.571 | 0.408 | 0.399 | 0.223 | ok |
| phase2_round020_global | total | 0.576 | 0.431 | 0.434 | 0.242 | ok |

## Total Split

| checkpoint | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: |
| warmup_global | 0.416 | 0.216 |
| phase1_round020_global | 0.424 | 0.222 |
| phase2_round020_global | 0.434 | 0.242 |
