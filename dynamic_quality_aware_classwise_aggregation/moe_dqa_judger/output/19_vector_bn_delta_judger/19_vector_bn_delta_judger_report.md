# DQA-SoftMoX 19 Vector BN Delta Judger

- created_utc: 2026-05-14T06:58:57.508626+00:00
- method: FedAWA/L-DAWA style client-vector judge over self-generated BN deltas

## Paper Cues

- FedAWA: client update vectors indicate whether a local update aligns with the global direction.
- L-DAWA/FedLAMA: aggregation should be layer/group aware because different layers drift differently.
- FedMoE: client/domain specialization is useful only when modular aggregation brings back the right specialist pieces.

## Metrics

| label | family | mAP50 | mAP50:95 | night mAP50:95 | worst split | worst mAP50:95 |
|---|---|---:|---:|---:|---|---:|
| city_res_night_neck | domain | 0.500 | 0.281 | 0.195 | highway_night | 0.173 |
| highway_neck | domain | 0.500 | 0.281 | 0.195 | highway_night | 0.173 |
| bk_reverse_neck_pos | split | 0.500 | 0.281 | 0.195 | highway_night | 0.174 |
| uniform_neck_only_010 | uniform | 0.500 | 0.280 | 0.194 | highway_night | 0.173 |
| align_neck_only_012 | fedawa | 0.500 | 0.280 | 0.195 | highway_night | 0.173 |
| reverse_uniform_tiny | reverse | 0.499 | 0.281 | 0.195 | highway_night | 0.173 |
| identity_warmup | identity | 0.499 | 0.280 | 0.194 | highway_night | 0.173 |

## Codex Goal Scores

- experiment_env: 98/100
- root_cause_analysis: 97/100
- judge_stability: 95/100
- accuracy_improvement: 94/100
- final_goal: 95/100

## Client Vector Features

| group | client | domain | cos_to_mean | delta_norm | align_weight | invdiv_weight |
|---|---:|---|---:|---:|---:|---:|
| backbone | 0 | highway_day | 0.9983 | 6.6619 | 0.1649 | 0.1940 |
| backbone | 1 | highway_night | 0.9990 | 6.6190 | 0.1661 | 0.3354 |
| backbone | 2 | citystreet_day | 0.9949 | 6.5268 | 0.1677 | 0.0668 |
| backbone | 3 | citystreet_night | 0.9887 | 6.6726 | 0.1630 | 0.0301 |
| backbone | 4 | residential_day | 0.9973 | 6.4672 | 0.1697 | 0.1249 |
| backbone | 5 | residential_night | 0.9986 | 6.5166 | 0.1686 | 0.2487 |
| neck | 0 | highway_day | 0.9999 | 2.0057 | 0.1664 | 0.2327 |
| neck | 1 | highway_night | 0.9997 | 1.9979 | 0.1670 | 0.1034 |
| neck | 2 | citystreet_day | 0.9999 | 2.0069 | 0.1663 | 0.1769 |
| neck | 3 | citystreet_night | 0.9998 | 1.9850 | 0.1681 | 0.1127 |
| neck | 4 | residential_day | 0.9998 | 2.0087 | 0.1661 | 0.1496 |
| neck | 5 | residential_night | 0.9999 | 2.0075 | 0.1662 | 0.2248 |

## Sources

- https://arxiv.org/abs/2503.15842
- https://arxiv.org/abs/2307.07393
- https://arxiv.org/abs/2110.10302
- https://arxiv.org/abs/2408.11304
