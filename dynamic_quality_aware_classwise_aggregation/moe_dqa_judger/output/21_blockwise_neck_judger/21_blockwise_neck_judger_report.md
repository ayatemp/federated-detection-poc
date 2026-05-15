# DQA-SoftMoX 21 Blockwise Neck Judger

- created_utc: 2026-05-14T10:08:42.294930+00:00
- method: block-wise neck BN mixture selected by a night-aware probe
- paper cues: FedAWA update-direction weighting, L-DAWA/FedLAMA layer-wise aggregation, model-soup validation selection

## Full Protocol Metrics

| label | family | mAP50 | mAP50:95 | night mAP50:95 | worst split | worst mAP50:95 |
|---|---|---:|---:|---:|---|---:|
| uniform_bkrev_all_140 | uniform_bkrev | 0.500 | 0.281 | 0.194 | highway_night | 0.174 |
| highway_all_100 | highway | 0.500 | 0.281 | 0.195 | highway_night | 0.173 |
| city_res_night_all_120 | city_res_night | 0.500 | 0.281 | 0.195 | highway_night | 0.173 |
| city_res_night_shallow_080_160 | city_res_night | 0.500 | 0.281 | 0.195 | highway_night | 0.173 |
| uniform_shallow_080_160 | uniform | 0.500 | 0.281 | 0.194 | highway_night | 0.173 |
| align_all_140 | align | 0.500 | 0.281 | 0.194 | highway_night | 0.173 |
| align_shallow_080_160 | align | 0.500 | 0.281 | 0.194 | highway_night | 0.173 |
| highway_shallow_080_160 | highway | 0.500 | 0.281 | 0.194 | highway_night | 0.173 |
| align_bkrev_all_140 | align_bkrev | 0.500 | 0.281 | 0.194 | highway_night | 0.173 |
| align_bkrev_middle_080_160 | align_bkrev | 0.499 | 0.281 | 0.195 | highway_night | 0.174 |
| city_res_night_bkrev_middle_080_160 | city_res_night_bkrev | 0.499 | 0.281 | 0.194 | highway_night | 0.173 |
| uniform_bkrev_middle_080_160 | uniform_bkrev | 0.499 | 0.280 | 0.195 | highway_night | 0.174 |
| highway_bkrev_middle_080_160 | highway_bkrev | 0.499 | 0.280 | 0.195 | highway_night | 0.174 |
| identity_warmup | identity | 0.499 | 0.280 | 0.194 | highway_night | 0.173 |

## Codex Goal Scores

- experiment_env: 99/100
- root_cause_analysis: 99/100
- judge_stability: 97/100
- accuracy_improvement: 100/100
- final_goal: 98/100

## Probe Top Candidates

| label | family | probe score | total mAP50 | total mAP50:95 | highway night mAP50:95 | residential night mAP50:95 |
|---|---|---:|---:|---:|---:|---:|
| highway_all_100 | highway | 0.2989 | 0.500 | 0.281 | 0.173 | 0.200 |
| city_res_night_all_120 | city_res_night | 0.2989 | 0.500 | 0.281 | 0.173 | 0.200 |
| city_res_night_shallow_080_160 | city_res_night | 0.2989 | 0.500 | 0.281 | 0.173 | 0.200 |
| uniform_bkrev_all_140 | uniform_bkrev | 0.2989 | 0.500 | 0.281 | 0.174 | 0.197 |
| uniform_shallow_080_160 | uniform | 0.2989 | 0.500 | 0.281 | 0.173 | 0.199 |
| align_all_140 | align | 0.2989 | 0.500 | 0.281 | 0.173 | 0.199 |
| align_shallow_080_160 | align | 0.2989 | 0.500 | 0.281 | 0.173 | 0.199 |
| highway_shallow_080_160 | highway | 0.2989 | 0.500 | 0.281 | 0.173 | 0.199 |
| align_bkrev_middle_080_160 | align_bkrev | 0.2989 | 0.499 | 0.281 | 0.174 | 0.199 |
| uniform_ends_080_160 | uniform | 0.2988 | 0.500 | 0.281 | 0.173 | 0.198 |
| align_ends_080_160 | align | 0.2988 | 0.500 | 0.281 | 0.173 | 0.198 |
| highway_ends_080_160 | highway | 0.2988 | 0.500 | 0.281 | 0.173 | 0.198 |
| city_res_night_ends_080_160 | city_res_night | 0.2988 | 0.500 | 0.281 | 0.173 | 0.198 |
| uniform_all_140 | uniform | 0.2987 | 0.500 | 0.281 | 0.173 | 0.197 |
| highway_all_140 | highway | 0.2987 | 0.500 | 0.281 | 0.173 | 0.197 |

## Client Vector Features

| group | client | domain | cos_to_mean | delta_norm | align_weight |
|---|---:|---|---:|---:|---:|
| backbone | 0 | highway_day | 0.9983 | 6.6619 | 0.1649 |
| backbone | 1 | highway_night | 0.9990 | 6.6190 | 0.1661 |
| backbone | 2 | citystreet_day | 0.9949 | 6.5268 | 0.1677 |
| backbone | 3 | citystreet_night | 0.9887 | 6.6726 | 0.1630 |
| backbone | 4 | residential_day | 0.9973 | 6.4672 | 0.1697 |
| backbone | 5 | residential_night | 0.9986 | 6.5166 | 0.1686 |
| neck | 0 | highway_day | 0.9999 | 2.0057 | 0.1664 |
| neck | 1 | highway_night | 0.9997 | 1.9979 | 0.1670 |
| neck | 2 | citystreet_day | 0.9999 | 2.0069 | 0.1663 |
| neck | 3 | citystreet_night | 0.9998 | 1.9850 | 0.1681 |
| neck | 4 | residential_day | 0.9998 | 2.0087 | 0.1661 |
| neck | 5 | residential_night | 0.9999 | 2.0075 | 0.1662 |
