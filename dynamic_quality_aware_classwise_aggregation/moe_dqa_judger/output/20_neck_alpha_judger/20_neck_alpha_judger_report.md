# DQA-SoftMoX 20 Neck Alpha Judger

- created_utc: 2026-05-14T08:24:12.041651+00:00
- method: focused neck-BN alpha search from self-generated client deltas

## Metrics

| label | family | mAP50 | mAP50:95 | night mAP50:95 | worst split | worst mAP50:95 |
|---|---|---:|---:|---:|---|---:|
| uniform_neck_140 | uniform | 0.500 | 0.281 | 0.193 | highway_night | 0.173 |
| uniform_neck_160 | uniform | 0.500 | 0.281 | 0.194 | highway_night | 0.173 |
| align_neck_140 | align | 0.500 | 0.281 | 0.194 | highway_night | 0.173 |
| align_neck_160 | align | 0.500 | 0.281 | 0.194 | highway_night | 0.173 |
| city_res_night_neck_120 | city_res_night | 0.500 | 0.281 | 0.195 | highway_night | 0.173 |
| city_res_night_neck_140 | city_res_night | 0.500 | 0.281 | 0.193 | highway_night | 0.173 |
| highway_neck_100 | highway | 0.500 | 0.281 | 0.195 | highway_night | 0.173 |
| highway_neck_140 | highway | 0.500 | 0.281 | 0.194 | highway_night | 0.173 |
| identity_warmup | identity | 0.499 | 0.280 | 0.194 | highway_night | 0.173 |

## Codex Goal Scores

- experiment_env: 98/100
- root_cause_analysis: 98/100
- judge_stability: 96/100
- accuracy_improvement: 96/100
- final_goal: 96/100

## Takeaway

This run tests whether the slight positive signal from 19 is a real neck-specific optimum or just validation noise.

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
