# DQA-SoftMoX Domain Policy Learner 13

- created_utc: 2026-05-13T17:56:40.750246+00:00
- rows: 150
- max_total_drop: 0.0011
- GroupKFold domain CV MAE: 0.075338

## Learned Policy

| domain | selected | actual score | predicted | mAP50 | total |
|---|---|---:|---:|---:|---:|
| citystreet_day | r004_scaled04_best00_rand008_050 | 0.63105 | 0.58498 | 0.506 | 0.57410 |
| citystreet_night | r019_residential_night_precision | 0.46570 | 0.39822 | 0.377 | 0.57455 |
| highway_day | r021_tiny_repair_delta | 0.54770 | 0.65791 | 0.441 | 0.57450 |
| highway_night | r015_residential_night_precision | 0.38425 | 0.47273 | 0.308 | 0.57355 |
| residential_day | r006_scaled04_best01_rand000_050 | 0.67195 | 0.58621 | 0.538 | 0.57410 |
| residential_night | r019_residential_night_precision | 0.47640 | 0.42196 | 0.389 | 0.57455 |

## Summary

| policy | mean | day | night | worst | DRO | night mAP50 |
|---|---:|---:|---:|---:|---:|---:|
| incumbent_r002 | 0.52939 | 0.61670 | 0.44208 | 0.38425 | 0.46544 | 0.358 |
| learned_groupcv_policy | 0.52951 | 0.61690 | 0.44212 | 0.38425 | 0.46550 | 0.358 |
| oracle_policy | 0.53041 | 0.61760 | 0.44322 | 0.38545 | 0.46654 | 0.359 |

## Codex Goal Scores

- 実験環境: 93/100
- 原因分析: 89/100
- judge の安定化: 90/100
- 精度向上: 56/100
- 最終ゴール達成度: 79/100