# DQA-SoftMoX Night-Targeted Delta Optimizer 10

- created_utc: 2026-05-13T17:09:06.601183+00:00
- rounds: 4,15,19,21
- search: incumbent-rebased coefficients on night mini-slices
- baseline total score: 0.57455

## Top Night Probe Candidates

| rank | candidate | objective | night mean | night worst | night mAP50 |
|---:|---|---:|---:|---:|---:|
| 1 | r004_scaled04_best01_rand000_075 | 0.41525 | 0.46035 | 0.38680 | 0.373 |
| 2 | r021_city_night_recall | 0.41484 | 0.45985 | 0.38640 | 0.372 |
| 3 | r004_head_repair_moe_target | 0.41481 | 0.45963 | 0.38675 | 0.372 |
| 4 | r004_anti_drift_moe_only | 0.41481 | 0.45963 | 0.38675 | 0.372 |
| 5 | r015_rand000 | 0.41479 | 0.45973 | 0.38640 | 0.372 |
| 6 | r015_tiny_repair_delta | 0.41475 | 0.45962 | 0.38665 | 0.372 |
| 7 | r004_city_night_recall | 0.41472 | 0.45947 | 0.38665 | 0.372 |
| 8 | r019_residential_night_precision | 0.41471 | 0.45952 | 0.38640 | 0.372 |
| 9 | r019_anti_drift_moe_only | 0.41468 | 0.45950 | 0.38640 | 0.372 |
| 10 | r004_highway_night_guard | 0.41466 | 0.45973 | 0.38605 | 0.372 |
| 11 | r015_city_night_recall | 0.41466 | 0.45948 | 0.38640 | 0.372 |
| 12 | r019_head_repair_moe_target | 0.41466 | 0.45928 | 0.38675 | 0.372 |

## Full Domain Summary

| rank | candidate | total | day | night | worst | DRO | night mAP50 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | r021_city_night_recall | 0.57455 | 0.61665 | 0.44242 | 0.38425 | 0.46563 | 0.358 |
| 2 | r004_scaled04_best01_rand000_075 | 0.57455 | 0.61743 | 0.44218 | 0.38400 | 0.46560 | 0.358 |
| 3 | r004_head_repair_moe_target | 0.57450 | 0.61670 | 0.44220 | 0.38460 | 0.46558 | 0.358 |
| 4 | r004_anti_drift_moe_only | 0.57455 | 0.61662 | 0.44220 | 0.38460 | 0.46556 | 0.358 |
| 5 | r015_rand000 | 0.57455 | 0.61672 | 0.44208 | 0.38425 | 0.46544 | 0.358 |
| 6 | incumbent_r002 | 0.57455 | 0.61670 | 0.44208 | 0.38425 | 0.46544 | 0.358 |
| 7 | r015_residential_night_precision | 0.57355 | 0.61610 | 0.44217 | 0.38425 | 0.46537 | 0.358 |

## Dynamic Domain Router

| domain | selected | score | incumbent | delta | mAP50 delta |
|---|---|---:|---:|---:|---:|
| citystreet_day | r021_city_night_recall | 0.63110 | 0.63090 | +0.00020 | +0.000 |
| citystreet_night | r021_city_night_recall | 0.46555 | 0.46455 | +0.00100 | +0.001 |
| highway_day | r004_scaled04_best01_rand000_075 | 0.54865 | 0.54765 | +0.00100 | +0.001 |
| highway_night | r004_anti_drift_moe_only | 0.38460 | 0.38425 | +0.00035 | +0.000 |
| residential_day | r004_scaled04_best01_rand000_075 | 0.67260 | 0.67155 | +0.00105 | +0.001 |
| residential_night | r015_residential_night_precision | 0.47775 | 0.47745 | +0.00030 | +0.001 |

## Policy Summary

| policy | mean | day | night | worst | DRO | night mAP50 |
|---|---:|---:|---:|---:|---:|---:|
| incumbent_r002 | 0.52939 | 0.61670 | 0.44208 | 0.38425 | 0.46544 | 0.358 |
| night_domain_router | 0.53004 | 0.61745 | 0.44263 | 0.38460 | 0.46599 | 0.358 |

## Codex Goal Scores

- 実験環境: 90/100
- 原因分析: 84/100
- judge の安定化: 88/100
- 精度向上: 45/100
- 最終ゴール達成度: 73/100

## Interpretation

- 10番は、total mAP ではなく night/domain slice を目的関数にして係数を探索した。
- 精度向上スコアが100未満なら、次はこの結果を教師データにして係数提案モデルを作る。