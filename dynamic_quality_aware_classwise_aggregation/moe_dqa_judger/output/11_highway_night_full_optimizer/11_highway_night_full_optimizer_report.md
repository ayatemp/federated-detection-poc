# DQA-SoftMoX Highway-Night Full Optimizer 11

- created_utc: 2026-05-13T17:34:47.963602+00:00
- rounds: 4,6,9,15,16,19,21
- target: full highway_night score before total/domain validation

## Top Highway-Night Candidates

| rank | candidate | highway score | mAP50 | mAP50:95 | recall |
|---:|---|---:|---:|---:|---:|
| 1 | r019_residential_night_precision | 0.38530 | 0.309 | 0.177 | 0.287 |
| 2 | r021_tiny_repair_delta | 0.38485 | 0.308 | 0.178 | 0.291 |
| 3 | r004_head_repair_moe_target | 0.38460 | 0.308 | 0.178 | 0.286 |
| 4 | r004_anti_drift_moe_only | 0.38460 | 0.308 | 0.178 | 0.286 |
| 5 | r006_tiny_repair_delta | 0.38460 | 0.308 | 0.178 | 0.286 |
| 6 | r006_head_repair_moe_target | 0.38460 | 0.308 | 0.178 | 0.286 |
| 7 | r006_highway_night_guard | 0.38460 | 0.308 | 0.178 | 0.286 |
| 8 | r006_anti_drift_moe_only | 0.38460 | 0.308 | 0.178 | 0.286 |
| 9 | r009_residential_night_precision | 0.38460 | 0.308 | 0.178 | 0.286 |
| 10 | r015_tiny_repair_delta | 0.38460 | 0.308 | 0.178 | 0.286 |
| 11 | r019_head_repair_moe_target | 0.38460 | 0.308 | 0.178 | 0.286 |
| 12 | r021_residential_night_precision | 0.38450 | 0.308 | 0.177 | 0.291 |
| 13 | r004_city_night_recall | 0.38450 | 0.308 | 0.178 | 0.284 |
| 14 | r006_scaled04_best00_rand007_035 | 0.38430 | 0.308 | 0.177 | 0.287 |
| 15 | r019_city_night_recall | 0.38430 | 0.308 | 0.177 | 0.287 |

## Full Domain Summary

| rank | candidate | total | day | night | worst | DRO | night mAP50 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | r006_scaled04_best01_rand000_050 | 0.57410 | 0.61723 | 0.44240 | 0.38545 | 0.46598 | 0.358 |
| 2 | r019_residential_night_precision | 0.57455 | 0.61710 | 0.44247 | 0.38530 | 0.46596 | 0.358 |
| 3 | r003_tiny_all_s | 0.57350 | 0.61675 | 0.44255 | 0.38460 | 0.46580 | 0.358 |
| 4 | r003_rand003 | 0.57405 | 0.61662 | 0.44255 | 0.38465 | 0.46578 | 0.358 |
| 5 | r021_city_night_recall | 0.57455 | 0.61665 | 0.44242 | 0.38425 | 0.46563 | 0.358 |
| 6 | r004_scaled04_best01_rand000_075 | 0.57455 | 0.61743 | 0.44218 | 0.38400 | 0.46560 | 0.358 |
| 7 | r004_head_repair_moe_target | 0.57450 | 0.61670 | 0.44220 | 0.38460 | 0.46558 | 0.358 |
| 8 | r004_anti_drift_moe_only | 0.57455 | 0.61662 | 0.44220 | 0.38460 | 0.46556 | 0.358 |
| 9 | r006_tiny_repair_delta | 0.57445 | 0.61660 | 0.44220 | 0.38460 | 0.46556 | 0.358 |
| 10 | r005_scaled04_best01_sur00_02_025 | 0.57450 | 0.61670 | 0.44198 | 0.38470 | 0.46547 | 0.358 |
| 11 | r005_scaled04_best00_slight_extrapolate_target_025 | 0.57450 | 0.61670 | 0.44197 | 0.38465 | 0.46545 | 0.358 |
| 12 | r015_rand000 | 0.57455 | 0.61672 | 0.44208 | 0.38425 | 0.46544 | 0.358 |

## Dynamic Domain Router

| domain | selected | score | incumbent | delta | mAP50 delta |
|---|---|---:|---:|---:|---:|
| citystreet_day | r021_city_night_recall | 0.63110 | 0.63090 | +0.00020 | +0.000 |
| citystreet_night | r019_residential_night_precision | 0.46570 | 0.46455 | +0.00115 | +0.001 |
| highway_day | r006_scaled04_best01_rand000_050 | 0.54870 | 0.54765 | +0.00105 | +0.001 |
| highway_night | r006_scaled04_best01_rand000_050 | 0.38545 | 0.38425 | +0.00120 | +0.001 |
| residential_day | r019_residential_night_precision | 0.67300 | 0.67155 | +0.00145 | +0.000 |
| residential_night | r015_residential_night_precision | 0.47775 | 0.47745 | +0.00030 | +0.001 |

## Policy Summary

| policy | mean | day | night | worst | DRO | night mAP50 |
|---|---:|---:|---:|---:|---:|---:|
| incumbent_r002 | 0.52939 | 0.61670 | 0.44208 | 0.38425 | 0.46544 | 0.358 |
| night_domain_router | 0.53028 | 0.61760 | 0.44297 | 0.38545 | 0.46639 | 0.359 |

## Codex Goal Scores

- 実験環境: 91/100
- 原因分析: 86/100
- judge の安定化: 88/100
- 精度向上: 52/100
- 最終ゴール達成度: 75/100

## Interpretation

- 11番は mini proxy を外し、最弱の highway_night を full slice で直接 judge 信号にした。
- それでも精度向上が100未満なら、次は係数探索だけでなく checkpoint candidates を統合する policy learner に移る。