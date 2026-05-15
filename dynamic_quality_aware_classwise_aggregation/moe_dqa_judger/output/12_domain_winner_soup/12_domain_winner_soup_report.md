# DQA-SoftMoX Domain-Winner Soup 12

- created_utc: 2026-05-13T17:54:00.883891+00:00
- method: weighted soup over self-generated domain winners

## Total Evaluation

| rank | soup | total score | mAP50 | mAP50:95 | members |
|---:|---|---:|---:|---:|---|
| 1 | identity_incumbent | 0.57455 | 0.462 | 0.260 | incumbent:1.000 |
| 2 | balanced_domain_winners | 0.57455 | 0.462 | 0.260 | incumbent:0.400;r006_scaled:0.200;r019_resprec:0.200;r021_city:0.100;r015_resprec:0.100 |
| 3 | conservative_domain_winners | 0.57455 | 0.462 | 0.260 | incumbent:0.650;r006_scaled:0.120;r019_resprec:0.120;r021_city:0.060;r015_resprec:0.050 |
| 4 | night_heavy | 0.57455 | 0.462 | 0.260 | incumbent:0.350;r006_scaled:0.250;r019_resprec:0.250;r003_tiny_s:0.100;r015_resprec:0.050 |
| 5 | highway_city_pair | 0.57455 | 0.462 | 0.260 | incumbent:0.500;r006_scaled:0.250;r021_city:0.250 |
| 6 | highway_res_pair | 0.57455 | 0.462 | 0.260 | incumbent:0.500;r006_scaled:0.250;r019_resprec:0.250 |
| 7 | r019_lead | 0.57455 | 0.462 | 0.260 | incumbent:0.500;r019_resprec:0.350;r006_scaled:0.100;r021_city:0.050 |
| 8 | r006_lead | 0.57455 | 0.462 | 0.260 | incumbent:0.500;r006_scaled:0.350;r019_resprec:0.100;r021_city:0.050 |
| 9 | rand_bridge | 0.57455 | 0.462 | 0.260 | incumbent:0.450;r003_rand:0.200;r006_scaled:0.150;r019_resprec:0.150;r021_city:0.050 |
| 10 | low_incumbent_aggressive | 0.57455 | 0.462 | 0.260 | incumbent:0.200;r006_scaled:0.300;r019_resprec:0.300;r021_city:0.100;r015_resprec:0.100 |
| 11 | tiny_s_bridge | 0.57450 | 0.462 | 0.260 | incumbent:0.450;r003_tiny_s:0.200;r006_scaled:0.150;r019_resprec:0.150;r021_city:0.050 |

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
| 8 | highway_res_pair | 0.57455 | 0.61663 | 0.44222 | 0.38460 | 0.46558 | 0.358 |
| 9 | r004_anti_drift_moe_only | 0.57455 | 0.61662 | 0.44220 | 0.38460 | 0.46556 | 0.358 |
| 10 | night_heavy | 0.57455 | 0.61662 | 0.44220 | 0.38460 | 0.46556 | 0.358 |
| 11 | r006_tiny_repair_delta | 0.57445 | 0.61660 | 0.44220 | 0.38460 | 0.46556 | 0.358 |
| 12 | r005_scaled04_best01_sur00_02_025 | 0.57450 | 0.61670 | 0.44198 | 0.38470 | 0.46547 | 0.358 |
| 13 | r005_scaled04_best00_slight_extrapolate_target_025 | 0.57450 | 0.61670 | 0.44197 | 0.38465 | 0.46545 | 0.358 |
| 14 | r015_rand000 | 0.57455 | 0.61672 | 0.44208 | 0.38425 | 0.46544 | 0.358 |
| 15 | incumbent_r002 | 0.57455 | 0.61670 | 0.44208 | 0.38425 | 0.46544 | 0.358 |

## Dynamic Router Pool

| domain | selected | delta score | delta mAP50 |
|---|---|---:|---:|
| citystreet_day | r021_city_night_recall | +0.00020 | +0.000 |
| citystreet_night | r019_residential_night_precision | +0.00115 | +0.001 |
| highway_day | r006_scaled04_best01_rand000_050 | +0.00105 | +0.001 |
| highway_night | r006_scaled04_best01_rand000_050 | +0.00120 | +0.001 |
| residential_day | r019_residential_night_precision | +0.00145 | +0.000 |
| residential_night | r003_tiny_all_s | +0.00105 | +0.001 |

## Policy Summary

| policy | mean | night | worst | DRO |
|---|---:|---:|---:|---:|
| incumbent_r002 | 0.52939 | 0.44208 | 0.38425 | 0.46544 |
| night_domain_router | 0.53041 | 0.44322 | 0.38545 | 0.46654 |

## Codex Goal Scores

- 実験環境: 92/100
- 原因分析: 87/100
- judge の安定化: 88/100
- 精度向上: 56/100
- 最終ゴール達成度: 77/100

## Interpretation

- 12番は domain winner を単一 checkpoint に焼き込めるかを検証した。
- soup が単一モデルで伸びない場合、次は soup ではなく domain-aware routing/policy learning を本命にする。