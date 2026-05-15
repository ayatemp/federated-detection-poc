# DQA-SoftMoX 15 Multi-Round Group Residual Composer

- created_utc: 2026-05-14T04:25:18.278709+00:00
- method: body/head/router/expert groups borrow residuals from different self-generated rounds

## Total Evaluation

| rank | candidate | total score | mAP50 | mAP50:95 |
|---:|---|---:|---:|---:|
| 1 | mr_r006_moe_r019_head | 0.57455 | 0.462 | 0.260 |
| 2 | mr_r004_body_r006_moe_r019_head | 0.57455 | 0.462 | 0.260 |
| 3 | mr_city_res_highway_split | 0.57455 | 0.462 | 0.260 |
| 4 | mr_resnight_tiny_head_r006_moe | 0.57455 | 0.462 | 0.260 |
| 5 | mr_r019_head_router_only | 0.57455 | 0.462 | 0.260 |
| 6 | mr_night_recall_blend | 0.57455 | 0.462 | 0.260 |
| 7 | mr_conservative_all_winners | 0.57450 | 0.462 | 0.260 |
| 8 | mr_r006_highway_preserve | 0.57450 | 0.462 | 0.260 |

## Full Domain Summary

| rank | candidate | total | day | night | worst | DRO | night mAP50 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | r006_scaled04_best01_rand000_050 | 0.57410 | 0.61723 | 0.44240 | 0.38545 | 0.46598 | 0.358 |
| 2 | r019_residential_night_precision | 0.57455 | 0.61710 | 0.44247 | 0.38530 | 0.46596 | 0.358 |
| 3 | r003_tiny_all_s | 0.57350 | 0.61675 | 0.44255 | 0.38460 | 0.46580 | 0.358 |
| 4 | r003_rand003 | 0.57405 | 0.61662 | 0.44255 | 0.38465 | 0.46578 | 0.358 |
| 5 | mr_r006_moe_r019_head | 0.57455 | 0.61708 | 0.44218 | 0.38460 | 0.46565 | 0.358 |
| 6 | mr_city_res_highway_split | 0.57455 | 0.61708 | 0.44218 | 0.38460 | 0.46565 | 0.358 |
| 7 | r021_city_night_recall | 0.57455 | 0.61665 | 0.44242 | 0.38425 | 0.46563 | 0.358 |
| 8 | r004_scaled04_best01_rand000_075 | 0.57455 | 0.61743 | 0.44218 | 0.38400 | 0.46560 | 0.358 |
| 9 | r004_head_repair_moe_target | 0.57450 | 0.61670 | 0.44220 | 0.38460 | 0.46558 | 0.358 |
| 10 | highway_res_pair | 0.57455 | 0.61663 | 0.44222 | 0.38460 | 0.46558 | 0.358 |
| 11 | r004_anti_drift_moe_only | 0.57455 | 0.61662 | 0.44220 | 0.38460 | 0.46556 | 0.358 |
| 12 | night_heavy | 0.57455 | 0.61662 | 0.44220 | 0.38460 | 0.46556 | 0.358 |
| 13 | r006_tiny_repair_delta | 0.57445 | 0.61660 | 0.44220 | 0.38460 | 0.46556 | 0.358 |
| 14 | mr_r006_highway_preserve | 0.57450 | 0.61662 | 0.44218 | 0.38425 | 0.46548 | 0.358 |
| 15 | r005_scaled04_best01_sur00_02_025 | 0.57450 | 0.61670 | 0.44198 | 0.38470 | 0.46547 | 0.358 |
| 16 | r005_scaled04_best00_slight_extrapolate_target_025 | 0.57450 | 0.61670 | 0.44197 | 0.38465 | 0.46545 | 0.358 |
| 17 | mr_r019_head_router_only | 0.57455 | 0.61708 | 0.44185 | 0.38460 | 0.46545 | 0.358 |
| 18 | r015_rand000 | 0.57455 | 0.61672 | 0.44208 | 0.38425 | 0.46544 | 0.358 |

## Dynamic Router Pool

| domain | selected | delta score | delta mAP50 |
|---|---|---:|---:|
| citystreet_day | r021_city_night_recall | +0.00020 | +0.000 |
| citystreet_night | r019_residential_night_precision | +0.00115 | +0.001 |
| highway_day | r006_scaled04_best01_rand000_050 | +0.00105 | +0.001 |
| highway_night | r006_scaled04_best01_rand000_050 | +0.00120 | +0.001 |
| residential_day | mr_r006_moe_r019_head | +0.00150 | +0.000 |
| residential_night | r003_tiny_all_s | +0.00105 | +0.001 |

## Policy Summary

| policy | mean | night | worst | DRO |
|---|---:|---:|---:|---:|
| incumbent_r002 | 0.52939 | 0.44208 | 0.38425 | 0.46544 |
| night_domain_router | 0.53042 | 0.44322 | 0.38545 | 0.46654 |

## Codex Goal Scores

- experiment_env: 95/100
- root_cause_analysis: 92/100
- judge_stability: 91/100
- accuracy_improvement: 83/100
- final_goal: 89/100

## Takeaway

This tests the missing hypothesis from the previous loops: a single round is too coarse.  If no new candidate beats the existing domain-router pool, the next loop should change the training data/curriculum rather than only recombining checkpoints.