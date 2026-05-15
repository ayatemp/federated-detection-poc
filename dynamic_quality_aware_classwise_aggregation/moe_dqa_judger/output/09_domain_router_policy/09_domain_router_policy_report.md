# DQA-SoftMoX Domain Router Policy 09

- created_utc: 2026-05-13T15:26:57.005580+00:00
- min_total_score: 0.5735

## Policy

| domain | selected | score | incumbent | delta | mAP50 | delta mAP50 |
|---|---|---:|---:|---:|---:|---:|
| citystreet_day | r004_scaled04_best00_rand008_050 | 0.6310 | 0.6309 | +0.0001 | 0.506 | +0.000 |
| citystreet_night | r003_rand003 | 0.4656 | 0.4646 | +0.0010 | 0.377 | +0.001 |
| highway_day | r006_scaled04_best01_rand000_050 | 0.5487 | 0.5476 | +0.0010 | 0.442 | +0.001 |
| highway_night | r006_scaled04_best01_rand000_050 | 0.3854 | 0.3842 | +0.0012 | 0.309 | +0.001 |
| residential_day | r006_scaled04_best01_rand000_050 | 0.6720 | 0.6715 | +0.0004 | 0.538 | +0.000 |
| residential_night | r003_tiny_all_s | 0.4785 | 0.4774 | +0.0010 | 0.390 | +0.001 |

## Summary

| policy | mean | day | night | worst | DRO | night mAP50 |
|---|---:|---:|---:|---:|---:|---:|
| incumbent_r002 | 0.5294 | 0.6167 | 0.4421 | 0.3842 | 0.4654 | 0.358 |
| domain_router_oracle | 0.5302 | 0.6172 | 0.4432 | 0.3854 | 0.4664 | 0.359 |