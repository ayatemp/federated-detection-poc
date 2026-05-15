# DQA-SoftMoX Incumbent-Rebased Delta Judger 07

- created_utc: 2026-05-13T15:10:08.373096+00:00
- incumbent_score: 0.5746
- mini_splits: 3
- mini_images: 384

## Accepted Policy

| round | candidate | accepted | mAP50 | mAP50:95 | score | incumbent_after | reason |
|---:|---|---:|---:|---:|---:|---:|---|
| 2 | incumbent_r002 | True | 0.462 | 0.260 | 0.5746 | 0.5746 | initial incumbent |
| 3 | rand003 | False | 0.462 | 0.260 | 0.5741 | 0.5746 | rejected; keep incumbent |
| 4 | anti_drift_head | False | 0.462 | 0.260 | 0.5746 | 0.5746 | rejected; keep incumbent |
| 5 | scaled04_best01_sur00_02_025 | False | 0.462 | 0.260 | 0.5745 | 0.5746 | rejected; keep incumbent |
| 6 | anti_drift_head | False | 0.462 | 0.260 | 0.5746 | 0.5746 | rejected; keep incumbent |

## Top Proxy Candidates

| rank | round | candidate | proxy LCB | pred full | mean score | std |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 3 | tiny_all_s | 0.5969 | 0.5735 | 0.6281 | 0.0416 |
| 2 | 3 | rand003 | 0.5969 | 0.5735 | 0.6282 | 0.0416 |
| 3 | 3 | scaled04_best01_sur00_00_025 | 0.5968 | 0.5735 | 0.6281 | 0.0417 |
| 4 | 3 | scaled04_best00_sur00_02_025 | 0.5968 | 0.5735 | 0.6281 | 0.0417 |
| 5 | 3 | target_router_only | 0.5968 | 0.5735 | 0.6279 | 0.0415 |
| 6 | 3 | tiny_all_a | 0.5968 | 0.5735 | 0.6279 | 0.0415 |
| 7 | 3 | moe_a_only | 0.5968 | 0.5735 | 0.6279 | 0.0415 |
| 8 | 3 | head_s_moe_a | 0.5968 | 0.5735 | 0.6279 | 0.0415 |
| 9 | 3 | body_frozen_head_moe | 0.5968 | 0.5735 | 0.6279 | 0.0415 |
| 10 | 3 | source_repair_only_head | 0.5968 | 0.5735 | 0.6279 | 0.0415 |
| 11 | 3 | rand000 | 0.5968 | 0.5735 | 0.6279 | 0.0415 |
| 12 | 3 | anti_drift_head | 0.5968 | 0.5735 | 0.6279 | 0.0415 |
| 13 | 3 | rand001 | 0.5968 | 0.5734 | 0.6277 | 0.0412 |
| 14 | 3 | scaled04_best00_sur00_02_050 | 0.5968 | 0.5734 | 0.6285 | 0.0422 |
| 15 | 3 | scaled04_best01_sur00_00_050 | 0.5971 | 0.5734 | 0.6285 | 0.0418 |
| 16 | 3 | rand002 | 0.5968 | 0.5732 | 0.6276 | 0.0411 |
| 17 | 4 | scaled04_best00_rand008_050 | 0.5970 | 0.5724 | 0.6283 | 0.0417 |
| 18 | 4 | anti_drift_head | 0.5969 | 0.5724 | 0.6280 | 0.0415 |
| 19 | 4 | rand003 | 0.5968 | 0.5724 | 0.6280 | 0.0416 |
| 20 | 4 | scaled04_best01_rand000_050 | 0.5972 | 0.5724 | 0.6286 | 0.0418 |
| 21 | 4 | tiny_all_a | 0.5968 | 0.5724 | 0.6279 | 0.0415 |
| 22 | 4 | moe_a_only | 0.5968 | 0.5724 | 0.6279 | 0.0415 |
| 23 | 4 | head_s_moe_a | 0.5968 | 0.5724 | 0.6279 | 0.0415 |
| 24 | 4 | body_frozen_head_moe | 0.5968 | 0.5724 | 0.6279 | 0.0415 |
| 25 | 4 | target_router_only | 0.5968 | 0.5724 | 0.6279 | 0.0415 |
| 26 | 4 | rand001 | 0.5968 | 0.5724 | 0.6279 | 0.0415 |
| 27 | 4 | source_repair_only_head | 0.5969 | 0.5724 | 0.6280 | 0.0415 |
| 28 | 4 | scaled04_best00_rand008_025 | 0.5969 | 0.5724 | 0.6279 | 0.0414 |
| 29 | 4 | rand000 | 0.5969 | 0.5724 | 0.6279 | 0.0414 |
| 30 | 4 | scaled04_best01_rand000_025 | 0.5969 | 0.5724 | 0.6280 | 0.0415 |