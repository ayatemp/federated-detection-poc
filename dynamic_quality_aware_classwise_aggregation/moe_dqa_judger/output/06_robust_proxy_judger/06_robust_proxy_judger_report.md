# DQA-SoftMoX Robust Proxy Judger 06

- created_utc: 2026-05-13T14:38:11.446498+00:00
- candidate_count: 44
- mini_splits: 3
- mini_images: 384
- calibrator_train_count: 25
- calibrator_loo_mae: 0.00055

## Selected Policy

| round | label | source | role | mean proxy | std | LCB | pred full | full mAP50 | full mAP50:95 | full score |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | warmup_g0 | source | warmup | 0.6219 | 0.0389 | 0.5927 | 0.5692 | 0.458 | 0.255 | 0.5683 |
| 1 | 03_mix_judger_policy_r001_prior02 | 03_mix_judger_policy | learned_mix | 0.6287 | 0.0431 | 0.5964 | 0.5745 | 0.462 | 0.260 | 0.5746 |
| 2 | 03_mix_judger_policy_r002_prior07 | 03_mix_judger_policy | learned_mix | 0.6279 | 0.0415 | 0.5968 | 0.5745 | 0.462 | 0.260 | 0.5746 |
| 3 | 02_mix_weight_optimizer_expanded_r003_best02_sur00_01 | 02_mix_weight_optimizer_expanded | learned_mix | 0.6196 | 0.0441 | 0.5865 | 0.5710 | 0.459 | 0.258 | 0.5710 |
| 4 | 04_delta_expert_optimizer_r004_best01_rand000 | 04_delta_expert_optimizer | learned_mix | 0.6165 | 0.0397 | 0.5867 | 0.5671 | 0.456 | 0.256 | 0.5671 |
| 5 | r005_a | source | a | 0.6071 | 0.0396 | 0.5774 | 0.5604 | 0.450 | 0.253 | 0.5596 |
| 6 | r006_a | source | a | 0.5996 | 0.0369 | 0.5719 | 0.5599 | 0.445 | 0.250 | 0.5528 |

## Top Robust Candidates

| rank | round | label | source | role | LCB | pred full | known full |
|---:|---:|---|---|---|---:|---:|---:|
| 1 | 1 | 03_mix_judger_policy_r001_prior02 | 03_mix_judger_policy | learned_mix | 0.5964 | 0.5745 | 0.5746 |
| 2 | 2 | 03_mix_judger_policy_r002_prior07 | 03_mix_judger_policy | learned_mix | 0.5968 | 0.5745 | 0.5746 |
| 3 | 2 | 04_delta_expert_optimizer_r002_best00_sur00_01 | 04_delta_expert_optimizer | learned_mix | 0.5963 | 0.5745 | 0.5746 |
| 4 | 2 | r002_g | source | g | 0.5971 | 0.5745 | nan |
| 5 | 2 | 04_delta_expert_optimizer_r002_best01_sur00_00 | 04_delta_expert_optimizer | learned_mix | 0.5963 | 0.5745 | 0.5745 |
| 6 | 1 | r001_s | source | s | 0.5971 | 0.5745 | nan |
| 7 | 2 | r002_a | source | a | 0.5960 | 0.5745 | nan |
| 8 | 2 | 02_mix_weight_optimizer_expanded_r002_best00_prior07 | 02_mix_weight_optimizer_expanded | learned_mix | 0.5968 | 0.5745 | 0.5746 |
| 9 | 2 | 02_mix_weight_optimizer_expanded_r002_best02_sur01_01 | 02_mix_weight_optimizer_expanded | learned_mix | 0.5969 | 0.5745 | 0.5745 |
| 10 | 1 | 02_mix_weight_optimizer_expanded_r001_best02_sur01_01 | 02_mix_weight_optimizer_expanded | learned_mix | 0.5976 | 0.5740 | 0.5739 |
| 11 | 1 | 02_mix_weight_optimizer_expanded_r001_best01_sur01_03 | 02_mix_weight_optimizer_expanded | learned_mix | 0.5968 | 0.5739 | 0.5739 |
| 12 | 2 | 02_mix_weight_optimizer_expanded_r002_best01_sur00_02 | 02_mix_weight_optimizer_expanded | learned_mix | 0.5968 | 0.5736 | 0.5735 |
| 13 | 1 | 04_delta_expert_optimizer_r001_best00_sur00_00 | 04_delta_expert_optimizer | learned_mix | 0.5964 | 0.5735 | 0.5735 |
| 14 | 1 | 04_delta_expert_optimizer_r001_best01_rand007 | 04_delta_expert_optimizer | learned_mix | 0.5959 | 0.5735 | 0.5735 |
| 15 | 1 | 02_mix_weight_optimizer_expanded_r001_best00_sur01_00 | 02_mix_weight_optimizer_expanded | learned_mix | 0.5952 | 0.5725 | 0.5725 |
| 16 | 2 | r002_s | source | s | 0.5865 | 0.5714 | nan |
| 17 | 3 | 02_mix_weight_optimizer_expanded_r003_best02_sur00_01 | 02_mix_weight_optimizer_expanded | learned_mix | 0.5865 | 0.5710 | 0.5710 |
| 18 | 3 | r003_g | source | g | 0.5865 | 0.5710 | nan |
| 19 | 3 | 02_mix_weight_optimizer_expanded_r003_best00_prior00 | 02_mix_weight_optimizer_expanded | learned_mix | 0.5865 | 0.5710 | 0.5710 |
| 20 | 3 | 02_mix_weight_optimizer_expanded_r003_best01_sur00_00 | 02_mix_weight_optimizer_expanded | learned_mix | 0.5864 | 0.5710 | 0.5710 |