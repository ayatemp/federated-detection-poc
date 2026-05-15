# DQA-SoftMoX Mix Judger Policy 03

- created_utc: 2026-05-13T13:04:22.563659+00:00
- training_rows: 206
- full_training_rows: 19
- model_type: extratrees

## Leave-One-Round Full-Candidate CV

| round | selected | best | selected score | best score | regret |
|---:|---|---|---:|---:|---:|
| 1 | best00_sur01_00 | best01_prior02 | 0.5725 | 0.5746 | 0.0021 |
| 2 | best01_sur00_02 | best00_prior07 | 0.5735 | 0.5746 | 0.0010 |
| 3 | best00_prior00 | best02_sur00_01 | 0.5710 | 0.5710 | 0.0000 |
| 4 | best02_sur00_01 | best00_prior00 | 0.5656 | 0.5656 | 0.0000 |
| 5 | best02_prior01 | best00_prior00 | 0.5596 | 0.5597 | 0.0001 |

## Selected Policy Weights

| round | pred | body G/A/S | head G/A/S | moe G/A/S | pool | guard |
|---:|---:|---|---|---|---:|---|
| 1 | 0.5745 | 0.00/0.00/1.00 | 0.00/0.00/1.00 | 0.00/0.00/1.00 | 2101 |  |
| 2 | 0.5745 | 0.65/0.25/0.10 | 0.20/0.10/0.70 | 0.15/0.75/0.10 | 2089 |  |
| 3 | 0.5709 | 1.00/0.00/0.00 | 1.00/0.00/0.00 | 1.00/0.00/0.00 | 2089 | g_beats_children |
| 4 | 0.5657 | 1.00/0.00/0.00 | 1.00/0.00/0.00 | 1.00/0.00/0.00 | 2087 | repair_hurts_both |
| 5 | 0.5598 | 1.00/0.00/0.00 | 1.00/0.00/0.00 | 1.00/0.00/0.00 | 2090 | repair_hurts_both |

## Selected Full-Total Evaluation

| round | mAP50 | mAP50:95 | precision | recall | score |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.462 | 0.260 | 0.694 | 0.431 | 0.5746 |
| 2 | 0.462 | 0.260 | 0.694 | 0.431 | 0.5746 |
| 3 | 0.459 | 0.258 | 0.684 | 0.433 | 0.5710 |
| 4 | 0.455 | 0.255 | 0.690 | 0.428 | 0.5656 |
| 5 | 0.450 | 0.253 | 0.686 | 0.424 | 0.5597 |