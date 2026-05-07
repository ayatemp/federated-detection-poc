# 01 Experiment Index

This file organizes the completed `01` series for the scene/daynight DQA line.
The short version is:

- `01` showed that the repair-oriented loop can recover performance over rounds.
- `01_0` showed that most of that gain is explained by repeated supervised server repair.
- `01_1` tested constrained fixed-pseudoGT DQA variants, but none beat repair-only.
- `01_2` replaced fixed pseudoGT supervision with SSOD-style client training, but it also did not beat repair-only.

## Notebook Map

| notebook | output | runner | purpose | default status |
| --- | --- | --- | --- | --- |
| `notebooks/01_repair_oriented_scene_daynight_dqa.ipynb` | `output/01_repair_oriented_scene_daynight_dqa/` | `scripts/run_scene_daynight_dqa_01.py` | First repair-oriented DQA loop with 6 scene/daynight clients. | Completed; superseded by controls. |
| `notebooks/01_0_repair_baseline_comparison.ipynb` | `output/01_0_repair_baseline_comparison/` | `scripts/run_scene_daynight_dqa_01_0.py` | Matched controls: repair-only, pseudo FedAvg, pseudo DQA. | Completed; main baseline. |
| `notebooks/01_1_dqa_diagnostic_sweep.ipynb` | `output/01_1_dqa_diagnostic_sweep/` | `scripts/run_scene_daynight_dqa_01_1.py` | Improvement-only DQA sweep with constrained fixed-pseudoGT updates. | Completed; no final gain over repair-only. |
| `notebooks/01_2_ssod_pivot_dqa.ipynb` | `output/01_2_ssod_pivot/` | `scripts/run_scene_daynight_dqa_01_2.py` | SSOD client-training pivot; stable pseudo boxes are used for DQA reliability, not fixed labels. | Completed; no final gain over repair-only. |

## Final-Round Summary

Primary comparison should use final-round repaired `scene_daynight_total`
paper-protocol metrics, not the shorter normal validation printed during
training.

| experiment | condition | aggregate mAP50:95 | repaired mAP50 | repaired mAP50:95 | worst split | worst mAP50:95 | night avg mAP50:95 | note |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `01` | original DQA loop | 0.210 | 0.381 | 0.212 | n/a | n/a | n/a | Initial positive-looking result. |
| `01_0` | `repair_only` | n/a | 0.381 | 0.212 | highway_night | 0.152 | 0.173 | Main control baseline. |
| `01_0` | `pseudo_fedavg` | 0.196 | 0.379 | 0.212 | highway_night | 0.143 | 0.164 | Worse aggregate/worst split, repaired total ties. |
| `01_0` | `pseudo_dqa` | 0.210 | 0.381 | 0.211 | highway_night | 0.151 | 0.173 | DQA preserves aggregate better than FedAvg, but does not beat repair-only. |
| `01_1` | `dqa_current` | 0.210 | 0.381 | 0.212 | highway_night | 0.151 | 0.173 | Ties repair-only. |
| `01_1` | `dqa_head_lowbox` | 0.210 | 0.381 | 0.212 | highway_night | 0.152 | 0.174 | Best worst/night among 01_1, still no total gain. |
| `01_1` | `dqa_nonbackbone_lowbox` | 0.210 | 0.381 | 0.212 | highway_night | 0.152 | 0.174 | Similar to head-only. |
| `01_1` | `dqa_source_light` | 0.210 | 0.381 | 0.212 | highway_night | 0.152 | 0.173 | Source-light did not add final gain. |
| `01_1` | `dqa_target_double` | 0.210 | 0.381 | 0.212 | highway_night | 0.151 | 0.172 | Extra target pseudoGT did not help. |
| `01_2` | `ssod_dqa` | 0.210 | 0.381 | 0.212 | highway_night | 0.152 | 0.173 | SSOD pivot ties repair-only. |
| `01_2` | `ssod_dqa_head` | 0.210 | 0.381 | 0.212 | highway_night | 0.152 | 0.173 | SSOD head-only ties repair-only. |
| `01_2` | `ssod_dqa_nonbackbone` | 0.210 | 0.381 | 0.211 | highway_night | 0.152 | 0.173 | Slightly lower final repaired mAP50:95. |

## What The 01 Series Says

1. Server repair is doing most of the final mAP recovery.
   `repair_only` reaches `mAP50=0.381` and `mAP50:95=0.212`, which is the same
   final level reached by the DQA and SSOD-DQA variants.

2. DQA is useful as a protection mechanism before repair.
   In `01_0`, `pseudo_fedavg` ends with aggregate `mAP50:95=0.196`, while
   DQA-style aggregation reaches about `0.210`.  This is the strongest positive
   signal for DQA in this series.

3. DQA has not yet shown post-repair target-domain gain.
   After source repair, all 01_1 and 01_2 variants collapse to almost the same
   final total mAP as `repair_only`.

4. Worst-split and night metrics remain the key place to look.
   The final worst split is consistently `highway_night`, around
   `mAP50:95=0.151-0.152`.  Any next method should be judged by whether it
   improves this split without lowering total mAP.

## Current Interpretation

The 01 series does not support the claim that the current DQA client adaptation
improves final repaired average mAP beyond source repair.  It does support a
more conservative claim: DQA can make pseudoGT/SSOD client aggregation less
destructive than plain FedAvg before repair.

The next experiment should therefore change the objective from "more pseudoGT
client learning" to "make client updates produce target-domain information that
survives source repair."  Good candidates are:

- repair loss that preserves target consistency instead of pure source GT repair
- explicit worst-split/night balancing during aggregation
- class/scene reliability measured on unlabeled target consistency rather than confidence-only pseudo boxes
- a baseline that extends repair-only rounds to verify the plateau is real

The first follow-up from this conclusion is `02_target_consistency_repair_dqa`,
which keeps SSOD-DQA client adaptation but changes server repair to source GT
plus weak target consistency.

## Result Files

| file | role |
| --- | --- |
| `output/01_repair_oriented_scene_daynight_dqa/stats/01_round_metrics.csv` | Initial 01 round metrics. |
| `output/01_0_repair_baseline_comparison/stats/01_0_all_condition_metrics.csv` | Main repair-only / FedAvg / DQA control table. |
| `output/01_1_dqa_diagnostic_sweep/stats/01_1_all_condition_metrics.csv` | DQA diagnostic sweep table. |
| `output/01_2_ssod_pivot/stats/01_2_all_condition_metrics.csv` | SSOD pivot table. |
| `output/01_1_01_2_combined_report.md` | Combined 01_1 + 01_2 discussion sent to Discord. |
