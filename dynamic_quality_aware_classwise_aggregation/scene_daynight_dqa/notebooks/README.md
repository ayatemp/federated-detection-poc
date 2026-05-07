# Scene-Daynight DQA Notebooks

Use these notebooks in this order when checking the `01` series:

| order | notebook | role |
| ---: | --- | --- |
| 1 | `01_repair_oriented_scene_daynight_dqa.ipynb` | Original repair-oriented DQA loop. |
| 2 | `01_0_repair_baseline_comparison.ipynb` | Baseline controls. Treat `repair_only` as the main comparison point. |
| 3 | `01_1_dqa_diagnostic_sweep.ipynb` | DQA-only improvement sweep using constrained fixed-pseudoGT client updates. |
| 4 | `01_2_ssod_pivot_dqa.ipynb` | DQA-only SSOD pivot using target images through EfficientTeacher-style training. |
| 5 | `02_head_to_full_long_dqa.ipynb` | Main next hypothesis: long Phase1 head-only DQA followed by short Phase2 full-model DQA. |

Current conclusion: none of the `01_1` or `01_2` variants beat the final
`01_0/repair_only` repaired mAP50:95.  See
`../01_EXPERIMENT_INDEX.md` for the result table and interpretation.

The next active hypothesis is in `02_head_to_full_long_dqa.ipynb`.
