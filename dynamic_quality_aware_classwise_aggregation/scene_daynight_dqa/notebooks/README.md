# Scene-Daynight DQA Notebooks

Use these notebooks in this order when checking the `01` series:

| order | notebook | role |
| ---: | --- | --- |
| 1 | `01_repair_oriented_scene_daynight_dqa.ipynb` | Original repair-oriented DQA loop. |
| 2 | `01_0_repair_baseline_comparison.ipynb` | Baseline controls. Treat `repair_only` as the main comparison point. |
| 3 | `01_1_dqa_diagnostic_sweep.ipynb` | DQA-only improvement sweep using constrained fixed-pseudoGT client updates. |
| 4 | `01_2_ssod_pivot_dqa.ipynb` | DQA-only SSOD pivot using target images through EfficientTeacher-style training. |
| 5 | `02_head_to_full_long_dqa.ipynb` | Main next hypothesis: long Phase1 head-only DQA followed by short Phase2 full-model DQA. |
| 6 | `03_main_bn_residual_dqa_experiment.ipynb` | Main comparison: warmup, warmup+server repair, and BN-residual DQA+server repair. |
| 7 | `04_repair_shielded_local_expert_dqa.ipynb` | Reuses 03 checkpoints and evaluates repair-shielded local expert DQA candidates against the 03 table. |
| 8 | `05_expert_choice_pseudogt_router_dqa.ipynb` | Non-residual DQA: Expert-Choice balanced pseudoGT selection before client training. |
| 9 | `06_counterfactual_output_moe_dqa.ipynb` | Production 06 experiment: counterfactual-view pseudoGT experts plus output-space MoE, without residual checkpoint mixing. |
| 10 | `07_shared_soft_head_moe_dqa.ipynb` | Shared-detector soft Head-MoE: route-specific pseudoGT head deltas are softly mixed into the 03 shared checkpoint. |

Current conclusion: none of the `01_1` or `01_2` variants beat the final
`01_0/repair_only` repaired mAP50:95.  See
`../01_EXPERIMENT_INDEX.md` for the result table and interpretation.

The next active hypothesis is now the production 07 notebook.  06 showed that
counterfactual pseudoGT routing contains a night-domain signal, but independent
detector experts and output-space fusion were too weak.  07 keeps the 03 shared
detector as the trunk and injects route-specific expertise as soft head/neck
deltas instead.
