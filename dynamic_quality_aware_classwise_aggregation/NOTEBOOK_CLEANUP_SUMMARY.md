# Notebook Cleanup Summary

- 作成日時: 2026-05-17 12:39 UTC
- 対象: `dynamic_quality_aware_classwise_aggregation/**/*.ipynb`
- 方針: ノートブック自体は残し、コードセルとMarkdownセルも保持。実行出力、execution count、widget state metadataだけを削除。
- スキャンしたノートブック数: 160
- 軽量化したノートブック数: 75
- 軽量化前の合計サイズ: 80.2 MB
- 軽量化後の合計サイズ: 1.8 MB
- 削減量: 78.4 MB

## 今回整理したこと

今回は、過去実験の`output/`ディレクトリは削除していません。チェックポイント、ログ、評価値が論文用の比較で必要になる可能性があるためです。代わりに、IDEで開くと重くなりやすいノートブックの実行出力だけを消し、コードと説明は残しました。

現在の主実験は以下に集約されています。

- `dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/notebooks/04_canonical_fixed_dqa_upcycled_moe_v2.ipynb`
- `dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/scripts/run_dqa_anonymous_backbone_moe.py`
- `dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/04_canonical_fixed_dqa_upcycled_moe_v2/`
- 過去結果の要約: `dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/04_canonical_fixed_dqa_upcycled_moe_v2/previous_results_archive/README.md`

## ディレクトリ別のノートブックサイズ

| Area | Notebooks | Changed | Before | After | Saved |
|---|---:|---:|---:|---:|---:|
| `dynamic_quality_aware_classwise_aggregation/notebook` | 31 | 17 | 47.9 MB | 987.4 KB | 47.0 MB |
| `dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa` | 80 | 32 | 30.8 MB | 561.3 KB | 30.3 MB |
| `dynamic_quality_aware_classwise_aggregation/moe_dqa_judger` | 37 | 21 | 976.2 KB | 115.3 KB | 860.9 KB |
| `dynamic_quality_aware_classwise_aggregation/exploring` | 4 | 3 | 383.2 KB | 71.4 KB | 311.7 KB |
| `dynamic_quality_aware_classwise_aggregation/source_calibrated_localization_quality` | 2 | 2 | 153.4 KB | 60.0 KB | 93.3 KB |
| `dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region` | 4 | 0 | 23.0 KB | 23.0 KB | 0 B |
| `dynamic_quality_aware_classwise_aggregation/moe_dqa` | 1 | 0 | 12.8 KB | 12.8 KB | 0 B |
| `dynamic_quality_aware_classwise_aggregation/threshold_policy_model` | 1 | 0 | 7.7 KB | 7.7 KB | 0 B |

## 削減量が大きかったノートブック

| Notebook | Before | After | Saved |
|---|---:|---:|---:|
| `dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa/notebooks/03_main_bn_residual_dqa_experiment.ipynb` | 12.8 MB | 13.0 KB | 12.7 MB |
| `dynamic_quality_aware_classwise_aggregation/notebook/08_3_dqa_scene_phase2_update_gating_sweep_7h.ipynb` | 12.0 MB | 31.7 KB | 12.0 MB |
| `dynamic_quality_aware_classwise_aggregation/notebook/07_dqa_scene_learned_adaptive_policy_8h.ipynb` | 9.1 MB | 25.3 KB | 9.1 MB |
| `dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa/notebooks/02_head_to_full_long_dqa.ipynb` | 8.7 MB | 10.4 KB | 8.7 MB |
| `dynamic_quality_aware_classwise_aggregation/notebook/08_dqa_scene_tri_stage_pseudogt_policy_8h.ipynb` | 8.5 MB | 26.7 KB | 8.5 MB |
| `dynamic_quality_aware_classwise_aggregation/notebook/08_2_dqa_scene_phase2_head_protected_policy.ipynb` | 5.8 MB | 27.3 KB | 5.7 MB |
| `dynamic_quality_aware_classwise_aggregation/notebook/05_dqa_scene_class_profile_5h.ipynb` | 3.8 MB | 22.1 KB | 3.8 MB |
| `dynamic_quality_aware_classwise_aggregation/notebook/03_2_dqa_cwa_corrected_12h_evaluation.ipynb` | 2.6 MB | 63.6 KB | 2.6 MB |
| `dynamic_quality_aware_classwise_aggregation/notebook/04_3_2_dqa_ver2_scene_12h_evaluation.ipynb` | 2.1 MB | 58.4 KB | 2.1 MB |
| `dynamic_quality_aware_classwise_aggregation/notebook/04_2_dqa_ver2_evaluation.ipynb` | 2.1 MB | 54.5 KB | 2.0 MB |
| `dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa/aggressive_dqamox/notebooks/research_loop_until_060/017_27w_class_channel_moe.ipynb` | 1.9 MB | 3.3 KB | 1.9 MB |
| `dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa/aggressive_dqamox/notebooks/research_loop_until_060/015_27u_coco_bridge_moe.ipynb` | 1.4 MB | 2.0 KB | 1.4 MB |
| `dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa/notebooks/01_0_repair_baseline_comparison.ipynb` | 1.1 MB | 7.1 KB | 1.1 MB |
| `dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa/aggressive_dqamox/notebooks/research_loop_until_060/019_27y_split_scale_coco_moe.ipynb` | 1015.2 KB | 2.7 KB | 1012.5 KB |
| `dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa/aggressive_dqamox/notebooks/research_loop_until_060/018_27x_guarded_residual_class_moe.ipynb` | 985.2 KB | 2.7 KB | 982.4 KB |
| `dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa/notebooks/01_1_dqa_diagnostic_sweep.ipynb` | 971.7 KB | 11.3 KB | 960.4 KB |
| `dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa/notebooks/01_repair_oriented_scene_daynight_dqa.ipynb` | 876.6 KB | 7.7 KB | 868.9 KB |
| `dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa/aggressive_dqamox/notebooks/research_loop_until_060/014_27t_path_domain_routed_moe.ipynb` | 369.5 KB | 2.0 KB | 367.5 KB |
| `dynamic_quality_aware_classwise_aggregation/notebook/08_3_4_dqa_scene_phase2_feature_quality_sweep.ipynb` | 396.5 KB | 41.1 KB | 355.3 KB |
| `dynamic_quality_aware_classwise_aggregation/notebook/08_3_8_dqa_scene_phase2_rscolq_smooth_policy.ipynb` | 222.9 KB | 30.0 KB | 192.9 KB |
| `dynamic_quality_aware_classwise_aggregation/notebook/08_3_2_dqa_scene_phase2_fedsto_dqa_sweep.ipynb` | 210.4 KB | 32.8 KB | 177.6 KB |
| `dynamic_quality_aware_classwise_aggregation/notebook/08_3_7_dqa_scene_phase2_round_stable_scolq_policy.ipynb` | 199.6 KB | 28.3 KB | 171.2 KB |
| `dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa/aggressive_dqamox/notebooks/research_loop_until_060/016_27v_consensus_wbf_moe.ipynb` | 171.4 KB | 2.1 KB | 169.3 KB |
| `dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/notebooks/08_domain_slice_judger.executed.ipynb` | 145.5 KB | 3.0 KB | 142.5 KB |
| `dynamic_quality_aware_classwise_aggregation/exploring/02_3_dqa_control_sweep_6h.ipynb` | 131.8 KB | 9.5 KB | 122.3 KB |

## 軽量化後に大きいノートブック

| Notebook | Size |
|---|---:|
| `dynamic_quality_aware_classwise_aggregation/notebook/03_2_dqa_cwa_corrected_12h_evaluation.ipynb` | 63.6 KB |
| `dynamic_quality_aware_classwise_aggregation/notebook/02_3_dqa_cwa_14h_evaluation.ipynb` | 59.0 KB |
| `dynamic_quality_aware_classwise_aggregation/notebook/04_3_2_dqa_ver2_scene_12h_evaluation.ipynb` | 58.4 KB |
| `dynamic_quality_aware_classwise_aggregation/notebook/04_2_dqa_ver2_evaluation.ipynb` | 54.5 KB |
| `dynamic_quality_aware_classwise_aggregation/notebook/00_dqa_status_and_full_supervised_baseline.ipynb` | 45.8 KB |
| `dynamic_quality_aware_classwise_aggregation/notebook/08_3_4_dqa_scene_phase2_feature_quality_sweep.ipynb` | 41.1 KB |
| `dynamic_quality_aware_classwise_aggregation/notebook/08_3_3_dqa_scene_phase2_anti_drift_sweep.ipynb` | 38.9 KB |
| `dynamic_quality_aware_classwise_aggregation/source_calibrated_localization_quality/01_train_and_select_scolq.ipynb` | 36.1 KB |
| `dynamic_quality_aware_classwise_aggregation/notebook/08_3_2_dqa_scene_phase2_fedsto_dqa_sweep.ipynb` | 32.8 KB |
| `dynamic_quality_aware_classwise_aggregation/notebook/08_3_dqa_scene_phase2_update_gating_sweep_7h.ipynb` | 31.7 KB |
| `dynamic_quality_aware_classwise_aggregation/notebook/03_3_dqa_cwa_scene_12h_reproduction.ipynb` | 30.4 KB |
| `dynamic_quality_aware_classwise_aggregation/notebook/03_dqa_cwa_corrected_12h_reproduction.ipynb` | 30.2 KB |
| `dynamic_quality_aware_classwise_aggregation/notebook/08_3_8_dqa_scene_phase2_rscolq_smooth_policy.ipynb` | 30.0 KB |
| `dynamic_quality_aware_classwise_aggregation/notebook/02_2_dqa_cwa_14h_reproduction.ipynb` | 30.0 KB |
| `dynamic_quality_aware_classwise_aggregation/notebook/08_3_6_dqa_scene_phase2_scolq_policy.ipynb` | 29.0 KB |
| `dynamic_quality_aware_classwise_aggregation/notebook/08_3_7_dqa_scene_phase2_round_stable_scolq_policy.ipynb` | 28.3 KB |
| `dynamic_quality_aware_classwise_aggregation/notebook/04_3_dqa_ver2_scene_12h_reproduction.ipynb` | 27.9 KB |
| `dynamic_quality_aware_classwise_aggregation/notebook/04_dqa_ver2_reproduction.ipynb` | 27.8 KB |
| `dynamic_quality_aware_classwise_aggregation/notebook/08_2_dqa_scene_phase2_head_protected_policy.ipynb` | 27.3 KB |
| `dynamic_quality_aware_classwise_aggregation/exploring/02_warmup_continuation_patterns.ipynb` | 27.2 KB |

## まだ容量を使っている場所

ディスク容量を大きく使っているのはノートブックではなく、過去実験の出力ディレクトリです。直近のスキャンでは以下が大きいです。

- `dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa/aggressive_dqamox/output`: 約83 GB
- `dynamic_quality_aware_classwise_aggregation/moe_dqa/output`: 約32 GB
- `dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output`: 約21 GB
- `dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa/output`: 約22 GB
- `dynamic_quality_aware_classwise_aggregation/dqa_moe_trust_region/output/04_canonical_fixed_dqa_upcycled_moe_v2`: 約4.5 GB

次に本格的に容量を空けるなら、古い実験ごとに最終mAP、重要ログ、設定だけをMarkdown/CSVに保存してから、不要な`output/`をアーカイブまたは削除するのが一番効きます。
