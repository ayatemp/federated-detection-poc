# DQA Threshold Policy Model

This directory trains a small learned policy that predicts pseudoGT gates from
DQA05 client/class pseudo-label statistics.

Inputs:

- `stats_dqa05_scene_class_profile_5h/phase2_round*_client*.json`
- `efficientteacher_dqa05_scene_class_profile_5h/runs/dqa_phase2_round*_server/results.csv`

Outputs:

- `artifacts/dqa05_threshold_policy.joblib`
- `artifacts/dqa05_threshold_policy_report.json`
- `artifacts/dqa05_policy_dataset.csv`
- `artifacts/dqa05_threshold_predictions.csv`
- `artifacts/latest_policy_decision.json`

Important limitation: DQA05 used one fixed threshold schedule, so it does not
contain direct labels for the true globally optimal threshold.  The training
target is an offline oracle proxy derived from next-round server mAP drift and
pseudo-label quality/count statistics.  This is suitable as a lightweight
learned DQA policy prototype and should be validated by a follow-up DQA06/07 run.

Run:

```bash
python dynamic_quality_aware_classwise_aggregation/threshold_policy_model/train_threshold_policy.py
```

Infer from an existing stats directory:

```bash
python dynamic_quality_aware_classwise_aggregation/threshold_policy_model/infer_threshold_policy.py \
  --stats-root dynamic_quality_aware_classwise_aggregation/stats_dqa05_scene_class_profile_5h
```
