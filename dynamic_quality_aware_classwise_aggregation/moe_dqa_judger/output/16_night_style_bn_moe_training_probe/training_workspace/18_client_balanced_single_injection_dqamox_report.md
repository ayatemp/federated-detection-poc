# 18 Fixed-Path Localization-Curriculum Full-From-Warmup DQA-MoX Report

- created_utc: 2026-05-14T04:47:33.976952+00:00
- protocol: `scene_daynight_dqa_18_client_balanced_single_injection_dqamox_v1`
- workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/16_night_style_bn_moe_training_probe/training_workspace`
- target_map50: 0.550
- experts: K=4, top_k=2, temperature=1.38
- schedule: warmup 50 epochs, repair baseline 0 rounds, phase1 1 rounds, phase2 0 rounds

## Metrics

| condition | mAP50 | mAP50:95 | delta vs repair | worst split | worst mAP50:95 |
|---|---:|---:|---:|---|---:|
| warmup | 0.460000 | 0.259000 | 0.058000 | highway_night | 0.173000 |
| warmup + server repair | 0.378000 | 0.201000 | 0.000000 | highway_night | 0.134000 |
| warmup + fixed-pseudo-label-path DQA-MoX aggregate | 0.459000 | 0.258000 | 0.057000 | highway_night | 0.174000 |
| warmup + fixed-pseudo-label-path DQA-MoX + server repair | 0.459000 | 0.258000 | 0.057000 | highway_night | 0.174000 |

## Interpretation Hooks

- When router specialization is enabled, client/domain/class assignment is written to `18_p*_router_specialization.csv` and gated by DQA pseudoGT quality.
- The new part is pseudoGT selection: expert-choice buckets reduce class imbalance before client training and before DQA statistics.
- DQA remains in the selected pseudoGT statistics and classwise server-anchored aggregation; MoE remains inside the detector head.
- The key comparison is `latent_dqamox_final_repair` vs `warmup_server_repair_final` on total and each scene/day-night split.
- The target for this run is to push final total mAP50 to at least the configured target.

## Run Manifest

```json
{
  "created_utc": "2026-05-14T04:29:40.830573+00:00",
  "protocol": "scene_daynight_dqa_18_client_balanced_single_injection_dqamox_v1",
  "workspace": "/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/16_night_style_bn_moe_training_probe/training_workspace",
  "estimated_runtime": "4h14m00s",
  "architecture": {
    "head": "LatentMoEYoloV5",
    "num_experts": 4,
    "top_k": 2,
    "router_temperature": 1.38,
    "moe_scale": 1.0,
    "router_balance_weight": 0.018,
    "router_entropy_weight": 0.0025,
    "router_specialization_weight": 0.105,
    "router_specialization_map": "domain4",
    "router_specialization_min_quality": 0.62,
    "router_specialization_min_boxes": 220.0,
    "router_specialization_max_weight": 0.07,
    "router_specialization_class_threshold": 0.28,
    "class_skew_residual": false,
    "class_skew_orthogonal_weight": 0.0,
    "class_skew_srip_weight": 0.0,
    "class_skew_residual_weight": 0.0,
    "expert_semantics": "DQA-gated client/domain/class specialization when router_specialization_weight > 0"
  },
  "schedule": {
    "warmup_epochs": 50,
    "repair_baseline_rounds": 0,
    "phase1_rounds": 1,
    "phase2_rounds": 0,
    "post_dqa_repair_rounds": 0,
    "client_sampling_ratio": 1.0,
    "client_sampling_seed": 160069,
    "phase1_train_scope": "bn_moe_head",
    "phase2_train_scope": "all",
    "curriculum_start_round": 999,
    "late_phase1_client_lr": 0.0005,
    "late_phase1_source_repeat": 2,
    "late_phase1_pseudo_repeat": 2,
    "late_phase1_loss_box": 0.0005,
    "client_loss_cls": 0.42,
    "client_loss_obj": 0.85,
    "server_repair_loss_cls": null,
    "server_repair_loss_obj": null
  },
  "target": {
    "metric": "paper_protocol_total_map50",
    "target_map50": 0.55
  },
  "pseudo_selection": {
    "method": "expert_choice_fedmox_full_balanced",
    "imgsz": 640,
    "pseudo_imgsz": 1152,
    "pseudo_teacher_checkpoints": [],
    "use_local_ema_teacher": true,
    "local_ema_teacher_role": "selected clients persist one local EMA teacher across rounds; server anchor is used only as fallback/comparison and is not stored as a client model",
    "expert_count": 4,
    "keep_fraction": 0.5,
    "max_class_fraction": 0.16,
    "actual_max_class_fraction": 0.24,
    "load_bias_strength": 0.45,
    "late_keep_fraction": 0.6,
    "late_max_class_fraction": 0.22,
    "late_actual_max_class_fraction": 0.28,
    "late_min_score": 0.24,
    "late_min_stability": 0.68,
    "learned_quality_pseudogt": {
      "enabled": false,
      "model": "",
      "role": "pseudoGT verifier only; replaces pseudo box score before expert-choice selection and DQA stats"
    }
  },
  "style_source_adaptation": {
    "enabled": true,
    "method": "FDA target-style source-GT replay",
    "repeat": 1,
    "source_limit": 1600,
    "beta": 0.0035,
    "imgsz": 640,
    "role": "client target appearance is injected into source images while source GT boxes remain the only supervised labels"
  },
  "post_dqa_consolidation": {
    "enabled": false,
    "rounds": 0,
    "train_scope": "neck_head",
    "lr": 0.0007,
    "loss_box": 0.05,
    "loss_cls": null,
    "loss_obj": null,
    "reason": "keep the early DQA/MoE specialization but stop repeated pseudoGT self-training drift"
  },
  "client_balanced_dqa_stats": {
    "enabled": true,
    "target": "max",
    "max_scale": 1.8,
    "reason": "night and rare-scene clients have fewer selected pseudo boxes, so raw count-weighted DQA can let easy day/citystreet clients dominate aggregation"
  },
  "aggregation_curriculum": {
    "early_server_anchor": 1.02,
    "early_min_server_alpha": 0.94,
    "early_residual_blend": 0.0,
    "moe_expert_blend": 0.085,
    "moe_router_blend": 0.2,
    "bn_blend": 0.16,
    "late_server_anchor": 0.35,
    "late_min_server_alpha": 0.35,
    "late_residual_blend": 0.08
  },
  "server": {
    "weather": "cloudy represented by BDD100K Kaggle weather='partly cloudy'",
    "train_list": "/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/16_night_style_bn_moe_training_probe/training_workspace/data_lists/server_cloudy_train.txt",
    "val_list": "/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/16_night_style_bn_moe_training_probe/training_workspace/data_lists/paper_eval_scene_daynight_total_val.txt",
    "source_val_list": "/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/16_night_style_bn_moe_training_probe/training_workspace/data_lists/server_cloudy_val.txt",
    "train_images": 4881,
    "val_images": 9087,
    "source_val_images": 738,
    "validation_target": "scene_daynight_total"
  },
  "clients": [
    {
      "id": 1,
      "name": "highway_night",
      "weather": "highway_night",
      "scene": "highway",
      "timeofday": "night"
    },
    {
      "id": 5,
      "name": "residential_night",
      "weather": "residential_night",
      "scene": "residential",
      "timeofday": "night"
    }
  ],
  "actual_runtime_seconds": 587.4984263638034,
  "actual_runtime_hms": "9m47s",
  "records": [
    {
      "condition": "warmup",
      "label": "warmup_global",
      "kind": "warmup",
      "phase": "",
      "round": "",
      "client": "",
      "variant": "",
      "path": "/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/16_night_style_bn_moe_training_probe/training_workspace/checkpoints/round000_latent_dqamox_warmup.pt"
    },
    {
      "condition": "latent_dqamox",
      "label": "latent_dqamox_p1_round001_client1_highway_night",
      "kind": "client",
      "phase": "1",
      "round": "1",
      "client": "client1_highway_night",
      "variant": "scope=bn_moe_head:lr=8.5e-05:source=2:pseudo=0:box=0.004:router_target=1:router_weight=0.064793",
      "path": "/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/16_night_style_bn_moe_training_probe/training_wor
```
