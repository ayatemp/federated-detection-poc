# 18 Fixed-Path Localization-Curriculum Full-From-Warmup DQA-MoX Report

- created_utc: 2026-05-14T06:08:50.155415+00:00
- protocol: `scene_daynight_dqa_18_client_balanced_single_injection_dqamox_v1`
- workspace: `/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/18_bn_only_pseudo_softmix_probe/training_workspace`
- target_map50: 0.600
- experts: K=4, top_k=2, temperature=1.45
- schedule: warmup 50 epochs, repair baseline 0 rounds, phase1 1 rounds, phase2 0 rounds

## Metrics

| condition | mAP50 | mAP50:95 | delta vs repair | worst split | worst mAP50:95 |
|---|---:|---:|---:|---|---:|
| warmup | 0.499000 | 0.280000 | 0.064000 | highway_night | 0.173000 |
| warmup + server repair | 0.404000 | 0.216000 | 0.000000 | highway_night | 0.133000 |
| warmup + fixed-pseudo-label-path DQA-MoX aggregate | 0.492000 | 0.274000 | 0.058000 | highway_night | 0.167000 |
| warmup + fixed-pseudo-label-path DQA-MoX + server repair | 0.492000 | 0.274000 | 0.058000 | highway_night | 0.167000 |

## Interpretation Hooks

- When router specialization is enabled, client/domain/class assignment is written to `18_p*_router_specialization.csv` and gated by DQA pseudoGT quality.
- The new part is pseudoGT selection: expert-choice buckets reduce class imbalance before client training and before DQA statistics.
- DQA remains in the selected pseudoGT statistics and classwise server-anchored aggregation; MoE remains inside the detector head.
- The key comparison is `latent_dqamox_final_repair` vs `warmup_server_repair_final` on total and each scene/day-night split.
- The target for this run is to push final total mAP50 to at least the configured target.

## Run Manifest

```json
{
  "created_utc": "2026-05-14T05:49:03.519171+00:00",
  "protocol": "scene_daynight_dqa_18_client_balanced_single_injection_dqamox_v1",
  "workspace": "/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/18_bn_only_pseudo_softmix_probe/training_workspace",
  "estimated_runtime": "4h14m00s",
  "architecture": {
    "head": "LatentMoEYoloV5",
    "num_experts": 4,
    "top_k": 2,
    "router_temperature": 1.45,
    "moe_scale": 1.0,
    "router_balance_weight": 0.02,
    "router_entropy_weight": 0.003,
    "router_specialization_weight": 0.05,
    "router_specialization_map": "domain4",
    "router_specialization_min_quality": 0.65,
    "router_specialization_min_boxes": 180.0,
    "router_specialization_max_weight": 0.035,
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
    "client_sampling_seed": 180181,
    "phase1_train_scope": "bn",
    "phase2_train_scope": "all",
    "curriculum_start_round": 999,
    "late_phase1_client_lr": 0.0005,
    "late_phase1_source_repeat": 2,
    "late_phase1_pseudo_repeat": 2,
    "late_phase1_loss_box": 0.0005,
    "client_loss_cls": 0.2,
    "client_loss_obj": 0.45,
    "server_repair_loss_cls": null,
    "server_repair_loss_obj": null
  },
  "target": {
    "metric": "paper_protocol_total_map50",
    "target_map50": 0.6
  },
  "pseudo_selection": {
    "method": "expert_choice_fedmox_full_balanced",
    "imgsz": 960,
    "pseudo_imgsz": 1280,
    "pseudo_teacher_checkpoints": [],
    "use_local_ema_teacher": true,
    "local_ema_teacher_role": "selected clients persist one local EMA teacher across rounds; server anchor is used only as fallback/comparison and is not stored as a client model",
    "expert_count": 4,
    "keep_fraction": 0.5,
    "max_class_fraction": 0.14,
    "actual_max_class_fraction": 0.22,
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
    "enabled": false,
    "method": "FDA target-style source-GT replay",
    "repeat": 0,
    "source_limit": 1,
    "beta": 0.0,
    "imgsz": 960,
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
    "max_scale": 1.4,
    "reason": "night and rare-scene clients have fewer selected pseudo boxes, so raw count-weighted DQA can let easy day/citystreet clients dominate aggregation"
  },
  "aggregation_curriculum": {
    "early_server_anchor": 1.0,
    "early_min_server_alpha": 0.97,
    "early_residual_blend": 0.0,
    "moe_expert_blend": 0.0,
    "moe_router_blend": 0.0,
    "bn_blend": 0.38,
    "late_server_anchor": 0.35,
    "late_min_server_alpha": 0.35,
    "late_residual_blend": 0.08
  },
  "server": {
    "weather": "cloudy represented by BDD100K Kaggle weather='partly cloudy'",
    "train_list": "/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/18_bn_only_pseudo_softmix_probe/training_workspace/data_lists/server_cloudy_train.txt",
    "val_list": "/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/18_bn_only_pseudo_softmix_probe/training_workspace/data_lists/paper_eval_scene_daynight_total_val.txt",
    "source_val_list": "/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/18_bn_only_pseudo_softmix_probe/training_workspace/data_lists/server_cloudy_val.txt",
    "train_images": 4881,
    "val_images": 9087,
    "source_val_images": 738,
    "validation_target": "scene_daynight_total"
  },
  "clients": [
    {
      "id": 0,
      "name": "highway_day",
      "weather": "highway_day",
      "scene": "highway",
      "timeofday": "daytime"
    },
    {
      "id": 1,
      "name": "highway_night",
      "weather": "highway_night",
      "scene": "highway",
      "timeofday": "night"
    },
    {
      "id": 2,
      "name": "citystreet_day",
      "weather": "citystreet_day",
      "scene": "city street",
      "timeofday": "daytime"
    },
    {
      "id": 3,
      "name": "citystreet_night",
      "weather": "citystreet_night",
      "scene": "city street",
      "timeofday": "night"
    },
    {
      "id": 4,
      "name": "residential_day",
      "weather": "residential_day",
      "scene": "residential",
      "timeofday": "daytime"
    },
    {
      "id": 5,
      "name": "residential_night",
      "weather": "residential_night",
      "scene": "residential",
      "timeofday": "night"
    }
  ],
  "actual_runtime_seconds": 502.7818723358214,
  "actual_runtime_hms": "8m22s",
  "records": [
    {
      "condition": "warmup",
      "label": "warmup_global",
      "kind": "warmup",
      "phase": "",
      "round": "",
      "client": "",
      "variant": "",
      "path": "/app/Object_Detection/dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/18_bn_only_pseudo_sof
```
