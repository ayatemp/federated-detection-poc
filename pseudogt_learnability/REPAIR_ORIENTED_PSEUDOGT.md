# Repair-Oriented Pseudo-GT Plan

## Motivation

The 01 and 02 experiments suggest that the main target should not be
"pseudo-GT alone improves client checkpoints."  A more realistic target is:

> Client pseudo-GT updates inject target-domain information, and supervised
> server repair turns the aggregated model back into a stable detector while
> retaining some of that target-domain gain.

This changes the primary evaluation object from client or aggregate checkpoints
to the repaired global checkpoint after each round.

## What 02 Shows

Scene-total validation results:

| checkpoint | mAP@0.5 | mAP@0.5:0.95 | interpretation |
| --- | ---: | ---: | --- |
| warmup_global | 0.352 | 0.189 | baseline |
| stable_mix_backbone_aggregate_backbone | 0.253 | 0.140 | client adaptation is too destructive |
| stable_mix_backbone_server_repair | 0.377 | 0.210 | server repair recovers and improves |
| stable_mix_all_lowlr_aggregate_all | 0.308 | 0.174 | less destructive than backbone-only aggregation |
| stable_mix_all_lowlr_server_repair | 0.381 | 0.212 | best 02 result |

Main conclusions:

- Stable pseudo labels are not automatically safe as full YOLO labels.
- Aggregate performance can drop while repaired performance improves.
- The useful signal is therefore not "the pseudo labels are clean GT."
- The useful signal is closer to "the client update moves the model toward the
  target domain, and server repair restores detector calibration/localization."
- `stable_mix_all_lowlr` is the better base than `stable_mix_backbone` because
  it loses less before repair and gives the best repaired checkpoint.

## Current Server Repair

In 02, server repair is a one-epoch supervised fine-tuning step after client
aggregation:

| item | value |
| --- | --- |
| input checkpoint | aggregate checkpoint |
| train list | `output/02_stable_pseudogt/data_lists/server_cloudy_train.txt` |
| target/pseudo data | none |
| SSOD | disabled |
| train scope | all |
| epochs | 1 |
| learning rate | 0.0008 |

The source repair set is BDD100K `weather = partly cloudy`:

| split | images | boxes | boxes/image |
| --- | ---: | ---: | ---: |
| server_cloudy_train | 4881 | 97123 | 19.90 |
| server_cloudy_val | 738 | 14937 | 20.24 |

The server train scene composition is not uniform:

| scene | images |
| --- | ---: |
| city street | 2561 |
| highway | 1705 |
| residential | 580 |

The server train time-of-day composition is strongly daytime-heavy:

| timeofday | images |
| --- | ---: |
| daytime | 4262 |
| dawn/dusk | 570 |
| night | 49 |

This means server repair is strong at restoring source-calibrated detection
behavior, but it may erase target-domain information if client updates only
modify fragile head/localization parameters.

## Client Target Difference

Client targets are scene-specific and weather-mixed.  The labels below are used
only for diagnosis, not for training.

| target | images | boxes | boxes/image | night images |
| --- | ---: | ---: | ---: | ---: |
| client0_highway | 5000 | 70965 | 14.19 | 2006 |
| client1_citystreet | 5000 | 101179 | 20.24 | 2071 |
| client2_residential | 5000 | 81698 | 16.34 | 1164 |

Class distribution differs by scene:

| data | car | traffic sign | traffic light | person | truck |
| --- | ---: | ---: | ---: | ---: | ---: |
| server_cloudy_train | 57.4% | 18.9% | 11.8% | 6.6% | 3.1% |
| client0_highway | 61.4% | 25.4% | 7.7% | 1.4% | 2.7% |
| client1_citystreet | 49.9% | 18.2% | 17.9% | 9.5% | 2.0% |
| client2_residential | 73.7% | 13.1% | 7.3% | 3.0% | 1.5% |

The saved diagnostic table is:

```text
output/02_stable_pseudogt/stats/02_domain_class_distribution_from_bdd_json.csv
```

## Multi-Round Objective

Each round should be defined as:

1. Start from the current repaired global model.
2. Generate pseudo labels or pseudo-training signals on each client target set.
3. Train each client with source GT plus target pseudo signal.
4. Aggregate client checkpoints.
5. Run supervised server repair on source cloudy GT.
6. Evaluate the repaired global checkpoint.
7. Use the repaired global checkpoint as the next round's teacher/start model.

The repaired global checkpoint is the main product of the round.  Aggregate
metrics remain diagnostic, but they should not be the primary success criterion.

## Metrics To Use

### Primary Metrics

| metric | definition | purpose |
| --- | --- | --- |
| repaired_mAP50 | scene-total mAP@0.5 after server repair | main performance score |
| repaired_mAP50_95 | scene-total mAP@0.5:0.95 after server repair | main localization-aware score |
| final_round_mAP | repaired mAP at the configured final round | avoids GT-based best checkpoint selection |
| last_N_avg_mAP | average repaired mAP over the last N rounds | measures stable convergence |
| last_N_min_mAP | minimum repaired mAP over the last N rounds | detects late collapse |

### Diagnostic Metrics

| metric | definition | purpose |
| --- | --- | --- |
| aggregate_mAP | mAP immediately after client aggregation | measures how destructive client adaptation is |
| repair_gain | repaired_mAP - aggregate_mAP | measures how much server repair recovers |
| retained_gain | repaired_mAP - warmup_mAP | measures final useful gain |
| client_to_repair_gap | repaired_mAP - average_client_mAP | measures dependence on server repair |
| round_delta | repaired_mAP(round t) - repaired_mAP(round t-1) | measures round-to-round stability |
| plateau_slope | linear slope of repaired mAP over the last N rounds | detects continuing drift |

### Pseudo-Label Diagnostics

| metric | definition | purpose |
| --- | --- | --- |
| pseudo_images_kept | images with at least one pseudo label | detects over/under-selection |
| pseudo_boxes_per_image | kept pseudo boxes per kept image | detects too many box targets |
| mean_pseudo_conf | mean detector confidence | confidence diagnostic only |
| mean_pseudo_stability | mean augmentation stability | localization consistency diagnostic |
| class_distribution_delta | pseudo class distribution minus target/source reference | detects easy-class collapse |
| rare_class_drop | count or ratio for rider, motor, train, etc. | detects teacher blindness |

## Success Criteria

Do not select a checkpoint by validation GT.  Use fixed round count or the last
N rounds.

Recommended short-run success criteria:

- Final repaired mAP@0.5:0.95 is above warmup.
- Last-N average repaired mAP@0.5:0.95 is above warmup.
- Last-N minimum repaired mAP@0.5:0.95 does not fall back to warmup.
- Aggregate mAP may be lower than warmup, but it must not collapse so hard that
  server repair becomes the only source of improvement.
- Scene-wise gains should not come only from one split.

For the current warmup baseline:

| metric | warmup |
| --- | ---: |
| scene_total mAP@0.5 | 0.352 |
| scene_total mAP@0.5:0.95 | 0.189 |

The 02 repaired best is:

| checkpoint | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: |
| stable_mix_all_lowlr_server_repair | 0.381 | 0.212 |

The first multi-round target is not necessarily to beat the best single-round
02 result.  The first target is to show that repaired mAP rises or plateaus
instead of degrading as rounds increase.

## Design Direction For 03

Use server repair as a first-class component of the method.

Recommended base:

- Start from `stable_mix_all_lowlr`.
- Evaluate every round after server repair.
- Carry the repaired global model into the next round.
- Keep server repair supervised-only on source cloudy GT.

Change pseudo-GT from "ordinary GT replacement" to "repair-oriented target
signal":

1. Use a small high-quality tier for bbox regression.
   - stricter confidence/stability thresholds
   - lower max boxes per image
   - per-class cap
   - preferably top-k per image
2. Use the remaining target images as weak adaptation signal.
   - avoid broad bbox regression from medium-quality boxes
   - prefer objectness/class/feature/domain adaptation when possible
3. Keep source GT in client training.
   - client training should not become target-pseudo-only
   - source labels keep the head close enough for repair to work
4. Treat aggregate mAP as a safety diagnostic.
   - aggregate can be below warmup
   - severe aggregate collapse means client updates are too destructive
5. Treat repaired mAP as the main claim.
   - the method is a client-adaptation plus server-repair loop
   - not pseudo-GT-only supervised learning

## Interpretation For DQA

The DQA/FedSTO-style story should be reframed:

- Clients are not expected to produce independently strong detectors from
  pseudo-GT.
- Clients are expected to produce target-domain-biased updates.
- Aggregation combines those target-domain updates.
- Server repair restores detector calibration/localization with reliable source
  GT.
- DQA should eventually decide which client updates are useful because they
  survive server repair, not because they have high pseudo confidence before
  repair.
