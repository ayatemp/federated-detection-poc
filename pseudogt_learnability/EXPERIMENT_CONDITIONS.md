# Experiment Conditions

## Project Question

Can pseudo-GT client training improve the detector under the same scene-client
condition used by DQA?

This project treats that as a prerequisite for DQA.  If clients cannot learn
useful local updates from pseudo-GT, class-wise aggregation has no reliable
signal to exploit.

## Fixed Inputs

| item | value |
| --- | --- |
| dataset | BDD100K scene-client setup |
| model | EfficientTeacher YOLOv5L |
| start checkpoint | `checkpoints/round000_warmup.pt` |
| source of start checkpoint | DQA 08_4 `round000_warmup.pt` |
| image size | 640 |
| client target mode | unlabeled-only |
| server train list | `output/data_lists/server_cloudy_train.txt` |
| evaluation protocol | scene-wise validation |

## Clients

| client | scene | target images |
| --- | --- | ---: |
| client 0 | highway | 5000 |
| client 1 | city street | 5000 |
| client 2 | residential | 5000 |

## Evaluation Splits

| split | images | boxes |
| --- | ---: | ---: |
| highway | 2499 | 36377 |
| citystreet | 6112 | 127178 |
| residential | 1253 | 20855 |
| scene_total | 9864 | union |

## Baseline To Beat

From DQA 08_4 scene evaluation:

| checkpoint | scene_total mAP@0.5 | scene_total mAP@0.5:0.95 |
| --- | ---: | ---: |
| warmup | 0.352 | 0.189 |
| phase1 round003 | 0.375 | 0.203 |
| phase1 round012 | 0.369 | 0.199 |
| phase2 round001 | 0.370 | 0.203 |
| phase2 round012 | 0.348 | 0.189 |

The immediate target is not to beat the best Phase 1 result.  The immediate
target is to show that pseudo-GT client learning can produce at least one
useful local or aggregate update from the warmup checkpoint.

## 01 Profiles

| profile | scope | local epochs | main protection |
| --- | --- | ---: | --- |
| `backbone_obj_safe` | backbone | 3 | head is frozen; uncertain pseudo boxes mainly affect objectness |
| `neck_head_high_precision` | neck/head | 2 | pseudo boxes require higher confidence |
| `all_consistency_lowlr` | all | 2 | low LR, high thresholds, non-backbone orthogonal regularization |

All profiles use:

- lower pseudo-GT loss than the original FedSTO config
- reduced mixup/mosaic/cutout strength
- tri-stage objectness gating through the existing EfficientTeacher patch
- no DQA aggregation during local learning

## Outputs

All generated artifacts stay inside:

```text
pseudogt_learnability/output/
```

Important files after running 01:

| file | role |
| --- | --- |
| `output/stats/01_manifest.json` | exact run settings and checkpoint list |
| `output/stats/01_checkpoints.csv` | labels and paths for saved checkpoints |
| `output/checkpoints/*.pt` | normalized client, aggregate, and repair checkpoints |
| `output/validation_reports/paper_protocol_eval_summary.md` | readable scene evaluation summary |
| `output/validation_reports/paper_protocol_eval_summary.csv` | tabular scene metrics |

## Decision Rule

Continue toward DQA-style aggregation only if at least one of these happens:

1. An aggregate or server-repair checkpoint beats warmup on `scene_total`
   mAP@0.5:0.95.
2. A client checkpoint improves its matching scene split without a severe
   `scene_total` collapse.

If neither happens, the next project step should change pseudo-GT generation or
loss design, not the DQA aggregation rule.

## 02 Stable Pseudo-GT Condition

02 changes pseudo-GT generation rather than DQA aggregation.

| item | value |
| --- | --- |
| teacher | single copied warmup checkpoint |
| target views | identity and horizontal flip |
| pseudo-label acceptance | same class plus stable de-augmented box location |
| default view requirement | both views must support the box |
| default stability threshold | mean IoU to weighted box >= 0.58 |
| default stable score | mean confidence * stability >= 0.16 |
| training data | source cloudy labels plus stable target pseudo labels |
| default output root | `output/02_stable_pseudogt/` |

This intentionally avoids two-teacher designs.  The point is to ask whether a
single model can avoid self-training drift when bbox labels are allowed to
supervise regression only if their localization is augmentation-stable.

## 02 Profiles

| profile | scope | local epochs | main protection |
| --- | --- | ---: | --- |
| `stable_mix_backbone` | backbone | 3 | head is frozen; stable target boxes adapt features while source labels anchor the detector |
| `stable_mix_all_lowlr` | all | 2 | full-model update uses low LR, source anchor, and stability-gated target boxes |
| `stable_mix_neck_head` | neck/head | 2 | optional conservative head adaptation from high-quality stable boxes |

Important diagnostics after running 02:

| file | role |
| --- | --- |
| `output/02_stable_pseudogt/stats/02_pseudo_label_stats.csv` | per-client stable pseudo-label counts and means |
| `output/02_stable_pseudogt/stats/02_*_stable_boxes.csv` | accepted pseudo boxes with confidence, stability, and score |
| `output/02_stable_pseudogt/stats/02_manifest.json` | exact run settings and checkpoint list |
| `output/02_stable_pseudogt/validation_reports/paper_protocol_eval_summary.csv` | scene-wise mAP comparison |

## 03 Repair-Oriented Multi-Round Condition

03 treats server repair as part of the method rather than as a secondary
checkpoint.  The main unit is:

1. current repaired global model generates stable pseudo labels
2. clients train from that model with source cloudy GT plus target pseudo labels
3. client checkpoints are aggregated
4. aggregate checkpoint is repaired on source cloudy GT
5. repaired checkpoint is evaluated and becomes the next round's global model

Default 03 settings:

| item | value |
| --- | --- |
| profile | `repair_oriented_all_lowlr` |
| client epochs | 1 |
| client train scope | all |
| client LR | 0.0005 |
| source repeat in client train | 2 |
| pseudo repeat in client train | 1 |
| server repair epochs | 1 |
| server repair LR | 0.0008 |
| default rounds | 3 |
| default output root | `output/03_repair_oriented_multiround/` |

Default pseudo-label filtering is stricter than 02:

| item | value |
| --- | ---: |
| confidence threshold | 0.25 |
| match IoU | 0.60 |
| min stability | 0.72 |
| min score | 0.28 |
| max boxes/image | 12 |
| max class fraction | 0.45 |

Primary 03 metrics:

| metric | role |
| --- | --- |
| final repaired mAP@0.5:0.95 | fixed-final-round score without GT-based checkpoint selection |
| last-N average repaired mAP@0.5:0.95 | plateau/convergence score |
| last-N minimum repaired mAP@0.5:0.95 | late-collapse check |
| repair gain | diagnostic: repaired mAP minus aggregate mAP |
| retained gain | diagnostic: repaired mAP minus warmup mAP |

Important outputs:

| file | role |
| --- | --- |
| `output/03_repair_oriented_multiround/stats/03_round_metrics.csv` | per-round repaired, aggregate, repair-gain, and retained-gain metrics |
| `output/03_repair_oriented_multiround/stats/03_round_metrics_summary.json` | final/last-N metrics |
| `output/03_repair_oriented_multiround/stats/03_round*_pseudo_label_stats.csv` | per-round pseudo-label diagnostics |
| `output/03_repair_oriented_multiround/stats/03_checkpoints.csv` | saved checkpoints |
