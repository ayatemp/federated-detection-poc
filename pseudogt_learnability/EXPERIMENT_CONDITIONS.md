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

