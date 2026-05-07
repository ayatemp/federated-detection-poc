# Experiment Conditions

## Question

Can DQA become useful when clients differ not only by weather/domain but also by
scene, class distribution, and label density?

FedSTO mainly addresses labeled-server/unlabeled-client domain shift.  This
line tests whether DQA can add value when client updates have different
class-wise and scene-wise strengths.

## Client Splits

The clients are defined by BDD100K metadata only.  They do not use GT box counts
for assignment.

| client | scene | timeofday | default target images |
| --- | --- | --- | ---: |
| client0_highway_day | highway | daytime | 1500 |
| client1_highway_night | highway | night | 1500 |
| client2_citystreet_day | city street | daytime | 1500 |
| client3_citystreet_night | city street | night | 1500 |
| client4_residential_day | residential | daytime | 1500 |
| client5_residential_night | residential | night | 1500 |

The default 1500-image cap keeps clients balanced and runtime manageable.  The
runner exposes `--client-limit` when a larger or smaller condition is needed.

## Server Data

Server repair remains supervised-only on the same source-cloudy split used by
FedSTO/DQA:

```text
server_cloudy_train.txt
server_cloudy_val.txt
```

This intentionally preserves the FedSTO premise: source GT stabilizes detector
calibration/localization after client adaptation.

## Evaluation Splits

The evaluation includes all six scene-daynight splits plus their union:

| split | scene | timeofday |
| --- | --- | --- |
| highway_day | highway | daytime |
| highway_night | highway | night |
| citystreet_day | city street | daytime |
| citystreet_night | city street | night |
| residential_day | residential | daytime |
| residential_night | residential | night |
| scene_daynight_total | union | union |

## Method

Each round:

1. Start from the current repaired global model.
2. Generate stable pseudo labels for every client target set.
3. Train each client with source cloudy GT plus strict target pseudo labels.
4. Aggregate client checkpoints using server-anchored DQA-CWA v2.
5. Repair the aggregate on source cloudy GT.
6. Evaluate aggregate and repaired checkpoints.
7. Use the repaired checkpoint as the next round's global model.

## Primary Metrics

| metric | purpose |
| --- | --- |
| repaired mAP@0.5 | main detection score |
| repaired mAP@0.5:0.95 | main localization-aware score |
| final-round repaired mAP | fixed-round result without GT checkpoint picking |
| last-N average repaired mAP | plateau/stability score |
| last-N minimum repaired mAP | late-collapse check |

## Diagnostics

| metric | purpose |
| --- | --- |
| aggregate mAP | how destructive client updates are before repair |
| repair gain | repaired mAP minus aggregate mAP |
| retained gain | repaired mAP minus warmup mAP |
| DQA alpha | class-wise client/server weights used in aggregation |
| pseudo boxes per image | pseudo-label quantity drift |
| pseudo class distribution | easy-class or rare-class collapse |

## 01_0 Control Conditions

The first 01 result showed repaired performance improvements, but this can be
confounded by extra supervised source repair epochs.  The 01_0 notebook runs the
controls needed to separate the effects:

| condition | purpose |
| --- | --- |
| `repair_only` | tests whether warmup was simply under-trained by applying source repair for the same number of rounds |
| `pseudo_fedavg` | tests whether strict pseudoGT client adaptation helps without DQA |
| `pseudo_dqa` | repeats the 01 DQA-CWA v2 policy in the same comparison workspace |

The key claim is supported only if `pseudo_dqa` outperforms `repair_only`, or if
it keeps total mAP comparable while improving worst-split/day-night/classwise
gap metrics.

## 01_1 Diagnostic Sweep

The 01_0 result showed that repaired mAP is mostly explained by source repair:
`repair_only` and pseudoGT variants reached almost the same final mAP.  However,
DQA preserved aggregate mAP better than FedAvg before repair.  The 01_1 notebook
therefore tests whether DQA can become useful when client updates are constrained
to target-domain adaptation instead of full noisy pseudo-box supervision.

Notebook:

```text
notebooks/01_1_dqa_diagnostic_sweep.ipynb
```

Runner:

```text
scripts/run_scene_daynight_dqa_01_1.py
```

Default diagnostic settings follow the 01_0 full-run resource setting: 3
rounds, all 1500 target images per client, 2 GPUs, batch size 160, and 8
workers.

The notebook default is now improvement-only: `all` resolves to DQA variants
only.  `repair_only` and `fedavg_target_double` remain available by explicit
condition name for debugging, but they are not part of the default run because
the matched controls already live in 01_0.

| condition | purpose |
| --- | --- |
| `dqa_current` | 01_0-style source-heavy full-model DQA |
| `dqa_source_light` | tests whether reducing source dominance creates useful target signal |
| `dqa_target_double` | source once, target pseudoGT twice, with softer DQA anchor |
| `dqa_head_lowbox` | head-only target adaptation with weak bbox loss |
| `dqa_nonbackbone_lowbox` | neck/head adaptation with weak bbox loss |

01_1 supports three claims:

| claim | supporting evidence |
| --- | --- |
| pseudoGT client updates are destructive | FedAvg aggregate drops below warmup or needs large repair gain |
| DQA protects the global model | DQA aggregate mAP is higher than FedAvg aggregate mAP |
| DQA adds target-domain value | repaired mAP, worst-split mAP, or night average exceeds `repair_only` |

## 01_2 SSOD Pivot

If 01_1 still cannot make DQA outperform `repair_only`, the next hypothesis is
that fixed pseudoGT is the wrong client-training interface.  In 01_2, clients
train with EfficientTeacher/FedSTO-style SSOD on unlabeled target images.  Stable
pseudo boxes are generated only to estimate DQA class-wise reliability for
aggregation; they are not written back as fixed supervised target labels.

Notebook:

```text
notebooks/01_2_ssod_pivot_dqa.ipynb
```

Runner:

```text
scripts/run_scene_daynight_dqa_01_2.py
```

| condition | purpose |
| --- | --- |
| `ssod_dqa` | SSOD client training with DQA reliability from stable pseudo boxes |
| `ssod_dqa_head` | head-only SSOD target adaptation with weak bbox loss |
| `ssod_dqa_nonbackbone` | neck/head SSOD target adaptation with reduced bbox loss |

01_2 supports a different claim from 01_1:

The notebook default is improvement-only: `all` resolves to `ssod_dqa`,
`ssod_dqa_head`, and `ssod_dqa_nonbackbone`.  `repair_only` and `ssod_fedavg`
remain available by explicit condition name, but they are not part of the
default run.

| claim | supporting evidence |
| --- | --- |
| fixed pseudoGT supervision is the bottleneck | 01_1 fails but 01_2 SSOD variants improve repaired/worst/night mAP |
| target data matters beyond source repair | any SSOD-DQA variant beats `repair_only` on total, worst split, or night average |

The optional `ssod_fedavg` condition can still be run explicitly if a new SSOD
control is needed later.

## 02 Head-to-Full Long DQA

The active 02 experiment tests the new phase interpretation:

- Phase1 should be long and conservative so client/class/domain differences can
  emerge without pseudoGT damaging the detector backbone.
- Phase2 should be short and stronger so the Phase1 target signal can touch the
  full model before confidence/easy-sample bias dominates.
- Evaluation should be final-focused by default because a 30+2 round pilot is
  already a long run.

Notebook:

```text
notebooks/02_head_to_full_long_dqa.ipynb
```

Runner:

```text
scripts/run_scene_daynight_dqa_02_head_to_full.py
```

Default schedule:

| phase | rounds | client train scope | client LR | source:pseudo repeat | bbox loss | role |
| --- | ---: | --- | ---: | --- | ---: | --- |
| Phase1 | 30 | `neck_head` | 0.0008 | 1:2 | 0.005 | build stable DQA client/class/domain signal |
| Phase2 | 2 | `all` | 0.0003 | 1:1 | 0.010 | brief full-model target burst |

Final-focused paper-protocol evaluation includes only:

- `warmup_global`
- `phase1_final_aggregate`
- `phase1_final_repair`
- `phase2_final_aggregate`
- `phase2_final_repair`

The key claim is supported only if `phase2_final_repair` beats the 01_0
`repair_only` reference on total repaired mAP50:95, or improves the worst/night
client splits while preserving total mAP.

## Superseded 02 Target-Consistency Repair DQA

The 01 series did not show post-repair gains over `repair_only`, even when DQA
preserved the aggregate checkpoint better than FedAvg.  The 02 notebook tests
the hypothesis that source-only server repair erases target-domain information
introduced by the clients.

Notebook:

```text
notebooks/02_target_consistency_repair_dqa.ipynb
```

Runner:

```text
scripts/run_scene_daynight_dqa_02.py
```

Each round keeps the 01_2 SSOD-DQA client update, then changes server repair:

1. aggregate client checkpoints with DQA-CWA v2;
2. train the server on source cloudy GT as before;
3. also expose the server repair step to unlabeled client target images through
   weak SSOD target consistency on the neck/head repair scope;
4. keep pseudo bbox loss weak or disabled during repair so target consistency
   does not become ordinary noisy pseudo-box supervision.

Default improvement conditions:

| condition | purpose |
| --- | --- |
| `tc_repair_light` | target-consistency repair using all client target images uniformly |
| `tc_repair_night_focus` | same repair, but oversamples night clients and the persistent worst split `highway_night` |

The key claim is supported only if 02 beats `01_0/repair_only` on total
repaired mAP50:95, or improves `highway_night` / night-average mAP50:95 while
not lowering total mAP.
