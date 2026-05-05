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
