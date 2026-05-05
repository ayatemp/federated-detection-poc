# DQA-CWA Requirements Definition

Last updated: 2026-04-22

## 1. Purpose

This document defines the current implementation state, functional requirements,
research requirements, validation requirements, and remaining gaps for Dynamic
Quality-Aware Class-wise Aggregation (DQA-CWA).

DQA-CWA is a proposed extension on top of the FedSTO-style semi-supervised
federated object detection pipeline in `navigating_data_heterogeneity`. The
goal is to improve aggregation under weather-based non-IID client data by
changing only the aggregation step, while keeping the existing FedSTO training
protocol as intact as possible.

## 2. Research Background

The base project reproduces the FedSTO setting:

- One labeled server domain.
- Three unlabeled client domains.
- BDD100K-style weather heterogeneity.
- Server domain: cloudy-equivalent `partly cloudy`.
- Client domains: `overcast`, `rainy`, and `snowy`.
- Object detector: EfficientTeacher YOLOv5L-style detector.
- Target class space: 10 BDD100K detection classes.
- Paper-scale protocol target: warm-up 50 epochs, phase 1 for 100 rounds, phase
  2 for 150 rounds, one local epoch per round.

The current FedSTO reproduction path is implemented under:

```text
navigating_data_heterogeneity/
```

The DQA-CWA proposal is intentionally isolated under:

```text
dynamic_quality_aware_classwise_aggregation/
```

This separation allows the FedSTO baseline to keep running unchanged while
DQA-CWA is developed and evaluated as an additional method.

## 3. Problem Statement

In semi-supervised federated object detection, client updates can have different
quality per class because each client sees different weather conditions and
receives pseudo labels of different reliability.

FedAvg treats client updates uniformly at the parameter level. In a weather
non-IID setting, this can mix weak pseudo-label updates for a class with stronger
updates from another client. DQA-CWA addresses this by weighting classification
head rows per class using dynamic pseudo-label statistics.

The intended hypothesis is:

```text
For each object class, clients with more reliable pseudo labels for that class
should contribute more to the classification-head rows for that class.
```

## 4. Scope

### 4.1 In Scope

- Add class-wise dynamic aggregation to FedSTO phase 2.
- Use per-client, per-class pseudo-label statistics as reliability signals.
- Apply DQA only to detection-head classification rows.
- Keep backbone, neck, objectness rows, and box-regression rows on FedAvg.
- Preserve FedSTO phase 1 behavior.
- Preserve local training, server training, checkpoint reuse, and history resume
  behavior from the FedSTO runner.
- Store DQA outputs separately from the FedSTO baseline.
- Support restartable long-running experiments.
- Provide enough logging and state output to audit the aggregation weights.

### 4.2 Out Of Scope For The Current Scaffold

- Replacing EfficientTeacher's internal pseudo-label assigner.
- Changing client/server local training losses.
- Adding privacy-preserving noise or secure aggregation for DQA statistics.
- Full paper-quality comparison against all FedSTO baselines.
- Automatic extraction of pseudo-label statistics from EfficientTeacher internals.
  This is currently a required future integration point.

## 5. Current Implementation Status

### 5.1 Implemented Files

```text
dynamic_quality_aware_classwise_aggregation/
  README.md
  __init__.py
  collect_pseudo_stats.py
  dqa_cwa_aggregation.py
  run_dqa_cwa_fedsto.py
  stats/
```

### 5.2 Core Aggregation Module

Implemented in:

```text
dynamic_quality_aware_classwise_aggregation/dqa_cwa_aggregation.py
```

Current capabilities:

- Loads round-level client class statistics from JSON.
- Supports several JSON shapes:
  - `{"clients": [...]}`
  - `[...]`
  - `{"client_id": {...}}`
- Computes client-class reliability from pseudo-label counts and mean
  confidence.
- Maintains EMA state for counts, quality, and aggregation weights.
- Supports a labeled-server anchor.
- Applies FedAvg to all floating-point parameters by default.
- Replaces only class-specific classification rows in detection heads with
  DQA-weighted rows.
- Supports YOLOv5-style head tensors:
  - `head.m.0.weight`
  - `head.m.0.bias`
  - `head.m.1.weight`
  - `head.m.1.bias`
  - `head.m.2.weight`
  - `head.m.2.bias`
- Applies the same dynamic aggregation to EMA weights if all source checkpoints
  contain EMA.
- Provides a synthetic `self-test` command.

The current reliability formula is:

```text
R_{k,c}^{(t)}
  = log(1 + EMA(n_{k,c}^{(t)}))
    * EMA(q_{k,c}^{(t)})
    * stability
```

The normalized aggregation weight is:

```text
alpha_{k,c}^{(t)} = normalize_k(R_{k,c}^{(t)})
```

where:

- `k` is the source client or server anchor.
- `c` is the object class.
- `n` is pseudo-label count.
- `q` is mean pseudo-label confidence.
- `stability` penalizes abrupt quality changes.

Current conservative controls:

- `count_ema`
- `quality_ema`
- `alpha_ema`
- `temperature`
- `uniform_mix`
- `classwise_blend`
- `stability_lambda`
- `min_effective_count`
- `server_anchor`

### 5.3 DQA Runner

Implemented in:

```text
dynamic_quality_aware_classwise_aggregation/run_dqa_cwa_fedsto.py
```

Current capabilities:

- Imports and reuses the FedSTO setup and runner modules.
- Redirects generated configs, runs, checkpoints, and history into:

```text
dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa_cwa/
```

- Reuses FedSTO data list generation.
- Reuses EfficientTeacher training commands.
- Supports warm-up, phase 1, and phase 2.
- Starts DQA aggregation in phase 2 by default.
- Requires round-level DQA stats by default.
- Can fall back to FedAvg if `--fallback-fedavg-without-stats` is explicitly
  passed.
- Supports restart and checkpoint reuse through `history.json`.
- Cleans intermediate checkpoints unless `--keep-intermediate-checkpoints` is
  set.
- Writes compact training output to `dqa_cwa_latest.log` by default.
- Includes disk-space protection through `--min-free-gib`.

### 5.4 Pseudo-Label Statistics Collector

Implemented in:

```text
dynamic_quality_aware_classwise_aggregation/collect_pseudo_stats.py
```

Current capabilities:

- Reads pseudo-label dumps from text, CSV-like text, or JSON files.
- Supports rows shaped like:
  - `cls x y w h`
  - `cls x y w h confidence`
  - `image_id cls x y w h confidence`
- Produces per-client:
  - class counts
  - confidence sums
  - mean confidences
- Writes DQA-compatible JSON:

```json
{
  "clients": [
    {
      "id": "0",
      "counts": [0.0],
      "confidence_sums": [0.0],
      "mean_confidences": [0.0]
    }
  ]
}
```

### 5.5 Current DQA Experiment State

As of this document:

- DQA implementation files exist.
- The DQA aggregation self-test passes.
- A DQA dry run can generate configs and data lists.
- `dynamic_quality_aware_classwise_aggregation/stats/` is empty.
- There are no completed DQA-CWA training results yet.
- There is no DQA-CWA `history.json` yet.
- There are no DQA-CWA global checkpoints yet.

Therefore, the current DQA-CWA project state is:

```text
Research idea + runnable scaffold + aggregation implementation.
Not yet an evaluated experimental result.
```

### 5.6 Current FedSTO Baseline State

The FedSTO baseline workspace is:

```text
navigating_data_heterogeneity/efficientteacher_fedsto/
```

Observed current state:

- Phase 1 appears complete through 100 rounds.
- Phase 2 appears complete through at least 44 rounds.
- Phase 2 round 45 has partial client/server artifacts.
- The latest validation reports may not fully reflect the newest training
  history.

This baseline progress is useful because DQA-CWA can reuse the same dataset,
protocol, model family, and runner patterns. However, DQA-CWA should be run in
its own workspace to avoid contaminating the FedSTO baseline artifacts.

## 6. Functional Requirements

### FR-1: Round-Level Stats Loading

The DQA runner must load one JSON stats file for every DQA-enabled round.

Default expected path:

```text
dynamic_quality_aware_classwise_aggregation/stats/phase2_round001.json
```

Requirement:

- If DQA is enabled and the required stats file is missing, the runner must stop.
- Silent fallback to FedAvg is not allowed in real experiments.
- FedAvg fallback is allowed only when explicitly requested for smoke testing.

### FR-2: Reliability Computation

For each client and class, the system must compute reliability using:

- pseudo-label count
- mean pseudo-label confidence
- EMA-smoothed count
- EMA-smoothed quality
- stability penalty

Reliability must be non-negative and normalized across sources per class.

### FR-3: Class-Wise Aggregation

For active classes, the system must aggregate classification-head rows using
class-wise alpha values.

Requirement:

- Only class rows of the detection head are DQA-weighted.
- Non-class rows remain FedAvg.
- Inactive classes remain FedAvg.
- Output tensors must preserve original dtype and shape.

### FR-4: Server Anchor

The system must optionally include the labeled server checkpoint as a reliability
source for classification rows.

Requirement:

- `server_anchor = 0.0` disables the server anchor.
- `server_anchor > 0.0` adds the server as an additional source.
- The server anchor should prevent noisy pseudo-label-heavy clients from
  completely overwriting labeled-server knowledge.

### FR-5: FedAvg Compatibility

DQA-CWA must remain compatible with the existing FedSTO checkpoint format.

Requirement:

- It must load EfficientTeacher checkpoints containing `model`.
- It must support checkpoints containing `ema`.
- It must save a standard checkpoint usable as the next global checkpoint.

### FR-6: Runner Restartability

DQA-CWA experiments must be restartable.

Requirement:

- Completed global checkpoints are reused.
- `history.json` defines the continuous completed prefix.
- DQA EMA state is rebuilt from history if needed and all required stats files
  are available.
- Invalid checkpoints are detected and retrained.

### FR-7: Output Isolation

DQA-CWA outputs must not overwrite FedSTO baseline outputs.

Requirement:

All generated DQA artifacts should be stored under:

```text
dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa_cwa/
dynamic_quality_aware_classwise_aggregation/stats/
```

### FR-8: Stats Collection

The project must support converting pseudo-label dumps into round-level DQA
stats.

Current implementation:

- Standalone conversion script exists.

Required next step:

- Connect EfficientTeacher pseudo-label generation to automatic stats export.

## 7. Non-Functional Requirements

### NFR-1: Minimal Baseline Disruption

DQA-CWA must not require changes to the FedSTO baseline training semantics beyond
the aggregation step.

### NFR-2: Auditability

DQA-CWA must store enough state to inspect why a source received high or low
weight.

Required audit outputs:

- per-round `alpha`
- active class mask
- source IDs
- DQA config
- count EMA
- quality EMA

Current implementation stores these in the DQA state JSON.

### NFR-3: Reproducibility

Experiments must specify:

- dataset split
- model checkpoint
- warm-up epochs
- phase 1 rounds
- phase 2 rounds
- batch size
- number of GPUs
- DQA hyperparameters
- random seed, if supported by the runner/trainer

### NFR-4: Disk Safety

The runner must avoid filling disk during long training runs.

Current implementation:

- `--min-free-gib` gate.
- cleanup of completed intermediate checkpoints.
- optional `--keep-intermediate-checkpoints`.

### NFR-5: Runtime Visibility

The runner must avoid flooding notebooks with raw training output.

Current implementation:

- training output is appended to `dqa_cwa_latest.log` by default.
- raw streaming can be enabled with `--stream-train-output`.

## 8. Data Requirements

### 8.1 Dataset

Required dataset layout is inherited from the FedSTO reproduction:

```text
navigating_data_heterogeneity/data_paper20k/
  server/
    images/train/
    images/val/
    labels/train/
    labels/val/
  clients/
    client_0/images_unlabeled/
    client_1/images_unlabeled/
    client_2/images_unlabeled/
```

### 8.2 Class Space

The expected 10 classes are:

```text
person
rider
car
bus
truck
bike
motor
traffic light
traffic sign
train
```

### 8.3 DQA Stats Schema

Each DQA round must have one stats JSON file containing all clients participating
in that round.

Example:

```json
{
  "clients": [
    {
      "id": "0",
      "counts": [10, 0, 120, 4, 2, 0, 0, 16, 44, 0],
      "mean_confidences": [0.81, 0.0, 0.74, 0.69, 0.63, 0.0, 0.0, 0.71, 0.76, 0.0]
    }
  ]
}
```

Requirements:

- `counts` length must equal `num_classes`.
- `mean_confidences` length must equal `num_classes`.
- Client ordering should match the client checkpoint ordering.
- Client IDs should be stable across rounds.

## 9. Evaluation Requirements

### 9.1 Required Baselines

Minimum required comparison:

- FedSTO baseline with the same schedule.
- DQA-CWA with the same schedule.

Recommended additional comparisons:

- FedAvg-style full aggregation.
- Count-only DQA.
- Confidence-only DQA.
- DQA without server anchor.
- DQA without alpha EMA.
- DQA with `classwise_blend = 1.0`.
- DQA with classification rows only versus broader head aggregation, if tested.

### 9.2 Required Metrics

Object detection metrics:

- mAP@0.5
- mAP@0.5:0.95
- precision
- recall

DQA-specific diagnostics:

- per-round alpha heatmap
- per-class active mask
- per-client class count distribution
- per-client mean confidence distribution
- correlation between alpha and final per-class AP, if available

### 9.3 Required Evaluation Splits

At minimum:

- server cloudy validation

For paper-quality evaluation:

- server cloudy
- overcast client domain
- rainy client domain
- snowy client domain
- total combined evaluation

### 9.4 Acceptance Criteria For A Valid DQA Experiment

A DQA run is considered valid only if:

- Stats files exist for all DQA-enabled rounds.
- `--fallback-fedavg-without-stats` is not used.
- The DQA state file records non-uniform alpha for at least some active classes.
- The global checkpoints are produced from `aggregate_checkpoints` in
  `dqa_cwa_aggregation.py`.
- Evaluation is run on the produced DQA global checkpoints.
- FedSTO baseline and DQA-CWA use the same data split and comparable schedule.

## 10. Current Risks And Gaps

### Gap-1: Pseudo-Label Stats Are Not Automatically Exported

The biggest current blocker is that DQA-CWA requires per-round pseudo-label
statistics, but the EfficientTeacher training flow does not yet automatically
write the required stats files.

Impact:

- DQA-CWA cannot run as a real method until stats are generated.
- Manual stats collection is possible only if pseudo-label dumps are available.

Required fix:

- Add an EfficientTeacher hook or post-processing step that exports per-client,
  per-class pseudo-label counts and confidence means for each DQA-enabled round.

### Gap-2: Mean Confidence May Be Too Simple

Current quality uses mean pseudo-label confidence only.

Impact:

- Classification confidence may not represent localization quality.
- Clients with many overconfident bad boxes may be overweighted.

Possible extensions:

- objectness-aware quality
- box-size or IoU-consistency quality
- teacher-student agreement
- NMS stability
- per-class pseudo-label entropy

### Gap-3: Privacy Leakage From Stats

Counts and confidence values can reveal class distribution patterns for each
client.

Impact:

- This is acceptable for an initial research prototype.
- It should be discussed for a federated-learning paper.

Possible mitigations:

- coarse binning
- clipping
- additive noise
- secure aggregation of statistics

### Gap-4: Evaluation Reports May Be Stale

The existing FedSTO validation report may not reflect the latest baseline
history and checkpoints.

Required fix:

- Regenerate validation reports after baseline and DQA runs.
- Ensure final tables reference the exact evaluated checkpoints.

### Gap-5: Environment Dependencies

Checkpoint loading and EfficientTeacher imports require the vendor runtime
dependencies.

Observed missing dependencies in the current environment:

- `seaborn`
- `tensorboard`

Impact:

- Some checkpoint inspection or training paths may fail until dependencies are
  installed.

## 11. Implementation Milestones

### Milestone 1: Make DQA Stats Automatic

Deliverables:

- Per-client pseudo-label stats are exported for each phase 2 round.
- Stats are saved to `dynamic_quality_aware_classwise_aggregation/stats/`.
- A smoke run verifies that DQA aggregation is reached without fallback.

Acceptance:

- `phase2_round001.json` is generated automatically.
- DQA state contains `last_alpha` and `last_active_classes`.

### Milestone 2: Pilot DQA Run

Recommended command shape:

```bash
python3 dynamic_quality_aware_classwise_aggregation/run_dqa_cwa_fedsto.py \
  --warmup-epochs 20 \
  --phase1-rounds 40 \
  --phase2-rounds 60 \
  --batch-size 64 \
  --workers 0 \
  --gpus 2 \
  --min-free-gib 80
```

Acceptance:

- DQA run completes.
- DQA global checkpoints are produced.
- DQA state evolves over phase 2 rounds.
- No fallback FedAvg is used in phase 2.

### Milestone 3: Baseline Versus DQA Evaluation

Deliverables:

- FedSTO baseline evaluation table.
- DQA-CWA evaluation table.
- Per-domain metrics.
- Per-class metrics, if available.
- Alpha heatmap figures.

Acceptance:

- DQA-CWA is compared against FedSTO under the same dataset split and schedule.
- Results include mAP@0.5 and mAP@0.5:0.95.
- The report identifies whether DQA improves, degrades, or has mixed impact.

### Milestone 4: Ablation Study

Deliverables:

- count-only DQA
- confidence-only DQA
- no server anchor
- no alpha EMA
- blend sweep

Acceptance:

- The paper claim can explain which component contributes to the improvement.

## 12. Research Claim Requirements

The final research claim should not be:

```text
DQA-CWA is better because it uses confidence.
```

The stronger intended claim is:

```text
In semi-supervised federated object detection with weather-induced non-IID
client domains, pseudo-label reliability differs by class and client. Applying
quality-aware class-wise aggregation only to classification-head rows improves
or stabilizes global detection performance while preserving the FedSTO training
protocol.
```

To support this claim, the final report must show:

- FedSTO baseline comparison.
- Per-domain evaluation.
- Per-class or class-group analysis.
- DQA alpha diagnostics.
- Ablation evidence.
- Clear statement of when DQA helps and when it does not.

## 13. Definition Of Done

DQA-CWA is considered research-ready when:

- Automatic stats export is implemented.
- A full or pilot DQA run finishes without FedAvg fallback.
- FedSTO and DQA results are evaluated with the same protocol.
- At least one ablation verifies that class-wise quality weighting matters.
- Alpha diagnostics are available and interpretable.
- Known gaps and limitations are documented.

It is considered paper-ready when, in addition:

- The evaluation covers all relevant weather domains.
- Results are stable over multiple seeds or repeated runs where feasible.
- Privacy and communication overhead are discussed.
- The novelty is positioned against class-wise and quality-aware federated
  aggregation prior work.
