# Dynamic Quality-Aware Class-wise Aggregation

![DQA-CWA banner](../assets/DQA.png)

This directory contains the runnable scaffold for the DQA-CWA research idea.
It is intentionally separate from `navigating_data_heterogeneity` so the FedSTO
baseline can be inspected and reproduced independently.

## Core Idea

DQA-CWA changes only the aggregation step:

```text
R_{k,c}^{(t)} = log(1 + EMA(n_{k,c}^{(t)})) * EMA(q_{k,c}^{(t)}) * stability
alpha_{k,c}^{(t)} = normalize_k(R_{k,c}^{(t)})
```

Only the classification rows of the detection head are aggregated with
`alpha_{k,c}`. Backbone, neck, box regression, and objectness stay on FedAvg.

The implementation also keeps the method conservative:

- log-count saturation for noisy pseudo-label-heavy clients
- EMA smoothing for count, confidence, and alpha
- temperature and uniform mixing to avoid over-sharp weights
- optional labeled-server anchor for classification rows
- FedAvg/DQA blending via `classwise_blend`

## Files

- `dqa_cwa_aggregation.py`: core reliability scoring and class-wise head aggregation
- `run_dqa_cwa_fedsto.py`: FedSTO-style runner that stores generated outputs in this directory
- `collect_pseudo_stats.py`: converts pseudo-label dumps into round-level stats JSON
- `generate_dqa_cwa_notebook.py`: writes a notebook that can launch, monitor, and evaluate DQA the same way `03_fedsto_exact_reproduction.ipynb` does for FedSTO

Generated files are written under:

```text
dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa_cwa/
dynamic_quality_aware_classwise_aggregation/stats/
```

Paper-style evaluation is available through:

```text
dynamic_quality_aware_classwise_aggregation/evaluate_paper_protocol.py
```

Notebook entry points:

```text
dynamic_quality_aware_classwise_aggregation/01_dqa_cwa_reproduction.ipynb
dynamic_quality_aware_classwise_aggregation/02_dqa_cwa_exact_reproduction.ipynb
dynamic_quality_aware_classwise_aggregation/02_2_dqa_cwa_14h_reproduction.ipynb
dynamic_quality_aware_classwise_aggregation/02_3_dqa_cwa_14h_evaluation.ipynb
dynamic_quality_aware_classwise_aggregation/03_dqa_cwa_corrected_12h_reproduction.ipynb
dynamic_quality_aware_classwise_aggregation/03_2_dqa_cwa_corrected_12h_evaluation.ipynb
```

## Run Safety

The runner is restartable and disk-conscious by default:

- `history.json` is used to resume from the last continuous completed round.
- Existing valid `last.pt` / global checkpoints are reused after interruption.
- Completed-round intermediate `last.pt`, `best.pt`, and start checkpoints are deleted after each global checkpoint is written.
- Full EfficientTeacher training output is appended to `efficientteacher_dqa_cwa/dqa_cwa_latest.log` so notebooks stay responsive.
- `--min-free-gib` defaults to `70`; if free space drops below that, the runner cleans completed intermediates and then stops with a clear error instead of failing inside `torch.save`.

Use `--keep-intermediate-checkpoints` only when you intentionally need every per-run checkpoint for debugging. Use `--stream-train-output` only when running outside a notebook and you want the raw training logs in the terminal.

## Dry Run

```bash
python3 dynamic_quality_aware_classwise_aggregation/run_dqa_cwa_fedsto.py --dry-run
```

To refresh the notebook after edits:

```bash
python3 dynamic_quality_aware_classwise_aggregation/generate_dqa_cwa_notebook.py
```

The `01` notebook is the fast pilot path. The `02` notebook is the paper-scale path and uses its own workspace (`efficientteacher_dqa_cwa_exact`) and stats directory (`stats_exact`) so it does not collide with the pilot run. The `02_2` notebook is the same-day middle path, tuned from the completed FedSTO runtime log to target roughly 13-14 hours in its own workspace (`efficientteacher_dqa_cwa_14h`) and stats directory (`stats_14h`). The `02_3` notebook is the read-only evaluation pass for that 13-14 hour workspace: it aggregates `results.csv` files into a compact training summary, renders mAP/precision/recall plots, and lines up the DQA server checkpoints against the FedSTO baseline when those artifacts are present. The `02_4` notebook is the dedicated paper-protocol evaluation pass: it runs the shared per-weather validation script against the selected DQA workspace and reshapes the resulting `paper_protocol_eval_summary.csv` into checkpoint-by-split tables and plots.

The `03` and `03_2` notebooks are the corrected path to use next. They apply the FedSTO Algorithm 1 order exactly: client update, client aggregation, server update on labeled data, then server-updated global checkpoint. DQA-CWA starts at phase 1 in this path, so every post-warmup federated round uses dynamic class-wise aggregation rather than FedSTO aggregation.

## Stats Format

For each post-warmup DQA round, the runner expects per-client pseudo-label stats.
For example, phase 1 round 1 is:

```text
dynamic_quality_aware_classwise_aggregation/stats/phase1_round001.json
```

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

Pseudo-label text dumps can be converted with:

```bash
python3 dynamic_quality_aware_classwise_aggregation/collect_pseudo_stats.py \
  --num-classes 10 \
  --client 0=/path/to/client0/pseudo_labels \
  --client 1=/path/to/client1/pseudo_labels \
  --client 2=/path/to/client2/pseudo_labels \
  --out dynamic_quality_aware_classwise_aggregation/stats/phase1_round001.json
```

Supported text rows include:

```text
cls x y w h
cls x y w h confidence
image_id cls x y w h confidence
```

## Fast Pilot Defaults

The default runner settings target a roughly 10-12 hour pilot on a 2x RTX 6000 Ada node:

```bash
python3 dynamic_quality_aware_classwise_aggregation/run_dqa_cwa_fedsto.py
```

This is equivalent to:

```bash
python3 dynamic_quality_aware_classwise_aggregation/run_dqa_cwa_fedsto.py \
  --warmup-epochs 8 \
  --phase1-rounds 15 \
  --phase2-rounds 35 \
  --batch-size 64 \
  --workers 0 \
  --gpus 2 \
  --min-free-gib 70
```

For a more stable 24-hour pilot, override these with `--warmup-epochs 20 --phase1-rounds 40 --phase2-rounds 60`. For paper-scale reproduction, override these with `--warmup-epochs 50 --phase1-rounds 100 --phase2-rounds 150`.

By default, DQA-CWA starts in phase 1, so every federated round after warm-up is
true DQA. If a required stats file is missing, the runner stops instead of
silently pretending the proposed method ran. For smoke tests only, pass
`--fallback-fedavg-without-stats`.

## Paper-Style Evaluation

The DQA runner reuses `setup_fedsto_exact_reproduction.py`, so it inherits the
same paper-alignment fixes as the FedSTO baseline:

- pseudo-label classification loss is enabled (`Lu_cls`-side mismatch reduced)
- per-weather labeled validation splits are materialized for `cloudy`,
  `overcast`, `rainy`, `snowy`, plus `total`

After a run, evaluate DQA checkpoints with the shared paper-style protocol:

```bash
python3 dynamic_quality_aware_classwise_aggregation/evaluate_paper_protocol.py
```

This writes results under:

```text
dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa_cwa/validation_reports/
```

You can also target specific checkpoints or splits, for example:

```bash
python3 dynamic_quality_aware_classwise_aggregation/evaluate_paper_protocol.py \
  --splits cloudy,total \
  --checkpoint final=dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa_cwa/global_checkpoints/phase2_round035_global.pt
```
