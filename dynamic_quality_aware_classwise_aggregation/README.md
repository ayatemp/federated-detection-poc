# Dynamic Quality-Aware Class-wise Aggregation

This directory contains the runnable scaffold for the DQA-CWA research idea.
It is intentionally separate from `navigating_data_heterogeneity` so the FedSTO
baseline can keep running unchanged.

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

Generated files are written under:

```text
dynamic_quality_aware_classwise_aggregation/efficientteacher_dqa_cwa/
dynamic_quality_aware_classwise_aggregation/stats/
```

## Dry Run

```bash
python3 dynamic_quality_aware_classwise_aggregation/run_dqa_cwa_fedsto.py --dry-run
```

## Stats Format

For phase 2 round 1, the runner expects:

```text
dynamic_quality_aware_classwise_aggregation/stats/phase2_round001.json
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
  --out dynamic_quality_aware_classwise_aggregation/stats/phase2_round001.json
```

Supported text rows include:

```text
cls x y w h
cls x y w h confidence
image_id cls x y w h confidence
```

## Full Run

```bash
python3 dynamic_quality_aware_classwise_aggregation/run_dqa_cwa_fedsto.py \
  --warmup-epochs 50 \
  --phase1-rounds 100 \
  --phase2-rounds 150 \
  --batch-size 64 \
  --workers 0 \
  --gpus 2
```

By default, DQA-CWA starts in phase 2. If a required stats file is missing, the
runner stops instead of silently pretending the proposed method ran. For smoke
tests only, pass `--fallback-fedavg-without-stats`.

