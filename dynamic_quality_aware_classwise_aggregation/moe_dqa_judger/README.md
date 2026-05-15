# MOE x DQA Judger

This workspace is for developing the DQA-SoftMoX judger before putting it into
the full FL loop.  The goal is to learn how to mix three checkpoints:

- `G_t`: previous global checkpoint
- `A_t`: DQA client aggregate checkpoint
- `S_t`: server repaired checkpoint

The first probe uses existing artifacts from `moe_dqa/output/01_dqa_fedmox_yolo_full`
and builds a module-wise soft mixture:

```text
body = mix(G_t, A_t, S_t)
head = mix(G_t, A_t, S_t)
moe  = mix(G_t, A_t, S_t)
```

The initial `judger_v0` is intentionally small.  It learns from the historical
round traces as a bootstrap model, predicts body/head/moe mixture weights, and
writes one deployable mixed checkpoint per round.  The notebook can run two
rounds first and extend to five rounds when the two-round probe is promising.

## Notebooks

- `notebooks/01_judger_probe.ipynb`: bootstrap judger from existing round traces.
- `notebooks/02_mix_weight_optimizer.ipynb`: black-box coefficient search over
  module-wise `G/A/S` mixtures, using mini validation and full-total confirmation.
- `notebooks/03_mix_judger_policy.ipynb`: trains a reusable score judger from
  notebook 02 trials, selects mixture coefficients from generated candidates, and
  evaluates the selected checkpoints on the full total split.

## Current Learned Policy

The best policy found so far is not a fixed mixture:

```text
round 1: use S_t / server repair
round 2: body = 0.65G + 0.25A + 0.10S
         head = 0.20G + 0.10A + 0.70S
         moe  = 0.15G + 0.75A + 0.10S
round 3+: freeze to G_t when source-anchor drift is detected
```

This matches the observed failure mode: repeated repair/self-training starts to
drag the source-anchor metrics down after the second round, so the learned
selector needs a drift guard rather than blindly replacing the parent model with
the latest repaired checkpoint.
