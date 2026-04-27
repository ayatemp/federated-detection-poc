# EfficientTeacher Single-Client Baseline

This directory adds a DQA-matched EfficientTeacher baseline without changing the
existing DQA/FedSTO implementation.

The protocol is intentionally narrow:

- one unlabeled client, defaulting to `client_0_overcast`
- the same BDD100K paper20k split used by DQA/FedSTO
- the same EfficientTeacher YOLOv5L vendor tree and pretrained checkpoint
- a cloudy labeled server warm-up
- one pseudo-label client epoch per round
- one labeled server-GT epoch after every client epoch
- ET and LocalEMA notebooks can seed warm-up from the completed DQA warm-up checkpoint
- plain ET defaults to no cross-round local EMA carryover
- LocalEMA is available as a separate comparison run
- the same phase schedule used by the current guarded DQA notebook:
  `warmup=15`, `phase1=14`, `phase2=27`, `batch=64`, `gpus=2`

The runner still uses the FedSTO checkpoint helpers for phase-1 backbone-only
aggregation. Because there is only one client, phase-2 aggregation is equivalent
to taking that client checkpoint before the server GT update.

## Notebooks

```text
efficient_teacher/01_efficient_teacher_training.ipynb
efficient_teacher/00_localema_training.ipynb
efficient_teacher/01_2_efficient_teacher_evaluation.ipynb
```

Regenerate them after editing notebook templates with:

```bash
python3 efficient_teacher/generate_notebooks.py
```

## CLI

Dry-run the full command/config path:

```bash
python3 efficient_teacher/run_efficient_teacher_single_client.py --dry-run
```

Launch or resume the default DQA-matched run:

```bash
python3 efficient_teacher/run_efficient_teacher_single_client.py
```

Launch or resume the LocalEMA comparison:

```bash
python3 efficient_teacher/run_efficient_teacher_single_client.py \
  --workspace-root efficient_teacher/efficientteacher_localema \
  --local-ema
```

Evaluate selected checkpoints with the shared paper-style per-weather protocol:

```bash
python3 efficient_teacher/evaluate_paper_protocol.py
```

Outputs are written under:

```text
efficient_teacher/efficientteacher_single_client/
efficient_teacher/efficientteacher_localema/
```

## Reproduction Notes

This is a controlled reproduction scaffold rather than a bitwise guarantee. It
pins the dataset interface, model family, schedule, client count, and checkpoint
flow to the local DQA/FedSTO code. It is not an exact official EfficientTeacher
paper reproduction because the model, dataset split, round schedule, and
server-GT update are deliberately matched to DQA. GPU kernels, DDP ordering,
library versions, and EfficientTeacher augmentations can still introduce small
non-bitwise differences across machines.
