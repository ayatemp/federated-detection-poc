# MOE x DQA Experiments

This directory is a clean workspace for the MoE and DQA line of experiments.
It is separate from the older `scene_daynight_dqa/notebooks` and
`scene_daynight_dqa/aggressive_dqamox` directories so the current research
thread has a smaller surface area.

## 01 DQA-FedMoX-YOLO-Full

Notebook:

```text
notebooks/01_dqa_fedmox_yolo_full.ipynb
```

Generator:

```text
scripts/make_01_dqa_fedmox_yolo_full_notebook.py
```

The notebook runs one full, no-early-stop protocol that produces three
comparison rows from the same workspace:

1. warmup only
2. warmup plus source-GT server repair
3. DQA-routed MoE YOLO full training

The full run uses FedMoX-style 33% client sampling, FedSTO-style persistent
local EMA teachers, and DQA-guided expert/router aggregation.
