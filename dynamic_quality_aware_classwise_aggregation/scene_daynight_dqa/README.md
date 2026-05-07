# Scene-Daynight DQA

This subproject keeps the next DQA line separate from the older exploratory
notebooks in `dynamic_quality_aware_classwise_aggregation/`.

The goal is to test DQA under a more explicit heterogeneity condition:

- scene heterogeneity: highway, city street, residential
- time/domain heterogeneity: daytime vs night
- natural class and label-density shifts induced by those splits

Existing top-level notebooks are left untouched.  New notebooks, runners, and
outputs for this line live here:

```text
dynamic_quality_aware_classwise_aggregation/scene_daynight_dqa/
```

## Layout

| path | role |
| --- | --- |
| `notebooks/` | runnable notebooks |
| `scripts/` | setup and runner scripts |
| `output/` | generated data lists, checkpoints, reports, and stats |
| `EXPERIMENT_CONDITIONS.md` | fixed setup and evaluation rules |
| `01_EXPERIMENT_INDEX.md` | completed 01-series notebook map, results, and interpretation |

## 01 Notebook

```text
notebooks/01_repair_oriented_scene_daynight_dqa.ipynb
```

The first notebook uses a repair-oriented loop:

1. generate strict stable pseudo labels from the current repaired global model
2. train six clients with source GT plus target pseudo labels
3. aggregate clients with server-anchored DQA-CWA v2
4. repair the aggregate on supervised source-cloudy GT
5. evaluate the repaired global checkpoint and carry it into the next round

Primary metrics are repaired global mAP values, not client-only scores.

## 01 Series Result Index

The completed `01`, `01_0`, `01_1`, and `01_2` notebooks are summarized in:

```text
01_EXPERIMENT_INDEX.md
```

Use that file as the entry point before rerunning any 01-series notebook.

## 02 Notebook

```text
notebooks/02_head_to_full_long_dqa.ipynb
```

The active 02 notebook tests a FedSTO-style head-to-full DQA schedule:
long Phase1 head/neck-only client adaptation followed by a short Phase2
full-model low-LR burst.  It uses final-focused paper-protocol evaluation so a
30+2 round pilot remains practical.
