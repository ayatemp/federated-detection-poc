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
