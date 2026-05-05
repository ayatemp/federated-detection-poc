# SCoLQ: Source-Calibrated Localization Quality Judge

SCoLQ is a bbox-quality judge for DQA phase 2.

The goal is to stop treating teacher confidence as localization quality.  SCoLQ
learns, on source/server labeled data only, whether a teacher pseudo box is
likely to have good IoU with ground truth.  The resulting score can then be used
in DQA phase 2 to gate or weight pseudo boxes without selecting checkpoints by
target validation GT.

Notebook entry point:

```text
dynamic_quality_aware_classwise_aggregation/source_calibrated_localization_quality/01_train_and_select_scolq.ipynb
dynamic_quality_aware_classwise_aggregation/source_calibrated_localization_quality/02_train_and_validate_round_stable_scolq.ipynb
```

Planned artifacts:

```text
artifacts/scolq_dataset.csv
artifacts/scolq_best.joblib
artifacts/rscolq_best.joblib
reports/model_ranking.csv
reports/feature_importance.csv
reports/rscolq_model_ranking.csv
reports/rscolq_0836_round_diagnostics.csv
predictions/
```

The first notebook compares feature families instead of committing to one
hand-picked quality signal:

- confidence-only baseline
- geometry features
- per-image and per-class prediction context
- same-class and cross-class overlap/crowding
- source class priors
- optional augmented-inference agreement
- optional multi-scale agreement

The selection criterion is source-only: quality AP, calibration, and precision
at useful pseudo-box coverage.  Target GT is intentionally not used.

The second notebook builds **R-SCoLQ** (Round-Stable SCoLQ).  It keeps the
source-only bbox quality model, but adds an anti-inflation round policy from
08_3_6 pseudo-label stats so that repeated phase-2 rounds do not reward growing
pseudo-box counts and rising self-scores by default.  Target validation metrics
from 08_3_6 are used only as diagnostics, not as artifact selection inputs.
