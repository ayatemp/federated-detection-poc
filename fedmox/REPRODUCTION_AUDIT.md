# FedMox Reproduction Audit

Verdict after reading the full paper and supplementary material: this directory
is **not 100/100** as a full FedMox reproduction. It is a paper-faithful
algorithmic reproduction layer, but it does not yet reproduce the full detector
training stack or the reported metrics.

## Score

**58 / 100 for full-paper reproduction.**

| Area | Score | Reason |
| --- | ---: | --- |
| PSSFL protocol and aggregation equations | 23 / 25 | Warm-up, client sampling, weighted FedAvg, Soft Mixture, server-after-client order, backbone freezing, and task-head exchange are implemented. True client parallelism is represented as a callable loop rather than a COALA simulation. |
| MoE mechanism | 14 / 20 | Spatial 1x1 top-1 routing and fixed-dimensional ROI top-1 routing are implemented semantically. The generic module still computes all experts before masking, so it does not reproduce the paper's single-expert-level compute claim. Router gradient handling is also an implementation choice because the paper specifies hard-max routing but not the estimator used in code. |
| Paper constants, splits, and reported targets | 18 / 20 | Dataset counts, expert counts, thresholds, optimizer settings, and reported target metrics are captured in config files and checked. Exact random seeds and selected alpha for final runs are not published in the text. |
| Detector and SSL training integration | 3 / 25 | The actual MMDetection Faster R-CNN + ViT-Adapter + Soft Teacher + COALA training implementation is not present. Only framework-neutral hooks and constants exist. |
| Empirical result reproduction | 0 / 10 | No 50-round BDD100K/SODA10M/Cityscapes run has been executed from this directory, so the paper tables are targets, not reproduced results. |

## Corrected After Audit

| Issue | Why it mattered | Fix |
| --- | --- | --- |
| Federated state used the whole model by default | The paper defines the global FL model as the task head `w`; the frozen FM backbone is sent only once before FL. | `PSSFLRunner` now exchanges a selected task-head state dict by default and excludes keys whose path contains `backbone`. |
| Backbone freezing was only a helper, not part of runner setup | The paper freezes the FM backbone before initializing and training the detection head. | `PSSFLRunner` now freezes `backbone` parameters on initialization by default. |
| Literal hard argmax made the router untrainable | The paper uses hard-max top-1 routing, but a raw argmax has no useful gradient for learning the router. | MoE modules now use hard forward routing with a straight-through softmax backward path by default. |
| MoE was described too generously | The module matched top-1 output semantics but computed all experts before masking. | The code and audit now explicitly mark this as semantic, not FLOP-level, reproduction; exact sparse compute remains a detector-specific gap. |
| Non-floating state tensors were averaged | PyTorch state dicts can include counters such as integer `num_batches_tracked`; averaging them is not meaningful. | Aggregation copies non-floating tensors instead of averaging or soft-mixing them. |
| Detector details were incomplete in config | The supplementary material gives concrete Faster R-CNN, ViT-Adapter, RPN, ROI, Soft Teacher, and optimizer settings. | `FedMoxPaperConfig` now includes those paper values. |
| `soft_mixture_alpha=0.5` was an arbitrary default | The paper treats alpha as a tunable hyperparameter and does not publish one default value in text. | Alpha is now required via `PSSFLRunner(..., soft_mixture_alpha=...)` or `FedMoxPaperConfig.soft_mixture_alpha`. |
| Reported metrics were not captured as verification targets | A reproduction needs known target values for paper-table comparison. | `configs/paper_results.toml` records the main, scaling, and ablation target totals. |
| Test path assumed pytest was installed | The environment does not include pytest. | Added `scripts/run_core_tests.py` to run the same checks with the standard library. |

## Blocking Gaps To Reach 100

The local package is now a paper-faithful algorithmic implementation, but a full
empirical reproduction still requires:

- MMDetection Faster R-CNN integration.
- ViT-Adapter-Small with DINOv2 and MS-COCO adapter pretraining.
- Soft Teacher training loop.
- COALA-style federated simulation.
- Exact BDD100K, SODA10M, and Cityscapes materialization and random sampling.
- Official seeds or official code. The arXiv page exposes the paper and TeX
  source, but the paper metadata did not provide an official GitHub repository.

Because those pieces are not fully specified in executable form by the paper
artifact, claiming bit-level or metric-level "100%" reproduction would be
misleading. The package now explicitly separates verified paper equations and
protocol constants from the external training stack required for full results.
