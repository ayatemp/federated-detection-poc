# FedMox Reproduction Audit

Verdict: the first pass was **not** a full FedMox reproduction. It captured the
main equations, but it missed several paper-critical implementation constraints.
This audit records what was corrected and what still cannot be honestly claimed
without the missing external artifacts.

## Corrected After Audit

| Issue | Why it mattered | Fix |
| --- | --- | --- |
| Federated state used the whole model by default | The paper defines the global FL model as the task head `w`; the frozen FM backbone is sent only once before FL. | `PSSFLRunner` now exchanges a selected task-head state dict by default and excludes keys whose path contains `backbone`. |
| Literal hard argmax made the router untrainable | The paper uses hard-max top-1 routing, but a raw argmax has no useful gradient for learning the router. | MoE modules now use hard forward routing with a straight-through softmax backward path by default. |
| Non-floating state tensors were averaged | PyTorch state dicts can include counters such as integer `num_batches_tracked`; averaging them is not meaningful. | Aggregation copies non-floating tensors instead of averaging or soft-mixing them. |
| Detector details were incomplete in config | The supplementary material gives concrete Faster R-CNN, ViT-Adapter, RPN, ROI, Soft Teacher, and optimizer settings. | `FedMoxPaperConfig` now includes those paper values. |
| Test path assumed pytest was installed | The environment does not include pytest. | Added `scripts/run_core_tests.py` to run the same checks with the standard library. |

## Still Not A 100% Empirical Reproduction

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
