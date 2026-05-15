# FedMox Reproduction

This top-level workspace implements the paper core of **FedMox: Federated
Mixture of Experts** from arXiv:2508.16568, "Closer to Reality: Practical
Semi-Supervised Federated Learning for Foundation Model Adaptation".

The implementation is structured as a paper-faithful reproduction layer:

- `fedmox/moe.py`: sparse top-1 MoE modules.
  - RPN-style feature maps use a spatial `1x1` convolution router.
  - ROI-style fixed vectors use a traditional linear top-1 router.
- `fedmox/aggregation.py`: weighted FedAvg and FedMox Soft Mixture.
- `fedmox/pssfl.py`: the PSSFL execution order from the supplementary algorithm.
- `configs/`: BDD100K, SODA10M, and Cityscapes protocol facts from the paper.
- `PAPER_TRACE.md`: line-by-line reproduction audit against the paper source.

## Quick Checks

```bash
cd /app/Object_Detection
PYTHONPATH=fedmox python fedmox/scripts/verify_paper_trace.py
PYTHONPATH=fedmox pytest -q fedmox/tests
```

## Reproduction Scope

This directory faithfully implements the algorithmic components that the paper
defines: spatial sparse MoE, ROI top-1 MoE, FedAvg, Soft Mixture, PSSFL ordering,
and the published dataset/training protocol defaults.

The full detector training stack still requires the heavy external dependencies
named by the paper: MMDetection, ViT-Adapter-Small/DINOv2, Soft Teacher, COALA,
and the driving datasets. Because the paper does not publish an official code
repository in the arXiv metadata, this repo keeps those integrations as explicit
adapters rather than pretending an unverified MMDetection config is bit-identical.
