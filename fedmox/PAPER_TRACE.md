# FedMox Paper Trace

Source checked: arXiv:2508.16568, submitted 2025-08-22.

## Implemented Exactly

| Paper requirement | Local implementation |
| --- | --- |
| Freeze foundation-model backbone; train task head only | Exposed as the expected model boundary in `PSSFLRunner`; train callables receive and update the task-head state only. |
| Warm-up server training before FL | `FedMoxPaperConfig.warmup_epochs = 50`; first step in `PSSFLRunner.run`. |
| Each round selects `M = r * N` online clients | `PSSFLRunner.select_clients`, default `client_sampling_ratio = 0.33`. |
| Client-side low-resolution unsupervised training | Runner delegates to `client_train(..., client_epochs_per_round=1)`; config stores `(640, 360)`. |
| Weighted FedAvg `sum_i n_i / n * w_i` | `fedmox.aggregation.fedavg`. |
| Soft Mixture `alpha * w^(t) + (1-alpha) * wbar^(t+1)` | `fedmox.aggregation.soft_mixture`. |
| Server-side high-resolution supervised training after aggregation | Runner loads mixed weights, then calls `server_train(..., server_epochs_per_round=1)`; config stores `(1280, 720)`. |
| Spatial sparse MoE router for variable-size feature maps | `SpatialTop1MoE`: `Conv2d(C, K, 1)` + hard top-1 mask for `[B, C, H, W]`. |
| Traditional top-1 ROI router for fixed-size ROI features | `Top1MoE`: `Linear(D, K)` + hard top-1 mask. |
| BDD100K/SODA10M/Cityscapes expert counts | `FedMoxPaperConfig.experts_for_dataset`: 4, 3, 3. |
| Soft Teacher pseudo-label thresholds and unsupervised weight | Stored in `FedMoxPaperConfig`. |
| Dataset split counts and client-domain allocation | YAML files in `configs/`. |

## Paper Details Captured

- BDD100K: server domain Cloudy; client domains Overcast, Rainy, Snowy; 2k labeled server images; 5k/5k/8k unlabeled client images; 1.5k validation samples split 300/600/300/300; K=4; metric mAP@50.
- SODA10M: server domain Clear; client domains Overcast and Rainy; 2k labeled server images; 10k/8k unlabeled client images; 2.3k validation samples split 1000/1000/300; K=3; metric mAP.
- Cityscapes: city-based domain generalization; 2k server images from 18 cities; 18k client images from 23 cities; 500 test images from 3 cities; K=3; metric mAP@50.
- Training defaults: AdamW, lr 0.0001, weight decay 0.05, cosine annealing with 5-epoch warmup, 50 warm-up epochs, 50 FL rounds, 1 client epoch and 1 server epoch per round.

## Not Claiming 100% Bit-Level Reproduction Yet

The paper does not provide official source code or exact random seeds in the arXiv artifact. The detector stack depends on MMDetection, ViT-Adapter, Soft Teacher, COALA, and dataset materialization details. This directory therefore reaches a paper-faithful algorithmic reproduction and records the remaining external integration points explicitly.

To claim full empirical reproduction, the next required step is to bind these modules into a concrete MMDetection Faster R-CNN + ViT-Adapter config and run the published 50-round protocols on the exact dataset splits.
