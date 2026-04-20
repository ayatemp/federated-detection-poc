# FedSTO Paper Gap Audit

This audit compares the older `01_2_fedsto_paper_aligned.ipynb` prototype and the newer
`03_fedsto_exact_reproduction.ipynb` reproduction route with the public FedSTO paper and official
repository metadata.

Sources:
- Paper: https://papers.nips.cc/paper_files/paper/2023/file/066e4dbfeccb5dc2851acd5eca584937-Paper-Conference.pdf
- Official repository: https://github.com/Kthyeon/ssfod

## Current Match Summary

`01_2_fedsto_paper_aligned.ipynb` remains a paper-aligned prototype.

`03_fedsto_exact_reproduction.ipynb` is now the production reproduction route. It vendors the
official SSFOD dataset repository metadata and the EfficientTeacher YOLOv5L SSOD trainer, then
adds the FedSTO-specific controls that are required by the paper: phase-wise backbone-only
training, local client EMA persistence, full-parameter Phase 2, and spectral orthogonal
regularization on non-backbone parameters.

Approximate implementation fidelity of `01_2`: 65-75%.
Approximate implementation fidelity of `03`: public-code maximum, except for assets the authors
did not publish.

This is not a scientific score. It means the notebook now reproduces most high-level experimental
structure, data partitioning, model family, and training schedule, but it does not yet reproduce the
paper's loss-level SSOD trainer or gradient-level ORN implementation.

## Same Or Very Close

| Component | Current state | Status |
|---|---|---|
| SSFOD setting | Server has labeled data, clients have unlabeled data only | Same |
| Non-IID weather split | Server cloudy-equivalent `partly cloudy`; clients `overcast`, `rainy`, `snowy` | Very close |
| Number of clients | 1 server + 3 clients | Same |
| BDD100K scale | 4,881 labeled server train + 15,000 unlabeled client train = 19,881 | Very close to paper's approx. 20K |
| BDD100K classes | 10-class BDD detection mapping | Same target class space |
| Detector family | YOLOv5 Large-compatible checkpoint (`yolov5lu.pt`) | Close, but not exact original YOLOv5L stack |
| Training rounds | warm-up 50, phase1 100, phase2 150 | Same |
| Local epochs | 1 local epoch per round | Same |
| Learning rate | constant `lr0=0.01` | Same stated value |
| NMS thresholds | confidence 0.1, IoU 0.65 | Same stated values |
| EMA decay | 0.999 | Same stated value |
| Phase 1 concept | Client-side backbone selective training; server updates with labeled data | Same high-level algorithm |
| Phase 2 concept | Full-parameter training with neck/head ORN target | Same high-level algorithm |
| Alternating server/client training | Clients train on unlabeled pseudo labels, server trains on labeled data | Same high-level framework |

## Different Or Partial

| Component | Paper target | Current implementation | Gap |
|---|---|---|---|
| YOLO implementation | YOLOv5 Large as used in the paper's original stack | Ultralytics v8 `YOLO("yolov5l.pt")` resolves to `yolov5lu.pt` | Need original YOLOv5/Efficient Teacher trainer stack for exactness |
| Pseudo label assigner | Semi-Efficient Teacher-style PLA with adaptive unsupervised loss | Uses high-confidence hard YOLO labels only | Missing soft objectness and unreliable pseudo-label loss routing |
| Unsupervised loss | `Lu = Lu_cls + Lu_reg + Lu_obj`, with threshold-dependent cls/reg/objectness behavior | Standard Ultralytics supervised loss on generated pseudo-label text files | Major mismatch |
| Soft objectness | Scores below tau1 -> zero, above tau2 -> pseudo objectness, middle -> soft objectness | Middle/low scores are not trained with soft targets | Major mismatch |
| Regression routing | Regression also trained for objectness > 0.99 cases | Only boxes kept as hard labels are trained | Partial |
| ORN | Orthogonal regularization during client/server orthogonal updates | Post-training orthogonal projection on neck/head weights | Major mismatch |
| Local EMA | Each client maintains a local EMA model across its local training trajectory | Current notebook creates fresh EMA teacher from broadcast model each selected round | Partial; must persist per-client EMA state |
| Layer partition | Paper's YOLOv5 backbone/neck/head split in original architecture | Heuristic: backbone range 0-9, neck 10-21, head last layer | Needs exact split from original code/config |
| Data split exactness | Authors selected 20,000 BDD points | Local mirror yields 19,881 using available cloudy-equivalent labeled train records | Very close, but not exact same sample IDs |
| Augmentations | Mosaic, horizontal flip, large scale jitter, graying, Gaussian blur, cutout, color conversion | Ultralytics default augmentations, not fully pinned to paper list | Partial |
| Loss weights | class/object balance 0.3/0.7, anchor threshold 4.0 | Not explicitly pinned in current trainer args | Partial |
| Evaluation | Paper reports mAP@0.5 across server/client weather splits and total | Current history tracks server validation mAP only | Partial |
| Baselines | Fully supervised, partially supervised, global model, local EMA model, FedAvg/FedProx/FedBN comparisons | Not implemented in `01_2` | Missing for paper-level reproduction |

## What Is Needed For 100%

Completed in `03`:

1. Move away from Ultralytics high-level `model.train()` for client unsupervised training.
2. Implement or port an Efficient Teacher / YOLOv5 trainer with:
   - pseudo-label assigner,
   - threshold-dependent classification/regression/objectness losses,
   - soft objectness targets,
   - objectness-gated regression,
   - class/object balance and anchor threshold matching the paper.
3. Implement ORN as a gradient-level regularization term on neck/head during Phase 2, not as a post-hoc projection.
4. Keep persistent client-local EMA teachers across rounds and update them according to the EMA equation.
5. Pin the original YOLOv5 Large architecture/checkpoint/training code instead of using `yolov5lu.pt` through Ultralytics v8.

Still dependent on unpublished or external material:

6. Recreate exact sample IDs if the authors' split files are available; otherwise, the local split is paper-scale but not sample-identical.
7. Add paper baselines and evaluation protocol:
   - fully supervised centralized,
   - partially supervised centralized,
   - vanilla SSFL/global pseudo labeler,
   - local EMA pseudo labeler,
   - FedSTO,
   - per-weather and total mAP@0.5.

## Bottom Line

`01_2` is a strong prototype.

`03` is the serious reproduction path: official SSFOD data setup metadata, EfficientTeacher YOLOv5L
SSOD, paper-scale BDD100K split, 10 classes, 50/100/150 schedule, local EMA, selective backbone
training, full-parameter Phase 2, and non-backbone orthogonal regularization. The official SSFOD
repository currently says the training implementation is "Still in progress. To be uploaded!!", so a
bit-for-bit reproduction of unpublished author code and exact private sample IDs is not possible from
public artifacts alone.
