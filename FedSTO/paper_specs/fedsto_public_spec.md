# FedSTO Public Specification

## Confirmed Sources

- Paper: https://arxiv.org/abs/2310.17097
- NeurIPS PDF: https://papers.nips.cc/paper_files/paper/2023/file/066e4dbfeccb5dc2851acd5eca584937-Paper-Conference.pdf
- Supplemental: https://papers.neurips.cc/paper_files/paper/2023/file/066e4dbfeccb5dc2851acd5eca584937-Supplemental-Conference.pdf
- Official public repository: https://github.com/Kthyeon/ssfod
- Papers with Code entry: https://paperswithcode.com/paper/navigating-data-heterogeneity-in-federated

## Publicly Stated FedSTO Design

| Item | Public specification | Implementation in this directory |
|---|---|---|
| Task | Semi-supervised federated object detection | Implemented |
| Label placement | Server has labeled data; clients have unlabeled data only | Implemented |
| BDD non-IID setup | Server cloudy, clients overcast/rainy/snowy | Implemented with BDD Kaggle `partly cloudy` as cloudy |
| Data scale | About 20k BDD points, 1 server + 3 clients | Implemented as 4,881 server train + 15,000 client train |
| Detector | YOLOv5 Large | Implemented with EfficientTeacher YOLOv5L checkpoint |
| Warmup | 50 supervised server rounds/epochs | Implemented |
| Phase 1 | 100 rounds selective backbone training | Implemented |
| Phase 2 | 150 rounds full-parameter training | Implemented |
| Local epoch | 1 local epoch per round | Implemented |
| Client participation | Algorithm samples clients; 1-server/3-client BDD ratio not explicitly stated | Implemented as full participation of all 3 clients per round |
| Client pseudo labeler | Local EMA model per client, reinitialized from global after server broadcast | Implemented as default; cross-round persistence is ablation-only |
| EMA decay | 0.999 | Implemented |
| EMA schedule | Fixed alpha in paper equation | Implemented as fixed EMA; EfficientTeacher cosine EMA and ModelEMA ramp disabled for FedSTO configs |
| Pseudo label assigner | Semi-Efficient Teacher-style PLA | Implemented via EfficientTeacher SSOD trainer |
| Ignore thresholds | Low/high 0.1/0.6 | Implemented |
| NMS | confidence 0.1, IoU 0.65 | Implemented |
| Loss balance | class 0.3, object 0.7, anchor threshold 4.0 | Implemented |
| Augmentation | Mosaic, LR flip, large scale jitter, graying, Gaussian blur, cutout, color conversion | Implemented as close EfficientTeacher SSOD augmentation path; exact graying/Gaussian/color-conversion parity depends on EfficientTeacher defaults |
| Orthogonal enhancement | Non-backbone orthogonal regularization in Phase 2 | Implemented as gradient-level spectral orthogonal regularization |
| Orthogonal coefficient | Not publicly specified | Uses `1e-4`; must be treated as an implementation choice |
| Evaluation | mAP@0.5 on cloudy/overcast/rainy/snowy/total | Implemented |

## Public Results To Compare Against

BDD100K non-IID FedSTO mAP@0.5:

| Split | Paper |
|---|---:|
| Cloudy | 0.596 |
| Overcast | 0.607 |
| Rainy | 0.590 |
| Snowy | 0.580 |
| Total | 0.593 |

BDD100K IID FedSTO mAP@0.5:

| Split | Paper |
|---|---:|
| Cloudy | 0.591 |
| Overcast | 0.634 |
| Rainy | 0.614 |
| Snowy | 0.595 |
| Total | 0.609 |

## Known Public-Artifact Blockers

These are not engineering omissions in this directory; they are unavailable public artifacts:

1. The official repository currently does not contain the FedSTO training implementation.
2. The exact BDD100K sample IDs selected by the authors are not published.
3. The original runtime environment and any local EfficientTeacher edits used by the authors are not published.

For that reason, the correct standard is **maximal public-spec reproduction**, not bit-for-bit author-code identity. This is not honestly 100% identical to the authors' private run.
