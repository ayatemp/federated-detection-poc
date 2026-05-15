# Codex Goal: DQA-MoE To mAP50 0.60

作成日: 2026-05-11

## 目的

FedMoX/FedSTO に近い scene-daynight federated object detection 設定で、DQA-MoE/DQA-MoX 系の total `mAP50 >= 0.600` を目指す。

Codex は単に既存 notebook を実行するのではなく、各 trial ごとに以下を自律的に行う。

1. 直近までの結果を読み、何が効いた/効かなかったかを短くまとめる。
2. 関連論文を再確認し、使う発想を明示する。
3. MoE 周りで攻めた仮説を1つ決める。
4. notebook と研究メモを生成する。
5. 実行する。
6. warmup と比較し、明らかに伸びない trial は打ち切る。
7. 結果を CSV/Markdown と Discord に記録する。
8. `mAP50 >= 0.600` に届かなければ、原因を更新して次 trial に進む。

## 現時点の結果からの前提

- best: `22_twoepoch_conservative_neckhead_single_injection_dqamox`
  - final `mAP50=0.462`
  - final `mAP50:95=0.260`
- output-space MoE は弱い。
  - 06 の WBF/output fusion は個別 expert より落ちた。
- full latent MoE は drift しやすい。
  - 08 は warmup `0.460` から final `0.394` へ悪化。
- single injection + neck/head + 強い anchor は安定する。
  - 18-22 は warmup 近傍を維持し、22 だけ小さく上回った。
- 残ボトルネックは `highway_night`。

## 論文から使う発想

- FedMoX/PSSFL:
  - sparse MoE
  - spatial router
  - Soft-Mixture
  - server labeled / client unlabeled mismatch の安定化
- FedSTO:
  - server-only labels + client-only unlabeled non-IID
  - selective training then full-parameter training
- SSOD pseudo-label quality:
  - classification confidence だけでは不足
  - localization quality / uncertainty を pseudoGT 選別と loss 重みに使う
  - class-specific/adaptive threshold が class imbalance に効く

## 打ち切りルール

基本ルール:

- final/途中評価で DQA-MoE branch が warmup を超えない場合、その trial は継続価値が低い。
- warmup 超えが見えないまま long schedule に入る場合は、次の仮説へ切り替える。
- `mAP50=0.600` との差が大きく、かつ warmup からの差分が実質ゼロの trial は、攻め方が間違っていると判断する。

運用上は、まず `warmup_map50`, `dqa_aggregate_map50`, `dqa_repair_map50`, `best_map50` を記録し、warmup との差を Discord に出す。明確な伸びがなければ、round/epoch を増やすより MoE routing, pseudoGT quality, split-specific expert design を変える。

## 次に優先する仮説

1. localization-quality strict-to-open MoE
   - 前半は pseudo box の stability を強く要求する。
   - 後半だけ threshold を少し開く。
   - box loss は小さく保ち、誤った bbox regression の蓄積を避ける。

2. highway-night specialist routing
   - total average ではなく worst split を改善対象にする。
   - night/highway expert を head/neck delta に限定する。

3. high-capacity soft router
   - K=6/K=8, top-k=2/3 で expert collapse を避ける。
   - ただし full latent drift を避けるため、初期は neck/head に閉じる。

