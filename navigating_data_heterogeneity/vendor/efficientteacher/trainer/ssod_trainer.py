#Copyright (c) 2023, Alibaba Group
"""
Train an object detection model using domain adaptation  @ruiyang

"""
from cgitb import enable
from email.utils import encode_rfc2231
import json
import os

from .trainer import Trainer
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
# import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW, SGD, lr_scheduler
from tqdm import tqdm
from datetime import timedelta

# import val # for end-of-epoch mAP
from models.backbone.experimental import attempt_load
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_git_status, check_img_size, check_requirements, \
    check_file, check_yaml, check_suffix, print_args, print_mutation, set_logging, one_cycle, colorstr, methods
from utils.downloads import attempt_download
from models.loss.loss import DomainLoss, TargetLoss
from models.loss import build_ssod_loss
from utils.plots import plot_labels, plot_evolve
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, intersect_dicts, select_device, \
    torch_distributed_zero_first, is_parallel,time_sync, SemiSupModelEMA, CosineEMA
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.metrics import fitness
# from ..val import run # for end-of-epoch mAP
from utils.plots import plot_images, plot_labels, plot_results,  plot_images_debug, output_to_target
from utils.datasets_ssod import create_target_dataloader, augment_hsv, cutout
from utils.self_supervised_utils import FairPseudoLabel
from utils.labelmatch import LabelMatch 
from utils.self_supervised_utils import check_pseudo_label_with_gt, check_pseudo_label
from models.detector.yolo_ssod import Model
import torchvision
import copy
from utils.fedsto_regularization import apply_fedsto_train_scope, class_skew_head_regularization, spectral_orthogonal_regularization

LOGGER = logging.getLogger(__name__)

class SSODTrainer(Trainer):
    def __init__(self, cfg, device, callbacks, LOCAL_RANK, RANK, WORLD_SIZE):
        self.cfg = cfg
        self.set_env(cfg, device, LOCAL_RANK, RANK, WORLD_SIZE, callbacks)

        self.build_model(cfg, device)
        self.build_optimizer(cfg)
        self.build_dataloader(cfg, callbacks)
       
        LOGGER.info(f'Image sizes {self.imgsz} train, {self.imgsz} val\n'
                f'Using {self.train_loader.num_workers} dataloader workers\n'
                f"Logging results to {colorstr('bold', self.save_dir)}\n"
                f'Starting training for {self.epochs} epochs...')
        # self.no_aug_epochs = cfg.hyp.no_aug_epochs
        # burn_epochs = cfg.hyp.burn_epochs
        if cfg.SSOD.pseudo_label_type == 'FairPseudoLabel':
            self.pseudo_label_creator = FairPseudoLabel(cfg)
        elif cfg.SSOD.pseudo_label_type == 'LabelMatch':
            self.pseudo_label_creator = LabelMatch(cfg, int(self.unlabeled_dataset.__len__()/self.WORLD_SIZE), self.label_num_per_image, cls_ratio_gt= self.cls_ratio_gt)

        self.build_ddp_model(cfg, device)
        self.device = device
    
    def set_env(self, cfg, device, LOCAL_RANK, RANK, WORLD_SIZE, callbacks):
        super().set_env(cfg, device, LOCAL_RANK, RANK, WORLD_SIZE, callbacks)
        self.data_dict['target'] = cfg.Dataset.target
        self.target_with_gt = cfg.SSOD.ssod_hyp.with_gt
        self.break_epoch = -1
        self.epoch_adaptor = cfg.SSOD.epoch_adaptor
        self.da_loss_weights = cfg.SSOD.da_loss_weights
        self.cosine_ema = cfg.SSOD.cosine_ema
        self.fixed_accumulate = cfg.SSOD.fixed_accumulate
        self.dqa_pseudo_stats_out = os.getenv("DQA_PSEUDO_STATS_OUT", "").strip()
        self.dqa_client_id = os.getenv("DQA_CLIENT_ID", "").strip()
        self.dqa_phase = os.getenv("DQA_PHASE", "").strip()
        self.dqa_round = os.getenv("DQA_ROUND", "").strip()
        self.dqa_quality_mode = os.getenv(
            "DQA0834_STATS_QUALITY_MODE",
            os.getenv("DQA_STATS_QUALITY_MODE", "confidence"),
        ).strip().lower()
        self.dqa0836_scolq_enabled = (
            os.getenv("DQA0836_SCOLQ_ENABLE", "").strip().lower() in {"1", "true", "yes", "on"}
            or os.getenv("DQA0837_RSCOLQ_ENABLE", "").strip().lower() in {"1", "true", "yes", "on"}
            or os.getenv("DQA0838_RSCOLQ_ENABLE", "").strip().lower() in {"1", "true", "yes", "on"}
            or os.getenv("DQA0839_RSCOLQ_ENABLE", "").strip().lower() in {"1", "true", "yes", "on"}
        )
        self.dqa0836_scolq_model_path = os.getenv(
            "DQA0836_SCOLQ_MODEL",
            os.getenv("DQA0837_RSCOLQ_MODEL", os.getenv("DQA0838_RSCOLQ_MODEL", os.getenv("DQA0839_RSCOLQ_MODEL", ""))),
        ).strip()
        self.dqa0836_scolq_score_power = float(os.getenv(
            "DQA0836_SCOLQ_SCORE_POWER",
            os.getenv("DQA0837_RSCOLQ_SCORE_POWER", os.getenv("DQA0838_RSCOLQ_SCORE_POWER", os.getenv("DQA0839_RSCOLQ_SCORE_POWER", "1.0"))),
        ))
        self.dqa0837_rscolq_enabled = (
            os.getenv("DQA0837_RSCOLQ_ENABLE", "").strip().lower() in {"1", "true", "yes", "on"}
            or os.getenv("DQA0838_RSCOLQ_ENABLE", "").strip().lower() in {"1", "true", "yes", "on"}
            or os.getenv("DQA0839_RSCOLQ_ENABLE", "").strip().lower() in {"1", "true", "yes", "on"}
        )
        self.dqa0837_rscolq_global_multiplier = float(os.getenv(
            "DQA0837_RSCOLQ_GLOBAL_MULTIPLIER",
            os.getenv("DQA0838_RSCOLQ_GLOBAL_MULTIPLIER", os.getenv("DQA0839_RSCOLQ_GLOBAL_MULTIPLIER", "1.0")),
        ))
        self.dqa0837_rscolq_class_multipliers = os.getenv(
            "DQA0837_RSCOLQ_CLASS_MULTIPLIERS",
            os.getenv("DQA0838_RSCOLQ_CLASS_MULTIPLIERS", os.getenv("DQA0839_RSCOLQ_CLASS_MULTIPLIERS", "")),
        ).strip()
        self.dqa0838_rscolq_raw_stats = (
            os.getenv("DQA0838_RSCOLQ_RAW_STATS", "").strip().lower() in {"1", "true", "yes", "on"}
            or os.getenv("DQA0839_RSCOLQ_RAW_STATS", "").strip().lower() in {"1", "true", "yes", "on"}
        )
        self.dqa0838_last_raw_rscolq_scores = None
        self.dqa0836_scolq_bundle = None
        self.dqa0836_scolq_features = None
        self.dqa0836_source_gt_prior = None
        self.dqa0835_memory_enabled = os.getenv("DQA0835_PSEUDO_MEMORY", "").strip().lower() in {"1", "true", "yes", "on"}
        self.dqa0835_memory_path = os.getenv("DQA0835_PSEUDO_MEMORY_PATH", "").strip()
        self.dqa0835_memory_iou = float(os.getenv("DQA0835_MEMORY_IOU", "0.55"))
        self.dqa0835_memory_merge_iou = float(os.getenv("DQA0835_MEMORY_MERGE_IOU", "0.70"))
        self.dqa0835_stable_rounds = max(int(os.getenv("DQA0835_STABLE_ROUNDS", "2")), 1)
        self.dqa0835_new_score_cap = float(os.getenv("DQA0835_NEW_SCORE_CAP", "0.70"))
        self.dqa0835_matched_score_cap = float(os.getenv("DQA0835_MATCHED_SCORE_CAP", "0.74"))
        self.dqa0835_stable_score_floor = float(os.getenv("DQA0835_STABLE_SCORE_FLOOR", "0.78"))
        self.dqa0835_stable_obj_floor = float(os.getenv("DQA0835_STABLE_OBJ_FLOOR", "0.85"))
        self.dqa0835_stable_cls_floor = float(os.getenv("DQA0835_STABLE_CLS_FLOOR", "0.85"))
        self.dqa0835_max_entries_per_image = max(int(os.getenv("DQA0835_MAX_ENTRIES_PER_IMAGE", "80")), 1)
        self.dqa0835_memory = self._load_dqa0835_pseudo_memory()
        self.dqa0835_base_memory = copy.deepcopy(self.dqa0835_memory)
        self.dqa0835_seen_memory_keys = set()
        self.dqa0835_memory_matches = 0
        self.dqa0835_memory_stable = 0
        self.dqa0835_memory_total = 0
        self.dqa_track_pseudo_stats = bool(self.dqa_pseudo_stats_out) and cfg.SSOD.train_domain
        if self.dqa_track_pseudo_stats:
            self.dqa_pseudo_counts = torch.zeros(int(cfg.Dataset.nc), dtype=torch.float64)
            self.dqa_pseudo_confidence_sums = torch.zeros(int(cfg.Dataset.nc), dtype=torch.float64)
            self.dqa_pseudo_objectness_sums = torch.zeros(int(cfg.Dataset.nc), dtype=torch.float64)
            self.dqa_pseudo_class_confidence_sums = torch.zeros(int(cfg.Dataset.nc), dtype=torch.float64)
            self.dqa_pseudo_localization_sums = torch.zeros(int(cfg.Dataset.nc), dtype=torch.float64)
            self.dqa_pseudo_feature_quality_sums = torch.zeros(int(cfg.Dataset.nc), dtype=torch.float64)
            self.dqa_pseudo_feature_contrast_sums = torch.zeros(int(cfg.Dataset.nc), dtype=torch.float64)
            self.dqa_pseudo_quality_sums = torch.zeros(int(cfg.Dataset.nc), dtype=torch.float64)
        else:
            self.dqa_pseudo_counts = None
            self.dqa_pseudo_confidence_sums = None
            self.dqa_pseudo_objectness_sums = None
            self.dqa_pseudo_class_confidence_sums = None
            self.dqa_pseudo_localization_sums = None
            self.dqa_pseudo_feature_quality_sums = None
            self.dqa_pseudo_feature_contrast_sums = None
            self.dqa_pseudo_quality_sums = None

    def _dqa0835_rank_memory_path(self) -> Path | None:
        if not self.dqa0835_memory_enabled or not self.dqa0835_memory_path:
            return None
        base = Path(self.dqa0835_memory_path)
        if self.RANK in (-1, 0) and not dist.is_available():
            return base
        rank = self.RANK if self.RANK not in (-1, None) else 0
        return base.with_name(f"{base.stem}.rank{rank}{base.suffix or '.json'}")

    @staticmethod
    def _dqa0835_xywh_iou(box_a, box_b) -> float:
        ax, ay, aw, ah = [float(x) for x in box_a]
        bx, by, bw, bh = [float(x) for x in box_b]
        ax1, ay1, ax2, ay2 = ax - aw / 2.0, ay - ah / 2.0, ax + aw / 2.0, ay + ah / 2.0
        bx1, by1, bx2, by2 = bx - bw / 2.0, by - bh / 2.0, bx + bw / 2.0, by + bh / 2.0
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        area_a = max(0.0, aw) * max(0.0, ah)
        area_b = max(0.0, bw) * max(0.0, bh)
        union = area_a + area_b - inter
        return inter / union if union > 1e-12 else 0.0

    def _dqa0835_merge_memory_entries(self, entries):
        cleaned = []
        for entry in entries:
            try:
                cls_id = int(entry.get("cls", -1))
                box = [float(x) for x in entry.get("box", [])[:4]]
                if cls_id < 0 or len(box) != 4:
                    continue
                cleaned.append(
                    {
                        "cls": cls_id,
                        "box": [min(max(float(x), 0.0), 1.0) for x in box],
                        "score": min(max(float(entry.get("score", 0.0)), 0.0), 1.0),
                        "obj": min(max(float(entry.get("obj", entry.get("score", 0.0))), 0.0), 1.0),
                        "cls_conf": min(max(float(entry.get("cls_conf", entry.get("score", 0.0))), 0.0), 1.0),
                        "stability": max(int(entry.get("stability", 1)), 1),
                    }
                )
            except (TypeError, ValueError):
                continue
        cleaned.sort(key=lambda x: (x["stability"], x["score"], x["obj"], x["cls_conf"]), reverse=True)
        kept = []
        for entry in cleaned:
            duplicate = False
            for existing in kept:
                if existing["cls"] == entry["cls"] and self._dqa0835_xywh_iou(existing["box"], entry["box"]) >= self.dqa0835_memory_merge_iou:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(entry)
            if len(kept) >= self.dqa0835_max_entries_per_image:
                break
        return kept

    def _load_dqa0835_pseudo_memory(self) -> dict:
        if not self.dqa0835_memory_enabled or not self.dqa0835_memory_path:
            return {}
        base = Path(self.dqa0835_memory_path)
        candidates = [base]
        candidates.extend(sorted(base.parent.glob(f"{base.stem}.rank*{base.suffix or '.json'}")))
        memory = {}
        for path in candidates:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            for key, entries in payload.items():
                if not isinstance(entries, list):
                    continue
                memory.setdefault(str(key), []).extend(entries)
        return {key: self._dqa0835_merge_memory_entries(entries) for key, entries in memory.items()}

    def _dqa0835_target_key(self, target_paths, image_idx: int) -> str:
        try:
            return str(target_paths[image_idx])
        except Exception:
            return f"rank{self.RANK}_image{image_idx}"

    def _refine_dqa0835_targets_with_memory(self, unlabeled_targets, target_paths, invalid_target_shape):
        if not self.dqa0835_memory_enabled or invalid_target_shape:
            return unlabeled_targets
        if not isinstance(unlabeled_targets, torch.Tensor) or unlabeled_targets.ndim != 2 or unlabeled_targets.shape[0] == 0:
            return unlabeled_targets
        if unlabeled_targets.shape[1] <= 6:
            return unlabeled_targets

        refined = unlabeled_targets.clone()
        raw = unlabeled_targets.detach().cpu()
        updates: dict[str, list[dict]] = {}
        for row_idx, row in enumerate(raw):
            try:
                image_idx = int(row[0].item())
                cls_id = int(row[1].item())
                box = [float(x) for x in row[2:6].tolist()]
                score = min(max(float(row[6].item()), 0.0), 1.0)
                obj = min(max(float(row[7].item()) if raw.shape[1] > 7 else score, 0.0), 1.0)
                cls_conf = min(max(float(row[8].item()) if raw.shape[1] > 8 else score, 0.0), 1.0)
            except (TypeError, ValueError):
                continue

            key = self._dqa0835_target_key(target_paths, image_idx)
            best_match = None
            best_iou = 0.0
            for entry in self.dqa0835_base_memory.get(key, []):
                if int(entry.get("cls", -1)) != cls_id:
                    continue
                iou = self._dqa0835_xywh_iou(box, entry.get("box", []))
                if iou > best_iou:
                    best_iou = iou
                    best_match = entry

            previous_stability = int(best_match.get("stability", 0)) if best_match and best_iou >= self.dqa0835_memory_iou else 0
            new_stability = previous_stability + 1 if previous_stability > 0 else 1
            self.dqa0835_memory_total += 1
            if previous_stability > 0:
                self.dqa0835_memory_matches += 1

            if new_stability >= self.dqa0835_stable_rounds:
                refined[row_idx, 6] = max(float(refined[row_idx, 6].item()), self.dqa0835_stable_score_floor)
                if refined.shape[1] > 7:
                    refined[row_idx, 7] = max(float(refined[row_idx, 7].item()), self.dqa0835_stable_obj_floor)
                if refined.shape[1] > 8:
                    refined[row_idx, 8] = max(float(refined[row_idx, 8].item()), self.dqa0835_stable_cls_floor)
                self.dqa0835_memory_stable += 1
            elif previous_stability > 0:
                refined[row_idx, 6] = min(max(float(refined[row_idx, 6].item()), score), self.dqa0835_matched_score_cap)
            else:
                refined[row_idx, 6] = min(float(refined[row_idx, 6].item()), self.dqa0835_new_score_cap)

            updates.setdefault(key, []).append(
                {
                    "cls": cls_id,
                    "box": [min(max(float(x), 0.0), 1.0) for x in box],
                    "score": score,
                    "obj": obj,
                    "cls_conf": cls_conf,
                    "stability": new_stability,
                }
            )
            self.dqa0835_seen_memory_keys.add(key)

        for key, entries in updates.items():
            self.dqa0835_memory[key] = self._dqa0835_merge_memory_entries(
                list(self.dqa0835_memory.get(key, [])) + entries
            )
        return refined

    def _write_dqa0835_pseudo_memory(self):
        if not self.dqa0835_memory_enabled or not self.dqa0835_memory_path:
            return
        out_path = self._dqa0835_rank_memory_path()
        if out_path is None:
            return
        payload = {
            key: self.dqa0835_memory.get(key, [])
            for key in sorted(self.dqa0835_seen_memory_keys)
            if self.dqa0835_memory.get(key)
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        LOGGER.info(
            "Wrote DQA08_3_5 pseudo-label memory to %s "
            "(images=%d, total=%d, matched=%d, stable=%d)",
            out_path,
            len(payload),
            self.dqa0835_memory_total,
            self.dqa0835_memory_matches,
            self.dqa0835_memory_stable,
        )

    def _dqa_feature_quality(self, unlabeled_targets, feature_maps):
        targets = unlabeled_targets.detach().to(self.device)
        n = targets.shape[0]
        default = torch.ones(n, dtype=torch.float64, device=self.device) * 0.5
        if not isinstance(feature_maps, (list, tuple)) or len(feature_maps) == 0:
            return default.cpu(), default.cpu()

        maps = [feature.detach() for feature in feature_maps if isinstance(feature, torch.Tensor) and feature.ndim == 4]
        if not maps:
            return default.cpu(), default.cpu()

        boxes = targets[:, 2:6].to(torch.float32).clamp(0.0, 1.0)
        x_center, y_center, width, height = boxes.unbind(dim=1)
        size = torch.maximum(width, height)
        level_ids = torch.zeros(n, dtype=torch.long, device=self.device)
        if len(maps) > 1:
            level_ids = torch.where(size > 0.14, torch.ones_like(level_ids), level_ids)
        if len(maps) > 2:
            level_ids = torch.where(size > 0.32, torch.ones_like(level_ids) * 2, level_ids)
        level_ids = level_ids.clamp(0, len(maps) - 1)

        energy = torch.zeros(n, dtype=torch.float32, device=self.device)
        contrast = torch.zeros(n, dtype=torch.float32, device=self.device)
        image_ids = targets[:, 0].to(torch.long)

        for level_idx, fmap in enumerate(maps):
            mask = level_ids == level_idx
            if not mask.any():
                continue
            fmap = fmap.float()
            b, _, h, w = fmap.shape
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            img_idx = image_ids[idx].clamp(0, b - 1)
            xs = (x_center[idx] * (w - 1)).round().to(torch.long).clamp(0, w - 1)
            ys = (y_center[idx] * (h - 1)).round().to(torch.long).clamp(0, h - 1)

            vectors = fmap[img_idx, :, ys, xs]
            energy[idx] = vectors.pow(2).mean(dim=1).sqrt()

            activation = fmap.abs().mean(dim=1, keepdim=True)
            local = torch.nn.functional.avg_pool2d(activation, kernel_size=3, stride=1, padding=1)
            center = activation[img_idx, 0, ys, xs]
            neighborhood = local[img_idx, 0, ys, xs].clamp_min(1e-6)
            contrast[idx] = center / neighborhood

        def robust_normalize(values):
            if values.numel() <= 1:
                return torch.ones_like(values) * 0.5
            low = torch.quantile(values, 0.10)
            high = torch.quantile(values, 0.90)
            denom = (high - low).clamp_min(1e-6)
            return ((values - low) / denom).clamp(0.0, 1.0)

        feature_quality = robust_normalize(energy)
        feature_contrast = robust_normalize(contrast)
        return feature_quality.to(torch.float64).cpu(), feature_contrast.to(torch.float64).cpu()

    @staticmethod
    def _dqa0836_label_path_from_image(image_path: Path) -> Path:
        parts = list(image_path.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            return Path(*parts).with_suffix(".txt")
        return image_path.with_suffix(".txt")

    @staticmethod
    def _dqa0836_xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return np.stack([x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0], axis=1)

    @staticmethod
    def _dqa0836_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if len(a) == 0 or len(b) == 0:
            return np.zeros((len(a), len(b)), dtype=np.float64)
        lt = np.maximum(a[:, None, :2], b[None, :, :2])
        rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
        wh = np.clip(rb - lt, 0.0, None)
        inter = wh[:, :, 0] * wh[:, :, 1]
        area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
        area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
        return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-12, None)

    def _dqa0836_load_source_prior(self, bundle: dict) -> np.ndarray:
        if self.dqa0836_source_gt_prior is not None:
            return self.dqa0836_source_gt_prior

        counts = np.zeros(self.nc, dtype=np.float64)
        list_path = Path(str(bundle.get("source_train_list", "")))
        if list_path.exists():
            for raw in list_path.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                label_path = self._dqa0836_label_path_from_image(Path(raw))
                if not label_path.exists():
                    continue
                for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(float(parts[0]))
                    except ValueError:
                        continue
                    if 0 <= cls_id < self.nc:
                        counts[cls_id] += 1.0

        self.dqa0836_source_gt_prior = (counts + 1.0) / (counts.sum() + self.nc)
        return self.dqa0836_source_gt_prior

    def _dqa0836_load_scolq_bundle(self):
        if not self.dqa0836_scolq_enabled:
            return None
        if self.dqa0836_scolq_bundle is not None:
            return self.dqa0836_scolq_bundle
        if not self.dqa0836_scolq_model_path:
            LOGGER.warning("DQA08_3_6 SCoLQ is enabled but DQA0836_SCOLQ_MODEL is empty; using teacher confidence.")
            self.dqa0836_scolq_enabled = False
            return None

        model_path = Path(self.dqa0836_scolq_model_path)
        if not model_path.exists():
            LOGGER.warning("DQA08_3_6 SCoLQ model is missing at %s; using teacher confidence.", model_path)
            self.dqa0836_scolq_enabled = False
            return None

        try:
            import joblib
        except Exception as exc:
            LOGGER.warning("Could not import joblib for SCoLQ (%s); using teacher confidence.", exc)
            self.dqa0836_scolq_enabled = False
            return None

        self.dqa0836_scolq_bundle = joblib.load(model_path)
        self.dqa0836_scolq_features = list(self.dqa0836_scolq_bundle.get("features", []))
        if not self.dqa0836_scolq_features:
            LOGGER.warning("SCoLQ bundle at %s has no feature list; using teacher confidence.", model_path)
            self.dqa0836_scolq_enabled = False
            return None
        self._dqa0836_load_source_prior(self.dqa0836_scolq_bundle)
        LOGGER.info(
            "Loaded DQA08_3_6 SCoLQ bundle from %s with %d features.",
            model_path,
            len(self.dqa0836_scolq_features),
        )
        return self.dqa0836_scolq_bundle

    def _dqa0836_targets_to_feature_frame(self, unlabeled_targets):
        import pandas as pd

        raw = unlabeled_targets.detach().cpu().numpy()
        n = raw.shape[0]
        image_ids = raw[:, 0].astype(np.int64)
        cls = raw[:, 1].astype(np.int64).clip(0, self.nc - 1)
        x = raw[:, 2].astype(np.float64)
        y = raw[:, 3].astype(np.float64)
        w = raw[:, 4].astype(np.float64)
        h = raw[:, 5].astype(np.float64)
        conf = np.clip(raw[:, 6].astype(np.float64), 1e-6, 1.0 - 1e-6)

        image_pred_count = np.ones(n, dtype=np.float64)
        class_pred_count = np.ones(n, dtype=np.float64)
        rank_conf = np.ones(n, dtype=np.float64)
        max_iou_same = np.zeros(n, dtype=np.float64)
        max_iou_any = np.zeros(n, dtype=np.float64)
        near_same_50 = np.zeros(n, dtype=np.float64)
        near_any_50 = np.zeros(n, dtype=np.float64)

        boxes = self._dqa0836_xywh_to_xyxy(np.stack([x, y, w, h], axis=1))
        for image_id in np.unique(image_ids):
            idx = np.where(image_ids == image_id)[0]
            image_pred_count[idx] = float(len(idx))
            order = idx[np.argsort(-conf[idx], kind="mergesort")]
            rank_conf[order] = np.arange(1, len(order) + 1, dtype=np.float64)
            for cls_id in np.unique(cls[idx]):
                class_pred_count[idx[cls[idx] == cls_id]] = float(np.sum(cls[idx] == cls_id))
            ious = self._dqa0836_iou_matrix(boxes[idx], boxes[idx])
            if len(idx):
                np.fill_diagonal(ious, 0.0)
                same = cls[idx][:, None] == cls[idx][None, :]
                max_iou_any[idx] = ious.max(axis=1)
                max_iou_same[idx] = (ious * same).max(axis=1)
                near_any_50[idx] = (ious >= 0.5).sum(axis=1)
                near_same_50[idx] = ((ious >= 0.5) & same).sum(axis=1)

        source_prior = self.dqa0836_source_gt_prior
        if source_prior is None:
            source_prior = np.ones(self.nc, dtype=np.float64) / max(self.nc, 1)
        batch_counts = np.bincount(cls, minlength=self.nc).astype(np.float64)
        batch_prior = batch_counts / max(float(n), 1.0)
        area = w * h
        aspect = w / np.clip(h, 1e-6, None)

        data = {
            "cls": cls,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "conf": conf,
            "image_pred_count": image_pred_count,
            "class_pred_count": class_pred_count,
            "rank_conf": rank_conf,
            "rank_conf_norm": rank_conf / np.maximum(image_pred_count, 1.0),
            "max_iou_same_pred": max_iou_same,
            "max_iou_any_pred": max_iou_any,
            "near_same_count_50": near_same_50,
            "near_any_count_50": near_any_50,
            "area": area,
            "log_area": np.log1p(area),
            "aspect": aspect,
            "log_aspect": np.log(np.clip(aspect, 1e-6, None)),
            "edge_dist": np.minimum.reduce([x, y, 1.0 - x, 1.0 - y]),
            "conf_logit": np.log(conf / np.clip(1.0 - conf, 1e-6, None)),
            "source_gt_class_prior": source_prior[cls],
            "source_gt_class_log_prior": np.log(source_prior[cls]),
            "split_pred_class_prior": batch_prior[cls],
            "pred_gt_prior_ratio": batch_prior[cls] / np.clip(source_prior[cls], 1e-9, None),
            "aug640_iou": np.nan,
            "aug640_conf": np.nan,
            "aug640_matched": 0.0,
            "plain512_iou": np.nan,
            "plain512_conf": np.nan,
            "plain512_matched": 0.0,
            "plain768_iou": np.nan,
            "plain768_conf": np.nan,
            "plain768_matched": 0.0,
            "agreement_iou_mean": np.nan,
            "agreement_match_count": 0.0,
        }
        for feature in self.dqa0836_scolq_features:
            data.setdefault(feature, 0.0)
        return pd.DataFrame(data)[self.dqa0836_scolq_features]

    def _apply_dqa0836_scolq(self, unlabeled_targets, invalid_target_shape):
        self.dqa0838_last_raw_rscolq_scores = None
        if not self.dqa0836_scolq_enabled or invalid_target_shape:
            return unlabeled_targets
        if not isinstance(unlabeled_targets, torch.Tensor):
            unlabeled_targets = torch.as_tensor(unlabeled_targets)
        if unlabeled_targets.ndim != 2 or unlabeled_targets.shape[0] == 0 or unlabeled_targets.shape[1] <= 6:
            return unlabeled_targets

        bundle = self._dqa0836_load_scolq_bundle()
        if bundle is None:
            return unlabeled_targets

        try:
            frame = self._dqa0836_targets_to_feature_frame(unlabeled_targets)
            scores = bundle["pipeline"].predict_proba(frame)[:, 1]
        except Exception as exc:
            LOGGER.warning("DQA08_3_6 SCoLQ scoring failed (%s); using teacher confidence for this batch.", exc)
            return unlabeled_targets

        scores = np.clip(scores.astype(np.float64), 0.0, 1.0)
        if self.dqa0836_scolq_score_power != 1.0:
            scores = np.power(scores, self.dqa0836_scolq_score_power)
        raw_scores = scores.copy()
        if self.dqa0837_rscolq_enabled:
            scores = scores * max(self.dqa0837_rscolq_global_multiplier, 0.0)
            class_multipliers = self._dqa0837_rscolq_class_multiplier_array()
            if class_multipliers is not None:
                cls_ids = unlabeled_targets[:, 1].detach().cpu().numpy().astype(np.int64).clip(0, self.nc - 1)
                scores = scores * class_multipliers[cls_ids]
            scores = np.clip(scores, 0.0, 1.0)
        self.dqa0838_last_raw_rscolq_scores = torch.as_tensor(raw_scores, dtype=torch.float64)
        refined = unlabeled_targets.clone()
        refined[:, 6] = torch.as_tensor(scores, dtype=refined.dtype, device=refined.device)
        return refined

    def _dqa0837_rscolq_class_multiplier_array(self):
        if not self.dqa0837_rscolq_class_multipliers:
            return None
        try:
            values = json.loads(self.dqa0837_rscolq_class_multipliers)
        except Exception:
            values = [part.strip() for part in self.dqa0837_rscolq_class_multipliers.split(",") if part.strip()]
        if not isinstance(values, list) or len(values) != self.nc:
            LOGGER.warning(
                "Ignoring DQA08_3_7 R-SCoLQ class multipliers with invalid length: %s",
                self.dqa0837_rscolq_class_multipliers,
            )
            return None
        return np.asarray([max(float(value), 0.0) for value in values], dtype=np.float64)

    def _update_dqa_pseudo_stats(self, unlabeled_targets, invalid_target_shape, feature_maps=None):
        if not self.dqa_track_pseudo_stats or invalid_target_shape:
            return
        if not isinstance(unlabeled_targets, torch.Tensor):
            unlabeled_targets = torch.as_tensor(unlabeled_targets)
        if unlabeled_targets.ndim != 2 or unlabeled_targets.shape[0] == 0:
            return

        targets = unlabeled_targets.detach().cpu()
        class_ids = targets[:, 1].to(torch.int64)
        valid = (class_ids >= 0) & (class_ids < self.nc)
        if not valid.any():
            return

        targets = targets[valid]
        class_ids = targets[:, 1].to(torch.int64)
        if targets.shape[1] > 6:
            confidences = targets[:, 6].to(torch.float64).clamp(0.0, 1.0)
        else:
            confidences = torch.ones_like(class_ids, dtype=torch.float64)
        raw_rscolq_confidences = None
        if self.dqa0838_rscolq_raw_stats and self.dqa0838_last_raw_rscolq_scores is not None:
            raw_scores = self.dqa0838_last_raw_rscolq_scores
            if raw_scores.numel() == valid.numel():
                raw_rscolq_confidences = raw_scores[valid].to(torch.float64).clamp(0.0, 1.0)
        if targets.shape[1] > 7:
            objectness = targets[:, 7].to(torch.float64).clamp(0.0, 1.0)
        else:
            objectness = confidences.clone()
        if targets.shape[1] > 8:
            class_confidences = targets[:, 8].to(torch.float64).clamp(0.0, 1.0)
        else:
            class_confidences = confidences.clone()

        boxes = targets[:, 2:6].to(torch.float64)
        finite_boxes = torch.isfinite(boxes).all(dim=1)
        x_center, y_center, width, height = boxes.unbind(dim=1)
        x1 = x_center - width / 2.0
        y1 = y_center - height / 2.0
        x2 = x_center + width / 2.0
        y2 = y_center + height / 2.0
        overflow = (
            torch.relu(-x1)
            + torch.relu(-y1)
            + torch.relu(x2 - 1.0)
            + torch.relu(y2 - 1.0)
        )
        positive_area = (width > 0.0) & (height > 0.0)
        localization = (1.0 - overflow.clamp(0.0, 1.0)).clamp(0.0, 1.0)
        localization = torch.where(finite_boxes & positive_area, localization, torch.zeros_like(localization))
        quality = (
            0.50 * confidences
            + 0.20 * objectness
            + 0.20 * class_confidences
            + 0.10 * localization
        ).clamp(0.0, 1.0)
        feature_quality = torch.ones_like(confidences, dtype=torch.float64) * 0.5
        feature_contrast = torch.ones_like(confidences, dtype=torch.float64) * 0.5
        if self.dqa_quality_mode in {
            "feature_saliency",
            "feature_contrast",
            "feature_balanced",
            "feature_conservative",
            "feature_no_conf",
        }:
            feature_quality, feature_contrast = self._dqa_feature_quality(targets, feature_maps)
            feature_quality = feature_quality.to(torch.float64).clamp(0.0, 1.0)
            feature_contrast = feature_contrast.to(torch.float64).clamp(0.0, 1.0)
        feature_balanced = (
            0.35 * feature_quality
            + 0.25 * feature_contrast
            + 0.20 * objectness
            + 0.10 * class_confidences
            + 0.10 * localization
        ).clamp(0.0, 1.0)

        if self.dqa_quality_mode == "feature_saliency":
            quality = (
                0.65 * feature_quality
                + 0.15 * objectness
                + 0.10 * class_confidences
                + 0.10 * localization
            ).clamp(0.0, 1.0)
        elif self.dqa_quality_mode == "feature_contrast":
            quality = (
                0.65 * feature_contrast
                + 0.15 * feature_quality
                + 0.10 * objectness
                + 0.10 * localization
            ).clamp(0.0, 1.0)
        elif self.dqa_quality_mode == "feature_balanced":
            quality = feature_balanced
        elif self.dqa_quality_mode == "feature_conservative":
            quality = torch.minimum(quality, feature_balanced)
        elif self.dqa_quality_mode == "feature_no_conf":
            quality = (
                0.50 * feature_quality
                + 0.30 * feature_contrast
                + 0.20 * localization
            ).clamp(0.0, 1.0)
        elif self.dqa_quality_mode in {"scolq", "rscolq", "source_calibrated_localization"}:
            quality = confidences.clamp(0.0, 1.0)
        elif self.dqa_quality_mode in {"rscolq_raw", "round_stable_raw"}:
            quality = (raw_rscolq_confidences if raw_rscolq_confidences is not None else confidences).clamp(0.0, 1.0)

        counts = torch.bincount(class_ids, minlength=self.nc).to(torch.float64)
        confidence_sums = torch.zeros(self.nc, dtype=torch.float64)
        objectness_sums = torch.zeros(self.nc, dtype=torch.float64)
        class_confidence_sums = torch.zeros(self.nc, dtype=torch.float64)
        localization_sums = torch.zeros(self.nc, dtype=torch.float64)
        feature_quality_sums = torch.zeros(self.nc, dtype=torch.float64)
        feature_contrast_sums = torch.zeros(self.nc, dtype=torch.float64)
        quality_sums = torch.zeros(self.nc, dtype=torch.float64)
        confidence_sums.index_add_(0, class_ids, confidences)
        objectness_sums.index_add_(0, class_ids, objectness)
        class_confidence_sums.index_add_(0, class_ids, class_confidences)
        localization_sums.index_add_(0, class_ids, localization)
        feature_quality_sums.index_add_(0, class_ids, feature_quality)
        feature_contrast_sums.index_add_(0, class_ids, feature_contrast)
        quality_sums.index_add_(0, class_ids, quality)
        self.dqa_pseudo_counts += counts
        self.dqa_pseudo_confidence_sums += confidence_sums
        self.dqa_pseudo_objectness_sums += objectness_sums
        self.dqa_pseudo_class_confidence_sums += class_confidence_sums
        self.dqa_pseudo_localization_sums += localization_sums
        self.dqa_pseudo_feature_quality_sums += feature_quality_sums
        self.dqa_pseudo_feature_contrast_sums += feature_contrast_sums
        self.dqa_pseudo_quality_sums += quality_sums

    def _write_dqa_pseudo_stats(self):
        if not self.dqa_track_pseudo_stats:
            return

        counts = self.dqa_pseudo_counts.clone()
        confidence_sums = self.dqa_pseudo_confidence_sums.clone()
        objectness_sums = self.dqa_pseudo_objectness_sums.clone()
        class_confidence_sums = self.dqa_pseudo_class_confidence_sums.clone()
        localization_sums = self.dqa_pseudo_localization_sums.clone()
        feature_quality_sums = self.dqa_pseudo_feature_quality_sums.clone()
        feature_contrast_sums = self.dqa_pseudo_feature_contrast_sums.clone()
        quality_sums = self.dqa_pseudo_quality_sums.clone()
        if self.RANK != -1 and dist.is_available() and dist.is_initialized():
            counts_device = counts.to(self.device)
            confidence_sums_device = confidence_sums.to(self.device)
            objectness_sums_device = objectness_sums.to(self.device)
            class_confidence_sums_device = class_confidence_sums.to(self.device)
            localization_sums_device = localization_sums.to(self.device)
            feature_quality_sums_device = feature_quality_sums.to(self.device)
            feature_contrast_sums_device = feature_contrast_sums.to(self.device)
            quality_sums_device = quality_sums.to(self.device)
            dist.all_reduce(counts_device, op=dist.ReduceOp.SUM)
            dist.all_reduce(confidence_sums_device, op=dist.ReduceOp.SUM)
            dist.all_reduce(objectness_sums_device, op=dist.ReduceOp.SUM)
            dist.all_reduce(class_confidence_sums_device, op=dist.ReduceOp.SUM)
            dist.all_reduce(localization_sums_device, op=dist.ReduceOp.SUM)
            dist.all_reduce(feature_quality_sums_device, op=dist.ReduceOp.SUM)
            dist.all_reduce(feature_contrast_sums_device, op=dist.ReduceOp.SUM)
            dist.all_reduce(quality_sums_device, op=dist.ReduceOp.SUM)
            counts = counts_device.cpu()
            confidence_sums = confidence_sums_device.cpu()
            objectness_sums = objectness_sums_device.cpu()
            class_confidence_sums = class_confidence_sums_device.cpu()
            localization_sums = localization_sums_device.cpu()
            feature_quality_sums = feature_quality_sums_device.cpu()
            feature_contrast_sums = feature_contrast_sums_device.cpu()
            quality_sums = quality_sums_device.cpu()

        if self.RANK not in [-1, 0]:
            return

        mean_confidences = [
            (confidence_sums[idx] / counts[idx]).item() if counts[idx] > 0 else 0.0
            for idx in range(self.nc)
        ]
        mean_objectness = [
            (objectness_sums[idx] / counts[idx]).item() if counts[idx] > 0 else 0.0
            for idx in range(self.nc)
        ]
        mean_class_confidences = [
            (class_confidence_sums[idx] / counts[idx]).item() if counts[idx] > 0 else 0.0
            for idx in range(self.nc)
        ]
        mean_localization_qualities = [
            (localization_sums[idx] / counts[idx]).item() if counts[idx] > 0 else 0.0
            for idx in range(self.nc)
        ]
        mean_feature_qualities = [
            (feature_quality_sums[idx] / counts[idx]).item() if counts[idx] > 0 else 0.0
            for idx in range(self.nc)
        ]
        mean_feature_contrasts = [
            (feature_contrast_sums[idx] / counts[idx]).item() if counts[idx] > 0 else 0.0
            for idx in range(self.nc)
        ]
        mean_quality_scores = [
            (quality_sums[idx] / counts[idx]).item() if counts[idx] > 0 else 0.0
            for idx in range(self.nc)
        ]
        payload = {
            "id": self.dqa_client_id or self.save_dir.name,
            "phase": int(self.dqa_phase) if self.dqa_phase else None,
            "round": int(self.dqa_round) if self.dqa_round else None,
            "source_run": str(self.save_dir),
            "counts": [float(value) for value in counts.tolist()],
            "confidence_sums": [float(value) for value in confidence_sums.tolist()],
            "objectness_sums": [float(value) for value in objectness_sums.tolist()],
            "class_confidence_sums": [float(value) for value in class_confidence_sums.tolist()],
            "localization_sums": [float(value) for value in localization_sums.tolist()],
            "feature_quality_sums": [float(value) for value in feature_quality_sums.tolist()],
            "feature_contrast_sums": [float(value) for value in feature_contrast_sums.tolist()],
            "quality_sums": [float(value) for value in quality_sums.tolist()],
            "mean_confidences": mean_confidences,
            "mean_objectness": mean_objectness,
            "mean_class_confidences": mean_class_confidences,
            "mean_localization_qualities": mean_localization_qualities,
            "mean_feature_qualities": mean_feature_qualities,
            "mean_feature_contrasts": mean_feature_contrasts,
            "mean_quality_scores": mean_quality_scores,
            "quality_mode": self.dqa_quality_mode,
        }
        out_path = Path(self.dqa_pseudo_stats_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info(f"Wrote DQA pseudo-label stats to {out_path}")
    
    def build_optimizer(self, cfg, optinit=True, weight_masks=None, ckpt=None):
        super().build_optimizer(cfg, optinit, weight_masks, ckpt)
        # Scheduler
        if cfg.SSOD.multi_step_lr:
            milestones = cfg.SSOD.milestones
            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
            self.scheduler.last_epoch = self.epoch - 1  # do not move
            print('self scheduler:', milestones)
            self.scaler = amp.GradScaler(enabled=self.cuda)

    def build_model(self, cfg, device):
        # use DomainAdpatationModel
        check_suffix(cfg.weights, '.pt')  # check weights
        pretrained = cfg.weights.endswith('.pt')
        if pretrained:
            with torch_distributed_zero_first(self.LOCAL_RANK):
                weights = attempt_download(cfg.weights)  # download if not found locally
            ckpt = torch.load(weights, map_location=device, weights_only=False)  # load checkpoint
            self.model = Model(cfg or ckpt['model'].yaml).to(device)  # create
            exclude = ['anchor'] if (cfg or cfg.Model.anchors) and not cfg.resume else []  # exclude keys
            csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            if cfg.prune_finetune:
                dynamic_load(self.model, csd)
                self.model.info()
            csd = intersect_dicts(csd, self.model.state_dict(), exclude=exclude)  # intersect
            self.model.load_state_dict(csd, strict=False)  # load
            LOGGER.info(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from {weights}')  # report
        else:
            self.model = Model(cfg).to(device)  # create
        # Freeze
        freeze = [f'model.{x}.' for x in range(cfg.freeze_layer_num)]  # layers to freeze
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print(f'freezing {k}')
                v.requires_grad = False
        apply_fedsto_train_scope(self.model, cfg.FedSTO.train_scope)
        # EMA
        self.ema = ModelEMA(self.model)        
        if self.cfg.hyp.burn_epochs > 0:
            self.semi_ema = None
            # self.ema = ModelEMA(self.model, decay=self.cfg.SSOD.ema_rate)
        else:
            if self.cosine_ema:
                self.semi_ema = CosineEMA(self.ema.ema, decay_start=self.cfg.SSOD.ema_rate, total_epoch=self.epochs)
            else:
                self.semi_ema = SemiSupModelEMA(self.ema.ema, self.cfg.SSOD.ema_rate)

        # Resume
        self.start_epoch = 0
        pretrained = cfg.weights.endswith('.pt')
        if pretrained:
            if ckpt['optimizer'] is not None:
                try:
                    self.optimizer.load_state_dict(ckpt['optimizer'])
                except:
                    LOGGER.info('pretrain model with different type of optimizer')
                # best_fitness = ckpt['best_fitness']

            # EMA
            if self.ema and ckpt.get('ema'):
                self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict(), strict=False)
                self.ema.updates = ckpt['updates']
            
            if self.semi_ema and ckpt.get('ema'):
                self.semi_ema.ema.load_state_dict(ckpt['ema'].float().state_dict(), strict=False)
            # EMA
            # if self.ema and ckpt.get('ema'):
            #     self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict(), strict=False)
            #     self.ema.updates = ckpt['updates']

            # Epochs
            self.start_epoch = ckpt['epoch'] + 1
            if cfg.resume:
                assert self.start_epoch > 0, f'{weights} training to {self.epochs} epochs is finished, nothing to resume.'
            if self.epochs < self.start_epoch:
                LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs.")
                self.epochs += ckpt['epoch']  # finetune additional epochs

            del ckpt, csd
        self.epoch = self.start_epoch
        # self.ema.update_decay(self.epoch, self.cfg.hyp.burn_epochs)
        self.model_type = self.model.model_type

        # load teacher model and extract class idx
        self.extra_teacher_models = []
        self.extra_teacher_class_idxs = []  
        if len(self.cfg.SSOD.extra_teachers) > 0 and len(self.cfg.SSOD.extra_teachers_class_names) > 0:
            assert(len(self.cfg.SSOD.extra_teachers) == len(self.cfg.SSOD.extra_teachers_class_names)  )
            for i, extra_teacher_path in enumerate(self.cfg.SSOD.extra_teachers):
                teacher_model = attempt_load(extra_teacher_path, map_location=device) 
                self.extra_teacher_models.append(teacher_model)
                if self.RANK in [-1 , 0]: 
                    print('load  {} teacher model and class...'.format(i))

                teacher_class_idx = {}  #{origin_class_idx : new_class_idx}
                assert len(self.cfg.SSOD.extra_teachers_class_names[i]) > 0
                if self.RANK in [-1 , 0]: 
                    print("origin name: {} current name: {}".format(teacher_model.names, self.cfg.Dataset.names) )
                for na in self.cfg.SSOD.extra_teachers_class_names[i]:
                    origin_idx = -1; curr_idx = -1
                    for idx, origin_name in enumerate(teacher_model.names):
                        if na == origin_name:
                            origin_idx = idx
                            break 
                    for idx, name  in enumerate(self.cfg.Dataset.names):
                        if na == name:
                            curr_idx = idx
                    if len(self.cfg.SSOD.extra_teachers_class_names[i]) == 1: #teacher model是通过single-cls 训练的,这里 将原来的class-idx改为0
                        if self.RANK in [-1 , 0]: 
                            print('single cls change ')
                        origin_idx = 0
                    teacher_class_idx[origin_idx] = curr_idx

                if self.RANK in [-1 , 0]: 
                    print('class_idx dic: ', teacher_class_idx)
                self.extra_teacher_class_idxs.append(teacher_class_idx)
                assert len(self.extra_teacher_class_idxs) == len(self.extra_teacher_models)
                assert len(self.extra_teacher_models) > 0

    def build_dataloader(self, cfg, callbacks):
        # Image sizes
        gs = max(int(self.model.stride.max()), 32)  # grid size (max stride)
        # Model parameters
        self.imgsz = check_img_size(cfg.Dataset.img_size, gs, floor=gs * 2)  # verify imgsz is gs-multiple

        # DP mode
        if self.cuda and self.RANK == -1 and torch.cuda.device_count() > 1:
            logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
            self.model = torch.nn.DataParallel(self.model)

        # SyncBatchNorm
        if self.sync_bn and self.cuda and self.RANK != -1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(device)
            LOGGER.info('Using SyncBatchNorm()')

        # Trainloader
        self.train_loader, self.dataset = create_dataloader(self.data_dict['train'], self.imgsz, self.batch_size // self.WORLD_SIZE, gs, self.single_cls,
                                              hyp=cfg.hyp, augment=True, cache=cfg.cache, rect=cfg.rect, rank=self.LOCAL_RANK,
                                              workers=cfg.Dataset.workers, prefix=colorstr('train: '), cfg=cfg)
        self.cls_ratio_gt = self.dataset.cls_ratio_gt 
        self.label_num_per_image = self.dataset.label_num_per_image
        # Trainloader for semi supervised training 
        self.unlabeled_dataloader, self.unlabeled_dataset = create_target_dataloader(self.data_dict['target'], self.imgsz, self.batch_size // self.WORLD_SIZE, gs, self.single_cls,
                                              hyp=cfg.hyp, augment=True, cache=cfg.cache, rect=cfg.rect, rank=self.LOCAL_RANK,
                                              workers=cfg.Dataset.workers, cfg=cfg, prefix=colorstr('target: '))

        mlc = int(np.concatenate(self.dataset.labels, 0)[:, 0].max())  # max label class
        self.nb = len(self.train_loader)  # number of batches
        assert mlc < self.nc, f'Label class {mlc} exceeds nc={self.nc} in {cfg.Dataset.data_name}. Possible class labels are 0-{self.nc - 1}'

        # Process 0
        if self.RANK in [-1, 0]:
            self.val_loader = create_dataloader(self.data_dict['val'] , self.imgsz, self.batch_size // self.WORLD_SIZE * 2, gs, self.single_cls,
                                       hyp=cfg.hyp, cache=None if self.noval else cfg.cache, rect=True, rank=-1,
                                       workers=cfg.Dataset.workers, pad=0.5,
                                       prefix=colorstr('val: '), cfg=cfg)[0]

            if not cfg.resume:
                labels = np.concatenate(self.dataset.labels, 0)
                if self.plots:
                    plot_labels(labels, self.names, self.save_dir)

                # Anchors
                if not cfg.noautoanchor:
                    check_anchors(self.dataset, model=self.model, thr=cfg.Loss.anchor_t, imgsz=self.imgsz)
                self.model.half().float()  # pre-reduce anchor precision

            callbacks.run('on_pretrain_routine_end')
        self.no_aug_epochs = cfg.hyp.no_aug_epochs


    def build_ddp_model(self, cfg, device):
        super().build_ddp_model(cfg, device)
        # if cfg.Loss.type == 'ComputeLoss': 
        self.compute_un_sup_loss = build_ssod_loss(self.model, cfg)
        self.domain_loss = DomainLoss()
        self.target_loss = TargetLoss()
    
    def update_train_logger(self):
        for (imgs, targets, paths, _) in self.train_loader:  # batch -------------------------------------------------------------
            imgs = imgs.to(self.device, non_blocking=True).float() / self.norm_scale  # uint8 to float32, 0-255 to 0.0-1.0
            with amp.autocast(enabled=self.cuda):
                pred, sup_feats = self.model(imgs)  # forward
                loss, loss_items = self.compute_loss(pred, targets.to(self.device))  # loss scaled by batch_size
                if self.model_type in ['yolox', 'tal']:
                    un_sup_loss, un_sup_loss_items = self.compute_un_sup_loss(pred, pred, targets.to(self.device))  
                else:
                    un_sup_loss, un_sup_loss_items = self.compute_un_sup_loss(pred, targets.to(self.device))  
            if self.RANK in [-1, 0]:
                for loss_key in loss_items.keys():
                    self.log_contents.append(loss_key)
            if (self.epoch >= self.cfg.hyp.burn_epochs):
                if self.RANK in [-1, 0]:
                    for loss_key in un_sup_loss_items.keys():
                        self.log_contents.append(loss_key)
            break
        if self.cfg.SSOD.train_domain == True and self.epoch >= self.cfg.hyp.burn_epochs:
            if self.RANK in [-1, 0]:
            # self.log_contents.append('hit_miss')
            # self.log_contents.append('hit_total')
                self.log_contents.append('tp')
                self.log_contents.append('fp_cls')
                self.log_contents.append('fp_loc')
                self.log_contents.append('pse_num')
                self.log_contents.append('gt_num')
        LOGGER.info(('\n' + '%10s' * len(self.log_contents)) % tuple(self.log_contents))
        
    
    def train_in_epoch(self, callbacks):
        if ( self.epoch < self.cfg.hyp.burn_epochs):
            if self.cfg.SSOD.with_da_loss:
                self.train_without_unlabeled_da(callbacks)
            else:
                self.train_without_unlabeled(callbacks)
            if self.RANK in [-1, 0]:
                print('burn_in_epoch: {}, cur_epoch: {}'.format(self.cfg.hyp.burn_epochs, self.epoch) )
        else:
            # if self.epoch == self.cfg.hyp.burn_epochs and self.cfg.hyp.burn_epochs > 0:
            if self.epoch == self.cfg.hyp.burn_epochs:
                msd = self.model.module.state_dict() if is_parallel(self.model) else self.model.state_dict()  # model state_dict
                for k, v in self.ema.ema.state_dict().items():
                    if v.dtype.is_floating_point:
                        msd[k] = v
                    # if self.RANK in [-1, 0]:
                    #     print('ema:', v)
                    #     print('msd:', msd[k])
                if self.cosine_ema:
                    self.semi_ema = CosineEMA(self.ema.ema, decay_start=self.cfg.SSOD.ema_rate, total_epoch=self.epochs - self.cfg.hyp.burn_epochs)
                else:
                    self.semi_ema = SemiSupModelEMA(self.ema.ema, self.cfg.SSOD.ema_rate)
            self.train_with_unlabeled(callbacks)
    
    def after_epoch(self, callbacks, val):
        if self.cfg.SSOD.pseudo_label_type == 'LabelMatch' and self.epoch >= self.cfg.SSOD.dynamic_thres_epoch:
            self.pseudo_label_creator.update_epoch_cls_thr(self.epoch - self.start_epoch)
            self.compute_un_sup_loss.ignore_thres_high = self.pseudo_label_creator.cls_thr_high
            self.compute_un_sup_loss.ignore_thres_low = self.pseudo_label_creator.cls_thr_low
            # print(self.RANK,  self.pseudo_label_creator.cls_thr_high, self.pseudo_label_creator.cls_thr_low)
        if self.epoch >= self.cfg.hyp.burn_epochs:
            if self.model_type == 'tal':
                self.compute_un_sup_loss.cur_epoch = self.epoch - self.cfg.hyp.burn_epochs
            if self.cosine_ema:
                self.semi_ema.update_decay(self.epoch - self.cfg.hyp.burn_epochs)        
        if self.RANK in [-1, 0]:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=self.epoch)
            self.ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (self.epoch + 1 == self.epochs)
            if not self.noval or final_epoch:  # Calculate mAP
                val_ssod = self.cfg.SSOD.train_domain
                # if (self.epoch >= self.cfg.hyp.burn_epochs):
                # if (1):
                self.results, maps, _, cls_thr = val.run(self.data_dict,
                                           batch_size=self.batch_size // self.WORLD_SIZE * 2,
                                           imgsz=self.imgsz,
                                           model=deepcopy(de_parallel(self.model)),
                                           conf_thres=self.cfg.val_conf_thres, 
                                           single_cls=self.single_cls,
                                           dataloader=self.val_loader,
                                           save_dir=self.save_dir,
                                           plots=False,
                                           callbacks=callbacks,
                                           compute_loss=self.compute_loss,
                                           num_points=self.cfg.Dataset.np,
                                           val_ssod=val_ssod,
                                           val_kp=self.cfg.Dataset.val_kp)
                self.model.train()
                if (self.epoch >= self.cfg.hyp.burn_epochs):
                    self.results, maps, _, cls_thr = val.run(self.data_dict,
                                           batch_size=self.batch_size // self.WORLD_SIZE * 2,
                                           imgsz=self.imgsz,
                                           model=self.semi_ema.ema,
                                           conf_thres=self.cfg.val_conf_thres, 
                                           single_cls=self.single_cls,
                                           dataloader=self.val_loader,
                                           save_dir=self.save_dir,
                                           plots=False,
                                           callbacks=callbacks,
                                           compute_loss=self.compute_loss,
                                           num_points = self.cfg.Dataset.np,
                                           val_ssod=val_ssod,
                                           val_kp=self.cfg.Dataset.val_kp)
                else:
                    self.results, maps, _, cls_thr = val.run(self.data_dict,
                                           batch_size=self.batch_size // self.WORLD_SIZE * 2,
                                           imgsz=self.imgsz,
                                           model=self.ema.ema,
                                           conf_thres=self.cfg.val_conf_thres, 
                                           single_cls=self.single_cls,
                                           dataloader=self.val_loader,
                                           save_dir=self.save_dir,
                                           plots=False,
                                           callbacks=callbacks,
                                           compute_loss=self.compute_loss,
                                           num_points = self.cfg.Dataset.np,
                                           val_ssod=val_ssod,
                                           val_kp=self.cfg.Dataset.val_kp)

            # Update best mAP
            fi = fitness(np.array(self.results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > self.best_fitness:
                self.best_fitness = fi
            log_vals = list(self.meter.get_avg())[:3] + list(self.results) + self.lr
            callbacks.run('on_fit_epoch_end', log_vals, self.epoch, self.best_fitness, fi)

            # Save model
            if (not self.nosave) or (final_epoch):  # if save
                if self.epoch >= self.cfg.hyp.burn_epochs:
                    ckpt = {'epoch': self.epoch,
                        'best_fitness': self.best_fitness,
                        'model': deepcopy(de_parallel(self.model)).half(),
                        'ema': deepcopy(self.semi_ema.ema).half(),
                        'updates': self.ema.updates,
                        'optimizer': self.optimizer.state_dict(),
                        'wandb_id':  None}
                else:
                    ckpt = {'epoch': self.epoch,
                        'best_fitness': self.best_fitness,
                        'model': deepcopy(de_parallel(self.model)).half(),
                        'ema': deepcopy(self.ema.ema).half(),
                        'updates': self.ema.updates,
                        'optimizer': self.optimizer.state_dict(),
                        'wandb_id':  None}

                # Save last, best and delete
                torch.save(ckpt, self.last)
                if self.best_fitness == fi:
                    torch.save(ckpt, self.best)
                if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
                    w = self.save_dir / 'weights'  # weights dir
                    torch.save(ckpt, w / f'epoch{self.epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', self.last, self.epoch, final_epoch, self.best_fitness, fi)

    def train_without_unlabeled(self, callbacks):
        pbar = enumerate(self.train_loader)
        if self.RANK in [-1, 0]:
            pbar = tqdm(pbar, total=self.nb)  # progress bar

        self.optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + self.nb * self.epoch  # number integrated batches (since train start)
            imgs = imgs.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            # Forward
            #with torch.autograd.set_detect_anomaly(True):
            with amp.autocast(enabled=self.cuda):
                pred, sup_feats = self.model(imgs)  # forward
                loss, loss_items = self.compute_loss(pred, targets.to(self.device))  # loss scaled by batch_size 

                if self.RANK != -1:
                    loss *= self.WORLD_SIZE  # gradient averaged between devices in DDP mode
                    # print(self.WORLD_SIZE)
                    # print('scale loss:', loss_items)

                loss = loss + 0 * (sup_feats[0].mean() + sup_feats[1].mean() + sup_feats[2].mean())

            self.update_optimizer(loss, ni) 

            # Log
            if self.RANK in [-1, 0]:
                self.meter.update(loss_items)
                mloss_count= len(self.meter.meters.items())
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * (mloss_count+2)) % (
                    f'{self.epoch}/{self.epochs - 1}', mem,  targets.shape[0], imgs.shape[-1], *self.meter.get_avg()))
                callbacks.run('on_train_batch_end', ni, self.model, imgs, targets, paths, self.plots, self.sync_bn, self.cfg.Dataset.np)
        # end batch ------------------------------------------------------------------------------------------------
        # Scheduler
        self.lr = [x['lr'] for x in self.optimizer.param_groups]  # for loggers
        self.scheduler.step()  

    def update_optimizer(self, loss, ni):
        # Backward
        orth_loss = spectral_orthogonal_regularization(
            self.model,
            self.cfg.FedSTO.orthogonal_weight,
            self.cfg.FedSTO.orthogonal_scope,
        )
        if orth_loss is not None:
            loss = loss + orth_loss
        class_skew_loss = class_skew_head_regularization(
            self.model,
            self.cfg.ClassSkewFedSTO.orthogonal_weight,
            self.cfg.ClassSkewFedSTO.srip_weight,
            self.cfg.ClassSkewFedSTO.residual_weight,
        )
        if class_skew_loss is not None:
            loss = loss + class_skew_loss
        self.scaler.scale(loss).backward()
                
        if self.fixed_accumulate:
            self.accumulate = 1
        else:
            self.accumulate = max(round(64 / self.batch_size), 1) 

        #warmup setting
        if ni <= self.nw:
            xi = [0, self.nw]
            if self.fixed_accumulate:
                self.accumulate = max(1, np.interp(ni, xi, [1, 1]).round())
            else:
                self.accumulate = max(1, np.interp(ni, xi, [1, 64 / self.batch_size]).round())
            for j, x in enumerate(self.optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(ni, xi, [self.warmup_bias_lr if j == 2 else 0.0, x['initial_lr'] * self.lf(self.epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [self.warmup_momentum, self.momentum])
                # Optimize

        if ni - self.last_opt_step >= self.accumulate:
            self.scaler.step(self.optimizer)  # optimizer.step
            self.scaler.update()
            self.optimizer.zero_grad()
            self.ema.update(self.model)
            if self.semi_ema:
                self.semi_ema.update(self.ema.ema)
            self.last_opt_step = ni

    def train_without_unlabeled_da(self, callbacks):
        pbar = enumerate(self.train_loader)
        if self.RANK in [-1, 0]:
            pbar = tqdm(pbar, total=self.nb)  # progress bar

        self.optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + self.nb * self.epoch  # number integrated batches (since train start)
            imgs = imgs.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            target_imgs, target_targets, target_paths, _, target_imgs_ori, target_M = next(self.unlabeled_dataloader.__iter__())
            target_imgs_ori = target_imgs_ori.to(self.device, non_blocking=True).float() / 255.0 
            total_imgs = torch.cat([imgs, target_imgs_ori], 0)
            n_img, _, _, _ = imgs.shape
            # Forward
            #with torch.autograd.set_detect_anomaly(True):
            with amp.autocast(enabled=self.cuda):
                total_pred, total_feature = self.model(total_imgs)  # forward

                sup_pred, sup_feature, un_sup_pred, un_sup_feature = self.split_predict_and_feature(total_pred, total_feature, n_img)
                loss, loss_items = self.compute_loss(sup_pred, targets.to(self.device))  # loss scaled by batch_size 
                d_loss = self.domain_loss(sup_feature)
                t_loss = self.target_loss(un_sup_feature) 

                loss = loss + d_loss * self.da_loss_weights + t_loss * self.da_loss_weights + 0 * un_sup_pred[0].mean() + 0 * un_sup_pred[1].mean() + 0 * un_sup_pred[2].mean()
                # else:
                    # loss = loss + 0 * d_loss + 0 * t_loss + 0 * un_sup_pred[0].mean() + 0 * un_sup_pred[1].mean() + 0 * un_sup_pred[2].mean()

                if self.RANK != -1:
                    loss *= self.WORLD_SIZE  # gradient averaged between devices in DDP mode

            self.update_optimizer(loss, ni) 

            # Log
            if self.RANK in [-1, 0]:
                self.meter.update(loss_items)
                mloss_count= len(self.meter.meters.items())
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * (mloss_count+2)) % (
                    f'{self.epoch}/{self.epochs - 1}', mem,  targets.shape[0], imgs.shape[-1], *self.meter.get_avg()))
                callbacks.run('on_train_batch_end', ni, self.model, imgs, targets, paths, self.plots, self.sync_bn, self.cfg.Dataset.np)
        # end batch ------------------------------------------------------------------------------------------------
        # Scheduler
        self.lr = [x['lr'] for x in self.optimizer.param_groups]  # for loggers
        self.scheduler.step()  

    def after_train(self, callbacks, val):
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, clss)
        self._write_dqa_pseudo_stats()
        self._write_dqa0835_pseudo_memory()
        skip_best_val = os.getenv('ET_SKIP_AFTER_TRAIN_BEST_VAL', '').strip().lower() in {'1', 'true', 'yes', 'on'}
        if self.RANK in [-1, 0]:
            for f in self.last, self.best:
                if f.exists():
                    strip_optimizer(f)  # strip optimizers
                    if f is self.best:
                        if skip_best_val:
                            LOGGER.info(f'\nSkipping final best-checkpoint validation for {f} because ET_SKIP_AFTER_TRAIN_BEST_VAL is enabled.')
                        else:
                            LOGGER.info(f'\nValidating {f}...')
                            try:
                                # val_ssod = self.cfg.SSOD.train_domain
                                results, _, _, _ = val.run(self.data_dict,
                                                    batch_size=self.batch_size // self.WORLD_SIZE * 2,
                                                    imgsz=self.imgsz,
                                                    model=attempt_load(f, self.device).half(),
                                                    conf_thres=self.cfg.val_conf_thres,
                                                    iou_thres=0.65,  # best pycocotools results at 0.65
                                                    single_cls=self.single_cls,
                                                    dataloader=self.val_loader,
                                                    save_dir=self.save_dir,
                                                    save_json=False,
                                                    verbose=True,
                                                    plots=True,
                                                    callbacks=callbacks,
                                                    compute_loss=self.compute_loss,
                                                    num_points=self.cfg.Dataset.np,
                                                    val_ssod=self.cfg.SSOD.train_domain,
                                                    val_kp=self.cfg.Dataset.val_kp)  # val best model with plots
                            except Exception as exc:
                                LOGGER.warning(f'Final best-checkpoint validation failed for {f}: {exc}')

            callbacks.run('on_train_end', self.last, self.best, self.plots, self.epoch)
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
       
        torch.cuda.empty_cache()
        return results
    
    def split_predict_and_feature(self, total_pred, total_feature, n_img):
            sup_feature = [total_feature[0][:n_img, :, :, :], total_feature[1][:n_img, :, :, :], total_feature[2][:n_img, :, :, :]]
            un_sup_feature = [total_feature[0][n_img:, :, :, :], total_feature[1][n_img:, :, :, :], total_feature[2][n_img:, :, :, :]]
            if self.model_type == 'yolov5':
                sup_pred = [total_pred[0][:n_img, :, :, :, :], total_pred[1][:n_img, :, :, :, :], total_pred[2][:n_img, :, :, :, :]]
                un_sup_pred = [total_pred[0][n_img:, :, :, :, :], total_pred[1][n_img:, :, :, :, :], total_pred[2][n_img:, :, :, :, :]]
            elif self.model_type in ['yolox', 'yoloxkp']:
                sup_pred = [total_pred[0][:n_img, :, :], total_pred[1][:n_img, :, :], total_pred[2][:n_img, :, :]]
                un_sup_pred = [total_pred[0][n_img:, :, :], total_pred[1][n_img:, :, :], total_pred[2][n_img:, :, :]]
            elif self.model_type == 'tal':
                sup_pred = [[total_pred[0][0][:n_img, :, :, :], total_pred[0][1][:n_img, :, :, :], total_pred[0][2][:n_img, :, :, :]], total_pred[1][:n_img, :, :], total_pred[2][:n_img, :, :]]
                un_sup_pred = [[total_pred[0][0][n_img:, :, :, :], total_pred[0][1][n_img:, :, :, :], total_pred[0][2][n_img:, :, :, :]], total_pred[1][n_img:, :, :], total_pred[2][n_img:, :, :]]
            # elif self.model_type == 'yoloxkp':
            #     sup_pred = [total_pred[0][:n_img, :, :], total_pred[1], total_pred[2], total_pred[3], total_pred[4], total_pred[5]]
            #     un_sup_pred = [total_pred[0][n_img:, :, :], total_pred[1], total_pred[2], total_pred[3], total_pred[4], total_pred[5]]
            else:
                raise NotImplementedError
            return sup_pred, sup_feature, un_sup_pred, un_sup_feature
    
    def train_instance(self, imgs, targets, paths, unlabeled_imgs, unlabeled_imgs_ori, unlabeled_gt, unlabeled_paths, unlabeled_M, ni, pbar, callbacks):
        n_img, _, _, _ = imgs.shape
        n_pse_img, _,_,_ = unlabeled_imgs.shape
        invalid_target_shape = True
        unlabeled_targets = torch.zeros(8)

        # Teacher Model Forward
        extra_teacher_outs = []
        with amp.autocast(enabled=self.cuda):
            #build pseudo label via pred from teacher model
            with torch.no_grad():
                if self.model_type in ['yolov5']:
                    (teacher_pred, train_out), teacher_feature = self.ema.ema(unlabeled_imgs_ori, augment=False)
                # elif self.model_type == 'tal':
                #     teacher_pred, teacher_feature = self.ema.ema(unlabeled_imgs_ori, augment=False)
                # elif self.model_type == 'yoloxkp':
                #     teacher_pred, teacher_feature = self.ema.ema(unlabeled_imgs_ori, augment=False)
                    # teacher_pred = torch.cat(outputs, 1)
                else:
                    raise NotImplementedError
                
                if len(self.extra_teacher_models) > 0 :
                    for teacher_model in self.extra_teacher_models:
                        teacher_out = teacher_model(unlabeled_imgs_ori)[0]
                        extra_teacher_outs.append(teacher_out)

        if len(self.extra_teacher_models) > 0 and len(extra_teacher_outs) > 0 :
            unlabeled_targets, unlabeled_imgs, invalid_target_shape = self.pseudo_label_creator.create_pseudo_label_online_with_extra_teachers(teacher_pred, extra_teacher_outs, copy.deepcopy(unlabeled_imgs), unlabeled_M, self.extra_teacher_class_idxs, self.RANK)
        elif len(self.extra_teacher_models) == 0 :
            if self.cfg.SSOD.pseudo_label_type == 'LabelMatch':
                self.pseudo_label_creator.update(targets, n_img, n_pse_img)
            unlabeled_targets, invalid_target_shape = self.pseudo_label_creator.create_pseudo_label_online_with_gt(teacher_pred, copy.deepcopy(unlabeled_imgs), unlabeled_M, copy.deepcopy(unlabeled_imgs_ori), unlabeled_gt, self.RANK)
            unlabeled_imgs = unlabeled_imgs.to(self.device)
        else:    
            raise NotImplementedError

        unlabeled_targets = self._refine_dqa0835_targets_with_memory(
            unlabeled_targets,
            unlabeled_paths,
            invalid_target_shape,
        )
        unlabeled_targets = self._apply_dqa0836_scolq(unlabeled_targets, invalid_target_shape)

        with amp.autocast(enabled=self.cuda):
            if self.cfg.FedSTO.unlabeled_only_client:
                un_sup_pred, un_sup_feature = self.model(unlabeled_imgs)
                sup_loss = un_sup_pred[0].sum() * 0.0
                sup_loss_items = dict(box=0, obj=0, cls=0)
            else:
                total_imgs = torch.cat([imgs, unlabeled_imgs], 0)
                total_pred, total_feature = self.model(total_imgs)  # forward
                sup_pred, sup_feature, un_sup_pred, un_sup_feature = self.split_predict_and_feature(total_pred, total_feature, n_img)
                sup_loss, sup_loss_items = self.compute_loss(sup_pred, targets.to(self.device)) 

                #计算domain adaptation部分loss
                d_loss = self.domain_loss(sup_feature)
                t_loss = self.target_loss(un_sup_feature) 
                if self.cfg.SSOD.with_da_loss:
                    sup_loss = sup_loss + d_loss * self.da_loss_weights + t_loss * self.da_loss_weights
                else:
                    sup_loss = sup_loss + d_loss * 0 + t_loss * 0
            self._update_dqa_pseudo_stats(unlabeled_targets, invalid_target_shape, un_sup_feature)
            # total_t2 = time_sync()
            if self.RANK != -1:
                sup_loss *= self.WORLD_SIZE  # gradient averaged between devices in DDP mode
            if( invalid_target_shape ): #伪标签生成质量没有达到要求之前不计算loss
                un_sup_loss = torch.zeros(1, device=self.device) 
                un_sup_loss_items = dict(ss_box=0, ss_obj=0, ss_cls=0)
                un_sup_loss = un_sup_loss * 0.0
            else:
                un_sup_loss, un_sup_loss_items = self.compute_un_sup_loss(un_sup_pred, unlabeled_targets.to(self.device))  
                # un_sup_loss = un_sup_loss * self.cfg.SSOD.teacher_loss_weight
            if self.RANK != -1:
                un_sup_loss *= self.WORLD_SIZE
        loss = sup_loss + un_sup_loss * self.cfg.SSOD.teacher_loss_weight
        # if self.cfg.SSOD.imitate_teacher:
            # loss += loss_imitate
        self.update_optimizer(loss, ni) 
        
        # Log
        if self.RANK in [-1, 0]:
            self.meter.update(sup_loss_items)
            self.meter.update(un_sup_loss_items)
            if invalid_target_shape: #no pseudo label created
                hit_rate = dict(tp=0, fp_cls=0, fp_loc=0, pse_num=0, gt_num=0)
                self.meter.update(hit_rate)
            else:
                if self.target_with_gt:
                    tp_rate, fp_cls_rate, fp_loc_rate, pse_num, gt_num = check_pseudo_label_with_gt(unlabeled_targets, unlabeled_gt, \
                        ignore_thres_low=self.compute_un_sup_loss.ignore_thres_low, ignore_thres_high=self.compute_un_sup_loss.ignore_thres_high, \
                        batch_size=self.batch_size // self.WORLD_SIZE)
                else:
                    tp_rate, fp_loc_rate, pse_num, gt_num = check_pseudo_label(unlabeled_targets, \
                        ignore_thres_low=self.compute_un_sup_loss.ignore_thres_low, ignore_thres_high=self.compute_un_sup_loss.ignore_thres_high, \
                            batch_size=self.batch_size // self.WORLD_SIZE)
                    fp_cls_rate = 0
                hit_rate = dict(tp=tp_rate, fp_cls=fp_cls_rate, fp_loc=fp_loc_rate, pse_num=pse_num, gt_num=gt_num)
                self.meter.update(hit_rate)


            mloss_count= len(self.meter.meters.items())
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%10s' * 2 + '%10.4g' * (mloss_count + 2 )) % (
                f'{self.epoch}/{self.epochs - 1}', mem, targets.shape[0], imgs.shape[-1], *self.meter.get_avg()))
            
            callbacks.run('on_train_batch_end', ni, self.model, imgs, targets, paths, self.plots, self.sync_bn, self.cfg.Dataset.np)

    def train_with_unlabeled(self, callbacks):
        # hit_rate = dict(p_rate=0, r_rate=0); 

        if self.epoch_adaptor:
            self.nb = len(self.unlabeled_dataloader)  # number of batches
            pbar = enumerate(self.unlabeled_dataloader)
            if self.RANK in [-1, 0]:
                pbar = tqdm(pbar, total=self.nb)  # progress bar
            self.optimizer.zero_grad()
            for i , (target_imgs, target_gt, target_paths, _, target_imgs_ori, target_M) in pbar:
                ni = i + self.nb * self.epoch  # number integrated batches (since train start)
                imgs, targets, paths, _ = next(self.train_loader.__iter__())
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0 
                target_imgs = target_imgs.to(self.device, non_blocking=True).float() / 255.0 
                target_imgs_ori = target_imgs_ori.to(self.device, non_blocking=True).float() / 255.0 
                self.train_instance(imgs, targets, paths, target_imgs, target_imgs_ori, target_gt, target_paths, target_M, ni, pbar, callbacks)
        else:
            pbar = enumerate(self.train_loader)
            if self.RANK in [-1, 0]:
                pbar = tqdm(pbar, total=self.nb)  # progress bar
            self.optimizer.zero_grad()
            for i , (imgs, targets, paths, _) in pbar:
                ni = i + self.nb * self.epoch  # number integrated batches (since train start)
                target_imgs, target_gt, target_paths, _, target_imgs_ori, target_M= next(self.unlabeled_dataloader.__iter__())
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0 
                target_imgs = target_imgs.to(self.device, non_blocking=True).float() / 255.0 
                target_imgs_ori = target_imgs_ori.to(self.device, non_blocking=True).float() / 255.0 
                self.train_instance(imgs, targets, paths, target_imgs, target_imgs_ori, target_gt, target_paths, target_M, ni, pbar, callbacks)
            
        # end batch ------------------------------------------------------------------------------------------------
        
        # Scheduler
        self.lr = [x['lr'] for x in self.optimizer.param_groups]  # for loggers
        self.scheduler.step()  
