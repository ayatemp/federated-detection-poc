import json
from pathlib import Path
from textwrap import dedent


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


NOTEBOOK_PATH = Path(
    "/Users/kakuayato/my-company/masters_research/navigating_data_heterogeneity/01_fedsto_ssfod.ipynb"
)


cells = [
    md(
        """
        # FedSTO Notebook: Navigating Data Heterogeneity in Federated Learning

        この notebook は、論文 **"Navigating Data Heterogeneity in Federated Learning: A Semi-Supervised Federated Object Detection" (NeurIPS 2023)** の FedSTO を、研究用に再現しやすい notebook 形式でまとめたものです。

        この notebook が扱うもの:
        - labels-at-server の SSFOD セットアップ
        - warm-up -> Selective Training -> Full Parameter Training の FL ループ
        - client ごとの local EMA teacher
        - pseudo label 生成
        - backbone / neck / head の分割と phase ごとの凍結
        - weighted aggregation
        - orthogonal enhancement の実用実装

        重要:
        - 論文の骨格は忠実に実装しています。
        - ただし、`tau1/tau2`, optimizer, batch size, exact ORN coefficient など論文本体が明示していない部分は、実験再現のための既定値を入れています。
        - Ultralytics 高レベル API を使う都合上、pseudo label の soft objectness を使う完全な論文損失とは一致しません。ここでは `tau_high` 以上を hard pseudo labels に落として学習し、medium / low confidence の統計も別保存します。
        - ORN は「neck/head への orthogonal regularization」を狙った実用実装として、phase 2 の学習後に重みを orthogonal projection で整える補助も入れています。
        """
    ),
    md(
        """
        ## 想定ディレクトリ構成

        最低限、以下のような構成を想定しています。

        ```text
        masters_research/navigating_data_heterogeneity/
        ├── 01_fedsto_ssfod.ipynb
        ├── data/
        │   ├── server/
        │   │   ├── images/
        │   │   │   ├── train/
        │   │   │   └── val/
        │   │   ├── labels/
        │   │   │   ├── train/
        │   │   │   └── val/
        │   │   └── data.yaml
        │   └── clients/
        │       ├── client_0/
        │       │   └── images_unlabeled/
        │       ├── client_1/
        │       │   └── images_unlabeled/
        │       └── client_2/
        │           └── images_unlabeled/
        └── runs/
        ```

        `server/data.yaml` は通常の YOLO 形式です。  
        client 側はラベル不要で、画像だけ置いてあれば notebook 側で pseudo-labeled dataset を生成します。

        ## exactness メモ

        論文準拠:
        - warm-up 50, phase1 100, phase2 150
        - local epoch = 1
        - local EMA = 0.999
        - phase1 は backbone のみ更新
        - phase2 は full parameter training + orthogonal enhancement

        論文未確定なので既定値で埋めているもの:
        - `tau_low=0.1`, `tau_high=0.6`
        - `client_sampling_ratio=1.0`
        - `optimizer=SGD`
        - `lambda_orn`
        """
    ),
    code(
        """
        # 必要に応じて最初に実行:
        # %pip install -q "numpy<2" "opencv-python<4.11" "ultralytics>=8.2,<9" pyyaml pandas matplotlib
        #
        # この環境では NumPy 2.x と OpenCV / Ultralytics の ABI 不整合が起きることがあります。
        # import error が出たら上の install を実行して kernel を再起動してください。

        import os
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("OMP_NUM_THREADS", "1")

        import json
        import math
        import random
        import shutil
        import time
        from copy import deepcopy
        from dataclasses import asdict, dataclass, field
        from pathlib import Path
        from typing import Dict, Iterable, List, Optional, Sequence, Tuple

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        import yaml
        from ultralytics import YOLO

        print("torch:", torch.__version__)
        print("cuda:", torch.cuda.is_available())
        print("mps:", getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

        if torch.cuda.is_available():
            DEVICE = "cuda:0"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            DEVICE = "mps"
        else:
            DEVICE = "cpu"

        print("device:", DEVICE)
        """
    ),
    code(
        """
        @dataclass
        class FedSTOConfig:
            base_model: str = "yolov5l.pt"
            seed: int = 42

            project_root: Path = Path("/Users/kakuayato/my-company/masters_research/navigating_data_heterogeneity")
            data_root: Path = project_root / "data"
            run_root: Path = project_root / "runs"

            server_data_yaml: Path = data_root / "server" / "data.yaml"
            clients_root: Path = data_root / "clients"

            warmup_rounds: int = 50
            phase1_rounds: int = 100
            phase2_rounds: int = 150
            local_epochs: int = 1

            imgsz: int = 640
            batch: int = 8
            workers: int = 0
            lr0: float = 0.01
            client_sampling_ratio: float = 1.0

            ema_decay: float = 0.999
            tau_low: float = 0.1
            tau_high: float = 0.6
            regression_obj_gate: float = 0.99
            nms_conf_thres: float = 0.1
            nms_iou_thres: float = 0.65
            lambda_orn: float = 1e-4

            phase1_backbone_layers: Tuple[int, ...] = tuple(range(0, 10))
            phase1_neck_layers: Tuple[int, ...] = tuple(range(10, 24))
            phase1_head_layers: Tuple[int, ...] = (24,)

            save_plots: bool = True
            symlink_images: bool = True
            copy_labels_for_server: bool = False
            demo_mode: bool = False
            demo_warmup_rounds: int = 1
            demo_phase1_rounds: int = 2
            demo_phase2_rounds: int = 2

            def effective_warmup_rounds(self) -> int:
                return self.demo_warmup_rounds if self.demo_mode else self.warmup_rounds

            def effective_phase1_rounds(self) -> int:
                return self.demo_phase1_rounds if self.demo_mode else self.phase1_rounds

            def effective_phase2_rounds(self) -> int:
                return self.demo_phase2_rounds if self.demo_mode else self.phase2_rounds

            def experiment_dir(self) -> Path:
                return self.run_root / time.strftime("%Y%m%d_%H%M%S")


        def seed_everything(seed: int = 42) -> None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)


        def ensure_dir(path: Path) -> Path:
            path.mkdir(parents=True, exist_ok=True)
            return path


        def save_yaml(path: Path, payload: dict) -> None:
            ensure_dir(path.parent)
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


        def load_yaml(path: Path) -> dict:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)


        def list_images(root: Path) -> List[Path]:
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


        def discover_client_unlabeled_dirs(clients_root: Path) -> List[Path]:
            dirs = []
            for client_dir in sorted(clients_root.glob("client_*")):
                candidate = client_dir / "images_unlabeled"
                if candidate.exists():
                    dirs.append(candidate)
            return dirs


        cfg = FedSTOConfig()
        seed_everything(cfg.seed)
        ensure_dir(cfg.project_root)
        ensure_dir(cfg.run_root)

        client_unlabeled_dirs = discover_client_unlabeled_dirs(cfg.clients_root)
        print("server_data_yaml:", cfg.server_data_yaml)
        print("client_unlabeled_dirs:", [str(p) for p in client_unlabeled_dirs])
        print("demo_mode:", cfg.demo_mode)
        """
    ),
    code(
        """
        def get_core_model(yolo: YOLO):
            model = yolo.model
            if hasattr(model, "model"):
                return model.model
            return model


        def infer_partitions(yolo: YOLO, cfg: FedSTOConfig) -> Dict[str, List[int]]:
            layers = get_core_model(yolo)
            n_layers = len(layers)
            default = {
                "backbone": [i for i in cfg.phase1_backbone_layers if i < n_layers],
                "neck": [i for i in cfg.phase1_neck_layers if i < n_layers],
                "head": [i for i in cfg.phase1_head_layers if i < n_layers],
            }
            assigned = set(default["backbone"]) | set(default["neck"]) | set(default["head"])
            if not assigned:
                return {
                    "backbone": list(range(max(1, n_layers - 2))),
                    "neck": [max(0, n_layers - 2)],
                    "head": [max(0, n_layers - 1)],
                }
            leftovers = [i for i in range(n_layers) if i not in assigned]
            default["neck"].extend(leftovers)
            return default


        def set_trainable_partitions(yolo: YOLO, train_parts: Sequence[str], cfg: FedSTOConfig) -> Dict[str, List[int]]:
            partitions = infer_partitions(yolo, cfg)
            train_idx = set()
            for key in train_parts:
                train_idx.update(partitions[key])

            for idx, module in enumerate(get_core_model(yolo)):
                requires_grad = idx in train_idx
                for param in module.parameters(recurse=True):
                    param.requires_grad = requires_grad
            return partitions


        def reset_all_trainable(yolo: YOLO) -> None:
            for param in yolo.model.parameters():
                param.requires_grad = True


        def average_state_dicts(weighted_states: List[Tuple[float, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            assert weighted_states, "weighted_states must not be empty"
            total = sum(weight for weight, _ in weighted_states)
            ref = weighted_states[0][1]
            avg = {}
            for key in ref:
                stacked = []
                for weight, state in weighted_states:
                    tensor = state[key].detach().float().cpu()
                    stacked.append((weight / total) * tensor)
                avg[key] = torch.stack(stacked, dim=0).sum(dim=0)
            return avg


        def save_yolo_checkpoint(yolo: YOLO, path: Path) -> Path:
            ensure_dir(path.parent)
            yolo.save(str(path))
            return path


        def load_yolo(source: Path | str) -> YOLO:
            return YOLO(str(source))


        def extract_partition_state(yolo: YOLO, partition_names: Sequence[str], cfg: FedSTOConfig) -> Dict[str, torch.Tensor]:
            partitions = infer_partitions(yolo, cfg)
            keep = set()
            for part in partition_names:
                keep.update(partitions[part])

            extracted = {}
            for name, tensor in yolo.model.state_dict().items():
                matched = False
                for idx in keep:
                    prefix = f"model.{idx}."
                    if name.startswith(prefix):
                        extracted[name] = tensor.detach().cpu().clone()
                        matched = True
                        break
                if not matched and not name.startswith("model."):
                    extracted[name] = tensor.detach().cpu().clone()
            return extracted


        def load_partial_state(yolo: YOLO, partial_state: Dict[str, torch.Tensor]) -> None:
            current = yolo.model.state_dict()
            for key, tensor in partial_state.items():
                if key in current and current[key].shape == tensor.shape:
                    current[key] = tensor.to(current[key].device, dtype=current[key].dtype)
            yolo.model.load_state_dict(current, strict=False)


        def orthogonal_projection_(weight: torch.Tensor) -> torch.Tensor:
            shape = weight.shape
            matrix = weight.reshape(shape[0], -1)
            transposed = False
            if matrix.shape[0] > matrix.shape[1]:
                matrix = matrix.T
                transposed = True
            q, _ = torch.linalg.qr(matrix, mode="reduced")
            if transposed:
                q = q.T
            q = q.reshape(shape)
            return q


        def apply_orthogonal_enhancement_(yolo: YOLO, cfg: FedSTOConfig) -> int:
            partitions = infer_partitions(yolo, cfg)
            target_layers = set(partitions["neck"]) | set(partitions["head"])
            touched = 0
            for idx, module in enumerate(get_core_model(yolo)):
                if idx not in target_layers:
                    continue
                if hasattr(module, "conv") and hasattr(module.conv, "weight"):
                    with torch.no_grad():
                        module.conv.weight.copy_(orthogonal_projection_(module.conv.weight.data))
                    touched += 1
                elif hasattr(module, "weight") and isinstance(module.weight, torch.Tensor) and module.weight.ndim >= 2:
                    with torch.no_grad():
                        module.weight.copy_(orthogonal_projection_(module.weight.data))
                    touched += 1
            return touched


        def summarize_trainable_params(yolo: YOLO) -> pd.DataFrame:
            rows = []
            for name, p in yolo.model.named_parameters():
                rows.append({
                    "name": name,
                    "shape": tuple(p.shape),
                    "requires_grad": bool(p.requires_grad),
                    "numel": int(p.numel()),
                })
            return pd.DataFrame(rows)
        """
    ),
    code(
        """
        class LocalEMATeacher:
            def __init__(self, source_model: YOLO, decay: float = 0.999):
                self.decay = decay
                self.teacher = load_yolo(save_yolo_checkpoint(source_model, Path("/tmp/fedsto_ema_bootstrap.pt")))
                self.teacher.model.eval()

            def reset_from_global(self, global_model: YOLO) -> None:
                self.teacher.model.load_state_dict(deepcopy(global_model.model.state_dict()))
                self.teacher.model.eval()

            def update_from_student(self, student_model: YOLO) -> None:
                with torch.no_grad():
                    teacher_state = self.teacher.model.state_dict()
                    student_state = student_model.model.state_dict()
                    for key in teacher_state:
                        if key in student_state and teacher_state[key].dtype.is_floating_point:
                            teacher_state[key].mul_(self.decay).add_(student_state[key].detach().cpu(), alpha=1 - self.decay)
                        elif key in student_state:
                            teacher_state[key].copy_(student_state[key].detach().cpu())
                    self.teacher.model.load_state_dict(teacher_state, strict=False)
                self.teacher.model.eval()


        def symlink_or_copy(src: Path, dst: Path, symlink: bool = True) -> None:
            ensure_dir(dst.parent)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if symlink:
                dst.symlink_to(src)
            else:
                shutil.copy2(src, dst)


        def yolo_xyxy_to_normalized_xywh(box_xyxy, width: int, height: int) -> Tuple[float, float, float, float]:
            x1, y1, x2, y2 = box_xyxy
            xc = ((x1 + x2) / 2.0) / width
            yc = ((y1 + y2) / 2.0) / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            return xc, yc, w, h


        def write_yolo_labels(label_path: Path, lines: List[str]) -> None:
            ensure_dir(label_path.parent)
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\\n".join(lines))
                if lines:
                    f.write("\\n")


        def build_dataset_yaml(dataset_root: Path, nc: int, names: Sequence[str]) -> Path:
            yaml_path = dataset_root / "data.yaml"
            save_yaml(
                yaml_path,
                {
                    "path": str(dataset_root.resolve()),
                    "train": "images/train",
                    "val": "images/val",
                    "nc": int(nc),
                    "names": list(names),
                },
            )
            return yaml_path


        def split_for_train_val(items: List[Path], val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Path], List[Path]]:
            items = list(items)
            rnd = random.Random(seed)
            rnd.shuffle(items)
            n_val = max(1, int(len(items) * val_ratio)) if len(items) > 1 else len(items)
            return items[n_val:], items[:n_val]


        def infer_class_names_from_server_yaml(server_yaml: Path) -> Tuple[int, List[str]]:
            meta = load_yaml(server_yaml)
            names = meta["names"]
            if isinstance(names, dict):
                names = [names[i] for i in sorted(names)]
            return int(meta["nc"]), list(names)


        def generate_pseudo_labeled_dataset(
            teacher_model: YOLO,
            unlabeled_dir: Path,
            out_dir: Path,
            cfg: FedSTOConfig,
        ) -> Tuple[Path, pd.DataFrame]:
            nc, names = infer_class_names_from_server_yaml(cfg.server_data_yaml)
            images = list_images(unlabeled_dir)
            train_imgs, val_imgs = split_for_train_val(images, val_ratio=0.1, seed=cfg.seed)

            records = []
            for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
                image_dst = ensure_dir(out_dir / "images" / split_name)
                label_dst = ensure_dir(out_dir / "labels" / split_name)

                preds = teacher_model.predict(
                    source=[str(p) for p in split_imgs],
                    conf=0.001,
                    iou=cfg.nms_iou_thres,
                    stream=True,
                    verbose=False,
                    device=DEVICE,
                )

                for src_path, pred in zip(split_imgs, preds):
                    symlink_or_copy(src_path, image_dst / src_path.name, symlink=cfg.symlink_images)

                    img_h, img_w = pred.orig_shape
                    lines = []
                    high_count = 0
                    medium_count = 0
                    low_count = 0

                    if pred.boxes is not None and len(pred.boxes) > 0:
                        xyxy = pred.boxes.xyxy.detach().cpu().numpy()
                        confs = pred.boxes.conf.detach().cpu().numpy()
                        clss = pred.boxes.cls.detach().cpu().numpy().astype(int)
                        for box, conf, cls_idx in zip(xyxy, confs, clss):
                            if conf >= cfg.tau_high:
                                xc, yc, w, h = yolo_xyxy_to_normalized_xywh(box, img_w, img_h)
                                lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                                high_count += 1
                            elif conf >= cfg.tau_low:
                                medium_count += 1
                            else:
                                low_count += 1

                    write_yolo_labels(label_dst / f"{src_path.stem}.txt", lines)
                    records.append(
                        {
                            "image": str(src_path),
                            "split": split_name,
                            "num_high": high_count,
                            "num_medium": medium_count,
                            "num_low": low_count,
                        }
                    )

            yaml_path = build_dataset_yaml(out_dir, nc, names)
            stats_df = pd.DataFrame(records)
            stats_df.to_csv(out_dir / "pseudo_label_stats.csv", index=False)
            return yaml_path, stats_df
        """
    ),
    code(
        """
        def train_with_ultralytics(
            model_path: Path | str,
            data_yaml: Path,
            out_project: Path,
            run_name: str,
            cfg: FedSTOConfig,
            train_parts: Optional[Sequence[str]] = None,
            apply_orn_after: bool = False,
        ) -> Tuple[Path, dict]:
            model = load_yolo(model_path)
            if train_parts is None:
                reset_all_trainable(model)
                partitions = infer_partitions(model, cfg)
            else:
                partitions = set_trainable_partitions(model, train_parts=train_parts, cfg=cfg)

            results = model.train(
                data=str(data_yaml),
                epochs=cfg.local_epochs,
                imgsz=cfg.imgsz,
                batch=cfg.batch,
                device=DEVICE,
                workers=cfg.workers,
                lr0=cfg.lr0,
                project=str(out_project),
                name=run_name,
                exist_ok=True,
                verbose=False,
                plots=False,
                save=True,
                val=True,
            )

            best_path = Path(results.save_dir) / "weights" / "best.pt"
            last_path = Path(results.save_dir) / "weights" / "last.pt"
            chosen_path = best_path if best_path.exists() else last_path

            trained = load_yolo(chosen_path)
            orn_modules = 0
            if apply_orn_after:
                orn_modules = apply_orthogonal_enhancement_(trained, cfg)
                save_yolo_checkpoint(trained, chosen_path)

            metrics = {
                "save_dir": str(results.save_dir),
                "orn_modules": int(orn_modules),
                "partition_backbone": partitions["backbone"],
                "partition_neck": partitions["neck"],
                "partition_head": partitions["head"],
            }
            return chosen_path, metrics


        def evaluate_model(model_path: Path | str, data_yaml: Path, cfg: FedSTOConfig, name: str) -> dict:
            model = load_yolo(model_path)
            result = model.val(
                data=str(data_yaml),
                imgsz=cfg.imgsz,
                batch=cfg.batch,
                device=DEVICE,
                workers=cfg.workers,
                verbose=False,
                plots=False,
                project=str(cfg.run_root / "eval"),
                name=name,
                exist_ok=True,
            )
            return {
                "map50": float(result.box.map50),
                "map50_95": float(result.box.map),
            }


        def sample_client_indices(num_clients: int, ratio: float, seed: int, round_idx: int) -> List[int]:
            rng = random.Random(seed + round_idx)
            all_idx = list(range(num_clients))
            k = max(1, math.ceil(num_clients * ratio))
            rng.shuffle(all_idx)
            return sorted(all_idx[:k])


        def warmup_server(global_model_path: Path, cfg: FedSTOConfig, exp_dir: Path) -> Path:
            warmup_dir = ensure_dir(exp_dir / "warmup")
            trained_path, _ = train_with_ultralytics(
                model_path=global_model_path,
                data_yaml=cfg.server_data_yaml,
                out_project=warmup_dir,
                run_name="server_warmup",
                cfg=cfg,
                train_parts=None,
                apply_orn_after=False,
            )
            return trained_path


        def client_phase1_update(
            client_id: int,
            global_model_path: Path,
            unlabeled_dir: Path,
            cfg: FedSTOConfig,
            exp_dir: Path,
        ) -> Tuple[Dict[str, torch.Tensor], pd.DataFrame, dict]:
            student = load_yolo(global_model_path)
            teacher = LocalEMATeacher(student, decay=cfg.ema_decay)
            teacher.reset_from_global(student)

            pseudo_root = ensure_dir(exp_dir / "phase1" / f"client_{client_id}" / "pseudo")
            pseudo_yaml, stats_df = generate_pseudo_labeled_dataset(teacher.teacher, unlabeled_dir, pseudo_root, cfg)

            client_run_dir = ensure_dir(exp_dir / "phase1" / "client_train")
            trained_path, train_meta = train_with_ultralytics(
                model_path=global_model_path,
                data_yaml=pseudo_yaml,
                out_project=client_run_dir,
                run_name=f"client_{client_id}",
                cfg=cfg,
                train_parts=("backbone",),
                apply_orn_after=False,
            )

            trained = load_yolo(trained_path)
            teacher.update_from_student(trained)
            backbone_state = extract_partition_state(trained, partition_names=("backbone",), cfg=cfg)

            meta = {
                "client_id": client_id,
                "pseudo_yaml": str(pseudo_yaml),
                "train_meta": train_meta,
                "num_unlabeled_images": int(len(list_images(unlabeled_dir))),
                "num_pseudo_rows": int(len(stats_df)),
                "num_high_total": int(stats_df["num_high"].sum()) if len(stats_df) else 0,
            }
            return backbone_state, stats_df, meta


        def client_phase2_update(
            client_id: int,
            global_model_path: Path,
            unlabeled_dir: Path,
            cfg: FedSTOConfig,
            exp_dir: Path,
        ) -> Tuple[Dict[str, torch.Tensor], pd.DataFrame, dict]:
            student = load_yolo(global_model_path)
            teacher = LocalEMATeacher(student, decay=cfg.ema_decay)
            teacher.reset_from_global(student)

            pseudo_root = ensure_dir(exp_dir / "phase2" / f"client_{client_id}" / "pseudo")
            pseudo_yaml, stats_df = generate_pseudo_labeled_dataset(teacher.teacher, unlabeled_dir, pseudo_root, cfg)

            client_run_dir = ensure_dir(exp_dir / "phase2" / "client_train")
            trained_path, train_meta = train_with_ultralytics(
                model_path=global_model_path,
                data_yaml=pseudo_yaml,
                out_project=client_run_dir,
                run_name=f"client_{client_id}",
                cfg=cfg,
                train_parts=None,
                apply_orn_after=True,
            )

            trained = load_yolo(trained_path)
            teacher.update_from_student(trained)
            full_state = {k: v.detach().cpu().clone() for k, v in trained.model.state_dict().items()}
            meta = {
                "client_id": client_id,
                "pseudo_yaml": str(pseudo_yaml),
                "train_meta": train_meta,
                "num_unlabeled_images": int(len(list_images(unlabeled_dir))),
                "num_pseudo_rows": int(len(stats_df)),
                "num_high_total": int(stats_df["num_high"].sum()) if len(stats_df) else 0,
            }
            return full_state, stats_df, meta


        def server_phase1_update(
            global_model_path: Path,
            aggregated_backbone: Dict[str, torch.Tensor],
            cfg: FedSTOConfig,
            exp_dir: Path,
            round_idx: int,
        ) -> Path:
            server_model = load_yolo(global_model_path)
            load_partial_state(server_model, aggregated_backbone)
            bootstrap_path = save_yolo_checkpoint(server_model, exp_dir / "phase1" / f"server_bootstrap_round_{round_idx}.pt")
            trained_path, _ = train_with_ultralytics(
                model_path=bootstrap_path,
                data_yaml=cfg.server_data_yaml,
                out_project=ensure_dir(exp_dir / "phase1" / "server_train"),
                run_name=f"server_round_{round_idx}",
                cfg=cfg,
                train_parts=None,
                apply_orn_after=False,
            )
            return trained_path


        def server_phase2_update(
            global_model_path: Path,
            aggregated_full_model: Dict[str, torch.Tensor],
            cfg: FedSTOConfig,
            exp_dir: Path,
            round_idx: int,
        ) -> Path:
            server_model = load_yolo(global_model_path)
            server_model.model.load_state_dict(aggregated_full_model, strict=False)
            bootstrap_path = save_yolo_checkpoint(server_model, exp_dir / "phase2" / f"server_bootstrap_round_{round_idx}.pt")
            trained_path, _ = train_with_ultralytics(
                model_path=bootstrap_path,
                data_yaml=cfg.server_data_yaml,
                out_project=ensure_dir(exp_dir / "phase2" / "server_train"),
                run_name=f"server_round_{round_idx}",
                cfg=cfg,
                train_parts=None,
                apply_orn_after=True,
            )
            return trained_path
        """
    ),
    code(
        """
        def run_fedsto(cfg: FedSTOConfig) -> Tuple[pd.DataFrame, Path]:
            seed_everything(cfg.seed)
            exp_dir = ensure_dir(cfg.experiment_dir())
            save_yaml(exp_dir / "config.yaml", {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()})

            client_dirs = discover_client_unlabeled_dirs(cfg.clients_root)
            assert client_dirs, f"No client directories found under: {cfg.clients_root}"
            assert cfg.server_data_yaml.exists(), f"Server YAML not found: {cfg.server_data_yaml}"

            global_model_path = exp_dir / "global_round0.pt"
            shutil.copy2(cfg.base_model, global_model_path) if Path(cfg.base_model).exists() else save_yolo_checkpoint(load_yolo(cfg.base_model), global_model_path)

            history_rows = []

            print("=== Warm-up ===")
            for warm_round in range(1, cfg.effective_warmup_rounds() + 1):
                global_model_path = warmup_server(global_model_path, cfg, exp_dir)
                metrics = evaluate_model(global_model_path, cfg.server_data_yaml, cfg, name=f"warmup_eval_{warm_round}")
                history_rows.append({
                    "phase": "warmup",
                    "round": warm_round,
                    "num_clients": 0,
                    "map50": metrics["map50"],
                    "map50_95": metrics["map50_95"],
                })
                print(f"warmup round={warm_round}: mAP50={metrics['map50']:.4f}")

            print("\\n=== Phase 1: Selective Training ===")
            for round_idx in range(1, cfg.effective_phase1_rounds() + 1):
                selected = sample_client_indices(len(client_dirs), cfg.client_sampling_ratio, cfg.seed, round_idx)
                weighted_states = []
                client_meta_rows = []

                for client_id in selected:
                    state, stats_df, meta = client_phase1_update(
                        client_id=client_id,
                        global_model_path=global_model_path,
                        unlabeled_dir=client_dirs[client_id],
                        cfg=cfg,
                        exp_dir=exp_dir,
                    )
                    weight = max(1, meta["num_unlabeled_images"])
                    weighted_states.append((weight, state))
                    client_meta_rows.append(meta)

                aggregated_backbone = average_state_dicts(weighted_states)
                global_model_path = server_phase1_update(
                    global_model_path=global_model_path,
                    aggregated_backbone=aggregated_backbone,
                    cfg=cfg,
                    exp_dir=exp_dir,
                    round_idx=round_idx,
                )
                metrics = evaluate_model(global_model_path, cfg.server_data_yaml, cfg, name=f"phase1_eval_{round_idx}")
                history_rows.append({
                    "phase": "phase1",
                    "round": round_idx,
                    "num_clients": len(selected),
                    "map50": metrics["map50"],
                    "map50_95": metrics["map50_95"],
                    "selected_clients": ",".join(map(str, selected)),
                    "pseudo_high_total": int(sum(m["num_high_total"] for m in client_meta_rows)),
                })
                print(f"phase1 round={round_idx}: mAP50={metrics['map50']:.4f}, selected={selected}")

            print("\\n=== Phase 2: Full Parameter Training + Orthogonal Enhancement ===")
            for round_idx in range(1, cfg.effective_phase2_rounds() + 1):
                selected = sample_client_indices(len(client_dirs), cfg.client_sampling_ratio, cfg.seed + 1000, round_idx)
                weighted_states = []
                client_meta_rows = []

                for client_id in selected:
                    state, stats_df, meta = client_phase2_update(
                        client_id=client_id,
                        global_model_path=global_model_path,
                        unlabeled_dir=client_dirs[client_id],
                        cfg=cfg,
                        exp_dir=exp_dir,
                    )
                    weight = max(1, meta["num_unlabeled_images"])
                    weighted_states.append((weight, state))
                    client_meta_rows.append(meta)

                aggregated_full = average_state_dicts(weighted_states)
                global_model_path = server_phase2_update(
                    global_model_path=global_model_path,
                    aggregated_full_model=aggregated_full,
                    cfg=cfg,
                    exp_dir=exp_dir,
                    round_idx=round_idx,
                )
                metrics = evaluate_model(global_model_path, cfg.server_data_yaml, cfg, name=f"phase2_eval_{round_idx}")
                history_rows.append({
                    "phase": "phase2",
                    "round": round_idx,
                    "num_clients": len(selected),
                    "map50": metrics["map50"],
                    "map50_95": metrics["map50_95"],
                    "selected_clients": ",".join(map(str, selected)),
                    "pseudo_high_total": int(sum(m["num_high_total"] for m in client_meta_rows)),
                })
                print(f"phase2 round={round_idx}: mAP50={metrics['map50']:.4f}, selected={selected}")

            history = pd.DataFrame(history_rows)
            history.to_csv(exp_dir / "history.csv", index=False)
            return history, global_model_path
        """
    ),
    md(
        """
        ## 実行メモ

        この notebook は、`server/data.yaml` と `clients/client_*/images_unlabeled` が整っていれば、そのまま FedSTO 実験を回せる構成です。

        推奨の進め方:
        1. まず `demo_mode=True` で数ラウンドだけ試す
        2. 分割・pseudo labels・phase1 の freeze が意図通りか確認する
        3. 問題なければ `demo_mode=False` に戻して論文設定へ寄せる

        注意:
        - `base_model="yolov5l.pt"` はローカルに無いと初回ダウンロードが走ります
        - 環境により `cv2` と `numpy` の相性で import error が出る場合があります
        - medium confidence pseudo labels は統計保存のみで、学習には直接入れていません
        - ORN は論文の数式そのものを training loss に足す代わりに、phase2 後の orthogonal enhancement として適用しています
        """
    ),
    code(
        """
        # まずは demo_mode=True で疎通確認するのがおすすめです
        cfg = FedSTOConfig(
            demo_mode=True,
            batch=4,
            local_epochs=1,
            client_sampling_ratio=1.0,
        )

        print(cfg)

        # 実行する場合はコメントアウトを外してください
        # history, final_model_path = run_fedsto(cfg)
        # display(history.tail())
        # print("final_model_path:", final_model_path)
        """
    ),
    code(
        """
        # 実験後の可視化セル
        #
        # history が存在していれば mAP の推移を描きます。

        if "history" in globals():
            fig, ax = plt.subplots(figsize=(10, 5))
            for phase_name, sub in history.groupby("phase"):
                ax.plot(sub["round"], sub["map50"], marker="o", linewidth=2, label=f"{phase_name} mAP50")
            ax.set_xlabel("Round")
            ax.set_ylabel("mAP50")
            ax.set_title("FedSTO training curve")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()
        else:
            print("history is not defined yet. Run the training cell first.")
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote notebook: {NOTEBOOK_PATH}")
