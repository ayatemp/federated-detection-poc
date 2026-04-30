from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
ET_ROOT = REPO_ROOT / "navigating_data_heterogeneity" / "vendor" / "efficientteacher"
if str(ET_ROOT) not in sys.path:
    sys.path.insert(0, str(ET_ROOT))

# EfficientTeacher's plotting module imports seaborn at module load time, but this
# visualizer does not use the seaborn-dependent plotting helpers. Keep the import
# path usable in lean notebook kernels where seaborn is not installed.
if "seaborn" not in sys.modules:
    try:
        import seaborn  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["seaborn"] = types.ModuleType("seaborn")

from models.backbone.experimental import attempt_load  # noqa: E402
from utils.datasets import letterbox  # noqa: E402
from utils.general import non_max_suppression_ssod, scale_coords  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402


BDD_NAMES = [
    "person",
    "rider",
    "car",
    "bus",
    "truck",
    "bike",
    "motor",
    "traffic light",
    "traffic sign",
    "train",
]

VARIANT_CAPS = {
    "et_capped_balanced_obj": {"max_pseudo_per_image": 28, "max_pseudo_per_class_image": 8},
    "et_high_precision_capped": {"max_pseudo_per_image": 24, "max_pseudo_per_class_image": 6},
}

WEATHER_BY_CLIENT_ID = {0: "overcast", 1: "rainy", 2: "snowy"}

CLASS_COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]


@dataclass
class VisualCompareConfig:
    experiment_root: Path = REPO_ROOT / "efficient_teacher" / "efficientteacher_pseudogt_dqa_probe_2h"
    variant: str = "et_default_lr1e-3"
    client_id: int = 0
    image_list: Path | None = None
    weights: Path | None = None
    output_dir: Path | None = None
    num_images: int = 30
    start_index: int = 0
    sample_mode: str = "first"
    random_seed: int = 0
    img_size: int = 640
    raw_conf_thres: float = 0.001
    raw_iou_thres: float | None = None
    max_raw_detections: int = 300
    pseudo_max_detections: int = 300
    max_pseudo_per_image: int | None = None
    max_pseudo_per_class_image: int | None = None
    show_ignored_pseudo: bool = False
    panel_width: int = 640
    draw_raw_labels: bool = False
    draw_pseudo_labels: bool = True
    draw_gt_labels: bool = True
    device: str = ""
    half: bool = False


def default_image_list(exp_root: Path) -> Path:
    return exp_root / "mini_lists" / "server_cloudy_val_512.txt"


def variant_config_path(exp_root: Path, variant: str, client_id: int) -> Path:
    weather = WEATHER_BY_CLIENT_ID[int(client_id)]
    return exp_root / "configs" / f"{variant}_client{client_id}_{weather}.yaml"


def warmup_checkpoint() -> Path:
    return (
        REPO_ROOT
        / "dynamic_quality_aware_classwise_aggregation"
        / "efficientteacher_dqa_ver2_scene_12h"
        / "global_checkpoints"
        / "round000_warmup.pt"
    )


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        idx = parts.index("images")
    except ValueError:
        raise ValueError(f"Image path does not contain an images directory: {image_path}") from None
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_image_list(path: Path) -> list[Path]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(Path(text.split()[0]))
    return rows


def select_images(paths: list[Path], cfg: VisualCompareConfig) -> list[Path]:
    start = max(0, int(cfg.start_index))
    count = max(0, int(cfg.num_images))
    if cfg.sample_mode == "random":
        rng = np.random.default_rng(int(cfg.random_seed))
        pool = paths[start:]
        if not pool:
            return []
        idx = rng.choice(len(pool), size=min(count, len(pool)), replace=False)
        return [pool[int(i)] for i in idx]
    return paths[start : start + count]


def load_yolo_gt(image_path: Path, image_shape: tuple[int, int]) -> list[dict[str, Any]]:
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        return []
    h, w = image_shape
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, x, y, bw, bh = map(float, parts[:5])
        x1 = (x - bw / 2.0) * w
        y1 = (y - bh / 2.0) * h
        x2 = (x + bw / 2.0) * w
        y2 = (y + bh / 2.0) * h
        boxes.append({"xyxy": [x1, y1, x2, y2], "cls": int(cls), "conf": None, "state": "gt"})
    return boxes


def load_ssod_config(cfg: VisualCompareConfig) -> dict[str, Any]:
    path = variant_config_path(cfg.experiment_root, cfg.variant, cfg.client_id)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload.get("SSOD", {})


def resolve_caps(cfg: VisualCompareConfig) -> tuple[int, int]:
    defaults = VARIANT_CAPS.get(cfg.variant, {})
    max_img = cfg.max_pseudo_per_image
    max_cls = cfg.max_pseudo_per_class_image
    if max_img is None:
        max_img = int(defaults.get("max_pseudo_per_image", 0))
    if max_cls is None:
        max_cls = int(defaults.get("max_pseudo_per_class_image", 0))
    return int(max_img or 0), int(max_cls or 0)


def cap_detections(dets: torch.Tensor, max_per_image: int = 0, max_per_class_image: int = 0) -> torch.Tensor:
    if dets is None or dets.numel() == 0:
        return dets
    if max_per_class_image > 0:
        parts = []
        class_ids = dets[:, 5].to(torch.long)
        for class_id in torch.unique(class_ids):
            rows = dets[class_ids == class_id]
            order = rows[:, 4].argsort(descending=True)
            parts.append(rows[order[:max_per_class_image]])
        dets = torch.cat(parts, dim=0) if parts else dets[:0]
    if max_per_image > 0 and dets.shape[0] > max_per_image:
        order = dets[:, 4].argsort(descending=True)
        dets = dets[order[:max_per_image]]
    return dets


def tensor_to_boxes(dets: torch.Tensor, *, low: float | None = None, high: float | None = None) -> list[dict[str, Any]]:
    boxes = []
    if dets is None or dets.numel() == 0:
        return boxes
    for row in dets.detach().cpu().numpy():
        x1, y1, x2, y2, conf, cls, obj_conf, cls_conf = row.tolist()
        state = "raw"
        if low is not None and high is not None:
            if conf >= high:
                state = "reliable"
            elif conf >= low:
                state = "uncertain"
            else:
                state = "ignored"
        boxes.append(
            {
                "xyxy": [x1, y1, x2, y2],
                "cls": int(cls),
                "conf": float(conf),
                "obj_conf": float(obj_conf),
                "cls_conf": float(cls_conf),
                "state": state,
            }
        )
    return boxes


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        return ImageFont.load_default()


def draw_boxes(
    image_bgr: np.ndarray,
    boxes: list[dict[str, Any]],
    *,
    title: str,
    label_boxes: bool,
    panel_width: int,
) -> Image.Image:
    image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = _font()
    w, h = image.size
    line_width = max(2, round(min(w, h) / 360))

    for item in boxes:
        cls = int(item["cls"])
        x1, y1, x2, y2 = [float(v) for v in item["xyxy"]]
        state = item.get("state", "raw")
        if state == "gt":
            color = (0, 105, 255)
        elif state == "reliable":
            color = (0, 170, 80)
        elif state == "uncertain":
            color = (255, 145, 0)
        elif state == "ignored":
            color = (150, 150, 150)
        else:
            color = CLASS_COLORS[cls % len(CLASS_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        if not label_boxes:
            continue
        name = BDD_NAMES[cls] if 0 <= cls < len(BDD_NAMES) else str(cls)
        conf = item.get("conf")
        suffix = "" if conf is None else f" {conf:.2f}"
        if state in {"reliable", "uncertain", "ignored"}:
            suffix += f" {state[0].upper()}"
        text = f"{name}{suffix}"
        bbox = draw.textbbox((x1, y1), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        y_text = max(0, y1 - th - 4)
        draw.rectangle([x1, y_text, x1 + tw + 4, y_text + th + 4], fill=color)
        draw.text((x1 + 2, y_text + 2), text, fill=(255, 255, 255), font=font)

    if panel_width and image.size[0] != panel_width:
        new_h = max(1, round(image.size[1] * panel_width / image.size[0]))
        image = image.resize((panel_width, new_h), Image.Resampling.LANCZOS)

    title_h = 34
    titled = Image.new("RGB", (image.size[0], image.size[1] + title_h), (245, 245, 245))
    titled.paste(image, (0, title_h))
    draw = ImageDraw.Draw(titled)
    draw.text((10, 8), title, fill=(20, 20, 20), font=font)
    return titled


def hstack_panels(panels: list[Image.Image], gap: int = 8) -> Image.Image:
    height = max(panel.size[1] for panel in panels)
    width = sum(panel.size[0] for panel in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.size[0] + gap
    return canvas


class PseudoGTVisualComparer:
    def __init__(self, cfg: VisualCompareConfig) -> None:
        self.cfg = cfg
        self.cfg.experiment_root = Path(self.cfg.experiment_root)
        self.image_list = Path(self.cfg.image_list) if self.cfg.image_list else default_image_list(self.cfg.experiment_root)
        self.output_dir = Path(self.cfg.output_dir) if self.cfg.output_dir else (
            self.cfg.experiment_root / "results" / "visual_compare" / self.cfg.variant
        )
        self.weights = Path(self.cfg.weights) if self.cfg.weights else warmup_checkpoint()
        self.ssod = load_ssod_config(self.cfg)
        self.max_per_image, self.max_per_class_image = resolve_caps(self.cfg)

    def describe(self) -> dict[str, Any]:
        return {
            "image_list": str(self.image_list),
            "weights": str(self.weights),
            "output_dir": str(self.output_dir),
            "variant": self.cfg.variant,
            "client_id": self.cfg.client_id,
            "num_images": self.cfg.num_images,
            "raw_conf_thres": self.cfg.raw_conf_thres,
            "raw_iou_thres": self.raw_iou_thres,
            "pseudo_nms_conf_thres": self.pseudo_conf_thres,
            "pseudo_iou_thres": self.pseudo_iou_thres,
            "ignore_thres_low": self.ignore_low,
            "ignore_thres_high": self.ignore_high,
            "max_pseudo_per_image": self.max_per_image,
            "max_pseudo_per_class_image": self.max_per_class_image,
        }

    @property
    def raw_iou_thres(self) -> float:
        return float(self.cfg.raw_iou_thres if self.cfg.raw_iou_thres is not None else self.ssod.get("nms_iou_thres", 0.65))

    @property
    def pseudo_conf_thres(self) -> float:
        return float(self.ssod.get("nms_conf_thres", 0.1))

    @property
    def pseudo_iou_thres(self) -> float:
        return float(self.ssod.get("nms_iou_thres", 0.65))

    @property
    def ignore_low(self) -> float:
        return float(self.ssod.get("ignore_thres_low", self.pseudo_conf_thres))

    @property
    def ignore_high(self) -> float:
        return float(self.ssod.get("ignore_thres_high", self.ignore_low))

    def load_model(self):
        device = select_device(self.cfg.device)
        model = attempt_load(str(self.weights), device=device, fuse=True)
        model.eval()
        half = bool(self.cfg.half and device.type != "cpu")
        if half:
            model.half()
        stride = int(model.stride.max())
        return model, device, half, stride

    @torch.no_grad()
    def infer_image(self, model, device, half: bool, stride: int, image_path: Path):
        im0 = cv2.imread(str(image_path))
        if im0 is None:
            raise FileNotFoundError(image_path)
        img = letterbox(im0, self.cfg.img_size, stride=stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).to(device)
        tensor = tensor.half() if half else tensor.float()
        tensor /= 255.0
        if tensor.ndim == 3:
            tensor = tensor[None]
        pred = model(tensor, augment=False)[0]
        if isinstance(pred, tuple):
            pred = pred[0]

        raw = non_max_suppression_ssod(
            pred.clone(),
            conf_thres=float(self.cfg.raw_conf_thres),
            iou_thres=self.raw_iou_thres,
            multi_label=bool(self.ssod.get("multi_label", False)),
            num_points=0,
            max_det=int(self.cfg.max_raw_detections),
        )[0]
        pseudo = non_max_suppression_ssod(
            pred.clone(),
            conf_thres=self.pseudo_conf_thres,
            iou_thres=self.pseudo_iou_thres,
            multi_label=bool(self.ssod.get("multi_label", False)),
            num_points=0,
            max_det=int(self.cfg.pseudo_max_detections),
        )[0]
        pseudo = cap_detections(pseudo, self.max_per_image, self.max_per_class_image)

        if raw is not None and raw.numel():
            scale_coords(tensor.shape[2:], raw[:, :4], im0.shape).round()
        if pseudo is not None and pseudo.numel():
            scale_coords(tensor.shape[2:], pseudo[:, :4], im0.shape).round()
        return im0, raw, pseudo

    def render_one(self, model, device, half: bool, stride: int, image_path: Path, index: int) -> dict[str, Any]:
        im0, raw, pseudo = self.infer_image(model, device, half, stride, image_path)
        gt_boxes = load_yolo_gt(image_path, im0.shape[:2])
        raw_boxes = tensor_to_boxes(raw)
        pseudo_boxes_all = tensor_to_boxes(pseudo, low=self.ignore_low, high=self.ignore_high)
        pseudo_boxes = [
            box for box in pseudo_boxes_all if self.cfg.show_ignored_pseudo or box["state"] != "ignored"
        ]

        reliable = sum(1 for box in pseudo_boxes_all if box["state"] == "reliable")
        uncertain = sum(1 for box in pseudo_boxes_all if box["state"] == "uncertain")
        ignored = sum(1 for box in pseudo_boxes_all if box["state"] == "ignored")

        panels = [
            draw_boxes(
                im0,
                gt_boxes,
                title=f"GT labels: {len(gt_boxes)}",
                label_boxes=bool(self.cfg.draw_gt_labels),
                panel_width=int(self.cfg.panel_width),
            ),
            draw_boxes(
                im0,
                raw_boxes,
                title=f"Raw inference NMS: {len(raw_boxes)} conf>={self.cfg.raw_conf_thres}",
                label_boxes=bool(self.cfg.draw_raw_labels),
                panel_width=int(self.cfg.panel_width),
            ),
            draw_boxes(
                im0,
                pseudo_boxes,
                title=f"PseudoGT: {len(pseudo_boxes)} R={reliable} U={uncertain} I={ignored}",
                label_boxes=bool(self.cfg.draw_pseudo_labels),
                panel_width=int(self.cfg.panel_width),
            ),
        ]
        merged = hstack_panels(panels)
        out_path = self.output_dir / f"{index:03d}_{image_path.stem}_{self.cfg.variant}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.save(out_path, quality=92)

        return {
            "index": index,
            "image": str(image_path),
            "label": str(label_path_for_image(image_path)),
            "output": str(out_path),
            "gt_count": len(gt_boxes),
            "raw_count": len(raw_boxes),
            "pseudo_count": len(pseudo_boxes),
            "pseudo_reliable_count": reliable,
            "pseudo_uncertain_count": uncertain,
            "pseudo_ignored_count": ignored,
        }

    def run(self) -> pd.DataFrame:
        image_paths = select_images(read_image_list(self.image_list), self.cfg)
        model, device, half, stride = self.load_model()
        rows = []
        for i, path in enumerate(image_paths, start=int(self.cfg.start_index)):
            rows.append(self.render_one(model, device, half, stride, Path(path), i))
        df = pd.DataFrame(rows)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_dir / "visual_compare_summary.csv", index=False)
        (self.output_dir / "visual_compare_config.json").write_text(
            json.dumps(self.describe(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return df


def run_visual_compare(**kwargs) -> pd.DataFrame:
    return PseudoGTVisualComparer(VisualCompareConfig(**kwargs)).run()
