#!/usr/bin/env python3
"""Run Expert-Choice pseudoGT routing DQA.

05 is the non-residual follow-up to 03/04.  It does not compose checkpoints
after training.  Instead, it changes the pseudo-GT learning problem itself:

* generate stable pseudo boxes from the current global model;
* let several virtual experts select fixed-capacity box buckets;
* write a balanced pseudo dataset for each client;
* train clients on source GT + expert-choice pseudoGT;
* aggregate normally, then optionally run source/server repair.

The final table is written next to the 03 baselines for direct comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
PSEUDOGT_SCRIPTS = REPO_ROOT / "pseudogt_learnability" / "scripts"
PROTOCOL_VERSION = "scene_daynight_dqa_05_expert_choice_pseudogt_router_v1"

for path in (PROJECT_ROOT / "scripts", PROJECT_ROOT.parent, PSEUDOGT_SCRIPTS, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_pseudogt_learnability_02 as pl02  # noqa: E402
import run_pseudogt_learnability_03 as pl03  # noqa: E402
import run_scene_daynight_dqa_01_0 as base01_0  # noqa: E402
import run_scene_daynight_dqa_03_main_experiment as main03  # noqa: E402


DEFAULT_SOURCE_WORKSPACE = PROJECT_ROOT / "output" / "03_main_bn_residual_dqa_experiment"
DEFAULT_WORKSPACE = PROJECT_ROOT / "output" / "05_expert_choice_pseudogt_router_dqa"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: Any) -> float | None:
    return main03.as_float(value)


def round_tag(round_idx: int) -> str:
    return f"round{round_idx:03d}"


def seconds_to_hms(seconds: float | None) -> str:
    return main03.seconds_to_hms(seconds)


def parse_xyxy(raw: str) -> tuple[float, float, float, float]:
    values = [float(item) for item in str(raw).split()]
    if len(values) != 4:
        raise ValueError(f"Invalid xyxy: {raw!r}")
    return values[0], values[1], values[2], values[3]


def pseudo_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"Pseudo image path does not contain /images/: {image_path}") from exc
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def pseudo_image_label_pair(args: argparse.Namespace, client_tag: str, tag: str, image_path: Path) -> tuple[Path, Path]:
    """Return the pseudo image/label pair, tolerating rows that point to source images."""
    label_path = pseudo_label_path(image_path)
    if label_path.exists():
        return image_path, label_path

    stable_root = args.workspace_root / "pseudo_dataset" / f"03_{tag}_stable_aug" / client_tag
    stable_image = stable_root / "images" / "train" / image_path.name
    stable_label = stable_root / "labels" / "train" / f"{image_path.stem}.txt"
    if stable_image.exists() and stable_label.exists():
        return stable_image, stable_label
    return image_path, label_path


def xyxy_to_yolo_line(row: Mapping[str, Any], image_path: Path) -> str:
    with Image.open(image_path) as img:
        width, height = img.size
    x1, y1, x2, y2 = parse_xyxy(row["xyxy"])
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    x1 = min(max(float(x1), 0.0), width)
    x2 = min(max(float(x2), 0.0), width)
    y1 = min(max(float(y1), 0.0), height)
    y2 = min(max(float(y2), 0.0), height)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = max(0.0, x2 - x1) / width
    bh = max(0.0, y2 - y1) / height
    cls = int(row["class_id"])
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def expert_score(row: Mapping[str, Any], expert_id: int, class_counts: Counter[int], area_q1: float, area_q2: float) -> float:
    conf = float(row["conf"])
    stability = float(row["stability"])
    score = float(row["score"])
    area = float(row["area"])
    cls = int(row["class_id"])
    rarity = 1.0 / math.sqrt(float(class_counts[cls]) + 1.0)
    small_bonus = 1.0 if area <= area_q1 else 0.0
    large_bonus = 1.0 if area >= area_q2 else 0.0
    hard_stable = max(0.0, 1.0 - conf) * stability
    # Four virtual experts:
    # 0: high-stability clean regions
    # 1: rare-class regions
    # 2: small/scale-sensitive regions
    # 3: hard-but-stable regions
    mode = expert_id % 4
    if mode == 0:
        return 0.65 * score + 0.35 * stability
    if mode == 1:
        return 0.70 * score + 2.00 * rarity
    if mode == 2:
        return 0.60 * score + 0.20 * small_bonus + 0.08 * large_bonus
    return 0.55 * score + 0.35 * hard_stable + 0.10 * stability


def rank_rows_for_expert(
    rows: list[dict[str, Any]],
    expert_id: int,
    class_counts: Counter[int],
    area_q1: float,
    area_q2: float,
    load_bias: float,
) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: expert_score(row, expert_id, class_counts, area_q1, area_q2) + load_bias,
        reverse=True,
    )


def load_box_rows(box_table: Path) -> list[dict[str, Any]]:
    raw_rows = read_csv(box_table)
    line_index_by_image: defaultdict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        image = raw["image"]
        x1, y1, x2, y2 = parse_xyxy(raw["xyxy"])
        row: dict[str, Any] = dict(raw)
        row["line_index"] = line_index_by_image[image]
        row["area"] = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        line_index_by_image[image] += 1
        rows.append(row)
    return rows


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return values[idx]


def select_expert_choice_rows(
    rows: list[dict[str, Any]],
    *,
    expert_count: int,
    keep_fraction: float,
    max_class_fraction: float,
    load_biases: list[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return [], {"expert_counts": {}, "selected_boxes": 0}

    class_counts = Counter(int(row["class_id"]) for row in rows)
    areas = [float(row["area"]) for row in rows]
    area_q1 = quantile(areas, 0.33)
    area_q2 = quantile(areas, 0.67)
    target_total = max(expert_count, int(round(len(rows) * keep_fraction)))
    per_expert = max(1, math.ceil(target_total / expert_count))
    max_per_class = max(1, int(math.ceil(target_total * max_class_fraction)))

    selected_keys: set[tuple[str, int]] = set()
    selected: list[dict[str, Any]] = []
    selected_class_counts: Counter[int] = Counter()
    expert_counts: Counter[int] = Counter()
    load_biases = list(load_biases or [0.0] * expert_count)

    for expert_id in range(expert_count):
        bias = load_biases[expert_id] if expert_id < len(load_biases) else 0.0
        for row in rank_rows_for_expert(rows, expert_id, class_counts, area_q1, area_q2, bias):
            key = (row["image"], int(row["line_index"]))
            cls = int(row["class_id"])
            if key in selected_keys:
                continue
            if selected_class_counts[cls] >= max_per_class:
                continue
            selected_keys.add(key)
            selected_class_counts[cls] += 1
            expert_counts[expert_id] += 1
            copied = dict(row)
            copied["expert_id"] = expert_id
            selected.append(copied)
            if expert_counts[expert_id] >= per_expert:
                break

    if len(selected) < target_total:
        fallback = sorted(rows, key=lambda row: (float(row["score"]), float(row["stability"])), reverse=True)
        for row in fallback:
            key = (row["image"], int(row["line_index"]))
            cls = int(row["class_id"])
            if key in selected_keys:
                continue
            if selected_class_counts[cls] >= max_per_class:
                continue
            selected_keys.add(key)
            selected_class_counts[cls] += 1
            expert_id = min(range(expert_count), key=lambda item: expert_counts[item])
            expert_counts[expert_id] += 1
            copied = dict(row)
            copied["expert_id"] = expert_id
            selected.append(copied)
            if len(selected) >= target_total:
                break

    summary = {
        "input_boxes": len(rows),
        "target_boxes": target_total,
        "selected_boxes": len(selected),
        "expert_counts": {str(k): int(v) for k, v in sorted(expert_counts.items())},
        "selected_class_counts": {str(k): int(v) for k, v in sorted(selected_class_counts.items())},
        "area_q1": area_q1,
        "area_q2": area_q2,
        "max_per_class": max_per_class,
    }
    return selected, summary


def write_selected_pseudo_dataset(
    setup,
    args: argparse.Namespace,
    client_tag: str,
    round_idx: int,
    selected: list[dict[str, Any]],
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    tag = round_tag(round_idx)
    pseudo_root = args.workspace_root / "pseudo_dataset" / f"05_{tag}_expert_choice"
    image_dir = pseudo_root / client_tag / "images" / "train"
    label_dir = pseudo_root / client_tag / "labels" / "train"
    if args.force_pseudo and (pseudo_root / client_tag).exists():
        shutil.rmtree(pseudo_root / client_tag)
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    by_image: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_image[row["image"]].append(row)

    list_images: list[Path] = []
    selected_rows: list[dict[str, Any]] = []
    class_counts: Counter[int] = Counter()
    for raw_image, image_rows in sorted(by_image.items()):
        src_image, src_label = pseudo_image_label_pair(args, client_tag, tag, Path(raw_image))
        label_lines = src_label.read_text(encoding="utf-8").splitlines() if src_label.exists() else []
        out_image = image_dir / src_image.name
        out_label = label_dir / f"{src_image.stem}.txt"
        out_lines: list[str] = []
        for row in sorted(image_rows, key=lambda item: int(item["line_index"])):
            line_index = int(row["line_index"])
            if line_index < len(label_lines):
                out_lines.append(label_lines[line_index])
            elif row.get("xyxy"):
                out_lines.append(xyxy_to_yolo_line(row, src_image))
            else:
                continue
            cls = int(row["class_id"])
            class_counts[cls] += 1
            selected_rows.append(
                {
                    "round": tag,
                    "client": client_tag,
                    "expert_id": row["expert_id"],
                    "image": str(out_image.absolute()),
                    "source_pseudo_image": str(src_image.absolute()),
                    "source_image": row.get("source_image", ""),
                    "class_id": cls,
                    "conf": row.get("conf", ""),
                    "stability": row.get("stability", ""),
                    "score": row.get("score", ""),
                    "line_index": line_index,
                    "xyxy": row.get("xyxy", ""),
                }
            )
        if not out_lines:
            continue
        pl02.link_or_copy(src_image, out_image)
        out_label.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        # Keep the pseudo-dataset path itself. `resolve()` follows the image
        # symlink back to the original BDD image, which makes YOLO look for the
        # original label file instead of the generated pseudo label.
        list_images.append(out_image.absolute())

    train_list = setup.LIST_ROOT / f"pl05_{tag}_{client_tag}_expert_choice_train.txt"
    train_list.write_text("\n".join(str(path) for path in sorted(list_images)) + ("\n" if list_images else ""), encoding="utf-8")
    if not list_images:
        raise RuntimeError(f"Expert-choice selection produced no images for {tag} {client_tag}.")
    stats = {
        "train_list": str(train_list.resolve()),
        "pseudo_images_kept": len(list_images),
        "pseudo_boxes_kept": int(sum(class_counts.values())),
        "boxes_per_kept_image": float(sum(class_counts.values()) / max(1, len(list_images))),
        "class_counts": {str(k): int(v) for k, v in sorted(class_counts.items())},
    }
    return train_list, stats, selected_rows


def update_load_biases(expert_counts: Mapping[str, int], expert_count: int, strength: float) -> list[float]:
    counts = [float(expert_counts.get(str(idx), 0)) for idx in range(expert_count)]
    total = sum(counts) or 1.0
    target = total / expert_count
    # Under-used experts receive positive bias next round; over-used experts get
    # suppressed.  This is the pseudoGT analogue of loss-free balancing.
    return [strength * ((target - count) / target) for count in counts]


def apply_expert_choice_selection(
    setup,
    args: argparse.Namespace,
    pseudo_stats: dict[str, Any],
    round_idx: int,
    load_bias_state: dict[str, list[float]],
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    tag = round_tag(round_idx)
    stats_rows: list[dict[str, Any]] = []
    selected_box_rows: list[dict[str, Any]] = []
    out_clients: dict[str, Any] = {}
    next_bias_state: dict[str, list[float]] = {}

    for client_tag, client_stats in pseudo_stats["clients"].items():
        rows = load_box_rows(Path(client_stats["box_table"]))
        selected, selection_summary = select_expert_choice_rows(
            rows,
            expert_count=args.expert_count,
            keep_fraction=args.expert_keep_fraction,
            max_class_fraction=args.expert_max_class_fraction,
            load_biases=load_bias_state.get(client_tag, [0.0] * args.expert_count),
        )
        train_list, selected_stats, selected_rows = write_selected_pseudo_dataset(
            setup,
            args,
            client_tag,
            round_idx,
            selected,
        )
        selected_box_rows.extend(selected_rows)
        merged = {
            **client_stats,
            **selected_stats,
            "input_pseudo_boxes": selection_summary["input_boxes"],
            "target_selected_boxes": selection_summary["target_boxes"],
            "expert_counts": selection_summary["expert_counts"],
            "selected_class_counts": selection_summary["selected_class_counts"],
            "train_list": str(train_list.resolve()),
            "selection_summary": selection_summary,
        }
        out_clients[client_tag] = merged
        next_bias_state[client_tag] = update_load_biases(selection_summary["expert_counts"], args.expert_count, args.load_bias_strength)
        stats_rows.append(
            {
                "round": tag,
                "client": client_tag,
                "input_pseudo_boxes": selection_summary["input_boxes"],
                "selected_pseudo_boxes": selected_stats["pseudo_boxes_kept"],
                "selected_pseudo_images": selected_stats["pseudo_images_kept"],
                "boxes_per_kept_image": f"{selected_stats['boxes_per_kept_image']:.4f}",
                "expert_counts": json.dumps(selection_summary["expert_counts"], sort_keys=True),
                "next_load_biases": json.dumps(next_bias_state[client_tag]),
                "train_list": str(train_list.resolve()),
            }
        )

    write_csv(
        args.workspace_root / "stats" / f"05_{tag}_expert_choice_stats.csv",
        stats_rows,
        [
            "round",
            "client",
            "input_pseudo_boxes",
            "selected_pseudo_boxes",
            "selected_pseudo_images",
            "boxes_per_kept_image",
            "expert_counts",
            "next_load_biases",
            "train_list",
        ],
    )
    write_csv(
        args.workspace_root / "stats" / f"05_{tag}_expert_choice_boxes.csv",
        selected_box_rows,
        [
            "round",
            "client",
            "expert_id",
            "image",
            "source_pseudo_image",
            "source_image",
            "class_id",
            "conf",
            "stability",
            "score",
            "line_index",
            "xyxy",
        ],
    )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "round": tag,
        "clients": out_clients,
        "next_load_bias_state": next_bias_state,
    }
    (args.workspace_root / "stats" / f"05_{tag}_expert_choice_stats.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload, next_bias_state


@contextmanager
def patched_loss_box(loss_box: float | None) -> Iterator[None]:
    yield


def expert_choice_variant(args: argparse.Namespace) -> pl03.Variant:
    return pl03.Variant(
        name="expert_choice_pseudogt_router",
        train_scope=args.train_scope,
        aggregate_scope=args.aggregate_scope,
        client_epochs=1,
        client_lr0=args.client_lr,
        source_repeat=args.source_repeat,
        pseudo_repeat=args.pseudo_repeat,
        orthogonal_weight=args.orthogonal_weight,
        note="Expert-choice balanced pseudoGT selection before client training.",
    )


def write_client_config(
    setup,
    variant: pl03.Variant,
    client: dict[str, Any],
    start: Path,
    args: argparse.Namespace,
    round_idx: int,
) -> Path:
    tag = round_tag(round_idx)
    client_tag = f"client{client['id']}_{client['weather']}"
    run_name = f"pl05_{tag}_{variant.name}_{client_tag}"
    source_list = setup.LIST_ROOT / "server_cloudy_train.txt"
    pseudo_list = setup.LIST_ROOT / f"pl05_{tag}_{client_tag}_expert_choice_train.txt"
    cfg = setup.efficientteacher_config(
        name=run_name,
        train=source_list,
        val=setup.LIST_ROOT / "server_cloudy_val.txt",
        target=None,
        weights=str(start.resolve()),
        epochs=variant.client_epochs,
        train_scope=variant.train_scope,
        orthogonal_weight=variant.orthogonal_weight,
        batch_size=args.batch_size,
        workers=args.workers,
        device=pl03.config_device(args),
    )
    cfg["Dataset"]["train"] = pl03.train_expr(source_list, pseudo_list, variant)
    cfg["FedSTO"]["unlabeled_only_client"] = False
    cfg["SSOD"] = {"train_domain": False}
    pl03.apply_client_hyp(cfg, variant)
    if args.loss_box is not None:
        cfg.setdefault("Loss", {})
        cfg["Loss"]["box"] = float(args.loss_box)
    return setup.write_config(f"{run_name}.yaml", cfg)


def save_record(
    records: list[dict[str, str]],
    label: str,
    path: Path,
    kind: str,
    *,
    round_idx: int | str = "",
    client: str = "",
    variant: str = "",
) -> None:
    records.append(
        {
            "condition": "expert_choice_pseudogt_router",
            "label": label,
            "kind": kind,
            "round": str(round_idx),
            "client": client,
            "variant": variant,
            "path": str(path.resolve()),
        }
    )


def write_checkpoint_records(path: Path, records: list[dict[str, str]]) -> None:
    write_csv(path, records, ["condition", "label", "kind", "round", "client", "variant", "path"])


def run_expert_choice_round(
    setup,
    fedsto,
    current_global: Path,
    args: argparse.Namespace,
    clients: list[dict[str, Any]],
    *,
    round_idx: int,
    port_offset: int,
    load_bias_state: dict[str, list[float]],
) -> tuple[list[dict[str, str]], Path, dict[str, Any], dict[str, list[float]], int]:
    tag = round_tag(round_idx)
    variant = expert_choice_variant(args)
    print(f"\n=== expert_choice_pseudogt_router: {tag} ===")
    raw_pseudo_stats = pl03.generate_round_pseudo_labels(setup, current_global, args, clients, round_idx)
    expert_stats, next_load_bias_state = apply_expert_choice_selection(setup, args, raw_pseudo_stats, round_idx, load_bias_state)

    records: list[dict[str, str]] = []
    local_paths: list[Path] = []
    for client in clients:
        client_tag = f"client{client['id']}_{client['weather']}"
        start = fedsto.CLIENT_STATE_DIR / f"pl05_{tag}_{variant.name}_{client_tag}_start.pt"
        run_name = f"pl05_{tag}_{variant.name}_{client_tag}"
        raw_ckpt = fedsto.checkpoint_path(run_name)
        final_ckpt = args.workspace_root / "checkpoints" / f"{tag}_{variant.name}_{client_tag}.pt"

        if not args.dry_run and not fedsto.checkpoint_matches_protocol(start, PROTOCOL_VERSION):
            fedsto.make_start_checkpoint(current_global, start, protocol=PROTOCOL_VERSION, stage=f"{tag}_{variant.name}_{client_tag}_start")

        if not pl03.reusable_checkpoint(fedsto, final_ckpt, args.force):
            cfg = write_client_config(setup, variant, client, start, args, round_idx)
            raw_ckpt = pl03.run_train(
                setup,
                fedsto,
                cfg,
                dry_run=args.dry_run,
                gpus=args.gpus,
                master_port=args.master_port + port_offset,
            )
            port_offset += 1
            if not args.dry_run:
                fedsto.mark_checkpoint_protocol(raw_ckpt, PROTOCOL_VERSION, f"{tag}_{variant.name}_{client_tag}_raw")
                fedsto.make_start_checkpoint(raw_ckpt, final_ckpt, protocol=PROTOCOL_VERSION, stage=f"{tag}_{variant.name}_{client_tag}")
                pl03.cleanup_training_artifacts(raw_ckpt, start)

        local_paths.append(final_ckpt)
        save_record(records, f"{tag}_{client_tag}", final_ckpt, "client", round_idx=round_idx, client=client_tag, variant=variant.name)

    aggregate = args.workspace_root / "checkpoints" / f"{tag}_{variant.name}_aggregate.pt"
    if not args.dry_run and not pl03.reusable_checkpoint(fedsto, aggregate, args.force):
        fedsto.aggregate_checkpoints(local_paths, current_global, aggregate, backbone_only=(variant.aggregate_scope == "backbone"))
        fedsto.mark_checkpoint_protocol(aggregate, PROTOCOL_VERSION, f"{tag}_{variant.name}_aggregate")
    save_record(records, f"{tag}_expert_choice_aggregate", aggregate, "aggregate", round_idx=round_idx, variant=variant.name)

    if args.server_repair_epochs > 0:
        repair_records, repaired, port_offset = main03.run_server_repair_round(
            setup,
            fedsto,
            aggregate,
            args,
            condition="expert_choice_pseudogt_router",
            variant=main03.repair_variant("expert_choice_server_repair"),
            round_idx=round_idx,
            port_offset=port_offset,
        )
        for record in repair_records:
            record["condition"] = "expert_choice_pseudogt_router"
        records.extend(repair_records)
        next_global = repaired
    else:
        next_global = aggregate

    return records, next_global, expert_stats, next_load_bias_state, port_offset


def split_gap_metrics(by_label_split: dict[tuple[str, str], dict[str, str]], label: str) -> dict[str, Any]:
    return main03.split_gap_metrics(by_label_split, label)


def source_final_metrics(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_csv(args.source_workspace / "stats" / "03_main_experiment_final_metrics.csv")
    out = []
    for row in rows:
        copied: dict[str, Any] = dict(row)
        copied["experiment"] = "03_main"
        copied.setdefault("delta_vs_03_dqa_aggregate_map50_95", "0.000000" if row.get("checkpoint_label") == "bn_residual_dqa_final_aggregate" else "")
        out.append(copied)
    return out


def source_split_metrics(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [{**row, "experiment": "03_main"} for row in read_csv(args.source_workspace / "stats" / "03_main_experiment_split_metrics.csv")]


def write_final_metrics(args: argparse.Namespace, eval_records: list[dict[str, str]]) -> list[dict[str, Any]]:
    source_final = {
        row["checkpoint_label"]: row
        for row in read_csv(args.source_workspace / "stats" / "03_main_experiment_final_metrics.csv")
    }
    warm_m95 = as_float(source_final.get("warmup_global", {}).get("map50_95"))
    repair_m95 = as_float(source_final.get("warmup_server_repair_final", {}).get("map50_95"))
    dqa_agg_m95 = as_float(source_final.get("bn_residual_dqa_final_aggregate", {}).get("map50_95"))

    rows = [row for row in read_csv(args.workspace_root / "validation_reports" / "paper_protocol_eval_summary.csv") if row.get("status") == "ok"]
    totals = {row["checkpoint_label"]: row for row in rows if row.get("split") in {"scene_daynight_total", "total"}}
    by_label_split = {(row["checkpoint_label"], row["split"]): row for row in rows}
    meta = {row["label"]: row for row in eval_records}

    metric_rows = source_final_metrics(args)
    for row in metric_rows:
        row.setdefault("experiment", "03_main")
        row.setdefault("delta_vs_03_dqa_aggregate_map50_95", "")

    for label, total in totals.items():
        if label not in meta:
            continue
        m50 = as_float(total.get("map50"))
        m95 = as_float(total.get("map50_95"))
        gap = split_gap_metrics(by_label_split, label)
        metric_rows.append(
            {
                "experiment": "05_expert_choice",
                "checkpoint_label": label,
                "condition": "warmup + Expert-Choice pseudoGT router DQA",
                "kind": meta[label].get("kind", ""),
                "source_condition": "expert_choice_pseudogt_router",
                "round": meta[label].get("round", ""),
                "precision": total.get("precision", ""),
                "recall": total.get("recall", ""),
                "map50": "" if m50 is None else f"{m50:.6f}",
                "map50_95": "" if m95 is None else f"{m95:.6f}",
                "gain_vs_warmup_map50_95": "" if m95 is None or warm_m95 is None else f"{m95 - warm_m95:.6f}",
                "delta_vs_server_repair_map50_95": "" if m95 is None or repair_m95 is None else f"{m95 - repair_m95:.6f}",
                "delta_vs_03_dqa_aggregate_map50_95": "" if m95 is None or dqa_agg_m95 is None else f"{m95 - dqa_agg_m95:.6f}",
                **gap,
            }
        )

    fields = [
        "experiment",
        "checkpoint_label",
        "condition",
        "kind",
        "source_condition",
        "round",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "gain_vs_warmup_map50_95",
        "delta_vs_server_repair_map50_95",
        "delta_vs_03_dqa_aggregate_map50_95",
        "worst_split",
        "worst_split_map50_95",
        "day_avg_map50_95",
        "night_avg_map50_95",
        "day_night_gap_map50_95",
    ]
    for row in metric_rows:
        for field in fields:
            row.setdefault(field, "")
    write_csv(args.workspace_root / "stats" / "05_expert_choice_final_metrics.csv", metric_rows, fields)

    split_rows = source_split_metrics(args)
    for row in rows:
        label = row["checkpoint_label"]
        if label not in meta:
            continue
        split_rows.append(
            {
                "experiment": "05_expert_choice",
                "checkpoint_label": label,
                "condition": "warmup + Expert-Choice pseudoGT router DQA",
                "split": row["split"],
                "images": row.get("images", ""),
                "labels": row.get("labels", ""),
                "precision": row.get("precision", ""),
                "recall": row.get("recall", ""),
                "map50": row.get("map50", ""),
                "map50_95": row.get("map50_95", ""),
            }
        )
    write_csv(
        args.workspace_root / "stats" / "05_expert_choice_split_metrics.csv",
        split_rows,
        ["experiment", "checkpoint_label", "condition", "split", "images", "labels", "precision", "recall", "map50", "map50_95"],
    )
    return metric_rows


def write_report(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    ranked = sorted(
        [row for row in rows if as_float(row.get("map50_95")) is not None],
        key=lambda row: as_float(row.get("map50_95")) or -1.0,
        reverse=True,
    )
    lines = [
        "# 05 Expert-Choice pseudoGT Router DQA",
        "",
        f"- created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- protocol: {PROTOCOL_VERSION}",
        "",
        "## Ranking",
        "",
        "| rank | experiment | checkpoint | mAP50 | mAP50:95 | delta vs 03 DQA aggregate | condition |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(ranked, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row.get("experiment", "")),
                    str(row.get("checkpoint_label", "")),
                    str(row.get("map50", "")),
                    str(row.get("map50_95", "")),
                    str(row.get("delta_vs_03_dqa_aggregate_map50_95", "")),
                    str(row.get("condition", "")).replace("|", "/"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Design",
            "",
            "05 changes pseudoGT selection before training.  Each round creates stable pseudo boxes, then virtual experts select fixed-capacity class/scale/density buckets.  The detector is trained only on the selected balanced pseudoGT lists, then aggregated normally.",
            "",
        ]
    )
    (args.workspace_root / "05_expert_choice_pseudogt_router_report.md").write_text("\n".join(lines), encoding="utf-8")


def update_progress(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    *,
    round_idx: int,
    completed: int,
    total: int,
    start_time: float,
    checkpoint: Path,
) -> None:
    elapsed = time.monotonic() - start_time
    avg = elapsed / completed if completed else 0.0
    eta = avg * (total - completed)
    rows.append(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "expert_choice_pseudogt_router",
            "round": round_idx,
            "completed_steps": completed,
            "total_steps": total,
            "elapsed_seconds": f"{elapsed:.3f}",
            "eta_seconds": f"{eta:.3f}",
            "elapsed_hms": seconds_to_hms(elapsed),
            "eta_hms": seconds_to_hms(eta),
            "checkpoint": str(checkpoint.resolve()),
        }
    )
    write_csv(
        args.workspace_root / "stats" / "05_expert_choice_progress.csv",
        rows,
        ["created_utc", "stage", "round", "completed_steps", "total_steps", "elapsed_seconds", "eta_seconds", "elapsed_hms", "eta_hms", "checkpoint"],
    )


def tqdm_factory(args: argparse.Namespace, total: int):
    if args.no_progress:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:  # noqa: BLE001
        return None
    return tqdm(total=total, desc="05 Expert-Choice DQA", unit="round")


def notify(args: argparse.Namespace, message: str, *, title: str, status: str = "", error: str | None = None) -> None:
    try:
        from notebook_notify import notify_discord

        context = {
            "workspace": str(args.workspace_root.expanduser().resolve()),
            "rounds": args.rounds,
            "status": status,
        }
        metrics_path = args.workspace_root.expanduser().resolve() / "stats" / "05_expert_choice_final_metrics.csv"
        if metrics_path.exists():
            context["metrics_csv"] = str(metrics_path)
        if error:
            context["error"] = error[:500]
        print(notify_discord(message, title=title, context=context, fail_silently=True))
    except Exception as exc:  # noqa: BLE001
        print(f"Discord notification skipped: {exc}")


def str2bool(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-workspace", type=Path, default=DEFAULT_SOURCE_WORKSPACE)
    parser.add_argument("--warmup-checkpoint", type=Path, default=REPO_ROOT / "pseudogt_learnability" / "checkpoints" / "round000_warmup.pt")
    parser.add_argument("--client-limit", type=int, default=1500)
    parser.add_argument("--clients", default="all")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=33941)
    parser.add_argument("--device", default="")
    parser.add_argument("--train-scope", choices=["neck_head", "all", "backbone"], default="neck_head")
    parser.add_argument("--aggregate-scope", choices=["all", "backbone"], default="all")
    parser.add_argument("--client-lr", type=float, default=0.0008)
    parser.add_argument("--source-repeat", type=int, default=1)
    parser.add_argument("--pseudo-repeat", type=int, default=2)
    parser.add_argument("--loss-box", type=float, default=0.005)
    parser.add_argument("--orthogonal-weight", type=float, default=1e-4)
    parser.add_argument("--server-repair-epochs", type=int, default=1)
    parser.add_argument("--server-repair-lr", type=float, default=0.0008)
    parser.add_argument("--expert-count", type=int, default=4)
    parser.add_argument("--expert-keep-fraction", type=float, default=0.65)
    parser.add_argument("--expert-max-class-fraction", type=float, default=0.35)
    parser.add_argument("--load-bias-strength", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--nms-iou-thres", type=float, default=0.65)
    parser.add_argument("--match-iou", type=float, default=0.60)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--min-stability", type=float, default=0.72)
    parser.add_argument("--min-score", type=float, default=0.28)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-boxes-per-image", type=int, default=12)
    parser.add_argument("--max-images-per-client", type=int, default=0)
    parser.add_argument("--max-class-fraction", type=float, default=0.45)
    parser.add_argument("--min-class-keep", type=int, default=250)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--eval-splits", default="highway_day,highway_night,citystreet_day,citystreet_night,residential_day,residential_night,total")
    parser.add_argument("--eval-clients", action="store_true")
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--classwise", action="store_true")
    parser.add_argument("--no-eval-plots", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-pseudo", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--notify-start", action="store_true")
    parser.add_argument("--notify-end", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.source_workspace = args.source_workspace.expanduser().resolve()
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    (args.workspace_root / "stats").mkdir(parents=True, exist_ok=True)
    setup_args, setup, fedsto, manifest, clients, warmup = main03.prepare_workspace(args, args.workspace_root)

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION,
        "workspace": str(args.workspace_root.resolve()),
        "source_workspace": str(args.source_workspace.resolve()),
        "rounds": args.rounds,
        "policy": {
            "expert_count": args.expert_count,
            "expert_keep_fraction": args.expert_keep_fraction,
            "expert_max_class_fraction": args.expert_max_class_fraction,
            "load_bias_strength": args.load_bias_strength,
            "train_scope": args.train_scope,
            "aggregate_scope": args.aggregate_scope,
            "client_lr": args.client_lr,
            "source_repeat": args.source_repeat,
            "pseudo_repeat": args.pseudo_repeat,
            "loss_box": args.loss_box,
        },
        "server": manifest.get("server"),
        "clients": clients,
    }
    (args.workspace_root / "stats" / "05_expert_choice_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.setup_only:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("Setup complete.")
        return []

    setup_args.gpus = fedsto.resolve_gpus(setup_args.gpus)
    if not args.dry_run:
        fedsto.check_runtime_dependencies()

    records: list[dict[str, str]] = []
    save_record(records, "warmup_global", warmup, "warmup")
    current = warmup
    port_offset = 0
    load_bias_state: dict[str, list[float]] = {}
    expert_history: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    progress = tqdm_factory(args, args.rounds)
    start_time = time.monotonic()

    for idx in range(1, args.rounds + 1):
        round_records, current, expert_stats, load_bias_state, port_offset = run_expert_choice_round(
            setup,
            fedsto,
            current,
            setup_args,
            clients,
            round_idx=idx,
            port_offset=port_offset,
            load_bias_state=load_bias_state,
        )
        records.extend(round_records)
        expert_history.append(expert_stats)
        write_checkpoint_records(args.workspace_root / "stats" / "05_expert_choice_checkpoints.csv", records)
        update_progress(args, progress_rows, round_idx=idx, completed=idx, total=args.rounds, start_time=start_time, checkpoint=current)
        if progress is not None:
            progress.set_postfix(round=idx, eta=progress_rows[-1]["eta_hms"])
            progress.update(1)
    if progress is not None:
        progress.close()

    final_tag = round_tag(args.rounds)
    final_agg_label = f"{final_tag}_expert_choice_aggregate"
    final_repair_label = f"{final_tag}_expert_choice_pseudogt_router_server_repair"
    by_label = {row["label"]: row for row in records}
    eval_records = [
        {
            "condition": "warmup",
            "label": "warmup_global",
            "kind": "warmup",
            "round": "",
            "client": "",
            "variant": "",
            "path": str(warmup.resolve()),
        },
        {
            **by_label[final_agg_label],
            "label": "expert_choice_final_aggregate",
        },
    ]
    if args.server_repair_epochs > 0 and final_repair_label in by_label:
        eval_records.append({**by_label[final_repair_label], "label": "expert_choice_final_repair"})
    write_checkpoint_records(args.workspace_root / "stats" / "05_expert_choice_eval_checkpoints.csv", eval_records)

    run_payload = {
        **payload,
        "actual_runtime_seconds": time.monotonic() - start_time,
        "records": records,
        "expert_history": expert_history,
        "eval_records": eval_records,
    }
    (args.workspace_root / "stats" / "05_expert_choice_run_manifest.json").write_text(
        json.dumps(run_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = source_final_metrics(args)
    if args.evaluate:
        base01_0.run_evaluation(args, eval_records)
        rows = write_final_metrics(args, eval_records)
    else:
        # Still write the source baseline table so the notebook has a comparison
        # target before the long evaluation cell is run.
        fields = [
            "experiment",
            "checkpoint_label",
            "condition",
            "kind",
            "source_condition",
            "round",
            "precision",
            "recall",
            "map50",
            "map50_95",
            "gain_vs_warmup_map50_95",
            "delta_vs_server_repair_map50_95",
            "delta_vs_03_dqa_aggregate_map50_95",
            "worst_split",
            "worst_split_map50_95",
            "day_avg_map50_95",
            "night_avg_map50_95",
            "day_night_gap_map50_95",
        ]
        for row in rows:
            for field in fields:
                row.setdefault(field, "")
        write_csv(args.workspace_root / "stats" / "05_expert_choice_final_metrics.csv", rows, fields)
    write_report(args, rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.notify or args.notify_start:
        notify(args, "Scene-Daynight DQA 05 Expert-Choice pseudoGT router started.", title="DQA 05 start", status="started")
    status = "success"
    error: str | None = None
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = str(exc)
        raise
    finally:
        if args.notify or args.notify_end:
            notify(
                args,
                f"Scene-Daynight DQA 05 Expert-Choice pseudoGT router finished with status={status}.",
                title="DQA 05 finish",
                status=status,
                error=error,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
