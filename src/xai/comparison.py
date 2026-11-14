"""
IoU between XAI methods: do they highlight the same region of the coin?

For every trained run a few random test images are explained with every method (for the predicted class).
Per image and method the focus region = top-N% pixels that support the class. For every pair of methods
IoU of these regions is computed (mask IoU and IoU of the bbox around the largest region).
"""
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from src.constants import MODEL_CONFIGS, PRODUCTION_RUNS_DIR, SIDES
from src.dataset.custom_dataset import CoinDataset
from src.engine.inference import CoinInference
from src.xai.core.localization import attention_mask, bbox_iou, main_region_bbox, mask_iou


def find_runs(runs_dir=PRODUCTION_RUNS_DIR):
    runs = []
    for run_dir in sorted(Path(runs_dir).iterdir()):
        parts = run_dir.name.rsplit("_", 3)
        if (run_dir / "best_model.pth").is_file() and len(parts) == 4 \
                and parts[0] in MODEL_CONFIGS and parts[1] in SIDES:
            runs.append(run_dir)
    return runs


def find_production_run(side, runs_dir=PRODUCTION_RUNS_DIR):
    """The single production run trained on `side` (newest one if several are there)"""
    runs = [r for r in find_runs(runs_dir) if r.name.rsplit("_", 3)[1] == side]
    if not runs:
        raise FileNotFoundError(f"No {side} model in {runs_dir}")
    return sorted(runs, key=lambda r: r.name.rsplit("_", 2)[1:])[-1]


def evaluate_run(run_dir, rng, methods, data_root, catalogue_path, images_per_run=3,
                 top_percent=20.0, smooth_sigma=4.0):
    """Returns model_name, side, {pair: {"mask_iou": [...], "bbox_iou": [...]}} over the sampled images"""
    run_dir = Path(run_dir)
    model_name, side, _, _ = run_dir.name.rsplit("_", 3)
    inference = CoinInference(
        observe_run=run_dir if side == "Observe" else None,
        reverse_run=run_dir if side == "Reverse" else None,
        catalogue_path=catalogue_path,
        explainers=methods,
        data_root=data_root,
    )
    explainers = list(inference.explainers[side].values())
    pairs = list(combinations([e.name for e in explainers], 2))

    dataset = CoinDataset(Path(data_root) / "test", side)
    samples = rng.sample(dataset.samples, min(images_per_run, len(dataset.samples)))

    ious = {f"{a} vs {b}": {"mask_iou": [], "bbox_iou": []} for a, b in pairs}
    for path, _ in samples:
        image_tensor = dataset.base_transform(Image.open(path).convert("RGB"))
        pred_idx = int(inference._predict_probs(side, image_tensor).argmax())

        masks = {e.name: attention_mask(e.explain(image_tensor, pred_idx), top_percent, smooth_sigma)
                 for e in explainers}
        bboxes = {name: main_region_bbox(mask) for name, mask in masks.items()}
        for a, b in pairs:
            ious[f"{a} vs {b}"]["mask_iou"].append(mask_iou(masks[a], masks[b]))
            ious[f"{a} vs {b}"]["bbox_iou"].append(bbox_iou(bboxes[a], bboxes[b]))
        print(f"  {path.parent.name}/{path.name} done", flush=True)
    return model_name, side, ious


def compare_runs(runs, rng, **evaluate_kwargs):
    """
    Returns (rows, general):
        rows     per run and pair of methods: mean IoU over the run's images
        general  per pair of methods: mean IoU over all runs and images
    """
    rows, pooled = [], {}
    for run_dir in runs:
        print(f"=== {Path(run_dir).name} ===", flush=True)
        model_name, side, ious = evaluate_run(run_dir, rng, **evaluate_kwargs)
        for pair, values in ious.items():
            rows.append({"model": model_name, "side": side, "pair": pair, "images": len(values["mask_iou"]),
                         "mask_iou": float(np.nanmean(values["mask_iou"])),
                         "bbox_iou": float(np.nanmean(values["bbox_iou"]))})
            for metric in values:
                pooled.setdefault(pair, {}).setdefault(metric, []).extend(values[metric])

    general = [{"pair": pair, "images": len(v["mask_iou"]),
                "mask_iou": float(np.nanmean(v["mask_iou"])), "bbox_iou": float(np.nanmean(v["bbox_iou"]))}
               for pair, v in pooled.items()]
    return rows, general


def save_table(table, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in table)
