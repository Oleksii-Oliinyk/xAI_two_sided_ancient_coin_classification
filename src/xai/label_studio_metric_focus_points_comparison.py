"""
Do the CNNs look at the same parts of the coin as a human expert?

Ground truth: Label Studio export (dataset/data/test_set_focus_area_annotations.json). Every task is one
test image with
    side          choices  "Observe" / "Reverse"
    class_number  number   class folder of the image
    focus_area    brush    RLE mask of the semantically important region(s) drawn by the expert
and the image file name "<hash>-<stem>.png" -> dataset/data/test/<side>/<class_number>/<stem>.png

For every trained run (model x side) every annotated image of that side is explained by every XAI method
(for the true class by default) and the explanation map L is compared with the expert mask G
(both 224 x 224, the model input size):

    IoU            |S n G| / |S u G|
    Dice (DSC)     2 |S n G| / (|S| + |G|)
                   S = top-k pixels of the (positive, smoothed) map, k = |G|  -> S and G have the same area,
                   so a method is not rewarded / punished for how spread its map is
    Pointing Game  hit if argmax L lies inside G;  PG = #hits / (#hits + #misses)
    Energy-Based Pointing Game (Wang et al., Score-CAM, 2020, with the bbox replaced by the expert mask)
                   EBPG = sum_{(i,j) in G} L+(i,j) / sum_{(i,j)} L+(i,j),   L+ = max(L, 0)
                   share of the explanation's positive "energy" that falls into the expert region

Baselines (needed to decide whether a score is "high"):
    Random         a map without any information. Its expected scores are analytic, p = |G| / (H * W):
                   IoU = p / (2 - p), Dice = p, PG = p, EBPG = p
    Center prior   a centered Gaussian blob (coins are centered, so "look at the middle" is a strong guess);
                   evaluated exactly like an XAI method

Outputs (default runs/xai_focus_comparison/):
    gt_masks/<side>_<class>_<stem>.npy|.png  expert masks as 1 (important) / 0 (not important), original size
    per_image.csv                            every (model, side, image, method) with all metrics
    summary_by_run.csv                       mean per model, side, method
    summary_by_method.csv                    mean per method over all runs and images + lift over random
                                             + Wilcoxon p-values vs Random and vs Center prior
    figures/<run>/<side>_<class>_<stem>.png  input + expert mask + every method (heatmap, top-k contour)

Run from anywhere:
    python auto_commands/xai_focus_comparison.py            (all production models, all methods)
    python auto_commands/xai_focus_comparison.py --help
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
# LIME prints a progress bar for every image -> hide it (tqdm reads this env var)
os.environ.setdefault("TQDM_DISABLE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage, stats

from src.constants import EXPLAINERS, PRODUCTION_RUNS_DIR
from src.dataset.custom_dataset import IMAGE_SIZE, get_base_transform
from src.engine.inference import CoinInference
from src.xai.core.base import denormalize_gray
from src.xai.comparison import find_runs, save_table
from src.xai.core.localization import attention_mask, mask_iou

RANDOM = "Random"
CENTER = "Center prior"
METRICS = ("iou", "dice", "pointing_game", "ebpg")


# ----------------------------------------------------------------------------------------------------------
# Label Studio RLE -> 1/0 mask
# ----------------------------------------------------------------------------------------------------------
def decode_rle(rle):
    """
    Label Studio brush RLE (same algorithm as label_studio_converter.brush.decode_rle) -> flat uint8 array
    of length H * W * 4 (RGBA). Bit stream: 32 bits length, 5 bits word size - 1, 4 x 4 bits run-length
    field sizes - 1, then blocks of [1 bit repeat flag, 2 bits field size index, run length - 1, value(s)].
    """
    bits = np.unpackbits(np.asarray(rle, dtype=np.uint8))
    pos = 0

    def read(n):
        nonlocal pos
        value = 0
        for bit in bits[pos:pos + n]:
            value = (value << 1) | int(bit)
        pos += n
        return value

    total = read(32)
    word_size = read(5) + 1
    run_sizes = [read(4) + 1 for _ in range(4)]

    out = np.zeros(total, dtype=np.uint8)
    i = 0
    while i < total:
        repeat = read(1)
        j = i + 1 + read(run_sizes[read(2)])
        if repeat:
            out[i:j] = read(word_size)
            i = j
        else:
            while i < j:
                out[i] = read(word_size)
                i += 1
    return out


def brush_to_mask(result, alpha_threshold=128):
    """One 'brushlabels' result -> [H, W] uint8 mask, 1 = important pixel, 0 = not important.
    The brush is stored in the alpha channel with anti-aliased edges, so it is thresholded at 50 %."""
    h, w = result["original_height"], result["original_width"]
    alpha = decode_rle(result["value"]["rle"]).reshape(h, w, 4)[:, :, 3]
    return (alpha >= alpha_threshold).astype(np.uint8)


def load_annotations(annotation_path, data_root):
    """
    Returns [{"task_id", "side", "class_number", "image_path", "mask" [H, W] 1/0}, ...]
    Several brush strokes of one task are merged into one mask.
    """
    with open(annotation_path, encoding="utf-8") as f:
        tasks = json.load(f)

    items = []
    for task in tasks:
        if not task["annotations"]:
            continue
        results = task["annotations"][0]["result"]
        side = next(r["value"]["choices"][0] for r in results if r["type"] == "choices")
        class_number = int(next(r["value"]["number"] for r in results if r["type"] == "number"))
        brushes = [r for r in results if r["type"] == "brushlabels"]

        # Label Studio prefixes uploads with "<hash>-", the rest is the original file name
        file_name = Path(task["data"]["image"]).name.split("-", 1)[1]
        image_path = Path(data_root) / "test" / side / str(class_number) / file_name
        if not image_path.is_file():
            raise FileNotFoundError(f"Task {task['id']}: {image_path} not found")

        mask = np.zeros_like(brush_to_mask(brushes[0]))
        for brush in brushes:
            mask |= brush_to_mask(brush)

        size = Image.open(image_path).size
        if size != mask.shape[::-1]:
            raise ValueError(f"Task {task['id']}: mask {mask.shape[::-1]} != image {size}")

        items.append({"task_id": task["id"], "side": side, "class_number": class_number,
                      "image_path": image_path, "mask": mask})
    return items


def save_gt_masks(items, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        name = f"{item['side']}_{item['class_number']}_{item['image_path'].stem}"
        np.save(out_dir / f"{name}.npy", item["mask"])
        Image.fromarray(item["mask"] * 255).save(out_dir / f"{name}.png")


def resize_mask(mask, size=IMAGE_SIZE):
    """Same geometry as get_base_transform (plain resize to size x size), nearest so the mask stays 1/0"""
    return np.asarray(Image.fromarray(mask * 255).resize((size, size), Image.NEAREST)) > 127


# ----------------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------------
def dice(a, b):
    total = a.sum() + b.sum()
    if total == 0:
        return np.nan
    return float(2 * np.logical_and(a, b).sum() / total)


def pointing_game_hit(explanation, gt, smooth_sigma):
    """1 if the most important pixel lies inside the expert mask. The map is smoothed first like in
    attention_mask, so a single noisy SHAP pixel or a tie between LIME superpixel pixels does not decide."""
    positive = np.clip(np.asarray(explanation, dtype=np.float32), 0, None)
    if smooth_sigma > 0:
        positive = ndimage.gaussian_filter(positive, sigma=smooth_sigma)
    if positive.max() <= 0:
        return 0
    y, x = np.unravel_index(np.argmax(positive), positive.shape)
    return int(gt[y, x])


def energy_pointing_game(explanation, gt):
    """EBPG = sum of L+ inside the expert mask / sum of L+ over the whole image, L+ = max(L, 0)"""
    positive = np.clip(np.asarray(explanation, dtype=np.float64), 0, None)
    total = positive.sum()
    if total <= 0:
        return 0.0
    return float(positive[gt].sum() / total)


def compute_metrics(explanation, gt, smooth_sigma):
    top_percent = 100.0 * gt.mean()
    saliency = attention_mask(explanation, top_percent=top_percent, smooth_sigma=smooth_sigma)
    return {
        "iou": mask_iou(saliency, gt),
        "dice": dice(saliency, gt),
        "pointing_game": pointing_game_hit(explanation, gt, smooth_sigma),
        "ebpg": energy_pointing_game(explanation, gt),
    }, saliency


def random_baseline(gt):
    """Expected scores of a map without information (uniform / random top-k with |S| = |G|)"""
    p = float(gt.mean())
    return {"iou": p / (2 - p), "dice": p, "pointing_game": p, "ebpg": p}


def center_prior(size=IMAGE_SIZE, sigma_fraction=0.25):
    yy, xx = np.mgrid[:size, :size]
    c, s = (size - 1) / 2, sigma_fraction * size
    return np.exp(-((yy - c) ** 2 + (xx - c) ** 2) / (2 * s ** 2))


# ----------------------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------------------
def save_figure(image_tensor, gt, maps, title, save_path):
    """maps: {method: (explanation, signed, saliency mask)}"""
    image = denormalize_gray(image_tensor)
    columns = 1 + len(maps)
    fig, axes = plt.subplots(1, columns, figsize=(3.6 * columns, 4.2), squeeze=False)
    axes = axes[0]

    axes[0].imshow(image, cmap="gray")
    axes[0].imshow(np.ma.masked_where(~gt, gt), cmap="autumn", alpha=0.5)
    axes[0].set_title("Expert focus area")

    for ax, (name, (explanation, signed, saliency, metrics)) in zip(axes[1:], maps.items()):
        ax.imshow(image, cmap="gray")
        positive = np.clip(explanation, 0, None)
        ax.imshow(positive / (positive.max() or 1), cmap="jet", vmin=0, vmax=1, alpha=0.45)
        ax.contour(gt, levels=[0.5], colors="white", linewidths=1.2)
        ax.contour(saliency, levels=[0.5], colors="lime", linewidths=1.0, linestyles="--")
        ax.set_title(f"{name}\nIoU {metrics['iou']:.2f} | PG {metrics['pointing_game']} | "
                     f"EBPG {metrics['ebpg']:.2f}", fontsize=9)

    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{title}\nwhite = expert, green dashed = top-|G| pixels of the method", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=90)
    plt.close(fig)


def evaluate_run(run_dir, items, methods, data_root, catalogue_path, target, smooth_sigma, figures_dir):
    run_dir = Path(run_dir)
    model_name, side, _, _ = run_dir.name.rsplit("_", 3)
    side_items = [item for item in items if item["side"] == side]
    if not side_items:
        return []

    inference = CoinInference(
        observe_run=run_dir if side == "Observe" else None,
        reverse_run=run_dir if side == "Reverse" else None,
        catalogue_path=catalogue_path,
        explainers=methods,
        data_root=data_root,
    )
    explainers = list(inference.explainers[side].values())
    transform = get_base_transform()
    prior = center_prior()

    rows = []
    for item in side_items:
        image_tensor = transform(Image.open(item["image_path"]).convert("RGB"))
        gt = resize_mask(item["mask"])
        true_idx = inference.class_numbers.index(item["class_number"])
        pred_idx = int(inference._predict_probs(side, image_tensor).argmax())
        class_idx = true_idx if target == "true" else pred_idx

        base = {"model": model_name, "side": side, "class_number": item["class_number"],
                "image": item["image_path"].name, "predicted": inference.class_numbers[pred_idx],
                "correct": int(pred_idx == true_idx), "gt_area": float(gt.mean())}

        maps = {}
        for explainer in explainers:
            explanation = explainer.explain(image_tensor, class_idx)
            metrics, saliency = compute_metrics(explanation, gt, smooth_sigma)
            maps[explainer.name] = (explanation, explainer.signed, saliency, metrics)
            rows.append({**base, "method": explainer.name, **metrics})

        metrics, saliency = compute_metrics(prior, gt, smooth_sigma)
        maps[CENTER] = (prior, False, saliency, metrics)
        rows.append({**base, "method": CENTER, **metrics})
        rows.append({**base, "method": RANDOM, **random_baseline(gt)})

        if figures_dir is not None:
            title = (f"{run_dir.name} | {side} class {item['class_number']} / {item['image_path'].name} | "
                     f"predicted {inference.class_numbers[pred_idx]}")
            save_figure(image_tensor, gt, maps, title,
                        figures_dir / run_dir.name / f"{side}_{item['class_number']}_{item['image_path'].stem}.png")
        print(f"  {side}/{item['class_number']}/{item['image_path'].name} done", flush=True)
    return rows


def _mean(rows, metric):
    return float(np.nanmean([r[metric] for r in rows]))


def summarize(rows, keys):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    return [{**dict(zip(keys, key)), "images": len(group), **{m: _mean(group, m) for m in METRICS}}
            for key, group in groups.items()]


def summarize_methods(rows):
    """Mean per method over all runs and images, lift = method / random and paired Wilcoxon p-values
    (one-sided: is the method better than Random / than the Center prior on the same images?)"""
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    key = lambda r: (r["model"], r["side"], r["image"], r["class_number"])
    random_rows = {key(r): r for r in by_method[RANDOM]}
    center_rows = {key(r): r for r in by_method[CENTER]}

    table = []
    for method, group in by_method.items():
        row = {"method": method, "images": len(group), **{m: _mean(group, m) for m in METRICS}}
        for m in METRICS:
            row[f"{m}_lift"] = row[m] / _mean([random_rows[key(r)] for r in group], m)
        for name, reference in (("random", random_rows), ("center", center_rows)):
            for m in ("iou", "ebpg"):
                a = np.array([r[m] for r in group])
                b = np.array([reference[key(r)][m] for r in group])
                diff = a - b
                row[f"p_{m}_vs_{name}"] = (float(stats.wilcoxon(diff, alternative="greater").pvalue)
                                           if np.any(diff != 0) else np.nan)
        table.append(row)
    return table


def pars_args(argv=None):
    parser = argparse.ArgumentParser(description="XAI focus vs expert (Label Studio) focus areas")
    parser.add_argument('--annotations', type=str, default="dataset/data/test_set_focus_area_annotations.json")
    parser.add_argument('--runs', type=str, nargs='*', default=None,
                        help="Run folders to evaluate, default = every run in --runs-dir (production models)")
    parser.add_argument('--runs-dir', type=str, default=PRODUCTION_RUNS_DIR,
                        help="Folder with trained runs, e.g. runs/experiments for the non-production models")
    parser.add_argument('--data-root', type=str, default="dataset/data")
    parser.add_argument('--catalogue', type=str, default="dataset/data/catalogue.json")
    parser.add_argument('--methods', type=str, nargs='+', default=list(EXPLAINERS), choices=EXPLAINERS.keys())
    parser.add_argument('--target', type=str, default="true", choices=["true", "predicted"],
                        help="Explain the true class (expert mask belongs to it) or the predicted class")
    parser.add_argument('--smooth-sigma', type=float, default=4.0,
                        help="Gaussian blur (px) of maps before top-k / argmax, 0 = off")
    parser.add_argument('--no-figures', action='store_true')
    parser.add_argument('--output-dir', type=str, default="runs/xai_focus_comparison")
    return parser.parse_args(argv)


def main(argv=None):
    args = pars_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_annotations(args.annotations, args.data_root)
    save_gt_masks(items, out_dir / "gt_masks")
    print(f"{len(items)} expert masks saved to {out_dir / 'gt_masks'}")

    runs = [Path(r) for r in args.runs] if args.runs else find_runs(args.runs_dir)
    if not runs:
        raise SystemExit(f"No trained runs found in {args.runs_dir}")

    rows = []
    for run_dir in runs:
        print(f"=== {Path(run_dir).name} ===", flush=True)
        rows += evaluate_run(run_dir, items, args.methods, args.data_root, args.catalogue, args.target,
                             args.smooth_sigma, None if args.no_figures else out_dir / "figures")

    by_run = summarize(rows, ("model", "side", "method"))
    by_method = summarize_methods(rows)
    save_table(rows, out_dir / "per_image.csv")
    save_table(by_run, out_dir / "summary_by_run.csv")
    save_table(by_method, out_dir / "summary_by_method.csv")

    correct = [r["correct"] for r in rows if r["method"] == RANDOM]
    print(f"\nAccuracy on the annotated images: {np.mean(correct):.2%} ({sum(correct)}/{len(correct)})")
    print(f"\n{'model':<16} {'side':<8} {'method':<13} {'IoU':>6} {'Dice':>6} {'PG':>6} {'EBPG':>6}")
    for r in by_run:
        print(f"{r['model']:<16} {r['side']:<8} {r['method']:<13} "
              f"{r['iou']:>6.3f} {r['dice']:>6.3f} {r['pointing_game']:>6.3f} {r['ebpg']:>6.3f}")
    print(f"\nGENERAL (mean over all runs and images)\n"
          f"{'method':<13} {'n':>4} {'IoU':>6} {'Dice':>6} {'PG':>6} {'EBPG':>6} {'EBPG lift':>9} "
          f"{'p(IoU>rand)':>11} {'p(EBPG>rand)':>12} {'p(IoU>ctr)':>10} {'p(EBPG>ctr)':>11}")
    for r in by_method:
        print(f"{r['method']:<13} {r['images']:>4} {r['iou']:>6.3f} {r['dice']:>6.3f} {r['pointing_game']:>6.3f} "
              f"{r['ebpg']:>6.3f} {r['ebpg_lift']:>9.2f} {r['p_iou_vs_random']:>11.2g} "
              f"{r['p_ebpg_vs_random']:>12.2g} {r['p_iou_vs_center']:>10.2g} {r['p_ebpg_vs_center']:>11.2g}")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
