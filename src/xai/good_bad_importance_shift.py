"""
Does the coin's state of preservation change WHERE the CNN looks?

Data: dataset/xai_data/<side>/<class>/{Good,Bad}/<stem>.png, per class one well preserved (Good) and one worn
(Bad) coin of the same type (all of them are test-set images). Every image is explained by every XAI method
for its TRUE class with the production model of its side.

Good and Bad are two different physical coins, so their pixels are not aligned (other rotation, crop, die)
and a pixel-wise IoU between the two maps would mostly measure misalignment. Coins are centered and fill the
crop, therefore the image is divided into rotation-invariant radial zones (r = distance from the image center
divided by half the image size, after the 224 x 224 model resize):

    center      r <  1/3        main motif (head, bust, figure)
    middle      1/3 <= r < 2/3  outer parts of the motif, attributes, inner legend
    rim         2/3 <= r <= 1   legend and border of dots - first to wear off
    background  r > 1           corners outside the inscribed circle

Per image and method (L+ = positive part of the map = evidence FOR the true class):
    <zone>         % of the L+ energy that lies in the zone (zones add up to 100 %)
    conc50         % of the image area that holds 50 % of the L+ energy (smaller = more focused)
    mean_radius    energy weighted mean r, in % of the coin radius (smaller = closer to the center)
    confidence     model probability of the true class, %
Per Good/Bad pair:
    delta          Bad - Good, in percentage points
    shift          0.5 * sum over zones |Bad - Good|  = % of the importance that moved to another zone
                   (total variation distance, 0 = identical zone distribution, 100 = no overlap at all)
    shift_baseline Good vs ordinary coins: the same shift between the Good coin and N other test images of the
                   class (--baseline-images, mean over them) = how much the zones move between ANY two coins of
                   one type. Wear only matters if shift (Good vs Bad) is clearly larger than shift_baseline.
Significance: paired two-sided Wilcoxon signed-rank test over the classes (Good vs Bad of the same class,
              and shift vs shift_baseline of the same class).

Outputs (default runs/xai_quality_shift/):
    per_image.csv            every (model, side, class, quality, method) with all metrics
    pairs.csv                every (model, side, class, method): Good, Bad and delta of every metric + shift
    summary.csv              per scope (Observe / Reverse / All) and method: mean Good, Bad, delta, p-value
    figures/<side>_<class>.png   Good row and Bad row, every method with the zone circles

Run from anywhere:
    python auto_commands/xai_quality_shift.py
    python auto_commands/xai_quality_shift.py --help
"""
import argparse
import os
import random
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
from scipy import stats

from src.constants import EXPLAINERS, IMAGE_EXTENSIONS, PRODUCTION_RUNS_DIR, SIDES
from src.dataset.custom_dataset import IMAGE_SIZE, get_base_transform
from src.engine.inference import CoinInference
from src.xai.comparison import find_production_run, save_table
from src.xai.core.base import denormalize_gray

QUALITIES = ("Good", "Bad")
ZONES = ("center", "middle", "rim", "background")
METRICS = (*ZONES, "conc50", "mean_radius", "confidence")
ALL = "All"


# ----------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------
def load_items(xai_root):
    """[{"side", "class_number", "quality", "image_path"}, ...] for every class that has both Good and Bad"""
    items = []
    for side in SIDES:
        side_root = Path(xai_root) / side
        if not side_root.is_dir():
            continue
        for class_dir in sorted(side_root.iterdir(), key=lambda d: int(d.name)):
            found = {}
            for quality in QUALITIES:
                files = sorted(f for f in (class_dir / quality).glob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)
                if files:
                    found[quality] = files[0]
                if len(files) > 1:
                    print(f"Warning: {class_dir / quality} has {len(files)} images, only {files[0].name} is used")
            if len(found) < 2:
                print(f"Warning: {class_dir} skipped, Good and Bad image are both needed")
                continue
            for quality, path in found.items():
                items.append({"side": side, "class_number": int(class_dir.name), "quality": quality,
                              "image_path": path})
    return items


# ----------------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------------
def radius_map(size=IMAGE_SIZE):
    """distance of every pixel from the image center, 1 = half the image size (edge of the inscribed circle)"""
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    return np.sqrt((yy - c) ** 2 + (xx - c) ** 2) / (size / 2)


def zone_masks(radius, bounds):
    inner, outer = bounds
    return {
        "center": radius < inner,
        "middle": (radius >= inner) & (radius < outer),
        "rim": (radius >= outer) & (radius <= 1.0),
        "background": radius > 1.0,
    }


def importance_metrics(explanation, radius, masks):
    positive = np.clip(np.asarray(explanation, dtype=np.float64), 0, None)
    total = positive.sum()
    if total <= 0:
        return {m: np.nan for m in (*ZONES, "conc50", "mean_radius")}

    metrics = {zone: 100.0 * positive[mask].sum() / total for zone, mask in masks.items()}
    # smallest share of pixels that holds half of the energy
    cumulative = np.cumsum(np.sort(positive.ravel())[::-1])
    metrics["conc50"] = 100.0 * (np.searchsorted(cumulative, 0.5 * total) + 1) / positive.size
    metrics["mean_radius"] = 100.0 * float((positive * radius).sum() / total)
    return metrics


def zone_shift(good, bad):
    """total variation distance between the zone distributions, in %"""
    return 0.5 * sum(abs(bad[z] - good[z]) for z in ZONES)


def wilcoxon_p(good, bad):
    diff = np.asarray(bad, dtype=float) - np.asarray(good, dtype=float)
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2 or not np.any(diff != 0):
        return np.nan
    return float(stats.wilcoxon(diff).pvalue)


# ----------------------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------------------
def save_figure(results, method_names, bounds, title, save_path):
    """results: {quality: (image_tensor, {method: (explanation, metrics)}, confidence, predicted)}"""
    columns = 1 + len(method_names)
    fig, axes = plt.subplots(len(QUALITIES), columns, figsize=(3.4 * columns, 4.4 * len(QUALITIES)), squeeze=False)
    c, half = (IMAGE_SIZE - 1) / 2, IMAGE_SIZE / 2

    for row, quality in zip(axes, QUALITIES):
        image_tensor, maps, confidence, predicted = results[quality]
        image = denormalize_gray(image_tensor)
        row[0].imshow(image, cmap="gray")
        row[0].set_title(f"{quality} coin\np(true) {confidence:.1f}% | pred {predicted}", fontsize=9)
        for ax, name in zip(row[1:], method_names):
            explanation, metrics = maps[name]
            positive = np.clip(explanation, 0, None)
            ax.imshow(image, cmap="gray")
            ax.imshow(positive / (positive.max() or 1), cmap="jet", vmin=0, vmax=1, alpha=0.45)
            ax.set_title(f"{name}\nC {metrics['center']:.0f}% | M {metrics['middle']:.0f}% | "
                         f"R {metrics['rim']:.0f}% | B {metrics['background']:.0f}%", fontsize=8)
        for ax in row:
            for r in (*bounds, 1.0):
                ax.add_patch(plt.Circle((c, c), r * half, fill=False, color="white", lw=0.8, ls="--"))
            ax.axis("off")

    fig.suptitle(f"{title}\nshare of positive importance per zone: C center, M middle, R rim, B background",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=90)
    plt.close(fig)


def evaluate_side(run_dir, items, methods, data_root, catalogue_path, bounds, figures_dir, baseline_images, rng):
    run_dir = Path(run_dir)
    model_name, side, _, _ = run_dir.name.rsplit("_", 3)
    side_items = [item for item in items if item["side"] == side]
    if not side_items:
        return []
    print(f"=== {run_dir.name}: {len(side_items) // 2} Good/Bad pairs ===", flush=True)

    inference = CoinInference(
        observe_run=run_dir if side == "Observe" else None,
        reverse_run=run_dir if side == "Reverse" else None,
        catalogue_path=catalogue_path,
        explainers=methods,
        data_root=data_root,
    )
    explainers = list(inference.explainers[side].values())
    transform = get_base_transform()
    radius = radius_map()
    masks = zone_masks(radius, bounds)

    rows = []
    for class_number in sorted({item["class_number"] for item in side_items}):
        class_items = [i for i in side_items if i["class_number"] == class_number]
        # ordinary coins of the same class = reference for the normal coin-to-coin variation
        used = {i["image_path"].name for i in class_items}
        class_dir = Path(data_root) / "test" / side / str(class_number)
        others = sorted(f for f in class_dir.glob("*") if f.suffix.lower() in IMAGE_EXTENSIONS and f.name not in used)
        others = rng.sample(others, min(baseline_images, len(others)))
        class_items += [{"quality": "Other", "image_path": f} for f in others]

        results = {}
        for item in class_items:
            image_tensor = transform(Image.open(item["image_path"]).convert("RGB"))
            true_idx = inference.class_numbers.index(class_number)
            probs = inference._predict_probs(side, image_tensor)
            pred_idx = int(probs.argmax())
            confidence = 100.0 * float(probs[true_idx])

            maps = {}
            for explainer in explainers:
                explanation = explainer.explain(image_tensor, true_idx)
                metrics = importance_metrics(explanation, radius, masks)
                maps[explainer.name] = (explanation, metrics)
                rows.append({"model": model_name, "side": side, "class_number": class_number,
                             "quality": item["quality"], "image": item["image_path"].name,
                             "predicted": inference.class_numbers[pred_idx], "correct": int(pred_idx == true_idx),
                             "method": explainer.name, **metrics, "confidence": confidence})
            if item["quality"] in QUALITIES:
                results[item["quality"]] = (image_tensor, maps, confidence, inference.class_numbers[pred_idx])

        if figures_dir is not None:
            save_figure(results, [e.name for e in explainers], bounds,
                        f"{run_dir.name} | {side} class {class_number}", figures_dir / f"{side}_{class_number}.png")
        print(f"  {side}/{class_number} done", flush=True)
    return rows


def make_pairs(rows):
    by_key = {}
    for r in rows:
        group = by_key.setdefault((r["model"], r["side"], r["class_number"], r["method"]), {"Other": []})
        if r["quality"] == "Other":
            group["Other"].append(r)
        else:
            group[r["quality"]] = r

    pairs = []
    for (model, side, class_number, method), q in by_key.items():
        good, bad = q["Good"], q["Bad"]
        pair = {"model": model, "side": side, "class_number": class_number, "method": method,
                "good_image": good["image"], "bad_image": bad["image"],
                "good_correct": good["correct"], "bad_correct": bad["correct"]}
        for m in METRICS:
            pair[f"{m}_good"], pair[f"{m}_bad"], pair[f"{m}_delta"] = good[m], bad[m], bad[m] - good[m]
        pair["shift"] = zone_shift(good, bad)
        others = q["Other"]
        pair["baseline_images"] = len(others)
        pair["shift_baseline"] = float(np.mean([zone_shift(good, o) for o in others])) if others else np.nan
        pair["confidence_other"] = float(np.mean([o["confidence"] for o in others])) if others else np.nan
        pairs.append(pair)
    return pairs


def summarize(pairs):
    """per scope (each side + All) and method: mean Good / Bad / delta and Wilcoxon p of every metric"""
    table = []
    methods = list(dict.fromkeys(p["method"] for p in pairs))
    sides = [s for s in SIDES if any(p["side"] == s for p in pairs)]
    for scope in (*sides, ALL):
        for method in methods:
            group = [p for p in pairs if p["method"] == method and scope in (ALL, p["side"])]
            row = {"scope": scope, "method": method, "pairs": len(group),
                   "good_accuracy": 100.0 * np.mean([p["good_correct"] for p in group]),
                   "bad_accuracy": 100.0 * np.mean([p["bad_correct"] for p in group]),
                   "shift": float(np.nanmean([p["shift"] for p in group])),
                   "shift_median": float(np.nanmedian([p["shift"] for p in group])),
                   "shift_baseline": float(np.nanmean([p["shift_baseline"] for p in group])),
                   "shift_vs_baseline_p": wilcoxon_p([p["shift_baseline"] for p in group],
                                                     [p["shift"] for p in group]),
                   "confidence_other": float(np.nanmean([p["confidence_other"] for p in group]))}
            for m in METRICS:
                good = [p[f"{m}_good"] for p in group]
                bad = [p[f"{m}_bad"] for p in group]
                row[f"{m}_good"] = float(np.nanmean(good))
                row[f"{m}_bad"] = float(np.nanmean(bad))
                row[f"{m}_delta"] = row[f"{m}_bad"] - row[f"{m}_good"]
                row[f"{m}_p"] = wilcoxon_p(good, bad)
            table.append(row)
    return table


def print_summary(table):
    labels = {"center": "Center", "middle": "Middle", "rim": "Rim", "background": "Backgr.",
              "conc50": "Area50", "mean_radius": "Radius", "confidence": "p(true)"}

    def cell(row, m):
        star = "*" if row[f"{m}_p"] < 0.05 else " "
        return f"{row[f'{m}_good']:5.1f}>{row[f'{m}_bad']:5.1f} ({row[f'{m}_delta']:+5.1f}){star}"

    for scope in dict.fromkeys(r["scope"] for r in table):
        rows = [r for r in table if r["scope"] == scope]
        print(f"\n{scope.upper()} ({rows[0]['pairs']} Good/Bad pairs, accuracy Good {rows[0]['good_accuracy']:.0f}% "
              f"-> Bad {rows[0]['bad_accuracy']:.0f}%)   values in %, Good>Bad (delta pp), * p < 0.05")
        print(f"{'method':<11} {'shift':>6} " + " ".join(f"{labels[m]:>20}" for m in METRICS))
        for r in rows:
            print(f"{r['method']:<11} {r['shift']:6.1f} " + " ".join(f"{cell(r, m):>20}" for m in METRICS))
        if not np.isnan(rows[0]["shift_baseline"]):
            print(f"  baseline (Good vs ordinary coins of the class, p(true) of those coins "
                  f"{rows[0]['confidence_other']:.1f}%):")
            for r in rows:
                star = "*" if r["shift_vs_baseline_p"] < 0.05 else " "
                print(f"  {r['method']:<11} shift Good-Bad {r['shift']:5.1f}  vs  Good-Ordinary {r['shift_baseline']:5.1f}"
                      f"   p = {r['shift_vs_baseline_p']:.3g}{star}")


def pars_args(argv=None):
    parser = argparse.ArgumentParser(description="Shift of XAI importance regions between Good and Bad coins")
    parser.add_argument('--xai-data', type=str, default="dataset/xai_data",
                        help="Folder <side>/<class>/{Good,Bad}/<image>")
    parser.add_argument('--runs-dir', type=str, default=PRODUCTION_RUNS_DIR,
                        help="Folder with one trained run per side (production models by default)")
    parser.add_argument('--data-root', type=str, default="dataset/data")
    parser.add_argument('--catalogue', type=str, default="dataset/data/catalogue.json")
    parser.add_argument('--methods', type=str, nargs='+', default=list(EXPLAINERS), choices=EXPLAINERS.keys())
    parser.add_argument('--zone-bounds', type=float, nargs=2, default=[1 / 3, 2 / 3],
                        help="Radii (fraction of the coin radius) of center|middle and middle|rim borders")
    parser.add_argument('--baseline-images', type=int, default=3,
                        help="Other test images per class used as the normal coin-to-coin variation, 0 = off")
    parser.add_argument('--seed', type=int, default=42, help="Random choice of the baseline images")
    parser.add_argument('--no-figures', action='store_true')
    parser.add_argument('--output-dir', type=str, default="runs/xai_quality_shift")
    return parser.parse_args(argv)


def main(argv=None):
    args = pars_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(args.xai_data)
    if not items:
        raise SystemExit(f"No Good/Bad pairs found in {args.xai_data}")
    print(f"{len(items) // 2} Good/Bad pairs in {args.xai_data}")

    rows = []
    for side in SIDES:
        if any(item["side"] == side for item in items):
            rows += evaluate_side(find_production_run(side, args.runs_dir), items, args.methods, args.data_root,
                                  args.catalogue, tuple(args.zone_bounds),
                                  None if args.no_figures else out_dir / "figures",
                                  args.baseline_images, random.Random(args.seed))

    pairs = make_pairs(rows)
    table = summarize(pairs)
    save_table(rows, out_dir / "per_image.csv")
    save_table(pairs, out_dir / "pairs.csv")
    save_table(table, out_dir / "summary.csv")

    print_summary(table)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
