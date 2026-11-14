"""
IoU table between XAI methods on random test images (see xai/comparison.py).

Outputs (default runs/xai_comparison/):
    iou_table.csv    per run and pair of methods: mean IoU over the run's images
    iou_general.csv  per pair of methods: mean IoU over all runs and images
"""
import argparse
import os
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
# LIME prints a progress bar for every image -> hide it (tqdm reads this env var)
os.environ.setdefault("TQDM_DISABLE", "1")

from src.constants import EXPLAINERS, PRODUCTION_RUNS_DIR
from src.xai.comparison import compare_runs, find_runs, save_table


def pars_args():
    parser = argparse.ArgumentParser(description="IoU table between XAI methods on random test images")
    parser.add_argument('--runs', type=str, nargs='*', default=None,
                        help="Run folders to evaluate, default = every run in --runs-dir (production models)")
    parser.add_argument('--runs-dir', type=str, default=PRODUCTION_RUNS_DIR,
                        help="Folder with trained runs, e.g. runs/experiments for the non-production models")
    parser.add_argument('--data-root', type=str, default="dataset/data")
    parser.add_argument('--catalogue', type=str, default="dataset/data/catalogue.json")
    parser.add_argument('--methods', type=str, nargs='+', default=list(EXPLAINERS), choices=EXPLAINERS.keys())
    parser.add_argument('--images-per-run', type=int, default=3, help="Random test images explained per run")
    parser.add_argument('--top-percent', type=float, default=20.0,
                        help="Percent of most important pixels that form the focus region of a method")
    parser.add_argument('--smooth-sigma', type=float, default=4.0,
                        help="Gaussian blur (px) of maps before thresholding, 0 = off")
    parser.add_argument('--output-dir', type=str, default="runs/xai_comparison")
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = pars_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [Path(r) for r in args.runs] if args.runs else find_runs(args.runs_dir)
    if not runs:
        raise SystemExit(f"No trained runs found in {args.runs_dir}")

    rows, general = compare_runs(
        runs, random.Random(args.seed),
        methods=args.methods,
        data_root=args.data_root,
        catalogue_path=args.catalogue,
        images_per_run=args.images_per_run,
        top_percent=args.top_percent,
        smooth_sigma=args.smooth_sigma,
    )
    save_table(rows, out_dir / "iou_table.csv")
    save_table(general, out_dir / "iou_general.csv")

    print(f"\n{'model':<16} {'side':<8} {'pair':<24} {'mask IoU':>9} {'bbox IoU':>9}")
    for r in rows:
        print(f"{r['model']:<16} {r['side']:<8} {r['pair']:<24} {r['mask_iou']:>9.3f} {r['bbox_iou']:>9.3f}")
    print(f"\nGENERAL (mean over all runs and images)\n{'pair':<24} {'images':>6} {'mask IoU':>9} {'bbox IoU':>9}")
    for r in general:
        print(f"{r['pair']:<24} {r['images']:>6} {r['mask_iou']:>9.3f} {r['bbox_iou']:>9.3f}")
    print(f"\nSaved to {out_dir / 'iou_table.csv'} and {out_dir / 'iou_general.csv'}")


if __name__ == "__main__":
    main()
