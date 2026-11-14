import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset.preprocessing import DatasetPreprocessing


def pars_args():
    parser = argparse.ArgumentParser(
        description="Preprocess data into train/val/test split dividing class samples equally"
    )
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help="Root directory for dataset (contains 'Observe' and 'Reverse' folders)"
    )
    parser.add_argument(
        '--output-folder',
        type=str,
        required=True,
        help="Directory where train/val/test folders will be created"
    )
    parser.add_argument(
        '--catalogue',
        type=str,
        default=None,
        help="Excel catalogue of classes, defaults to <data-root>/RRC60_catalogue.xlsx"
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.15,
        help="Fraction of each class that goes to val and to test"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits"
    )
    return parser.parse_args()


def main():
    args = pars_args()

    preprocessing = DatasetPreprocessing(
        data_root=args.data_root,
        output_folder=args.output_folder,
        catalogue_path=args.catalogue,
        split_ratio=args.split_ratio,
        seed=args.seed,
    )
    splits = preprocessing.data_split()
    preprocessing.save_split(splits)
    preprocessing.save_catalogue()


if __name__ == "__main__":
    main()
