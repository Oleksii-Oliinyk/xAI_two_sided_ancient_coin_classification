import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset.feature_extractor import CatalogueFeatureExtractor


def pars_args():
    parser = argparse.ArgumentParser(
        description="Split catalogue descriptions into features and rank them per class from most to least unique"
    )
    parser.add_argument('--catalogue', type=str, default="dataset/data/catalogue.json",
                        help="catalogue.json created by data_split")
    parser.add_argument('--output', type=str, default="dataset/data/catalogue_features",
                        help="Output path without extension, .csv and .json are saved")
    parser.add_argument('--similarity-threshold', type=float, default=0.75,
                        help="Similarity from which two features count as the same when counting similar classes")
    return parser.parse_args()


def main():
    args = pars_args()

    extractor = CatalogueFeatureExtractor(args.catalogue, similarity_threshold=args.similarity_threshold)
    features = extractor.extract_all()
    rows = extractor.compute_uniqueness(features)
    extractor.save_csv(rows, f"{args.output}.csv")
    extractor.save_json(rows, f"{args.output}.json")


if __name__ == "__main__":
    main()
