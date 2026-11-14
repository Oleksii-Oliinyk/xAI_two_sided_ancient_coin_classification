import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.constants import EXPLAINERS
from src.engine.inference import CoinInference
from src.xai.comparison import find_production_run


def pars_args():
    parser = argparse.ArgumentParser(description="Classify a coin from its Observe and/or Reverse image")
    parser.add_argument('--observe-image', type=str, default=None, help="Image of the Observe side")
    parser.add_argument('--reverse-image', type=str, default=None, help="Image of the Reverse side")
    parser.add_argument('--observe-run', type=str, default=None,
                        help="Run folder of the Observe model, default = the Observe model in runs/production")
    parser.add_argument('--reverse-run', type=str, default=None,
                        help="Run folder of the Reverse model, default = the Reverse model in runs/production")
    parser.add_argument('--catalogue', type=str, default="dataset/data/catalogue.json",
                        help="catalogue.json created by data_split")
    parser.add_argument('--explain', type=str, nargs='*', default=None, choices=EXPLAINERS.keys(),
                        help="Explanation methods to run, e.g. --explain gradcam shap. "
                             "--explain alone = all methods. Figures go to runs/xai/")
    args = parser.parse_args()

    if args.explain == []:
        args.explain = list(EXPLAINERS)

    if args.observe_image is None and args.reverse_image is None:
        parser.error("give at least one of --observe-image / --reverse-image")
    if args.observe_image and not args.observe_run:
        args.observe_run = find_production_run("Observe")
    if args.reverse_image and not args.reverse_run:
        args.reverse_run = find_production_run("Reverse")
    return args


def main():
    args = pars_args()

    inference = CoinInference(
        observe_run=args.observe_run if args.observe_image else None,
        reverse_run=args.reverse_run if args.reverse_image else None,
        catalogue_path=args.catalogue,
        explainers=args.explain,
    )
    prediction = inference.predict(observe_image=args.observe_image, reverse_image=args.reverse_image)
    print(prediction)


if __name__ == "__main__":
    main()
