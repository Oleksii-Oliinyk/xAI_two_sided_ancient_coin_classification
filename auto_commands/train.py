import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.engine.train import ClassifierTrain
from src.constants import MODEL_CONFIGS, SIDES


def pars_args():
    parser = argparse.ArgumentParser(description="Train a coin classification model")
    parser.add_argument('--data-path', type=str, default="dataset/data",
                        help="Folder with train/val/test splits")
    parser.add_argument('--side', type=str, required=True, choices=SIDES,
                        help="Which coin side to train the model on")
    parser.add_argument('--model', type=str, default="resnet18", choices=MODEL_CONFIGS.keys())
    parser.add_argument('--epochs', type=int, default=10)
    return parser.parse_args()


def main():
    args = pars_args()

    trainer = ClassifierTrain(
        data_root=args.data_path,
        side=args.side,
        model_name=args.model,
        model_config=MODEL_CONFIGS[args.model],
        num_epochs=args.epochs,
    )
    trainer.fit()
    trainer.test()


if __name__ == "__main__":
    main()
