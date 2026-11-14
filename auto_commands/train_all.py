import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.constants import MODEL_CONFIGS, SIDES


TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"


def pars_args():
    parser = argparse.ArgumentParser(description="Train every model type on every side, one after another")
    parser.add_argument('--epochs', type=int, default=50)
    return parser.parse_args()


def main():
    args = pars_args()
    results = []

    for model_name in MODEL_CONFIGS:
        for side in SIDES:
            command = [sys.executable, str(TRAIN_SCRIPT), "--side", side, "--model", model_name,
                       "--epochs", str(args.epochs)]
            print(f"\n=== {model_name} | {side} ===\n{' '.join(command)}", flush=True)

            start = time.time()
            # separate process per run: GPU memory and the wandb run are fully released in between,
            # and one crashed run doesn't stop the rest
            return_code = subprocess.run(command).returncode
            minutes = (time.time() - start) / 60
            results.append((model_name, side, "ok" if return_code == 0 else f"FAILED ({return_code})", minutes))

    print("\n=== Summary ===")
    for model_name, side, status, minutes in results:
        print(f"{model_name:<16} {side:<8} {status:<12} {minutes:6.1f} min")


if __name__ == "__main__":
    main()
