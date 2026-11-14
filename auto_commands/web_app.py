import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
# default paths (runs/production, dataset/data) are relative to the project root
os.chdir(PROJECT_ROOT)

import uvicorn

from src.web.server import create_app
from src.xai.comparison import find_production_run


def pars_args():
    parser = argparse.ArgumentParser(description="Web app: upload a coin photo, predict, explore XAI heatmaps")
    parser.add_argument('--host', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--observe-run', type=str, default=None,
                        help="Run folder of the Observe model, default = the Observe model in runs/production")
    parser.add_argument('--reverse-run', type=str, default=None,
                        help="Run folder of the Reverse model, default = the Reverse model in runs/production")
    parser.add_argument('--catalogue', type=str, default="dataset/data/catalogue.json",
                        help="catalogue.json created by data_split")
    parser.add_argument('--data-root', type=str, default="dataset/data",
                        help="Dataset splits, SHAP takes its background images from <data-root>/train")
    args = parser.parse_args()
    args.observe_run = args.observe_run or find_production_run("Observe")
    args.reverse_run = args.reverse_run or find_production_run("Reverse")
    return args


def main():
    args = pars_args()
    app = create_app(
        observe_run=args.observe_run,
        reverse_run=args.reverse_run,
        catalogue_path=args.catalogue,
        data_root=args.data_root,
    )
    print(f"Open http://{args.host}:{args.port} in the browser")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
