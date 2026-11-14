"""
XAI focus vs expert (Label Studio) focus areas, see src/xai/label_studio_metric_focus_points_comparison.py.

Works from any folder (switches to the project root), every argument is passed through:
    python auto_commands/xai_focus_comparison.py                      all production models, all methods
    python auto_commands/xai_focus_comparison.py --methods gradcam shap --no-figures
    python auto_commands/xai_focus_comparison.py --runs-dir runs/experiments
    python auto_commands/xai_focus_comparison.py --help

The console output is also written to <output-dir>/run.log (default runs/xai_focus_comparison/run.log).
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # all default paths (dataset/data, runs/...) are relative to the project root

from src.utils.tee import log_stdout_to
from src.xai.label_studio_metric_focus_points_comparison import main, pars_args


if __name__ == "__main__":
    argv = sys.argv[1:]
    with log_stdout_to(Path(pars_args(argv).output_dir) / "run.log"):
        main(argv)
