"""
Shift of XAI importance regions between Good and Bad (worn) coins, see src/xai/good_bad_importance_shift.py.

Works from any folder (switches to the project root), every argument is passed through:
    python auto_commands/xai_quality_shift.py                       production models, all methods
    python auto_commands/xai_quality_shift.py --methods gradcam shap --no-figures
    python auto_commands/xai_quality_shift.py --zone-bounds 0.4 0.7
    python auto_commands/xai_quality_shift.py --help

The console output is also written to <output-dir>/run.log (default runs/xai_quality_shift/run.log).
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # all default paths (dataset/..., runs/...) are relative to the project root

from src.utils.tee import log_stdout_to
from src.xai.good_bad_importance_shift import main, pars_args


if __name__ == "__main__":
    argv = sys.argv[1:]
    with log_stdout_to(Path(pars_args(argv).output_dir) / "run.log"):
        main(argv)
