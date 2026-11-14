CLASS_NUMBER = 60
SIDES = ("Observe", "Reverse")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

from src.models.efficientnet import CoinDetectionEfficientNetB0
from src.models.mobilenet import CoinDetectionMobileNetV3
from src.models.resnet18 import CoinDetectionResNet18
from src.xai.core.gradcam import GradCAMExplainer, GradCAMPlusPlusExplainer
from src.xai.core.lime_explainer import LimeExplainer
from src.xai.core.shap_explainer import ShapExplainer

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# runs/production:  models actually used (best ones, currently both resnet18 runs) -> default for inference / XAI
# runs/experiments: every other trained model, new training runs land here and are moved to production by hand
RUNS_DIR = "runs"
PRODUCTION_RUNS_DIR = f"{RUNS_DIR}/production"
EXPERIMENT_RUNS_DIR = f"{RUNS_DIR}/experiments"

WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
WANDB_PROJECT = "coin-classification"
WANDB_ENTITY = "oleksii-oliinyk-lviv-polytechnic-national-university"

MODEL_CONFIGS = {
    "resnet18": {
        "model": CoinDetectionResNet18,
        "lr": 1e-4,
        "batch_size": 32,
        "target_layer": "backbone.layer4",  # last conv block, used by Grad-CAM(++)
    },
    "mobilenet_v3": {
        "model": CoinDetectionMobileNetV3,
        "lr": 3e-4,
        "batch_size": 32,
        "target_layer": "backbone.features",
    },
    "efficientnet_b0": {
        "model": CoinDetectionEfficientNetB0,
        "lr": 3e-4,
        "batch_size": 32,
        "target_layer": "backbone.features",
    },
}

EXPLAINERS = {
    "gradcam": GradCAMExplainer,
    "gradcam++": GradCAMPlusPlusExplainer,
    "lime": LimeExplainer,
    "shap": ShapExplainer,
}
