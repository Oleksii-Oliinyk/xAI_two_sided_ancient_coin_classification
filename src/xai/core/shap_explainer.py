import random
from pathlib import Path

import numpy as np
import shap
import torch
import torch.nn as nn

from src.dataset.custom_dataset import CoinDataset
from src.xai.core.base import BaseExplainer


class _ClassProbability(nn.Module):
    """Wraps a classifier so it outputs only the softmax probability of one class"""

    def __init__(self, model, class_idx):
        super().__init__()
        self.model = model
        self.class_idx = class_idx

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)[:, self.class_idx:self.class_idx + 1]


class ShapExplainer(BaseExplainer):
    """
    SHAP GradientExplainer (expected gradients, Lundberg & Lee, 2017): compares the input with
    background images from train/<side> and attributes the class probability to every pixel.
    """
    name = "SHAP"
    signed = True

    background_size = 32  # reference images SHAP compares the input against
    nsamples = 100        # (background, interpolation) samples per explanation, more = smoother but slower
    seed = 42

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.background = self._load_background()

    def _load_background(self):
        dataset = CoinDataset(Path(self.data_root) / "train", self.side)
        size = min(self.background_size, len(dataset))
        indices = random.Random(self.seed).sample(range(len(dataset)), size)
        return torch.stack([dataset[i][0] for i in indices]).to(self.device)

    def explain(self, image_tensor, class_idx):
        explainer = shap.GradientExplainer(_ClassProbability(self.model, class_idx), self.background)
        shap_values = explainer.shap_values(image_tensor.unsqueeze(0).to(self.device), nsamples=self.nsamples)
        # [1, 3, H, W, 1] -> [H, W], channels are summed: the image is grayscale repeated to 3 channels
        return np.asarray(shap_values)[0, :, :, :, 0].sum(axis=0)
