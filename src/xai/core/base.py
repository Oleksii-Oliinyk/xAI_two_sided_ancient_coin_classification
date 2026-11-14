import numpy as np
import torch

from src.dataset.custom_dataset import MEAN, STD


class BaseExplainer():
    """
    Common interface of all explainers: explain(image_tensor, class_idx) -> map [H, W]

    signed = True:  map has +/- values (red supports the class, blue works against it)  -> SHAP, LIME
    signed = False: map is non-negative importance scaled to 0..1                         -> Grad-CAM(++)
    """
    name = ""
    signed = False

    def __init__(self, model, device, target_layer, data_root, side):
        """
        model:        trained classifier in eval mode
        device:       torch device the model lives on
        target_layer: name of the last conv block (MODEL_CONFIGS[...]["target_layer"]), used by Grad-CAM(++)
        data_root:    folder with train/val/test splits, used by SHAP for background images
        side:         'Observe' or 'Reverse'
        """
        self.model = model
        self.device = device
        self.target_layer = target_layer
        self.data_root = data_root
        self.side = side

    def explain(self, image_tensor, class_idx):
        raise NotImplementedError


def denormalize_gray(image_tensor):
    """[3, H, W] normalized tensor -> [H, W] grayscale image in 0..1"""
    image = image_tensor[0].cpu().numpy() * STD[0] + MEAN[0]
    return np.clip(image, 0, 1)


def normalize_gray_batch(images):
    """[N, H, W] grayscale images in 0..1 -> [N, 3, H, W] normalized tensor (inverse of denormalize_gray)"""
    x = torch.as_tensor(images, dtype=torch.float32).unsqueeze(1).repeat(1, 3, 1, 1)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (x - mean) / std
