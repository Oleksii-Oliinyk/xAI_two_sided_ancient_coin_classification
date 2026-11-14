import torch.nn as nn
from src.constants import CLASS_NUMBER
from torchvision import models


class CoinDetectionEfficientNetB0(nn.Module):
    def __init__(self):
        """EfficientNet-B0, native input resolution 224x224"""
        super(CoinDetectionEfficientNetB0, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone.classifier[-1] = nn.Linear(self.backbone.classifier[-1].in_features, CLASS_NUMBER)

    def forward(self, x):
        return self.backbone(x)
