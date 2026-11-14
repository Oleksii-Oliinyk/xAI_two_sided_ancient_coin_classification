import torch.nn as nn
from src.constants import CLASS_NUMBER
from torchvision import models


class CoinDetectionMobileNetV3(nn.Module):
    def __init__(self):
        """MobileNetV3-Large, pretrained on ImageNet at 224x224"""
        super(CoinDetectionMobileNetV3, self).__init__()
        self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.backbone.classifier[-1] = nn.Linear(self.backbone.classifier[-1].in_features, CLASS_NUMBER)

    def forward(self, x):
        return self.backbone(x)
