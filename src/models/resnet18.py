import torch.nn as nn
from src.constants import CLASS_NUMBER
from torchvision import models


class CoinDetectionResNet18(nn.Module):
    def __init__(self):
        """"""
        super(CoinDetectionResNet18, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, CLASS_NUMBER)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.backbone(x)
