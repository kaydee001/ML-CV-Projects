import torch
import torch.nn as nn
from torchvision import models

class FlowerClassifier(nn.Module):
    # we are using Flower-102 dataset
    def __init__(self, num_classes=102):
        super(FlowerClassifier, self).__init__()

        # loading resnet pretrained on ImageNet  
        self.model = models.resnet18(weights="IMAGENET1K_V1")

        # replacing final layer to match our dataset -> 102 classes
        num_features = self.model.fc.in_features # 512 features from ResNet18
        self.model.fc = nn.Linear(num_features, num_classes) # 512 -> 102

    def forward(self, x):
        return self.model(x)
