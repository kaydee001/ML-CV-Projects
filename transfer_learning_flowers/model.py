import torch
import torch.nn as nn
from torchvision import models

class FlowerClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(FlowerClassifier, self).__init__()

        # loading pre trained resnet from imagenet
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        # no of input features of final layer -> 512 for resnet 
        num_features = self.model.fc.in_features
        # replace final layer -> 1000 imagenet classes => 5 flower classes
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # passing the input thru modified resnet
        return self.model(x)
    
if __name__ == "__main__":
    print("creating flower classifier")
    model = FlowerClassifier(num_classes=5)

    # dummy input -> 4 images, 3 channels, 224x224
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)

    print(f"input shape : {dummy_input.shape}")
    print(f"output shape : {output.shape}")
    print(f"model has {sum(p.numel() for p in model.parameters())} parameters")
    print("model working")