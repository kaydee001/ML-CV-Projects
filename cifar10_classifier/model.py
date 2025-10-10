import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) # input channels (3), output channels (16), 3x3 filter, padding=1         
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 2x2 pooling which reduces the size by half
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # input channels (16), output channels (32), 3x3 filter, padding=1

    def forward(self, x):
        pass