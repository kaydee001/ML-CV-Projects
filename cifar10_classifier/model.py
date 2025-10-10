import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) # input channels (3), output channels (16), 3x3 filter, padding=1         
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 2x2 pooling which reduces the size by half
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # input channels (16), output channels (32), 3x3 filter, padding=1

        self.fc1 = nn.Linear(32*8*8, 128)
        self.fc2  = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        # flatting from 2d to 1d
        x = torch.flatten(x, start_dim=1) # flatten everything the "batch" dimension (ie no of images)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    dummy_input = torch.randn(4, 3, 32, 32)
    model = SimpleCNN()
    output = model(dummy_input)

    print(f"input shape : {dummy_input.shape}")
    print(f"output shape : {output.shape}")
    print(f"output sample : {output[0]}")