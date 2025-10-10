import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'using device : {device}')

def main():
    # converting image to tensor and then normalizing the rgb channels
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # loading the data set
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # produces batch of size = 4 images, 3 input channels -> RGB, 32x32 pixels
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # converting the train_loader into an iterator (the 1st batch only)
    images, labels = next(iter(train_loader))
    print(f"batch shape : {images.shape}")
    print(f"labels : {labels}")

if __name__ == "__main__":
    main()