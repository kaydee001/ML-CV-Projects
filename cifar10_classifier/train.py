import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import SimpleCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'using device : {device}')

def main():
    # converting image to tensor and then normalizing the rgb channels
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # loading the data set
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # produces batch of size = 4 images, 3 input channels -> RGB, 32x32 pixels
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # converting the train_loader into an iterator (the 1st batch only)
    images, labels = next(iter(train_loader))
    # print(f"batch shape : {images.shape}")
    # print(f"labels : {labels}")

    # processing power shifted to gpu
    model = SimpleCNN().to(device)
    # standard loss for classification
    criterion = nn.CrossEntropyLoss()
    # using adam optimizer with learning rate=0.01 on model.parameters()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # print(f"model created {sum(p.numel() for p in model.parameters())}")
    print(f"model on device: {next(model.parameters()).device}")

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        
        for batch_idx,  (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            if(batch_idx+1)%100 == 0:
                print(f"epoch [{epoch+1}/{num_epochs}], step [{batch_idx+1}/{len(train_loader)}], loss: {loss.item():.4f}")
        
        print(f'epoch [{epoch+1}/{num_epochs}] complete')


if __name__ == "__main__":
    main()