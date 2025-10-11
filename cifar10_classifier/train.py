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

    # loading the testing dataset
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # no shuffle for testing
    test_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    # converting the train_loader into an iterator (the 1st batch only)
    images, labels = next(iter(train_loader))
    # print(f"batch shape : {images.shape}")
    # print(f"labels : {labels}")

    # processing power shifted to gpu
    model = SimpleCNN().to(device)
    # standard loss for classification
    criterion = nn.CrossEntropyLoss()
    # using adam optimizer with learning rate=0.001 on model.parameters()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # print(f"model created {sum(p.numel() for p in model.parameters())}")
    print(f"model on device: {next(model.parameters()).device}")

    def evaluate(model, test_loader, device):
        # inheriting from nn.Module
        model.eval()

        correct = 0 # no of correct predictions
        total = 0 # total no of images seen

        # dont calculate the gradients -> faster (IMP)
        with torch.no_grad():
            for images, labels in test_loader: 
                images = images.to(device)
                labels = labels.to(device)

                # only forward pass and getting the predictions
                outputs = model(images)

                #tracking the stats
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()

        accuracy = 100 * correct/total
        return accuracy

    # training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        # inheriting from nn.Module
        model.train()
        
        running_loss = 0.0
        correct = 0.0 # no of correct predictions
        total = 0.0 # total no of images seen
        
        for batch_idx,  (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # zero the gradients (IMP) -> to clear the old gradients
            optimizer.zero_grad()
            # forward pass and loss calculation 
            outputs = model(images)
            loss = criterion(outputs, labels)
            # backward pass
            loss.backward()
            # updating the weights
            optimizer.step()

            # tracking the stats
            # adding the total cumulative loss -> converting loss tensor to python number 
            running_loss += loss.item()
            # ignores the max_value,  
            _, predicted = torch.max(outputs.data, 1)
            # batch size -> total no of images in the current batch
            total += labels.size(0)
            # counting the no of prediction which match the true labels
            correct += (predicted==labels).sum().item()

            if(batch_idx+1)%100 == 0:
                avg_loss = running_loss/100
                accuracy = 100 * correct/total
                print(f"epoch [{epoch+1}/{num_epochs}], step [{batch_idx+1}/{len(train_loader)}], loss : {avg_loss:.4f}, accuracy : {accuracy:.2f}%")
                # reset for the next 100 batches
                running_loss = 0.0
        
        epoch_acc = 100 * correct/total
        print(f"epoch [{epoch+1}/{num_epochs}] complete with accuracy : {epoch_acc}")

        test_acc = evaluate(model, test_loader, device)
        print(f"test accuracy : {test_acc:.2f}% \n")


if __name__ == "__main__":
    main()