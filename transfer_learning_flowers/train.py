import torch
import torch.nn as nn
import torch.optim as optim
from model import FlowerClassifier
from utils import get_data_loaders

def train_model(model, train_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device : {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if(batch_idx + 1) % 10 == 0:
                print(f"epoch [{epoch+1}/{num_epochs}], batch [{batch_idx+1}/{len(train_loader)}], loss: {loss.item():.4f}")

        avg_loss = running_loss/len(train_loader)
        print(f"epoch : {epoch+1} complete, avg loss : {avg_loss:.4f}")

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted==labels).sum().item()

        accuracy = 100*correct / total
        print(f"test accuracy : {accuracy:.2f}")
        return accuracy

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FlowerClassifier(num_classes=5)
    train_loader, test_loader = get_data_loaders('data/flower_photos', batch_size=32)

    train_model(model, train_loader, num_epochs=10)
    evaluate_model(model, test_loader, device)