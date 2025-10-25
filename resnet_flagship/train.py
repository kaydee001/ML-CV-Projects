import torch 
import torch.nn as nn
import torch.optim as optim
from model import FlowerClassifier
from data import get_data_loaders

def train_model(model, train_loader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    print(f"using device : {device}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    loss_history = []

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

            running_loss += loss.item()

            if (batch_idx+1) % 10 == 0:
                print(f"epoch [{epoch+1}/{num_epochs}], batch [{batch_idx+1}/{len(train_loader)}], loss: {loss.item():.4f}")

        avg_loss = running_loss/len(train_loader)
        loss_history.append(avg_loss)
        print(f"epoch : {epoch+1} complete, avg loss : {avg_loss:.4f}")

    return loss_history

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # find max along dimension1 -> class
            _, predicted = torch.max(outputs, 1) # output.shape -> (batch_size, no_of_classes   )

            total += labels.size(0)
            # .item() to convert it from tensor boolean to python int 
            correct += (predicted == labels).sum().item()

        accuracy = 100*correct / total
        print(f"test accuracy : {accuracy:.2f}")
        return accuracy

def save_model(model, save_path="models/example.pth"):
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # state_dict() to only save the weights
    torch.save(model.state_dict(), save_path)
    print(f"model saved to {save_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="new")

    parser.add_argument('--data_dir', type=str, default='data/flower_photos', help="path to flowers dataset")
    parser.add_argument('--epochs', type=int, default=10, help="no of epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--save_path', type=str, default='models/example.pth', help="where to save model")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FlowerClassifier(num_classes=102)
    train_loader, test_loader = get_data_loaders(batch_size=args.batch_size)

    loss_history = train_model(model, train_loader, num_epochs=args.epochs)

    evaluate_model(model, test_loader, device)

    save_model(model, args.save_path)

