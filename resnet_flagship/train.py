import torch 
import torch.nn as nn
import torch.optim as optim
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import FlowerClassifier
from data import get_data_loaders
from visualize import plot_loss_history
from torch.amp import autocast, GradScaler

def train_model(model, train_loader, val_loader, class_weights, num_epochs=10):
    start_time = time.time()

    scaler = GradScaler()

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    print(f"using device : {device}")
    model.to(device)

    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00001)

    loss_history = []
    val_acc_history = []
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        epoch_time = time.time()

        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader): 
            images = images.to(device)
            labels = labels.to(device)

            with autocast(device_type='cuda'):         
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if (batch_idx+1) % 10 == 0:
                print(f"epoch [{epoch+1}/{num_epochs}], batch [{batch_idx+1}/{len(train_loader)}], loss: {loss.item():.4f}")

        avg_loss = running_loss/len(train_loader)
        loss_history.append(avg_loss)

        val_acc = evaluate_model(model, val_loader, device)
        val_acc_history.append(val_acc)

        print(f"epoch : {epoch+1} complete, avg loss : {avg_loss:.4f}, val acc = {val_acc:.2f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, "models/best_model.pth")
            print(f"new best model saved, val_acc : {best_val_acc:.2f}")

        scheduler.step()
        print(f"learning rate : {scheduler.get_last_lr()[0]:.6f}")

        total_time = time.time() - start_time
        print(f"total time : {total_time:.2f}s -> ({total_time/60:.2f} mins)")
        print(f"avg time per epoch : {total_time/num_epochs:.2f}s")

    return loss_history, val_acc_history

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    start_test_time = time.time()

    print("testing the model")

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

        end_test_time = time.time()    
        print(f"testing done, time takem : {end_test_time-start_test_time:.2f}s")
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
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(batch_size=args.batch_size, test_batch_size=128)

    loss_history, val_acc_history = train_model(model, train_loader, val_loader, class_weights, num_epochs=args.epochs)
    plot_loss_history(loss_history, val_acc_history, save_path="loss_curve.png")

    evaluate_model(model, test_loader, device)

    save_model(model, args.save_path)