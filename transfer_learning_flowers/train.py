import torch
import torch.nn as nn
import torch.optim as optim
from model import FlowerClassifier
from utils import get_data_loaders

def train_model(model, train_loader, num_epochs=10):
    # setup up device -> cpu to gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device : {device}")
    model = model.to(device)

    # loss function -> CrossEntropyLoss (commonly used in multi-class classification)
    criterion = nn.CrossEntropyLoss()

    # adam optimizer to update weights on gradients with learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training loop
    for epoch in range(num_epochs):
        model.train() # set model to training model (enables dropuout)
        running_loss = 0.0

        # tracking loss history
        loss_history = []

        # processing data in batches -> 32 images at a time
        for batch_idx, (images, labels) in enumerate(train_loader):
            # move to device
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)
            # loss calc
            loss = criterion(outputs, labels)
            # backward pass 
            optimizer.zero_grad() # clearing out old gradients
            loss.backward() # calculating new gradients
            # updating weight
            optimizer.step() 
            
            # calculating cumulative loss for this epoch
            running_loss += loss.item()

            if(batch_idx + 1) % 10 == 0:
                print(f"epoch [{epoch+1}/{num_epochs}], batch [{batch_idx+1}/{len(train_loader)}], loss: {loss.item():.4f}")

        # calculating avg loss for this epoch
        avg_loss = running_loss/len(train_loader)
        loss_history.append(avg_loss)
        print(f"epoch : {epoch+1} complete, avg loss : {avg_loss:.4f}")

    return loss_history

def evaluate_model(model, test_loader, device):
    model.eval() # set model to evaluating model (disables dropout)
    correct = 0
    total = 0

    # disabling gradient calc
    with torch.no_grad():
        for images, labels in test_loader:
            # move to device
            images, labels = images.to(device), labels.to(device)

            # get predictions
            outputs = model(images)
            # get class with highest probability
            _, predicted = torch.max(outputs, 1)

            # count the total no of correct predictions
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

        # calculating accuracy percentage
        accuracy = 100*correct / total
        print(f"test accuracy : {accuracy:.2f}")
        return accuracy

def save_model(model, save_path='models/flower_classifier.pth'):
    import os

    # creating models/ directory if it doesnt exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # save model weights
    torch.save(model.state_dict(), save_path)
    print(f"model saved to {save_path}")

if __name__ == "__main__":
    import argparse

    # setting up command line argument parsing
    parser = argparse.ArgumentParser(description="train flower classifier")

    # defining arguments
    parser.add_argument('--data_dir', type=str, default='data/flower_photos', help="path to flowers dataset")
    parser.add_argument('--epochs', type=int, default=10, help="no of epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--save_path', type=str, default='models/flower_classifier.pth', help="where to save model")

    # parse arguments from command line
    args = parser.parse_args()

    # model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FlowerClassifier(num_classes=5)
    train_loader, test_loader = get_data_loaders(args.data_dir, batch_size=args.batch_size)

    # training the model
    print(f"training with {args.epochs} epochs, lr = {args.lr}")
    loss_history = train_model(model, train_loader, num_epochs=args.epochs)
    train_model(model, train_loader, num_epochs=args.epochs)

    # evaluate the model -> testing
    evaluate_model(model, test_loader, device)

    # saving the model
    save_model(model, args.save_path)

    from visualize import plot_loss_curve
    plot_loss_curve(loss_history, save_path="loss_curve.png")
    print(f"training complete")