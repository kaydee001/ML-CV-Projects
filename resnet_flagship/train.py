import torch 
import torch.nn as nn
import torch.optim as optim
import time
import logging
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import FlowerClassifier
from data import get_data_loaders
from visualize import plot_loss_history
from torch.amp import autocast, GradScaler

# loading hyperparameters from config file
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# setup loggging for both console and file -> with timestamps
def setup_logging(log_file="training.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # clear existing handlers
    logger.handlers = []

    # file handler -> to save logs to file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # console handler -> to print logs to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # format : timestamp-level-message -> everything which was in print statement
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def train_model(model, train_loader, val_loader, class_weights, config):
    start_time = time.time()

    # GradScaler for mixed precision stability
    scaler = GradScaler()

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    logging.info(f"using device : {device}")
    model.to(device)

    # weighted loss to handle class imbalance -> rear classes get higher weight
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # loading all hyperparameters from the config file
    num_epochs = config['training']['epochs']
    # using AdamW -> weight decay
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    # smooth learning rate decay
    scheduler = CosineAnnealingLR(optimizer, T_max=config['scheduler']['T_max'], eta_min=config['scheduler']['eta_min'])

    # tracking
    loss_history = []
    val_acc_history = []
    best_val_acc = 0.0
    epochs_no_improve = 0 # for early stopping

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # training loop
        for batch_idx, (images, labels) in enumerate(train_loader): 
            images = images.to(device)
            labels = labels.to(device)

            # mixed precision forwad pass
            with autocast(device_type='cuda'):         
                outputs = model(images)
                loss = criterion(outputs, labels)

            # backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # gradient clipping -> prevent exploding gradients 
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['training']['max_grad_norm'])

            # updating the weights
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if (batch_idx+1) % 10 == 0:
                logging.info(f"epoch [{epoch+1}/{num_epochs}], batch [{batch_idx+1}/{len(train_loader)}], loss: {loss.item():.4f}")

        avg_loss = running_loss/len(train_loader)
        loss_history.append(avg_loss)

        # validation
        val_acc = evaluate(model, val_loader, device, dataset_name="validation")
        val_acc_history.append(val_acc)

        logging.info(f"epoch : {epoch+1} complete, avg loss : {avg_loss:.4f}, val acc = {val_acc:.2f}")
        logging.info("---")

        # early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, config['save']['best_model_path'])
            logging.info(f"new best model saved, best validation accuracy : {best_val_acc:.2f}")
            epochs_no_improve = 0 # resetting the counter
        else:
            epochs_no_improve += 1
            logging.info(f"no improvement for {epochs_no_improve} epochs(s)")

            # stop if no_improv for (patience) epochs
            if epochs_no_improve >= config['training']['early_stopping_patience']:
                logging.info(f"early stopping triggered! no improvement for {epochs_no_improve} epoch(s)")
                logging.info(f"best validation accuracy : {best_val_acc:.2f}")
                break # exit training

        # update learning rate           
        scheduler.step()
        logging.info(f"learning rate : {scheduler.get_last_lr()[0]:.6f}")

        # tracking time taken
        total_time = time.time() - start_time
        logging.info(f"total time : {total_time:.2f}s -> ({total_time/60:.2f} mins)")
        logging.info(f"avg time per epoch : {total_time/num_epochs:.2f}s")
        logging.info("---")

    return loss_history, val_acc_history

def evaluate(model, data_loader, device, dataset_name="validation"):
    model.eval() # disables dropout etc
    correct = 0
    total = 0

    start_test_time = time.time()
    # no gradient calculation
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # find max along dimension1 -> class
            _, predicted = torch.max(outputs, 1) # output.shape -> (batch_size, no_of_classes)

            total += labels.size(0)
            # .item() to convert it from tensor boolean to python int 
            correct += (predicted == labels).sum().item()

        accuracy = 100*correct / total

        end_test_time = time.time()-start_test_time
        logging.info(f"{dataset_name} accuracy : {accuracy:.2f}, time taken : {end_test_time:.2f}s")

        return accuracy

def save_model(model, save_path="models/example.pth"):
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # state_dict() to only save the weights
    torch.save(model.state_dict(), save_path)
    logging.info(f"model saved to {save_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="new")

    parser.add_argument('--config', type=str, default='config.yaml', help="path to config file")
    args = parser.parse_args()

    # load config and setup logging
    config = load_config(args.config)    
    setup_logging(log_file=config['logging']['log_file'])
    logging.info("-"*10)
    logging.info(f"starting training with config file : {args.config}")
    logging.info("-"*10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # creating the model
    model = FlowerClassifier(num_classes=config['model']['num_classes'])

    # load data with class weights for imbalance handling
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(batch_size=config['training']['batch_size'], test_batch_size=config['data']['test_batch_size'])

    # training the model
    loss_history, val_acc_history = train_model(model, train_loader, val_loader, class_weights, config=config)

    # plotting the cuves
    plot_loss_history(loss_history, val_acc_history, save_path="loss_curve.png")

    logging.info("--- final test evaluation ---")
    evaluate(model, test_loader, device, dataset_name="test")
    
    # save final model
    save_model(model, config['save']['final_model_path'])