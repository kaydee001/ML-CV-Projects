import torch
from model import FlowerClassifier
from utils import get_data_loaders
import matplotlib.pyplot as plt

def visualize_predictions(model, test_loader, device, num_images=8, save_path='predictions.png'):
    model.eval() # model set to evaluation mode

    # class names -> in alphabetically order (IMP)
    class_names = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

    # get a random batch everytime
    import random
    all_batches = list(test_loader)
    images, labels = random.choice(all_batches)

    # moving the batch from cpu -> gpu
    images = images.to(device)
    labels = labels.to(device)

    # no gradient calculations
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # moving back from gpu -> cpu for plotting
    images = images.cpu()
    labels = labels.cpu()
    predicted = predicted.cpu()

    # subplot grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    # denormalize values to reverse imagenet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i in range(num_images):
        # denormalize image for displaying
        img = images[i]
        img = img*std + mean # reverse normalization
        img = torch.clamp(img, 0, 1) # clip to valid range -> 0,1

        # converting from (channels, width, height) -> (width, height, channels) for matplotlib
        img = img.permute(1, 2, 0).numpy()

        axes[i].imshow(img)
        axes[i].axis("off")

        # get predictions and true labels
        true_label = class_names[labels[i]]
        pred_label = class_names[predicted[i]]

        if predicted[i] == labels[i]:
            color = "green"
            title = f"✓{pred_label}"
        else:
            color = "red"
            title = f"✗ prediction : {pred_label}\n true : {true_label}"

        axes[i].set_title(title, fontsize=12, color=color, fontweight="bold")

    # saving the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"predictions saved to {save_path}")
    plt.close()

def plot_loss_curve(loss_history, save_path='loss_curve.png'):
    # plotting loss over epochs
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(loss_history)+1)
    # plotting loss curve
    plt.plot(epochs, loss_history)

    plt.xlabel("epochs")
    plt.ylabel("avg loss")
    plt.title("training loss")

    # save and close
    plt.savefig(save_path)
    plt.close()
    print(f"loss curve saved to {save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading the trained model
    model = FlowerClassifier(num_classes=5)
    model.load_state_dict(torch.load('models/flower_classifier.pth'))
    model = model.to(device)

    # get test data
    _, test_loader = get_data_loaders('data/flower_photos', batch_size=32)
    # generate visualizations
    visualize_predictions(model, test_loader, device)