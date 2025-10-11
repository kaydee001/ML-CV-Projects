import matplotlib.pyplot as plt
import torch
import numpy as np

CIFAR_CLASSES  = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def visualize_predicitons(model, test_loader, device, num_images=16):
    # inheriting from nn.Module -> disables training specific behaviour
    model.eval()

    # get only 1 batch of the images -> in this case, num_images
    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)

    # get predictions
    with torch.no_grad():
        output = model(images)
        # ? output.data
        _, predicted = torch.max(output, 1)
    
    # moving back to cpu for visualization
    images = images.cpu()
    labels = labels.cpu()
    predicted = predicted.cpu()

    # plot the predictions along with the image from CIFAR10 image dataset
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    # flatten 2d array to 1d for easier indexing
    axes = axes.ravel()

    for i in range(num_images):
        # get the image and convert it from tensor to numpy 
        img = images[i].numpy().transpose(1, 2, 0) # converting from [channels, height, width] to [height, width, channels]
        # denormalization -> from [-1, 1] back to [0, 1]
        img = img*0.5 + 0.5
        # values stay in b/w the range of 0 and 1
        img = np.clip(img, 0, 1)

        # display the image
        axes[i].imshow(img)
        axes[i].axis('off')

        # get labels
        true_label = CIFAR_CLASSES[labels[i]]
        predicted_label = CIFAR_CLASSES[predicted[i]]

        # if the label is same as predicted; color = green else red
        color = 'green' if labels[i] == predicted[i] else 'red'

        axes[i].set_title(f"true : {true_label}\n predicted : {predicted_label}", color=color, fontsize=10)

    plt.tight_layout()  
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    print("predictions saved")
    plt.close()

# saving the trained model object
def save_model(model, path='model.pth'):
    # returns a dict of all learned weights and biases
    torch.save(model.state_dict(), path)
    print(f"model saved to {path}")

# loading the weights and biases
def load_model(model, path='model.pth'):
    model.load_state_dict(torch.load(path))
    # disabling dropout and fixes batch_norm
    model.eval()
    print(f"model loaded from {path}")
    return model