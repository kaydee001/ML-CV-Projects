import torch
import matplotlib.pyplot as plt
from torchvision.datasets import Flowers102
from data import train_transforms, AlbumentationTransforms
import numpy as np

def visualize_augmentations(num_samples=9):
    dataset = Flowers102(root="data", split="train", download=True)
    img, label = dataset[0]

    img_np = np.array(img)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    axes[0].imshow(img_np)
    axes[0].set_title("original image")
    axes[0].axis("off")

    for i in range(1, num_samples):
        augmented = train_transforms(image=img_np)
        aug_img = augmented['image']

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        aug_img = aug_img * std + mean
        aug_img = torch.clamp(aug_img, 0, 1)
        aug_img = aug_img.permute(1, 2, 0).numpy()

        axes[i].imshow(aug_img)
        axes[i].set_title(f"augmented {i}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("augmentation_samples.png")
    print("augmentation samples image saved")

if __name__ == "__main__":
    visualize_augmentations()