import torch
import albumentations as A
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import Flowers102
from albumentations.pytorch import ToTensorV2

# wrapper to use albumentations with torchvision datasets
class AlbumentationTransforms:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = np.array(img) # convert PIL to numpy
        augmented = self.transform(image=img)
        return augmented['image']

# training augmentations -> realisitc variations
train_transforms = A.Compose([A.Resize(256, 256),
           A.CenterCrop(224, 224), # resnet input size
           A.HorizontalFlip(p=0.5), # 50% chance of flip
           A.Rotate(limit=15, p=0.5), # random rotation -> ± 15 deg
           A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0, p=0.3),
           A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7), # lighting variations
           A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # imagenet normalization
           ToTensorV2()])

# validation transforms -> no augmentations, only preprocessing
val_transforms = A.Compose([A.Resize(256, 256),
                            A.CenterCrop(224, 224),
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ToTensorV2()])

# weight calculation for imbalanced classes
def get_class_weights(dataset):
    class_counts = {}

    # counting samples per class
    for _, label in dataset:
        class_counts[label] = class_counts.get(label, 0) + 1

    total_samples = len(dataset)
    num_classes = len(class_counts)

    # weight = total_samples / (num_classes*class_count)
    weights = []
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 1)
        weight = total_samples / (num_classes*count)
        weights.append(weight)

    return torch.FloatTensor(weights)

# creating train/val/test loaders with augmentations and class weights
def get_data_loaders(batch_size=32, test_batch_size=128):

    # wrap transforms
    train_transform = AlbumentationTransforms(train_transforms)
    val_transform = AlbumentationTransforms(val_transforms)

    # loading Flower-102 dataset
    full_train = Flowers102(root="data", split="train", download=True, transform=train_transform)

    # 80-20 split -> for train and validation
    train_size = int(0.8*(len(full_train)))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
    
    # test set -> completely separate
    test_dataset = Flowers102(root="data", split="test", download=True, transform=val_transform)

    # calculating class weights for imbalanced training
    class_weights = get_class_weights(train_dataset)

    # creating data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_weights