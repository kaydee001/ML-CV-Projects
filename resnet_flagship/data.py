import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import Flowers102

train_transforms = transforms.Compose([transforms.Resize(256), 
                                       transforms.CenterCrop(224), 
                                       transforms.ToTensor(), 
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def get_class_weights(dataset):
    class_counts = {}

    for _, label in dataset:
        class_counts[label] = class_counts.get(label, 0) + 1

    total_samples = len(dataset)
    num_classes = len(class_counts)

    weights = []
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 1)
        weight = total_samples / (num_classes*count)
        weights.append(weight)

    return torch.FloatTensor(weights)

def get_data_loaders(batch_size=32):

    full_train = Flowers102(root="data", split="train", download=True, transform=train_transforms)

    train_size = int(0.8*(len(full_train)))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
    
    test_dataset = Flowers102(root="data", split="test", download=True, transform=train_transforms)

    class_weights = get_class_weights(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_weights