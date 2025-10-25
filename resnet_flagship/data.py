from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import Flowers102

train_transforms = transforms.Compose([transforms.Resize(256), 
                                       transforms.CenterCrop(224), 
                                       transforms.ToTensor(), 
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def get_data_loaders(batch_size=32):
    train_dataset = Flowers102(root="data", split="train", download=True, transform=train_transforms)
    test_dataset = Flowers102(root="data", split="test", download=True, transform=train_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader