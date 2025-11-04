import os
import numpy as np
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

class BuildingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        all_files = os.listdir(data_dir)
        self.image_files = [f for f in all_files if f.endswith("_sat.jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        mask_name = image_name.replace("_sat.jpg", "_mask.png")

        img_path = os.path.join(self.data_dir, image_name)
        mask_path = os.path.join(self.data_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = np.array(image)
        mask = np.array(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask
    
train_transforms = A.Compose([A.Resize(512, 512),
                              A.HorizontalFlip(p=0.5),
                              A.VerticalFlip(p=0.5),
                              A.RandomRotate90(p=0.5),
                              A.RandomBrightnessContrast(p=0.2),
                              A.Normalize(mean=[0.455, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                              A.ToTensorV2()])

val_transforms = A.Compose([A.Resize(512, 512),
                            A.Normalize(mean=[0.455, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            A.ToTensorV2()])