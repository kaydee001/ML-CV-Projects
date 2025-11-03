import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class BuildingDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        all_files = os.listdir(data_dir)
        self.image_files = [f for f in all_files if f.endswith("_sat.jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        mask_name = image_name.replace("_sat.jpg", "_mask.png")

        img_path = os.path.join(self.data_dir, image_name)
        mask_path = os.path.join(self.data_dir, mask_name)

        image = Image.open(img_path)
        mask = Image.open(mask_path)

        image = np.array(image)
        mask = np.array(mask)

        return image, mask