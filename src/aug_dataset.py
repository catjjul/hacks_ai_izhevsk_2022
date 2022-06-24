from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch
from src.config import DATA_PATH


class AugImageDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data_df = data_df
        self.transform = transform

    def __getitem__(self, idx):
        image_name, label = self.data_df.iloc[idx]['img_num'], self.data_df.iloc[idx]['number_of_houses']

        image = Image.open(f"{DATA_PATH}/train/{image_name}")
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)['image']

        return image, torch.tensor(label).long()

    def __len__(self):
        return len(self.data_df)
