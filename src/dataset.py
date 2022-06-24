from PIL import Image
from torch.utils.data import Dataset
import torch
from src.config import DATA_PATH


class ImageDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data_df = data_df
        self.transform = transform

    def __getitem__(self, idx):
        image_name, label = self.data_df.iloc[idx]['img_num'], self.data_df.iloc[idx]['number_of_houses']
        image = Image.open(f"{DATA_PATH}/train/{image_name}")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label).long()

    def __len__(self):
        return len(self.data_df)


class TestImageDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data_df = data_df
        self.transform = transform

    def __getitem__(self, idx):
        image_name = self.data_df.iloc[idx]['img_num']
        image = Image.open(f"{DATA_PATH}/test/{image_name}")

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.data_df)
