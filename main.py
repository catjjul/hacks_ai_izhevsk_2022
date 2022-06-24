import random

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset

from torchvision import models

from src.config import DEVICE, DATA_PATH, BATCH_SIZE, RANDOM_SEED
from src.dataset import ImageDataset, TestImageDataset
from src.aug_dataset import AugImageDataset
from src.train_pipeline import train
from src.transform import train_transform, valid_transform
from src.augment import train_augmentation


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_prediction(trained_model):
    test_df = pd.read_csv(DATA_PATH + 'sample_solution.csv')
    test_df = test_df.drop(["number_of_houses"], axis=1)
    test_dataset = TestImageDataset(test_df, valid_transform)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=1,
                                 pin_memory=True,
                                 num_workers=2)

    predicted_labels = []
    with torch.no_grad():
        for images in test_dataloader:
            images = images.to(DEVICE)
            output = trained_model(images)
            predicted_labels.append(torch.argmax(output).item() + 1)

    sub = pd.read_csv(DATA_PATH + 'sample_solution.csv')
    sub['number_of_houses'] = predicted_labels
    sub.to_csv('sub.csv', index=False)


seed_everything(RANDOM_SEED)

# read csv and print some pictures
train_df = pd.read_csv(DATA_PATH + 'train.csv')

# clean data for cuda work
train_df.number_of_houses = train_df.number_of_houses - 1
train_df = train_df[train_df.number_of_houses < 25]
n_classes = train_df.number_of_houses.nunique()  # 25

train_data = ImageDataset(train_df, train_transform)

train_aug_data = AugImageDataset(train_df, train_augmentation)
train_data = ConcatDataset([train_data, train_aug_data])

train_dataset, val_dataset = train_test_split(train_data, test_size=0.2, random_state=RANDOM_SEED)
print(f'Training sample size: {len(train_dataset)}'
      f'Validation sample size: {len(val_dataset)}\n')

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

model = models.resnet50(pretrained=True)

for name, child in model.named_children():
    if name in ['layer3', 'layer4', 'fc']:
        print(name + ' has been unfrozen')
        for param in child.parameters():
            param.requires_grad = True
    else:
        for param in child.parameters():
            param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(p=0.6),
    nn.Linear(512, n_classes)
)
model.fc.requires_grad = True

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=0.001)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                          max_lr=1e-3, epochs=10, steps_per_epoch=len(train_dataloader))

model, history = train(
    model, criterion, optimizer,
    train_dataloader, val_dataloader,
    num_epochs=12
)

make_prediction(model)
