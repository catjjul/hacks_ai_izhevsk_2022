import random

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from torchvision import models

from src.config import DEVICE, DATA_PATH, BATCH_SIZE, RANDOM_SEED
from src.dataset import ImageDataset, TestImageDataset
from src.aug_dataset import AugImageDataset
from src.train_pipeline import train
from src.transform import train_transform, valid_transform


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def print_one_and_two_random_buildings(df: pd.DataFrame):
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Одно здание {" " * 105} Два здания', fontsize=14)

    for i, n in zip(range(4), df[df['number_of_houses'] == 1].sample(4, random_state=42)['img_num']):
        axs[i // 2, (i % 2)].imshow(plt.imread(f"{DATA_PATH}/train/{n}"))
        axs[i // 2, (i % 2)].axis('off')

    for i, n in zip(range(4), df[df['number_of_houses'] == 2].sample(4, random_state=42)['img_num']):
        axs[i // 2, (i % 2) + 2].imshow(plt.imread(f"{DATA_PATH}/train/{n}"))
        axs[i // 2, (i % 2) + 2].axis('off')

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)


def print_random_pictures(data: torch.utils.data.Dataset):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, label = data[sample_idx]
        inp = img.numpy().transpose((1, 2, 0))
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(inp)
    plt.show()


def print_classification_report(trained_model, dataloader: torch.utils.data.DataLoader):
    real = []
    pred = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, y = batch
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_pred = trained_model(x)
            y_true = y.detach().cpu().numpy().tolist()
            y_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1).tolist()
            real.extend(y_true)
            pred.extend(y_pred)
    print(classification_report(real, pred))


def prediction(trained_model):
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
print_one_and_two_random_buildings(train_df)

# clean data for cuda work
train_df.number_of_houses = train_df.number_of_houses - 1
train_df = train_df[train_df.number_of_houses < 25]
n_classes = train_df.number_of_houses.nunique()  # 25

train_data = ImageDataset(train_df, train_transform)
print_random_pictures(train_data)

train_aug_data = AugImageDataset(train_df, train_transform)
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
optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=0.001)

model, history = train(
        model, criterion, optimizer,
        train_dataloader, val_dataloader,
        num_epochs=12
    )

print_classification_report(model, val_dataloader)
prediction(model)
