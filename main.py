import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm
"""
print('System Version:', sys.version)
print('PyTorch version', torch.__version__)
print('Torchvision version', torchvision.__version__)
print('Numpy version', np.__version__)
print('Pandas version', pd.__version__)
"""


class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes


dataset = PlayingCardDataset(data_dir='archive/train')
print(len(dataset))
image, label = dataset[6000]
print(label)
print(image)
data_dir = 'archive/train'
target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}
print(target_to_class)

transform = transforms.Compose([transforms.Resize((128, 128)),transforms.ToTensor()])

data_dir = 'archive/train'
dataset = PlayingCardDataset(data_dir, transform)

image, label = dataset[100]
print(image.shape)

for images, label in dataset:
    print("through dataset :)")
    break

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for image, labels in dataset:
    print("through dataset :)")
    break

print(f"images.shape --->  {images.shape} labels.shape --> {labels.shape}")


