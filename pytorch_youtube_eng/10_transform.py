import math

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset


class WineDataset(Dataset):
    def __init__(self, transform=None):
        xy = np.loadtxt(
            "./data/wine/wine.csv",
            dtype=np.float32,
            delimiter=",",
            skiprows=1,
        )
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]  # label
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


class MyToTensor:
    def __call__(self, sample):
        x, y = sample
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return x, y


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target


# dataset = WineDataset(transform=MyToTensor())
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)


composed = torchvision.transforms.Compose(
    [
        MyToTensor(),
        MulTransform(2),
    ]
)
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features, labels)
