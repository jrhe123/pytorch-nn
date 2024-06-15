import math

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset


class WineDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt(
            "./data/wine/wine.csv",
            dtype=np.float32,
            delimiter=",",
            skiprows=1,
        )
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])  # label
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset = WineDataset()
dataloader = DataLoader(
    dataset=dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
)

# training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)
print(f"Total samples: {total_samples}")
print(f"Iterations per epoch: {n_iterations}")

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward, backward, update
        if (i + 1) % 5 == 0:
            print(f"epoch {epoch+1} iter {i+1}")


# dataloader with custom dataset
