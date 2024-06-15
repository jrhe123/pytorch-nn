# https://pytorch.org/docs/stable/nn.html
# 1. forward
# 2. backward

# 数据降维
# 下采样: Max_pool2d


# 上采样: Max_uppool2d


import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor(
    [
        [1, 2, 0, 3, 1],
        [0, 1, 2, 3, 1],
        [1, 2, 1, 0, 0],
        [5, 2, 3, 1, 1],
        [2, 1, 0, 1, 1],
    ],
    dtype=torch.float32,
)
input = torch.reshape(input, (-1, 1, 5, 5))
# print(input.shape)


dataset = torchvision.datasets.CIFAR10(
    "./dataset",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
dataloader = DataLoader(dataset, batch_size=64)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3,
            ceil_mode=True,
        )

    def forward(self, input):
        output = self.maxpool1(input)
        return output


model = MyModel()
# output = model(input)
# print(output)

writer = SummaryWriter("p15")
step = 0
for data in dataloader:
    imgs, targets = data
    outputs = model(imgs)

    writer.add_images("input", imgs, step)
    writer.add_images("outputs", outputs, step)

    step = step + 1


writer.close()

# python3 15_max_pool.py
# tensorboard --logdir=p15 --port=6006
