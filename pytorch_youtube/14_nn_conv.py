# https://pytorch.org/docs/stable/nn.html
# 1. forward
# 2. backward

# step: 卷积 -> 非线性 -> 卷积 -> 非线性
# 卷积: conv1 / conv2
# 非线性: relu / sigmoid

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(
    root="./dataset",
    transform=torchvision.transforms.ToTensor(),
    train=False,
    download=True,
)
dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
)


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


model = MyModel()
print(model)

writer = SummaryWriter("./p14")
step = 0
for data in dataloader:
    imgs, targets = data
    outputs = model(imgs)

    # [64, 3, 32, 32]
    print(imgs.shape)
    # [64, 6, 30, 30]
    print(outputs.shape)

    outputs = torch.reshape(outputs, (-1, 3, 30, 30))

    writer.add_images("input", imgs, step)
    writer.add_images("outputs", outputs, step)

    step = step + 1

writer.close()
