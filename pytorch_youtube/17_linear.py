import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# k * x + b
# k: weight
# b: bias


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = torch.nn.Linear(196608, 10)

    def forward(self, x):
        output = self.linear1(x)
        return output


dataset = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
dataloader = DataLoader(dataset, batch_size=64)
writer = SummaryWriter("p17")


model = MyModel()

step = 0
for data in dataloader:
    imgs, targets = data
    output = torch.reshape(imgs, (1, 1, 1, -1))
    output = model(output)

    step = step + 1

writer.close()

# python3 xxx.py
# tensorboard --logdir=xxx --port=6006
