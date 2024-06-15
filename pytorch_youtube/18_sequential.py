import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# build CIFAR10 network


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # given: (see pic cifar 10)
        # channel: 3
        # w / h: 32
        # kernel_size: 5
        # dilation: 1 (default)

        # calculate
        # stride: 1
        # padding: 2
        self.conv1 = torch.nn.Conv2d(3, 32, 5, padding=2)
        # maxpool: 2
        self.maxpool1 = torch.nn.MaxPool2d(2)

        self.conv2 = torch.nn.Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = torch.nn.MaxPool2d(2)

        self.conv3 = torch.nn.Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = torch.nn.MaxPool2d(2)

        # flatten
        self.flatten = torch.nn.Flatten()

        # linear
        self.linear1 = torch.nn.Linear(1024, 64)
        # classification 10 different types
        self.linear2 = torch.nn.Linear(64, 10)

        self.model1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 5, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 5, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(1024, 64),
            torch.nn.Linear(64, 10),
        )

    def forward(self, x):
        output = self.model1(x)
        return output


model = MyModel()
input = torch.ones((64, 3, 32, 32))
output = model(input)

print(output.shape)

writer = SummaryWriter("p18")
writer.add_graph(model, input)
writer.close()

# python3 xxx.py
# tensorboard --logdir=xxx --port=6006
