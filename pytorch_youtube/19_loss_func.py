import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 反向传播: grad

# loss / mse
inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([4, 5, 6], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = torch.nn.L1Loss()
result = loss(inputs, targets)
print(result)

loss_mse = torch.nn.MSELoss()
result2 = loss_mse(inputs, targets)
print(result2)


# classification: person / dog / cat
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = torch.nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)


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


dataset = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
dataloader = DataLoader(dataset, batch_size=1)


loss_cross_2 = torch.nn.CrossEntropyLoss()
model = MyModel()
# 优化器
optim = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optim,
    step_size=5,
    gamma=0.1,
)

# every 5 steps, we update the lr rate
# 0.01 -> 0.001 -> 0.0001

for epcho in range(20):
    running_loss = 0.0

    for data in dataloader:
        imgs, targets = data
        output = model(imgs)
        result_loss = loss_cross_2(output, targets)

        # 反向传播
        optim.zero_grad()
        result_loss.backward()
        scheduler.step()  # update lr rate with scheduler

        running_loss = running_loss + result_loss

    print("running_loss: ", running_loss)


# python3 xxx.py
# tensorboard --logdir=xxx --port=6006
