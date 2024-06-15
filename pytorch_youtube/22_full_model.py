import time

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# check mps
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print(
            "MPS not available because the current PyTorch install was not "
            "built with MPS enabled."
        )
    else:
        print(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine."
        )

# mac m1/2 chip
mps_device = torch.device("mps")

train_data = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
test_data = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("train size: {}".format(train_data_size))
print("test size: {}".format(test_data_size))

# data loader
train_data_loader = DataLoader(
    train_data,
    batch_size=64,
)
test_data_loader = DataLoader(
    test_data,
    batch_size=64,
)


# build nn
class Full_Model(torch.nn.Module):
    def __init__(self):
        super(Full_Model, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, 1, 2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 5, 1, 2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 5, 1, 2),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(1024, 64),
            torch.nn.Linear(64, 10),
        )

    def forward(self, x):
        output = self.model(x)
        return output


model = Full_Model()
# mps
model = model.to(mps_device)

# loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
# mps
loss_fn = loss_fn.to(mps_device)

learning_rate = 0.01
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
)

total_train_step = 0
total_test_step = 0
epoch = 10

# tensorboard
writer = SummaryWriter("p22")


for i in range(epoch):
    # record time elapse
    start_time = time.time()
    print("--------Epoch {} starting--------".format(i + 1))

    # start training
    model.train()
    for data in train_data_loader:
        imgs, targets = data
        # mps
        imgs = imgs.to(mps_device)
        targets = targets.to(mps_device)

        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1

        if total_train_step % 100 == 0:
            print(
                "total_train_step: {}, loss: {}".format(total_train_step, loss.item())
            )
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # start evaluation
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_data_loader:
            imgs, targets = data
            # mps
            imgs = imgs.to(mps_device)
            targets = targets.to(mps_device)

            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            total_test_loss = total_test_loss + loss

            # 整体正确的个数
            # argmax(1)
            # 1: 横向
            # 0: 纵向
            accuracy = outputs.argmax(1).eq(targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("total_test_loss: {}".format(total_test_loss))
    print("total_accuracy: {}".format(total_accuracy / test_data_size))

    writer.add_scalar(
        "test_loss",
        total_test_loss,
        total_test_step,
    )
    writer.add_scalar(
        "test_accuracy",
        total_accuracy / test_data_size,
        total_test_step,
    )
    total_test_step = total_test_step + 1

    torch.save(model, "full_{}_model.pth".format(i))
    print("model saved")

    end_time = time.time()
    print("time spent: ", end_time - start_time)


writer.close()
