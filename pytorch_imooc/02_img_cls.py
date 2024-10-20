import torch
import torch.utils.data as data_utils
import torchvision
import torchvision.datasets as dataset

# data
train_data = dataset.MNIST(
    root="./data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
test_data = dataset.MNIST(
    root="./data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

# batch size
train_loader = data_utils.DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True,
)
test_loader = data_utils.DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=False,
)


# model
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.fc = torch.nn.Linear(
            14 * 14 * 32,
            10,
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out


model = MyModel()

# loss and optimizer
loss_func = torch.nn.CrossEntropyLoss()
opimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
)

# train
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = loss_func(outputs, labels)

        # BP
        opimizer.zero_grad()
        loss.backward()
        opimizer.step()

        if (i + 1) % 100 == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                    epoch + 1,
                    10,
                    i + 1,
                    len(train_loader),
                    loss.item(),
                )
            )

    # eval / test
    model.eval()
    with torch.no_grad():
        loss_test = 0
        accuracy = 0
        for i, (images, labels) in enumerate(test_loader):
            outputs = model(images)
            loss_test += loss_func(outputs, labels)

            _, pred = outputs.max(1)
            num_correct = (pred == labels).sum()
            accuracy += num_correct.item()

        accuracy = accuracy / len(test_data)
        loss_test = loss_test / (len(test_loader) // 64)

        print(
            "epcho is {}, accuracy is {}, loss is {}".format(
                epoch + 1,
                accuracy,
                loss_test.item(),
            )
        )

# save model
torch.save(model, "mnist_model.pkl")
