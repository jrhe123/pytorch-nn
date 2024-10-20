import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/mnist")

device = torch.device("mps")

# hyperparameters
input_size = 784  # 28x28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)
test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
)

examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(samples[i][0], cmap="gray")
    plt.title("Label: " + str(labels[i].item()))
    plt.xticks([])
    plt.yticks([])

# plt.show()
img_grid = torchvision.utils.make_grid(samples)
writer.add_image("MNIST Images", img_grid)
# writer.close()
# sys.exit()


class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MyModel, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output = self.l1(x)
        output = self.relu(output)
        output = self.l2(output)
        return output


model = MyModel(
    input_size,
    hidden_size,
    num_classes,
).to(device)

# apply softmax here
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)
writer.add_graph(model, samples.reshape(-1, 28 * 28))

# train loop
n_total_steps = len(train_loader)
running_loss = 0.0
running_correct = 0

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 100, 1, 28, 28
        # =>
        # 100, 784
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # update
        optimizer.step()

        # metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()

        # logging
        if (i + 1) % 100 == 0:
            writer.add_scalar(
                "traning loss",
                running_loss / 100,
                epoch * n_total_steps + i,
            )
            writer.add_scalar(
                "accuracy",
                running_correct / 100,
                epoch * n_total_steps + i,
            )
            running_loss = 0.0
            running_correct = 0

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}"
            )

# test loop
labels_list = []
preds = []
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum()

        class_prediciton = [F.softmax(output, dim=0) for output in outputs]
        preds.append(class_prediciton)
        labels_list.append(predicted)

    preds = torch.cat([torch.stack(batch) for batch in preds])
    labels_list = torch.cat(labels_list)

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy of the model on the 10000 test images: {acc}%")

    classes = range(10)
    for i in classes:
        labels_i = labels == i
        preds_i = preds[:, i]
        writer.add_pr_curve(
            str(i),
            labels_i,
            preds_i,
            global_step=0,
        )

        print(f"Class {i}")
        print(f"Confidence: {preds_i[labels_i].mean():.4f}")
        print(f"True: {labels_i.sum().item()}")
        print(f"False: {labels_i.size(0) - labels_i.sum().item()}")

# save model
# torch.save(model.state_dict(), "./models/model.ckpt")

writer.close()
