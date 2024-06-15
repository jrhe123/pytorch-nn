import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Use ReLU in the hidden layers of deep neural networks for faster and more efficient training. It's a good general-purpose activation function for most hidden layers.

# Use Sigmoid in the output layer when you are dealing with binary classification problems and need the output to represent probabilities between 0 and 1.

input = torch.tensor(
    [
        [1, -0.5],
        [-1, 3],
    ],
    dtype=float,
)
input = torch.reshape(input, (-1, 1, 2, 2))

# -1: calculate this dimension automatically.
# 1: new tensor will have 1 channel or feature map
# 2, 2: 2x2 matrix

print(input)


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.relu1 = torch.nn.ReLU(inplace=False)
        self.sigmod1 = torch.nn.Sigmoid()

    def forward(self, x):
        # output = self.relu1(x)
        output = self.sigmod1(x)
        return output


model = MyModel()
output = model(input)
print("out: ", output)


dataset = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
dataloader = DataLoader(dataset, batch_size=64)
writer = SummaryWriter("p16")

step = 0
for data in dataloader:
    imgs, targets = data
    output = model(imgs)

    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()


# python3 xxx.py
# tensorboard --logdir=xxx --port=6006
