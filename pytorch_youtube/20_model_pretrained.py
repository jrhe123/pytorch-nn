import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# train_data = torchvision.datasets.ImageNet(
#     root="./dataset_image_net",
#     split="train",
#     transform=torchvision.transforms.ToTensor(),
#     download=True,
# )

# vgg16_false = torchvision.models.vgg16(
#     pretrained=False,
# )
vgg16_true = torchvision.models.vgg16(
    pretrained=True,
)

train_data = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

# add addition layer: 1000 -> 10
vgg16_true.classifier.add_module("add_linear", torch.nn.Linear(1000, 10))
print(vgg16_true)

# replace existing layer
vgg16_true.classifier[6] = torch.nn.Linear(4096, 10)
print(vgg16_true)
