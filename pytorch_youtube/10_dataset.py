import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
    ]
)

train_set = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=True,
    download=True,
    transform=dataset_transform,
)
test_set = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=False,
    download=True,
    transform=dataset_transform,
)

# print(test_set[0])
# <PIL.Image.Image image mode=RGB size=32x32 at 0x13C37D890>, 3

# img, target = test_set[0]
# print(img)
# print(target)
# img.show()

# tensor
# print(test_set[0])
writer = SummaryWriter("p10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()


# python3 10_dataset.py
# tensorboard --logdir=p10 --port=6006
