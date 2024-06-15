import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
    ]
)
test_set = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=False,
    download=True,
    transform=dataset_transform,
)
test_loader = DataLoader(
    dataset=test_set,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    drop_last=False,
)

# img, target = test_set[0]
# print(img.shape)
# print(target)

writer = SummaryWriter("p11")
for epoch in range(2):
    step = 0
    for data in test_loader:
        # batch is 4, so 4 images in one data loaded
        imgs, target = data
        writer.add_images("test_img", imgs, step)
        step = step + 1

writer.close()
