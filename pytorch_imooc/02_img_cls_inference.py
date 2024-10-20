import cv2
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data_utils
import torchvision
import torchvision.datasets as dataset

# data
test_data = dataset.MNIST(
    root="./data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

# batch size
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


model = torch.load("mnist_model.pkl")

# eval / test
with torch.no_grad():
    loss_test = 0
    accuracy = 0
    for i, (images, labels) in enumerate(test_loader):
        outputs = model(images)

        _, pred = outputs.max(1)
        num_correct = (pred == labels).sum()
        accuracy += num_correct.item()

        for idx in range(images.shape[0]):
            img = images.numpy()[idx].reshape(28, 28)
            img = img.transpose(1, 2, 0)
            label = labels.numpy()[idx]
            pred_label = pred.numpy()[idx]
            print(
                "prediction: {}, label: {}, accuracy: {}".format(
                    pred_label,
                    label,
                    "True" if pred_label == label else "False",
                )
            )
            cv2.imshow("image", img)
            cv2.waitKey(0)

    accuracy = accuracy / len(test_data)
    loss_test = loss_test / (len(test_loader) // 64)

    print(
        "accuracy is {}, loss is {}".format(
            accuracy,
            loss_test.item(),
        )
    )
