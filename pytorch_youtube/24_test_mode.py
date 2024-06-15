import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

mps_device = torch.device("mps")

image_path = "./imgs/dog.png"
# Convert image to RGB to ensure 3 channels
image = Image.open(image_path).convert("RGB")

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
    ]
)
image = transform(image)


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


image = torch.reshape(image, (1, 3, 32, 32))
model = torch.load("./full_9_model.pth")

model = model.to(mps_device)
image = image.to(mps_device)

model.eval()
with torch.no_grad():
    output = model(image)

print(output)
print(output.argmax(1))

# [0]: airplane
# [1]: automobile
# [2]: bird
# [3]: cat
# [4]: deer
# [5]: dog
# [6]: frog
# [7]: horse
# [8]: ship
# [9]: truck

# tensor(
#     [
#         [
#             -5.5299,
#             -2.9782,
#             -0.3614,
#             4.0862,
#             -0.8710,
#             5.1698,
#             1.6638,
#             1.6666,
#             -0.5850,
#             -0.4703,
#         ]
#     ],
#     device="mps:0",
# )
# tensor([5], device="mps:0")
