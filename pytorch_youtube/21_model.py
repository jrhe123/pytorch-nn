import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# METHOD 1: save model
vgg16 = torchvision.models.vgg16(
    pretrained=False,
)
torch.save(vgg16, "vgg16_model.pth")

# load model
vgg16_2 = torch.load("vgg16_model.pth")


# METHOD 2: save state dict
torch.save(vgg16.state_dict(), "vgg16_state_dict.pth")

# load state dict
vgg16_3 = torchvision.models.vgg16(
    pretrained=False,
).load_state_dict(torch.load("vgg16_state_dict.pth"))
