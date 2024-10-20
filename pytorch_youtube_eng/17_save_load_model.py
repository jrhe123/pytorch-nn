import torch
import torch.nn as nn


class MyModel(nn.module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


model = MyModel()
for param in model.parameters():
    print(param)

FILE = "model.pth"
torch.save(model.state_dict(), FILE)

# Load the model
load_model = MyModel()
load_model.load_state_dict(torch.load(FILE))
load_model.eval()

for param in load_model.parameters():
    print(param)


learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(optimizer.state_dict())

# checkpoint = {
#     "epoch": 99,
#     "model_state": model.state_dict(),
#     "optimizer_state": optimizer.state_dict(),
#     "loss": 1.0,
# }
# torch.save(checkpoint, "checkpoint.pth")

loaded_checkpoint = torch.load("checkpoint.pth")
epoch = loaded_checkpoint["epoch"]
model = MyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0)
model.load_state_dict(loaded_checkpoint["model_state"])
optimizer.load_state_dict(loaded_checkpoint["optimizer_state"])

print("optimizer state dict", optimizer.state_dict())
