import re

import numpy as np
import torch

with open("data/BostonHousing.csv", "r") as f:
    lines = f.readlines()

data = []
for item in lines[1:]:
    cleaned_line = re.sub(r"\s{2,}", " ", item).strip()
    cleaned_line = cleaned_line.replace('"', "")
    data.append(cleaned_line.split(","))

data = np.array(data).astype(np.float32)
print(data.shape)

Y = data[:, -1]
x = data[:, 0:-1]

Y_train = Y[0:496, ...]
x_train = x[0:496, ...]

Y_test = Y[496:, ...]
x_test = x[496:, ...]


class MyModel(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(MyModel, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, 100)
        self.predict = torch.nn.Linear(100, n_output)

    def forward(self, x):
        out = self.hidden(x)
        out = torch.relu(out)
        out = self.predict(out)
        return out


# load model
model = MyModel(13, 1)
model.load_state_dict(torch.load("model.pkl"))
# loss func
loss_func = torch.nn.MSELoss()

model.eval()
with torch.no_grad():
    x_data = torch.tensor(x_test, dtype=torch.float32)
    Y_data = torch.tensor(Y_test, dtype=torch.float32)

    pred = model.forward(x_data)
    pred = torch.squeeze(pred)
    loss = loss_func(pred, Y_data) * 0.001
    print("Loss_test: {}".format(loss.item()))
