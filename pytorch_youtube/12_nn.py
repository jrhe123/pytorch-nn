# https://pytorch.org/docs/stable/nn.html
# 1. forward
# 2. backward

# step: 卷积 -> 非线性 -> 卷积 -> 非线性
# 卷积: conv1 / conv2
# 非线性: relu / sigmoid

import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


model = MyModel()
x = torch.tensor(1.0)
output = model(x)
print(output)
