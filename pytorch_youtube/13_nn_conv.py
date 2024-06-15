# https://pytorch.org/docs/stable/nn.html
# 1. forward
# 2. backward

# step: 卷积 -> 非线性 -> 卷积 -> 非线性
# 卷积: conv1 / conv2
# 非线性: relu / sigmoid

import torch
import torch.nn.functional as F

input = torch.tensor(
    [
        [1, 2, 0, 3, 1],
        [0, 1, 2, 3, 1],
        [1, 2, 1, 0, 0],
        [5, 2, 3, 1, 1],
        [2, 1, 0, 1, 1],
    ]
)

kernel = torch.tensor(
    [
        [1, 2, 1],
        [0, 1, 0],
        [2, 1, 0],
    ]
)

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(input.shape)
print(kernel.shape)

# stride: each step move
output = F.conv2d(input, kernel, stride=1)
print(output)
