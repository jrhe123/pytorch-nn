import numpy as np
import torch

# 2 dimensions
x = torch.empty(2, 2)
print(x)

x = torch.rand(2, 2)
print(x)

x = torch.zeros(2, 2)
print(x)

x = torch.ones(2, 2)
print(x)

x = torch.ones(2, 2, dtype=torch.float32)
print(x)
print(x.size())

x = torch.tensor([2.5, 1.0])
print(x)

x = torch.tensor([2, 2])
y = torch.tensor([2, 2])
z = x + y
z = torch.add(x, y)
x.add_(y)
print(z)

z = x * y
z = torch.mul(x, y)
x.mul_(y)
print(z)

x = torch.rand(5, 3)
print(x)
print(x[:, :])
# column 0
print(x[:, 0])
# row 0
print(x[0, :])


print(x[1, 1])
print(x[1, 1].item())

# resize
x = torch.rand(4, 4)
# 2 x 8
a = x.view(-1, 8)
print(a)


a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

c = torch.from_numpy(b)
print(c)
