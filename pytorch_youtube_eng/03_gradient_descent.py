import torch

# 3 random values
x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
print(y)

z = y * y * 2
print(z)

z = z.mean()
print(z)

z.backward()
print("grad: ", x.grad)


# x.requires_grad(False)
y = x.detach()
print(y)

with torch.no_grad():
    y = x + 2
    print(y)


weights = torch.ones(4, requires_grad=True)
optimizer = torch.optim.SGD(weights, lr=0.01)

for epoch in range(1):
    model_output = (weights * 3).sum()
    model_output.backward()

    print(weights.grad)

    weights.grad.zero_()
