import torch
import torch.nn as nn

# f = w * x
# f = 2 * x

x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

x_test = torch.tensor([[5]], dtype=torch.float32)

# # initial weight
# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

n_samples, n_features = x.shape
input_size = n_features
output_size = n_features


class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


model = MyModel(input_size, output_size)
print(f"Prediction before training: f(5) = {model(x_test).item():.3f}")


# use nn loss func & optimizer
learning_rate = 0.01
n_iters = 10
loss = nn.MSELoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
)

# training
for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(x)

    # loss
    l = loss(y, y_pred)

    # gradients = backward pass
    # dl / dw
    l.backward()

    # update weights
    optimizer.step()

    # zero the gradients after updating
    optimizer.zero_grad()

    if epoch % 1 == 0:
        [w, b] = model.parameters()
        print(f"epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}")


print(f"Prediction after training: f(5) = {model(x_test).item():.3f}")
