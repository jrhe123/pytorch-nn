import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets

# 0. prepare data
X_numpy, y_numpy = datasets.make_regression(
    n_samples=100,
    n_features=1,
    noise=20,
    random_state=1,
)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
# reshape y shape
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# 1. model
input_size = n_features
output_size = 1
model = nn.Linear(
    input_size,
    output_size,
)

# 2. loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
)

# 3. training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # forward pass and loss
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # backward pass
    loss.backward()

    # update
    optimizer.step()

    # zero the gradients
    optimizer.zero_grad()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")

# 4. plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, "ro", label="Real data")
plt.plot(X_numpy, predicted, "b-", label="Predicted data")
plt.legend()
plt.show()
