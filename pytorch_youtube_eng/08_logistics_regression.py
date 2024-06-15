import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 0. prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1234,
)
# scale (recommended for logistics regression, standardization)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# reshape
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# 1. model
# f = w * x + b, sigmoid at the end
class MyModel(nn.Module):
    def __init__(self, input_size):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


input_size = n_features
model = MyModel(input_size)

# 2. loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
)

# 3. training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # forward pass and loss
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # backward pass
    loss.backward()

    # update
    optimizer.step()

    # zero the gradients
    optimizer.zero_grad()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")

# 4. evaluation
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()

    accuracy = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f"Accuracy: {accuracy:.2f}")

# plot
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test.numpy(),
    cmap="viridis",
)
