import numpy as np
import torch
import torch.nn as nn

#
#        -> 2.0              -> 0.65
# Linear -> 1.0  -> Softmax  -> 0.25   -> CrossEntropy(y, y_hat)
#        -> 0.1              -> 0.1
#
#     scores(logits)      probabilities
#                           sum = 1.0
#


# def cross_entropy(actual, prediction):
#     loss = -np.sum(actual * np.log(prediction))
#     return loss

# # y must be one hot encoded
# # if class 0: [1,0,0]
# # if class 1: [0,1,0]
# # if class 2: [0,0,1]
# y = np.array([1, 0, 0])

# y_pred_good = np.array([0.7, 0.2, 0.1])
# y_pred_bad = np.array([0.1, 0.3, 0.6])

# l1 = cross_entropy(y, y_pred_good)
# l2 = cross_entropy(y, y_pred_bad)

# print(l1, l2)


# ====================

# CrossEntropyLoss = LogSoftmax + NullLoss
loss = nn.CrossEntropyLoss()

# y has class labels, not one-hot
Y = torch.tensor([0])

# nsamples * nclasses = 1 x 3
# class 0: 2.0
# class 1: 1.0
# class 2: 0.1
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])

# class 0: 0.5
# class 1: 2.0
# class 2: 0.3
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(f"PyTorch Loss1: {l1.item():.4f}")
print(f"PyTorch Loss2: {l2.item():.4f}")


# get predictions
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(
    f"Actual class: {Y.item()}, Y_pred1: {predictions1.item()}, Y_pred2: {predictions2.item()}"
)
