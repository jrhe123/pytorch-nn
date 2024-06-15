import numpy as np
import torch
import torch.nn as nn


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print("softmax numpy: ", outputs)


x = torch.tensor([2.0, 1.0, 0.1])
outputs = nn.functional.softmax(x, dim=0)
print("softmax torch: ", outputs)
