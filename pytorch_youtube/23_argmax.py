import torch

outputs = torch.tensor(
    [
        [0.1, 0.2],
        [0.3, 0.4],
    ]
)

print(outputs.argmax(1))

preds = outputs.argmax(1)
targets = torch.tensor([0, 1])

# 整体正确的个数
print((preds == targets).sum())
