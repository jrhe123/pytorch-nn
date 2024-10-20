import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import umap
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset

# 假设有一个虚构的人口普查数据集
data = {
    "age": [25, 32, 47, 51, 62, 23, 35, 46, 53, 21],
    "education_num": [10, 12, 14, 13, 9, 11, 15, 10, 13, 12],
    "hours_per_week": [40, 50, 60, 45, 30, 20, 55, 45, 50, 35],
}

# 创建DataFrame
df = pd.DataFrame(data)


# 创建一个PyTorch数据集类
class CensusDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.features = self.data[["age", "education_num", "hours_per_week"]].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        return features


# 初始化数据集
dataset = CensusDataset(df)
data_loader = DataLoader(dataset, batch_size=5, shuffle=True)


# 定义自编码器模型
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 2), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(2, 3), nn.ReLU())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# 初始化模型、损失函数和优化器
model = AutoEncoder()
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    for data in data_loader:
        # 前向传播
        reconstructed = model(data)
        loss = criterion(reconstructed, data)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 获取编码后的数据
encoded_data = []
with torch.no_grad():
    for data in data_loader:
        encoded = model.encoder(data)
        encoded_data.append(encoded)
encoded_data = torch.cat(encoded_data).numpy()

# 使用UMAP进行可视化
umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, metric="correlation")
umap_embedding = umap_model.fit_transform(encoded_data)

plt.figure(figsize=(8, 6))
plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c="blue", label="UMAP")
plt.legend()
plt.title("UMAP Visualization")
plt.show()

# 使用t-SNE进行可视化
# tsne_model = TSNE(n_components=2, perplexity=5, n_iter=300)
# tsne_embedding = tsne_model.fit_transform(encoded_data)

# plt.figure(figsize=(8, 6))
# plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c="red", label="t-SNE")
# plt.legend()
# plt.title("t-SNE Visualization")
# plt.show()
