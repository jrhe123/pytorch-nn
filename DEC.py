import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# 数据准备和标准化
data = {
    "age": [25, 32, 47, 51, 62, 23, 35, 46, 53, 21, 34, 41, 29, 57, 49],
    "education_num": [10, 12, 14, 13, 9, 11, 15, 10, 13, 12, 11, 10, 14, 15, 12],
    "hours_per_week": [40, 50, 60, 45, 30, 20, 55, 45, 50, 35, 40, 50, 60, 45, 30],
    "income": [50, 60, 70, 80, 90, 40, 55, 65, 75, 85, 95, 55, 65, 75, 85],
}
df = pd.DataFrame(data)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)


class CensusDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data[idx], dtype=torch.float32)
        return features


dataset = CensusDataset(scaled_data)
data_loader = DataLoader(dataset, batch_size=5, shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# 定义模型、损失函数和优化器
input_dim = scaled_data.shape[1]
hidden_dim = 128
latent_dim = 10
model = AutoEncoder(input_dim, hidden_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练自编码器
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for data in data_loader:
        optimizer.zero_grad()
        encoded, decoded = model(data)
        loss = criterion(decoded, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(data_loader.dataset):.4f}"
    )

# 获取编码后的数据
encoded_data = []
with torch.no_grad():
    for data in data_loader:
        encoded, _ = model(data)
        encoded_data.append(encoded)
encoded_data = torch.cat(encoded_data).numpy()

# 使用K-means在潜在空间进行初始聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(encoded_data)


# 深度嵌入聚类 (DEC)
class DEC(nn.Module):
    def __init__(self, autoencoder, n_clusters=3, alpha=1.0):
        super(DEC, self).__init__()
        self.autoencoder = autoencoder
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.clustering_layer = nn.Parameter(torch.Tensor(n_clusters, latent_dim))
        torch.nn.init.xavier_normal_(self.clustering_layer.data)

    def forward(self, x):
        encoded, decoded = self.autoencoder(x)
        q = 1.0 / (
            1.0
            + torch.sum(torch.pow(encoded.unsqueeze(1) - self.clustering_layer, 2), 2)
            / self.alpha
        )
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return encoded, decoded, q


# 初始化 DEC 模型
dec_model = DEC(model, n_clusters=3)
optimizer = optim.Adam(dec_model.parameters(), lr=0.001)


# KL散度损失函数
def kl_loss(q, p):
    return torch.mean(torch.sum(p * torch.log(p / q), dim=1))


# 聚类目标分布
def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


# 训练 DEC 模型
num_epochs = 100
for epoch in range(num_epochs):
    dec_model.train()
    train_loss = 0
    for data in data_loader:
        optimizer.zero_grad()
        encoded, decoded, q = dec_model(data)
        p = target_distribution(q).detach()
        loss = criterion(decoded, data) + kl_loss(q, p)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(data_loader.dataset):.4f}"
    )

# 获取编码后的数据和聚类结果
encoded_data = []
dec_labels = []
with torch.no_grad():
    for data in data_loader:
        encoded, _, q = dec_model(data)
        encoded_data.append(encoded)
        dec_labels.append(torch.argmax(q, dim=1))
encoded_data = torch.cat(encoded_data).numpy()
dec_labels = torch.cat(dec_labels).numpy()

# 使用UMAP进行降维
umap_model = umap.UMAP(n_components=2)
umap_data = umap_model.fit_transform(encoded_data)

# 可视化UMAP降维结果
plt.figure(figsize=(8, 6))
scatter = plt.scatter(umap_data[:, 0], umap_data[:, 1], c=dec_labels, cmap="viridis")
plt.title("UMAP Visualization with DEC Clusters")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.colorbar(scatter, label="Cluster")
plt.show()

# 计算并打印每个特征在不同簇下的统计信息
print("DEC Clusters:")
for feature in ["age", "education_num", "hours_per_week", "income"]:
    print(f"\nFeature: {feature}")
    for cluster in np.unique(dec_labels):
        cluster_data = df[dec_labels == cluster][feature]
        print(f"Cluster {cluster}:")
        print(f"  Count: {cluster_data.count()}")
        print(f"  Mean: {cluster_data.mean()}")
        print(f"  Std: {cluster_data.std()}")

# 绘制每个特征在不同簇下的直方图
features = ["age", "education_num", "hours_per_week", "income"]
for feature in features:
    plt.figure(figsize=(10, 4))
    for cluster in np.unique(dec_labels):
        cluster_data = df[dec_labels == cluster][feature]
        plt.hist(cluster_data, bins=10, alpha=0.5, label=f"Cluster {cluster}")
    plt.title(f"Distribution of {feature} (DEC)")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
