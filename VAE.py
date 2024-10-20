import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# 假设有一个虚拟的人口普查数据集
data = {
    "age": [25, 32, 47, 51, 62, 23, 35, 46, 53, 21, 34, 41, 29, 57, 49],
    "education_num": [10, 12, 14, 13, 9, 11, 15, 10, 13, 12, 11, 10, 14, 15, 12],
    "hours_per_week": [40, 50, 60, 45, 30, 20, 55, 45, 50, 35, 40, 50, 60, 45, 30],
    "income": [50, 60, 70, 80, 90, 40, 55, 65, 75, 85, 95, 55, 65, 75, 85],
}

# 创建DataFrame
df = pd.DataFrame(data)

# 标准化数据
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)


# 创建一个PyTorch数据集类
class CensusDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data[idx], dtype=torch.float32)
        return features


# 初始化数据集
dataset = CensusDataset(scaled_data)
data_loader = DataLoader(dataset, batch_size=5, shuffle=True)


# 定义变分自动编码器（VAE）模型
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


# 初始化VAE模型、优化器和损失函数
input_dim = scaled_data.shape[1]
hidden_dim = 128
latent_dim = 2
model = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练VAE模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for data in data_loader:
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
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
        mu, logvar = model.encode(data)
        z = model.reparameterize(mu, logvar)
        encoded_data.append(z)
encoded_data = torch.cat(encoded_data).numpy()

# 使用UMAP进行降维
umap_model = umap.UMAP(n_components=2)
umap_data = umap_model.fit_transform(encoded_data)

# 使用K-means进行聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(encoded_data)

# 可视化UMAP降维结果
plt.figure(figsize=(8, 6))
scatter = plt.scatter(umap_data[:, 0], umap_data[:, 1], c=kmeans_labels, cmap="viridis")
plt.title("UMAP Visualization with VAE and K-means Clusters")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.colorbar(scatter, label="Cluster")
plt.show()

# 计算并打印每个特征在不同簇下的统计信息
print("K-means Clusters:")
for feature in ["age", "education_num", "hours_per_week", "income"]:
    print(f"\nFeature: {feature}")
    for cluster in np.unique(kmeans_labels):
        cluster_data = df[kmeans_labels == cluster][feature]
        print(f"Cluster {cluster}:")
        print(f"  Count: {cluster_data.count()}")
        print(f"  Mean: {cluster_data.mean()}")
        print(f"  Std: {cluster_data.std()}")

# 绘制每个特征在不同簇下的直方图
features = ["age", "education_num", "hours_per_week", "income"]
for feature in features:
    plt.figure(figsize=(10, 4))
    for cluster in np.unique(kmeans_labels):
        cluster_data = df[kmeans_labels == cluster][feature]
        plt.hist(cluster_data, bins=10, alpha=0.5, label=f"Cluster {cluster}")
    plt.title(f"Distribution of {feature} (K-means)")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# 在编码后的数据上使用K-means进行聚类
kmeans_encoded = KMeans(n_clusters=3, random_state=42)
kmeans_encoded_labels = kmeans_encoded.fit_predict(encoded_data)
silhouette_encoded = silhouette_score(encoded_data, kmeans_encoded_labels)
print(f"Silhouette Score on encoded data: {silhouette_encoded:.4f}")
