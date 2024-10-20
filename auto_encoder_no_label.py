import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


# 定义更复杂的自编码器模型
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# 初始化模型、损失函数和优化器
model = AutoEncoder()
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1
)  # 学习率调度器

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
    scheduler.step()  # 更新学习率

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 获取编码后的数据
encoded_data = []
with torch.no_grad():
    for data in data_loader:
        encoded = model.encoder(data)
        encoded_data.append(encoded)
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
plt.title("UMAP Visualization of Encoded Data with K-means Clusters")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.colorbar(scatter, label="Cluster")
plt.show()

# 计算并打印每个特征在不同簇下的统计信息
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
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


# 特征重要性评估
importances = []
for feature in features:
    importance = 0
    for cluster in np.unique(kmeans_labels):
        cluster_data = df[kmeans_labels == cluster][feature]
        importance += abs(cluster_data.mean() - df[feature].mean()) / df[feature].std()
    importances.append(importance)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(len(features)), importances, align="center")
plt.xticks(range(len(features)), features, rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()


# 在原始数据上使用K-means进行聚类
# kmeans_orig = KMeans(n_clusters=3, random_state=42)
# kmeans_orig_labels = kmeans_orig.fit_predict(scaled_data)
# silhouette_orig = silhouette_score(scaled_data, kmeans_orig_labels)
# print(f"Silhouette Score on original data: {silhouette_orig:.4f}")

# 在编码后的数据上使用K-means进行聚类
kmeans_encoded = KMeans(n_clusters=3, random_state=42)
kmeans_encoded_labels = kmeans_encoded.fit_predict(encoded_data)
silhouette_encoded = silhouette_score(encoded_data, kmeans_encoded_labels)
print(f"Silhouette Score on encoded data: {silhouette_encoded:.4f}")
