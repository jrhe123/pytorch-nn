import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 假设有一个虚构的人口普查数据集
data = {
    "age": [25, 32, 47, 51, 62, 23, 35, 46, 53, 21],
    "education_num": [10, 12, 14, 13, 9, 11, 15, 10, 13, 12],
    "hours_per_week": [40, 50, 60, 45, 30, 20, 55, 45, 50, 35],
}

# 创建DataFrame
df = pd.DataFrame(data)

# 标准化数据
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# 使用PCA进行降维
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# 使用K-means进行聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(pca_data)
labels = kmeans.labels_

# 可视化聚类结果
plt.figure(figsize=(10, 8))
plt.scatter(
    pca_data[:, 0],
    pca_data[:, 1],
    c=labels,
    cmap="viridis",
    marker="o",
    edgecolor="k",
    s=100,
)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c="red",
    marker="x",
    label="Centroids",
)
plt.title("K-means Clustering with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()

# 打印PCA解释的方差比例
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
