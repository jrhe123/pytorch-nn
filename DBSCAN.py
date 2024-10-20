import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

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

# 使用UMAP进行降维
umap_model = umap.UMAP(n_components=2)
umap_data = umap_model.fit_transform(scaled_data)

# DBSCAN 聚类
dbscan = DBSCAN(eps=0.5, min_samples=2)
dbscan_labels = dbscan.fit_predict(scaled_data)

# 层次聚类
hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(scaled_data)

# 可视化UMAP降维结果 (DBSCAN)
plt.figure(figsize=(8, 6))
plt.scatter(umap_data[:, 0], umap_data[:, 1], c=dbscan_labels, cmap="viridis")
plt.title("UMAP Visualization with DBSCAN Clusters")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.colorbar(label="Cluster")
plt.show()

# 可视化UMAP降维结果 (Hierarchical Clustering)
plt.figure(figsize=(8, 6))
plt.scatter(umap_data[:, 0], umap_data[:, 1], c=hierarchical_labels, cmap="viridis")
plt.title("UMAP Visualization with Hierarchical Clusters")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.colorbar(label="Cluster")
plt.show()

# 计算并打印每个特征在不同簇下的统计信息 (DBSCAN)
print("DBSCAN Clusters:")
for feature in ["age", "education_num", "hours_per_week", "income"]:
    print(f"\nFeature: {feature}")
    for cluster in np.unique(dbscan_labels):
        cluster_data = df[dbscan_labels == cluster][feature]
        print(f"Cluster {cluster}:")
        print(f"  Count: {cluster_data.count()}")
        print(f"  Mean: {cluster_data.mean()}")
        print(f"  Std: {cluster_data.std()}")

# 计算并打印每个特征在不同簇下的统计信息 (Hierarchical Clustering)
print("Hierarchical Clusters:")
for feature in ["age", "education_num", "hours_per_week", "income"]:
    print(f"\nFeature: {feature}")
    for cluster in np.unique(hierarchical_labels):
        cluster_data = df[hierarchical_labels == cluster][feature]
        print(f"Cluster {cluster}:")
        print(f"  Count: {cluster_data.count()}")
        print(f"  Mean: {cluster_data.mean()}")
        print(f"  Std: {cluster_data.std()}")

# 绘制每个特征在不同簇下的直方图 (DBSCAN)
features = ["age", "education_num", "hours_per_week", "income"]
for feature in features:
    plt.figure(figsize=(10, 4))
    for cluster in np.unique(dbscan_labels):
        cluster_data = df[dbscan_labels == cluster][feature]
        plt.hist(cluster_data, bins=10, alpha=0.5, label=f"Cluster {cluster}")
    plt.title(f"Distribution of {feature} (DBSCAN)")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# 绘制每个特征在不同簇下的直方图 (Hierarchical Clustering)
for feature in features:
    plt.figure(figsize=(10, 4))
    for cluster in np.unique(hierarchical_labels):
        cluster_data = df[hierarchical_labels == cluster][feature]
        plt.hist(cluster_data, bins=10, alpha=0.5, label=f"Cluster {cluster}")
    plt.title(f"Distribution of {feature} (Hierarchical Clustering)")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
