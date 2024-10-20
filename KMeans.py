import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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

# 使用PCA降维
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# 使用K-means进行聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(pca_data)

# 计算Silhouette Score
silhouette_avg = silhouette_score(pca_data, kmeans_labels)
print(f"Silhouette Score: {silhouette_avg:.4f}")

# 可视化PCA降维后的K-means聚类结果
plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans_labels, cmap="viridis")
plt.title("PCA Visualization with K-means Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
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
