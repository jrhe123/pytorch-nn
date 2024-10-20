import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 假设有一个虚拟的人口普查数据集
data = {
    "age": [25, 32, 47, 51, 62, 23, 35, 46, 53, 21, 34, 41, 29, 57, 49],
    "education_num": [10, 12, 14, 13, 9, 11, 15, 10, 13, 12, 11, 10, 14, 15, 12],
    "hours_per_week": [40, 50, 60, 45, 30, 20, 55, 45, 50, 35, 40, 50, 60, 45, 30],
    "income": [50, 60, 70, 80, 90, 40, 55, 65, 75, 85, 95, 55, 65, 75, 85],
    "label": [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1],  # 假设有一个二分类标签
}
df = pd.DataFrame(data)

# 特征和标签分离
X = df.drop("label", axis=1)
y = df["label"]

# 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 初始化随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Class 0", "Class 1"],
    yticklabels=["Class 0", "Class 1"],
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# 可视化特征重要性
feature_importances = clf.feature_importances_
features = X.columns
indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), feature_importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()
