import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# 加载数据
def load_mall_data():
    df = pd.read_csv("D:\资料\机器学习\hw4\Mall_Customers.csv")
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X, df

# 简化版带约束KMeans
class ConstrainedKMeans:
    def __init__(self, n_clusters=5, weights=None, size_constraints=None):
        self.n_clusters = n_clusters
        self.weights = weights
        self.size_constraints = size_constraints
        
    def fit(self, X):
        n_samples = X.shape[0]
        
        # 随机初始化质心
        np.random.seed(42)
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices]
        
        # 如果没有大小约束，设置默认
        if self.size_constraints is None:
            min_size = 0
            max_size = n_samples
            self.size_constraints = [(min_size, max_size) for _ in range(self.n_clusters)]
        
        for _ in range(100):
            # 计算距离
            distances = self._compute_distance(X, self.centroids)
            
            # 分配标签
            labels = np.argmin(distances, axis=1)
            
            # 调整以满足最小约束
            labels = self._adjust_constraints(X, distances, labels)
            
            # 更新质心
            new_centroids = np.zeros_like(self.centroids)
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[i] = self.centroids[i]
            
            # 检查收敛
            if np.allclose(new_centroids, self.centroids, atol=1e-4):
                break
                
            self.centroids = new_centroids
        
        self.labels = labels
        return self
    
    def _compute_distance(self, X, centroids):
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        
        for i in range(self.n_clusters):
            if self.weights is None:
                distances[:, i] = np.sqrt(np.sum((X - centroids[i])**2, axis=1))
            else:
                weights = np.array(self.weights)
                weighted_diff = weights * (X - centroids[i])
                distances[:, i] = np.sqrt(np.sum(weighted_diff**2, axis=1))
        
        return distances
    
    def _adjust_constraints(self, X, distances, labels):
        n_samples = X.shape[0]
        min_sizes = [c[0] for c in self.size_constraints]
        
        # 确保最小约束
        for i in range(self.n_clusters):
            count = np.sum(labels == i)
            if count < min_sizes[i]:
                not_in_i = np.where(labels != i)[0]
                if len(not_in_i) > 0:
                    needed = min_sizes[i] - count
                    costs = distances[not_in_i, i]
                    idxs = not_in_i[np.argsort(costs)[:needed]]
                    labels[idxs] = i
        
        return labels

# 加载数据
X_scaled, X_original, df = load_mall_data()
n_samples = X_original.shape[0]
colors = ['red', 'blue', 'green', 'orange', 'purple']

print(f"总客户数: {n_samples}")

# 场景1: 收入特征权重加倍
print("\n=== 场景1: 收入特征权重加倍 ===")
weights1 = [1.0, 2.0, 1.0]  # 收入权重加倍
size_constraints1 = None  # 没有大小约束

model1 = ConstrainedKMeans(n_clusters=5, weights=weights1, 
                          size_constraints=size_constraints1)
model1.fit(X_scaled)
labels1 = model1.labels

# 场景2: 每个簇至少包含20%的客户
print("\n=== 场景2: 每个簇至少包含20%的客户 ===")
weights2 = None  # 没有特征权重
min_size = int(0.20 * n_samples)
size_constraints2 = [(min_size, n_samples) for _ in range(5)]

model2 = ConstrainedKMeans(n_clusters=5, weights=weights2, 
                          size_constraints=size_constraints2)
model2.fit(X_scaled)
labels2 = model2.labels

# 绘制两个场景的3D聚类结果
fig = plt.figure(figsize=(14, 6))

# 场景1: 收入特征权重加倍的3D可视化
ax1 = fig.add_subplot(121, projection='3d')
for i in range(5):
    ax1.scatter(X_original[labels1 == i, 0],  # Age
               X_original[labels1 == i, 1],  # Annual Income
               X_original[labels1 == i, 2],  # Spending Score
               s=50, c=colors[i], label=f'Cluster {i}', alpha=0.7)

ax1.set_title('Scenario 1: Income Weight Doubled\n(Weights=[1.0, 2.0, 1.0])', fontsize=12)
ax1.set_xlabel('Age', fontsize=10)
ax1.set_ylabel('Annual Income (k$)', fontsize=10)
ax1.set_zlabel('Spending Score (1-100)', fontsize=10)
ax1.legend()

# 场景2: 最小20%约束的3D可视化
ax2 = fig.add_subplot(122, projection='3d')
for i in range(5):
    ax2.scatter(X_original[labels2 == i, 0],  # Age
               X_original[labels2 == i, 1],  # Annual Income
               X_original[labels2 == i, 2],  # Spending Score
               s=50, c=colors[i], label=f'Cluster {i}', alpha=0.7)

ax2.set_title('Scenario 2: Min 20% per Cluster\n(No feature weighting)', fontsize=12)
ax2.set_xlabel('Age', fontsize=10)
ax2.set_ylabel('Annual Income (k$)', fontsize=10)
ax2.set_zlabel('Spending Score (1-100)', fontsize=10)
ax2.legend()

plt.tight_layout()
plt.show()

# 打印簇分布信息
print("\n场景1簇分布 (收入权重加倍):")
for i in range(5):
    count = np.sum(labels1 == i)
    percentage = (count / n_samples) * 100
    print(f"  Cluster {i}: {count} customers ({percentage:.1f}%)")

print("\n场景2簇分布 (最小20%约束):")
for i in range(5):
    count = np.sum(labels2 == i)
    percentage = (count / n_samples) * 100
    print(f"  Cluster {i}: {count} customers ({percentage:.1f}%)")
    if percentage < 20:
        print(f"    Warning: Below 20% minimum!")