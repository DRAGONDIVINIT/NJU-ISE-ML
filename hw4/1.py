import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 加载数据
def load_mall_data():
    df = pd.read_csv("D:\资料\机器学习\hw4\Mall_Customers.csv")
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X, df

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None
        self.inertia_ = None  # 簇内平方和
        self.random_state = random_state
        
    def _initialize_centroids(self, X):
        """随机初始化聚类中心"""
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        # 随机选择k个样本作为初始聚类中心
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[indices]
    
    def _compute_distance(self, X, centroids):
        """计算每个样本到各个聚类中心的距离"""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.sqrt(np.sum((X - centroids[i]) ** 2, axis=1))
        return distances
    
    def _assign_clusters(self, distances):
        """分配每个样本到最近的聚类中心"""
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        """更新聚类中心为簇内样本的均值"""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # 如果簇为空，随机重新初始化该聚类中心
                new_centroids[i] = X[np.random.randint(0, X.shape[0])]
        return new_centroids
    
    def fit(self, X):
        """
        实现K-means算法
        参数:
            X: shape (n_samples, n_features)
        返回:
            self
        """
        # 1. 随机初始化聚类中心
        self.centroids = self._initialize_centroids(X)
        
        # 2. 迭代优化直到收敛
        for iteration in range(self.max_iters):
            # 计算距离并分配标签
            distances = self._compute_distance(X, self.centroids)
            labels = self._assign_clusters(distances)
            
            # 更新聚类中心
            new_centroids = self._update_centroids(X, labels)
            
            # 检查是否收敛（聚类中心不再变化）
            if np.allclose(self.centroids, new_centroids):
                print(f"KMeans收敛于第 {iteration+1} 次迭代")
                break
                
            self.centroids = new_centroids
        
        # 保存最终结果
        distances = self._compute_distance(X, self.centroids)
        self.labels = self._assign_clusters(distances)
        
        # 计算簇内平方和
        self.inertia_ = 0
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                self.inertia_ += np.sum(np.sum((cluster_points - self.centroids[i]) ** 2, axis=1))
        
        return self
        
    def predict(self, X):
        """返回每个样本的聚类标签"""
        distances = self._compute_distance(X, self.centroids)
        return self._assign_clusters(distances)
    
    def fit_predict(self, X):
        """拟合数据并返回标签"""
        self.fit(X)
        return self.labels

def plot_clusters_3d(X, labels, centroids=None, title="KMeans聚类结果"):
    """绘制聚类结果的3D散点图"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 为每个簇分配颜色
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        cluster_points = X[labels == label]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                  c=[color], label=f'簇 {label}', alpha=0.7, s=50)
    
    # 绘制聚类中心
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                  c='black', marker='X', s=200, label='聚类中心', alpha=1.0, linewidths=2)
    
    ax.set_xlabel('年龄 (标准化)', fontsize=12)
    ax.set_ylabel('年收入 (标准化)', fontsize=12)
    ax.set_zlabel('消费积分 (标准化)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_clusters_2d(X, labels, centroids=None, feature_names=None):
    """绘制聚类结果的2D散点图（特征两两组合）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    if feature_names is None:
        feature_names = ['特征1', '特征2', '特征3']
    
    # 为每个簇分配颜色
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # 特征组合：0-1, 0-2, 1-2, 还有一个子图空着
    combinations = [(0, 1), (0, 2), (1, 2)]
    
    for idx, (i, j) in enumerate(combinations):
        ax = axes[idx // 2, idx % 2]
        
        for label, color in zip(unique_labels, colors):
            cluster_points = X[labels == label]
            ax.scatter(cluster_points[:, i], cluster_points[:, j],
                      c=[color], label=f'簇 {label}', alpha=0.6, s=40)
        
        # 绘制聚类中心
        if centroids is not None:
            ax.scatter(centroids[:, i], centroids[:, j],
                      c='black', marker='X', s=150, label='聚类中心', alpha=1.0, linewidths=2)
        
        ax.set_xlabel(feature_names[i], fontsize=11)
        ax.set_ylabel(feature_names[j], fontsize=11)
        ax.set_title(f'{feature_names[i]} vs {feature_names[j]}', fontsize=12)
        if idx == 0:
            ax.legend()
    
    # 移除最后一个空子图
    axes[1, 1].axis('off')
    
    plt.suptitle('KMeans聚类结果可视化', fontsize=16)
    plt.tight_layout()
    plt.show()

# 主程序
def main():
    # 加载数据
    X_scaled, X_original, df = load_mall_data()
    feature_names = ['Age', 'Annual Income', 'Spending Score']
    
    print(f"数据集形状: {X_scaled.shape}")
    print(f"样本数: {X_scaled.shape[0]}, 特征数: {X_scaled.shape[1]}")
    
    # 使用肘部法则确定最佳K值（可选）
    print("\n计算不同K值的簇内平方和...")
    inertias = []
    K_range = range(1, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, max_iters=100, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        print(f"K={kk}: 簇内平方和 = {kmeans.inertia_:.2f}")
    
    # 绘制肘部法则图
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('K值', fontsize=12)
    plt.ylabel('簇内平方和', fontsize=12)
    plt.title('肘部法则', fontsize=14)
    plt.xticks(K_range)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 选择K=5进行聚类（根据肘部法则）
    print("\n使用K=5进行KMeans聚类...")
    kmeans = KMeans(n_clusters=5, max_iters=100, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    print(f"聚类完成！")
    print(f"聚类中心形状: {kmeans.centroids.shape}")
    print(f"最终簇内平方和: {kmeans.inertia_:.2f}")
    
    # 统计每个簇的样本数
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"簇 {label}: {count} 个样本 ({count/X_scaled.shape[0]*100:.1f}%)")
    
    # 3D可视化
    plot_clusters_3d(X_scaled, labels, kmeans.centroids, 
                     title="客户分群分析 (3D可视化)")
    
    # 2D可视化
    plot_clusters_2d(X_scaled, labels, kmeans.centroids, feature_names)
    
    # 将聚类结果添加到原始数据中
    df['Cluster'] = labels
    
    # 分析每个簇的特征
    print("\n各簇特征统计:")
    cluster_stats = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
    print(cluster_stats.round(2))
    
    # 绘制簇特征条形图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, feature in enumerate(['Age', 'Annual Income (k$)', 'Spending Score (1-100)']):
        axes[idx].bar(cluster_stats.index, cluster_stats[feature])
        axes[idx].set_xlabel('簇')
        axes[idx].set_ylabel(feature)
        axes[idx].set_title(f'各簇平均{feature}')
        axes[idx].set_xticks(cluster_stats.index)
    
    plt.suptitle('簇特征分析', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return kmeans, df

if __name__ == "__main__":
    kmeans_model, df_with_clusters = main()