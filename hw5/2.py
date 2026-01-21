import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 1. 生成瑞士卷数据集
X, color = make_swiss_roll(n_samples=3000, random_state=0)

print(f"数据形状: {X.shape}")
print(f"颜色数据形状: {color.shape}")

# 2. 可视化原始瑞士卷数据
fig = plt.figure(figsize=(15, 10))

# 原始数据3D可视化
ax1 = fig.add_subplot(231, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral, s=10)
ax1.set_title('原始瑞士卷数据 (3D)')
ax1.view_init(10, -60)

# 2D投影可视化
ax2 = fig.add_subplot(232)
ax2.scatter(X[:, 0], X[:, 2], c=color, cmap=plt.cm.Spectral, s=10)
ax2.set_title('X-Z平面投影')
ax2.set_xlabel('X')
ax2.set_ylabel('Z')

ax3 = fig.add_subplot(233)
ax3.scatter(X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral, s=10)
ax3.set_title('Y-Z平面投影')
ax3.set_xlabel('Y')
ax3.set_ylabel('Z')

plt.tight_layout()
plt.show()

# 3. 自定义LLE实现
def custom_LLE(X, n_components=2, n_neighbors=12, reg=1e-3):
    """
    自定义局部线性嵌入(LLE)实现
    
    参数:
    X: 输入数据，形状(n_samples, n_features)
    n_components: 降维后的维度
    n_neighbors: 邻居数量
    reg: 正则化参数
    
    返回:
    Y: 低维嵌入，形状(n_samples, n_components)
    """
    n_samples, n_features = X.shape
    
    # Step 1: 找到每个点的k个最近邻居
    knn = NearestNeighbors(n_neighbors=n_neighbors+1)
    knn.fit(X)
    distances, indices = knn.kneighbors(X)
    
    # 移除自身
    indices = indices[:, 1:]  # 移除第一个点(自身)
    
    # Step 2: 计算重构权重矩阵W
    W = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        # 获取邻居
        neighbors = indices[i]
        Xi = X[i] - X[neighbors]  # 中心化
        
        # 计算局部协方差矩阵
        C = np.dot(Xi, Xi.T)
        
        # 添加正则化项
        C.flat[::n_neighbors+1] += reg * np.trace(C)
        
        # 求解权重: C * w = 1
        try:
            w = np.linalg.solve(C, np.ones(n_neighbors))
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            w = np.linalg.pinv(C).dot(np.ones(n_neighbors))
        
        # 归一化权重
        w = w / np.sum(w)
        
        # 存储权重
        W[i, neighbors] = w
    
    # Step 3: 计算低维嵌入Y
    # M = (I - W)^T (I - W)
    I = np.eye(n_samples)
    M = (I - W).T.dot(I - W)
    
    # 计算M的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    
    # 选择最小的n_components+1个特征值对应的特征向量
    # 忽略最小的特征值(接近0)
    Y = eigenvectors[:, 1:n_components+1]
    
    return Y

# 4. 不同参数下的LLE实验
fig = plt.figure(figsize=(20, 15))

# 参数组合
param_combinations = [
    {'n_neighbors': 8, 'reg': 1e-3, 'title': '邻居数=8, 正则化=1e-3'},
    {'n_neighbors': 12, 'reg': 1e-3, 'title': '邻居数=12, 正则化=1e-3'},
    {'n_neighbors': 20, 'reg': 1e-3, 'title': '邻居数=20, 正则化=1e-3'},
    {'n_neighbors': 12, 'reg': 1e-5, 'title': '邻居数=12, 正则化=1e-5'},
    {'n_neighbors': 12, 'reg': 1e-1, 'title': '邻居数=12, 正则化=1e-1'},
    {'n_neighbors': 30, 'reg': 1e-3, 'title': '邻居数=30, 正则化=1e-3'},
]

# 使用自定义LLE
for i, params in enumerate(param_combinations, 1):
    ax = fig.add_subplot(2, 3, i)
    
    try:
        # 使用自定义LLE
        Y_custom = custom_LLE(
            X, 
            n_components=2,
            n_neighbors=params['n_neighbors'],
            reg=params['reg']
        )
        
        ax.scatter(Y_custom[:, 0], Y_custom[:, 1], c=color, cmap=plt.cm.Spectral, s=10)
        ax.set_title(f'自定义LLE: {params["title"]}')
        ax.set_xlabel('LLE Component 1')
        ax.set_ylabel('LLE Component 2')
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
        ax.set_title(f'失败: {params["title"]}')

plt.tight_layout()
plt.suptitle('自定义LLE在不同参数下的结果', fontsize=16, y=1.02)
plt.show()

# 5. 使用sklearn的LLE进行比较
fig = plt.figure(figsize=(20, 15))

# 不同方法和参数的比较
methods_params = [
    {'n_neighbors': 12, 'method': 'standard', 'title': '标准LLE, 邻居数=12'},
    {'n_neighbors': 12, 'method': 'modified', 'title': '改进LLE, 邻居数=12'},
    {'n_neighbors': 8, 'method': 'standard', 'title': '标准LLE, 邻居数=8'},
    {'n_neighbors': 20, 'method': 'standard', 'title': '标准LLE, 邻居数=20'},
    {'n_neighbors': 30, 'method': 'standard', 'title': '标准LLE, 邻居数=30'},
    {'n_neighbors': 12, 'method': 'hessian', 'title': 'Hessian LLE, 邻居数=12'},
]

for i, params in enumerate(methods_params, 1):
    ax = fig.add_subplot(2, 3, i)
    
    try:
        lle = LocallyLinearEmbedding(
            n_components=2,
            n_neighbors=params['n_neighbors'],
            method=params['method'],
            random_state=42
        )
        
        Y_sklearn = lle.fit_transform(X)
        
        ax.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=color, cmap=plt.cm.Spectral, s=10)
        ax.set_title(f'sklearn LLE: {params["title"]}')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
        ax.set_title(f'失败: {params["title"]}')

plt.tight_layout()
plt.suptitle('sklearn LLE在不同参数下的结果', fontsize=16, y=1.02)
plt.show()

# 6. 邻居数影响的分析
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

n_neighbors_list = [5, 8, 12, 20, 30, 50]

for idx, n_neighbors in enumerate(n_neighbors_list):
    ax = axes[idx]
    
    try:
        lle = LocallyLinearEmbedding(
            n_components=2,
            n_neighbors=n_neighbors,
            method='standard',
            random_state=42
        )
        
        Y = lle.fit_transform(X)
        
        ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, s=10)
        ax.set_title(f'邻居数 = {n_neighbors}')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        
        # 计算重构误差
        reconstruction_error = lle.reconstruction_error_
        ax.text(0.05, 0.95, f'误差: {reconstruction_error:.2e}', 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error', ha='center', va='center')
        ax.set_title(f'邻居数 = {n_neighbors}')

plt.tight_layout()
plt.suptitle('邻居数对LLE结果的影响', fontsize=16, y=1.02)
plt.show()