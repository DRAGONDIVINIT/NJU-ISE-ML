from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 加载 LFW 数据集
lfw_people = fetch_lfw_people(min_faces_per_person=100, resize=0.4)

# 2. 获取数据
X = lfw_people.data  

# Todo1: 写出PCA函数
def PCA(X, n_components=5):
    """
    主成分分析（PCA）实现
    
    参数:
    X: 输入数据，形状为 (n_samples, n_features)
    n_components: 要提取的主成分数量
    
    返回:
    eigenfaces: 特征脸，形状为 (n_components, height * width)
    """
    # 1. 数据中心化：减去均值
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    # 2. 计算协方差矩阵
    # 如果样本数小于特征数，使用SVD方法计算更高效
    n_samples, n_features = X.shape
    
    if n_samples < n_features:
        # 使用SVD方法
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        # 主成分是Vt的前n_components行
        components = Vt[:n_components]
    else:
        # 计算协方差矩阵
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # 3. 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 4. 对特征值和特征向量进行排序（降序）
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        eigenvalues = eigenvalues[sorted_indices]
        
        # 5. 选择前n_components个主成分
        components = eigenvectors[:, :n_components].T
    
    # 将主成分转换为特征脸形式
    eigenfaces = components.reshape((n_components, *lfw_people.images.shape[1:]))
    
    return eigenfaces

# 应用PCA
eigenfaces = PCA(X, n_components=5)

# Todo2: 可视化5个主成分对应的特征脸
def plot_eigenfaces(eigenfaces, image_shape):
    """
    可视化特征脸
    
    参数:
    eigenfaces: 特征脸数组
    image_shape: 图像原始形状
    """
    n_components = eigenfaces.shape[0]
    
    # 创建子图
    fig, axes = plt.subplots(1, n_components, figsize=(15, 3))
    
    # 如果只有一个主成分，axes不是数组，需要转换
    if n_components == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        # 获取第i个特征脸并重塑为图像形状
        eigenface = eigenfaces[i].reshape(image_shape)
        
        # 显示图像
        ax.imshow(eigenface, cmap='gray')
        ax.set_title(f'主成分 {i+1}')
        ax.axis('off')
    
    plt.suptitle('前5个主成分对应的特征脸', fontsize=14)
    plt.tight_layout()
    plt.show()

# 可视化特征脸
plot_eigenfaces(eigenfaces, lfw_people.images.shape[1:])