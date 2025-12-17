# -*- coding: utf-8 -*-
"""
欧氏对齐（Euclidean Alignment, EA）模块

原理:
- 计算训练集的参考协方差矩阵
- 通过白化/对齐操作，使数据在通道空间中对齐
- 不引入任何可学习参数，纯数据预处理步骤

优势:
- 提升跨会话/跨日稳定性
- 在MI任务和BCI Competition IV 2a上被广泛验证
- 零参数开销

参考:
- He & Wu (2019). Transfer Learning for Brain-Computer Interfaces
- Zanini et al. (2018). Transfer Learning: A Riemannian Geometry Framework
"""

import numpy as np
from scipy.linalg import sqrtm, inv


class EuclideanAlignment:
    """
    欧氏对齐（EA）预处理器
    
    使用方法:
    1. fit(X_train): 在训练集上拟合参考协方差和对齐矩阵
    2. transform(X): 对任意数据集应用对齐变换
    """
    
    def __init__(self):
        """初始化EA对齐器"""
        self.R_ref = None  # 参考协方差矩阵
        self.W = None      # 白化/对齐矩阵
        self.is_fitted = False
        
    def fit(self, X_train):
        """
        在训练集上拟合EA参数
        
        Args:
            X_train: (N, C, T) numpy数组，训练集EEG数据
                     N = 样本数
                     C = 通道数
                     T = 时间点数
        
        计算过程:
        1. 计算每个trial的协方差矩阵 Ri
        2. 对所有Ri取平均，得到参考协方差 R_ref
        3. 计算白化矩阵 W = R_ref^(-1/2)
        """
        print(f"🔧 欧氏对齐 (EA) - 拟合阶段")
        print(f"  训练集数据: {X_train.shape}")
        
        N, C, T = X_train.shape
        
        # 1. 计算每个trial的协方差矩阵
        cov_matrices = []
        for i in range(N):
            # X_train[i]: (C, T)
            # 协方差矩阵: (C, C)
            cov = np.cov(X_train[i], rowvar=True)  # rowvar=True: 每行是一个变量(通道)
            cov_matrices.append(cov)
        
        cov_matrices = np.array(cov_matrices)  # (N, C, C)
        
        # 2. 计算参考协方差矩阵（所有trial的平均）
        self.R_ref = np.mean(cov_matrices, axis=0)  # (C, C)
        
        # 3. 计算白化矩阵 W = R_ref^(-1/2)
        # 使用矩阵平方根的逆
        try:
            # 方法1: 直接使用sqrtm
            R_ref_sqrt = sqrtm(self.R_ref)
            self.W = inv(R_ref_sqrt)
        except np.linalg.LinAlgError:
            print("  ⚠️ 矩阵求逆失败，使用特征值分解方法")
            # 方法2: 特征值分解（更稳定）
            eigvals, eigvecs = np.linalg.eigh(self.R_ref)
            # 避免数值不稳定，添加小的正则化项
            eigvals = np.maximum(eigvals, 1e-6)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
            self.W = eigvecs @ D_inv_sqrt @ eigvecs.T
        
        # 确保W是实数矩阵（有时sqrtm会引入微小虚部）
        if np.iscomplexobj(self.W):
            print("  ⚠️ 白化矩阵包含虚部，取实部")
            self.W = np.real(self.W)
        
        self.is_fitted = True
        
        print(f"  ✅ 参考协方差矩阵: {self.R_ref.shape}")
        print(f"  ✅ 白化矩阵: {self.W.shape}")
        print(f"  ✅ 协方差矩阵条件数: {np.linalg.cond(self.R_ref):.2e}")
        print(f"  ✅ 白化矩阵条件数: {np.linalg.cond(self.W):.2e}")
        
        return self
    
    def transform(self, X):
        """
        应用EA对齐变换
        
        Args:
            X: (N, C, T) numpy数组，待对齐的EEG数据
        
        Returns:
            X_aligned: (N, C, T) numpy数组，对齐后的EEG数据
        
        变换过程:
        对每个trial: X_aligned[i] = W @ X[i]
        其中 W 是白化矩阵，使得变换后的数据协方差接近单位矩阵
        """
        if not self.is_fitted:
            raise ValueError("必须先调用 fit() 方法拟合参数！")
        
        print(f"🔧 欧氏对齐 (EA) - 变换阶段")
        print(f"  输入数据: {X.shape}")
        
        N, C, T = X.shape
        X_aligned = np.zeros_like(X)
        
        # 对每个trial应用对齐矩阵
        for i in range(N):
            # X[i]: (C, T)
            # W: (C, C)
            # W @ X[i]: (C, T)
            X_aligned[i] = self.W @ X[i]
        
        print(f"  ✅ 对齐完成: {X_aligned.shape}")
        
        return X_aligned
    
    def fit_transform(self, X_train):
        """
        拟合并变换训练集（便捷方法）

        Args:
            X_train: (N, C, T) numpy数组，训练集EEG数据

        Returns:
            X_train_aligned: (N, C, T) numpy数组，对齐后的训练集
        """
        self.fit(X_train)
        return self.transform(X_train)


def verify_alignment(X_original, X_aligned):
    """
    验证EA对齐效果

    Args:
        X_original: (N, C, T) 原始数据
        X_aligned: (N, C, T) 对齐后数据

    打印:
        - 原始数据的平均协方差矩阵
        - 对齐后数据的平均协方差矩阵（应接近单位矩阵）
    """
    print(f"\n{'='*80}")
    print(f"📊 EA对齐效果验证")
    print(f"{'='*80}")

    N, C, T = X_original.shape

    # 计算原始数据的平均协方差
    cov_orig_list = []
    for i in range(N):
        cov = np.cov(X_original[i], rowvar=True)
        cov_orig_list.append(cov)
    cov_orig_mean = np.mean(cov_orig_list, axis=0)

    # 计算对齐后数据的平均协方差
    cov_aligned_list = []
    for i in range(N):
        cov = np.cov(X_aligned[i], rowvar=True)
        cov_aligned_list.append(cov)
    cov_aligned_mean = np.mean(cov_aligned_list, axis=0)

    # 打印统计信息
    print(f"原始数据:")
    print(f"  平均协方差矩阵对角线均值: {np.mean(np.diag(cov_orig_mean)):.4f}")
    print(f"  平均协方差矩阵对角线标准差: {np.std(np.diag(cov_orig_mean)):.4f}")
    print(f"  平均协方差矩阵非对角线均值: {np.mean(np.abs(cov_orig_mean - np.diag(np.diag(cov_orig_mean)))):.4f}")

    print(f"\n对齐后数据:")
    print(f"  平均协方差矩阵对角线均值: {np.mean(np.diag(cov_aligned_mean)):.4f}")
    print(f"  平均协方差矩阵对角线标准差: {np.std(np.diag(cov_aligned_mean)):.4f}")
    print(f"  平均协方差矩阵非对角线均值: {np.mean(np.abs(cov_aligned_mean - np.diag(np.diag(cov_aligned_mean)))):.4f}")

    # 计算与单位矩阵的距离
    I = np.eye(C)
    frobenius_distance = np.linalg.norm(cov_aligned_mean - I, 'fro')
    print(f"\n  与单位矩阵的Frobenius距离: {frobenius_distance:.4f}")
    print(f"  (理想情况下应接近0)")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    """测试EA模块"""
    print("="*80)
    print("🧪 欧氏对齐（EA）模块测试")
    print("="*80)

    # 生成模拟EEG数据
    np.random.seed(42)
    N_train = 100
    N_test = 50
    C = 22  # 通道数
    T = 1000  # 时间点数

    # 训练集：添加一些通道间的相关性
    X_train = np.random.randn(N_train, C, T)
    for i in range(N_train):
        # 添加通道间相关性
        correlation_matrix = np.random.randn(C, C)
        correlation_matrix = correlation_matrix @ correlation_matrix.T
        L = np.linalg.cholesky(correlation_matrix + np.eye(C) * 0.1)
        X_train[i] = L @ X_train[i]

    # 测试集：不同的相关性结构
    X_test = np.random.randn(N_test, C, T)
    for i in range(N_test):
        correlation_matrix = np.random.randn(C, C) * 0.5
        correlation_matrix = correlation_matrix @ correlation_matrix.T
        L = np.linalg.cholesky(correlation_matrix + np.eye(C) * 0.1)
        X_test[i] = L @ X_test[i]

    print(f"\n训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")

    # 创建EA对齐器
    ea = EuclideanAlignment()

    # 拟合并变换训练集
    print(f"\n{'='*80}")
    X_train_aligned = ea.fit_transform(X_train)

    # 变换测试集
    print(f"\n{'='*80}")
    X_test_aligned = ea.transform(X_test)

    # 验证对齐效果
    verify_alignment(X_train, X_train_aligned)
    verify_alignment(X_test, X_test_aligned)

    print("✅ EA模块测试完成！")

