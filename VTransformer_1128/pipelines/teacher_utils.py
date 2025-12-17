# -*- coding: utf-8 -*-
"""
V5.1 教师模型工具函数

实现4个教师视角：
1. CSP + LDA
2. Riemannian Tangent Space + LR
3. CEMD IMF Energy + LR  
4. STFT Band Power + LR
"""

import numpy as np
from scipy.signal import butter, filtfilt, stft
from scipy.linalg import sqrtm, inv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

try:
    from PyEMD import CEMD
    HAS_CEMD = True
except ImportError:
    HAS_CEMD = False
    print("警告: PyEMD未安装，CEMD教师将不可用")


# ============================================================
# Teacher 1: CSP + LDA
# ============================================================

def apply_csp(X, y, n_components=3):
    """
    应用Common Spatial Patterns
    
    Args:
        X: (N, C, T) EEG数据
        y: (N,) 标签
        n_components: CSP成分数（每个类别）
    
    Returns:
        W: (C, n_components*2) CSP滤波器
    """
    N, C, T = X.shape
    n_classes = len(np.unique(y))
    
    # 计算每个类别的平均协方差矩阵
    cov_matrices = []
    for label in range(n_classes):
        X_class = X[y == label]
        # 计算归一化协方差矩阵
        cov = np.zeros((C, C))
        for trial in X_class:
            trial_cov = np.cov(trial)
            trace = np.trace(trial_cov)
            if trace > 0:
                cov += trial_cov / trace
        cov /= len(X_class)
        cov_matrices.append(cov)
    
    # 两类CSP（简化为第一类 vs 其他类）
    cov_1 = cov_matrices[0]
    cov_rest = np.mean(cov_matrices[1:], axis=0)
    
    # 广义特征值分解
    try:
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(cov_rest, cov_1))
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
    except:
        # 如果失败，返回单位矩阵的子集
        eigenvalues = np.ones(C)
        eigenvectors = np.eye(C)
    
    # 排序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # 选择前n_components和后n_components个
    W = np.hstack([eigenvectors[:, :n_components], eigenvectors[:, -n_components:]])
    
    return W


def extract_csp_features(X, W):
    """
    提取CSP特征（log-variance）
    
    Args:
        X: (N, C, T) EEG数据
        W: (C, n_filters) CSP滤波器
    
    Returns:
        features: (N, n_filters) CSP特征
    """
    N, C, T = X.shape
    n_filters = W.shape[1]
    
    features = np.zeros((N, n_filters))
    for i in range(N):
        # 应用CSP滤波器
        X_filtered = W.T @ X[i]  # (n_filters, T)
        # 计算log-variance
        var = np.var(X_filtered, axis=1)
        features[i] = np.log(var + 1e-10)
    
    return features


def teacher_csp_lda(X, y, fs=250):
    """
    Teacher 1: CSP + LDA
    
    使用10折CV生成logits
    
    Args:
        X: (N, C, T) EA对齐后的EEG数据
        y: (N,) 标签
        fs: 采样率
    
    Returns:
        logits: (N, n_classes) 软标签
    """
    print("\n" + "="*80)
    print("Teacher 1: CSP + LDA")
    print("="*80)
    
    N, C, T = X.shape
    n_classes = len(np.unique(y))
    logits = np.zeros((N, n_classes))
    
    # 滤波到Mu (8-13Hz) 和 Beta (13-30Hz)
    print("  滤波到 Mu (8-13Hz) 和 Beta (13-30Hz)...")
    b_mu, a_mu = butter(4, [8, 13], btype='band', fs=fs)
    b_beta, a_beta = butter(4, [13, 30], btype='band', fs=fs)
    
    X_mu = np.zeros_like(X)
    X_beta = np.zeros_like(X)
    for i in range(N):
        for ch in range(C):
            X_mu[i, ch, :] = filtfilt(b_mu, a_mu, X[i, ch, :])
            X_beta[i, ch, :] = filtfilt(b_beta, a_beta, X[i, ch, :])
    
    # 10折CV
    print("  10折交叉验证生成logits...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"    Fold {fold+1}/10...")
        
        # CSP for Mu band
        W_mu = apply_csp(X_mu[train_idx], y[train_idx], n_components=3)
        feat_mu_train = extract_csp_features(X_mu[train_idx], W_mu)
        feat_mu_val = extract_csp_features(X_mu[val_idx], W_mu)
        
        # CSP for Beta band
        W_beta = apply_csp(X_beta[train_idx], y[train_idx], n_components=3)
        feat_beta_train = extract_csp_features(X_beta[train_idx], W_beta)
        feat_beta_val = extract_csp_features(X_beta[val_idx], W_beta)
        
        # 拼接特征
        X_train = np.hstack([feat_mu_train, feat_beta_train])
        X_val = np.hstack([feat_mu_val, feat_beta_val])
        
        # 训练LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y[train_idx])
        
        # 预测验证集logits
        logits[val_idx] = lda.predict_proba(X_val)
    
    # 验证准确率
    y_pred = np.argmax(logits, axis=1)
    acc = np.mean(y_pred == y)
    print(f"  ✅ CSP+LDA 准确率: {acc*100:.2f}%")
    
    return logits


# ============================================================
# Teacher 2: Riemannian Tangent Space + LR
# ============================================================

def compute_covariance_matrices(X):
    """
    计算协方差矩阵
    
    Args:
        X: (N, C, T) EEG数据
    
    Returns:
        covs: (N, C, C) 协方差矩阵
    """
    N, C, T = X.shape
    covs = np.zeros((N, C, C))
    
    for i in range(N):
        covs[i] = np.cov(X[i])
    
    return covs


def geometric_mean_covariance(covs):
    """
    计算协方差矩阵的几何平均
    
    Args:
        covs: (N, C, C) 协方差矩阵
    
    Returns:
        mean_cov: (C, C) 几何平均
    """
    N, C, _ = covs.shape
    
    # 简化版：使用算术平均近似
    mean_cov = np.mean(covs, axis=0)
    
    # 确保正定
    mean_cov = (mean_cov + mean_cov.T) / 2
    mean_cov += 1e-6 * np.eye(C)
    
    return mean_cov


def tangent_space_projection(covs, mean_cov):
    """
    投影到切空间
    
    Args:
        covs: (N, C, C) 协方差矩阵
        mean_cov: (C, C) 参考点（几何平均）
    
    Returns:
        tangent_vectors: (N, C*(C+1)//2) 切空间向量
    """
    N, C, _ = covs.shape
    
    # 计算参考点的逆平方根
    try:
        mean_cov_inv_sqrt = inv(sqrtm(mean_cov))
    except:
        # 如果失败，使用对角矩阵近似
        eigenvalues, eigenvectors = np.linalg.eigh(mean_cov)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        mean_cov_inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    
    tangent_vectors = []
    for i in range(N):
        # 投影: log(M^{-1/2} * C * M^{-1/2})
        transformed = mean_cov_inv_sqrt @ covs[i] @ mean_cov_inv_sqrt
        
        # 矩阵对数（简化：使用对角化）
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(transformed)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            log_mat = eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.T
        except:
            log_mat = np.log(np.maximum(transformed, 1e-10))
        
        # 向量化（仅取上三角）
        vec = log_mat[np.triu_indices(C)]
        tangent_vectors.append(vec)
    
    return np.array(tangent_vectors)


def teacher_riemannian_lr(X, y):
    """
    Teacher 2: Riemannian Tangent Space + LR
    
    Args:
        X: (N, C, T) EA对齐后的EEG数据
        y: (N,) 标签
    
    Returns:
        logits: (N, n_classes) 软标签
    """
    print("\n" + "="*80)
    print("Teacher 2: Riemannian Tangent Space + LR")
    print("="*80)
    
    N, C, T = X.shape
    n_classes = len(np.unique(y))
    logits = np.zeros((N, n_classes))
    
    print("  计算协方差矩阵...")
    covs = compute_covariance_matrices(X)
    
    # 10折CV
    print("  10折交叉验证生成logits...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"    Fold {fold+1}/10...")
        
        # 计算训练集的几何平均
        mean_cov = geometric_mean_covariance(covs[train_idx])
        
        # 投影到切空间
        X_train = tangent_space_projection(covs[train_idx], mean_cov)
        X_val = tangent_space_projection(covs[val_idx], mean_cov)
        
        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # 训练LR
        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X_train, y[train_idx])
        
        # 预测验证集logits
        logits[val_idx] = lr.predict_proba(X_val)
    
    # 验证准确率
    y_pred = np.argmax(logits, axis=1)
    acc = np.mean(y_pred == y)
    print(f"  ✅ Riemannian+LR 准确率: {acc*100:.2f}%")
    
    return logits


# ============================================================
# Teacher 3: CEMD IMF Energy + LR
# ============================================================

def extract_imf_energy_features(X, fs=250):
    """
    提取CEMD IMF能量特征
    
    Args:
        X: (N, C, T) EEG数据
        fs: 采样率
    
    Returns:
        features: (N, n_features) IMF能量特征
    """
    if not HAS_CEMD:
        print("  ⚠️ PyEMD未安装，使用简化的频带能量替代...")
        # 退化为简单的频带能量
        return extract_band_power_features(X, fs)
    
    N, C, T = X.shape
    
    # 对于每个通道，提取前5个IMF的能量
    n_imfs = 5
    features = np.zeros((N, C * n_imfs))
    
    print("  提取CEMD IMF能量特征...")
    for i in range(N):
        if (i + 1) % 50 == 0:
            print(f"    处理 {i+1}/{N}...")
        
        for ch in range(C):
            try:
                # CEMD分解
                cemd = CEMD()
                IMFs = cemd(X[i, ch, :])
                
                # 提取前n_imfs个IMF的能量
                for imf_idx in range(min(n_imfs, len(IMFs))):
                    energy = np.sum(IMFs[imf_idx] ** 2)
                    features[i, ch * n_imfs + imf_idx] = np.log(energy + 1e-10)
            except:
                # 如果CEMD失败，使用零填充
                features[i, ch * n_imfs:(ch + 1) * n_imfs] = 0
    
    return features


def extract_band_power_features(X, fs=250):
    """
    提取频带功率特征（CEMD的简化替代）
    
    Args:
        X: (N, C, T) EEG数据
        fs: 采样率
    
    Returns:
        features: (N, C*n_bands) 频带功率特征
    """
    N, C, T = X.shape
    
    # 5个频带
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 40)]
    n_bands = len(bands)
    
    features = np.zeros((N, C * n_bands))
    
    for i in range(N):
        for ch in range(C):
            for band_idx, (low, high) in enumerate(bands):
                # 滤波
                b, a = butter(4, [low, high], btype='band', fs=fs)
                filtered = filtfilt(b, a, X[i, ch, :])
                # 能量
                energy = np.sum(filtered ** 2)
                features[i, ch * n_bands + band_idx] = np.log(energy + 1e-10)
    
    return features


def teacher_cemd_lr(X, y, fs=250):
    """
    Teacher 3: CEMD IMF Energy + LR
    
    Args:
        X: (N, C, T) EA对齐后的EEG数据
        y: (N,) 标签
        fs: 采样率
    
    Returns:
        logits: (N, n_classes) 软标签
    """
    print("\n" + "="*80)
    print("Teacher 3: CEMD IMF Energy + LR")
    print("="*80)
    
    N, C, T = X.shape
    n_classes = len(np.unique(y))
    logits = np.zeros((N, n_classes))
    
    # 提取IMF能量特征
    features = extract_imf_energy_features(X, fs)
    
    # 10折CV
    print("  10折交叉验证生成logits...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"    Fold {fold+1}/10...")
        
        X_train = features[train_idx]
        X_val = features[val_idx]
        
        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # 训练LR
        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X_train, y[train_idx])
        
        # 预测验证集logits
        logits[val_idx] = lr.predict_proba(X_val)
    
    # 验证准确率
    y_pred = np.argmax(logits, axis=1)
    acc = np.mean(y_pred == y)
    print(f"  ✅ CEMD+LR 准确率: {acc*100:.2f}%")
    
    return logits


# ============================================================
# Teacher 4: STFT Band Power + LR
# ============================================================

def extract_stft_band_power(X, fs=250):
    """
    提取STFT频带功率特征
    
    Args:
        X: (N, C, T) EEG数据
        fs: 采样率
    
    Returns:
        features: (N, C*n_bands) STFT功率特征
    """
    N, C, T = X.shape
    
    # Mu (8-13Hz) 和 Beta (13-30Hz)
    mu_range = (8, 13)
    beta_range = (13, 30)
    
    features = np.zeros((N, C * 2))  # 每个通道2个频带
    
    print("  计算STFT频带功率...")
    for i in range(N):
        if (i + 1) % 50 == 0:
            print(f"    处理 {i+1}/{N}...")
        
        for ch in range(C):
            # STFT
            f, t, Zxx = stft(X[i, ch, :], fs=fs, nperseg=128, noverlap=64)
            
            # 功率谱
            power = np.abs(Zxx) ** 2
            
            # Mu频带功率
            mu_idx = np.where((f >= mu_range[0]) & (f <= mu_range[1]))[0]
            mu_power = np.mean(power[mu_idx, :])
            features[i, ch * 2] = np.log(mu_power + 1e-10)
            
            # Beta频带功率
            beta_idx = np.where((f >= beta_range[0]) & (f <= beta_range[1]))[0]
            beta_power = np.mean(power[beta_idx, :])
            features[i, ch * 2 + 1] = np.log(beta_power + 1e-10)
    
    return features


def teacher_stft_lr(X, y, fs=250):
    """
    Teacher 4: STFT Band Power + LR
    
    Args:
        X: (N, C, T) EA对齐后的EEG数据
        y: (N,) 标签
        fs: 采样率
    
    Returns:
        logits: (N, n_classes) 软标签
    """
    print("\n" + "="*80)
    print("Teacher 4: STFT Band Power + LR")
    print("="*80)
    
    N, C, T = X.shape
    n_classes = len(np.unique(y))
    logits = np.zeros((N, n_classes))
    
    # 提取STFT功率特征
    features = extract_stft_band_power(X, fs)
    
    # 10折CV
    print("  10折交叉验证生成logits...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"    Fold {fold+1}/10...")
        
        X_train = features[train_idx]
        X_val = features[val_idx]
        
        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # 训练LR
        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X_train, y[train_idx])
        
        # 预测验证集logits
        logits[val_idx] = lr.predict_proba(X_val)
    
    # 验证准确率
    y_pred = np.argmax(logits, axis=1)
    acc = np.mean(y_pred == y)
    print(f"  ✅ STFT+LR 准确率: {acc*100:.2f}%")
    
    return logits

