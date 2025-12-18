# -*- coding: utf-8 -*-
"""
统一数据加载器 - V4.2/V4.2.1

整合功能:
1. 基础数据加载（load_train_eval_separately等）
2. 信号切片（slice_raw_signal）
3. 图构建（build_plv_adjacency_matrix）
4. DE特征计算（可选，保留向后兼容）

版本说明:
- 与V4.1完全相同，但为V4.2系列独立维护
- 保证每个版本都有完整的代码文件集

使用范围: V3.6, V4.1及后续版本
"""
import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import hilbert, butter, filtfilt, sosfilt
import torch
from tqdm import tqdm


# ============================================================
# 基础数据加载函数
# ============================================================

def _parse_subject_id(subject_id):
    """Convert 'A01' style subject id to integer 1..9."""
    if isinstance(subject_id, (int, np.integer)):
        return int(subject_id)
    sid = str(subject_id).strip().upper()
    if sid.startswith("A"):
        sid = sid[1:]
    return int(sid)


def _load_train_eval_separately_moabb(
    subject_id,
    *,
    trial_start_offset=2.0,
    trial_length=4.0,
    return_run_indices=False,
):
    """
    Load BNCI2014_001 via MOABB and return (X_train,y_train,X_eval,y_eval[,run_idx_train,run_idx_eval]).

    Notes:
      - Returns *raw* (unfiltered) EEG, consistent with the original .mat loader.
      - Uses stim-channel events (1..4) and crops [trial_start_offset, trial_start_offset+trial_length).
      - Ensures exactly T = trial_length * sfreq samples (e.g., 4s*250=1000).
    """
    try:
        import mne
        from moabb.datasets import BNCI2014_001
    except Exception as e:  # pragma: no cover
        raise RuntimeError("MOABB + MNE are required to load BNCI2014_001") from e

    subj = _parse_subject_id(subject_id)
    dataset = BNCI2014_001()
    data = dataset.get_data(subjects=[subj])[subj]

    def _epoch_session(session_key: str):
        session_runs = data[session_key]
        xs, ys, runs = [], [], []
        run_counter = 0
        for run_id in sorted(session_runs.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            raw = session_runs[run_id]

            stim_picks = mne.pick_types(raw.info, stim=True)
            stim_channel = raw.ch_names[int(stim_picks[0])] if len(stim_picks) else "stim"
            events = mne.find_events(raw, shortest_event=0, stim_channel=stim_channel, verbose=False)

            eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False)
            if len(eeg_picks) < 22:
                raise RuntimeError(f"Expected >=22 EEG channels, got {len(eeg_picks)}")

            sfreq = float(raw.info["sfreq"])
            tmin = float(trial_start_offset)
            tmax = float(trial_start_offset + trial_length - 1.0 / sfreq)

            epochs = mne.Epochs(
                raw,
                events,
                event_id=dataset.event_id,
                tmin=tmin,
                tmax=tmax,
                baseline=None,
                preload=True,
                picks=eeg_picks,
                detrend=None,
                verbose=False,
            )
            x = epochs.get_data().astype(np.float32, copy=False)  # [N,C,T]
            y = epochs.events[:, 2].astype(np.int64, copy=False)  # 1..4
            if x.size == 0:
                continue
            xs.append(x)
            ys.append(y)
            runs.append(np.full((x.shape[0],), run_counter, dtype=np.int64))
            run_counter += 1

        if not xs:
            return None, None, None
        X = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        run_idx = np.concatenate(runs, axis=0)
        return X, y, run_idx

    X_train, y_train, run_train = _epoch_session("0train")
    X_eval, y_eval, run_eval = _epoch_session("1test")
    if X_train is None or X_eval is None:
        raise RuntimeError(f"MOABB load produced empty data for subject {subject_id}")

    if return_run_indices:
        return X_train, y_train, X_eval, y_eval, run_train, run_eval
    return X_train, y_train, X_eval, y_eval


def load_train_eval_separately(mat_path, subject_id, trial_start_offset=2.0, trial_length=4.0, return_run_indices=False):
    """
    分别加载训练集(T)和评估集(E) - 返回原始未滤波数据

    ⚠️ 重要: 返回未滤波数据，由后续流程根据需要滤波
    - 避免双重滤波bug
    - 允许不同流使用不同滤波参数

    Args:
        mat_path: 数据目录路径
        subject_id: 被试ID (e.g., 'A01')
        trial_start_offset: MI开始时间（默认2.0秒）
        trial_length: MI持续时间（默认4.0秒）
        return_run_indices: 是否返回run索引（用于run级加权）

    Returns:
        X_train: (N_train, 22, 1000) 训练集原始EEG
        y_train: (N_train,) 训练集标签 (0-3)
        X_eval: (N_eval, 22, 1000) 评估集原始EEG
        y_eval: (N_eval,) 评估集标签 (0-3)
        run_indices_train: (N_train,) 训练集run索引 (0-5)，仅当return_run_indices=True时返回
        run_indices_eval: (N_eval,) 评估集run索引 (0-5)，仅当return_run_indices=True时返回
    """
    print(f"正在加载被试 {subject_id} (分离T和E)...")
    print(f"  ✅ MI时间窗: {trial_start_offset}s-{trial_start_offset+trial_length}s")
    print(f"  ⚠️ 返回原始信号（未滤波）")
    if return_run_indices:
        print(f"  📊 返回run索引（用于run级加权）")

    # Prefer local .mat if available; otherwise fall back to MOABB (BNCI2014_001).
    train_file = os.path.join(str(mat_path), f"{subject_id}T.mat")
    eval_file = os.path.join(str(mat_path), f"{subject_id}E.mat")
    use_mat = os.path.exists(train_file) and os.path.exists(eval_file)

    if use_mat:
        if return_run_indices:
            X_train, y_train, run_indices_train = extract_trials_from_mat(
                train_file, trial_start_offset, trial_length, return_run_indices=True
            )
        else:
            X_train, y_train = extract_trials_from_mat(
                train_file, trial_start_offset, trial_length, return_run_indices=False
            )
    else:
        print(f"  ℹ️ 未找到本地.mat：{train_file} / {eval_file}，改用 MOABB(BNCI2014_001) 加载。")
        if return_run_indices:
            X_train, y_train, X_eval, y_eval, run_indices_train, run_indices_eval = _load_train_eval_separately_moabb(
                subject_id,
                trial_start_offset=trial_start_offset,
                trial_length=trial_length,
                return_run_indices=True,
            )
        else:
            X_train, y_train, X_eval, y_eval = _load_train_eval_separately_moabb(
                subject_id,
                trial_start_offset=trial_start_offset,
                trial_length=trial_length,
                return_run_indices=False,
            )
    print(f"  T (训练集): {X_train.shape[0]} trials")
    if return_run_indices:
        print(f"    Run分布: {[np.sum(run_indices_train == r) for r in range(6)]}")

    # 加载评估集E（MOABB 分支已提前加载）
    if use_mat:
        if return_run_indices:
            X_eval, y_eval, run_indices_eval = extract_trials_from_mat(
                eval_file, trial_start_offset, trial_length, return_run_indices=True
            )
        else:
            X_eval, y_eval = extract_trials_from_mat(
                eval_file, trial_start_offset, trial_length, return_run_indices=False
            )
    print(f"  E (评估集): {X_eval.shape[0]} trials")
    if return_run_indices:
        print(f"    Run分布: {[np.sum(run_indices_eval == r) for r in range(6)]}")

    # 标签转换: 1-4 → 0-3
    y_train = y_train - 1
    y_eval = y_eval - 1

    print(f"  训练集: {X_train.shape}, 每类: {np.bincount(y_train)}")
    print(f"  评估集: {X_eval.shape}, 每类: {np.bincount(y_eval)}")
    print(f"  ✅ 返回原始未滤波数据")

    if return_run_indices:
        return X_train, y_train, X_eval, y_eval, run_indices_train, run_indices_eval
    return X_train, y_train, X_eval, y_eval


def load_single_session(mat_path, subject_id, session='T', trial_start_offset=2.0, trial_length=4.0, return_run_indices=False):
    """
    加载单个session的数据（T或E）

    Args:
        mat_path: 数据目录
        subject_id: 被试ID (e.g., 'A01')
        session: 'T' 或 'E'
        trial_start_offset: trial开始时间（秒）
        trial_length: trial长度（秒）
        return_run_indices: 是否返回run索引

    Returns:
        X_session: (N, 22, 1000)
        y_session: (N,)
        run_indices_session: (N,) 仅当return_run_indices=True时返回
    """
    session = session.upper()
    if session not in ['T', 'E']:
        raise ValueError(f"session必须为'T'或'E'，收到: {session}")

    print(f"正在加载被试 {subject_id} 的单个session: {session}")
    print(f"  ✅ MI时间窗: {trial_start_offset}s-{trial_start_offset+trial_length}s")
    print(f"  ⚠️ 返回原始信号（未滤波）")

    mat_file = os.path.join(str(mat_path), f"{subject_id}{session}.mat")
    use_mat = os.path.exists(mat_file)

    if use_mat:
        if return_run_indices:
            X_session, y_session, run_indices = extract_trials_from_mat(
                mat_file, trial_start_offset, trial_length, return_run_indices=True
            )
        else:
            X_session, y_session = extract_trials_from_mat(
                mat_file, trial_start_offset, trial_length, return_run_indices=False
            )
    else:
        print(f"  ℹ️ 未找到本地.mat：{mat_file}，改用 MOABB(BNCI2014_001) 加载。")
        X_train, y_train, X_eval, y_eval, run_train, run_eval = _load_train_eval_separately_moabb(
            subject_id,
            trial_start_offset=trial_start_offset,
            trial_length=trial_length,
            return_run_indices=True,
        )
        if session == "T":
            X_session, y_session, run_indices = X_train, y_train, run_train
        else:
            X_session, y_session, run_indices = X_eval, y_eval, run_eval

    if X_session is None:
        raise RuntimeError(f"{subject_id}{session} 无有效trial")

    y_session = y_session - 1

    print(f"  Session {session}: {X_session.shape[0]} trials, 每类: {np.bincount(y_session)}")
    if return_run_indices:
        print(f"    Run分布: {[np.sum(run_indices == r) for r in range(6)]}")

    if return_run_indices:
        return X_session, y_session, run_indices
    return X_session, y_session


def extract_trials_from_mat(mat_file, trial_start_offset, trial_length, return_run_indices=False):
    """
    从.mat文件提取trials

    Args:
        mat_file: .mat文件路径
        trial_start_offset: trial开始偏移（秒）
        trial_length: trial长度（秒）
        return_run_indices: 是否返回run索引

    Returns:
        X: (N, 22, T) EEG数据
        y: (N,) 标签（原始1-4）
        run_indices: (N,) run索引 (0-5)，仅当return_run_indices=True时返回
    """
    data = loadmat(mat_file)
    data_array = data['data']

    X_list, y_list, run_list = [], [], []
    run_counter = 0  # 用于连续编号run索引

    for session_idx in range(data_array.shape[1]):
        session = data_array[0, session_idx]

        X_continuous = session['X'][0, 0]  # (timepoints, channels)
        y_labels = session['y'][0, 0]  # (n_trials, 1)
        trial_starts = session['trial'][0, 0]  # (n_trials, 1)
        fs = int(session['fs'][0, 0][0, 0])

        if len(y_labels) == 0:
            continue

        offset_samples = int(trial_start_offset * fs)
        length_samples = int(trial_length * fs)

        # 只使用前22个通道（标准EEG通道）
        X_continuous = X_continuous[:, :22]

        for trial_idx in range(len(trial_starts)):
            trial_start = int(trial_starts[trial_idx, 0])

            start_idx = trial_start + offset_samples
            end_idx = start_idx + length_samples

            if end_idx <= X_continuous.shape[0]:
                # 转置: (T, C) → (C, T)
                trial_data = X_continuous[start_idx:end_idx, :].T
                X_list.append(trial_data)
                y_list.append(y_labels[trial_idx, 0])
                run_list.append(run_counter)  # 使用连续的run索引

        # 如果这个session有数据，run计数器+1
        if len(y_labels) > 0:
            run_counter += 1

    if len(X_list) == 0:
        if return_run_indices:
            return None, None, None
        return None, None

    if return_run_indices:
        return np.array(X_list), np.array(y_list), np.array(run_list)
    return np.array(X_list), np.array(y_list)


# ============================================================
# 信号切片函数（V3.6+）
# ============================================================

def slice_raw_signal(X, n_slices=5, verbose=True):
    """
    将原始EEG信号切片（用于GNN节点特征提取）
    
    Args:
        X: (N, C, T) 原始EEG数据
        n_slices: 切片数量
        
    Returns:
        X_sliced: (N, n_slices, C, slice_length)
    
    Example:
        X: (288, 22, 1000)
        n_slices=5
        → X_sliced: (288, 5, 22, 200)
    """
    N, C, T = X.shape
    slice_length = T // n_slices
    
    if verbose:
        print(f"  🔪 切片原始信号: {n_slices}片 × {slice_length}点/片")
    
    X_sliced = np.zeros((N, n_slices, C, slice_length), dtype=np.float32)
    
    for i in range(n_slices):
        start = i * slice_length
        end = (i + 1) * slice_length
        X_sliced[:, i, :, :] = X[:, :, start:end]
    
    if verbose:
        print(f"  ✅ 切片完成: {X_sliced.shape}")
    return X_sliced


# ============================================================
# 图构建函数
# ============================================================

def build_plv_adjacency_matrix(X, threshold=0.8):
    """
    构建PLV（相位锁定值）邻接矩阵
    
    Args:
        X: (N, C, T) EEG数据
        threshold: PLV阈值（默认0.8）
    
    Returns:
        plv_normalized: (C, C) 归一化PLV邻接矩阵
    """
    print(f"  构建PLV邻接矩阵 (threshold={threshold})...")
    
    n_trials, n_channels, n_timepoints = X.shape
    
    # 计算解析信号和瞬时相位
    X_analytic = hilbert(X, axis=2)
    instantaneous_phase = np.angle(X_analytic)
    
    # 初始化PLV矩阵
    plv_matrix = np.zeros((n_channels, n_channels))
    
    # 计算所有通道对的PLV
    for i in range(n_channels):
        for j in range(i, n_channels):
            if i == j:
                plv_matrix[i, j] = 1.0
                continue
            
            # 相位差
            phase_diff = instantaneous_phase[:, i, :] - instantaneous_phase[:, j, :]
            
            # PLV计算
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_matrix[i, j] = plv_matrix[j, i] = plv
    
    # 阈值处理
    plv_matrix[plv_matrix < threshold] = 0
    np.fill_diagonal(plv_matrix, 1.0)
    
    # 归一化: D^(-1/2) * A * D^(-1/2)
    D = np.sum(plv_matrix, axis=1)
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    D_matrix_inv_sqrt = np.diag(D_inv_sqrt)
    plv_normalized = np.matmul(np.matmul(D_matrix_inv_sqrt, plv_matrix), D_matrix_inv_sqrt)
    
    print(f"  ✅ PLV矩阵完成")
    
    return plv_normalized


def normalize_adjacency_matrix(adj_matrix):
    """
    归一化邻接矩阵（使用PyTorch）
    
    Args:
        adj_matrix: numpy array或torch tensor
    
    Returns:
        normalized_adj: torch tensor
    """
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = torch.FloatTensor(adj_matrix)
    
    D = torch.sum(adj_matrix, dim=1)
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
    D_matrix_inv_sqrt = torch.diag(D_inv_sqrt)
    
    return torch.matmul(torch.matmul(D_matrix_inv_sqrt, adj_matrix), D_matrix_inv_sqrt)


# ============================================================
# DE特征计算（保留向后兼容，V3.6+不使用）
# ============================================================

def compute_de_features(X, fs=250, cache_file=None):
    """
    预计算微分熵（DE）特征
    
    注意: V3.6/V4.1不使用DE特征，此函数保留向后兼容
    
    Args:
        X: (n_trials, n_channels, n_timepoints) EEG数据
        fs: 采样率
        cache_file: 缓存文件路径
    
    Returns:
        de_features: (n_trials, n_channels, 5) 5个频带的DE特征
    """
    # 检查缓存
    if cache_file and os.path.exists(cache_file):
        print(f"  ✅ 从缓存加载DE特征: {cache_file}")
        return np.load(cache_file)
    
    print(f"  🔧 预计算DE特征（5频带）...")
    
    # 5个频带：δ, θ, μ, β, γ
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 40)]
    
    n_trials, n_channels, n_timepoints = X.shape
    de_features = np.zeros((n_trials, n_channels, len(bands)), dtype=np.float32)
    
    # 设计滤波器
    nyq = fs / 2
    sos_filters = []
    for low, high in bands:
        sos = butter(4, [low/nyq, high/nyq], btype='band', output='sos')
        sos_filters.append(sos)
    
    # 批量计算
    for trial in range(n_trials):
        if trial % 100 == 0:
            print(f"    处理 trial {trial}/{n_trials}...")
        
        for ch in range(n_channels):
            sig = X[trial, ch, :]
            
            for band_idx, sos in enumerate(sos_filters):
                # 带通滤波
                filtered = sosfilt(sos, sig)
                
                # 微分熵: 0.5 * log(2π e σ²)
                var = np.var(filtered)
                de = 0.5 * np.log(2 * np.pi * np.e * (var + 1e-10))
                de_features[trial, ch, band_idx] = de
    
    # 保存缓存
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.save(cache_file, de_features)
        print(f"  ✅ DE特征已保存: {cache_file}")
    
    print(f"  ✅ DE特征计算完成: {de_features.shape}")
    
    return de_features


def compute_sliced_de_features(X, fs=250, n_slices=5, cache_file=None):
    """
    预计算切片的微分熵（DE）特征
    
    注意: V3.6/V4.1不使用，此函数保留向后兼容
    
    Args:
        X: (n_trials, n_channels, n_timepoints) EEG数据
        fs: 采样率
        n_slices: 切片数量
        cache_file: 缓存文件路径
    
    Returns:
        de_features_sliced: (n_trials, n_slices, n_channels, 5)
    """
    # 检查缓存
    if cache_file and os.path.exists(cache_file):
        print(f"  ✅ 从缓存加载 Sliced DE 特征: {cache_file}")
        return np.load(cache_file)
    
    print(f"  🔧 预计算 Sliced DE 特征 ({n_slices} slices, 5 bands)...")
    
    # 5个频带
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 40)]
    
    n_trials, n_channels, n_timepoints = X.shape
    slice_length = n_timepoints // n_slices
    
    if slice_length == 0:
        raise ValueError(f"时间点 {n_timepoints} 太短，无法分为 {n_slices} 片")
    
    print(f"  每个切片长度: {slice_length} 个时间点 ({slice_length/fs:.2f}秒)")
    
    # 初始化结果
    de_features_sliced = np.zeros((n_trials, n_slices, n_channels, len(bands)), dtype=np.float32)
    
    # 设计滤波器
    nyq = fs / 2
    sos_filters = []
    for low, high in bands:
        low = max(low, 0.1)
        high = min(high, nyq - 0.1)
        if low >= high:
            continue
        sos = butter(4, [low/nyq, high/nyq], btype='band', output='sos')
        sos_filters.append(sos)
    
    # 批量计算
    for trial in tqdm(range(n_trials), desc="  计算Sliced DE"):
        for sl_idx in range(n_slices):
            start = sl_idx * slice_length
            end = (sl_idx + 1) * slice_length
            
            for ch in range(n_channels):
                sig_slice = X[trial, ch, start:end]
                
                for band_idx, sos in enumerate(sos_filters):
                    filtered = sosfilt(sos, sig_slice)
                    var = np.var(filtered)
                    de = 0.5 * np.log(2 * np.pi * np.e * (var + 1e-10))
                    de_features_sliced[trial, sl_idx, ch, band_idx] = de
    
    # 保存缓存
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.save(cache_file, de_features_sliced)
        print(f"  ✅ Sliced DE 特征已保存: {cache_file}")
    
    print(f"  ✅ Sliced DE 特征计算完成: {de_features_sliced.shape}")
    
    return de_features_sliced


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("="*80)
    print("🧪 统一数据加载器测试")
    print("="*80)
    
    # 测试数据加载
    print("\n1. 测试数据加载...")
    X_train, y_train, X_eval, y_eval = load_train_eval_separately(
        'BCIIV2a_mat', 'A01'
    )
    print(f"✅ 数据加载成功")
    print(f"   训练集: {X_train.shape}")
    print(f"   评估集: {X_eval.shape}")
    
    # 测试信号切片
    print("\n2. 测试信号切片...")
    X_train_sliced = slice_raw_signal(X_train, n_slices=5)
    print(f"✅ 切片成功: {X_train_sliced.shape}")
    
    # 测试PLV图构建
    print("\n3. 测试PLV图构建...")
    adj_plv = build_plv_adjacency_matrix(X_train, threshold=0.8)
    print(f"✅ PLV矩阵构建成功: {adj_plv.shape}")
    
    # 测试归一化
    print("\n4. 测试邻接矩阵归一化...")
    adj_norm = normalize_adjacency_matrix(adj_plv)
    print(f"✅ 归一化成功: {adj_norm.shape}, type: {type(adj_norm)}")
    
    print("\n" + "="*80)
    print("✅ 所有测试通过！")
    print("="*80)
