from __future__ import annotations

import numpy as np


def csp_filters(C1: np.ndarray, C2: np.ndarray, m: int = 2, eps: float = 1e-6) -> np.ndarray:
    """
    Compute CSP spatial filters for a binary problem using mean covariances.
    Returns W of shape [2m, n_ch].
    """
    n_ch = int(C1.shape[0])
    C1 = C1 + eps * np.eye(n_ch, dtype=C1.dtype)
    C2 = C2 + eps * np.eye(n_ch, dtype=C2.dtype)
    Cc = C1 + C2

    d, U = np.linalg.eigh(Cc)
    idx = np.argsort(d)[::-1]
    d = d[idx]
    U = U[:, idx]

    # Whitening: P = diag(1/sqrt(d)) @ U^T
    P = (U / np.sqrt(d + eps)[None, :]).T
    S1 = P @ C1 @ P.T
    d1, B = np.linalg.eigh(S1)
    idx = np.argsort(d1)[::-1]
    B = B[:, idx]
    W = (B.T @ P).astype(np.float32)
    return np.concatenate([W[:m], W[-m:]], axis=0).astype(np.float32)


def fit_fbcsp_ovr_filters(
    cov_fb: np.ndarray,  # [B, N, C, C]
    y: np.ndarray,  # [N]
    train_idx: np.ndarray,  # [Ntr]
    sel: np.ndarray,  # [S]
    *,
    m: int = 2,
    eps: float = 1e-6,
    n_classes: int = 4,
) -> list[np.ndarray]:
    """
    Fit FilterBank-CSP filters for OVR multi-class:
      one CSP per (band, class).

    Returns list of W matrices with length B*n_classes, each [2m, S].
    """
    bands = int(cov_fb.shape[0])
    cov_train = cov_fb[:, train_idx][:, :, sel][:, :, :, sel].astype(np.float32, copy=False)  # [B, Ntr, S, S]
    y_train = y[train_idx]

    filters: list[np.ndarray] = []
    for b in range(bands):
        cov_b = cov_train[b]
        for c in range(int(n_classes)):
            c1 = cov_b[y_train == c].mean(axis=0)
            c2 = cov_b[y_train != c].mean(axis=0)
            filters.append(csp_filters(c1, c2, m=m, eps=eps))
    return filters


def transform_fbcsp_features(
    cov_fb: np.ndarray,  # [B, N, C, C]
    idx: np.ndarray,  # [N]
    sel: np.ndarray,  # [S]
    filters: list[np.ndarray],
    *,
    m: int = 2,
    eps: float = 1e-6,
    n_classes: int = 4,
) -> np.ndarray:
    """
    Transform covariances into FBCSP log-variance features using pre-fit filters.

    Returns X_feat: [N, B*n_classes*2m]
    """
    bands = int(cov_fb.shape[0])
    cov_sel = cov_fb[:, idx][:, :, sel][:, :, :, sel].astype(np.float32, copy=False)  # [B, N, S, S]

    feats = []
    fi = 0
    for b in range(bands):
        cov_b = cov_sel[b]
        for _c in range(int(n_classes)):
            W = filters[fi]
            fi += 1
            var = np.einsum("fi,nij,fj->nf", W, cov_b, W)  # [N, 2m]
            var = var / (var.sum(axis=1, keepdims=True) + eps)
            feats.append(np.log(var + eps))
    return np.concatenate(feats, axis=1).astype(np.float32)

