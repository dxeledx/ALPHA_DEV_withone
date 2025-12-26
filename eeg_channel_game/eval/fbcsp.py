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


def _apply_diag_gating_to_cov(
    cov: np.ndarray,  # [..., C, C]
    mask: np.ndarray,  # [C] in {0,1}
    *,
    mask_eps: float = 0.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Apply channel mask as a diagonal gate (W = diag(g)), where g = mask*(1-mask_eps)+mask_eps.
    Returns W*C*W, followed by trace normalization.
    """
    mask = mask.astype(np.float32, copy=False).reshape(-1)
    g = mask * (1.0 - float(mask_eps)) + float(mask_eps)
    w = (g[:, None] * g[None, :]).astype(np.float32, copy=False)  # [C, C]
    cov = cov.astype(np.float32, copy=False) * w  # broadcast on leading dims
    tr = np.einsum("...cc->...", cov).astype(np.float32, copy=False)
    cov = cov / (tr[..., None, None] + float(eps))
    return cov.astype(np.float32, copy=False)


def _shrinkage_cov(C: np.ndarray, lam: float) -> np.ndarray:
    lam = float(lam)
    if lam <= 0.0:
        return C
    n_ch = int(C.shape[0])
    tr = float(np.trace(C))
    return ((1.0 - lam) * C + lam * (tr / float(n_ch)) * np.eye(n_ch, dtype=C.dtype)).astype(np.float32)


def csp_filters_mask_penalized(
    C1: np.ndarray,
    C2: np.ndarray,
    *,
    mask: np.ndarray,  # [C] in {0,1}
    n_filters: int = 2,
    eps: float = 1e-6,
    shrinkage: float = 0.1,
    ridge: float = 1e-3,
    mask_penalty: float = 0.1,
) -> np.ndarray:
    """
    Mask-aware regularized CSP:
      - optional covariance shrinkage (towards scaled identity)
      - ridge for numerical stability
      - mask-penalty term beta*P (P=diag(1-mask)) added to both classes
    """
    n_ch = int(C1.shape[0])
    mask = mask.astype(np.float32, copy=False).reshape(n_ch)
    P = np.diag(1.0 - mask).astype(np.float32, copy=False)

    C1 = 0.5 * (C1 + C1.T)
    C2 = 0.5 * (C2 + C2.T)
    C1 = _shrinkage_cov(C1.astype(np.float32, copy=False), shrinkage)
    C2 = _shrinkage_cov(C2.astype(np.float32, copy=False), shrinkage)
    C1 = C1 + float(ridge) * np.eye(n_ch, dtype=np.float32)
    C2 = C2 + float(ridge) * np.eye(n_ch, dtype=np.float32)
    if float(mask_penalty) > 0.0:
        C1 = C1 + float(mask_penalty) * P
        C2 = C2 + float(mask_penalty) * P
    return csp_filters(C1, C2, m=int(n_filters), eps=float(eps))


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


def fit_mr_fbcsp_ovr_filters(
    cov_fb: np.ndarray,  # [B, N, C, C]
    y: np.ndarray,  # [N]
    train_idx: np.ndarray,  # [Ntr]
    *,
    mask: np.ndarray,  # [C] in {0,1}
    n_filters: int = 2,
    eps: float = 1e-6,
    mask_eps: float = 0.0,
    shrinkage: float = 0.1,
    ridge: float = 1e-3,
    mask_penalty: float = 0.1,
    n_classes: int = 4,
) -> list[np.ndarray]:
    """
    MR-FBCSP (mask-aware regularized FBCSP) for OVR multi-class:
      - fixed C=22 dimensionality
      - masked covariances (W*C*W with trace renorm)
      - shrinkage + ridge
      - mask-penalized CSP
    """
    bands = int(cov_fb.shape[0])
    cov_train = cov_fb[:, train_idx].astype(np.float32, copy=False)  # [B, Ntr, C, C]
    cov_train = _apply_diag_gating_to_cov(cov_train, mask, mask_eps=float(mask_eps), eps=float(eps))
    y_train = y[train_idx]

    filters: list[np.ndarray] = []
    for b in range(bands):
        cov_b = cov_train[b]
        for c in range(int(n_classes)):
            c1 = cov_b[y_train == c].mean(axis=0)
            c2 = cov_b[y_train != c].mean(axis=0)
            filters.append(
                csp_filters_mask_penalized(
                    c1,
                    c2,
                    mask=mask,
                    n_filters=int(n_filters),
                    eps=float(eps),
                    shrinkage=float(shrinkage),
                    ridge=float(ridge),
                    mask_penalty=float(mask_penalty),
                )
            )
    return filters


def transform_mr_fbcsp_features(
    cov_fb: np.ndarray,  # [B, N, C, C]
    idx: np.ndarray,  # [N]
    *,
    mask: np.ndarray,  # [C] in {0,1}
    filters: list[np.ndarray],
    n_filters: int = 2,
    eps: float = 1e-6,
    mask_eps: float = 0.0,
    n_classes: int = 4,
) -> np.ndarray:
    """
    Transform (masked) covariances into MR-FBCSP log-variance features.
    Returns X_feat: [N, B*n_classes*2m]
    """
    bands = int(cov_fb.shape[0])
    cov_sel = cov_fb[:, idx].astype(np.float32, copy=False)  # [B, N, C, C]
    cov_sel = _apply_diag_gating_to_cov(cov_sel, mask, mask_eps=float(mask_eps), eps=float(eps))

    feats = []
    fi = 0
    for b in range(bands):
        cov_b = cov_sel[b]
        for _c in range(int(n_classes)):
            W = filters[fi]
            fi += 1
            var = np.einsum("fi,nij,fj->nf", W, cov_b, W)  # [N, 2m]
            var = var / (var.sum(axis=1, keepdims=True) + float(eps))
            feats.append(np.log(var + float(eps)))
    return np.concatenate(feats, axis=1).astype(np.float32)
