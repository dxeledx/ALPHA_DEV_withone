from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_bandpower_fft(
    x: np.ndarray,
    sfreq: float,
    bands: Iterable[tuple[float, float]],
    eps: float = 1e-12,
) -> np.ndarray:
    """
    x: float32 [N, C, T]
    returns: float32 [N, C, B] log-bandpower
    """
    if x.ndim != 3:
        raise ValueError("x must be [N, C, T]")
    n_times = x.shape[-1]
    window = np.hanning(n_times).astype(np.float32)
    window_power = float(np.sum(window**2))
    xw = x * window[None, None, :]

    spec = np.fft.rfft(xw, axis=-1)
    psd = (np.abs(spec) ** 2) / (sfreq * window_power + eps)
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)

    bands = list(bands)
    out = np.empty((x.shape[0], x.shape[1], len(bands)), dtype=np.float32)
    for bi, (fmin, fmax) in enumerate(bands):
        idx = (freqs >= float(fmin)) & (freqs < float(fmax))
        if not np.any(idx):
            raise ValueError(f"No FFT bins for band [{fmin}, {fmax}) at sfreq={sfreq} n_times={n_times}")
        out[..., bi] = np.log(psd[..., idx].mean(axis=-1) + eps).astype(np.float32)
    return out


def compute_quality_features(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Minimal per-epoch quality features.
    x: float32 [N, C, T]
    returns: float32 [N, C, Q] with Q=2: [logvar, kurtosis]
    """
    if x.ndim != 3:
        raise ValueError("x must be [N, C, T]")
    xc = x - x.mean(axis=-1, keepdims=True)
    m2 = np.mean(xc**2, axis=-1)  # [N, C]
    m4 = np.mean(xc**4, axis=-1)  # [N, C]
    kurtosis = m4 / (m2**2 + eps)
    logvar = np.log(m2 + eps)
    return np.stack([logvar, kurtosis], axis=-1).astype(np.float32)


def mean_abs_corr_from_bandpower(bp: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Redundancy proxy: average absolute Pearson correlation between channels,
    computed on bandpower features.

    bp: float32 [N, C, B]
    returns: float32 [C, C]
    """
    if bp.ndim != 3:
        raise ValueError("bp must be [N, C, B]")
    n_trials, n_ch, n_bands = bp.shape
    corr_sum = np.zeros((n_ch, n_ch), dtype=np.float64)
    for b in range(n_bands):
        xb = bp[:, :, b].astype(np.float64, copy=False)  # [N, C]
        xb = xb - xb.mean(axis=0, keepdims=True)
        xb = xb / (xb.std(axis=0, keepdims=True) + eps)
        corr = (xb.T @ xb) / max(1, n_trials - 1)
        corr_sum += np.abs(corr)
    corr_mean = corr_sum / float(n_bands)
    np.fill_diagonal(corr_mean, 0.0)
    return corr_mean.astype(np.float32)

