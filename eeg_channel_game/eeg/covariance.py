from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_cov_fb(
    x: np.ndarray,
    sfreq: float,
    bands: Iterable[tuple[float, float]],
    *,
    iir_order: int = 4,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute per-epoch normalized covariance matrices for each frequency band.

    x: float32 [N, C, T]  (EEG only)
    returns: float32 [B, N, C, C]
    """
    try:
        import mne
    except Exception as e:  # pragma: no cover
        raise RuntimeError("mne is required for compute_cov_fb") from e

    if x.ndim != 3:
        raise ValueError("x must be [N, C, T]")
    bands = list(bands)
    n_epochs, n_ch, _ = x.shape
    cov_fb = np.empty((len(bands), n_epochs, n_ch, n_ch), dtype=np.float32)
    iir_params = dict(order=iir_order, ftype="butter")

    for bi, (l_freq, h_freq) in enumerate(bands):
        xf = mne.filter.filter_data(
            x.astype(np.float64, copy=False),
            sfreq=float(sfreq),
            l_freq=float(l_freq),
            h_freq=float(h_freq),
            method="iir",
            iir_params=iir_params,
            verbose=False,
        ).astype(np.float32, copy=False)
        cov = np.einsum("nct,ndt->ncd", xf, xf).astype(np.float64, copy=False)  # [N, C, C]
        tr = np.einsum("ncc->n", cov)
        cov = cov / (tr[:, None, None] + eps)
        cov_fb[bi] = cov.astype(np.float32)

    return cov_fb
