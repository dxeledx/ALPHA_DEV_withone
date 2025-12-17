from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eeg_channel_game.eeg.features import mean_abs_corr_from_bandpower


@dataclass(frozen=True)
class FoldStats:
    bp_mean: np.ndarray  # [22, B]
    bp_std: np.ndarray  # [22, B]
    fisher: np.ndarray  # [22, B]
    redund_corr: np.ndarray  # [22, 22]
    quality_mean: np.ndarray  # [22, Q]
    artifact_corr_eog: np.ndarray  # [22]
    resid_ratio: np.ndarray  # [22]


def compute_fold_stats(
    *,
    bp: np.ndarray,  # [N, 22, B] (train-session subset)
    q: np.ndarray,  # [N, 22, Q]
    y: np.ndarray,  # [N]
    artifact_corr_eog: np.ndarray,  # [22]
    resid_ratio: np.ndarray,  # [22]
) -> FoldStats:
    if bp.ndim != 3:
        raise ValueError("bp must be [N, 22, B]")
    if q.ndim != 3:
        raise ValueError("q must be [N, 22, Q]")
    if bp.shape[0] != q.shape[0] or bp.shape[0] != y.shape[0]:
        raise ValueError("bp/q/y must share N")
    if bp.shape[1] != 22:
        raise ValueError("Expected 22 EEG channels")

    bp_mean = bp.mean(axis=0).astype(np.float32)
    bp_std = bp.std(axis=0).astype(np.float32)
    quality_mean = q.mean(axis=0).astype(np.float32)

    # Fisher score (ANOVA F) on flattened bandpower features, then reshape.
    try:
        from sklearn.feature_selection import f_classif
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for fisher scores") from e

    x_feat = bp.reshape(bp.shape[0], -1)
    f, _ = f_classif(x_feat, y)
    fisher = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).reshape(22, -1).astype(np.float32)

    redund_corr = mean_abs_corr_from_bandpower(bp)

    return FoldStats(
        bp_mean=bp_mean,
        bp_std=bp_std,
        fisher=fisher,
        redund_corr=redund_corr,
        quality_mean=quality_mean,
        artifact_corr_eog=artifact_corr_eog.astype(np.float32, copy=False),
        resid_ratio=resid_ratio.astype(np.float32, copy=False),
    )

