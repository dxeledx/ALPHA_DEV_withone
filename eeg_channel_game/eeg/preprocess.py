from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EOGRegression:
    coef: np.ndarray  # [n_eog, n_eeg]
    intercept: np.ndarray  # [n_eeg]

    def apply(self, eeg: np.ndarray, eog: np.ndarray) -> np.ndarray:
        """
        eeg: float32 [N, n_eeg, T]
        eog: float32 [N, n_eog, T]
        """
        if eeg.ndim != 3 or eog.ndim != 3:
            raise ValueError("eeg/eog must be 3D arrays: [N, C, T]")
        if eeg.shape[0] != eog.shape[0] or eeg.shape[2] != eog.shape[2]:
            raise ValueError("eeg/eog must share N and T dims")
        n = eeg.shape[0]
        n_eeg = eeg.shape[1]
        n_eog = eog.shape[1]
        if self.coef.shape != (n_eog, n_eeg):
            raise ValueError(f"coef shape mismatch: {self.coef.shape} vs {(n_eog, n_eeg)}")
        if self.intercept.shape != (n_eeg,):
            raise ValueError(f"intercept shape mismatch: {self.intercept.shape} vs {(n_eeg,)}")

        x = eog.transpose(0, 2, 1).reshape(-1, n_eog)  # [N*T, n_eog]
        y = eeg.transpose(0, 2, 1).reshape(-1, n_eeg)  # [N*T, n_eeg]
        y_hat = x @ self.coef + self.intercept[None, :]
        y_clean = y - y_hat
        return y_clean.reshape(n, -1, n_eeg).transpose(0, 2, 1).astype(np.float32, copy=False)


def fit_eog_regression(eeg: np.ndarray, eog: np.ndarray, ridge: float = 1e-3) -> EOGRegression:
    """
    Fit EEG <- EOG linear regression (with intercept), then EEG_clean = EEG - (EOG @ coef + intercept).
    Uses only the provided data (caller must ensure no eval-session leakage).
    """
    if eeg.ndim != 3 or eog.ndim != 3:
        raise ValueError("eeg/eog must be 3D arrays: [N, C, T]")
    if eeg.shape[0] != eog.shape[0] or eeg.shape[2] != eog.shape[2]:
        raise ValueError("eeg/eog must share N and T dims")
    n_eeg = eeg.shape[1]
    n_eog = eog.shape[1]

    x = eog.transpose(0, 2, 1).reshape(-1, n_eog).astype(np.float64, copy=False)
    y = eeg.transpose(0, 2, 1).reshape(-1, n_eeg).astype(np.float64, copy=False)

    ones = np.ones((x.shape[0], 1), dtype=x.dtype)
    x_aug = np.concatenate([x, ones], axis=1)  # [N*T, n_eog+1]

    xtx = x_aug.T @ x_aug
    xtx += ridge * np.eye(xtx.shape[0], dtype=xtx.dtype)
    xty = x_aug.T @ y
    w = np.linalg.solve(xtx, xty)  # [n_eog+1, n_eeg]

    coef = w[:-1, :].astype(np.float32)
    intercept = w[-1, :].astype(np.float32)
    return EOGRegression(coef=coef, intercept=intercept)


def mean_abs_eeg_eog_corr(eeg: np.ndarray, eog: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Returns mean absolute correlation with EOG per EEG channel.
    eeg: [N, n_eeg, T]
    eog: [N, n_eog, T]
    -> [n_eeg]
    """
    if eeg.ndim != 3 or eog.ndim != 3:
        raise ValueError("eeg/eog must be 3D arrays: [N, C, T]")
    if eeg.shape[0] != eog.shape[0] or eeg.shape[2] != eog.shape[2]:
        raise ValueError("eeg/eog must share N and T dims")

    x_eeg = eeg.transpose(0, 2, 1).reshape(-1, eeg.shape[1]).astype(np.float64, copy=False)
    x_eog = eog.transpose(0, 2, 1).reshape(-1, eog.shape[1]).astype(np.float64, copy=False)

    x_eeg = x_eeg - x_eeg.mean(axis=0, keepdims=True)
    x_eog = x_eog - x_eog.mean(axis=0, keepdims=True)

    x_eeg = x_eeg / (x_eeg.std(axis=0, keepdims=True) + eps)
    x_eog = x_eog / (x_eog.std(axis=0, keepdims=True) + eps)

    corr = (x_eeg.T @ x_eog) / max(1, x_eeg.shape[0] - 1)  # [n_eeg, n_eog]
    return np.mean(np.abs(corr), axis=1).astype(np.float32)

