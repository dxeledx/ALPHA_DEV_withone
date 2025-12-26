from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeepPreprocessConfig:
    """
    Preprocessing aligned with common Braindecode baselines:
      - optional CAR (common average reference)
      - exponential moving standardization (recommended) or z-score

    Note: we apply this on epoched trials [N, C, T] for simplicity.
    """

    # Referencing / scaling
    use_car: bool = True
    scale: float = 1.0

    # Standardization
    standardize: str = "exp_moving"  # exp_moving | zscore | none

    # exponential moving standardization (Braindecode-style)
    em_factor_new: float = 0.001
    em_init_block_size: int = 100
    em_eps: float = 1e-4


def _apply_car(x: np.ndarray) -> np.ndarray:
    # x: [N, C, T]
    return (x - x.mean(axis=1, keepdims=True)).astype(np.float32, copy=False)


def _exponential_moving_standardize_np(
    x: np.ndarray, *, factor_new: float, init_block_size: int, eps: float
) -> np.ndarray:
    """
    Simple per-trial, per-channel exponential moving standardization.
    x: [N, C, T] float32
    """
    x = x.astype(np.float32, copy=False)
    n, c, t = x.shape
    out = np.empty_like(x, dtype=np.float32)

    init = int(init_block_size)
    init = max(1, min(init, t))
    alpha = float(factor_new)
    eps = float(eps)

    for i in range(n):
        xi = x[i]  # [C, T]
        oi = out[i]
        mean = xi[:, :init].mean(axis=1, keepdims=True)
        var = xi[:, :init].var(axis=1, keepdims=True)
        oi[:, :init] = (xi[:, :init] - mean) / np.sqrt(var + eps)

        m = mean[:, 0]
        v = var[:, 0]
        for tt in range(init, t):
            xt = xi[:, tt]
            m = (1.0 - alpha) * m + alpha * xt
            v = (1.0 - alpha) * v + alpha * (xt - m) ** 2
            oi[:, tt] = (xt - m) / np.sqrt(v + eps)

    return out


def _maybe_exponential_moving_standardize(
    x: np.ndarray, *, factor_new: float, init_block_size: int, eps: float
) -> np.ndarray:
    """
    Prefer Braindecode's official implementation when available.
    """
    try:
        from braindecode.preprocessing import exponential_moving_standardize

        out = np.empty_like(x, dtype=np.float32)
        for i in range(x.shape[0]):
            out[i] = exponential_moving_standardize(
                x[i].astype(np.float32, copy=False),
                factor_new=float(factor_new),
                init_block_size=int(init_block_size),
                eps=float(eps),
            ).astype(np.float32, copy=False)
        return out
    except Exception:
        return _exponential_moving_standardize_np(
            x, factor_new=float(factor_new), init_block_size=int(init_block_size), eps=float(eps)
        )


def apply_deep_preprocess_splits(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_eval: np.ndarray,
    *,
    cfg: DeepPreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the same preprocessing to (train, val, eval). For z-score, statistics
    are computed on x_train only.
    """
    x_train = x_train.astype(np.float32, copy=False)
    x_val = x_val.astype(np.float32, copy=False)
    x_eval = x_eval.astype(np.float32, copy=False)

    if float(cfg.scale) != 1.0:
        x_train = (x_train * float(cfg.scale)).astype(np.float32, copy=False)
        x_val = (x_val * float(cfg.scale)).astype(np.float32, copy=False)
        x_eval = (x_eval * float(cfg.scale)).astype(np.float32, copy=False)

    if bool(cfg.use_car):
        x_train = _apply_car(x_train)
        x_val = _apply_car(x_val)
        x_eval = _apply_car(x_eval)

    mode = str(cfg.standardize).lower()
    if mode in {"exp", "exp_moving", "exp-moving", "ema"}:
        x_train = _maybe_exponential_moving_standardize(
            x_train,
            factor_new=float(cfg.em_factor_new),
            init_block_size=int(cfg.em_init_block_size),
            eps=float(cfg.em_eps),
        )
        x_val = _maybe_exponential_moving_standardize(
            x_val,
            factor_new=float(cfg.em_factor_new),
            init_block_size=int(cfg.em_init_block_size),
            eps=float(cfg.em_eps),
        )
        x_eval = _maybe_exponential_moving_standardize(
            x_eval,
            factor_new=float(cfg.em_factor_new),
            init_block_size=int(cfg.em_init_block_size),
            eps=float(cfg.em_eps),
        )
    elif mode in {"z", "zscore", "standard", "standardize"}:
        mu = x_train.mean(axis=(0, 2), keepdims=True)
        sd = x_train.std(axis=(0, 2), keepdims=True) + 1e-6
        x_train = ((x_train - mu) / sd).astype(np.float32, copy=False)
        x_val = ((x_val - mu) / sd).astype(np.float32, copy=False)
        x_eval = ((x_eval - mu) / sd).astype(np.float32, copy=False)
    elif mode in {"none", "no"}:
        pass
    else:
        raise ValueError(f"Unknown standardize mode: {cfg.standardize!r}")

    return x_train, x_val, x_eval

