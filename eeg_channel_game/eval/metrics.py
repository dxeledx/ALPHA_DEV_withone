from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        from sklearn.metrics import cohen_kappa_score
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for kappa") from e
    return float(cohen_kappa_score(y_true, y_pred))

