from __future__ import annotations

import numpy as np


def riemann_ts_lr_channel_scores(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cov_estimator: str = "oas",
    ts_metric: str = "riemann",
    c: float = 1.0,
    max_iter: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """
    Riemannian (covariance + tangent space) channel importance.

    This computes SPD covariance matrices from time-domain epochs, maps them to
    tangent space, fits a linear classifier (LogReg), and converts the learned
    weights back to a symmetric matrix to derive per-channel importance scores.

    Parameters
    ----------
    X:
        EEG epochs, shape [N, C, T] (N trials, C channels, T timepoints).
    y:
        Labels, shape [N].
    cov_estimator:
        Covariance estimator for pyriemann.estimation.Covariances (default: "oas").
    ts_metric:
        Metric for tangent space mapping (default: "riemann").
    c, max_iter, seed:
        LogisticRegression hyperparameters.

    Returns
    -------
    scores:
        Non-negative channel scores, shape [C]. Larger => more important.
    """
    if X.ndim != 3:
        raise ValueError("X must be [N, C, T]")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be [N] and match X.shape[0]")
    n_ch = int(X.shape[1])
    if n_ch <= 0:
        raise ValueError("X must have at least 1 channel")

    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from pyriemann.utils.tangentspace import unupper
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pyriemann is required for riemann_ts_lr baseline") from e

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for riemann_ts_lr baseline") from e

    cov = Covariances(estimator=str(cov_estimator)).transform(X.astype(np.float32, copy=False))
    ts = TangentSpace(metric=str(ts_metric)).fit_transform(cov)

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(
            penalty="l2",
            C=float(c),
            max_iter=int(max_iter),
            solver="lbfgs",
            random_state=int(seed),
        ),
    )
    clf.fit(ts.astype(np.float32, copy=False), y.astype(np.int64, copy=False))
    lr: LogisticRegression = clf[-1]
    coef = np.asarray(lr.coef_, dtype=np.float64, order="C")

    # Convert weight vectors back to symmetric matrices and aggregate across classes.
    score_mat = np.zeros((n_ch, n_ch), dtype=np.float64)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    for cvec in coef:
        w = unupper(cvec)
        score_mat += w * w

    ch_scores = np.sqrt(np.sum(score_mat, axis=1)).astype(np.float32, copy=False)
    ch_scores = np.nan_to_num(ch_scores, nan=0.0, posinf=0.0, neginf=0.0)
    return ch_scores
