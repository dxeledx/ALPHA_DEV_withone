from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    train_idx: np.ndarray  # indices into training-session trials
    val_idx: np.ndarray


def make_stratified_splits(
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> list[Split]:
    try:
        from sklearn.model_selection import StratifiedKFold
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for splits") from e

    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits: list[Split] = []
    for tr, va in skf.split(np.zeros_like(y), y):
        splits.append(Split(train_idx=tr.astype(np.int64), val_idx=va.astype(np.int64)))
    return splits

