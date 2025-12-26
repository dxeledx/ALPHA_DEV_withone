from __future__ import annotations

from typing import Any

import numpy as np

from eeg_channel_game.eeg.fold_sampler import FoldData
from eeg_channel_game.eval.evaluator_base import EvaluatorBase
from eeg_channel_game.utils.bitmask import popcount


class L0LrWeightEvaluator(EvaluatorBase):
    """
    L0 proxy based on embedded logistic-regression channel importance (lr_weight).

    Motivation: lr_weight is a strong embedded baseline in this project; using it as the
    cheap proxy for early MCTS/value bootstrapping can improve sample-efficiency under
    sparse terminal rewards.
    """

    def __init__(
        self,
        *,
        lambda_cost: float = 0.05,
        beta_redund: float = 0.2,
        artifact_gamma: float = 0.0,
    ):
        self.lambda_cost = float(lambda_cost)
        self.beta_redund = float(beta_redund)
        self.artifact_gamma = float(artifact_gamma)
        self._cache: dict[tuple[int, int, int], tuple[float, dict[str, Any]]] = {}

    def evaluate(self, key: int, fold: FoldData) -> tuple[float, dict[str, Any]]:
        cache_key = (fold.subject, fold.split_id, int(key))
        if cache_key in self._cache:
            return self._cache[cache_key]

        stats = fold.stats
        n_ch = popcount(key)
        if n_ch == 0:
            out = (-1.0, {"n_ch": 0, "relevance": 0.0, "redund": 0.0, "artifact": 0.0})
            self._cache[cache_key] = out
            return out

        sel = np.array([(key >> i) & 1 for i in range(22)], dtype=bool)

        # lr_weight is non-negative and can be heavy-tailed; log1p stabilizes scale.
        ch_score = np.log1p(np.maximum(stats.lr_weight, 0.0)).astype(np.float32, copy=False)  # [22]
        relevance = float(ch_score[sel].mean())

        if n_ch <= 1:
            redund = 0.0
        else:
            sidx = np.where(sel)[0]
            mat = stats.redund_corr[np.ix_(sidx, sidx)].astype(np.float64, copy=False)
            triu = mat[np.triu_indices_from(mat, k=1)]
            redund = float(triu.mean()) if triu.size else 0.0

        artifact = float(stats.artifact_corr_eog[sel].mean()) if sel.any() else 0.0
        cost = float(n_ch) / 22.0

        reward = relevance - self.beta_redund * redund - self.lambda_cost * cost - self.artifact_gamma * artifact
        info: dict[str, Any] = {
            "n_ch": n_ch,
            "relevance": relevance,
            "redund": redund,
            "artifact": artifact,
            "cost": cost,
            "reward": reward,
        }
        out = (float(reward), info)
        self._cache[cache_key] = out
        return out

