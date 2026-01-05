from __future__ import annotations

from typing import Any

import numpy as np

from eeg_channel_game.eeg.fold_sampler import FoldData
from eeg_channel_game.eval.evaluator_base import EvaluatorBase
from eeg_channel_game.utils.bitmask import mask_to_key, popcount


class DeltaFull22Evaluator(EvaluatorBase):
    """
    Reward normalization wrapper for multi-subject training.

    For a given (subject, split), subtract the base evaluator's reward on the full-22 set:
      r_delta(key) = r_base(key) - r_base(full22)

    This keeps within-subject ordering unchanged (constant shift) while aligning reward scales
    across subjects, which is critical for a single shared policy/value network.
    """

    def __init__(self, base: EvaluatorBase, *, baseline_key: int = (1 << 22) - 1):
        self.base = base
        self.baseline_key = int(baseline_key)
        self._cache: dict[tuple[int, int, int], tuple[float, dict[str, Any]]] = {}

    def evaluate(self, key: int, fold: FoldData) -> tuple[float, dict[str, Any]]:
        cache_key = (fold.subject, fold.split_id, int(key))
        if cache_key in self._cache:
            return self._cache[cache_key]

        r_raw, info_raw = self.base.evaluate(int(key), fold)
        r0, _ = self.base.evaluate(self.baseline_key, fold)

        r_delta = float(r_raw - r0)
        info: dict[str, Any] = dict(info_raw or {})
        info["reward_raw"] = float(r_raw)
        info["reward_baseline_full22"] = float(r0)
        info["reward"] = r_delta
        info["normalize_mode"] = "delta_full22"

        out = (r_delta, info)
        self._cache[cache_key] = out
        return out


class AdvantageMaxBaselineEvaluator(EvaluatorBase):
    """
    Reward normalization wrapper for multi-subject / multi-K training.

    For each (subject, split, K), subtract the stronger baseline between:
      - full22 reward
      - lr_weight top-K reward, where K = popcount(key)

    Formally:
      r_adv(key) = r_base(key) - max(r_base(full22), r_base(lr_weight_topK(K))).

    Notes:
    - Baselines are computed using train-session-only statistics already stored in fold.stats.
    - For a fixed K, this is a constant shift within (subject, split), so it preserves ordering
      among K-sized subsets while aligning reward scales against a strong deterministic baseline.
    """

    def __init__(self, base: EvaluatorBase, *, baseline_full22_key: int = (1 << 22) - 1):
        self.base = base
        self.baseline_full22_key = int(baseline_full22_key)
        self._cache: dict[tuple[int, int, int], tuple[float, dict[str, Any]]] = {}
        self._full22_cache: dict[tuple[int, int], float] = {}
        self._lr_topk_cache: dict[tuple[int, int, int], tuple[int, float]] = {}
        self._baseline_max_cache: dict[tuple[int, int, int], tuple[float, dict[str, Any]]] = {}

    @staticmethod
    def _lr_weight_topk_key(fold: FoldData, k: int) -> int:
        k = int(k)
        if k <= 0:
            return 0
        if k >= 22:
            return (1 << 22) - 1
        scores = np.asarray(fold.stats.lr_weight, dtype=np.float32)
        if scores.shape != (22,):
            raise ValueError(f"Expected fold.stats.lr_weight shape (22,), got {scores.shape}")
        order = np.argsort(-scores, kind="stable")
        sel = order[:k].astype(np.int64, copy=False)
        mask = np.zeros((22,), dtype=np.int8)
        mask[sel] = 1
        return mask_to_key(mask)

    def _full22_reward(self, fold: FoldData) -> float:
        k = (int(fold.subject), int(fold.split_id))
        if k in self._full22_cache:
            return float(self._full22_cache[k])
        r0, _ = self.base.evaluate(self.baseline_full22_key, fold)
        self._full22_cache[k] = float(r0)
        return float(r0)

    def _lr_topk_reward(self, fold: FoldData, k: int) -> tuple[int, float]:
        ck = (int(fold.subject), int(fold.split_id), int(k))
        hit = self._lr_topk_cache.get(ck)
        if hit is not None:
            return int(hit[0]), float(hit[1])
        key = self._lr_weight_topk_key(fold, int(k))
        r, _ = self.base.evaluate(int(key), fold)
        out = (int(key), float(r))
        self._lr_topk_cache[ck] = out
        return out

    def _baseline_max(self, fold: FoldData, k: int) -> tuple[float, dict[str, Any]]:
        ck = (int(fold.subject), int(fold.split_id), int(k))
        hit = self._baseline_max_cache.get(ck)
        if hit is not None:
            return float(hit[0]), dict(hit[1])

        r_full22 = self._full22_reward(fold)
        lr_key, r_lr = self._lr_topk_reward(fold, int(k))
        r_max = float(max(r_full22, r_lr))
        info = {
            "reward_baseline_full22": float(r_full22),
            "baseline_full22_key": int(self.baseline_full22_key),
            "reward_baseline_lr_weight_topk": float(r_lr),
            "baseline_lr_weight_topk_key": int(lr_key),
            "reward_baseline_max": float(r_max),
        }
        out = (r_max, info)
        self._baseline_max_cache[ck] = out
        return out

    def evaluate(self, key: int, fold: FoldData) -> tuple[float, dict[str, Any]]:
        cache_key = (int(fold.subject), int(fold.split_id), int(key))
        if cache_key in self._cache:
            return self._cache[cache_key]

        r_raw, info_raw = self.base.evaluate(int(key), fold)
        k = popcount(int(key))
        r0, base_info = self._baseline_max(fold, int(k))
        r_adv = float(r_raw - r0)

        info: dict[str, Any] = dict(info_raw or {})
        info["reward_raw"] = float(r_raw)
        info.update(base_info)
        info["reward"] = float(r_adv)
        info["normalize_mode"] = "adv_lrmax"
        info["k"] = int(k)

        out = (r_adv, info)
        self._cache[cache_key] = out
        return out
