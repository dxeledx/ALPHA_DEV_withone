from __future__ import annotations

from typing import Any

from eeg_channel_game.eeg.fold_sampler import FoldData
from eeg_channel_game.eval.evaluator_base import EvaluatorBase


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

