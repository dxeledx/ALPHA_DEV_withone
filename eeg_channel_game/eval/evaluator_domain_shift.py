from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eeg_channel_game.eeg.fold_sampler import FoldData
from eeg_channel_game.eval.evaluator_base import EvaluatorBase
from eeg_channel_game.utils.bitmask import popcount


class DomainShiftPenaltyEvaluator(EvaluatorBase):
    """
    Wrap an evaluator with an unlabeled cross-session domain-shift penalty.

    This is SAFE for training because it only reads eval-session *features* (e.g., bandpower),
    never eval labels.

    Default penalty:
      D(S) = mean_{i in S} || mu_T(i) - mu_E(i) ||_2
    where mu_T is the train-session mean bandpower computed on the fold's train split,
    and mu_E is the eval-session mean bandpower (all eval trials, label-free).
    """

    def __init__(
        self,
        base: EvaluatorBase,
        *,
        eta: float = 0.0,
        mode: str = "bp_mean_l2",  # currently only bp_mean_l2
        data_root: str | Path = Path("eeg_channel_game") / "data",
        variant: str | None = None,
    ):
        self.base = base
        self.eta = float(eta)
        self.mode = str(mode)
        self.data_root = Path(data_root)
        self.variant = variant

        self._eval_bp_mean: dict[int, np.ndarray] = {}  # subject -> [22,B]
        self._cache: dict[tuple[int, int, int], tuple[float, dict[str, Any]]] = {}

    def _cache_dir(self, subject: int) -> Path:
        if self.variant is None:
            legacy = self.data_root / "cache" / f"subj{int(subject):02d}"
            if legacy.exists():
                return legacy
            v = "default"
        else:
            v = str(self.variant)
        return self.data_root / "cache" / v / f"subj{int(subject):02d}"

    def _get_eval_bp_mean(self, subject: int) -> np.ndarray:
        subject = int(subject)
        x = self._eval_bp_mean.get(subject)
        if x is not None:
            return x
        path = self._cache_dir(subject) / "sessionE_bp.npz"
        if not path.exists():
            # No eval features available; fall back to zeros (no penalty).
            mean = np.zeros((22, 1), dtype=np.float32)
        else:
            bp = np.load(path)["bp"].astype(np.float32, copy=False)  # [N,22,B]
            mean = bp.mean(axis=0).astype(np.float32, copy=False)
        self._eval_bp_mean[subject] = mean
        return mean

    def evaluate(self, key: int, fold: FoldData) -> tuple[float, dict[str, Any]]:
        cache_key = (fold.subject, fold.split_id, int(key))
        if cache_key in self._cache:
            return self._cache[cache_key]

        r, info = self.base.evaluate(int(key), fold)
        info2: dict[str, Any] = dict(info or {})

        eta = float(self.eta)
        if eta <= 0.0:
            out = (float(r), info2)
            self._cache[cache_key] = out
            return out

        n_ch = popcount(key)
        if n_ch == 0:
            info2["domain_shift"] = 0.0
            info2["domain_shift_eta"] = float(eta)
            info2["domain_shift_mode"] = str(self.mode)
            out = (float(r), info2)
            self._cache[cache_key] = out
            return out

        sel = [i for i in range(22) if (int(key) >> i) & 1]

        if self.mode != "bp_mean_l2":
            raise ValueError(f"Unknown domain-shift mode: {self.mode}")

        mu_t = fold.stats.bp_mean.astype(np.float32, copy=False)  # [22,B]
        mu_e = self._get_eval_bp_mean(int(fold.subject))
        if mu_e.shape[1] != mu_t.shape[1]:
            # Variant mismatch; do not crash training.
            shift = 0.0
        else:
            diff = mu_t - mu_e  # [22,B]
            per_ch = np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32, copy=False)  # [22]
            shift = float(np.mean(per_ch[np.asarray(sel, dtype=np.int64)]))

        r2 = float(r - eta * float(shift))
        info2["domain_shift"] = float(shift)
        info2["domain_shift_eta"] = float(eta)
        info2["domain_shift_mode"] = str(self.mode)
        info2["reward"] = float(r2)

        out = (r2, info2)
        self._cache[cache_key] = out
        return out

