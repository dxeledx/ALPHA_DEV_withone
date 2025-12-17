from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eeg_channel_game.eeg.fold_sampler import FoldData
from eeg_channel_game.eval.fbcsp import fit_fbcsp_ovr_filters, transform_fbcsp_features
from eeg_channel_game.eval.evaluator_base import EvaluatorBase
from eeg_channel_game.eval.metrics import accuracy, cohen_kappa
from eeg_channel_game.utils.bitmask import popcount


class L1FBCSPEvaluator(EvaluatorBase):
    def __init__(
        self,
        *,
        lambda_cost: float = 0.05,
        artifact_gamma: float = 0.0,
        m: int = 2,
        eps: float = 1e-6,
        cv_folds: int = 1,
        robust_mode: str = "mean_std",
        robust_beta: float = 0.5,
        data_root: str | Path = Path("eeg_channel_game") / "data",
        variant: str | None = None,
    ):
        self.lambda_cost = float(lambda_cost)
        self.artifact_gamma = float(artifact_gamma)
        self.m = int(m)
        self.eps = float(eps)
        self.cv_folds = int(cv_folds)
        self.robust_mode = str(robust_mode)
        self.robust_beta = float(robust_beta)
        self.data_root = Path(data_root)
        self.variant = variant

        self._cov_cache: dict[int, np.ndarray] = {}
        self._cache: dict[tuple[int, int, int], tuple[float, dict[str, Any]]] = {}

    def _load_cov_fb(self, subject: int) -> np.ndarray:
        subject = int(subject)
        if subject in self._cov_cache:
            return self._cov_cache[subject]
        if self.variant is None:
            path = self.data_root / "cache" / f"subj{subject:02d}" / "sessionT_cov_fb.npz"
        else:
            path = self.data_root / "cache" / str(self.variant) / f"subj{subject:02d}" / "sessionT_cov_fb.npz"
        d = np.load(path)
        cov_fb = d["cov_fb"].astype(np.float32, copy=False)
        self._cov_cache[subject] = cov_fb
        return cov_fb

    def _robust_kappa(self, kappas: np.ndarray) -> float:
        if kappas.size == 0:
            return 0.0
        if self.robust_mode == "mean":
            return float(kappas.mean())
        if self.robust_mode == "q20":
            return float(np.quantile(kappas, 0.2))
        if self.robust_mode == "mean_std":
            return float(kappas.mean() - self.robust_beta * kappas.std(ddof=0))
        raise ValueError(f"Unknown robust_mode={self.robust_mode}")

    def evaluate(self, key: int, fold: FoldData) -> tuple[float, dict[str, Any]]:
        cache_key = (fold.subject, fold.split_id, int(key))
        if cache_key in self._cache:
            return self._cache[cache_key]

        n_ch = popcount(key)
        if n_ch < 2:
            out = (-1.0, {"n_ch": n_ch, "kappa": 0.0, "acc": 0.0})
            self._cache[cache_key] = out
            return out

        sel = np.array([i for i in range(22) if (int(key) >> i) & 1], dtype=np.int64)
        sd = fold.subject_data
        cov_fb = self._load_cov_fb(fold.subject)

        try:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import StratifiedKFold
        except Exception as e:  # pragma: no cover
            raise RuntimeError("scikit-learn is required for L1 evaluator") from e

        base_idx = fold.split.train_idx
        y_base = sd.y_train[base_idx]

        kappas = []
        accs = []
        if self.cv_folds <= 1:
            tr_idx = fold.split.train_idx
            va_idx = fold.split.val_idx
            filters = fit_fbcsp_ovr_filters(cov_fb, sd.y_train, tr_idx, sel, m=self.m, eps=self.eps)
            x_tr = transform_fbcsp_features(cov_fb, tr_idx, sel, filters, m=self.m, eps=self.eps)
            x_va = transform_fbcsp_features(cov_fb, va_idx, sel, filters, m=self.m, eps=self.eps)
            clf = make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            )
            clf.fit(x_tr, sd.y_train[tr_idx])
            y_pred = clf.predict(x_va)
            kappas.append(cohen_kappa(sd.y_train[va_idx], y_pred))
            accs.append(accuracy(sd.y_train[va_idx], y_pred))
        else:
            skf = StratifiedKFold(
                n_splits=int(self.cv_folds),
                shuffle=True,
                random_state=int(10_000 * fold.subject + fold.split_id),
            )
            for tr_rel, va_rel in skf.split(np.zeros_like(y_base), y_base):
                tr_idx = base_idx[tr_rel]
                va_idx = base_idx[va_rel]
                filters = fit_fbcsp_ovr_filters(cov_fb, sd.y_train, tr_idx, sel, m=self.m, eps=self.eps)
                x_tr = transform_fbcsp_features(cov_fb, tr_idx, sel, filters, m=self.m, eps=self.eps)
                x_va = transform_fbcsp_features(cov_fb, va_idx, sel, filters, m=self.m, eps=self.eps)
                clf = make_pipeline(
                    StandardScaler(with_mean=True, with_std=True),
                    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                )
                clf.fit(x_tr, sd.y_train[tr_idx])
                y_pred = clf.predict(x_va)
                kappas.append(cohen_kappa(sd.y_train[va_idx], y_pred))
                accs.append(accuracy(sd.y_train[va_idx], y_pred))

        kappas_arr = np.array(kappas, dtype=np.float32)
        accs_arr = np.array(accs, dtype=np.float32)
        kappa_robust = self._robust_kappa(kappas_arr)

        artifact = float(fold.stats.artifact_corr_eog[sel].mean())
        cost = float(n_ch) / 22.0
        reward = float(kappa_robust - self.lambda_cost * cost - self.artifact_gamma * artifact)

        info: dict[str, Any] = {
            "n_ch": n_ch,
            "kappa_robust": float(kappa_robust),
            "kappa_mean": float(kappas_arr.mean()) if kappas_arr.size else 0.0,
            "kappa_std": float(kappas_arr.std(ddof=0)) if kappas_arr.size else 0.0,
            "kappa_q20": float(np.quantile(kappas_arr, 0.2)) if kappas_arr.size else 0.0,
            "acc_mean": float(accs_arr.mean()) if accs_arr.size else 0.0,
            "acc_std": float(accs_arr.std(ddof=0)) if accs_arr.size else 0.0,
            "cv_folds": int(self.cv_folds),
            "robust_mode": self.robust_mode,
            "artifact": artifact,
            "cost": cost,
            "reward": reward,
        }
        out = (reward, info)
        self._cache[cache_key] = out
        return out
