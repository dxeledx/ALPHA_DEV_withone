from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eeg_channel_game.eeg.deep_preprocess import DeepPreprocessConfig, apply_deep_preprocess_splits
from eeg_channel_game.eeg.fold_sampler import FoldData
from eeg_channel_game.eval.evaluator_base import EvaluatorBase
from eeg_channel_game.eval.metrics import accuracy, cohen_kappa
from eeg_channel_game.utils.bitmask import popcount


def _model_forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    braindecode models sometimes return [B, C] or [B, C, T'] (dense predictions).
    We unify to [B, C] logits by averaging over the last dim if needed.
    """
    out = model(x)
    if out.ndim == 3:
        out = out.mean(dim=-1)
    if out.ndim != 2:
        raise ValueError(f"Unexpected model output shape: {tuple(out.shape)}")
    return out


def _dense_ce_loss(loss_fn: nn.Module, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    If logits are dense [B, C, P], apply CroppedLoss-like behavior by repeating labels across P.
    Otherwise, standard CE on [B, C].
    """
    if logits.ndim == 3:
        b, n_classes, n_preds = logits.shape
        logits2 = logits.permute(0, 2, 1).reshape(int(b) * int(n_preds), int(n_classes))
        y2 = y.repeat_interleave(int(n_preds))
        return loss_fn(logits2, y2)
    return loss_fn(logits, y)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _sample_random_masks(
    *,
    rng: np.random.Generator,
    batch_size: int,
    n_ch: int,
    k_min: int,
    k_max: int,
    p_full: float,
) -> np.ndarray:
    n_ch = int(n_ch)
    k_min = int(k_min)
    k_max = int(k_max)
    if k_min < 2:
        k_min = 2
    if k_max > n_ch:
        k_max = n_ch
    if k_min > k_max:
        k_min = k_max

    out = np.zeros((int(batch_size), n_ch), dtype=np.float32)
    for i in range(int(batch_size)):
        if float(p_full) > 0.0 and float(rng.random()) < float(p_full):
            out[i, :] = 1.0
            continue
        k = int(rng.integers(k_min, k_max + 1))
        idx = rng.choice(n_ch, size=k, replace=False)
        out[i, idx] = 1.0
    return out


@dataclass(frozen=True)
class L1DeepMaskedTrainConfig:
    """
    Deep proxy reward on 0train only (no 1test label access):
      - Train ShallowFBCSPNet on fold.train_idx with random channel-dropout
      - Early-stop on fold.val_idx (full-22)
      - Reward = kappa (robust across seeds) on fold.val_idx with a fixed subset mask
    """

    preproc: DeepPreprocessConfig = DeepPreprocessConfig()

    # subset augmentation
    k_min: int = 4
    k_max: int = 14
    p_full: float = 0.2

    # cropped training (dense predictions)
    pool_mode: str = "max"
    final_conv_length: int = 30

    # optimization
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 30


@dataclass
class _TrainedSeed:
    seed: int
    model: nn.Module
    train_time_s: float


@dataclass
class _TrainerCacheEntry:
    cfg: L1DeepMaskedTrainConfig
    x_val: np.ndarray
    y_val: np.ndarray
    seeds: dict[int, _TrainedSeed]


class L1DeepMaskedEvaluator(EvaluatorBase):
    """
    L1.5 deep proxy evaluator (SAFE for training reward):
      - Only uses 0train labels (FoldData.subject_data.y_train)
      - Trains once per (subject, split) and caches
      - Evaluates any subset by masking channels on the fold's val set

    This is intended to align Phase B reward with the eventual deep L2 narrative,
    without touching 1test labels.
    """

    def __init__(
        self,
        *,
        lambda_cost: float = 0.05,
        artifact_gamma: float = 0.0,
        robust_mode: str = "mean_std",  # mean | q20 | mean_std
        robust_beta: float = 0.5,
        seeds: Iterable[int] = (0,),
        device: str = "cuda",
        cfg: L1DeepMaskedTrainConfig = L1DeepMaskedTrainConfig(),
    ):
        self.lambda_cost = float(lambda_cost)
        self.artifact_gamma = float(artifact_gamma)
        self.robust_mode = str(robust_mode)
        self.robust_beta = float(robust_beta)
        self.seeds = tuple(int(s) for s in seeds)
        self.device = str(device)
        self.cfg = cfg

        self._trainer_cache: dict[tuple[Any, ...], _TrainerCacheEntry] = {}
        self._cache: dict[tuple[int, int, int], tuple[float, dict[str, Any]]] = {}

    def _robust(self, xs: np.ndarray) -> float:
        if xs.size == 0:
            return 0.0
        if self.robust_mode == "mean":
            return float(xs.mean())
        if self.robust_mode == "q20":
            return float(np.quantile(xs, 0.2))
        if self.robust_mode == "mean_std":
            return float(xs.mean() - self.robust_beta * xs.std(ddof=0))
        raise ValueError(f"Unknown robust_mode={self.robust_mode}")

    def _get_or_train(self, fold: FoldData) -> _TrainerCacheEntry:
        train_idx_i64 = np.asarray(fold.split.train_idx, dtype=np.int64)
        val_idx_i64 = np.asarray(fold.split.val_idx, dtype=np.int64)
        cache_key = (
            "l1_deep_masked",
            int(fold.subject),
            int(fold.split_id),
            train_idx_i64.tobytes(),
            val_idx_i64.tobytes(),
            self.seeds,
            self.device,
            self.cfg,
        )
        entry = self._trainer_cache.get(cache_key)
        if entry is not None:
            return entry

        try:
            from braindecode.models import ShallowFBCSPNet
        except Exception as e:  # pragma: no cover
            raise RuntimeError("braindecode is required for L1DeepMaskedEvaluator") from e

        device_t = torch.device(str(self.device))
        if device_t.type == "cuda" and not torch.cuda.is_available():
            device_t = torch.device("cpu")

        sd = fold.subject_data
        x_tr = sd.X_train[train_idx_i64].astype(np.float32, copy=False)  # [Ntr, C, T]
        x_va = sd.X_train[val_idx_i64].astype(np.float32, copy=False)  # [Nva, C, T]
        # reuse x_va slot for API convenience
        x_tr, x_va, _ = apply_deep_preprocess_splits(x_tr, x_va, x_va, cfg=self.cfg.preproc)
        y_tr = sd.y_train[train_idx_i64].astype(np.int64, copy=False)
        y_va = sd.y_train[val_idx_i64].astype(np.int64, copy=False)

        ds_tr = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
        ds_va = TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va))
        dl_tr = DataLoader(ds_tr, batch_size=int(self.cfg.batch_size), shuffle=True, drop_last=False)
        dl_va = DataLoader(ds_va, batch_size=int(self.cfg.batch_size), shuffle=False, drop_last=False)

        trained: dict[int, _TrainedSeed] = {}
        for seed in self.seeds:
            _set_seed(int(seed))
            t0 = time.time()

            model = ShallowFBCSPNet(
                n_chans=int(sd.X_train.shape[1]),
                n_outputs=4,
                n_times=int(x_tr.shape[-1]),
                pool_mode=str(self.cfg.pool_mode),
                final_conv_length=int(self.cfg.final_conv_length),
                add_log_softmax=False,
            ).to(device_t)
            if hasattr(model, "to_dense_prediction_model"):
                try:
                    model.to_dense_prediction_model()
                except Exception:
                    pass

            opt = torch.optim.AdamW(
                model.parameters(), lr=float(self.cfg.lr), weight_decay=float(self.cfg.weight_decay)
            )
            loss_fn = nn.CrossEntropyLoss()

            best_state: dict[str, torch.Tensor] | None = None
            best_val = float("inf")
            bad = 0

            rng = np.random.default_rng(int(seed))
            for _ep in range(int(self.cfg.epochs)):
                model.train()
                for xb, yb in dl_tr:
                    xb = xb.to(device_t)  # [B, C, T]
                    yb = yb.to(device_t)
                    masks = _sample_random_masks(
                        rng=rng,
                        batch_size=int(yb.shape[0]),
                        n_ch=int(sd.X_train.shape[1]),
                        k_min=int(self.cfg.k_min),
                        k_max=int(self.cfg.k_max),
                        p_full=float(self.cfg.p_full),
                    )
                    gate = torch.from_numpy(masks).to(device_t)[:, :, None].to(dtype=xb.dtype)
                    logits_raw = model(xb * gate)
                    loss = _dense_ce_loss(loss_fn, logits_raw, yb)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                # early-stopping: full-22 val loss
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for xb, yb in dl_va:
                        xb = xb.to(device_t)
                        yb = yb.to(device_t)
                        logits_raw = model(xb)
                        loss = _dense_ce_loss(loss_fn, logits_raw, yb)
                        val_losses.append(float(loss.detach().cpu().item()))
                val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

                if val_loss + 1e-6 < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= int(self.cfg.patience):
                        break

            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            dt = float(time.time() - t0)
            trained[int(seed)] = _TrainedSeed(seed=int(seed), model=model, train_time_s=dt)

        entry = _TrainerCacheEntry(cfg=self.cfg, x_val=x_va, y_val=y_va, seeds=trained)
        self._trainer_cache[cache_key] = entry
        return entry

    def evaluate(self, key: int, fold: FoldData) -> tuple[float, dict[str, Any]]:
        cache_key = (fold.subject, fold.split_id, int(key))
        if cache_key in self._cache:
            return self._cache[cache_key]

        n_ch = popcount(key)
        if n_ch < 2:
            out = (-1.0, {"n_ch": n_ch, "kappa_robust": 0.0, "acc_mean": 0.0})
            self._cache[cache_key] = out
            return out

        sel = [i for i in range(22) if (int(key) >> i) & 1]
        entry = self._get_or_train(fold)

        device_t = torch.device(str(self.device))
        if device_t.type == "cuda" and not torch.cuda.is_available():
            device_t = torch.device("cpu")

        # eval on fold.val with fixed subset mask
        mask_vec = np.zeros((22,), dtype=np.float32)
        mask_vec[np.asarray(sel, dtype=np.int64)] = 1.0

        ds = TensorDataset(torch.from_numpy(entry.x_val), torch.from_numpy(entry.y_val))
        dl = DataLoader(ds, batch_size=int(entry.cfg.batch_size), shuffle=False, drop_last=False)

        per_seed = []
        kappas = []
        accs = []
        times = []
        with torch.no_grad():
            for seed, ts in entry.seeds.items():
                model = ts.model
                preds = []
                targs = []
                for xb, yb in dl:
                    xb = xb.to(device_t)
                    gate = torch.from_numpy(mask_vec).to(device_t)[None, :, None].to(dtype=xb.dtype)
                    gate = gate.repeat(int(yb.shape[0]), 1, 1)
                    logits = _model_forward_logits(model, xb * gate)
                    yhat = torch.argmax(logits, dim=1).detach().cpu().numpy()
                    preds.append(yhat)
                    targs.append(yb.numpy())

                y_pred = np.concatenate(preds) if preds else np.array([], dtype=np.int64)
                y_true = np.concatenate(targs) if targs else np.array([], dtype=np.int64)
                kappa = float(cohen_kappa(y_true, y_pred))
                acc = float(accuracy(y_true, y_pred))
                kappas.append(kappa)
                accs.append(acc)
                times.append(float(ts.train_time_s))
                per_seed.append({"seed": int(seed), "kappa": kappa, "acc": acc, "train_time_s": float(ts.train_time_s)})

        kappas_np = np.array(kappas, dtype=np.float32)
        accs_np = np.array(accs, dtype=np.float32)
        kappa_robust = float(self._robust(kappas_np))

        artifact = float(fold.stats.artifact_corr_eog[np.asarray(sel, dtype=np.int64)].mean())
        cost = float(n_ch) / 22.0
        reward = float(kappa_robust - self.lambda_cost * cost - self.artifact_gamma * artifact)

        info: dict[str, Any] = {
            "n_ch": int(n_ch),
            "kappa_robust": float(kappa_robust),
            "kappa_mean": float(kappas_np.mean()) if kappas_np.size else 0.0,
            "kappa_std": float(kappas_np.std(ddof=0)) if kappas_np.size else 0.0,
            "kappa_q20": float(np.quantile(kappas_np, 0.2)) if kappas_np.size else 0.0,
            "acc_mean": float(accs_np.mean()) if accs_np.size else 0.0,
            "acc_std": float(accs_np.std(ddof=0)) if accs_np.size else 0.0,
            "robust_mode": str(self.robust_mode),
            "robust_beta": float(self.robust_beta),
            "artifact": float(artifact),
            "cost": float(cost),
            "reward": float(reward),
            "per_seed": per_seed,
            "train_cfg": {
                "k_min": int(entry.cfg.k_min),
                "k_max": int(entry.cfg.k_max),
                "p_full": float(entry.cfg.p_full),
                "pool_mode": str(entry.cfg.pool_mode),
                "final_conv_length": int(entry.cfg.final_conv_length),
                "epochs": int(entry.cfg.epochs),
                "batch_size": int(entry.cfg.batch_size),
                "lr": float(entry.cfg.lr),
                "weight_decay": float(entry.cfg.weight_decay),
                "patience": int(entry.cfg.patience),
                "preproc": {
                    "use_car": bool(entry.cfg.preproc.use_car),
                    "scale": float(entry.cfg.preproc.scale),
                    "standardize": str(entry.cfg.preproc.standardize),
                    "em_factor_new": float(entry.cfg.preproc.em_factor_new),
                    "em_init_block_size": int(entry.cfg.preproc.em_init_block_size),
                    "em_eps": float(entry.cfg.preproc.em_eps),
                },
            },
        }
        out = (reward, info)
        self._cache[cache_key] = out
        return out

