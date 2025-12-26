from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eeg_channel_game.eeg.deep_preprocess import DeepPreprocessConfig, apply_deep_preprocess_splits
from eeg_channel_game.eeg.io import SubjectData
from eeg_channel_game.eval.metrics import accuracy, cohen_kappa


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


@dataclass(frozen=True)
class ShallowMaskedTrainConfig:
    preproc: DeepPreprocessConfig = DeepPreprocessConfig()

    # subset augmentation (channel dropout)
    k_min: int = 4
    k_max: int = 14
    p_full: float = 0.2

    # braindecode-style cropped training (dense predictions)
    pool_mode: str = "max"  # max is compatible with to_dense_prediction_model
    final_conv_length: int = 30

    # optimization
    epochs: int = 300
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 30


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


def _dense_ce_loss(loss_fn: nn.Module, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    If logits are dense [B, C, P], apply CroppedLoss-like behavior by repeating labels across P.
    Otherwise, standard CE on [B, C].
    """
    if logits.ndim == 3:
        # [B, C, P] -> [B*P, C]
        b, n_classes, n_preds = logits.shape
        logits2 = logits.permute(0, 2, 1).reshape(int(b) * int(n_preds), int(n_classes))
        y2 = y.repeat_interleave(int(n_preds))
        return loss_fn(logits2, y2)
    return loss_fn(logits, y)


@dataclass
class _TrainedSeed:
    seed: int
    model: nn.Module
    train_time_s: float


@dataclass
class _TrainerCacheEntry:
    cfg: ShallowMaskedTrainConfig
    x_eval: np.ndarray
    y_eval: np.ndarray
    seeds: dict[int, _TrainedSeed]


_TRAINER_CACHE: dict[tuple[Any, ...], _TrainerCacheEntry] = {}


def evaluate_l2_shallow_masked_train_eval(
    *,
    subject_data: SubjectData,
    sel_idx: list[int],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seeds: Iterable[int] = (0, 1, 2),
    device: str = "cuda",
    cfg: ShallowMaskedTrainConfig = ShallowMaskedTrainConfig(),
) -> dict[str, Any]:
    """
    Train a single ShallowConvNet (Braindecode ShallowFBCSPNet) on full-22 with random channel-dropout,
    then evaluate on eval session with a fixed subset mask (sel_idx).

    Important: Training is independent of sel_idx; sel_idx is only used as the test-time mask.
    """
    sel_idx = list(map(int, sel_idx))
    if not sel_idx:
        raise ValueError("sel_idx must be non-empty for L2 evaluation")
    if subject_data.X_eval is None or subject_data.y_eval is None:
        raise ValueError("subject_data must include eval session for L2 evaluation (include_eval=True)")

    n_ch = int(subject_data.X_train.shape[1])
    train_idx_i64 = np.asarray(train_idx, dtype=np.int64)
    val_idx_i64 = np.asarray(val_idx, dtype=np.int64)
    seeds_t = tuple(int(s) for s in seeds)

    cache_key = (
        "shallow_masked",
        int(subject_data.subject),
        tuple(int(x) for x in subject_data.X_train.shape),
        tuple(int(x) for x in subject_data.X_eval.shape) if subject_data.X_eval is not None else None,
        float(subject_data.sfreq),
        tuple(subject_data.ch_names),
        train_idx_i64.tobytes(),
        val_idx_i64.tobytes(),
        seeds_t,
        str(device),
        cfg,
    )

    device_t = torch.device(str(device))
    if device_t.type == "cuda" and not torch.cuda.is_available():
        device_t = torch.device("cpu")

    entry = _TRAINER_CACHE.get(cache_key)
    if entry is None:
        try:
            from braindecode.models import ShallowFBCSPNet
        except Exception as e:  # pragma: no cover
            raise RuntimeError("braindecode is required for shallow_masked L2 evaluator") from e

        x_tr_full = subject_data.X_train[train_idx_i64].astype(np.float32, copy=False)  # [Ntr, C, T]
        x_va_full = subject_data.X_train[val_idx_i64].astype(np.float32, copy=False)  # [Nva, C, T]
        x_eval = subject_data.X_eval.astype(np.float32, copy=False)  # [Ne, C, T]
        x_tr_full, x_va_full, x_eval = apply_deep_preprocess_splits(x_tr_full, x_va_full, x_eval, cfg=cfg.preproc)

        y_tr = subject_data.y_train[train_idx_i64].astype(np.int64, copy=False)
        y_va = subject_data.y_train[val_idx_i64].astype(np.int64, copy=False)
        y_eval = subject_data.y_eval.astype(np.int64, copy=False)

        ds_tr = TensorDataset(torch.from_numpy(x_tr_full), torch.from_numpy(y_tr))
        ds_va = TensorDataset(torch.from_numpy(x_va_full), torch.from_numpy(y_va))
        dl_tr = DataLoader(ds_tr, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
        dl_va = DataLoader(ds_va, batch_size=int(cfg.batch_size), shuffle=False, drop_last=False)

        trained: dict[int, _TrainedSeed] = {}
        for seed in seeds_t:
            _set_seed(int(seed))
            t0 = time.time()

            model = ShallowFBCSPNet(
                n_chans=int(n_ch),
                n_outputs=4,
                n_times=int(x_tr_full.shape[-1]),
                pool_mode=str(cfg.pool_mode),
                final_conv_length=int(cfg.final_conv_length),
                add_log_softmax=False,
            ).to(device_t)
            if hasattr(model, "to_dense_prediction_model"):
                # Braindecode-style cropped training (dense predictions)
                try:
                    model.to_dense_prediction_model()
                except Exception:
                    # fall back to trialwise if the model cannot be converted
                    pass
            opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
            loss_fn = nn.CrossEntropyLoss()

            best_state: dict[str, torch.Tensor] | None = None
            best_val = float("inf")
            bad = 0

            rng = np.random.default_rng(int(seed))
            for _ep in range(int(cfg.epochs)):
                model.train()
                for xb, yb in dl_tr:
                    xb = xb.to(device_t)  # [B, C, T]
                    yb = yb.to(device_t)
                    masks = _sample_random_masks(
                        rng=rng,
                        batch_size=int(yb.shape[0]),
                        n_ch=n_ch,
                        k_min=int(cfg.k_min),
                        k_max=int(cfg.k_max),
                        p_full=float(cfg.p_full),
                    )
                    gate = torch.from_numpy(masks).to(device_t)[:, :, None].to(dtype=xb.dtype)  # [B, C, 1]
                    logits_raw = model(xb * gate)
                    loss = _dense_ce_loss(loss_fn, logits_raw, yb)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                # val (use full-22 for a stable early-stopping signal)
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
                    if bad >= int(cfg.patience):
                        break

            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            dt = float(time.time() - t0)
            trained[int(seed)] = _TrainedSeed(seed=int(seed), model=model, train_time_s=dt)

        entry = _TrainerCacheEntry(cfg=cfg, x_eval=x_eval, y_eval=y_eval, seeds=trained)
        _TRAINER_CACHE[cache_key] = entry

    # eval with fixed subset mask
    mask_vec = np.zeros((n_ch,), dtype=np.float32)
    mask_vec[np.asarray(sel_idx, dtype=np.int64)] = 1.0

    ds_te = TensorDataset(torch.from_numpy(entry.x_eval), torch.from_numpy(entry.y_eval))
    dl_te = DataLoader(ds_te, batch_size=int(cfg.batch_size), shuffle=False, drop_last=False)

    per_seed = []
    kappas = []
    accs = []
    times = []
    with torch.no_grad():
        for seed in seeds_t:
            ts = entry.seeds[int(seed)]
            model = ts.model
            preds = []
            targs = []
            for xb, yb in dl_te:
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
    times_np = np.array(times, dtype=np.float32)

    return {
        "model": "shallow_masked",
        "n_ch": int(len(sel_idx)),
        "n_ch_input": int(n_ch),
        "seeds": list(map(int, seeds_t)),
        "kappa_mean": float(kappas_np.mean()),
        "kappa_std": float(kappas_np.std(ddof=0)),
        "kappa_q20": float(np.quantile(kappas_np, 0.2)),
        "acc_mean": float(accs_np.mean()),
        "acc_std": float(accs_np.std(ddof=0)),
        "train_time_s_mean": float(times_np.mean()),
        "train_time_s_sum": float(times_np.sum()),
        "per_seed": per_seed,
        "train_cfg": {
            "preproc": {
                "use_car": bool(entry.cfg.preproc.use_car),
                "scale": float(entry.cfg.preproc.scale),
                "standardize": str(entry.cfg.preproc.standardize),
                "em_factor_new": float(entry.cfg.preproc.em_factor_new),
                "em_init_block_size": int(entry.cfg.preproc.em_init_block_size),
                "em_eps": float(entry.cfg.preproc.em_eps),
            },
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
        },
    }
