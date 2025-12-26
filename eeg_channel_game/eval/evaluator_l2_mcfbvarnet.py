from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eeg_channel_game.eeg.io import SubjectData
from eeg_channel_game.eval.metrics import accuracy, cohen_kappa
from eeg_channel_game.model.mcfbvarnet import MCFBVarNet, MCFBVarNetConfig


DEFAULT_FB_BANDS: list[tuple[float, float]] = [
    (8.0, 13.0),  # mu
    (13.0, 30.0),  # beta
    (8.0, 30.0),  # mu+beta
    (4.0, 40.0),  # wide
]


@dataclass(frozen=True)
class MCFBVarNetTrainConfig:
    fb_bands: tuple[tuple[float, float], ...] = tuple(DEFAULT_FB_BANDS)
    fb_iir_order: int = 4

    # subset augmentation (channel dropout)
    k_min: int = 4
    k_max: int = 14
    p_full: float = 0.2

    # optimization
    epochs: int = 300
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-2
    patience: int = 30


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _compute_filterbank(
    x: np.ndarray,
    *,
    sfreq: float,
    bands: Iterable[tuple[float, float]],
    iir_order: int,
) -> np.ndarray:
    """
    x: [N, C, T] float32
    returns: [N, B, C, T] float32
    """
    try:
        import mne
    except Exception as e:  # pragma: no cover
        raise RuntimeError("mne is required for MC-FBVarNet filter-bank") from e

    if x.ndim != 3:
        raise ValueError(f"x must be [N,C,T], got {tuple(x.shape)}")
    x = x.astype(np.float64, copy=False)

    bands = list(bands)
    out = np.empty((x.shape[0], len(bands), x.shape[1], x.shape[2]), dtype=np.float32)
    iir_params = dict(order=int(iir_order), ftype="butter")
    for bi, (l_freq, h_freq) in enumerate(bands):
        xf = mne.filter.filter_data(
            x,
            sfreq=float(sfreq),
            l_freq=float(l_freq),
            h_freq=float(h_freq),
            method="iir",
            iir_params=iir_params,
            verbose=False,
        ).astype(np.float32, copy=False)
        out[:, bi] = xf
    return out


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


@dataclass
class _TrainedSeed:
    seed: int
    model: nn.Module
    train_time_s: float


@dataclass
class _TrainerCacheEntry:
    cfg: MCFBVarNetTrainConfig
    model_cfg: MCFBVarNetConfig
    x_eval_fb: np.ndarray
    y_eval: np.ndarray
    seeds: dict[int, _TrainedSeed]


_TRAINER_CACHE: dict[tuple[Any, ...], _TrainerCacheEntry] = {}


def evaluate_l2_mcfbvarnet_train_eval(
    *,
    subject_data: SubjectData,
    sel_idx: list[int],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seeds: Iterable[int] = (0, 1, 2),
    device: str = "cuda",
    cfg: MCFBVarNetTrainConfig = MCFBVarNetTrainConfig(),
    model_cfg: MCFBVarNetConfig = MCFBVarNetConfig(),
) -> dict[str, Any]:
    """
    Train MC-FBVarNet on training session (subset-augmented), test on eval session with a fixed mask.

    Important: Training is independent of sel_idx; sel_idx is only used as the test-time mask.
    We cache the trained model(s) per (subject_data object, split, hyperparams) inside the current process.
    """
    sel_idx = list(map(int, sel_idx))
    if not sel_idx:
        raise ValueError("sel_idx must be non-empty for L2 evaluation")

    if subject_data.X_eval is None or subject_data.y_eval is None:
        raise ValueError("subject_data must include eval session for L2 evaluation (include_eval=True)")

    n_ch = int(subject_data.X_train.shape[1])
    if n_ch != int(model_cfg.n_chans):
        raise ValueError(f"Expected n_chans={model_cfg.n_chans}, got {n_ch}")

    # cache key: keep it in-process and memory-safe (do not hash full arrays)
    train_idx_i64 = np.asarray(train_idx, dtype=np.int64)
    val_idx_i64 = np.asarray(val_idx, dtype=np.int64)
    seeds_t = tuple(int(s) for s in seeds)
    cache_key = (
        "mcfbvarnet",
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
        model_cfg,
    )

    device_t = torch.device(str(device))
    if device_t.type == "cuda" and not torch.cuda.is_available():
        device_t = torch.device("cpu")

    entry = _TRAINER_CACHE.get(cache_key)
    if entry is None:
        # 1) filter-bank + standardize once per split
        x_train_fb = _compute_filterbank(
            subject_data.X_train,
            sfreq=float(subject_data.sfreq),
            bands=cfg.fb_bands,
            iir_order=int(cfg.fb_iir_order),
        )
        x_eval_fb = _compute_filterbank(
            subject_data.X_eval,
            sfreq=float(subject_data.sfreq),
            bands=cfg.fb_bands,
            iir_order=int(cfg.fb_iir_order),
        )

        x_tr_full = x_train_fb[train_idx_i64]
        x_va_full = x_train_fb[val_idx_i64]
        y_tr = subject_data.y_train[train_idx_i64].astype(np.int64, copy=False)
        y_va = subject_data.y_train[val_idx_i64].astype(np.int64, copy=False)

        mu = x_tr_full.mean(axis=(0, 3), keepdims=True)
        sd = x_tr_full.std(axis=(0, 3), keepdims=True) + 1e-6
        x_tr_full = ((x_tr_full - mu) / sd).astype(np.float32, copy=False)
        x_va_full = ((x_va_full - mu) / sd).astype(np.float32, copy=False)
        x_eval_fb = ((x_eval_fb - mu) / sd).astype(np.float32, copy=False)

        ds_tr = TensorDataset(torch.from_numpy(x_tr_full), torch.from_numpy(y_tr))
        ds_va = TensorDataset(torch.from_numpy(x_va_full), torch.from_numpy(y_va))
        dl_tr = DataLoader(ds_tr, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
        dl_va = DataLoader(ds_va, batch_size=int(cfg.batch_size), shuffle=False, drop_last=False)

        trained: dict[int, _TrainedSeed] = {}
        for seed in seeds_t:
            _set_seed(int(seed))
            t0 = time.time()
            model = MCFBVarNet(model_cfg).to(device_t)
            opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
            loss_fn = nn.CrossEntropyLoss()

            best_state: dict[str, torch.Tensor] | None = None
            best_val = float("inf")
            bad = 0

            rng = np.random.default_rng(int(seed))
            for _ep in range(int(cfg.epochs)):
                model.train()
                for xb, yb in dl_tr:
                    xb = xb.to(device_t)
                    yb = yb.to(device_t)
                    masks = _sample_random_masks(
                        rng=rng,
                        batch_size=int(yb.shape[0]),
                        n_ch=n_ch,
                        k_min=int(cfg.k_min),
                        k_max=int(cfg.k_max),
                        p_full=float(cfg.p_full),
                    )
                    mb = torch.from_numpy(masks).to(device_t)
                    logits = model(xb, mb)
                    loss = loss_fn(logits, yb)
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
                        mb = torch.ones((int(yb.shape[0]), n_ch), dtype=xb.dtype, device=device_t)
                        logits = model(xb, mb)
                        loss = loss_fn(logits, yb)
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

        entry = _TrainerCacheEntry(
            cfg=cfg,
            model_cfg=model_cfg,
            x_eval_fb=x_eval_fb.astype(np.float32, copy=False),
            y_eval=subject_data.y_eval.astype(np.int64, copy=False),
            seeds=trained,
        )
        _TRAINER_CACHE[cache_key] = entry

    # 2) test on eval session with a fixed mask
    mask_vec = np.zeros((n_ch,), dtype=np.float32)
    mask_vec[np.asarray(sel_idx, dtype=np.int64)] = 1.0

    ds_te = TensorDataset(torch.from_numpy(entry.x_eval_fb), torch.from_numpy(entry.y_eval))
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
                mb = torch.from_numpy(mask_vec).to(device_t)[None, :].repeat(int(yb.shape[0]), 1)
                logits = model(xb, mb)
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
        "model": "mcfbvarnet",
        "n_ch": int(len(sel_idx)),
        "n_ch_input": int(n_ch),
        "fb_bands": [list(map(float, b)) for b in entry.cfg.fb_bands],
        "seeds": list(map(int, seeds_t)),
        "kappa_mean": float(kappas_np.mean()),
        "kappa_std": float(kappas_np.std(ddof=0)),
        "kappa_q20": float(np.quantile(kappas_np, 0.2)),
        "acc_mean": float(accs_np.mean()),
        "acc_std": float(accs_np.std(ddof=0)),
        "train_time_s_mean": float(times_np.mean()),
        "train_time_s_sum": float(times_np.sum()),
        "per_seed": per_seed,
    }
