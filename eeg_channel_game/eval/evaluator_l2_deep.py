from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eeg_channel_game.eeg.fold_sampler import FoldData
from eeg_channel_game.eeg.io import SubjectData
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


def _make_braindecode_model(
    *,
    name: str,
    n_chans: int,
    n_times: int,
    n_outputs: int,
) -> nn.Module:
    try:
        from braindecode.models import EEGNetv4, ShallowFBCSPNet
    except Exception as e:  # pragma: no cover
        raise RuntimeError("braindecode is required for L2 evaluator") from e

    name = name.lower()
    if name in {"eegnet", "eegnetv4"}:
        return EEGNetv4(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)
    if name in {"shallow", "shallowfbcsp", "shallowfbcspnet"}:
        return ShallowFBCSPNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, add_log_softmax=False)
    raise ValueError(f"Unknown model name: {name}")


def _make_tcformer_model(*, n_chans: int, n_outputs: int) -> nn.Module:
    """
    TCFormer integration (external folder).

    NOTE: TCFormerCustom upstream code uses absolute imports like `from utils...`,
    so we add `TCFormerCustom/` to sys.path and import from its internal modules.
    """
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    tc_root = repo_root / "TCFormerCustom"
    if not tc_root.exists():
        raise RuntimeError("TCFormerCustom/ not found at repo root; please keep the folder or adjust the path.")

    tc_root_str = str(tc_root)
    if tc_root_str not in sys.path:
        sys.path.insert(0, tc_root_str)

    try:
        from models.tcformer import TCFormerModule  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import TCFormer from TCFormerCustom. "
            "Make sure dependencies (einops/pytorch_lightning/torchmetrics) are installed."
        ) from e

    return TCFormerModule(
        n_channels=int(n_chans),
        n_classes=int(n_outputs),
        F1=32,
        temp_kernel_lengths=(20, 32, 64),
        d_group=16,
        D=2,
        pool_length_1=8,
        pool_length_2=7,
        dropout_conv=0.4,
        use_group_attn=True,
        q_heads=4,
        kv_heads=2,
        trans_depth=5,
        trans_dropout=0.4,
        tcn_depth=2,
        kernel_length_tcn=4,
        dropout_tcn=0.3,
    )


def _make_model(*, name: str, n_chans: int, n_times: int, n_outputs: int) -> nn.Module:
    name = str(name).lower()
    if name in {"tcformer"}:
        return _make_tcformer_model(n_chans=n_chans, n_outputs=n_outputs)
    return _make_braindecode_model(name=name, n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)


@dataclass(frozen=True)
class L2Result:
    kappa: float
    acc: float
    train_time_s: float


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _train_one_seed(
    *,
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int,
) -> L2Result:
    _set_seed(seed)
    t0 = time.time()

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.CrossEntropyLoss()

    ds_tr = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    ds_va = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    dl_tr = DataLoader(ds_tr, batch_size=int(batch_size), shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=int(batch_size), shuffle=False, drop_last=False)

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    bad = 0

    for _ep in range(int(epochs)):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = _model_forward_logits(model, xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = _model_forward_logits(model, xb)
                loss = loss_fn(logits, yb)
                val_losses.append(float(loss.detach().cpu().item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # test
    model.eval()
    ds_te = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    dl_te = DataLoader(ds_te, batch_size=int(batch_size), shuffle=False, drop_last=False)
    preds = []
    targs = []
    with torch.no_grad():
        for xb, yb in dl_te:
            xb = xb.to(device)
            logits = _model_forward_logits(model, xb)
            yhat = torch.argmax(logits, dim=1).detach().cpu().numpy()
            preds.append(yhat)
            targs.append(yb.numpy())
    y_pred = np.concatenate(preds) if preds else np.array([], dtype=np.int64)
    y_true = np.concatenate(targs) if targs else np.array([], dtype=np.int64)

    dt = float(time.time() - t0)
    return L2Result(kappa=cohen_kappa(y_true, y_pred), acc=accuracy(y_true, y_pred), train_time_s=dt)


def evaluate_l2_deep_train_eval(
    *,
    subject_data: SubjectData,
    sel_idx: list[int],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    model_name: str = "eegnetv4",
    seeds: Iterable[int] = (0, 1, 2),
    device: str = "cuda",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
) -> dict[str, Any]:
    sel_idx = list(map(int, sel_idx))
    if not sel_idx:
        raise ValueError("sel_idx must be non-empty for L2 evaluation")

    if subject_data.X_eval is None or subject_data.y_eval is None:
        raise ValueError("subject_data must include eval session for L2 evaluation (include_eval=True)")

    model_name = str(model_name)
    y_train_full = subject_data.y_train.astype(np.int64, copy=False)
    y_eval = subject_data.y_eval.astype(np.int64, copy=False)

    x_train_full = subject_data.X_train[:, sel_idx, :].astype(np.float32, copy=False)
    x_eval = subject_data.X_eval[:, sel_idx, :].astype(np.float32, copy=False)

    x_tr = x_train_full[train_idx]
    y_tr = y_train_full[train_idx]
    x_va = x_train_full[val_idx]
    y_va = y_train_full[val_idx]

    mu = x_tr.mean(axis=(0, 2), keepdims=True)
    sd = x_tr.std(axis=(0, 2), keepdims=True) + 1e-6
    x_tr = ((x_tr - mu) / sd).astype(np.float32)
    x_va = ((x_va - mu) / sd).astype(np.float32)
    x_eval_std = ((x_eval - mu) / sd).astype(np.float32)

    n_chans_model = int(len(sel_idx))

    # to torch: [N, C, T]
    device_t = torch.device(device)
    results: list[L2Result] = []
    seed_list = [int(s) for s in seeds]
    for s in seed_list:
        model = _make_model(name=model_name, n_chans=n_chans_model, n_times=x_tr.shape[-1], n_outputs=4)
        res = _train_one_seed(
            model=model,
            x_train=x_tr,
            y_train=y_tr,
            x_val=x_va,
            y_val=y_va,
            x_test=x_eval_std,
            y_test=y_eval,
            device=device_t,
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(lr),
            weight_decay=float(weight_decay),
            patience=int(patience),
            seed=int(s),
        )
        results.append(res)

    kappas = np.array([r.kappa for r in results], dtype=np.float32)
    accs = np.array([r.acc for r in results], dtype=np.float32)
    times = np.array([r.train_time_s for r in results], dtype=np.float32)

    return {
        "model": str(model_name),
        "n_ch": int(len(sel_idx)),
        "seeds": seed_list,
        "kappa_mean": float(kappas.mean()),
        "kappa_std": float(kappas.std(ddof=0)),
        "kappa_q20": float(np.quantile(kappas, 0.2)),
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std(ddof=0)),
        "train_time_s_mean": float(times.mean()),
        "train_time_s_sum": float(times.sum()),
        "per_seed": [
            {"seed": int(s), "kappa": float(r.kappa), "acc": float(r.acc), "train_time_s": float(r.train_time_s)}
            for s, r in zip(seed_list, results)
        ],
    }


class L2DeepEvaluator(EvaluatorBase):
    """
    L2 (high-fidelity) evaluator:
      train on training session (with a train/val split),
      test on evaluation session (1test).

    NOTE: Do NOT use this as the reward during RL training/search, otherwise it leaks eval-session labels.
    """

    def __init__(
        self,
        *,
        model_name: str = "eegnetv4",
        device: str = "cuda",
        seeds: Iterable[int] = (0, 1, 2),
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        lambda_cost: float = 0.05,
        artifact_gamma: float = 0.0,
    ):
        self.model_name = str(model_name)
        self.device = str(device)
        self.seeds = tuple(int(s) for s in seeds)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.lambda_cost = float(lambda_cost)
        self.artifact_gamma = float(artifact_gamma)
        self._cache: dict[tuple[int, int, int], tuple[float, dict[str, Any]]] = {}

    def evaluate(self, key: int, fold: FoldData) -> tuple[float, dict[str, Any]]:
        cache_key = (fold.subject, fold.split_id, int(key))
        if cache_key in self._cache:
            return self._cache[cache_key]

        n_ch = popcount(key)
        if n_ch < 2:
            out = (-1.0, {"n_ch": n_ch, "kappa_mean": 0.0, "acc_mean": 0.0})
            self._cache[cache_key] = out
            return out

        sel = [i for i in range(22) if (int(key) >> i) & 1]
        detail = evaluate_l2_deep_train_eval(
            subject_data=fold.subject_data,
            sel_idx=sel,
            train_idx=fold.split.train_idx,
            val_idx=fold.split.val_idx,
            model_name=self.model_name,
            seeds=self.seeds,
            device=self.device,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            patience=self.patience,
        )

        artifact = float(fold.stats.artifact_corr_eog[np.array(sel, dtype=np.int64)].mean())
        cost = float(n_ch) / 22.0
        reward = float(detail["kappa_q20"] - self.lambda_cost * cost - self.artifact_gamma * artifact)

        info: dict[str, Any] = {
            "reward": reward,
            "artifact": artifact,
            "cost": cost,
            **detail,
        }
        out = (reward, info)
        self._cache[cache_key] = out
        return out
