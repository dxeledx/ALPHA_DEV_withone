from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eeg_channel_game.eeg.fold_sampler import FoldData
from eeg_channel_game.utils.bitmask import popcount


@dataclass(frozen=True)
class StateObs:
    tokens: np.ndarray  # float32 [T=24, D]
    action_mask: np.ndarray  # bool [A=23]
    key: int


class StateBuilder:
    def __init__(
        self,
        *,
        ch_names: list[str],
        d_in: int = 64,
        b_max: int = 10,
        min_selected_for_stop: int = 2,
    ):
        self.ch_names = list(ch_names)
        self.d_in = int(d_in)
        self.b_max = int(b_max)
        self.min_selected_for_stop = int(min_selected_for_stop)
        self._ch_pos = self._make_channel_pos(self.ch_names)  # [22, 3]

    @staticmethod
    def _make_channel_pos(ch_names: list[str]) -> np.ndarray:
        try:
            import mne
        except Exception:
            return np.zeros((len(ch_names), 3), dtype=np.float32)

        montage = mne.channels.make_standard_montage("standard_1005")
        ch_pos = montage.get_positions()["ch_pos"]
        pos = np.zeros((len(ch_names), 3), dtype=np.float32)
        for i, name in enumerate(ch_names):
            if name in ch_pos:
                pos[i] = np.asarray(ch_pos[name], dtype=np.float32)
        # normalize (per-dimension) for stability
        mean = pos.mean(axis=0, keepdims=True)
        std = pos.std(axis=0, keepdims=True) + 1e-8
        pos = (pos - mean) / std
        return pos.astype(np.float32)

    def build(
        self,
        key: int,
        fold: FoldData,
        *,
        b_max: int | None = None,
        min_selected_for_stop: int | None = None,
    ) -> StateObs:
        key = int(key)
        b_max = int(self.b_max) if b_max is None else int(b_max)
        min_selected_for_stop = (
            int(self.min_selected_for_stop) if min_selected_for_stop is None else int(min_selected_for_stop)
        )
        n_sel = popcount(key)
        sel = np.array([(key >> i) & 1 for i in range(22)], dtype=np.int8)
        stats = fold.stats
        bands = list(getattr(fold.subject_data, "bands", []))  # [(fmin,fmax), ...]

        # Per-channel features
        feats: list[np.ndarray] = []
        sel_idx = np.where(sel == 1)[0]
        for i in range(22):
            onehot = np.zeros((22,), dtype=np.float32)
            onehot[i] = 1.0
            is_sel = np.array([float(sel[i])], dtype=np.float32)

            if sel_idx.size <= 1:
                redund_mean = 0.0
                redund_max = 0.0
            else:
                others = sel_idx[sel_idx != i]
                if others.size == 0:
                    redund_mean = 0.0
                    redund_max = 0.0
                else:
                    r = stats.redund_corr[i, others]
                    redund_mean = float(r.mean())
                    redund_max = float(r.max())

            ch_feat = np.concatenate(
                [
                    onehot,
                    is_sel,
                    stats.bp_mean[i].astype(np.float32, copy=False),
                    stats.bp_std[i].astype(np.float32, copy=False),
                    stats.fisher[i].astype(np.float32, copy=False),
                    np.array([redund_mean, redund_max], dtype=np.float32),
                    stats.quality_mean[i].astype(np.float32, copy=False),
                    np.array([float(stats.artifact_corr_eog[i])], dtype=np.float32),
                    np.array([float(stats.resid_ratio[i])], dtype=np.float32),
                    self._ch_pos[i].astype(np.float32, copy=False),
                ],
                axis=0,
            )
            feats.append(ch_feat)

        ch_tokens = np.stack(feats, axis=0)  # [22, D_raw]

        # Global tokens (CLS/CTX)
        cls = np.zeros((ch_tokens.shape[1],), dtype=np.float32)
        ctx = np.zeros((ch_tokens.shape[1],), dtype=np.float32)
        cls[0] = n_sel / 22.0
        cls[1] = (b_max - n_sel) / max(1, b_max)
        cls[2] = float(stats.fisher[sel_idx].mean()) if sel_idx.size else 0.0
        cls[3] = float(stats.artifact_corr_eog[sel_idx].mean()) if sel_idx.size else 0.0
        ctx[:4] = cls[:4]

        tokens_raw = np.concatenate([cls[None, :], ch_tokens, ctx[None, :]], axis=0)  # [24, D_raw]

        if tokens_raw.shape[1] > self.d_in:
            raise ValueError(f"State feature dim {tokens_raw.shape[1]} exceeds d_in={self.d_in}")
        if tokens_raw.shape[1] < self.d_in:
            pad = np.zeros((tokens_raw.shape[0], self.d_in - tokens_raw.shape[1]), dtype=np.float32)
            tokens = np.concatenate([tokens_raw, pad], axis=1)
        else:
            tokens = tokens_raw

        # Subject-conditioned context: replicate a small per-(subject,fold) descriptor into the padded dims.
        # This lets a single shared policy/value network adapt its channel-selection strategy across subjects.
        ctx_dim = int(self.d_in - tokens_raw.shape[1])
        if ctx_dim > 0:
            ctx_feat = np.zeros((ctx_dim,), dtype=np.float32)

            fisher = stats.fisher.astype(np.float32, copy=False)
            fisher_mean = float(np.mean(fisher))
            fisher_all = float(np.mean(fisher, axis=1).mean())  # == fisher_mean, but explicit

            # [0] overall separability (log-scaled)
            ctx_feat[0] = float(np.log1p(max(0.0, fisher_mean)))

            # [1] left-vs-right motor lateralization (tanh-scaled)
            if ctx_dim > 1:
                left_names = {"FC3", "C5", "C3", "C1", "CP3", "CP1"}
                right_names = {"FC4", "C6", "C4", "C2", "CP4", "CP2"}
                left_idx = [i for i, n in enumerate(self.ch_names) if n in left_names]
                right_idx = [i for i, n in enumerate(self.ch_names) if n in right_names]
                if left_idx and right_idx:
                    f_left = float(np.mean(fisher[left_idx]))
                    f_right = float(np.mean(fisher[right_idx]))
                    denom = abs(fisher_all) + 1e-6
                    ctx_feat[1] = float(np.tanh((f_left - f_right) / denom))
                else:
                    ctx_feat[1] = 0.0

            # [2] mu-vs-beta bandpower balance (tanh on log-ratio)
            if ctx_dim > 2 and stats.bp_mean.ndim == 2 and len(bands) == stats.bp_mean.shape[1]:
                centers = np.array([(float(a) + float(b)) / 2.0 for a, b in bands], dtype=np.float32)
                mu_idx = int(np.argmin(np.abs(centers - 10.0)))
                beta_idx = np.where((centers >= 18.0) & (centers <= 30.0))[0]
                if beta_idx.size == 0:
                    beta_idx = np.where((centers >= 16.0) & (centers <= 32.0))[0]
                mu = float(stats.bp_mean[:, mu_idx].mean())
                beta = float(stats.bp_mean[:, beta_idx].mean()) if beta_idx.size else float(stats.bp_mean.mean())
                ctx_feat[2] = float(np.tanh(mu - beta))

            # [3] artifact level (mean abs corr with EOG), scaled to [-1,1]
            if ctx_dim > 3:
                art = float(np.mean(stats.artifact_corr_eog))
                ctx_feat[3] = float(np.clip(2.0 * art - 1.0, -1.0, 1.0))

            # [4] residual ratio after EOG regression, scaled to [-1,1]
            if ctx_dim > 4:
                rr = float(np.mean(stats.resid_ratio))
                ctx_feat[4] = float(np.clip(2.0 * rr - 1.0, -1.0, 1.0))

            # Extra slots (only used if d_in is increased):
            # [5] fisher std (log-scaled)
            if ctx_dim > 5:
                ctx_feat[5] = float(np.log1p(max(0.0, float(np.std(fisher, ddof=0)))))

            # [6] mean log-variance (tanh-scaled)
            if ctx_dim > 6 and stats.quality_mean.ndim == 2 and stats.quality_mean.shape[1] >= 1:
                q_logvar = float(np.mean(stats.quality_mean[:, 0]))
                ctx_feat[6] = float(np.tanh(q_logvar / 2.0))

            # [7] mean kurtosis (excess over 3), tanh-scaled
            if ctx_dim > 7 and stats.quality_mean.ndim == 2 and stats.quality_mean.shape[1] >= 2:
                q_kurt = float(np.mean(stats.quality_mean[:, 1]))
                ctx_feat[7] = float(np.tanh((q_kurt - 3.0) / 2.0))

            # [8] global redundancy proxy (mean abs corr), scaled to [-1,1]
            if ctx_dim > 8 and stats.redund_corr.ndim == 2:
                # redund_corr diagonal is 0; this is a cheap overall redundancy score.
                r_mean = float(np.mean(stats.redund_corr))
                ctx_feat[8] = float(np.clip(2.0 * r_mean - 1.0, -1.0, 1.0))

            # Remaining dims stay 0 for now.
            tokens[:, -ctx_dim:] = ctx_feat[None, :]

        action_mask = np.ones((23,), dtype=bool)
        if n_sel >= b_max:
            action_mask[:22] = False
            action_mask[22] = True
        else:
            action_mask[:22] = (sel == 0)
            action_mask[22] = n_sel >= min_selected_for_stop

        return StateObs(tokens=tokens.astype(np.float32, copy=False), action_mask=action_mask, key=key)
