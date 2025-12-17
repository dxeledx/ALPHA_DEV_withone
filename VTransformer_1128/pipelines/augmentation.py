# -*- coding: utf-8 -*-
"""EEG 数据增强模块 (LiteGNNT V5.1)."""

import numpy as np


class EEGAugmentation:
    """与旧版 LiteGNNT 相同的 EEG 数据增强器."""

    def __init__(self, aug_prob=0.7, noise_level=0.05, scale_range=(0.9, 1.1),
                 shift_range=(-50, 50), drop_prob=0.1):
        self.aug_prob = aug_prob
        self.noise_level = noise_level
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.drop_prob = drop_prob

    def _apply_prob(self):
        return np.random.rand() <= self.aug_prob

    def sign_flipping(self, X):
        if not self._apply_prob():
            return X
        return -X

    def random_noise(self, X, noise_level=None):
        if not self._apply_prob():
            return X
        if noise_level is None:
            noise_level = self.noise_level
        std = np.std(X, axis=-1, keepdims=True)
        noise = np.random.randn(*X.shape) * std * noise_level
        return X + noise

    def time_shift(self, X, max_shift=None):
        if not self._apply_prob():
            return X
        if max_shift is None:
            max_shift = int(max(abs(self.shift_range[0]), abs(self.shift_range[1])))
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift > 0:
            return np.pad(X, ((0, 0), (0, 0), (shift, 0)), mode="constant")[:, :, :-shift]
        if shift < 0:
            return np.pad(X, ((0, 0), (0, 0), (0, -shift)), mode="constant")[:, :, -shift:]
        return X

    def amplitude_scale(self, X, scale_range=None):
        if not self._apply_prob():
            return X
        if scale_range is None:
            scale_range = self.scale_range
        scale = np.random.uniform(*scale_range)
        return X * scale

    def channel_dropout(self, X, drop_prob=None):
        if not self._apply_prob():
            return X
        if drop_prob is None:
            drop_prob = self.drop_prob
        mask = np.random.rand(*X.shape[:2], 1) > drop_prob
        return X * mask + X.mean(axis=1, keepdims=True) * (1 - mask)

    def time_masking(self, X, max_mask_length=50, n_masks=1):
        if not self._apply_prob():
            return X
        X_aug = X.copy()
        B, _, T = X.shape
        for _ in range(n_masks):
            mask_length = np.random.randint(1, max_mask_length + 1)
            mask_start = np.random.randint(0, max(1, T - mask_length))
            X_aug[:, :, mask_start:mask_start + mask_length] = 0
        return X_aug

    def frequency_shift(self, X, fs=250, max_shift_hz=2):
        if not self._apply_prob():
            return X
        shift_hz = np.random.uniform(-max_shift_hz, max_shift_hz)
        t = np.arange(X.shape[-1]) / fs
        modulation = np.cos(2 * np.pi * shift_hz * t)
        return X * modulation

    def segment_shuffle(self, X, n_segments=5):
        if not self._apply_prob():
            return X
        B, C, T = X.shape
        seg_len = T // n_segments
        X_aug = np.zeros_like(X)
        for b in range(B):
            segments = [X[b, :, i*seg_len:(i+1)*seg_len] for i in range(n_segments)]
            np.random.shuffle(segments)
            X_aug[b] = np.concatenate(segments, axis=1)
        return X_aug

    def apply_augmentation(self, X, augmentations):
        X_aug = X.copy()
        for aug in augmentations:
            if aug in {"sign_flip", "sign_flipping"}:
                X_aug = self.sign_flipping(X_aug)
            elif aug == "noise":
                X_aug = self.random_noise(X_aug)
            elif aug == "shift":
                X_aug = self.time_shift(X_aug)
            elif aug == "scale":
                X_aug = self.amplitude_scale(X_aug)
            elif aug == "channel_dropout":
                X_aug = self.channel_dropout(X_aug)
            elif aug == "time_mask":
                X_aug = self.time_masking(X_aug)
            elif aug == "freq_shift":
                X_aug = self.frequency_shift(X_aug)
            elif aug == "segment_shuffle":
                X_aug = self.segment_shuffle(X_aug)
        return X_aug
