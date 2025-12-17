from __future__ import annotations

import numpy as np


def mask_to_key(mask: np.ndarray) -> int:
    mask = np.asarray(mask).astype(np.uint8, copy=False)
    key = 0
    for i, v in enumerate(mask.tolist()):
        if v:
            key |= 1 << i
    return int(key)


def key_to_mask(key: int, n_ch: int = 22) -> np.ndarray:
    key = int(key)
    out = np.zeros((n_ch,), dtype=np.int8)
    for i in range(n_ch):
        out[i] = 1 if (key >> i) & 1 else 0
    return out


def apply_action(key: int, action: int) -> int:
    """
    action: 0..21 channel, 22 STOP (no-op)
    """
    key = int(key)
    if action == 22:
        return key
    return key | (1 << int(action))


def popcount(key: int) -> int:
    key = int(key)
    # Python < 3.10 has no int.bit_count()
    if hasattr(int, "bit_count"):
        return key.bit_count()  # type: ignore[attr-defined]
    return bin(key).count("1")
