from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, *, capacity: int = 200_000, n_actions: int = 23, seed: int = 42):
        self.capacity = int(capacity)
        self.n_actions = int(n_actions)
        self.rng = np.random.default_rng(seed)

        self.keys = np.zeros((self.capacity,), dtype=np.int64)
        self.subjects = np.zeros((self.capacity,), dtype=np.int16)
        self.split_ids = np.zeros((self.capacity,), dtype=np.int16)
        self.b_max = np.zeros((self.capacity,), dtype=np.int16)
        self.min_selected_for_stop = np.zeros((self.capacity,), dtype=np.int16)
        self.pi = np.zeros((self.capacity, self.n_actions), dtype=np.float16)
        self.z = np.zeros((self.capacity,), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        *,
        key: int,
        subject: int,
        split_id: int,
        b_max: int,
        min_selected_for_stop: int,
        pi: np.ndarray,
        z: float,
    ) -> None:
        i = self.ptr
        self.keys[i] = int(key)
        self.subjects[i] = int(subject)
        self.split_ids[i] = int(split_id)
        self.b_max[i] = int(b_max)
        self.min_selected_for_stop[i] = int(min_selected_for_stop)
        self.pi[i] = pi.astype(np.float16, copy=False)
        self.z[i] = float(z)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if self.size == 0:
            raise RuntimeError("Cannot sample from empty buffer")
        batch_size = int(batch_size)
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            "key": self.keys[idx],
            "subject": self.subjects[idx],
            "split_id": self.split_ids[idx],
            "b_max": self.b_max[idx],
            "min_selected_for_stop": self.min_selected_for_stop[idx],
            "pi": self.pi[idx].astype(np.float32),
            "z": self.z[idx].astype(np.float32),
        }

    def as_arrays(self) -> dict[str, np.ndarray]:
        sl = slice(None, self.size) if self.size < self.capacity else slice(None)
        return {
            "key": self.keys[sl],
            "subject": self.subjects[sl],
            "split_id": self.split_ids[sl],
            "b_max": self.b_max[sl],
            "min_selected_for_stop": self.min_selected_for_stop[sl],
            "z": self.z[sl],
        }
