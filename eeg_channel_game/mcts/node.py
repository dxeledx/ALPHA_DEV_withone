from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Node:
    P: np.ndarray  # float32 [A]
    N: np.ndarray  # int32 [A]
    W: np.ndarray  # float32 [A]
    Q: np.ndarray  # float32 [A]
    is_expanded: bool = False

    @staticmethod
    def empty(n_actions: int) -> "Node":
        return Node(
            P=np.zeros((n_actions,), dtype=np.float32),
            N=np.zeros((n_actions,), dtype=np.int32),
            W=np.zeros((n_actions,), dtype=np.float32),
            Q=np.zeros((n_actions,), dtype=np.float32),
            is_expanded=False,
        )

