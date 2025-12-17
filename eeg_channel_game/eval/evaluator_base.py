from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from eeg_channel_game.eeg.fold_sampler import FoldData


class EvaluatorBase(ABC):
    @abstractmethod
    def evaluate(self, key: int, fold: FoldData) -> tuple[float, dict[str, Any]]:
        raise NotImplementedError

