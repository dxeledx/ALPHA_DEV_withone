from __future__ import annotations

from typing import Any

from eeg_channel_game.eeg.fold_sampler import FoldData
from eeg_channel_game.eval.evaluator_base import EvaluatorBase
from eeg_channel_game.game.state_builder import StateBuilder, StateObs
from eeg_channel_game.utils.bitmask import popcount


class EEGChannelGame:
    def __init__(
        self,
        *,
        fold: FoldData,
        state_builder: StateBuilder,
        evaluator: EvaluatorBase,
        b_max: int = 10,
        min_selected_for_stop: int | None = None,
    ):
        self.fold = fold
        self.state_builder = state_builder
        self.evaluator = evaluator
        self.b_max = int(b_max)
        self.min_selected_for_stop = (
            int(state_builder.min_selected_for_stop) if min_selected_for_stop is None else int(min_selected_for_stop)
        )
        self.key = 0

    def reset(self) -> StateObs:
        self.key = 0
        return self.state_builder.build(
            self.key,
            self.fold,
            b_max=self.b_max,
            min_selected_for_stop=self.min_selected_for_stop,
        )

    def step(self, action: int) -> tuple[StateObs, float, bool, dict[str, Any]]:
        action = int(action)
        info: dict[str, Any] = {}
        n_sel = popcount(self.key)

        if n_sel >= self.b_max:
            done = True
        elif action == 22:
            # STOP is only valid after selecting a minimum number of channels.
            done = n_sel >= self.min_selected_for_stop
            if not done:
                info["invalid_stop"] = True
        else:
            if (self.key >> action) & 1:
                # invalid action; should be masked out by MCTS/policy
                done = False
            else:
                self.key = self.key | (1 << action)
                done = popcount(self.key) >= self.b_max

        if done:
            reward, detail = self.evaluator.evaluate(self.key, self.fold)
            info.update(detail)
        else:
            reward = 0.0

        obs = self.state_builder.build(
            self.key,
            self.fold,
            b_max=self.b_max,
            min_selected_for_stop=self.min_selected_for_stop,
        )
        return obs, float(reward), bool(done), info
