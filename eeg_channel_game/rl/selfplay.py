from __future__ import annotations

from typing import Any

import numpy as np

from eeg_channel_game.game.env import EEGChannelGame
from eeg_channel_game.mcts.mcts import MCTS
from eeg_channel_game.rl.replay_buffer import ReplayBuffer
from eeg_channel_game.utils.bitmask import popcount


def _sample_from_pi(pi: np.ndarray, tau: float, rng: np.random.Generator) -> int:
    pi = pi.astype(np.float64, copy=False)
    if tau <= 1e-6:
        return int(np.argmax(pi))
    x = np.power(pi, 1.0 / float(tau))
    x = np.maximum(x, 0.0)
    s = float(x.sum())
    if not np.isfinite(s) or s <= 0.0:
        return int(np.argmax(pi))
    x = x / s
    # numpy requires probabilities sum to 1 (tight tolerance)
    x[-1] = 1.0 - float(x[:-1].sum())
    if x[-1] < 0.0:
        x = np.maximum(x, 0.0)
        x = x / float(x.sum())
    return int(rng.choice(len(pi), p=x))


def play_one_game(
    *,
    env: EEGChannelGame,
    mcts: MCTS,
    buffer: ReplayBuffer,
    rng: np.random.Generator,
    temp_warmup_steps: int = 3,
    tau: float = 1.0,
    final_tau: float = 0.1,
) -> dict[str, Any]:
    traj: list[tuple[int, int, int, int, int, np.ndarray]] = []
    _ = env.reset()
    done = False
    info: dict[str, Any] = {}

    while not done:
        n_sel = popcount(env.key)
        pi = mcts.run(
            root_key=env.key,
            fold=env.fold,
            add_root_noise=True,
            b_max=int(env.b_max),
            min_selected_for_stop=int(env.min_selected_for_stop),
            rng=rng,
        )  # [23]
        cur_tau = float(tau) if n_sel < int(temp_warmup_steps) else float(final_tau)
        a = _sample_from_pi(pi, cur_tau, rng)
        traj.append(
            (
                int(env.key),
                int(env.fold.subject),
                int(env.fold.split_id),
                int(env.b_max),
                int(env.min_selected_for_stop),
                pi,
            )
        )
        _, r, done, info = env.step(a)

    z = float(r)
    for key, subject, split_id, b_max, min_selected_for_stop, pi in traj:
        buffer.add(
            key=key,
            subject=subject,
            split_id=split_id,
            b_max=b_max,
            min_selected_for_stop=min_selected_for_stop,
            pi=pi,
            z=z,
        )
    return info
