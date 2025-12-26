from __future__ import annotations

import math

import numpy as np
import torch

from eeg_channel_game.eeg.fold_sampler import FoldData
from eeg_channel_game.eval.evaluator_base import EvaluatorBase
from eeg_channel_game.game.state_builder import StateBuilder
from eeg_channel_game.mcts.node import Node
from eeg_channel_game.mcts.transposition import TranspositionTable
from eeg_channel_game.model.policy_value_net import PolicyValueNet
from eeg_channel_game.utils.bitmask import apply_action, popcount


class MCTS:
    def __init__(
        self,
        *,
        net: PolicyValueNet,
        state_builder: StateBuilder,
        evaluator: EvaluatorBase,
        leaf_evaluator: EvaluatorBase | None = None,
        leaf_value_mix_alpha: float = 1.0,
        leaf_value_proxy_scale: float = 1.0,
        policy_prior_eta: float = 0.0,
        policy_prior_temperature: float = 1.0,
        n_sim: int = 256,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        device: str = "cpu",
    ):
        self.net = net
        self.state_builder = state_builder
        self.evaluator = evaluator
        self.leaf_evaluator = leaf_evaluator
        self.leaf_value_mix_alpha = float(leaf_value_mix_alpha)
        self.leaf_value_proxy_scale = float(leaf_value_proxy_scale)
        self.policy_prior_eta = float(policy_prior_eta)
        self.policy_prior_temperature = float(policy_prior_temperature)
        self.n_sim = int(n_sim)
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_eps = float(dirichlet_eps)
        self.device = torch.device(device)
        self.tt = TranspositionTable()
        self.n_actions = 23

    def reset(self) -> None:
        self.tt.clear()

    def run(
        self,
        *,
        root_key: int,
        fold: FoldData,
        add_root_noise: bool = True,
        b_max: int | None = None,
        min_selected_for_stop: int | None = None,
    ) -> np.ndarray:
        root_key = int(root_key)
        b_max = int(self.state_builder.b_max) if b_max is None else int(b_max)
        min_selected_for_stop = (
            int(self.state_builder.min_selected_for_stop)
            if min_selected_for_stop is None
            else int(min_selected_for_stop)
        )

        root = self._get_or_expand(
            root_key,
            fold,
            add_root_noise=add_root_noise,
            b_max=b_max,
            min_selected_for_stop=min_selected_for_stop,
        )

        for _ in range(self.n_sim):
            self._simulate(root_key, fold, b_max=b_max, min_selected_for_stop=min_selected_for_stop)

        pi = root.N.astype(np.float32)
        s = float(pi.sum())
        if s <= 0.0:
            pi = np.zeros_like(pi)
            pi[22] = 1.0
            return pi
        pi /= s
        return pi

    def _infer(
        self, key: int, fold: FoldData, *, b_max: int, min_selected_for_stop: int
    ) -> tuple[np.ndarray, float]:
        obs = self.state_builder.build(key, fold, b_max=b_max, min_selected_for_stop=min_selected_for_stop)
        tokens = torch.from_numpy(obs.tokens[None]).to(self.device)
        action_mask = torch.from_numpy(obs.action_mask[None]).to(self.device)
        self.net.eval()
        with torch.no_grad():
            logits, value = self.net(tokens, action_mask=action_mask)
            p = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy().astype(np.float32)
            v = float(value[0].detach().cpu().item())

        # Optional: mix in an embedded-model prior (lr_weight) to stabilize early search.
        eta = float(self.policy_prior_eta)
        if eta > 0.0:
            scores = np.log1p(np.maximum(fold.stats.lr_weight.astype(np.float32, copy=False), 0.0))  # [22]
            sel = np.array([(int(key) >> i) & 1 for i in range(22)], dtype=bool)
            s = scores.copy()
            s[sel] = -1e9
            temp = float(self.policy_prior_temperature)
            if temp <= 1e-6:
                temp = 1.0
            ex = np.exp((s - float(np.max(s))) / temp).astype(np.float32, copy=False)
            ex[sel] = 0.0
            p_prior = np.zeros((self.n_actions,), dtype=np.float32)
            p_prior[:22] = ex
            # STOP gets no heuristic mass.
            ps = float(p_prior.sum())
            if ps > 0.0:
                p_prior /= ps
                eta2 = float(min(1.0, max(0.0, eta)))
                p = (1.0 - eta2) * p + eta2 * p_prior

        # enforce mask and renormalize
        p = p * obs.action_mask.astype(np.float32)
        ps = float(p.sum())
        if ps > 0:
            p /= ps
        return p, v

    def _valid_actions(self, key: int, *, b_max: int, min_selected_for_stop: int) -> np.ndarray:
        key = int(key)
        n_sel = popcount(key)
        mask = np.ones((self.n_actions,), dtype=bool)
        if n_sel >= b_max:
            mask[:22] = False
            mask[22] = True
            return mask
        for a in range(22):
            mask[a] = ((key >> a) & 1) == 0
        mask[22] = n_sel >= min_selected_for_stop
        return mask

    def _get_or_expand(
        self,
        key: int,
        fold: FoldData,
        *,
        add_root_noise: bool = False,
        b_max: int,
        min_selected_for_stop: int,
    ) -> Node:
        node = self.tt.get(key)
        if node is not None and node.is_expanded:
            return node

        p, v = self._infer(key, fold, b_max=b_max, min_selected_for_stop=min_selected_for_stop)
        if add_root_noise:
            valid = self._valid_actions(key, b_max=b_max, min_selected_for_stop=min_selected_for_stop)
            if valid.any():
                noise = np.random.dirichlet([self.dirichlet_alpha] * int(valid.sum())).astype(np.float32)
                p2 = p.copy()
                p2[valid] = (1.0 - self.dirichlet_eps) * p[valid] + self.dirichlet_eps * noise
                p2[~valid] = 0.0
                s = float(p2.sum())
                if s > 0:
                    p = p2 / s

        new = Node.empty(self.n_actions)
        new.P = p.astype(np.float32, copy=False)
        new.is_expanded = True
        self.tt.put(key, new)
        _ = v  # value handled at expansion time
        return new

    def _select_action(self, node: Node, valid: np.ndarray) -> int:
        n_sum = float(node.N.sum())
        u = self.c_puct * node.P * math.sqrt(n_sum + 1e-8) / (1.0 + node.N.astype(np.float32))
        score = node.Q + u
        score = score.astype(np.float64, copy=False)
        score[~valid] = -1e18
        return int(np.argmax(score))

    def _mixed_leaf_value(self, *, key: int, fold: FoldData, v_net: float) -> float:
        alpha = float(self.leaf_value_mix_alpha)
        if self.leaf_evaluator is None:
            return float(v_net)
        if alpha >= 1.0:
            return float(v_net)
        if alpha <= 0.0:
            v_proxy, _ = self.leaf_evaluator.evaluate(int(key), fold)
            return float(self.leaf_value_proxy_scale) * float(v_proxy)

        v_proxy, _ = self.leaf_evaluator.evaluate(int(key), fold)
        v_proxy = float(self.leaf_value_proxy_scale) * float(v_proxy)
        return float(alpha * float(v_net) + (1.0 - alpha) * float(v_proxy))

    def _simulate(self, root_key: int, fold: FoldData, *, b_max: int, min_selected_for_stop: int) -> None:
        key = int(root_key)
        path: list[tuple[int, int]] = []

        # Selection
        last_action = None
        while True:
            node = self.tt.get(key)
            if node is None or not node.is_expanded:
                break
            valid = self._valid_actions(key, b_max=b_max, min_selected_for_stop=min_selected_for_stop)
            a = self._select_action(node, valid)
            path.append((key, a))
            last_action = a
            if a == 22:
                key = key
                break
            key = apply_action(key, a)
            if popcount(key) >= b_max:
                break

        # Expansion / Evaluation
        is_stop = last_action == 22
        is_full = popcount(key) >= b_max
        is_terminal = is_stop or is_full

        if is_terminal:
            v, _ = self.evaluator.evaluate(key, fold)
        else:
            # Expand leaf and bootstrap with network value.
            p, v_net = self._infer(key, fold, b_max=b_max, min_selected_for_stop=min_selected_for_stop)
            v = self._mixed_leaf_value(key=key, fold=fold, v_net=float(v_net))
            leaf = Node.empty(self.n_actions)
            leaf.P = p
            leaf.is_expanded = True
            self.tt.put(key, leaf)

        # Backup
        for k, a in reversed(path):
            node = self.tt.get(k)
            if node is None:
                continue
            node.N[a] += 1
            node.W[a] += float(v)
            node.Q[a] = node.W[a] / float(node.N[a])
