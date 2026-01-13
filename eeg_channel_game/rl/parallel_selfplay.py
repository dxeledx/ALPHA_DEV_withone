from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from eeg_channel_game.eeg.fold_sampler import FoldSampler, FoldData
from eeg_channel_game.eeg.variant import variant_from_cfg
from eeg_channel_game.eval.evaluator_base import EvaluatorBase
from eeg_channel_game.eval.evaluator_domain_shift import DomainShiftPenaltyEvaluator
from eeg_channel_game.eval.evaluator_l0 import L0Evaluator
from eeg_channel_game.eval.evaluator_l0_lr_weight import L0LrWeightEvaluator
from eeg_channel_game.eval.evaluator_l1_deep_masked import L1DeepMaskedEvaluator, L1DeepMaskedTrainConfig
from eeg_channel_game.eval.evaluator_l1_fbcsp import L1FBCSPEvaluator
from eeg_channel_game.eval.evaluator_normalize import DeltaFull22Evaluator
from eeg_channel_game.eval.evaluator_normalize import AdvantageMaxBaselineEvaluator
from eeg_channel_game.game.env import EEGChannelGame
from eeg_channel_game.game.state_builder import StateBuilder
from eeg_channel_game.mcts.mcts import MCTS
from eeg_channel_game.model.policy_value_net import PolicyValueNet
from eeg_channel_game.utils.bitmask import popcount

_W: dict[str, Any] = {}


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
    x[-1] = 1.0 - float(x[:-1].sum())
    if x[-1] < 0.0:
        x = np.maximum(x, 0.0)
        x = x / float(x.sum())
    return int(rng.choice(len(pi), p=x))


def _make_evaluator(cfg: dict[str, Any], *, variant: str, name: str, device: str) -> EvaluatorBase:
    name = str(name)
    if name == "l0":
        return L0Evaluator(
            lambda_cost=float(cfg["reward"]["lambda_cost"]),
            beta_redund=float(cfg["reward"]["beta_redund"]),
            artifact_gamma=float(cfg["reward"].get("artifact_gamma", 0.0)),
        )
    if name in {"l0_lr_weight", "l0_lr", "lr_weight"}:
        return L0LrWeightEvaluator(
            lambda_cost=float(cfg["reward"]["lambda_cost"]),
            beta_redund=float(cfg["reward"]["beta_redund"]),
            artifact_gamma=float(cfg["reward"].get("artifact_gamma", 0.0)),
        )
    if name == "l1_fbcsp":
        l1_cfg = cfg.get("evaluator", {}).get("l1_fbcsp", {})
        return L1FBCSPEvaluator(
            lambda_cost=float(cfg["reward"]["lambda_cost"]),
            artifact_gamma=float(cfg["reward"].get("artifact_gamma", 0.0)),
            mode=str(l1_cfg.get("mode", "mr_fbcsp")),
            mask_eps=float(l1_cfg.get("mask_eps", 0.0)),
            csp_shrinkage=float(l1_cfg.get("csp_shrinkage", 0.1)),
            csp_ridge=float(l1_cfg.get("csp_ridge", 1e-3)),
            mask_penalty=float(l1_cfg.get("mask_penalty", 0.1)),
            cv_folds=int(l1_cfg.get("cv_folds", 3)),
            robust_mode=str(l1_cfg.get("robust_mode", "mean_std")),
            robust_beta=float(l1_cfg.get("robust_beta", 0.5)),
            variant=variant,
        )
    if name == "l1_deep_masked":
        l1_cfg = cfg.get("evaluator", {}).get("l1_deep_masked", {})
        train_cfg = L1DeepMaskedTrainConfig(
            k_min=int(l1_cfg.get("k_min", 4)),
            k_max=int(l1_cfg.get("k_max", 14)),
            p_full=float(l1_cfg.get("p_full", 0.2)),
            pool_mode=str(l1_cfg.get("pool_mode", "max")),
            final_conv_length=int(l1_cfg.get("final_conv_length", 30)),
            epochs=int(l1_cfg.get("epochs", 200)),
            batch_size=int(l1_cfg.get("batch_size", 64)),
            lr=float(l1_cfg.get("lr", 1e-3)),
            weight_decay=float(l1_cfg.get("weight_decay", 1e-4)),
            patience=int(l1_cfg.get("patience", 30)),
        )
        seeds = tuple(int(s) for s in str(l1_cfg.get("seeds", "0")).split(",") if s.strip())
        return L1DeepMaskedEvaluator(
            lambda_cost=float(cfg["reward"]["lambda_cost"]),
            artifact_gamma=float(cfg["reward"].get("artifact_gamma", 0.0)),
            robust_mode=str(l1_cfg.get("robust_mode", "mean_std")),
            robust_beta=float(l1_cfg.get("robust_beta", 0.5)),
            seeds=seeds or (0,),
            device=str(l1_cfg.get("device", device)),
            cfg=train_cfg,
        )
    raise ValueError(f"Unknown evaluator name: {name}")


def _wrap_evaluator(cfg: dict[str, Any], *, variant: str, ev: EvaluatorBase, data_root: Path) -> EvaluatorBase:
    # Optional: unlabeled cross-session domain-shift penalty (SAFE; uses eval features only).
    ds_cfg = cfg.get("reward", {}).get("domain_shift", {}) or {}
    ds_enabled = bool(ds_cfg.get("enabled", False))
    ds_eta = float(ds_cfg.get("eta", 0.0))
    ds_mode = str(ds_cfg.get("mode", "bp_mean_l2"))
    if ds_enabled and ds_eta > 0.0:
        ev = DomainShiftPenaltyEvaluator(ev, eta=ds_eta, mode=ds_mode, data_root=data_root, variant=variant)

    # Optional: per-subject reward normalization (delta vs full-22 baseline).
    normalize = cfg.get("reward", {}).get("normalize", False)
    norm_mode = str(normalize).lower()
    if norm_mode in {"1", "true", "yes", "delta_full22", "delta"}:
        ev = DeltaFull22Evaluator(ev)
    if norm_mode in {"adv_lrmax", "adv_full22_lrmax", "adv_full22_lr_weight", "adv_lr_weight_max"}:
        ev = AdvantageMaxBaselineEvaluator(ev)
    return ev


def init_worker(cfg: dict[str, Any], device: str, weights_path: str) -> None:
    """
    Multiprocessing worker initializer for parallel self-play.

    IMPORTANT:
    - Workers maintain their own evaluator caches (OK).
    - Workers lazily reload model weights from weights_path when task['iter'] changes.
    """
    global _W
    device = str(device)
    weights_path_p = Path(weights_path)

    variant = variant_from_cfg(cfg)
    subjects = [int(s) for s in cfg["data"]["subjects"]]
    seed = int(cfg["project"]["seed"])

    # Avoid CPU thread over-subscription when using many processes.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    sampler = FoldSampler(subjects=subjects, n_splits=5, seed=seed, variant=variant, include_eval=False)
    any_subject = subjects[0]
    ch_names = sampler.subject_data[any_subject].ch_names

    state_builder = StateBuilder(
        ch_names=ch_names,
        d_in=int(cfg["net"]["d_in"]),
        b_max=int(cfg["game"]["b_max"]),
        min_selected_for_stop=int(cfg["game"]["min_selected_for_stop"]),
    )

    # Evaluators per phase (wrapped consistently).
    train_cfg = cfg.get("train", {})
    switch_to_l1_iter = train_cfg.get("switch_to_l1_iter", None)
    if switch_to_l1_iter is None:
        evaluator_phase_a = cfg.get("evaluator", {}).get("name", "l0")
        evaluator_phase_b = evaluator_phase_a
        switch_to_l1_iter = int(train_cfg.get("num_iters", 0) + 1)
    else:
        evaluator_phase_a = cfg.get("evaluator", {}).get("phase_a", "l0")
        evaluator_phase_b = cfg.get("evaluator", {}).get("phase_b", "l1_fbcsp")
        switch_to_l1_iter = int(switch_to_l1_iter)

    data_root = Path("eeg_channel_game") / "data"
    evaluator_a = _wrap_evaluator(
        cfg,
        variant=variant,
        ev=_make_evaluator(cfg, variant=variant, name=str(evaluator_phase_a), device=device),
        data_root=data_root,
    )
    evaluator_b = _wrap_evaluator(
        cfg,
        variant=variant,
        ev=_make_evaluator(cfg, variant=variant, name=str(evaluator_phase_b), device=device),
        data_root=data_root,
    )

    # Leaf bootstrap proxy evaluator.
    leaf_cfg = cfg.get("mcts", {}).get("leaf_bootstrap", {}) or {}
    leaf_enabled = bool(leaf_cfg.get("enabled", False))
    leaf_proxy = str(leaf_cfg.get("proxy", "l0")).lower()
    leaf_evaluator: EvaluatorBase | None = None
    if leaf_enabled:
        if leaf_proxy in {"l0", "fisher"}:
            leaf_evaluator = _make_evaluator(cfg, variant=variant, name="l0", device=device)
        elif leaf_proxy in {"lr_weight", "l0_lr_weight", "l0_lr"}:
            leaf_evaluator = _make_evaluator(cfg, variant=variant, name="l0_lr_weight", device=device)
        else:
            raise ValueError(f"Unknown mcts.leaf_bootstrap.proxy={leaf_proxy!r} (expected: l0|lr_weight)")
        leaf_evaluator = _wrap_evaluator(cfg, variant=variant, ev=leaf_evaluator, data_root=data_root)

    net = PolicyValueNet(
        d_in=int(cfg["net"]["d_in"]),
        d_model=int(cfg["net"]["d_model"]),
        n_layers=int(cfg["net"]["n_layers"]),
        n_heads=int(cfg["net"]["n_heads"]),
        policy_mode=str(cfg.get("net", {}).get("policy_mode", "cls")),
        think_steps=int(cfg.get("net", {}).get("think_steps", 1) or 1),
        n_actions=23,
    ).to(torch.device(device))
    net.eval()

    infer_bs = int(cfg.get("mcts", {}).get("infer_batch_size", 1) or 1)

    mcts = MCTS(
        net=net,
        state_builder=state_builder,
        evaluator=evaluator_a,
        leaf_evaluator=leaf_evaluator,
        leaf_value_mix_alpha=1.0,
        leaf_value_proxy_scale=float(leaf_cfg.get("proxy_scale", 1.0)),
        policy_prior_eta=0.0,
        policy_prior_temperature=float(cfg.get("mcts", {}).get("policy_prior", {}).get("temperature", 1.0)),
        infer_batch_size=infer_bs,
        n_sim=int(cfg["mcts"]["n_sim"]),
        c_puct=float(cfg["mcts"]["c_puct"]),
        dirichlet_alpha=float(cfg["mcts"]["dirichlet_alpha"]),
        dirichlet_eps=float(cfg["mcts"]["dirichlet_eps"]),
        device=device,
    )

    # Cache training sampling knobs.
    b_max_choices = train_cfg.get("b_max_choices", None)
    if isinstance(b_max_choices, str):
        b_max_choices = [int(s.strip()) for s in b_max_choices.split(",") if s.strip()]
    elif isinstance(b_max_choices, (list, tuple)):
        b_max_choices = [int(x) for x in b_max_choices]
    else:
        b_max_choices = []
    force_exact_budget = bool(train_cfg.get("force_exact_budget", False))

    temp_cfg = cfg["mcts"]["temperature"]

    _W = {
        "cfg": cfg,
        "variant": variant,
        "subjects": subjects,
        "n_splits": int(sampler.n_splits),
        "sampler": sampler,
        "state_builder": state_builder,
        "evaluator_a": evaluator_a,
        "evaluator_b": evaluator_b,
        "evaluator_phase_a": str(evaluator_phase_a),
        "evaluator_phase_b": str(evaluator_phase_b),
        "switch_to_l1_iter": int(switch_to_l1_iter),
        "mcts": mcts,
        "net": net,
        "weights_path": weights_path_p,
        "loaded_iter": None,
        "b_max_choices": b_max_choices,
        "force_exact_budget": force_exact_budget,
        "temp_warmup_steps": int(temp_cfg["warmup_steps"]),
        "tau": float(temp_cfg["tau"]),
        "final_tau": float(temp_cfg["final_tau"]),
    }


def run_one_game(task: dict[str, Any]) -> dict[str, Any]:
    global _W
    it = int(task["iter"])
    seed = int(task["seed"])
    leaf_alpha = float(task.get("leaf_alpha", 1.0))
    policy_prior_eta = float(task.get("policy_prior_eta", 0.0))

    weights_path: Path = _W["weights_path"]
    loaded_iter = _W.get("loaded_iter", None)
    if loaded_iter != it:
        ckpt = torch.load(weights_path, map_location="cpu")
        sd = ckpt.get("model", ckpt)
        missing, unexpected = _W["net"].load_state_dict(sd, strict=False)
        if missing or unexpected:
            # Keep worker robust; training will surface if this is systematic.
            pass
        _W["loaded_iter"] = it

    rng = np.random.default_rng(seed)

    # Determine phase evaluator by iter.
    if it < int(_W["switch_to_l1_iter"]):
        evaluator: EvaluatorBase = _W["evaluator_a"]
        phase = str(_W["evaluator_phase_a"])
    else:
        evaluator = _W["evaluator_b"]
        phase = str(_W["evaluator_phase_b"])

    mcts: MCTS = _W["mcts"]
    mcts.evaluator = evaluator
    mcts.leaf_value_mix_alpha = float(leaf_alpha)
    mcts.policy_prior_eta = float(policy_prior_eta)

    # Fold sampling (deterministic per-task seed).
    subjects: list[int] = _W["subjects"]
    subject = int(rng.choice(subjects))
    split_id = int(rng.integers(0, int(_W["n_splits"])))
    fold: FoldData = _W["sampler"].get_fold(subject, split_id)

    # Per-episode budget.
    b_max_choices: list[int] = _W["b_max_choices"]
    state_builder: StateBuilder = _W["state_builder"]
    b_max_game = int(rng.choice(b_max_choices)) if b_max_choices else int(state_builder.b_max)
    min_stop = int(b_max_game) if bool(_W["force_exact_budget"]) else int(state_builder.min_selected_for_stop)

    env = EEGChannelGame(
        fold=fold,
        state_builder=state_builder,
        evaluator=evaluator,
        b_max=b_max_game,
        min_selected_for_stop=min_stop,
    )
    _ = env.reset()
    mcts.reset()

    traj: list[tuple[int, int, int, int, int, np.ndarray]] = []
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
        )
        cur_tau = float(_W["tau"]) if n_sel < int(_W["temp_warmup_steps"]) else float(_W["final_tau"])
        a = _sample_from_pi(pi, cur_tau, rng)
        traj.append((int(env.key), int(subject), int(split_id), int(env.b_max), int(env.min_selected_for_stop), pi))
        _, r, done, info = env.step(a)

    return {
        "iter": int(it),
        "phase": str(phase),
        "reward": float(r),
        "key": int(env.key),
        "info": info,
        "traj": traj,
    }
