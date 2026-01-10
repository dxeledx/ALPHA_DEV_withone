from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from eeg_channel_game.eeg.fold_sampler import FoldSampler
from eeg_channel_game.eeg.variant import variant_from_cfg
from eeg_channel_game.eval.evaluator_l0 import L0Evaluator
from eeg_channel_game.eval.evaluator_l0_lr_weight import L0LrWeightEvaluator
from eeg_channel_game.eval.evaluator_l1_fbcsp import L1FBCSPEvaluator
from eeg_channel_game.eval.evaluator_l1_deep_masked import L1DeepMaskedEvaluator, L1DeepMaskedTrainConfig
from eeg_channel_game.eval.evaluator_base import EvaluatorBase
from eeg_channel_game.eval.evaluator_domain_shift import DomainShiftPenaltyEvaluator
from eeg_channel_game.eval.evaluator_normalize import DeltaFull22Evaluator
from eeg_channel_game.eval.evaluator_normalize import AdvantageMaxBaselineEvaluator
from eeg_channel_game.game.env import EEGChannelGame
from eeg_channel_game.game.state_builder import StateBuilder
from eeg_channel_game.mcts.mcts import MCTS
from eeg_channel_game.model.policy_value_net import PolicyValueNet
from eeg_channel_game.rl.replay_buffer import ReplayBuffer
from eeg_channel_game.rl.selfplay import play_one_game
from eeg_channel_game.utils.config import RunPaths, make_run_paths
from eeg_channel_game.utils.seed import set_global_seed
from eeg_channel_game.utils.bitmask import key_to_mask, popcount


def _latest_checkpoint(ckpt_dir: Path) -> Path | None:
    ckpt_dir = Path(ckpt_dir)
    last = ckpt_dir / "last.pt"
    if last.exists():
        return last
    pts = sorted(ckpt_dir.glob("iter_*.pt"))
    if pts:
        return pts[-1]
    best = ckpt_dir / "best.pt"
    if best.exists():
        return best
    return None


def _move_optimizer_state_to_device(opt: torch.optim.Optimizer, device: torch.device) -> None:
    for st in opt.state.values():
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.to(device=device)


def _build_batch(
    *,
    sampler: FoldSampler,
    state_builder: StateBuilder,
    keys: np.ndarray,
    subjects: np.ndarray,
    split_ids: np.ndarray,
    b_max: np.ndarray,
    min_selected_for_stop: np.ndarray,
    teacher_temp: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    bsz = int(keys.shape[0])
    tokens = np.empty((bsz, 24, state_builder.d_in), dtype=np.float32)
    action_mask = np.empty((bsz, 23), dtype=bool)
    teacher_pi = None
    if teacher_temp is not None:
        teacher_pi = np.zeros((bsz, 23), dtype=np.float32)
    for i in range(bsz):
        fold = sampler.get_fold(int(subjects[i]), int(split_ids[i]))
        obs = state_builder.build(
            int(keys[i]),
            fold,
            b_max=int(b_max[i]),
            min_selected_for_stop=int(min_selected_for_stop[i]),
        )
        tokens[i] = obs.tokens
        action_mask[i] = obs.action_mask
        if teacher_pi is not None:
            # Teacher over actions from embedded lr_weight scores (valid channels only).
            scores = np.log1p(np.maximum(fold.stats.lr_weight.astype(np.float32, copy=False), 0.0))  # [22]
            sel = np.array([(int(keys[i]) >> j) & 1 for j in range(22)], dtype=bool)
            valid_ch = (~sel) & obs.action_mask[:22]
            if not np.any(valid_ch):
                # If no channels are valid, default to STOP.
                teacher_pi[i, 22] = 1.0
            else:
                s = scores.copy()
                s[~valid_ch] = -1e9
                t = float(teacher_temp)
                if t <= 1e-6:
                    t = 1.0
                x = np.exp((s - float(np.max(s[valid_ch]))) / t).astype(np.float32, copy=False)
                x[~valid_ch] = 0.0
                p = x / float(x.sum())
                teacher_pi[i, :22] = p
                teacher_pi[i, 22] = 0.0
    return tokens, action_mask, teacher_pi


def train(cfg: dict[str, Any]) -> RunPaths:
    seed = int(cfg["project"]["seed"])
    set_global_seed(seed)

    train_cfg = cfg["train"]
    out_dir = Path(cfg["project"]["out_dir"])

    # Safety: avoid accidentally overwriting an existing run directory.
    resume = bool(train_cfg.get("resume", False))
    overwrite = bool(cfg.get("project", {}).get("overwrite", False)) or bool(train_cfg.get("overwrite", False))
    if out_dir.exists():
        ckpt_dir = out_dir / "checkpoints"
        has_ckpts = ckpt_dir.exists() and any(ckpt_dir.glob("*.pt"))
        has_cfg = (out_dir / "config.json").exists()
        if (has_ckpts or has_cfg) and (not resume) and (not overwrite):
            raise FileExistsError(
                f"Output dir already has results: {out_dir}\n"
                "Choose a new project.out_dir, or set train.resume=true to continue, "
                "or set project.overwrite=true to overwrite."
            )
        if (has_ckpts or has_cfg) and overwrite and (not resume):
            print(f"[warn] overwrite enabled; existing files in {out_dir} may be replaced/merged.")

    paths = make_run_paths(out_dir)
    (paths.out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (paths.out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    def _to_float(x: Any) -> float | None:
        if isinstance(x, (int, float, np.floating)) and np.isfinite(x):
            return float(x)
        return None

    def _quantiles(xs: list[float], qs: tuple[float, ...] = (0.2, 0.5, 0.8)) -> dict[float, float]:
        if not xs:
            return {q: 0.0 for q in qs}
        arr = np.asarray(xs, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {q: 0.0 for q in qs}
        return {float(q): float(np.quantile(arr, q)) for q in qs}

    def _collect(info_list: list[dict[str, Any]], key: str) -> list[float]:
        out: list[float] = []
        for d in info_list:
            if not d:
                continue
            v = _to_float(d.get(key))
            if v is None:
                continue
            out.append(float(v))
        return out

    device = str(cfg["project"].get("device", "cuda"))
    subjects = [int(s) for s in cfg["data"]["subjects"]]
    variant = variant_from_cfg(cfg)

    sampler = FoldSampler(subjects=subjects, n_splits=5, seed=seed, variant=variant, include_eval=False)
    any_subject = subjects[0]
    ch_names = sampler.subject_data[any_subject].ch_names

    state_builder = StateBuilder(
        ch_names=ch_names,
        d_in=int(cfg["net"]["d_in"]),
        b_max=int(cfg["game"]["b_max"]),
        min_selected_for_stop=int(cfg["game"]["min_selected_for_stop"]),
    )

    def make_evaluator(name: str) -> EvaluatorBase:
        name = str(name)
        if name == "l0":
            ev: EvaluatorBase = L0Evaluator(
                lambda_cost=float(cfg["reward"]["lambda_cost"]),
                beta_redund=float(cfg["reward"]["beta_redund"]),
                artifact_gamma=float(cfg["reward"].get("artifact_gamma", 0.0)),
            )
            return ev
        if name in {"l0_lr_weight", "l0_lr", "lr_weight"}:
            ev = L0LrWeightEvaluator(
                lambda_cost=float(cfg["reward"]["lambda_cost"]),
                beta_redund=float(cfg["reward"]["beta_redund"]),
                artifact_gamma=float(cfg["reward"].get("artifact_gamma", 0.0)),
            )
            return ev
        if name == "l1_fbcsp":
            l1_cfg = cfg.get("evaluator", {}).get("l1_fbcsp", {})
            ev = L1FBCSPEvaluator(
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
            return ev
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
            ev = L1DeepMaskedEvaluator(
                lambda_cost=float(cfg["reward"]["lambda_cost"]),
                artifact_gamma=float(cfg["reward"].get("artifact_gamma", 0.0)),
                robust_mode=str(l1_cfg.get("robust_mode", "mean_std")),
                robust_beta=float(l1_cfg.get("robust_beta", 0.5)),
                seeds=tuple(int(s) for s in str(l1_cfg.get("seeds", "0")).split(",") if s.strip()),
                device=str(l1_cfg.get("device", device)),
                cfg=train_cfg,
            )
            return ev
        raise ValueError(f"Unknown evaluator name: {name}")

    switch_to_l1_iter = train_cfg.get("switch_to_l1_iter", None)
    clear_buffer_on_switch = bool(train_cfg.get("clear_buffer_on_switch", True))

    if switch_to_l1_iter is None:
        # backward-compatible: single evaluator
        evaluator_phase_a = cfg.get("evaluator", {}).get("name", "l0")
        evaluator_phase_b = evaluator_phase_a
        switch_to_l1_iter = int(train_cfg.get("num_iters", 0) + 1)
    else:
        evaluator_phase_a = cfg.get("evaluator", {}).get("phase_a", "l0")
        evaluator_phase_b = cfg.get("evaluator", {}).get("phase_b", "l1_fbcsp")
        switch_to_l1_iter = int(switch_to_l1_iter)

    evaluator_a = make_evaluator(str(evaluator_phase_a))
    evaluator_b = make_evaluator(str(evaluator_phase_b))

    # Optional: unlabeled cross-session domain-shift penalty (SAFE; uses eval features only).
    ds_cfg = cfg.get("reward", {}).get("domain_shift", {}) or {}
    ds_enabled = bool(ds_cfg.get("enabled", False))
    ds_eta = float(ds_cfg.get("eta", 0.0))
    ds_mode = str(ds_cfg.get("mode", "bp_mean_l2"))
    if ds_enabled and ds_eta > 0.0:
        evaluator_a = DomainShiftPenaltyEvaluator(evaluator_a, eta=ds_eta, mode=ds_mode, data_root=sampler.data_root, variant=variant)
        evaluator_b = DomainShiftPenaltyEvaluator(evaluator_b, eta=ds_eta, mode=ds_mode, data_root=sampler.data_root, variant=variant)

    # Optional: per-subject reward normalization (delta vs full-22 baseline).
    # This is a constant shift within each subject+split, so it preserves ordering but aligns scales across subjects.
    normalize = cfg.get("reward", {}).get("normalize", False)
    norm_mode = str(normalize).lower()

    def _apply_normalize(ev: EvaluatorBase) -> EvaluatorBase:
        if norm_mode in {"1", "true", "yes", "delta_full22", "delta"}:
            return DeltaFull22Evaluator(ev)
        if norm_mode in {"adv_lrmax", "adv_full22_lrmax", "adv_full22_lr_weight", "adv_lr_weight_max"}:
            return AdvantageMaxBaselineEvaluator(ev)
        return ev

    evaluator_a = _apply_normalize(evaluator_a)
    evaluator_b = _apply_normalize(evaluator_b)
    evaluator: EvaluatorBase = evaluator_a

    # Optional: mix network value with a cheap proxy at non-terminal MCTS leaves.
    # This helps early training when V(s) is noisy under sparse terminal rewards.
    leaf_cfg = cfg.get("mcts", {}).get("leaf_bootstrap", {}) or {}
    leaf_enabled = bool(leaf_cfg.get("enabled", False))
    leaf_alpha_start = float(leaf_cfg.get("alpha_start", 0.0))
    leaf_alpha_end = float(leaf_cfg.get("alpha_end", 1.0))
    leaf_warmup_iters = int(leaf_cfg.get("warmup_iters", 50))
    # Default: start mixing at Phase-B start (switch_to_l1_iter). Can be overridden to restart mixing on resume.
    leaf_start_iter = int(leaf_cfg.get("start_iter", switch_to_l1_iter))
    leaf_proxy_scale = float(leaf_cfg.get("proxy_scale", 1.0))
    leaf_proxy = str(leaf_cfg.get("proxy", "l0")).lower()

    # Optional: mix a heuristic policy prior (derived from fold stats) into the MCTS priors.
    pol_cfg = cfg.get("mcts", {}).get("policy_prior", {}) or {}
    pol_enabled = bool(pol_cfg.get("enabled", False))
    pol_eta_start = float(pol_cfg.get("eta_start", 0.0))
    pol_eta_end = float(pol_cfg.get("eta_end", 0.0))
    pol_warmup_iters = int(pol_cfg.get("warmup_iters", 50))
    pol_start_iter = int(pol_cfg.get("start_iter", 0))
    pol_temp = float(pol_cfg.get("temperature", 1.0))

    leaf_evaluator: EvaluatorBase | None = None
    if leaf_enabled:
        # Use a cheap proxy (fast, label-free on 0train). Default: fisher-based L0.
        if leaf_proxy in {"l0", "fisher"}:
            leaf_evaluator = L0Evaluator(
                lambda_cost=float(cfg["reward"]["lambda_cost"]),
                beta_redund=float(cfg["reward"]["beta_redund"]),
                artifact_gamma=float(cfg["reward"].get("artifact_gamma", 0.0)),
            )
        elif leaf_proxy in {"lr_weight", "l0_lr_weight", "l0_lr"}:
            leaf_evaluator = L0LrWeightEvaluator(
                lambda_cost=float(cfg["reward"]["lambda_cost"]),
                beta_redund=float(cfg["reward"]["beta_redund"]),
                artifact_gamma=float(cfg["reward"].get("artifact_gamma", 0.0)),
            )
        else:
            raise ValueError(f"Unknown mcts.leaf_bootstrap.proxy={leaf_proxy!r} (expected: l0|lr_weight)")
        if ds_enabled and ds_eta > 0.0:
            leaf_evaluator = DomainShiftPenaltyEvaluator(
                leaf_evaluator, eta=ds_eta, mode=ds_mode, data_root=sampler.data_root, variant=variant
            )
        leaf_evaluator = _apply_normalize(leaf_evaluator)

    # Optional: teacher KL / imitation-style regularization (lr_weight teacher).
    tcfg = train_cfg.get("teacher_kl", {}) or {}
    teacher_enabled = bool(tcfg.get("enabled", False))
    teacher_weight_start = float(tcfg.get("weight_start", 0.0))
    teacher_weight_end = float(tcfg.get("weight_end", 0.0))
    teacher_warmup_iters = int(tcfg.get("warmup_iters", 100))
    teacher_start_iter = int(tcfg.get("start_iter", switch_to_l1_iter))
    teacher_temp = float(tcfg.get("temperature", 1.0))

    net = PolicyValueNet(
        d_in=int(cfg["net"]["d_in"]),
        d_model=int(cfg["net"]["d_model"]),
        n_layers=int(cfg["net"]["n_layers"]),
        n_heads=int(cfg["net"]["n_heads"]),
        policy_mode=str(cfg.get("net", {}).get("policy_mode", "cls")),
        think_steps=int(cfg.get("net", {}).get("think_steps", 1) or 1),
        n_actions=23,
    ).to(torch.device(device))

    opt = torch.optim.AdamW(
        net.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    buffer = ReplayBuffer(capacity=int(cfg["train"]["buffer_capacity"]), n_actions=23, seed=seed)
    mcts = MCTS(
        net=net,
        state_builder=state_builder,
        evaluator=evaluator,
        leaf_evaluator=leaf_evaluator,
        leaf_value_mix_alpha=1.0,
        leaf_value_proxy_scale=float(leaf_proxy_scale),
        policy_prior_eta=0.0,
        policy_prior_temperature=float(pol_temp),
        infer_batch_size=int(cfg.get("mcts", {}).get("infer_batch_size", 1) or 1),
        n_sim=int(cfg["mcts"]["n_sim"]),
        c_puct=float(cfg["mcts"]["c_puct"]),
        dirichlet_alpha=float(cfg["mcts"]["dirichlet_alpha"]),
        dirichlet_eps=float(cfg["mcts"]["dirichlet_eps"]),
        device=device,
    )

    # Optional resume (NOTE: replay buffer is not serialized; resuming resets the buffer).
    start_iter = 0
    resume = bool(train_cfg.get("resume", False))
    resume_ckpt = train_cfg.get("resume_checkpoint", None)
    resume_optimizer = bool(train_cfg.get("resume_optimizer", True))

    # Checkpoint saving policy.
    ckpt_cfg = train_cfg.get("checkpoint", {}) or {}
    save_last = bool(ckpt_cfg.get("save_last", True))
    save_best = bool(ckpt_cfg.get("save_best", True))
    save_each_iter = bool(ckpt_cfg.get("save_each_iter", False))
    save_every = int(ckpt_cfg.get("save_every", 1) or 1)
    best_metric = str(ckpt_cfg.get("best_metric", "mean_reward")).lower()
    if best_metric not in {"mean_reward", "mean", "best_reward", "best"}:
        raise ValueError(f"train.checkpoint.best_metric={best_metric!r} (expected: mean_reward|best_reward)")
    best_metric_value = float("-inf")
    best_metric_iter = -1

    if resume:
        if isinstance(resume_ckpt, str) and resume_ckpt.lower() in {"last", "best"}:
            ckpt_path = paths.ckpt_dir / f"{resume_ckpt.lower()}.pt"
        else:
            ckpt_path = Path(resume_ckpt) if resume_ckpt else _latest_checkpoint(paths.ckpt_dir)
        if ckpt_path and ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            missing, unexpected = net.load_state_dict(ckpt.get("model", {}), strict=False)
            if missing or unexpected:
                print(f"[resume] WARNING: load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
            if "optimizer" in ckpt and resume_optimizer:
                try:
                    opt.load_state_dict(ckpt["optimizer"])
                    _move_optimizer_state_to_device(opt, torch.device(device))
                except Exception as e:
                    print(f"[resume] WARNING: failed to load optimizer state ({e}); continuing with fresh optimizer.")
            start_iter = int(ckpt.get("iter", -1)) + 1

            # phase alignment
            if start_iter > switch_to_l1_iter:
                evaluator = evaluator_b
                mcts.evaluator = evaluator
            # replay buffer reset for correctness (buffer not saved)
            buffer = ReplayBuffer(capacity=int(cfg["train"]["buffer_capacity"]), n_actions=23, seed=seed + start_iter)
            # resume best-metric tracking if present
            ckpt_metric = str(ckpt.get("best_metric", best_metric)).lower()
            if ckpt_metric == best_metric:
                best_metric_value = float(ckpt.get("best_metric_value", float("-inf")))
                best_metric_iter = int(ckpt.get("best_metric_iter", -1))
            else:
                print(
                    f"[resume] NOTE: checkpoint best_metric={ckpt_metric!r} != current {best_metric!r}; "
                    "resetting best-metric tracking."
                )
            print(f"[resume] checkpoint={ckpt_path} start_iter={start_iter} (buffer reset)")
        else:
            print("[resume] requested, but no checkpoint found; starting from scratch")

    rng = np.random.default_rng(seed + start_iter)
    temp_cfg = cfg["mcts"]["temperature"]
    calib_dir = paths.out_dir / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)

    # Training metrics (Q1-grade post-mortem diagnostics).
    metrics_mode = "a"
    if overwrite and (not resume):
        metrics_mode = "w"
    metrics_csv_path = paths.out_dir / "train_metrics.csv"
    metrics_jsonl_path = paths.out_dir / "train_metrics.jsonl"
    metrics_csv_f = open(metrics_csv_path, metrics_mode, encoding="utf-8", newline="")
    metrics_jsonl_f = open(metrics_jsonl_path, metrics_mode, encoding="utf-8")
    metrics_fields = [
        "iter",
        "phase",
        "buffer_size",
        "games_per_iter",
        "steps_per_iter",
        "batch_size",
        "device",
        "mcts_n_sim",
        "mcts_c_puct",
        "mcts_dirichlet_alpha",
        "mcts_dirichlet_eps",
        "mcts_infer_batch_size",
        "leaf_value_mix_alpha",
        "policy_prior_eta",
        "teacher_weight",
        "reward_mean",
        "reward_std",
        "reward_min",
        "reward_q20",
        "reward_median",
        "reward_q80",
        "reward_max",
        "reward_best",
        "reward_raw_mean",
        "reward_baseline_max_mean",
        "domain_shift_mean",
        "kappa_robust_mean",
        "kappa_mean_mean",
        "kappa_q20_mean",
        "acc_mean_mean",
        "n_ch_mean",
        "n_ch_std",
        "n_ch_min",
        "n_ch_max",
        "b_max_mean",
        "traj_len_mean",
        "pi_entropy_mean",
        "sp_subject_unique",
        "sp_split_unique",
        "stop_frac",
        "train_loss_total",
        "train_loss_pi",
        "train_loss_v",
        "train_loss_teacher",
        "train_policy_entropy",
        "train_grad_norm",
        "train_value_pred_mean",
        "train_value_tgt_mean",
        "time_selfplay_s",
        "time_train_s",
        "time_iter_s",
    ]
    metrics_writer = csv.DictWriter(metrics_csv_f, fieldnames=metrics_fields)
    if metrics_mode == "w" or metrics_csv_f.tell() == 0:
        metrics_writer.writeheader()
        metrics_csv_f.flush()

    # Compact hyperparameter snapshot for debugging (config.yaml is the full source of truth).
    hparams = {
        "project": {
            "out_dir": str(paths.out_dir),
            "seed": int(seed),
            "device": str(device),
        },
        "data": {"subjects": [int(s) for s in subjects], "variant": str(variant)},
        "game": {
            "b_max": int(cfg["game"]["b_max"]),
            "min_selected_for_stop": int(cfg["game"]["min_selected_for_stop"]),
        },
        "net": {
            "d_in": int(cfg["net"]["d_in"]),
            "d_model": int(cfg["net"]["d_model"]),
            "n_layers": int(cfg["net"]["n_layers"]),
            "n_heads": int(cfg["net"]["n_heads"]),
            "policy_mode": str(cfg.get("net", {}).get("policy_mode", "cls")),
            "think_steps": int(cfg.get("net", {}).get("think_steps", 1) or 1),
        },
        "mcts": {
            "n_sim": int(cfg["mcts"]["n_sim"]),
            "c_puct": float(cfg["mcts"]["c_puct"]),
            "dirichlet_alpha": float(cfg["mcts"]["dirichlet_alpha"]),
            "dirichlet_eps": float(cfg["mcts"]["dirichlet_eps"]),
            "infer_batch_size": int(cfg.get("mcts", {}).get("infer_batch_size", 1) or 1),
            "temperature": dict(cfg.get("mcts", {}).get("temperature", {})),
            "leaf_bootstrap": dict(cfg.get("mcts", {}).get("leaf_bootstrap", {}) or {}),
            "policy_prior": dict(cfg.get("mcts", {}).get("policy_prior", {}) or {}),
        },
        "evaluator": dict(cfg.get("evaluator", {}) or {}),
        "reward": {
            "lambda_cost": float(cfg["reward"]["lambda_cost"]),
            "beta_redund": float(cfg["reward"]["beta_redund"]),
            "artifact_gamma": float(cfg["reward"].get("artifact_gamma", 0.0)),
            "normalize": str(cfg.get("reward", {}).get("normalize", False)),
            "domain_shift": dict(cfg.get("reward", {}).get("domain_shift", {}) or {}),
        },
        "train": {
            "num_iters": int(cfg["train"]["num_iters"]),
            "games_per_iter": int(cfg["train"]["games_per_iter"]),
            "steps_per_iter": int(cfg["train"]["steps_per_iter"]),
            "batch_size": int(cfg["train"]["batch_size"]),
            "buffer_capacity": int(cfg["train"]["buffer_capacity"]),
            "lr": float(cfg["train"]["lr"]),
            "weight_decay": float(cfg["train"]["weight_decay"]),
            "switch_to_l1_iter": int(switch_to_l1_iter),
            "clear_buffer_on_switch": bool(clear_buffer_on_switch),
            "b_max_choices": cfg.get("train", {}).get("b_max_choices", None),
            "force_exact_budget": bool(cfg.get("train", {}).get("force_exact_budget", False)),
            "teacher_kl": dict(train_cfg.get("teacher_kl", {}) or {}),
            "selfplay": dict(train_cfg.get("selfplay", {}) or {}),
        },
        "_meta": dict(cfg.get("_meta", {}) or {}),
    }
    (paths.out_dir / "hparams.json").write_text(json.dumps(hparams, indent=2), encoding="utf-8")

    print(
        f"[run] out_dir={paths.out_dir} seed={seed} device={device} variant={variant} subjects={len(subjects)} "
        f"phase_a={evaluator_phase_a} phase_b={evaluator_phase_b} switch_to_l1_iter={switch_to_l1_iter}"
    )
    print(
        f"[hparams] mcts.n_sim={cfg['mcts']['n_sim']} c_puct={cfg['mcts']['c_puct']} "
        f"dir(alpha/eps)={cfg['mcts']['dirichlet_alpha']}/{cfg['mcts']['dirichlet_eps']} "
        f"infer_bs={cfg.get('mcts', {}).get('infer_batch_size', 1)} normalize={norm_mode} "
        f"ds(enabled={ds_enabled},eta={ds_eta},mode={ds_mode})"
    )

    # Optional: parallel self-play (thread workers sharing the same net).
    sp_cfg = train_cfg.get("selfplay", {}) or {}
    sp_workers = int(sp_cfg.get("num_workers", 0) or 0)
    sp_workers = max(0, int(sp_workers))
    sp_device_cfg = sp_cfg.get("device", None)
    sp_chunksize_cfg = int(sp_cfg.get("chunksize", 1) or 1)
    sp_executor = None
    sp_thread_workers: list[_SelfPlayWorker] = []
    if sp_device_cfg not in (None, "null", "None", "") and str(sp_device_cfg) != str(device):
        print(f"[selfplay] NOTE: train.selfplay.device={sp_device_cfg!r} ignored (thread backend uses project.device={device!r})")
    if sp_chunksize_cfg != 1:
        print(f"[selfplay] NOTE: train.selfplay.chunksize={sp_chunksize_cfg} ignored (thread backend)")

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

    def _make_phase_evaluator(phase_name: str) -> EvaluatorBase:
        ev = make_evaluator(str(phase_name))
        if ds_enabled and ds_eta > 0.0:
            ev = DomainShiftPenaltyEvaluator(ev, eta=ds_eta, mode=ds_mode, data_root=sampler.data_root, variant=variant)
        return _apply_normalize(ev)

    def _make_leaf_evaluator() -> EvaluatorBase | None:
        if not leaf_enabled:
            return None
        if leaf_proxy in {"l0", "fisher"}:
            ev: EvaluatorBase = L0Evaluator(
                lambda_cost=float(cfg["reward"]["lambda_cost"]),
                beta_redund=float(cfg["reward"]["beta_redund"]),
                artifact_gamma=float(cfg["reward"].get("artifact_gamma", 0.0)),
            )
        elif leaf_proxy in {"lr_weight", "l0_lr_weight", "l0_lr"}:
            ev = L0LrWeightEvaluator(
                lambda_cost=float(cfg["reward"]["lambda_cost"]),
                beta_redund=float(cfg["reward"]["beta_redund"]),
                artifact_gamma=float(cfg["reward"].get("artifact_gamma", 0.0)),
            )
        else:
            raise ValueError(f"Unknown mcts.leaf_bootstrap.proxy={leaf_proxy!r} (expected: l0|lr_weight)")
        if ds_enabled and ds_eta > 0.0:
            ev = DomainShiftPenaltyEvaluator(ev, eta=ds_eta, mode=ds_mode, data_root=sampler.data_root, variant=variant)
        return _apply_normalize(ev)

    class _SelfPlayWorker:
        def __init__(self) -> None:
            # Each worker has its own evaluator caches and MCTS tree (thread-safe).
            self.evaluator_a = _make_phase_evaluator(evaluator_phase_a)
            self.evaluator_b = _make_phase_evaluator(evaluator_phase_b)
            self.leaf_evaluator = _make_leaf_evaluator()
            self.mcts = MCTS(
                net=net,
                state_builder=state_builder,
                evaluator=self.evaluator_a,
                leaf_evaluator=self.leaf_evaluator,
                leaf_value_mix_alpha=1.0,
                leaf_value_proxy_scale=float(leaf_proxy_scale),
                policy_prior_eta=0.0,
                policy_prior_temperature=float(pol_temp),
                infer_batch_size=int(cfg.get("mcts", {}).get("infer_batch_size", 1) or 1),
                n_sim=int(cfg["mcts"]["n_sim"]),
                c_puct=float(cfg["mcts"]["c_puct"]),
                dirichlet_alpha=float(cfg["mcts"]["dirichlet_alpha"]),
                dirichlet_eps=float(cfg["mcts"]["dirichlet_eps"]),
                device=device,
            )

        def play_games(
            self,
            *,
            it: int,
            seeds: list[int],
            leaf_alpha: float,
            pol_eta: float,
            b_max_choices: list[int],
            force_exact_budget: bool,
        ) -> list[dict[str, Any]]:
            outs: list[dict[str, Any]] = []
            ev = self.evaluator_a if int(it) < int(switch_to_l1_iter) else self.evaluator_b
            self.mcts.evaluator = ev
            self.mcts.leaf_value_mix_alpha = float(leaf_alpha)
            self.mcts.policy_prior_eta = float(pol_eta)

            for seed_i in seeds:
                rng_i = np.random.default_rng(int(seed_i))
                subject_i = int(rng_i.choice(subjects))
                split_id_i = int(rng_i.integers(0, sampler.n_splits))
                fold_i = sampler.get_fold(subject_i, split_id_i)
                b_max_game = int(rng_i.choice(b_max_choices)) if b_max_choices else int(state_builder.b_max)
                min_stop = int(b_max_game) if force_exact_budget else int(state_builder.min_selected_for_stop)

                env = EEGChannelGame(
                    fold=fold_i,
                    state_builder=state_builder,
                    evaluator=ev,
                    b_max=b_max_game,
                    min_selected_for_stop=min_stop,
                )
                _ = env.reset()
                self.mcts.reset()

                traj: list[tuple[int, int, int, int, int, np.ndarray]] = []
                done = False
                info: dict[str, Any] = {}
                while not done:
                    n_sel = popcount(env.key)
                    pi = self.mcts.run(
                        root_key=env.key,
                        fold=env.fold,
                        add_root_noise=True,
                        b_max=int(env.b_max),
                        min_selected_for_stop=int(env.min_selected_for_stop),
                        rng=rng_i,
                    )
                    cur_tau = float(temp_cfg["tau"]) if n_sel < int(temp_cfg["warmup_steps"]) else float(
                        temp_cfg["final_tau"]
                    )
                    a = _sample_from_pi(pi, cur_tau, rng_i)
                    traj.append(
                        (
                            int(env.key),
                            int(subject_i),
                            int(split_id_i),
                            int(env.b_max),
                            int(env.min_selected_for_stop),
                            pi,
                        )
                    )
                    _, r, done, info = env.step(a)

                outs.append({"reward": float(r), "key": int(env.key), "info": info, "traj": traj})
            return outs

    if sp_workers > 0:
        from concurrent.futures import ThreadPoolExecutor

        sp_executor = ThreadPoolExecutor(max_workers=int(sp_workers))
        sp_thread_workers = [_SelfPlayWorker() for _ in range(int(sp_workers))]
        infer_bs = int(cfg.get("mcts", {}).get("infer_batch_size", 1) or 1)
        print(f"[selfplay] parallel workers={sp_workers} infer_batch_size={infer_bs}")

    def _run_one_iter(it: int) -> None:
        nonlocal evaluator, buffer, best_metric_value, best_metric_iter

        t_iter0 = time.perf_counter()

        # Leaf bootstrap schedule (Phase B only by default).
        if leaf_enabled and it >= leaf_start_iter:
            t = int(it - leaf_start_iter)
            if leaf_warmup_iters <= 0:
                alpha = float(leaf_alpha_end)
            else:
                frac = float(min(1.0, max(0.0, t / float(leaf_warmup_iters))))
                alpha = float(leaf_alpha_start + frac * (leaf_alpha_end - leaf_alpha_start))
            mcts.leaf_value_mix_alpha = float(alpha)
        else:
            mcts.leaf_value_mix_alpha = 1.0

        # Policy-prior schedule (typically early-only; default OFF).
        if pol_enabled and it >= pol_start_iter:
            t = int(it - pol_start_iter)
            if pol_warmup_iters <= 0:
                eta = float(pol_eta_end)
            else:
                frac = float(min(1.0, max(0.0, t / float(pol_warmup_iters))))
                eta = float(pol_eta_start + frac * (pol_eta_end - pol_eta_start))
            mcts.policy_prior_eta = float(eta)
        else:
            mcts.policy_prior_eta = 0.0

        # Teacher schedule (typically Phase B only).
        if teacher_enabled and it >= teacher_start_iter:
            t = int(it - teacher_start_iter)
            if teacher_warmup_iters <= 0:
                teacher_w = float(teacher_weight_end)
            else:
                frac = float(min(1.0, max(0.0, t / float(teacher_warmup_iters))))
                teacher_w = float(teacher_weight_start + frac * (teacher_weight_end - teacher_weight_start))
        else:
            teacher_w = 0.0

        # phase switch
        if it == switch_to_l1_iter:
            evaluator = evaluator_b
            mcts.evaluator = evaluator
            if clear_buffer_on_switch:
                buffer = ReplayBuffer(capacity=int(cfg["train"]["buffer_capacity"]), n_actions=23, seed=seed)
            print(f"[phase] switch iter={it} -> {evaluator_phase_b} (clear_buffer={clear_buffer_on_switch})")

        rewards: list[float] = []
        best = (-1e9, None, None)  # (reward, key, info)
        game_infos: list[dict[str, Any]] = []
        game_subjects: list[int] = []
        game_splits: list[int] = []
        game_b_max: list[int] = []
        game_n_ch: list[int] = []
        game_traj_len: list[int] = []
        game_pi_entropy: list[float] = []

        # Optional: sample a per-episode budget (b_max) to train a single policy/value that works across K.
        b_max_choices = train_cfg.get("b_max_choices", None)
        if isinstance(b_max_choices, str):
            b_max_choices = [int(s.strip()) for s in b_max_choices.split(",") if s.strip()]
        elif isinstance(b_max_choices, (list, tuple)):
            b_max_choices = [int(x) for x in b_max_choices]
        else:
            b_max_choices = []
        force_exact_budget = bool(train_cfg.get("force_exact_budget", False))

        # self-play
        t_sp0 = time.perf_counter()
        games_per_iter = int(cfg["train"]["games_per_iter"])
        if sp_executor is None:
            for _ in range(games_per_iter):
                fold = sampler.sample_fold()
                b_max_game = int(rng.choice(b_max_choices)) if b_max_choices else int(state_builder.b_max)
                min_stop = int(b_max_game) if force_exact_budget else int(state_builder.min_selected_for_stop)
                env = EEGChannelGame(
                    fold=fold,
                    state_builder=state_builder,
                    evaluator=evaluator,
                    b_max=b_max_game,
                    min_selected_for_stop=min_stop,
                )
                mcts.reset()
                info = play_one_game(
                    env=env,
                    mcts=mcts,
                    buffer=buffer,
                    rng=rng,
                    temp_warmup_steps=int(temp_cfg["warmup_steps"]),
                    tau=float(temp_cfg["tau"]),
                    final_tau=float(temp_cfg["final_tau"]),
                )
                r = float(info.get("reward", 0.0))
                rewards.append(r)
                game_infos.append(dict(info or {}))
                game_subjects.append(int(env.fold.subject))
                game_splits.append(int(env.fold.split_id))
                game_b_max.append(int(env.b_max))
                game_n_ch.append(int(popcount(env.key)))
                game_traj_len.append(int(_to_float(info.get("traj_len")) or 0))
                game_pi_entropy.append(float(_to_float(info.get("pi_entropy_mean")) or 0.0))
                if r > best[0]:
                    best = (r, int(env.key), info)
        else:
            leaf_alpha = float(mcts.leaf_value_mix_alpha)
            pol_eta = float(mcts.policy_prior_eta)
            seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(int(games_per_iter))]

            # Shard seeds to workers to amortize executor overhead.
            shards: list[list[int]] = [[] for _ in range(int(sp_workers))]
            for i, s in enumerate(seeds):
                shards[int(i) % int(sp_workers)].append(int(s))

            futs = []
            for worker, shard in zip(sp_thread_workers, shards):
                if not shard:
                    continue
                futs.append(
                    sp_executor.submit(
                        worker.play_games,
                        it=int(it),
                        seeds=shard,
                        leaf_alpha=float(leaf_alpha),
                        pol_eta=float(pol_eta),
                        b_max_choices=b_max_choices,
                        force_exact_budget=bool(force_exact_budget),
                    )
                )

            for fut in futs:
                outs = fut.result()
                for out in outs:
                    r = float(out.get("reward", 0.0))
                    rewards.append(r)
                    info = dict(out.get("info", {}) or {})
                    game_infos.append(info)
                    traj = out.get("traj", []) or []
                    if traj:
                        game_subjects.append(int(traj[0][1]))
                        game_splits.append(int(traj[0][2]))
                        game_b_max.append(int(traj[0][3]))
                        key_final = int(out.get("key", 0))
                        game_n_ch.append(int(popcount(key_final)))
                        game_traj_len.append(int(len(traj)))
                        eps = 1e-12
                        ent = []
                        for _, _, _, _, _, pi in traj:
                            p = np.asarray(pi, dtype=np.float64)
                            ent.append(float(-(p * np.log(p + eps)).sum()))
                        game_pi_entropy.append(float(np.mean(ent)) if ent else 0.0)
                    if r > best[0]:
                        best = (r, int(out.get("key", 0)), out.get("info", {}))
                    for key, subject, split_id, b_max, min_selected_for_stop, pi in out.get("traj", []):
                        buffer.add(
                            key=int(key),
                            subject=int(subject),
                            split_id=int(split_id),
                            b_max=int(b_max),
                            min_selected_for_stop=int(min_selected_for_stop),
                            pi=np.asarray(pi, dtype=np.float32),
                            z=float(r),
                        )
        t_sp1 = time.perf_counter()

        # training
        t_tr0 = time.perf_counter()
        net.train()
        steps_per_iter = int(cfg["train"]["steps_per_iter"])
        batch_size = int(cfg["train"]["batch_size"])
        loss_pi_sum = 0.0
        loss_v_sum = 0.0
        loss_teacher_sum = 0.0
        loss_sum = 0.0
        entropy_sum = 0.0
        grad_norm_sum = 0.0
        value_pred_sum = 0.0
        value_tgt_sum = 0.0
        for _ in range(int(cfg["train"]["steps_per_iter"])):
            batch = buffer.sample(batch_size)
            tokens_np, mask_np, teacher_np = _build_batch(
                sampler=sampler,
                state_builder=state_builder,
                keys=batch["key"],
                subjects=batch["subject"],
                split_ids=batch["split_id"],
                b_max=batch["b_max"],
                min_selected_for_stop=batch["min_selected_for_stop"],
                teacher_temp=float(teacher_temp) if teacher_w > 0.0 else None,
            )
            tokens = torch.from_numpy(tokens_np).to(device)
            action_mask = torch.from_numpy(mask_np).to(device)
            pi_tgt = torch.from_numpy(batch["pi"]).to(device)
            z = torch.from_numpy(batch["z"]).to(device)

            logits, value = net(tokens, action_mask=action_mask)
            logp = torch.log_softmax(logits, dim=-1)
            loss_pi = -(pi_tgt * logp).sum(dim=-1).mean()
            loss_v = torch.mean((value - z) ** 2)
            loss = loss_pi + loss_v
            with torch.no_grad():
                p = torch.softmax(logits, dim=-1)
                entropy = -(p * logp).sum(dim=-1).mean()
            if teacher_w > 0.0 and teacher_np is not None:
                teacher_tgt = torch.from_numpy(teacher_np).to(device)
                loss_teacher = -(teacher_tgt * logp).sum(dim=-1).mean()
                loss = loss + float(teacher_w) * loss_teacher
            else:
                loss_teacher = torch.zeros((), device=device)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            loss_pi_sum += float(loss_pi.detach().cpu().item())
            loss_v_sum += float(loss_v.detach().cpu().item())
            loss_teacher_sum += float(loss_teacher.detach().cpu().item())
            loss_sum += float(loss.detach().cpu().item())
            entropy_sum += float(entropy.detach().cpu().item())
            grad_norm_sum += float(gn.detach().cpu().item()) if torch.is_tensor(gn) else float(gn)
            value_pred_sum += float(value.detach().mean().cpu().item())
            value_tgt_sum += float(z.detach().mean().cpu().item())
        t_tr1 = time.perf_counter()

        steps_denom = float(max(1, steps_per_iter))
        loss_pi_mean = loss_pi_sum / steps_denom
        loss_v_mean = loss_v_sum / steps_denom
        loss_teacher_mean = loss_teacher_sum / steps_denom
        loss_mean = loss_sum / steps_denom
        entropy_mean = entropy_sum / steps_denom
        grad_norm_mean = grad_norm_sum / steps_denom
        value_pred_mean = value_pred_sum / steps_denom
        value_tgt_mean = value_tgt_sum / steps_denom

        mean_r = float(np.mean(rewards)) if rewards else 0.0
        std_r = float(np.std(rewards, ddof=0)) if rewards else 0.0
        q = _quantiles(rewards, qs=(0.2, 0.5, 0.8))
        reward_q20 = float(q[0.2])
        reward_med = float(q[0.5])
        reward_q80 = float(q[0.8])
        reward_min = float(np.min(rewards)) if rewards else 0.0
        reward_max = float(np.max(rewards)) if rewards else 0.0

        # Self-play diagnostics (aggregated).
        n_ch_mean = float(np.mean(game_n_ch)) if game_n_ch else 0.0
        n_ch_std = float(np.std(game_n_ch, ddof=0)) if game_n_ch else 0.0
        n_ch_min = int(np.min(game_n_ch)) if game_n_ch else 0
        n_ch_max = int(np.max(game_n_ch)) if game_n_ch else 0
        b_max_mean = float(np.mean(game_b_max)) if game_b_max else 0.0
        traj_len_mean = float(np.mean(game_traj_len)) if game_traj_len else 0.0
        pi_entropy_mean = float(np.mean(game_pi_entropy)) if game_pi_entropy else 0.0
        sp_subject_unique = int(len(set(int(s) for s in game_subjects))) if game_subjects else 0
        sp_split_unique = int(len(set(int(s) for s in game_splits))) if game_splits else 0
        stop_frac = float(
            np.mean([1.0 if int(n) < int(b) else 0.0 for n, b in zip(game_n_ch, game_b_max)])
        ) if game_n_ch and game_b_max and len(game_n_ch) == len(game_b_max) else 0.0

        _xs = _collect(game_infos, "reward_raw")
        reward_raw_mean = float(np.mean(_xs)) if _xs else 0.0
        _xs = _collect(game_infos, "reward_baseline_max")
        reward_baseline_max_mean = float(np.mean(_xs)) if _xs else 0.0
        _xs = _collect(game_infos, "domain_shift")
        domain_shift_mean = float(np.mean(_xs)) if _xs else 0.0
        _xs = _collect(game_infos, "kappa_robust")
        kappa_robust_mean = float(np.mean(_xs)) if _xs else 0.0
        _xs = _collect(game_infos, "kappa_mean")
        kappa_mean_mean = float(np.mean(_xs)) if _xs else 0.0
        _xs = _collect(game_infos, "kappa_q20")
        kappa_q20_mean = float(np.mean(_xs)) if _xs else 0.0
        _xs = _collect(game_infos, "acc_mean")
        acc_mean_mean = float(np.mean(_xs)) if _xs else 0.0

        phase = str(evaluator_phase_a) if it < switch_to_l1_iter else str(evaluator_phase_b)
        print(
            f"[iter {it:03d}] phase={phase} buffer={buffer.size} "
            f"mean_reward={mean_r:.4f} q20_reward={reward_q20:.4f} best_reward={best[0]:.4f} "
            f"loss(pi/v/t)={loss_pi_mean:.4f}/{loss_v_mean:.4f}/{loss_teacher_mean:.4f} "
            f"H(pi)={entropy_mean:.3f} stop_frac={stop_frac:.2f}"
        )

        metric_now = mean_r if best_metric in {"mean_reward", "mean"} else float(best[0])
        is_new_best = bool(metric_now > best_metric_value)
        if is_new_best:
            best_metric_value = float(metric_now)
            best_metric_iter = int(it)

        ckpt = {
            "iter": it,
            "model": net.state_dict(),
            "optimizer": opt.state_dict(),
            "buffer_size": buffer.size,
            "best_reward": best[0],
            "best_key": best[1],
            "best_info": best[2],
            "mean_reward": mean_r,
            "best_metric": best_metric,
            "best_metric_value": best_metric_value,
            "best_metric_iter": best_metric_iter,
            "cfg": cfg,
        }
        if save_last:
            torch.save(ckpt, paths.ckpt_dir / "last.pt")
        if save_best and is_new_best:
            torch.save(ckpt, paths.ckpt_dir / "best.pt")
        if save_each_iter and (it % save_every == 0):
            torch.save(ckpt, paths.ckpt_dir / f"iter_{it:03d}.pt")

        # Offline L2 calibration (never used for training reward)
        l2_every = int(train_cfg.get("l2_calib_every", 0) or 0)
        allow_eval_labels = bool(train_cfg.get("allow_eval_labels", False))
        if l2_every > 0 and allow_eval_labels and it >= switch_to_l1_iter and (it % l2_every == 0):
            try:
                from eeg_channel_game.eeg.io import load_subject_data
                from eeg_channel_game.eval.evaluator_l2_deep import evaluate_l2_deep_train_eval
            except Exception as e:  # pragma: no cover
                print(f"[l2-calib] skipped: {e}")
            else:
                top_n = int(train_cfg.get("l2_calib_top_n", 10))
                l2_model = str(train_cfg.get("l2_calib_model", "eegnetv4"))
                l2_device = str(train_cfg.get("l2_calib_device", device))
                l2_epochs = int(train_cfg.get("l2_calib_epochs", 20))
                l2_batch = int(train_cfg.get("l2_calib_batch_size", 64))
                l2_lr = float(train_cfg.get("l2_calib_lr", 1e-3))
                l2_wd = float(train_cfg.get("l2_calib_weight_decay", 1e-4))
                l2_pat = int(train_cfg.get("l2_calib_patience", 8))
                l2_seeds = train_cfg.get("l2_calib_seeds", [0, 1, 2])
                if isinstance(l2_seeds, str):
                    l2_seeds = [int(s.strip()) for s in l2_seeds.split(",") if s.strip()]
                elif isinstance(l2_seeds, int):
                    l2_seeds = [int(l2_seeds)]

                arr = buffer.as_arrays()
                if arr["z"].size == 0:
                    calib_rows = []
                else:
                    order = np.argsort(arr["z"])[::-1]
                    chosen = []
                    seen = set()
                    for idx in order:
                        tup = (int(arr["subject"][idx]), int(arr["split_id"][idx]), int(arr["key"][idx]))
                        if tup in seen:
                            continue
                        seen.add(tup)
                        chosen.append(int(idx))
                        if len(chosen) >= top_n:
                            break

                    calib_rows = []
                    for idx in chosen:
                        subj = int(arr["subject"][idx])
                        split_id = int(arr["split_id"][idx])
                        key = int(arr["key"][idx])
                        z = float(arr["z"][idx])

                        fold = sampler.get_fold(subj, split_id)
                        sd_eval = load_subject_data(
                            subj, data_root=sampler.data_root, variant=variant, include_eval=True
                        )
                        sel_mask = key_to_mask(key, n_ch=22)
                        sel_idx = [i for i in range(22) if int(sel_mask[i]) == 1]
                        if len(sel_idx) < 2:
                            continue

                        res = evaluate_l2_deep_train_eval(
                            subject_data=sd_eval,
                            sel_idx=sel_idx,
                            train_idx=fold.split.train_idx,
                            val_idx=fold.split.val_idx,
                            model_name=l2_model,
                            seeds=l2_seeds,
                            device=l2_device,
                            epochs=l2_epochs,
                            batch_size=l2_batch,
                            lr=l2_lr,
                            weight_decay=l2_wd,
                            patience=l2_pat,
                        )
                        calib_rows.append(
                            {
                                "iter": int(it),
                                "subject": subj,
                                "split_id": split_id,
                                "key": key,
                                "n_ch": int(len(sel_idx)),
                                "reward_l1": z,
                                "l2": res,
                            }
                        )

                # Correlation (sanity check): L1 reward vs L2 kappa
                if calib_rows:
                    xs = np.array([r["reward_l1"] for r in calib_rows], dtype=np.float32)
                    ys = np.array([r["l2"]["kappa_mean"] for r in calib_rows], dtype=np.float32)
                    if xs.size >= 2 and np.isfinite(xs).all() and np.isfinite(ys).all():
                        sx = float(xs.std(ddof=0))
                        sy = float(ys.std(ddof=0))
                        if sx > 1e-12 and sy > 1e-12:
                            xc = xs - float(xs.mean())
                            yc = ys - float(ys.mean())
                            corr = float((xc * yc).mean() / (sx * sy))
                        else:
                            corr = 0.0
                    else:
                        corr = 0.0
                    out = {
                        "iter": int(it),
                        "variant": variant,
                        "l2_model": l2_model,
                        "corr_reward_l2kappa": corr,
                        "rows": calib_rows,
                    }
                    out_path = calib_dir / f"l2_calib_iter{it:03d}.json"
                    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
                    print(f"[l2-calib] iter={it} top={len(calib_rows)} corr={corr:.3f} saved={out_path}")
        elif l2_every > 0 and (not allow_eval_labels) and it == switch_to_l1_iter:
            print("[l2-calib] disabled (train.allow_eval_labels=false)")

        # Persist per-iteration metrics for post-mortem.
        row = {
            "iter": int(it),
            "phase": phase,
            "buffer_size": int(buffer.size),
            "games_per_iter": int(games_per_iter),
            "steps_per_iter": int(steps_per_iter),
            "batch_size": int(batch_size),
            "device": str(device),
            "mcts_n_sim": int(cfg["mcts"]["n_sim"]),
            "mcts_c_puct": float(cfg["mcts"]["c_puct"]),
            "mcts_dirichlet_alpha": float(cfg["mcts"]["dirichlet_alpha"]),
            "mcts_dirichlet_eps": float(cfg["mcts"]["dirichlet_eps"]),
            "mcts_infer_batch_size": int(cfg.get("mcts", {}).get("infer_batch_size", 1) or 1),
            "leaf_value_mix_alpha": float(mcts.leaf_value_mix_alpha),
            "policy_prior_eta": float(mcts.policy_prior_eta),
            "teacher_weight": float(teacher_w),
            "reward_mean": float(mean_r),
            "reward_std": float(std_r),
            "reward_min": float(reward_min),
            "reward_q20": float(reward_q20),
            "reward_median": float(reward_med),
            "reward_q80": float(reward_q80),
            "reward_max": float(reward_max),
            "reward_best": float(best[0]),
            "reward_raw_mean": float(reward_raw_mean),
            "reward_baseline_max_mean": float(reward_baseline_max_mean),
            "domain_shift_mean": float(domain_shift_mean),
            "kappa_robust_mean": float(kappa_robust_mean),
            "kappa_mean_mean": float(kappa_mean_mean),
            "kappa_q20_mean": float(kappa_q20_mean),
            "acc_mean_mean": float(acc_mean_mean),
            "n_ch_mean": float(n_ch_mean),
            "n_ch_std": float(n_ch_std),
            "n_ch_min": int(n_ch_min),
            "n_ch_max": int(n_ch_max),
            "b_max_mean": float(b_max_mean),
            "traj_len_mean": float(traj_len_mean),
            "pi_entropy_mean": float(pi_entropy_mean),
            "sp_subject_unique": int(sp_subject_unique),
            "sp_split_unique": int(sp_split_unique),
            "stop_frac": float(stop_frac),
            "train_loss_total": float(loss_mean),
            "train_loss_pi": float(loss_pi_mean),
            "train_loss_v": float(loss_v_mean),
            "train_loss_teacher": float(loss_teacher_mean),
            "train_policy_entropy": float(entropy_mean),
            "train_grad_norm": float(grad_norm_mean),
            "train_value_pred_mean": float(value_pred_mean),
            "train_value_tgt_mean": float(value_tgt_mean),
            "time_selfplay_s": float(t_sp1 - t_sp0),
            "time_train_s": float(t_tr1 - t_tr0),
            "time_iter_s": float(time.perf_counter() - t_iter0),
        }
        metrics_writer.writerow({k: row.get(k, "") for k in metrics_fields})
        metrics_csv_f.flush()
        metrics_jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics_jsonl_f.flush()
    try:
        for it in range(int(start_iter), int(cfg["train"]["num_iters"])):
            _run_one_iter(int(it))
    finally:
        if sp_executor is not None:
            sp_executor.shutdown(wait=True)
        metrics_csv_f.close()
        metrics_jsonl_f.close()

    return paths
