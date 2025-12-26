from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eeg_channel_game.eeg.fold_sampler import FoldSampler
from eeg_channel_game.eeg.variant import variant_from_cfg
from eeg_channel_game.eval.evaluator_l0 import L0Evaluator
from eeg_channel_game.eval.evaluator_l0_lr_weight import L0LrWeightEvaluator
from eeg_channel_game.eval.evaluator_l1_fbcsp import L1FBCSPEvaluator
from eeg_channel_game.eval.evaluator_l1_deep_masked import L1DeepMaskedEvaluator, L1DeepMaskedTrainConfig
from eeg_channel_game.eval.evaluator_base import EvaluatorBase
from eeg_channel_game.eval.evaluator_domain_shift import DomainShiftPenaltyEvaluator
from eeg_channel_game.eval.evaluator_normalize import DeltaFull22Evaluator
from eeg_channel_game.game.env import EEGChannelGame
from eeg_channel_game.game.state_builder import StateBuilder
from eeg_channel_game.mcts.mcts import MCTS
from eeg_channel_game.model.policy_value_net import PolicyValueNet
from eeg_channel_game.rl.replay_buffer import ReplayBuffer
from eeg_channel_game.rl.selfplay import play_one_game
from eeg_channel_game.utils.config import RunPaths, make_run_paths
from eeg_channel_game.utils.seed import set_global_seed
from eeg_channel_game.utils.bitmask import key_to_mask


def _latest_checkpoint(ckpt_dir: Path) -> Path | None:
    pts = sorted(Path(ckpt_dir).glob("iter_*.pt"))
    return pts[-1] if pts else None


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
        has_ckpts = ckpt_dir.exists() and any(ckpt_dir.glob("iter_*.pt"))
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
    if str(normalize).lower() in {"1", "true", "yes", "delta_full22", "delta"}:
        evaluator_a = DeltaFull22Evaluator(evaluator_a)
        evaluator_b = DeltaFull22Evaluator(evaluator_b)
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
        if str(normalize).lower() in {"1", "true", "yes", "delta_full22", "delta"}:
            leaf_evaluator = DeltaFull22Evaluator(leaf_evaluator)

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
    if resume:
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
            print(f"[resume] checkpoint={ckpt_path} start_iter={start_iter} (buffer reset)")
        else:
            print("[resume] requested, but no checkpoint found; starting from scratch")

    rng = np.random.default_rng(seed + start_iter)
    temp_cfg = cfg["mcts"]["temperature"]
    calib_dir = paths.out_dir / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)

    for it in range(int(start_iter), int(cfg["train"]["num_iters"])):
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

        rewards = []
        best = (-1e9, None, None)  # (reward, key, info)

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
        for _ in range(int(cfg["train"]["games_per_iter"])):
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
            if r > best[0]:
                best = (r, int(env.key), info)

        # training
        net.train()
        for _ in range(int(cfg["train"]["steps_per_iter"])):
            batch = buffer.sample(int(cfg["train"]["batch_size"]))
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
            if teacher_w > 0.0 and teacher_np is not None:
                teacher_tgt = torch.from_numpy(teacher_np).to(device)
                loss_teacher = -(teacher_tgt * logp).sum(dim=-1).mean()
                loss = loss + float(teacher_w) * loss_teacher

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

        mean_r = float(np.mean(rewards)) if rewards else 0.0
        phase = str(evaluator_phase_a) if it < switch_to_l1_iter else str(evaluator_phase_b)
        print(f"[iter {it:03d}] phase={phase} buffer={buffer.size} mean_reward={mean_r:.4f} best_reward={best[0]:.4f}")

        ckpt = {
            "iter": it,
            "model": net.state_dict(),
            "optimizer": opt.state_dict(),
            "buffer_size": buffer.size,
            "best_reward": best[0],
            "best_key": best[1],
            "best_info": best[2],
            "cfg": cfg,
        }
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
                continue

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
                continue
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
                sd_eval = load_subject_data(subj, data_root=sampler.data_root, variant=variant, include_eval=True)
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
                out = {"iter": int(it), "variant": variant, "l2_model": l2_model, "corr_reward_l2kappa": corr, "rows": calib_rows}
                out_path = calib_dir / f"l2_calib_iter{it:03d}.json"
                out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
                print(f"[l2-calib] iter={it} top={len(calib_rows)} corr={corr:.3f} saved={out_path}")
        elif l2_every > 0 and (not allow_eval_labels) and it == switch_to_l1_iter:
            print("[l2-calib] disabled (train.allow_eval_labels=false)")

    return paths
