from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eeg_channel_game.eeg.fold_sampler import FoldSampler
from eeg_channel_game.eeg.variant import variant_from_cfg
from eeg_channel_game.eval.evaluator_l0 import L0Evaluator
from eeg_channel_game.eval.evaluator_l1_fbcsp import L1FBCSPEvaluator
from eeg_channel_game.eval.evaluator_base import EvaluatorBase
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
) -> tuple[np.ndarray, np.ndarray]:
    bsz = int(keys.shape[0])
    tokens = np.empty((bsz, 24, state_builder.d_in), dtype=np.float32)
    action_mask = np.empty((bsz, 23), dtype=bool)
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
    return tokens, action_mask


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
        if name == "l1_fbcsp":
            l1_cfg = cfg.get("evaluator", {}).get("l1_fbcsp", {})
            ev = L1FBCSPEvaluator(
                lambda_cost=float(cfg["reward"]["lambda_cost"]),
                artifact_gamma=float(cfg["reward"].get("artifact_gamma", 0.0)),
                cv_folds=int(l1_cfg.get("cv_folds", 3)),
                robust_mode=str(l1_cfg.get("robust_mode", "mean_std")),
                robust_beta=float(l1_cfg.get("robust_beta", 0.5)),
                variant=variant,
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

    # Optional: per-subject reward normalization (delta vs full-22 baseline).
    # This is a constant shift within each subject+split, so it preserves ordering but aligns scales across subjects.
    normalize = cfg.get("reward", {}).get("normalize", False)
    if str(normalize).lower() in {"1", "true", "yes", "delta_full22", "delta"}:
        evaluator_a = DeltaFull22Evaluator(evaluator_a)
        evaluator_b = DeltaFull22Evaluator(evaluator_b)
    evaluator: EvaluatorBase = evaluator_a

    net = PolicyValueNet(
        d_in=int(cfg["net"]["d_in"]),
        d_model=int(cfg["net"]["d_model"]),
        n_layers=int(cfg["net"]["n_layers"]),
        n_heads=int(cfg["net"]["n_heads"]),
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
    if resume:
        ckpt_path = Path(resume_ckpt) if resume_ckpt else _latest_checkpoint(paths.ckpt_dir)
        if ckpt_path and ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            missing, unexpected = net.load_state_dict(ckpt.get("model", {}), strict=False)
            if missing or unexpected:
                print(f"[resume] WARNING: load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
            if "optimizer" in ckpt:
                opt.load_state_dict(ckpt["optimizer"])
                _move_optimizer_state_to_device(opt, torch.device(device))
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
            tokens_np, mask_np = _build_batch(
                sampler=sampler,
                state_builder=state_builder,
                keys=batch["key"],
                subjects=batch["subject"],
                split_ids=batch["split_id"],
                b_max=batch["b_max"],
                min_selected_for_stop=batch["min_selected_for_stop"],
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
