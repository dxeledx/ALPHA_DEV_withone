from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eeg_channel_game.eeg.fold_stats import compute_fold_stats
from eeg_channel_game.eeg.io import load_subject_data
from eeg_channel_game.eeg.splits import Split
from eeg_channel_game.eeg.variant import variant_from_cfg
from eeg_channel_game.eval.fbcsp import fit_fbcsp_ovr_filters, transform_fbcsp_features
from eeg_channel_game.eval.evaluator_l0 import L0Evaluator
from eeg_channel_game.eval.evaluator_l1_fbcsp import L1FBCSPEvaluator
from eeg_channel_game.eval.evaluator_normalize import DeltaFull22Evaluator
from eeg_channel_game.eval.metrics import accuracy, cohen_kappa
from eeg_channel_game.game.env import EEGChannelGame
from eeg_channel_game.game.state_builder import StateBuilder
from eeg_channel_game.mcts.mcts import MCTS
from eeg_channel_game.model.policy_value_net import PolicyValueNet
from eeg_channel_game.utils.bitmask import key_to_mask
from eeg_channel_game.utils.config import load_config, make_run_paths
from eeg_channel_game.utils.visualization import plot_channel_mask_topomap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Use a trained policy/value net to search a channel subset for each subject (no eval-label leakage)."
    )
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--override",
        action="extend",
        nargs="+",
        default=[],
        help="YAML override(s) like project.out_dir=runs/x (repeatable)",
    )
    p.add_argument("--checkpoint", type=str, required=True, help="Path to a checkpoint (.pt) from run_train")
    p.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subject list (default: from config data.subjects)",
    )
    p.add_argument("--device", type=str, default=None, help="cuda|cpu (default: from config)")
    p.add_argument("--split-mode", type=str, default="full", help="full | split0..split4 (use 0train only)")
    p.add_argument("--restarts", type=int, default=1, help="Number of independent searches; keep best by 0train reward")
    p.add_argument("--stochastic", action="store_true", help="Stochastic search (root noise + action sampling)")
    p.add_argument("--tau", type=float, default=0.8, help="Sampling temperature when --stochastic")
    p.add_argument("--save-topomap", action="store_true", help="Save selected-channel topomap per subject")
    return p.parse_args()


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


def _eval_fbcsp_train_eval(
    *,
    variant: str,
    subject: int,
    sel_idx: list[int],
    m: int = 2,
    eps: float = 1e-6,
) -> dict[str, float]:
    sd = load_subject_data(subject, variant=variant, include_eval=True)
    if sd.y_eval is None:
        raise RuntimeError("Missing eval labels; did you run run_prepare_data?")

    data_root = Path("eeg_channel_game") / "data"
    cov_dir = data_root / "cache" / variant / f"subj{subject:02d}"
    cov_t = np.load(cov_dir / "sessionT_cov_fb.npz")["cov_fb"].astype(np.float32, copy=False)
    cov_e = np.load(cov_dir / "sessionE_cov_fb.npz")["cov_fb"].astype(np.float32, copy=False)

    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for FBCSP evaluation") from e

    sel = np.array(sel_idx, dtype=np.int64)
    tr_idx = np.arange(sd.y_train.shape[0], dtype=np.int64)
    ev_idx = np.arange(sd.y_eval.shape[0], dtype=np.int64)
    filters = fit_fbcsp_ovr_filters(cov_t, sd.y_train, tr_idx, sel, m=m, eps=eps)
    x_tr = transform_fbcsp_features(cov_t, tr_idx, sel, filters, m=m, eps=eps)
    x_ev = transform_fbcsp_features(cov_e, ev_idx, sel, filters, m=m, eps=eps)

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )
    clf.fit(x_tr, sd.y_train)
    y_pred = clf.predict(x_ev)
    return {"kappa": float(cohen_kappa(sd.y_eval, y_pred)), "acc": float(accuracy(sd.y_eval, y_pred))}


def _split_from_mode(y_train: np.ndarray, mode: str, seed: int) -> Split:
    mode = str(mode)
    if mode == "full":
        idx = np.arange(y_train.shape[0], dtype=np.int64)
        return Split(train_idx=idx, val_idx=np.array([], dtype=np.int64))

    if mode.startswith("split"):
        try:
            split_id = int(mode.replace("split", ""))
        except Exception as e:
            raise ValueError(f"Invalid split-mode: {mode}") from e
        if split_id < 0 or split_id > 4:
            raise ValueError("split-mode must be full or split0..split4")
        # mimic FoldSampler split strategy (StratifiedKFold on 0train)
        from eeg_channel_game.eeg.splits import make_stratified_splits

        splits = make_stratified_splits(y_train, n_splits=5, seed=seed)
        return splits[split_id]

    raise ValueError("split-mode must be full or split0..split4")


def _search_one_subject(
    *,
    cfg: dict[str, Any],
    variant: str,
    subject: int,
    net: PolicyValueNet,
    device: str,
    split_mode: str,
) -> dict[str, Any]:
    sd = load_subject_data(subject, variant=variant, include_eval=False)
    split = _split_from_mode(sd.y_train, split_mode, seed=int(cfg["project"]["seed"]))

    stats = compute_fold_stats(
        bp=sd.bp_train[split.train_idx],
        q=sd.q_train[split.train_idx],
        y=sd.y_train[split.train_idx],
        artifact_corr_eog=sd.artifact_corr_eog,
        resid_ratio=sd.resid_ratio,
    )
    from eeg_channel_game.eeg.fold_sampler import FoldData

    fold = FoldData(subject=int(subject), split_id=-1, stats=stats, split=split, subject_data=sd)

    state_builder = StateBuilder(
        ch_names=sd.ch_names,
        d_in=int(cfg["net"]["d_in"]),
        b_max=int(cfg["game"]["b_max"]),
        min_selected_for_stop=int(cfg["game"]["min_selected_for_stop"]),
    )

    # Search evaluator: use the same phase-B evaluator as training by default.
    train_cfg = cfg.get("train", {})
    switch_to_l1_iter = train_cfg.get("switch_to_l1_iter", None)
    if switch_to_l1_iter is None:
        eval_name = cfg.get("evaluator", {}).get("name", "l0")
    else:
        eval_name = cfg.get("evaluator", {}).get("phase_b", "l1_fbcsp")

    if eval_name == "l0":
        evaluator = L0Evaluator(
            lambda_cost=float(cfg["reward"]["lambda_cost"]),
            beta_redund=float(cfg["reward"]["beta_redund"]),
            artifact_gamma=float(cfg["reward"].get("artifact_gamma", 0.0)),
        )
    elif eval_name == "l1_fbcsp":
        l1_cfg = cfg.get("evaluator", {}).get("l1_fbcsp", {})
        evaluator = L1FBCSPEvaluator(
            lambda_cost=float(cfg["reward"]["lambda_cost"]),
            artifact_gamma=float(cfg["reward"].get("artifact_gamma", 0.0)),
            cv_folds=int(l1_cfg.get("cv_folds", 3)),
            robust_mode=str(l1_cfg.get("robust_mode", "mean_std")),
            robust_beta=float(l1_cfg.get("robust_beta", 0.5)),
            variant=variant,
        )
    else:
        raise ValueError(f"Unknown evaluator {eval_name}")

    normalize = cfg.get("reward", {}).get("normalize", False)
    if str(normalize).lower() in {"1", "true", "yes", "delta_full22", "delta"}:
        evaluator = DeltaFull22Evaluator(evaluator)

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

    restarts = int(cfg.get("search", {}).get("restarts", 1))
    stochastic = bool(cfg.get("search", {}).get("stochastic", False))
    tau = float(cfg.get("search", {}).get("tau", 0.8))

    env = EEGChannelGame(fold=fold, state_builder=state_builder, evaluator=evaluator, b_max=state_builder.b_max)
    base_seed = int(cfg["project"]["seed"]) + 10_000 * int(subject)
    best = (-1e18, None, None, None)  # (reward, key, info, actions)
    candidates = []
    for ri in range(int(restarts)):
        rng = np.random.default_rng(base_seed + int(ri))
        _ = env.reset()
        mcts.reset()

        done = False
        info: dict[str, Any] = {}
        actions = []
        while not done:
            add_noise = bool(stochastic) and int(env.key) == 0
            pi = mcts.run(root_key=env.key, fold=fold, add_root_noise=add_noise)
            if stochastic:
                a = _sample_from_pi(pi, tau=float(tau), rng=rng)
            else:
                a = int(np.argmax(pi))
            actions.append(int(a))
            _, r, done, info = env.step(a)

        key = int(env.key)
        candidates.append({"restart": int(ri), "key": key, "reward": float(r), "actions": actions})
        if float(r) > best[0]:
            best = (float(r), key, info, actions)

    best_reward, key, info, actions = best
    assert key is not None
    mask = key_to_mask(int(key), n_ch=22).astype(np.int8)
    sel_idx = np.where(mask == 1)[0].tolist()
    sel_names = [sd.ch_names[i] for i in sel_idx]

    return {
        "subject": int(subject),
        "variant": str(variant),
        "split_mode": str(split_mode),
        "b_max": int(state_builder.b_max),
        "min_selected_for_stop": int(state_builder.min_selected_for_stop),
        "key": int(key),
        "sel_idx": sel_idx,
        "sel_names": sel_names,
        "search_reward": float(best_reward),
        "search_info": info,
        "search_restarts": int(restarts),
        "search_stochastic": bool(stochastic),
        "search_tau": float(tau),
        "search_candidates": candidates,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    variant = variant_from_cfg(cfg)
    out_dir = Path(cfg["project"]["out_dir"])
    paths = make_run_paths(out_dir)

    subjects = cfg["data"]["subjects"]
    if args.subjects:
        subjects = [int(s.strip()) for s in str(args.subjects).split(",") if s.strip()]
    subjects = [int(s) for s in subjects]

    device = str(args.device) if args.device else str(cfg["project"].get("device", "cuda"))

    # Load net
    ckpt = torch.load(Path(args.checkpoint), map_location="cpu")
    net = PolicyValueNet(
        d_in=int(cfg["net"]["d_in"]),
        d_model=int(cfg["net"]["d_model"]),
        n_layers=int(cfg["net"]["n_layers"]),
        n_heads=int(cfg["net"]["n_heads"]),
        n_actions=23,
    ).to(torch.device(device))
    missing, unexpected = net.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"[search] WARNING: load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    net.eval()

    search_dir = paths.out_dir / "search"
    search_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for subj in subjects:
        # CLI overrides for search behavior (kept out of training config)
        cfg.setdefault("search", {})
        cfg["search"]["restarts"] = int(args.restarts)
        cfg["search"]["stochastic"] = bool(args.stochastic)
        cfg["search"]["tau"] = float(args.tau)

        res = _search_one_subject(
            cfg=cfg,
            variant=variant,
            subject=int(subj),
            net=net,
            device=device,
            split_mode=str(args.split_mode),
        )

        # Real protocol metric (0train -> 1test) using FBCSP(rLDA)
        te = _eval_fbcsp_train_eval(variant=variant, subject=int(subj), sel_idx=list(res["sel_idx"]))
        res["fbcsp_train_eval"] = te

        out_path = search_dir / f"subj{int(subj):02d}_search.json"
        out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")

        if args.save_topomap:
            fig_path = paths.fig_dir / f"subj{int(subj):02d}_search_topomap.png"
            sd_full = load_subject_data(int(subj), variant=variant, include_eval=False)
            plot_channel_mask_topomap(
                ch_names=sd_full.ch_names,
                mask=key_to_mask(int(res["key"]), n_ch=22).astype(np.float32),
                sfreq=sd_full.sfreq,
                title=f"Subject {int(subj):02d} | n={len(res['sel_idx'])}",
                save_path=fig_path,
                show_names=False,
            )

        summary_rows.append(
            {
                "subject": int(subj),
                "variant": str(variant),
                "split_mode": str(args.split_mode),
                "n_ch": int(len(res["sel_idx"])),
                "key": int(res["key"]),
                "search_reward": float(res["search_reward"]),
                "fbcsp_kappa": float(te["kappa"]),
                "fbcsp_acc": float(te["acc"]),
                "sel_idx": json.dumps(res["sel_idx"]),
            }
        )

        print(
            f"[search] subj={int(subj):02d} n={len(res['sel_idx'])} "
            f"reward={float(res['search_reward']):.4f} fbcsp_kappa/acc={float(te['kappa']):.4f}/{float(te['acc']):.4f}"
        )

    # Save summary CSV
    csv_path = search_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "subject",
                "variant",
                "split_mode",
                "n_ch",
                "key",
                "search_reward",
                "fbcsp_kappa",
                "fbcsp_acc",
                "sel_idx",
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[search] saved: {csv_path}")


if __name__ == "__main__":
    main()
