from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eeg_channel_game.eeg.fold_sampler import FoldSampler
from eeg_channel_game.eeg.io import load_subject_data
from eeg_channel_game.eeg.variant import variant_from_cfg
from eeg_channel_game.eval.fbcsp import fit_fbcsp_ovr_filters, transform_fbcsp_features
from eeg_channel_game.eval.evaluator_l0 import L0Evaluator
from eeg_channel_game.eval.evaluator_normalize import DeltaFull22Evaluator
from eeg_channel_game.eval.evaluator_l1_fbcsp import L1FBCSPEvaluator
from eeg_channel_game.eval.evaluator_l2_deep import evaluate_l2_deep_train_eval
from eeg_channel_game.eval.riemann import riemann_ts_lr_channel_scores
from eeg_channel_game.eval.metrics import accuracy, cohen_kappa
from eeg_channel_game.game.env import EEGChannelGame
from eeg_channel_game.game.state_builder import StateBuilder
from eeg_channel_game.mcts.mcts import MCTS
from eeg_channel_game.model.policy_value_net import PolicyValueNet
from eeg_channel_game.utils.bitmask import mask_to_key, key_to_mask
from eeg_channel_game.utils.config import load_config, make_run_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare channel-selection baselines at fixed K (0train -> 1test, FBCSP train->eval + L2)"
    )
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--override",
        action="extend",
        nargs="+",
        default=[],
        help="YAML override(s) like project.out_dir=runs/x (repeatable)",
    )
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--k", type=str, default="8,10", help="Comma-separated K list, e.g. 4,6,8,10")
    p.add_argument("--split-id", type=int, default=0, help="Fold split id (for SFS/L1 evaluator)")
    p.add_argument("--checkpoint", type=str, default=None, help="Our method checkpoint (.pt). If set, runs MCTS search.")
    p.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional output subdir tag. Writes to runs/<out>/compare_k/<tag>/ to avoid overwriting.",
    )

    p.add_argument("--random-n", type=int, default=200, help="Number of random subsets to sample (pick best by L1)")
    p.add_argument("--l1-cv-folds", type=int, default=3)
    p.add_argument("--l1-robust-mode", type=str, default="mean_std")
    p.add_argument("--l1-robust-beta", type=float, default=0.5)

    p.add_argument("--ours-restarts", type=int, default=4, help="Best-of-N searches for ours (keep best by 0train reward)")
    p.add_argument("--ours-stochastic", action="store_true", help="Stochastic search for ours (root noise + sampling)")
    p.add_argument("--ours-tau", type=float, default=0.8, help="Sampling temperature when --ours-stochastic")
    p.add_argument("--ours-n-sim", type=int, default=None, help="Override MCTS simulations for ours (default: config)")

    p.add_argument("--l2-model", type=str, default="eegnetv4")
    p.add_argument("--l2-device", type=str, default=None)
    p.add_argument("--l2-seeds", type=str, default="0,1,2")
    p.add_argument("--l2-epochs", type=int, default=30)
    p.add_argument("--l2-batch-size", type=int, default=64)
    p.add_argument("--l2-lr", type=float, default=1e-3)
    p.add_argument("--l2-weight-decay", type=float, default=1e-4)
    p.add_argument("--l2-patience", type=int, default=8)
    return p.parse_args()


def _topk_from_scores(scores: np.ndarray, k: int) -> list[int]:
    idx = np.argsort(scores)[::-1][: int(k)]
    return [int(i) for i in idx]


def _fisher_scores_from_fold(fold) -> np.ndarray:
    # fold.stats.fisher is [22, B]
    return fold.stats.fisher.mean(axis=1).astype(np.float32, copy=False)


def _mi_scores_bandpower(bp: np.ndarray, y: np.ndarray) -> np.ndarray:
    try:
        from sklearn.feature_selection import mutual_info_classif
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for mutual information baseline") from e

    # feature per channel = mean over bands (simple but stable)
    x = bp.reshape(bp.shape[0], -1)  # [N, 22*B]
    mi = mutual_info_classif(x, y, discrete_features=False, random_state=0)
    mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return mi.reshape(22, -1).mean(axis=1).astype(np.float32)


def _try_load_cov_fb(*, data_root: Path, variant: str, subject: int) -> tuple[np.ndarray, np.ndarray] | None:
    cov_dir = Path(data_root) / "cache" / str(variant) / f"subj{int(subject):02d}"
    try:
        cov_t = np.load(cov_dir / "sessionT_cov_fb.npz")["cov_fb"].astype(np.float32, copy=False)
        cov_e = np.load(cov_dir / "sessionE_cov_fb.npz")["cov_fb"].astype(np.float32, copy=False)
        return cov_t, cov_e
    except FileNotFoundError:
        return None


def _eval_fbcsp_train_eval(
    *,
    cov_train: np.ndarray,
    y_train: np.ndarray,
    cov_eval: np.ndarray,
    y_eval: np.ndarray,
    sel_idx: list[int],
    m: int = 2,
    eps: float = 1e-6,
) -> dict[str, float]:
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for FBCSP evaluation") from e

    sel = np.array(sel_idx, dtype=np.int64)
    tr_idx = np.arange(y_train.shape[0], dtype=np.int64)
    ev_idx = np.arange(y_eval.shape[0], dtype=np.int64)
    filters = fit_fbcsp_ovr_filters(cov_train, y_train, tr_idx, sel, m=m, eps=eps)
    x_tr = transform_fbcsp_features(cov_train, tr_idx, sel, filters, m=m, eps=eps)
    x_ev = transform_fbcsp_features(cov_eval, ev_idx, sel, filters, m=m, eps=eps)

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )
    clf.fit(x_tr, y_train)
    y_pred = clf.predict(x_ev)
    return {"acc": float(accuracy(y_eval, y_pred)), "kappa": float(cohen_kappa(y_eval, y_pred))}


def _sfs_by_l1(
    *,
    k: int,
    fold,
    evaluator: L1FBCSPEvaluator,
) -> list[int]:
    fisher = _fisher_scores_from_fold(fold)
    selected = [int(np.argmax(fisher))]
    while len(selected) < int(k):
        best = (-1e9, None)
        for ch in range(22):
            if ch in selected:
                continue
            cand = selected + [ch]
            mask = np.zeros((22,), dtype=np.int8)
            mask[cand] = 1
            key = mask_to_key(mask)
            r, _ = evaluator.evaluate(key, fold)
            if r > best[0]:
                best = (float(r), int(ch))
        assert best[1] is not None
        selected.append(int(best[1]))
    return selected


def _random_best_by_l1(
    *,
    k: int,
    n: int,
    fold,
    evaluator: L1FBCSPEvaluator,
    seed: int,
) -> tuple[list[int], float]:
    rng = np.random.default_rng(seed)
    best = (-1e9, None)
    for _ in range(int(n)):
        cand = rng.choice(22, size=int(k), replace=False).tolist()
        mask = np.zeros((22,), dtype=np.int8)
        mask[cand] = 1
        key = mask_to_key(mask)
        r, _ = evaluator.evaluate(key, fold)
        if r > best[0]:
            best = (float(r), cand)
    return list(best[1] or []), float(best[0])


def _ours_from_checkpoint(checkpoint: str, k: int) -> list[int]:
    ckpt = torch.load(Path(checkpoint), map_location="cpu")
    key = int(ckpt.get("best_key") or 0)
    mask = key_to_mask(key, n_ch=22)
    sel = [i for i in range(22) if int(mask[i]) == 1]
    if len(sel) == int(k):
        return sel
    return sel  # caller will adjust


def _adjust_to_k(sel: list[int], k: int, fisher_scores: np.ndarray) -> list[int]:
    sel = list(dict.fromkeys([int(s) for s in sel]))  # unique keep order
    if len(sel) > int(k):
        # keep the top-k by fisher among selected
        s = np.array(sel, dtype=np.int64)
        keep = s[np.argsort(fisher_scores[s])[::-1][: int(k)]]
        return [int(i) for i in keep]
    if len(sel) < int(k):
        # pad with best fisher channels
        for ch in _topk_from_scores(fisher_scores, 22):
            if ch not in sel:
                sel.append(int(ch))
            if len(sel) == int(k):
                break
    return sel


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


def _make_search_evaluator(*, cfg: dict, variant: str, device: str) -> object:
    """
    Build the phase-B evaluator for search (0train only). This must NOT touch eval-session labels.
    """
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
    elif eval_name == "l1_deep_masked":
        from eeg_channel_game.eval.evaluator_l1_deep_masked import L1DeepMaskedEvaluator, L1DeepMaskedTrainConfig

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
        seeds = tuple(int(s) for s in str(l1_cfg.get("seeds", "0")).split(",") if str(s).strip())
        evaluator = L1DeepMaskedEvaluator(
            lambda_cost=float(cfg["reward"]["lambda_cost"]),
            artifact_gamma=float(cfg["reward"].get("artifact_gamma", 0.0)),
            robust_mode=str(l1_cfg.get("robust_mode", "mean_std")),
            robust_beta=float(l1_cfg.get("robust_beta", 0.5)),
            seeds=seeds or (0,),
            device=str(l1_cfg.get("device", device)),
            cfg=train_cfg,
        )
    else:
        raise ValueError(f"Unknown evaluator {eval_name}")

    normalize = cfg.get("reward", {}).get("normalize", False)
    if str(normalize).lower() in {"1", "true", "yes", "delta_full22", "delta"}:
        evaluator = DeltaFull22Evaluator(evaluator)  # constant shift; does not change ordering within a fold
    return evaluator


def _ours_search_fixed_k(
    *,
    cfg: dict,
    fold,
    net: PolicyValueNet,
    device: str,
    k: int,
    evaluator: object,
    restarts: int,
    stochastic: bool,
    tau: float,
    n_sim: int | None,
) -> list[int]:
    # Force exactly K channels by setting b_max=K and min_selected_for_stop=K.
    sb = StateBuilder(
        ch_names=fold.subject_data.ch_names,
        d_in=int(cfg["net"]["d_in"]),
        b_max=int(k),
        min_selected_for_stop=int(k),
    )
    mcts = MCTS(
        net=net,
        state_builder=sb,
        evaluator=evaluator,  # type: ignore[arg-type]
        infer_batch_size=int(cfg.get("mcts", {}).get("infer_batch_size", 1) or 1),
        n_sim=int(cfg["mcts"]["n_sim"]) if n_sim is None else int(n_sim),
        c_puct=float(cfg["mcts"]["c_puct"]),
        dirichlet_alpha=float(cfg["mcts"]["dirichlet_alpha"]),
        dirichlet_eps=float(cfg["mcts"]["dirichlet_eps"]),
        device=device,
    )
    env = EEGChannelGame(fold=fold, state_builder=sb, evaluator=evaluator, b_max=int(k), min_selected_for_stop=int(k))  # type: ignore[arg-type]

    base_seed = int(cfg["project"]["seed"]) + 10_000 * int(fold.subject) + 100 * int(fold.split_id) + int(k)
    best = (-1e18, None)
    for ri in range(int(restarts)):
        rng = np.random.default_rng(base_seed + int(ri))
        _ = env.reset()
        mcts.reset()

        done = False
        r = 0.0
        while not done:
            add_noise = bool(stochastic) and int(env.key) == 0
            pi = mcts.run(
                root_key=env.key,
                fold=fold,
                add_root_noise=add_noise,
                b_max=int(k),
                min_selected_for_stop=int(k),
                rng=rng,
            )
            if stochastic:
                a = _sample_from_pi(pi, tau=float(tau), rng=rng)
            else:
                a = int(np.argmax(pi))
            _, r, done, _info = env.step(a)

        key = int(env.key)
        if float(r) > best[0]:
            best = (float(r), key)

    key = best[1]
    assert key is not None
    sel_mask = np.array([(int(key) >> i) & 1 for i in range(22)], dtype=np.int8)
    sel_idx = np.where(sel_mask == 1)[0].tolist()
    if len(sel_idx) != int(k):
        raise RuntimeError(f"Expected exactly K={k} channels, got {len(sel_idx)}")
    return [int(i) for i in sel_idx]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    variant = variant_from_cfg(cfg)
    out_dir = Path(cfg["project"]["out_dir"])
    paths = make_run_paths(out_dir)
    comp_dir = paths.out_dir / "compare_k"
    if args.tag:
        comp_dir = comp_dir / str(args.tag)
    comp_dir.mkdir(parents=True, exist_ok=True)

    subject = int(args.subject)
    ks = [int(s.strip()) for s in args.k.split(",") if s.strip()]
    l2_device = str(args.l2_device) if args.l2_device else str(cfg["project"].get("device", "cuda"))
    l2_seeds = [int(s.strip()) for s in args.l2_seeds.split(",") if s.strip()]

    # L1 evaluator (used for SFS/random selection only; L2 is final scoring)
    sampler = FoldSampler(subjects=[subject], n_splits=5, seed=int(cfg["project"]["seed"]), variant=variant, include_eval=False)
    fold = sampler.get_fold(subject, int(args.split_id))
    l1_cfg = cfg.get("evaluator", {}).get("l1_fbcsp", {})
    evaluator_l1 = L1FBCSPEvaluator(
        lambda_cost=float(cfg["reward"]["lambda_cost"]),
        artifact_gamma=float(cfg["reward"].get("artifact_gamma", 0.0)),
        mode=str(l1_cfg.get("mode", "mr_fbcsp")),
        mask_eps=float(l1_cfg.get("mask_eps", 0.0)),
        csp_shrinkage=float(l1_cfg.get("csp_shrinkage", 0.1)),
        csp_ridge=float(l1_cfg.get("csp_ridge", 1e-3)),
        mask_penalty=float(l1_cfg.get("mask_penalty", 0.1)),
        cv_folds=int(args.l1_cv_folds),
        robust_mode=str(args.l1_robust_mode),
        robust_beta=float(args.l1_robust_beta),
        variant=variant,
    )

    # Optional: load our policy/value net for MCTS search
    net = None
    search_device = str(cfg["project"].get("device", "cuda"))
    if not torch.cuda.is_available() and str(search_device).startswith("cuda"):
        search_device = "cpu"
    evaluator_search = None
    if args.checkpoint:
        ckpt = torch.load(Path(args.checkpoint), map_location="cpu")
        ckpt_cfg = ckpt.get("cfg", {}) or {}
        ckpt_net = ckpt_cfg.get("net", {}) if isinstance(ckpt_cfg, dict) else {}
        cfg_net = cfg.get("net", {}) or {}
        film_cfg = cfg_net.get("film", None)
        if not isinstance(film_cfg, dict):
            film_cfg = None
        if film_cfg is None and isinstance(ckpt_net, dict):
            film_cfg = ckpt_net.get("film", None)
            if not isinstance(film_cfg, dict):
                film_cfg = None
        film_enabled = bool(film_cfg.get("enabled", False)) if film_cfg is not None else False
        film_hidden_raw = film_cfg.get("hidden", None) if film_cfg is not None else None
        film_hidden = int(film_hidden_raw) if film_hidden_raw is not None else None
        net = PolicyValueNet(
            d_in=int(cfg_net.get("d_in", ckpt_net.get("d_in", 64))),
            d_model=int(cfg_net.get("d_model", ckpt_net.get("d_model", 128))),
            n_layers=int(cfg_net.get("n_layers", ckpt_net.get("n_layers", 4))),
            n_heads=int(cfg_net.get("n_heads", ckpt_net.get("n_heads", 4))),
            policy_mode=str(cfg_net.get("policy_mode", ckpt_net.get("policy_mode", "cls"))),
            think_steps=int(cfg_net.get("think_steps", ckpt_net.get("think_steps", 1)) or 1),
            n_actions=23,
            film_enabled=bool(film_enabled),
            film_hidden=film_hidden,
        ).to(torch.device(search_device))
        missing, unexpected = net.load_state_dict(ckpt.get("model", {}), strict=False)
        if missing or unexpected:
            print(f"[compare] WARNING: load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
        net.eval()
        evaluator_search = _make_search_evaluator(cfg=cfg, variant=variant, device=search_device)
        print(f"[compare] ours checkpoint={args.checkpoint}")

    # L2 scoring uses eval session labels
    sd = load_subject_data(subject, variant=variant, include_eval=True)
    if sd.y_eval is None:
        raise RuntimeError("Missing eval labels; did you run run_prepare_data?")

    cov_pair = _try_load_cov_fb(data_root=sampler.data_root, variant=variant, subject=subject)
    if cov_pair is None:
        print("[compare] FBCSP train->eval skipped: missing cov_fb cache (rerun run_prepare_data without --no-cov).")
    else:
        cov_t, cov_e = cov_pair
        full = _eval_fbcsp_train_eval(
            cov_train=cov_t,
            y_train=sd.y_train,
            cov_eval=cov_e,
            y_eval=sd.y_eval,
            sel_idx=list(range(22)),
        )
        print(f"[compare] subj={subject:02d} FBCSP train->eval full-22: kappa/acc={full['kappa']:.4f}/{full['acc']:.4f}")

    fisher = _fisher_scores_from_fold(fold)
    mi = _mi_scores_bandpower(sd.bp_train, sd.y_train)
    riemann_scores = riemann_ts_lr_channel_scores(
        fold.subject_data.X_train[fold.split.train_idx],
        fold.subject_data.y_train[fold.split.train_idx],
        cov_estimator="oas",
        ts_metric="riemann",
        c=1.0,
        max_iter=2000,
        seed=int(cfg["project"]["seed"]),
    )

    rows = []
    for k in ks:
        if k < 2 or k > 22:
            raise ValueError("K must be in [2,22]")

        subsets: dict[str, list[int]] = {}
        subsets["fisher_topk"] = _topk_from_scores(fisher, k)
        subsets["mi_topk"] = _topk_from_scores(mi, k)
        subsets["riemann_ts_lr_topk"] = _topk_from_scores(riemann_scores, k)
        subsets["sfs_l1"] = _sfs_by_l1(k=k, fold=fold, evaluator=evaluator_l1)
        rnd_sel, rnd_score = _random_best_by_l1(k=k, n=int(args.random_n), fold=fold, evaluator=evaluator_l1, seed=123)
        subsets["random_best_l1"] = rnd_sel

        if net is not None and evaluator_search is not None:
            ours_sel = _ours_search_fixed_k(
                cfg=cfg,
                fold=fold,
                net=net,
                device=str(search_device),
                k=int(k),
                evaluator=evaluator_search,
                restarts=int(args.ours_restarts),
                stochastic=bool(args.ours_stochastic),
                tau=float(args.ours_tau),
                n_sim=int(args.ours_n_sim) if args.ours_n_sim is not None else None,
            )
            subsets["ours"] = ours_sel

        for name, sel in subsets.items():
            fbcsp_kappa = None
            fbcsp_acc = None
            if cov_pair is not None:
                cov_t, cov_e = cov_pair
                fbcsp_res = _eval_fbcsp_train_eval(
                    cov_train=cov_t,
                    y_train=sd.y_train,
                    cov_eval=cov_e,
                    y_eval=sd.y_eval,
                    sel_idx=sel,
                )
                fbcsp_kappa = fbcsp_res["kappa"]
                fbcsp_acc = fbcsp_res["acc"]

            res = evaluate_l2_deep_train_eval(
                subject_data=sd,
                sel_idx=sel,
                train_idx=fold.split.train_idx,
                val_idx=fold.split.val_idx,
                model_name=str(args.l2_model),
                seeds=l2_seeds,
                device=l2_device,
                epochs=int(args.l2_epochs),
                batch_size=int(args.l2_batch_size),
                lr=float(args.l2_lr),
                weight_decay=float(args.l2_weight_decay),
                patience=int(args.l2_patience),
            )
            rows.append(
                {
                    "subject": subject,
                    "variant": variant,
                    "k": int(k),
                    "method": name,
                    "sel": sel,
                    "l2_model": res["model"],
                    "kappa_mean": res["kappa_mean"],
                    "kappa_q20": res["kappa_q20"],
                    "acc_mean": res["acc_mean"],
                    "train_time_s_sum": res["train_time_s_sum"],
                    "fbcsp_kappa": fbcsp_kappa,
                    "fbcsp_acc": fbcsp_acc,
                    "random_best_l1_score": float(rnd_score) if name == "random_best_l1" else None,
                }
            )
            if fbcsp_kappa is None:
                print(
                    f"[compare] subj={subject:02d} K={k:02d} {name}: "
                    f"L2 kappa(mean/q20)={res['kappa_mean']:.4f}/{res['kappa_q20']:.4f}"
                )
            else:
                print(
                    f"[compare] subj={subject:02d} K={k:02d} {name}: "
                    f"L2 kappa(mean/q20)={res['kappa_mean']:.4f}/{res['kappa_q20']:.4f} | "
                    f"FBCSP kappa/acc={fbcsp_kappa:.4f}/{fbcsp_acc:.4f}"
                )

    out_path = comp_dir / f"subj{subject:02d}_compare_k.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        out_csv = comp_dir / f"subj{subject:02d}_compare_k.csv"
        df.to_csv(out_csv, index=False)
        print(f"[compare] saved: {out_csv}")
    except Exception:
        pass
    print(f"[compare] saved: {out_path}")


if __name__ == "__main__":
    main()
