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
from eeg_channel_game.model.uniform_net import UniformPolicyValueNet
from eeg_channel_game.utils.bitmask import mask_to_key
from eeg_channel_game.utils.config import load_config, make_run_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fixed-K Pareto curve: evaluate channel-selection methods across K on 0train->1test protocol."
    )
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--override",
        action="extend",
        nargs="+",
        default=[],
        help="YAML override(s) like project.out_dir=runs/x (repeatable)",
    )
    p.add_argument("--checkpoint", type=str, default=None, help="Our method checkpoint (.pt). Default: latest in out_dir.")
    p.add_argument("--subjects", type=str, default=None, help="Comma-separated list (default: from config)")
    p.add_argument("--k", type=str, default="4,6,8,10,12", help="Comma-separated K list, e.g. 4,6,8,10,12")
    p.add_argument("--methods", type=str, default="ours,fisher,mi,full22", help="Comma-separated methods")
    p.add_argument("--random-n", type=int, default=200, help="Random subsets for random_best_l1 (if enabled)")
    p.add_argument("--ours-restarts", type=int, default=1, help="Best-of-N searches for ours (selection by 0train reward)")
    p.add_argument("--ours-stochastic", action="store_true", help="Stochastic search for ours (root noise + sampling)")
    p.add_argument("--ours-tau", type=float, default=0.8, help="Sampling temperature when --ours-stochastic")
    p.add_argument("--uct-restarts", type=int, default=1, help="Best-of-N searches for uct (selection by 0train reward)")
    p.add_argument("--uct-stochastic", action="store_true", help="Stochastic search for uct (root noise + sampling)")
    p.add_argument("--uct-tau", type=float, default=0.8, help="Sampling temperature when --uct-stochastic")

    p.add_argument("--ga-restarts", type=int, default=1, help="GA restarts (keep best by L1 reward)")
    p.add_argument("--ga-pop", type=int, default=64, help="GA population size")
    p.add_argument("--ga-gens", type=int, default=50, help="GA generations")
    p.add_argument("--ga-elite", type=int, default=4, help="GA elitism (top-N kept each generation)")
    p.add_argument("--ga-cx", type=float, default=0.6, help="GA crossover probability")
    p.add_argument("--ga-mut", type=float, default=0.2, help="GA mutation probability")
    p.add_argument("--ga-seed", type=int, default=123, help="GA base seed")

    p.add_argument("--lr-c", type=float, default=1.0, help="LogReg C for lr_weight baseline")
    p.add_argument("--lr-max-iter", type=int, default=2000, help="LogReg max_iter for lr_weight baseline")
    p.add_argument("--lr-seed", type=int, default=0, help="LogReg random_state for lr_weight baseline")
    p.add_argument("--device", type=str, default=None, help="cuda|cpu (default: from config)")
    p.add_argument("--plot", action="store_true", help="Save a matplotlib curve plot (if available)")
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


def _latest_checkpoint(ckpt_dir: Path) -> Path:
    pts = sorted(ckpt_dir.glob("iter_*.pt"))
    if not pts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    return pts[-1]


def _topk(scores: np.ndarray, k: int) -> list[int]:
    idx = np.argsort(scores)[::-1][: int(k)]
    return [int(i) for i in idx]


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


def _mi_scores(bp: np.ndarray, y: np.ndarray) -> np.ndarray:
    try:
        from sklearn.feature_selection import mutual_info_classif
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for MI baseline") from e

    x = bp.reshape(bp.shape[0], -1)
    mi = mutual_info_classif(x, y, discrete_features=False, random_state=0)
    mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return mi.reshape(22, -1).mean(axis=1).astype(np.float32)


def _lr_weight_scores(*, bp: np.ndarray, y: np.ndarray, c: float = 1.0, max_iter: int = 2000, seed: int = 0) -> np.ndarray:
    """
    Embedded baseline: train multinomial logistic regression on bandpower features,
    then rank channels by the norm of their weights aggregated over bands/classes.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for lr_weight baseline") from e

    x = bp.reshape(bp.shape[0], -1).astype(np.float32, copy=False)  # [N, 22*B]
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(
            penalty="l2",
            C=float(c),
            max_iter=int(max_iter),
            solver="lbfgs",
            random_state=int(seed),
        ),
    )
    clf.fit(x, y)
    lr: LogisticRegression = clf[-1]
    coef = np.asarray(lr.coef_, dtype=np.float32)  # [n_classes, 22*B]
    if coef.ndim != 2 or coef.shape[1] != x.shape[1]:
        raise RuntimeError(f"Unexpected coef shape: {coef.shape}")
    n_bands = int(bp.shape[2])
    coef = coef.reshape(coef.shape[0], 22, n_bands)
    scores = np.sqrt(np.sum(coef**2, axis=(0, 2))).astype(np.float32)  # [22]
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def _ga_by_l1(
    *,
    k: int,
    fold,
    evaluator: L1FBCSPEvaluator,
    seed: int,
    pop_size: int = 64,
    n_gens: int = 50,
    elite: int = 4,
    cx_prob: float = 0.6,
    mut_prob: float = 0.2,
) -> tuple[list[int], float]:
    """
    Wrapper baseline: simple genetic algorithm over fixed-size channel subsets.
    Fitness = L1 reward (robust CV kappa - cost - artifact penalty) on 0train.
    """
    k = int(k)
    pop_size = int(pop_size)
    n_gens = int(n_gens)
    elite = int(min(max(0, elite), pop_size))
    cx_prob = float(cx_prob)
    mut_prob = float(mut_prob)

    rng = np.random.default_rng(int(seed))

    def to_key(sel: list[int]) -> int:
        mask = np.zeros((22,), dtype=np.int8)
        mask[sel] = 1
        return int(mask_to_key(mask))

    def sample_individual() -> list[int]:
        sel = rng.choice(22, size=k, replace=False).tolist()
        sel.sort()
        return [int(x) for x in sel]

    def crossover(a: list[int], b: list[int]) -> list[int]:
        pool = list(set(a) | set(b))
        rng.shuffle(pool)
        child = pool[:k]
        if len(child) < k:
            for ch in rng.permutation(22).tolist():
                if ch not in child:
                    child.append(int(ch))
                if len(child) == k:
                    break
        child = [int(x) for x in child]
        child.sort()
        return child

    def mutate(sel: list[int]) -> list[int]:
        if k <= 0:
            return sel
        out = list(sel)
        out_ch = int(rng.choice(out))
        out = [c for c in out if c != out_ch]
        # sample new channel not in set
        cur = set(out)
        cand = [c for c in range(22) if c not in cur]
        in_ch = int(rng.choice(cand))
        out.append(in_ch)
        out = [int(x) for x in out]
        out.sort()
        return out

    # init population
    population = [sample_individual() for _ in range(pop_size)]
    best = (-1e18, None)

    for _gen in range(n_gens):
        scored: list[tuple[float, list[int]]] = []
        for ind in population:
            r, _ = evaluator.evaluate(to_key(ind), fold)
            scored.append((float(r), ind))
        scored.sort(key=lambda t: t[0], reverse=True)

        if scored and scored[0][0] > best[0]:
            best = (float(scored[0][0]), list(scored[0][1]))

        next_pop = [list(ind) for _, ind in scored[:elite]]

        def tournament() -> list[int]:
            tsize = 3
            idx = rng.integers(0, len(scored), size=tsize)
            cand = [scored[int(i)] for i in idx]
            cand.sort(key=lambda t: t[0], reverse=True)
            return list(cand[0][1])

        while len(next_pop) < pop_size:
            p1 = tournament()
            child = list(p1)
            if rng.random() < cx_prob:
                p2 = tournament()
                child = crossover(p1, p2)
            if rng.random() < mut_prob:
                child = mutate(child)
            next_pop.append(child)

        population = next_pop

    assert best[1] is not None
    return list(best[1]), float(best[0])


def _sfs_by_l1(*, k: int, fold, evaluator: L1FBCSPEvaluator, fisher_scores: np.ndarray) -> list[int]:
    selected = [int(np.argmax(fisher_scores))]
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
    *, k: int, n: int, fold, evaluator: L1FBCSPEvaluator, seed: int
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


def _make_fold_full_train(*, subject: int, variant: str) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    # Load train-only subject data (no eval label usage for selection).
    sd = load_subject_data(subject, variant=variant, include_eval=False)
    idx = np.arange(sd.y_train.shape[0], dtype=np.int64)
    split = Split(train_idx=idx, val_idx=np.array([], dtype=np.int64))

    stats = compute_fold_stats(
        bp=sd.bp_train[idx],
        q=sd.q_train[idx],
        y=sd.y_train[idx],
        artifact_corr_eog=sd.artifact_corr_eog,
        resid_ratio=sd.resid_ratio,
    )

    from eeg_channel_game.eeg.fold_sampler import FoldData

    fold = FoldData(subject=int(subject), split_id=0, stats=stats, split=split, subject_data=sd)
    fisher_scores = stats.fisher.mean(axis=1).astype(np.float32, copy=False)
    mi_scores = _mi_scores(sd.bp_train, sd.y_train)
    return sd, fold, fisher_scores, mi_scores


def _ours_search_fixed_k(
    *,
    cfg: dict[str, Any],
    variant: str,
    fold,
    net: torch.nn.Module,
    device: str,
    k: int,
    restarts: int = 1,
    stochastic: bool = False,
    tau: float = 0.8,
    n_sim: int | None = None,
) -> tuple[int, list[int], dict[str, Any]]:
    # evaluator used only on 0train (no eval leakage)
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

    # IMPORTANT: force exactly K channels by setting b_max=K.
    sb = StateBuilder(
        ch_names=fold.subject_data.ch_names,
        d_in=int(cfg["net"]["d_in"]),
        b_max=int(k),
        min_selected_for_stop=int(k),
    )
    mcts = MCTS(
        net=net,
        state_builder=sb,
        evaluator=evaluator,
        n_sim=int(cfg["mcts"]["n_sim"]) if n_sim is None else int(n_sim),
        c_puct=float(cfg["mcts"]["c_puct"]),
        dirichlet_alpha=float(cfg["mcts"]["dirichlet_alpha"]),
        dirichlet_eps=float(cfg["mcts"]["dirichlet_eps"]),
        device=device,
    )
    env = EEGChannelGame(fold=fold, state_builder=sb, evaluator=evaluator, b_max=int(k))
    base_seed = int(cfg["project"]["seed"]) + 10_000 * int(fold.subject) + 100 * int(k)
    best = (-1e18, None, None)  # (reward, key, info)
    candidates = []
    for ri in range(int(restarts)):
        rng = np.random.default_rng(base_seed + int(ri))
        _ = env.reset()
        mcts.reset()

        done = False
        info: dict[str, Any] = {}
        while not done:
            add_noise = bool(stochastic) and int(env.key) == 0
            pi = mcts.run(
                root_key=env.key,
                fold=fold,
                add_root_noise=add_noise,
                b_max=int(k),
                min_selected_for_stop=int(k),
            )
            if stochastic:
                a = _sample_from_pi(pi, tau=float(tau), rng=rng)
            else:
                a = int(np.argmax(pi))
            _, r, done, info = env.step(a)

        key = int(env.key)
        candidates.append({"restart": int(ri), "key": key, "reward": float(r)})
        if float(r) > best[0]:
            best = (float(r), key, info)

    best_reward, key, info = best
    assert key is not None
    sel_mask = np.array([(int(key) >> i) & 1 for i in range(22)], dtype=np.int8)
    sel_idx = np.where(sel_mask == 1)[0].tolist()
    if len(sel_idx) != int(k):
        raise RuntimeError(f"Expected exactly K={k} channels, got {len(sel_idx)}")
    return int(key), sel_idx, {"reward": float(best_reward), "restarts": int(restarts), "stochastic": bool(stochastic), "tau": float(tau), "candidates": candidates, **(info or {})}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    variant = variant_from_cfg(cfg)
    out_dir = Path(cfg["project"]["out_dir"])
    paths = make_run_paths(out_dir)

    pareto_dir = paths.out_dir / "pareto"
    pareto_dir.mkdir(parents=True, exist_ok=True)

    if args.subjects:
        subjects = [int(s.strip()) for s in str(args.subjects).split(",") if s.strip()]
    else:
        subjects = [int(s) for s in cfg["data"]["subjects"]]

    ks = [int(s.strip()) for s in str(args.k).split(",") if s.strip()]
    if not ks:
        raise ValueError("Empty K list")

    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    want_ours = "ours" in methods
    want_uct = "uct" in methods

    device = str(args.device) if args.device else str(cfg["project"].get("device", "cuda"))

    net = None
    ckpt_path: Path | None = None
    if want_ours:
        ckpt_dir = paths.ckpt_dir
        ckpt_path = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(ckpt_dir)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        net = PolicyValueNet(
            d_in=int(cfg["net"]["d_in"]),
            d_model=int(cfg["net"]["d_model"]),
            n_layers=int(cfg["net"]["n_layers"]),
            n_heads=int(cfg["net"]["n_heads"]),
            n_actions=23,
        ).to(torch.device(device))
        missing, unexpected = net.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            print(f"[pareto] WARNING: load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
        net.eval()
        print(f"[pareto] checkpoint={ckpt_path}")

    uct_net = None
    if want_uct:
        uct_net = UniformPolicyValueNet(n_tokens=24, n_actions=23).to(torch.device(device))
        uct_net.eval()

    rows = []
    for subj in subjects:
        sd_train, fold, fisher_scores, mi_scores = _make_fold_full_train(subject=int(subj), variant=variant)
        lr_scores = _lr_weight_scores(
            bp=sd_train.bp_train,
            y=sd_train.y_train,
            c=float(args.lr_c),
            max_iter=int(args.lr_max_iter),
            seed=int(args.lr_seed),
        )

        # L1 evaluator used by SFS/random baselines (selection only, 0train).
        l1_cfg = cfg.get("evaluator", {}).get("l1_fbcsp", {})
        evaluator_l1 = L1FBCSPEvaluator(
            lambda_cost=float(cfg["reward"]["lambda_cost"]),
            artifact_gamma=float(cfg["reward"].get("artifact_gamma", 0.0)),
            cv_folds=int(l1_cfg.get("cv_folds", 3)),
            robust_mode=str(l1_cfg.get("robust_mode", "mean_std")),
            robust_beta=float(l1_cfg.get("robust_beta", 0.5)),
            variant=variant,
        )

        for k in ks:
            if k < 2 or k > 22:
                raise ValueError("K must be in [2,22]")

            # selection (train-only)
            subsets: dict[str, list[int]] = {}
            meta: dict[str, dict[str, Any]] = {}

            if "ours" in methods:
                assert net is not None
                key, sel, info = _ours_search_fixed_k(
                    cfg=cfg,
                    variant=variant,
                    fold=fold,
                    net=net,
                    device=device,
                    k=k,
                    restarts=int(args.ours_restarts),
                    stochastic=bool(args.ours_stochastic),
                    tau=float(args.ours_tau),
                )
                subsets["ours"] = sel
                meta["ours"] = {"key": int(key), **info}

            if "uct" in methods:
                assert uct_net is not None
                key, sel, info = _ours_search_fixed_k(
                    cfg=cfg,
                    variant=variant,
                    fold=fold,
                    net=uct_net,
                    device=device,
                    k=k,
                    restarts=int(args.uct_restarts),
                    stochastic=bool(args.uct_stochastic),
                    tau=float(args.uct_tau),
                )
                subsets["uct"] = sel
                meta["uct"] = {"key": int(key), **info}

            if "fisher" in methods:
                subsets["fisher"] = _topk(fisher_scores, k)

            if "mi" in methods:
                subsets["mi"] = _topk(mi_scores, k)

            if "lr_weight" in methods:
                subsets["lr_weight"] = _topk(lr_scores, k)

            if "sfs_l1" in methods:
                subsets["sfs_l1"] = _sfs_by_l1(k=k, fold=fold, evaluator=evaluator_l1, fisher_scores=fisher_scores)

            if "random_best_l1" in methods:
                sel, score = _random_best_by_l1(k=k, n=int(args.random_n), fold=fold, evaluator=evaluator_l1, seed=123)
                subsets["random_best_l1"] = sel
                meta["random_best_l1"] = {"l1_reward": float(score)}

            if "ga_l1" in methods:
                best = (-1e18, None)
                for ri in range(int(args.ga_restarts)):
                    sel, score = _ga_by_l1(
                        k=k,
                        fold=fold,
                        evaluator=evaluator_l1,
                        seed=int(args.ga_seed) + 10_000 * int(subj) + 100 * int(k) + int(ri),
                        pop_size=int(args.ga_pop),
                        n_gens=int(args.ga_gens),
                        elite=int(args.ga_elite),
                        cx_prob=float(args.ga_cx),
                        mut_prob=float(args.ga_mut),
                    )
                    if float(score) > best[0]:
                        best = (float(score), sel)
                assert best[1] is not None
                subsets["ga_l1"] = list(best[1])
                meta["ga_l1"] = {
                    "l1_reward": float(best[0]),
                    "ga_restarts": int(args.ga_restarts),
                    "ga_pop": int(args.ga_pop),
                    "ga_gens": int(args.ga_gens),
                    "ga_elite": int(args.ga_elite),
                    "ga_cx": float(args.ga_cx),
                    "ga_mut": float(args.ga_mut),
                }

            if "full22" in methods:
                subsets["full22"] = list(range(22))

            # evaluation (uses eval labels; OK for reporting)
            for mname, sel_idx in subsets.items():
                te = _eval_fbcsp_train_eval(variant=variant, subject=int(subj), sel_idx=sel_idx)
                rows.append(
                    {
                        "subject": int(subj),
                        "variant": str(variant),
                        "k": int(k),
                        "method": str(mname),
                        "n_ch": int(len(sel_idx)),
                        "sel_idx": json.dumps([int(i) for i in sel_idx]),
                        "fbcsp_kappa": float(te["kappa"]),
                        "fbcsp_acc": float(te["acc"]),
                        "meta": json.dumps(meta.get(mname, {})),
                    }
                )
                print(
                    f"[pareto] subj={int(subj):02d} K={int(k):02d} {mname}: "
                    f"kappa/acc={float(te['kappa']):.4f}/{float(te['acc']):.4f}"
                )

    # per-subject table
    by_subj_csv = pareto_dir / "pareto_by_subject.csv"
    with by_subj_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["subject", "variant", "k", "method", "n_ch", "fbcsp_kappa", "fbcsp_acc", "sel_idx", "meta"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"[pareto] saved: {by_subj_csv}")

    # summary table
    summary_rows = []
    methods_seen = sorted({r["method"] for r in rows})
    for mname in methods_seen:
        for k in sorted(set(ks)):
            vals = [r for r in rows if r["method"] == mname and int(r["k"]) == int(k)]
            if not vals:
                continue
            kappas = np.array([float(r["fbcsp_kappa"]) for r in vals], dtype=np.float32)
            accs = np.array([float(r["fbcsp_acc"]) for r in vals], dtype=np.float32)
            summary_rows.append(
                {
                    "method": mname,
                    "k": int(k),
                    "n": int(len(vals)),
                    "kappa_mean": float(kappas.mean()),
                    "kappa_std": float(kappas.std(ddof=0)),
                    "kappa_q20": float(np.quantile(kappas, 0.2)),
                    "acc_mean": float(accs.mean()),
                    "acc_std": float(accs.std(ddof=0)),
                    "acc_q20": float(np.quantile(accs, 0.2)),
                }
            )

    summary_csv = pareto_dir / "pareto_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["method", "k", "n", "kappa_mean", "kappa_std", "kappa_q20", "acc_mean", "acc_std", "acc_q20"],
        )
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[pareto] saved: {summary_csv}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[pareto] plot skipped: {e}")
        else:
            def _plot_metric(*, mean_key: str, std_key: str, ylabel: str, out_name: str) -> None:
                plt.figure(figsize=(7.5, 4.5))
                for mname in methods_seen:
                    xs = []
                    ys = []
                    yerr = []
                    for k in sorted(set(ks)):
                        row = next(
                            (r for r in summary_rows if r["method"] == mname and int(r["k"]) == int(k)),
                            None,
                        )
                        if row is None:
                            continue
                        xs.append(int(k))
                        ys.append(float(row[mean_key]))
                        yerr.append(float(row[std_key]))
                    if xs:
                        plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=mname)
                plt.xlabel("K (number of channels)")
                plt.ylabel(ylabel)
                plt.title("Fixed-K Pareto curve (mean±std over subjects)")
                plt.grid(True, alpha=0.3)
                plt.legend()
                fig_path = pareto_dir / out_name
                plt.tight_layout()
                plt.savefig(fig_path, dpi=200)
                plt.close()
                print(f"[pareto] saved: {fig_path}")

            _plot_metric(
                mean_key="kappa_mean",
                std_key="kappa_std",
                ylabel="FBCSP kappa (0train→1test)",
                out_name="pareto_kappa.png",
            )
            _plot_metric(
                mean_key="acc_mean",
                std_key="acc_std",
                ylabel="FBCSP accuracy (0train→1test)",
                out_name="pareto_acc.png",
            )


if __name__ == "__main__":
    main()
