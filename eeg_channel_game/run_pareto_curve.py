from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from eeg_channel_game.eeg.fold_stats import compute_fold_stats
from eeg_channel_game.eeg.io import load_subject_data
from eeg_channel_game.eeg.splits import Split
from eeg_channel_game.eeg.variant import variant_from_cfg
from eeg_channel_game.eval.fbcsp import fit_fbcsp_ovr_filters, transform_fbcsp_features
from eeg_channel_game.eval.evaluator_l0 import L0Evaluator
from eeg_channel_game.eval.evaluator_l0_lr_weight import L0LrWeightEvaluator
from eeg_channel_game.eval.evaluator_l1_fbcsp import L1FBCSPEvaluator
from eeg_channel_game.eval.evaluator_l1_deep_masked import L1DeepMaskedEvaluator, L1DeepMaskedTrainConfig
from eeg_channel_game.eval.evaluator_domain_shift import DomainShiftPenaltyEvaluator
from eeg_channel_game.eval.evaluator_normalize import DeltaFull22Evaluator
from eeg_channel_game.eval.evaluator_normalize import AdvantageMaxBaselineEvaluator
from eeg_channel_game.eval.riemann import riemann_ts_lr_channel_scores
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
    bool_action = getattr(argparse, "BooleanOptionalAction", None)
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
    p.add_argument("--k", type=str, default=None, help="Comma-separated K list, e.g. 4,6,8,10,12 (default: from config)")
    p.add_argument("--methods", type=str, default=None, help="Comma-separated methods (default: from config)")
    p.add_argument("--random-n", type=int, default=None, help="Random subsets for random_best_l1 (default: from config)")
    p.add_argument("--ours-restarts", type=int, default=None, help="Best-of-N searches for ours (default: from config)")
    if bool_action is None:  # pragma: no cover
        p.add_argument(
            "--ours-stochastic",
            action="store_true",
            default=None,
            help="Stochastic search for ours (default: from config)",
        )
    else:
        p.add_argument("--ours-stochastic", action=bool_action, default=None, help="Stochastic search for ours (default: from config)")
    p.add_argument("--ours-tau", type=float, default=None, help="Sampling temperature when stochastic (default: from config)")
    p.add_argument("--uct-restarts", type=int, default=None, help="Best-of-N searches for uct (default: from config)")
    if bool_action is None:  # pragma: no cover
        p.add_argument(
            "--uct-stochastic",
            action="store_true",
            default=None,
            help="Stochastic search for uct (default: from config)",
        )
    else:
        p.add_argument("--uct-stochastic", action=bool_action, default=None, help="Stochastic search for uct (default: from config)")
    p.add_argument("--uct-tau", type=float, default=None, help="Sampling temperature when stochastic (default: from config)")

    p.add_argument("--ga-restarts", type=int, default=None, help="GA restarts (default: from config)")
    p.add_argument("--ga-pop", type=int, default=None, help="GA population size (default: from config)")
    p.add_argument("--ga-gens", type=int, default=None, help="GA generations (default: from config)")
    p.add_argument("--ga-elite", type=int, default=None, help="GA elitism (default: from config)")
    p.add_argument("--ga-cx", type=float, default=None, help="GA crossover probability (default: from config)")
    p.add_argument("--ga-mut", type=float, default=None, help="GA mutation probability (default: from config)")
    p.add_argument("--ga-seed", type=int, default=None, help="GA base seed (default: from config)")

    p.add_argument("--lr-c", type=float, default=None, help="LogReg C for lr_weight baseline (default: from config)")
    p.add_argument("--lr-max-iter", type=int, default=None, help="LogReg max_iter for lr_weight baseline (default: from config)")
    p.add_argument("--lr-seed", type=int, default=None, help="LogReg random_state for lr_weight baseline (default: from config)")
    p.add_argument("--device", type=str, default=None, help="cuda|cpu (default: from config)")
    p.add_argument("--tag", type=str, default=None, help="Optional output tag; writes to pareto/<tag>/ to avoid overwriting")
    if bool_action is None:  # pragma: no cover
        p.add_argument(
            "--resume",
            action="store_true",
            default=None,
            help="Resume from an existing pareto_by_subject.csv (skip completed subject×K×method cells).",
        )
        p.add_argument(
            "--overwrite",
            action="store_true",
            default=None,
            help="Overwrite existing outputs in the tag directory (use with care).",
        )
        p.add_argument("--plot", action="store_true", default=None, help="Save curve plots (default: from config)")
    else:
        p.add_argument("--resume", action=bool_action, default=None, help="Resume from existing outputs (default: from config)")
        p.add_argument("--overwrite", action=bool_action, default=None, help="Overwrite existing outputs (default: from config)")
        p.add_argument("--plot", action=bool_action, default=None, help="Save curve plots (default: from config)")
    p.add_argument(
        "--baseline-cache",
        type=str,
        default=None,
        help=(
            "Cache non-agent baselines (fisher/mi/lr_weight/sfs/random/ga/full22) across runs to avoid recomputing. "
            "Use 'auto' (default) to write to results/baseline_cache/, or 'none' to disable."
        ),
    )
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


def _default_checkpoint(ckpt_dir: Path) -> Path:
    # Prefer the best checkpoint if present; otherwise fall back to last/latest.
    best = ckpt_dir / "best.pt"
    if best.exists():
        return best
    last = ckpt_dir / "last.pt"
    if last.exists():
        return last
    pts = sorted(ckpt_dir.glob("iter_*.pt"))
    if not pts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    return pts[-1]


def _snapshot_checkpoint(ckpt_path: Path, *, out_dir: Path, ckpt: dict[str, Any]) -> Path | None:
    """
    Copy the checkpoint used for evaluation into the pareto output directory.

    This avoids irreproducible "moving target" evaluations when training keeps updating
    runs/<run>/checkpoints/best.pt or last.pt while/after you run pareto eval.
    """

    try:
        it = ckpt.get("iter", None)
        suffix = f"_iter{int(it):03d}" if it is not None else ""
    except Exception:
        suffix = ""

    dst = out_dir / f"ours_checkpoint{suffix}.pt"
    if dst.exists():
        return dst

    tmp = out_dir / f".tmp_ours_checkpoint{suffix}.pt"
    try:
        # Save from the already-loaded checkpoint dict to avoid races where best/last.pt is overwritten
        # between torch.load() and the snapshot copy.
        torch.save(ckpt, tmp)
        tmp.replace(dst)
        return dst
    except Exception as e:
        print(f"[pareto] WARNING: failed to snapshot checkpoint to {dst} ({e})")
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return None


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


def _make_fold_full_train(
    *,
    subject: int,
    variant: str,
    seed: int,
    need_val: bool,
    split_id: int = 0,
) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    # Load train-only subject data (no eval label usage for selection).
    sd = load_subject_data(subject, variant=variant, include_eval=False)
    if need_val:
        # mimic FoldSampler split strategy (StratifiedKFold on 0train)
        from eeg_channel_game.eeg.splits import make_stratified_splits

        splits = make_stratified_splits(sd.y_train, n_splits=5, seed=int(seed))
        split = splits[int(split_id)]
    else:
        all_idx = np.arange(sd.y_train.shape[0], dtype=np.int64)
        split = Split(train_idx=all_idx, val_idx=np.array([], dtype=np.int64))
    idx = split.train_idx

    stats = compute_fold_stats(
        bp=sd.bp_train[idx],
        q=sd.q_train[idx],
        y=sd.y_train[idx],
        artifact_corr_eog=sd.artifact_corr_eog,
        resid_ratio=sd.resid_ratio,
    )

    from eeg_channel_game.eeg.fold_sampler import FoldData

    fold = FoldData(subject=int(subject), split_id=int(split_id) if need_val else 0, stats=stats, split=split, subject_data=sd)
    fisher_scores = stats.fisher.mean(axis=1).astype(np.float32, copy=False)
    mi_scores = _mi_scores(sd.bp_train[split.train_idx], sd.y_train[split.train_idx])
    return sd, fold, fisher_scores, mi_scores


def _ours_search_fixed_k(
    *,
    cfg: dict[str, Any],
    variant: str,
    fold,
    net: torch.nn.Module,
    device: str,
    k: int,
    evaluator,
    restarts: int = 1,
    stochastic: bool = False,
    tau: float = 0.8,
    n_sim: int | None = None,
) -> tuple[int, list[int], dict[str, Any]]:
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
        infer_batch_size=int(cfg.get("mcts", {}).get("infer_batch_size", 1) or 1),
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
                rng=rng,
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
    pareto_cfg = cfg.get("eval", {}).get("pareto", {}) or {}
    variant = variant_from_cfg(cfg)
    out_dir = Path(cfg["project"]["out_dir"])
    paths = make_run_paths(out_dir)

    pareto_dir = paths.out_dir / "pareto"
    tag = args.tag if args.tag is not None else pareto_cfg.get("tag", None)
    if tag:
        pareto_dir = pareto_dir / str(tag)
    pareto_dir.mkdir(parents=True, exist_ok=True)

    resume = bool(args.resume) if args.resume is not None else bool(pareto_cfg.get("resume", False))
    overwrite = bool(args.overwrite) if args.overwrite is not None else bool(pareto_cfg.get("overwrite", False))
    plot = bool(args.plot) if args.plot is not None else bool(pareto_cfg.get("plot", False))

    if resume and overwrite:
        raise ValueError("Choose either --resume or --overwrite, not both.")

    fieldnames = ["subject", "variant", "k", "method", "n_ch", "fbcsp_kappa", "fbcsp_acc", "sel_idx", "meta"]
    by_subj_csv = pareto_dir / "pareto_by_subject.csv"
    if by_subj_csv.exists() and (not resume) and (not overwrite):
        raise FileExistsError(
            f"Output already exists: {by_subj_csv}\n"
            "Use --resume to continue, --overwrite to replace, or choose a different --tag."
        )

    rows: list[dict[str, Any]] = []
    done: set[tuple[int, int, str]] = set()
    if resume and by_subj_csv.exists():
        with by_subj_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    subj = int(r["subject"])
                    k = int(r["k"])
                    method = str(r["method"])
                except Exception:
                    continue
                done.add((subj, k, method))
                try:
                    rows.append(
                        {
                            "subject": subj,
                            "variant": str(r.get("variant", "")),
                            "k": k,
                            "method": method,
                            "n_ch": int(r.get("n_ch", 0)),
                            "sel_idx": str(r.get("sel_idx", "")),
                            "fbcsp_kappa": float(r.get("fbcsp_kappa", "nan")),
                            "fbcsp_acc": float(r.get("fbcsp_acc", "nan")),
                            "meta": str(r.get("meta", "")),
                        }
                    )
                except Exception:
                    # keep resume robust to partially written/corrupted lines
                    continue
        print(f"[pareto] resume: loaded {len(rows)} rows from {by_subj_csv}")

    if args.subjects is not None:
        subjects = [int(s.strip()) for s in str(args.subjects).split(",") if s.strip()]
    else:
        subj_cfg = pareto_cfg.get("subjects", None)
        if isinstance(subj_cfg, str):
            subjects = [int(s.strip()) for s in subj_cfg.split(",") if s.strip()]
        elif isinstance(subj_cfg, list):
            subjects = [int(s) for s in subj_cfg]
        else:
            subjects = [int(s) for s in cfg["data"]["subjects"]]

    k_spec = args.k if args.k is not None else pareto_cfg.get("k", "4,6,8,10,12")
    if isinstance(k_spec, str):
        ks = [int(s.strip()) for s in str(k_spec).split(",") if s.strip()]
    else:
        ks = [int(s) for s in (k_spec or [])]
    if not ks:
        raise ValueError("Empty K list")

    methods_spec = args.methods if args.methods is not None else pareto_cfg.get("methods", "ours,fisher,mi,full22")
    if isinstance(methods_spec, str):
        methods = [m.strip() for m in str(methods_spec).split(",") if m.strip()]
    else:
        methods = [str(m) for m in (methods_spec or [])]
    want_ours = "ours" in methods
    want_uct = "uct" in methods
    want_riemann = "riemann_ts_lr" in methods

    device = str(args.device) if args.device else str(pareto_cfg.get("device", cfg["project"].get("device", "cuda")))

    random_n = int(args.random_n) if args.random_n is not None else int(pareto_cfg.get("random_n", 200))
    ours_restarts = int(args.ours_restarts) if args.ours_restarts is not None else int(pareto_cfg.get("ours_restarts", 1))
    ours_stochastic = bool(args.ours_stochastic) if args.ours_stochastic is not None else bool(pareto_cfg.get("ours_stochastic", False))
    ours_tau = float(args.ours_tau) if args.ours_tau is not None else float(pareto_cfg.get("ours_tau", 0.8))
    uct_restarts = int(args.uct_restarts) if args.uct_restarts is not None else int(pareto_cfg.get("uct_restarts", 1))
    uct_stochastic = bool(args.uct_stochastic) if args.uct_stochastic is not None else bool(pareto_cfg.get("uct_stochastic", False))
    uct_tau = float(args.uct_tau) if args.uct_tau is not None else float(pareto_cfg.get("uct_tau", 0.8))

    ga_restarts = int(args.ga_restarts) if args.ga_restarts is not None else int(pareto_cfg.get("ga_restarts", 1))
    ga_pop = int(args.ga_pop) if args.ga_pop is not None else int(pareto_cfg.get("ga_pop", 64))
    ga_gens = int(args.ga_gens) if args.ga_gens is not None else int(pareto_cfg.get("ga_gens", 50))
    ga_elite = int(args.ga_elite) if args.ga_elite is not None else int(pareto_cfg.get("ga_elite", 4))
    ga_cx = float(args.ga_cx) if args.ga_cx is not None else float(pareto_cfg.get("ga_cx", 0.6))
    ga_mut = float(args.ga_mut) if args.ga_mut is not None else float(pareto_cfg.get("ga_mut", 0.2))
    ga_seed = int(args.ga_seed) if args.ga_seed is not None else int(pareto_cfg.get("ga_seed", 123))

    lr_c = float(args.lr_c) if args.lr_c is not None else float(pareto_cfg.get("lr_c", 1.0))
    lr_max_iter = int(args.lr_max_iter) if args.lr_max_iter is not None else int(pareto_cfg.get("lr_max_iter", 2000))
    lr_seed = int(args.lr_seed) if args.lr_seed is not None else int(pareto_cfg.get("lr_seed", 0))

    baseline_cache_mode = str(args.baseline_cache) if args.baseline_cache is not None else str(pareto_cfg.get("baseline_cache", "auto"))

    # Search evaluator used by ours/uct during selection (0train only; no eval leakage).
    train_cfg = cfg.get("train", {})
    switch_to_l1_iter = train_cfg.get("switch_to_l1_iter", None)
    if switch_to_l1_iter is None:
        eval_name = cfg.get("evaluator", {}).get("name", "l0")
    else:
        eval_name = cfg.get("evaluator", {}).get("phase_b", "l1_fbcsp")

    if eval_name == "l0":
        evaluator_search = L0Evaluator(
            lambda_cost=float(cfg["reward"]["lambda_cost"]),
            beta_redund=float(cfg["reward"]["beta_redund"]),
            artifact_gamma=float(cfg["reward"].get("artifact_gamma", 0.0)),
        )
        need_val = False
    elif eval_name in {"l0_lr_weight", "l0_lr", "lr_weight"}:
        evaluator_search = L0LrWeightEvaluator(
            lambda_cost=float(cfg["reward"]["lambda_cost"]),
            beta_redund=float(cfg["reward"]["beta_redund"]),
            artifact_gamma=float(cfg["reward"].get("artifact_gamma", 0.0)),
        )
        need_val = False
    elif eval_name == "l1_fbcsp":
        l1_cfg = cfg.get("evaluator", {}).get("l1_fbcsp", {})
        evaluator_search = L1FBCSPEvaluator(
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
        need_val = False
    elif eval_name == "l1_deep_masked":
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
        evaluator_search = L1DeepMaskedEvaluator(
            lambda_cost=float(cfg["reward"]["lambda_cost"]),
            artifact_gamma=float(cfg["reward"].get("artifact_gamma", 0.0)),
            robust_mode=str(l1_cfg.get("robust_mode", "mean_std")),
            robust_beta=float(l1_cfg.get("robust_beta", 0.5)),
            seeds=seeds or (0,),
            device=str(l1_cfg.get("device", device)),
            cfg=train_cfg,
        )
        need_val = True
    else:
        raise ValueError(f"Unknown evaluator {eval_name}")

    # Optional: unlabeled cross-session domain-shift penalty (SAFE; uses eval features only).
    ds_cfg = cfg.get("reward", {}).get("domain_shift", {}) or {}
    ds_enabled = bool(ds_cfg.get("enabled", False))
    ds_eta = float(ds_cfg.get("eta", 0.0))
    ds_mode = str(ds_cfg.get("mode", "bp_mean_l2"))
    if ds_enabled and ds_eta > 0.0:
        evaluator_search = DomainShiftPenaltyEvaluator(
            evaluator_search,
            eta=ds_eta,
            mode=ds_mode,
            data_root=Path("eeg_channel_game") / "data",
            variant=variant,
        )

    normalize = cfg.get("reward", {}).get("normalize", False)
    norm_mode = str(normalize).lower()
    if norm_mode in {"1", "true", "yes", "delta_full22", "delta"}:
        evaluator_search = DeltaFull22Evaluator(evaluator_search)
    elif norm_mode in {"adv_lrmax", "adv_full22_lrmax", "adv_full22_lr_weight", "adv_lr_weight_max"}:
        evaluator_search = AdvantageMaxBaselineEvaluator(evaluator_search)

    # Optional: baseline cache (across runs) for all deterministic non-agent methods.
    baseline_cache_path: Path | None
    baseline_cache_mode_norm = str(baseline_cache_mode or "auto").lower()
    if baseline_cache_mode_norm in {"none", "off", "false", "0", ""}:
        baseline_cache_path = None
        baseline_cache: dict[tuple[int, int, str], dict[str, Any]] = {}
        baseline_done: set[tuple[int, int, str]] = set()
    else:
        # Hash the knobs that affect baseline selection/scoring so cache remains valid.
        l1_cfg = cfg.get("evaluator", {}).get("l1_fbcsp", {}) or {}
        cache_cfg = {
            "variant": str(variant),
            "reward": {
                "lambda_cost": float(cfg["reward"]["lambda_cost"]),
                "artifact_gamma": float(cfg["reward"].get("artifact_gamma", 0.0)),
            },
            "evaluator_l1_fbcsp": {
                "mode": str(l1_cfg.get("mode", "mr_fbcsp")),
                "mask_eps": float(l1_cfg.get("mask_eps", 0.0)),
                "csp_shrinkage": float(l1_cfg.get("csp_shrinkage", 0.1)),
                "csp_ridge": float(l1_cfg.get("csp_ridge", 1e-3)),
                "mask_penalty": float(l1_cfg.get("mask_penalty", 0.1)),
                "cv_folds": int(l1_cfg.get("cv_folds", 3)),
                "robust_mode": str(l1_cfg.get("robust_mode", "mean_std")),
                "robust_beta": float(l1_cfg.get("robust_beta", 0.5)),
            },
            "baseline_params": {
                "random_n": int(random_n),
                "ga_restarts": int(ga_restarts),
                "ga_pop": int(ga_pop),
                "ga_gens": int(ga_gens),
                "ga_elite": int(ga_elite),
                "ga_cx": float(ga_cx),
                "ga_mut": float(ga_mut),
                "ga_seed": int(ga_seed),
                "lr_c": float(lr_c),
                "lr_max_iter": int(lr_max_iter),
                "lr_seed": int(lr_seed),
            },
            "eval": {
                "fbcsp_m": 2,
                "fbcsp_eps": 1e-6,
            },
        }
        cache_blob = json.dumps(cache_cfg, sort_keys=True).encode("utf-8")
        cache_id = hashlib.sha1(cache_blob).hexdigest()[:12]
        baseline_cache_path = (Path("results") / "baseline_cache" / str(variant) / f"pareto_fbcsp_{cache_id}.csv").resolve()
        baseline_cache_path.parent.mkdir(parents=True, exist_ok=True)

        baseline_cache = {}
        baseline_done = set()
        if baseline_cache_path.exists():
            try:
                with baseline_cache_path.open("r", newline="", encoding="utf-8") as bf:
                    br = csv.DictReader(bf)
                    for r in br:
                        try:
                            key = (int(r["subject"]), int(r["k"]), str(r["method"]))
                        except Exception:
                            continue
                        baseline_cache[key] = dict(r)
                        baseline_done.add(key)
                print(f"[pareto] baseline-cache: loaded {len(baseline_done)} rows from {baseline_cache_path}")
            except Exception as e:
                print(f"[pareto] baseline-cache: failed to load ({e}); continuing without cache")
                baseline_cache = {}
                baseline_done = set()

    net = None
    ckpt_path: Path | None = None
    ckpt_iter: int | None = None
    ckpt_snapshot: Path | None = None
    if want_ours:
        ckpt_dir = paths.ckpt_dir
        ckpt_spec = args.checkpoint if args.checkpoint is not None else pareto_cfg.get("checkpoint", None)
        if isinstance(ckpt_spec, str) and ckpt_spec.lower() in {"last", "best"}:
            candidate = ckpt_dir / f"{ckpt_spec.lower()}.pt"
            ckpt_path = candidate if candidate.exists() else _default_checkpoint(ckpt_dir)
        elif ckpt_spec:
            ckpt_path = Path(str(ckpt_spec))
        else:
            ckpt_path = _default_checkpoint(ckpt_dir)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        try:
            ckpt_iter = int(ckpt.get("iter")) if ckpt.get("iter", None) is not None else None
        except Exception:
            ckpt_iter = None
        ckpt_cfg = ckpt.get("cfg", {}) or {}
        ckpt_net = ckpt_cfg.get("net", {}) if isinstance(ckpt_cfg, dict) else {}
        cfg_net = cfg.get("net", {}) or {}
        net = PolicyValueNet(
            d_in=int(cfg_net.get("d_in", ckpt_net.get("d_in", 64))),
            d_model=int(cfg_net.get("d_model", ckpt_net.get("d_model", 128))),
            n_layers=int(cfg_net.get("n_layers", ckpt_net.get("n_layers", 4))),
            n_heads=int(cfg_net.get("n_heads", ckpt_net.get("n_heads", 4))),
            policy_mode=str(cfg_net.get("policy_mode", ckpt_net.get("policy_mode", "cls"))),
            think_steps=int(cfg_net.get("think_steps", ckpt_net.get("think_steps", 1)) or 1),
            n_actions=23,
        ).to(torch.device(device))
        missing, unexpected = net.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            print(f"[pareto] WARNING: load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
        net.eval()
        print(f"[pareto] checkpoint={ckpt_path}")
        ckpt_snapshot = _snapshot_checkpoint(ckpt_path, out_dir=pareto_dir, ckpt=ckpt)

    uct_net = None
    if want_uct:
        uct_net = UniformPolicyValueNet(n_tokens=24, n_actions=23).to(torch.device(device))
        uct_net.eval()

    # Snapshot the effective config/CLI for reproducibility.
    run_meta = {
        "argv": [str(x) for x in sys.argv],
        "config_path": str(args.config),
        "overrides": list(args.override or []),
        "resolved": {
            "out_dir": str(out_dir),
            "tag": str(tag) if tag else None,
            "device": str(device),
            "checkpoint": str(args.checkpoint) if args.checkpoint is not None else pareto_cfg.get("checkpoint", None),
            "checkpoint_path": str(ckpt_path) if ckpt_path is not None else None,
            "checkpoint_iter": int(ckpt_iter) if ckpt_iter is not None else None,
            "checkpoint_snapshot_path": str(ckpt_snapshot) if ckpt_snapshot is not None else None,
            "subjects": [int(s) for s in subjects],
            "k": [int(k) for k in ks],
            "methods": [str(m) for m in methods],
            "resume": bool(resume),
            "overwrite": bool(overwrite),
            "plot": bool(plot),
            "baseline_cache": str(baseline_cache_mode),
            "baseline_cache_path": str(baseline_cache_path) if baseline_cache_path is not None else None,
            "random_n": int(random_n),
            "ours": {"restarts": int(ours_restarts), "stochastic": bool(ours_stochastic), "tau": float(ours_tau)},
            "uct": {"restarts": int(uct_restarts), "stochastic": bool(uct_stochastic), "tau": float(uct_tau)},
            "ga": {
                "restarts": int(ga_restarts),
                "pop": int(ga_pop),
                "gens": int(ga_gens),
                "elite": int(ga_elite),
                "cx": float(ga_cx),
                "mut": float(ga_mut),
                "seed": int(ga_seed),
            },
            "lr_weight": {"c": float(lr_c), "max_iter": int(lr_max_iter), "seed": int(lr_seed)},
        },
        "cfg": cfg,
    }
    (pareto_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    (pareto_dir / "run_config.yaml").write_text(yaml.safe_dump(run_meta, sort_keys=False), encoding="utf-8")
    (pareto_dir / "command.txt").write_text(" ".join(str(x) for x in sys.argv), encoding="utf-8")

    csv_mode = "a" if (resume and by_subj_csv.exists() and (not overwrite)) else "w"
    with by_subj_csv.open(csv_mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if csv_mode == "w":
            w.writeheader()
            f.flush()

        for subj in subjects:
            sd_train, fold, fisher_scores, mi_scores = _make_fold_full_train(
                subject=int(subj),
                variant=variant,
                seed=int(cfg["project"]["seed"]),
                need_val=bool(need_val),
                split_id=0,
            )
            lr_scores = _lr_weight_scores(
                bp=sd_train.bp_train[fold.split.train_idx],
                y=sd_train.y_train[fold.split.train_idx],
                c=float(lr_c),
                max_iter=int(lr_max_iter),
                seed=int(lr_seed),
            )
            riemann_scores = None
            if want_riemann:
                riemann_scores = riemann_ts_lr_channel_scores(
                    sd_train.X_train[fold.split.train_idx],
                    sd_train.y_train[fold.split.train_idx],
                    cov_estimator="oas",
                    ts_metric="riemann",
                    c=1.0,
                    max_iter=2000,
                    seed=int(lr_seed),
                )

            # L1 evaluator used by SFS/random baselines (selection only, 0train).
            l1_cfg = cfg.get("evaluator", {}).get("l1_fbcsp", {})
            evaluator_l1 = L1FBCSPEvaluator(
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

            for k in ks:
                if k < 2 or k > 22:
                    raise ValueError("K must be in [2,22]")

                pending = [m for m in methods if (int(subj), int(k), str(m)) not in done]
                if not pending:
                    continue

                # Fast path: if baseline-cache has rows for non-agent methods, reuse them directly.
                if baseline_cache_path is not None:
                    for mname in list(pending):
                        if mname in {"ours", "uct"}:
                            continue
                        ck = (int(subj), int(k), str(mname))
                        if ck not in baseline_cache:
                            continue
                        row = dict(baseline_cache[ck])
                        # Normalize dtypes and ensure all columns exist.
                        row_out = {
                            "subject": int(row.get("subject", subj)),
                            "variant": str(row.get("variant", variant)),
                            "k": int(row.get("k", k)),
                            "method": str(row.get("method", mname)),
                            "n_ch": int(row.get("n_ch", k)),
                            "sel_idx": str(row.get("sel_idx", "[]")),
                            "fbcsp_kappa": float(row.get("fbcsp_kappa", float("nan"))),
                            "fbcsp_acc": float(row.get("fbcsp_acc", float("nan"))),
                            "meta": str(row.get("meta", "{}")),
                        }
                        rows.append(row_out)
                        done.add((int(subj), int(k), str(mname)))
                        w.writerow(row_out)
                        f.flush()
                        print(
                            f"[pareto] subj={int(subj):02d} K={int(k):02d} {mname}: "
                            f"kappa/acc={float(row_out['fbcsp_kappa']):.4f}/{float(row_out['fbcsp_acc']):.4f} (cached)"
                        )
                        pending.remove(mname)
                    if not pending:
                        continue

                # selection (train-only)
                subsets: dict[str, list[int]] = {}
                meta: dict[str, dict[str, Any]] = {}

                if "ours" in pending:
                    assert net is not None
                    key, sel, info = _ours_search_fixed_k(
                        cfg=cfg,
                        variant=variant,
                        fold=fold,
                        net=net,
                        device=device,
                        k=k,
                        evaluator=evaluator_search,
                        restarts=int(ours_restarts),
                        stochastic=bool(ours_stochastic),
                        tau=float(ours_tau),
                    )
                    subsets["ours"] = sel
                    meta["ours"] = {"key": int(key), **info}

                if "uct" in pending:
                    assert uct_net is not None
                    key, sel, info = _ours_search_fixed_k(
                        cfg=cfg,
                        variant=variant,
                        fold=fold,
                        net=uct_net,
                        device=device,
                        k=k,
                        evaluator=evaluator_search,
                        restarts=int(uct_restarts),
                        stochastic=bool(uct_stochastic),
                        tau=float(uct_tau),
                    )
                    subsets["uct"] = sel
                    meta["uct"] = {"key": int(key), **info}

                if "fisher" in pending:
                    subsets["fisher"] = _topk(fisher_scores, k)

                if "mi" in pending:
                    subsets["mi"] = _topk(mi_scores, k)

                if "lr_weight" in pending:
                    subsets["lr_weight"] = _topk(lr_scores, k)

                if "riemann_ts_lr" in pending:
                    if riemann_scores is None:
                        raise RuntimeError("riemann_ts_lr requested but scores were not computed")
                    subsets["riemann_ts_lr"] = _topk(riemann_scores, k)

                if "sfs_l1" in pending:
                    subsets["sfs_l1"] = _sfs_by_l1(k=k, fold=fold, evaluator=evaluator_l1, fisher_scores=fisher_scores)

                if "random_best_l1" in pending:
                    sel, score = _random_best_by_l1(
                        k=k, n=int(random_n), fold=fold, evaluator=evaluator_l1, seed=123
                    )
                    subsets["random_best_l1"] = sel
                    meta["random_best_l1"] = {"l1_reward": float(score)}

                if "ga_l1" in pending:
                    best = (-1e18, None)
                    for ri in range(int(ga_restarts)):
                        sel, score = _ga_by_l1(
                            k=k,
                            fold=fold,
                            evaluator=evaluator_l1,
                            seed=int(ga_seed) + 10_000 * int(subj) + 100 * int(k) + int(ri),
                            pop_size=int(ga_pop),
                            n_gens=int(ga_gens),
                            elite=int(ga_elite),
                            cx_prob=float(ga_cx),
                            mut_prob=float(ga_mut),
                        )
                        if float(score) > best[0]:
                            best = (float(score), sel)
                    assert best[1] is not None
                    subsets["ga_l1"] = list(best[1])
                    meta["ga_l1"] = {
                        "l1_reward": float(best[0]),
                        "ga_restarts": int(ga_restarts),
                        "ga_pop": int(ga_pop),
                        "ga_gens": int(ga_gens),
                        "ga_elite": int(ga_elite),
                        "ga_cx": float(ga_cx),
                        "ga_mut": float(ga_mut),
                    }

                if "full22" in pending:
                    subsets["full22"] = list(range(22))

                # evaluation (uses eval labels; OK for reporting)
                for mname, sel_idx in subsets.items():
                    te = _eval_fbcsp_train_eval(variant=variant, subject=int(subj), sel_idx=sel_idx)
                    row = {
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
                    rows.append(row)
                    done.add((int(subj), int(k), str(mname)))
                    w.writerow(row)
                    f.flush()
                    print(
                        f"[pareto] subj={int(subj):02d} K={int(k):02d} {mname}: "
                        f"kappa/acc={float(te['kappa']):.4f}/{float(te['acc']):.4f}"
                    )

                    # Persist non-agent baselines across runs (so future tags don't recompute).
                    if baseline_cache_path is not None and mname not in {"ours", "uct"}:
                        ck = (int(subj), int(k), str(mname))
                        if ck not in baseline_done:
                            baseline_done.add(ck)
                            baseline_cache[ck] = dict(row)
                            # Append safely (write header if new file).
                            write_header = not baseline_cache_path.exists()
                            with baseline_cache_path.open("a", newline="", encoding="utf-8") as bf:
                                bw = csv.DictWriter(bf, fieldnames=fieldnames)
                                if write_header:
                                    bw.writeheader()
                                bw.writerow(row)

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

    if plot:
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
