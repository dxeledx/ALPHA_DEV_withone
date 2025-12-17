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
from eeg_channel_game.eval.evaluator_l1_fbcsp import L1FBCSPEvaluator
from eeg_channel_game.eval.evaluator_l2_deep import evaluate_l2_deep_train_eval
from eeg_channel_game.eval.metrics import accuracy, cohen_kappa
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
    p.add_argument("--checkpoint", type=str, default=None, help="Our method checkpoint (.pt) for subset (optional)")

    p.add_argument("--random-n", type=int, default=200, help="Number of random subsets to sample (pick best by L1)")
    p.add_argument("--l1-cv-folds", type=int, default=3)
    p.add_argument("--l1-robust-mode", type=str, default="mean_std")
    p.add_argument("--l1-robust-beta", type=float, default=0.5)

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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    variant = variant_from_cfg(cfg)
    out_dir = Path(cfg["project"]["out_dir"])
    paths = make_run_paths(out_dir)
    comp_dir = paths.out_dir / "compare_k"
    comp_dir.mkdir(parents=True, exist_ok=True)

    subject = int(args.subject)
    ks = [int(s.strip()) for s in args.k.split(",") if s.strip()]
    l2_device = str(args.l2_device) if args.l2_device else str(cfg["project"].get("device", "cuda"))
    l2_seeds = [int(s.strip()) for s in args.l2_seeds.split(",") if s.strip()]

    # L1 evaluator (used for SFS/random selection only; L2 is final scoring)
    sampler = FoldSampler(subjects=[subject], n_splits=5, seed=int(cfg["project"]["seed"]), variant=variant, include_eval=False)
    fold = sampler.get_fold(subject, int(args.split_id))
    evaluator_l1 = L1FBCSPEvaluator(
        lambda_cost=float(cfg["reward"]["lambda_cost"]),
        artifact_gamma=float(cfg["reward"].get("artifact_gamma", 0.0)),
        cv_folds=int(args.l1_cv_folds),
        robust_mode=str(args.l1_robust_mode),
        robust_beta=float(args.l1_robust_beta),
        variant=variant,
    )

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

    rows = []
    for k in ks:
        if k < 2 or k > 22:
            raise ValueError("K must be in [2,22]")

        subsets: dict[str, list[int]] = {}
        subsets["fisher_topk"] = _topk_from_scores(fisher, k)
        subsets["mi_topk"] = _topk_from_scores(mi, k)
        subsets["sfs_l1"] = _sfs_by_l1(k=k, fold=fold, evaluator=evaluator_l1)
        rnd_sel, rnd_score = _random_best_by_l1(k=k, n=int(args.random_n), fold=fold, evaluator=evaluator_l1, seed=123)
        subsets["random_best_l1"] = rnd_sel

        if args.checkpoint:
            ours = _ours_from_checkpoint(args.checkpoint, k)
            subsets["ours_ckpt"] = _adjust_to_k(ours, k, fisher)

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
