from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eeg_channel_game.eeg.variant import variant_from_cfg
from eeg_channel_game.eeg.io import load_subject_data
from eeg_channel_game.eval.fbcsp import fit_fbcsp_ovr_filters, transform_fbcsp_features
from eeg_channel_game.eval.evaluator_l2_deep import evaluate_l2_deep_train_eval
from eeg_channel_game.eval.metrics import accuracy, cohen_kappa
from eeg_channel_game.utils.bitmask import key_to_mask
from eeg_channel_game.utils.config import load_config, make_run_paths
from eeg_channel_game.utils.visualization import plot_channel_mask_topomap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint and plot selected channels")
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--override",
        action="extend",
        nargs="+",
        default=[],
        help="YAML override(s) like project.out_dir=runs/x (repeatable)",
    )
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint (default: latest)")
    p.add_argument("--show-names", action="store_true", help="Show channel names on topomap (can be cluttered)")
    p.add_argument("--l2", action="store_true", help="Run L2 deep evaluation (train session -> eval session)")
    p.add_argument(
        "--l2-model", type=str, default="eegnetv4", help="eegnetv4 | shallowfbcspnet | shallow_masked | mcfbvarnet"
    )
    p.add_argument("--l2-device", type=str, default=None, help="Override device, e.g. cuda or cpu")
    p.add_argument("--l2-epochs", type=int, default=30)
    p.add_argument("--l2-batch-size", type=int, default=64)
    p.add_argument("--l2-lr", type=float, default=1e-3)
    p.add_argument("--l2-weight-decay", type=float, default=1e-4)
    p.add_argument("--l2-patience", type=int, default=8)
    p.add_argument("--l2-seeds", type=str, default="0,1,2", help="Comma-separated seeds, e.g. 0,1,2")
    p.add_argument("--l2-full", action="store_true", help="Also evaluate on all 22 channels (slower)")
    p.add_argument("--l2-split-seed", type=int, default=42, help="Seed for train/val split within train session")
    return p.parse_args()


def _latest_checkpoint(ckpt_dir: Path) -> Path:
    pts = sorted(ckpt_dir.glob("iter_*.pt"))
    if not pts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    return pts[-1]


def _eval_bandpower_logreg(
    bp_train: np.ndarray, y_train: np.ndarray, bp_eval: np.ndarray, y_eval: np.ndarray
) -> dict[str, float]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for evaluation") from e

    x_tr = bp_train.reshape(bp_train.shape[0], -1)
    x_ev = bp_eval.reshape(bp_eval.shape[0], -1)
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=2000),
    )
    clf.fit(x_tr, y_train)
    y_pred = clf.predict(x_ev)
    return {"acc": accuracy(y_eval, y_pred), "kappa": cohen_kappa(y_eval, y_pred)}


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
    return {"acc": accuracy(y_eval, y_pred), "kappa": cohen_kappa(y_eval, y_pred)}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    variant = variant_from_cfg(cfg)
    out_dir = Path(cfg["project"]["out_dir"])
    paths = make_run_paths(out_dir)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(paths.ckpt_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    key = int(ckpt.get("best_key") or 0)
    mask = key_to_mask(key, n_ch=22).astype(np.int8)
    sel_idx = np.where(mask == 1)[0].tolist()

    sd = load_subject_data(int(args.subject), variant=variant, include_eval=True)
    if sd.y_eval is None or sd.bp_eval is None:
        raise RuntimeError("Expected eval labels/features; did you prepare data with include_eval=True?")
    sel_names = [sd.ch_names[i] for i in sel_idx]
    print(f"[eval] checkpoint={ckpt_path}")
    print(f"[eval] subject={sd.subject:02d} selected_n={len(sel_idx)} selected={sel_names}")

    if not sel_idx:
        print("[eval] WARNING: empty selection; skip classifier + topomap.")
        return

    res_sel = _eval_bandpower_logreg(sd.bp_train[:, sel_idx, :], sd.y_train, sd.bp_eval[:, sel_idx, :], sd.y_eval)
    res_full = _eval_bandpower_logreg(sd.bp_train, sd.y_train, sd.bp_eval, sd.y_eval)
    print(f"[eval] bandpower-logreg kappa/acc (selected): {res_sel['kappa']:.4f} / {res_sel['acc']:.4f}")
    print(f"[eval] bandpower-logreg kappa/acc (all 22):   {res_full['kappa']:.4f} / {res_full['acc']:.4f}")

    # Strong traditional baseline on the real protocol: train session -> eval session
    try:
        data_root = Path("eeg_channel_game") / "data"
        cov_dir = data_root / "cache" / variant / f"subj{sd.subject:02d}"
        cov_t = np.load(cov_dir / "sessionT_cov_fb.npz")["cov_fb"].astype(np.float32, copy=False)
        cov_e = np.load(cov_dir / "sessionE_cov_fb.npz")["cov_fb"].astype(np.float32, copy=False)
        fbcsp_sel = _eval_fbcsp_train_eval(
            cov_train=cov_t, y_train=sd.y_train, cov_eval=cov_e, y_eval=sd.y_eval, sel_idx=sel_idx
        )
        fbcsp_full = _eval_fbcsp_train_eval(
            cov_train=cov_t, y_train=sd.y_train, cov_eval=cov_e, y_eval=sd.y_eval, sel_idx=list(range(22))
        )
        print(f"[eval] FBCSP(rLDA) kappa/acc (selected): {fbcsp_sel['kappa']:.4f} / {fbcsp_sel['acc']:.4f}")
        print(f"[eval] FBCSP(rLDA) kappa/acc (all 22):   {fbcsp_full['kappa']:.4f} / {fbcsp_full['acc']:.4f}")
    except FileNotFoundError:
        print("[eval] FBCSP(rLDA) skipped: missing cov_fb cache (rerun run_prepare_data without --no-cov).")

    fig_path = paths.fig_dir / f"subj{sd.subject:02d}_selected_topomap.png"
    plot_channel_mask_topomap(
        ch_names=sd.ch_names,
        mask=mask.astype(np.float32),
        sfreq=sd.sfreq,
        title=f"Subject {sd.subject:02d} | n={len(sel_idx)}",
        save_path=fig_path,
        show_names=bool(args.show_names),
    )
    print(f"[eval] saved: {fig_path}")

    if args.l2:
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except Exception as e:  # pragma: no cover
            raise RuntimeError("scikit-learn is required for L2 deep evaluation") from e

        seed_list = [int(s.strip()) for s in args.l2_seeds.split(",") if s.strip()]
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=int(args.l2_split_seed))
        train_idx, val_idx = next(splitter.split(np.zeros_like(sd.y_train), sd.y_train))

        l2_device = str(args.l2_device) if args.l2_device else str(cfg["project"].get("device", "cuda"))
        eval_dir = paths.out_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        if not sel_idx:
            print("[l2] WARNING: empty selection; skip deep evaluation.")
            return

        res_sel = evaluate_l2_deep_train_eval(
            subject_data=sd,
            sel_idx=sel_idx,
            train_idx=train_idx,
            val_idx=val_idx,
            model_name=str(args.l2_model),
            seeds=seed_list,
            device=l2_device,
            epochs=int(args.l2_epochs),
            batch_size=int(args.l2_batch_size),
            lr=float(args.l2_lr),
            weight_decay=float(args.l2_weight_decay),
            patience=int(args.l2_patience),
        )
        out_path = eval_dir / f"subj{sd.subject:02d}_l2_selected.json"
        out_path.write_text(json.dumps(res_sel, indent=2), encoding="utf-8")
        print(f"[l2] selected kappa(mean/q20)={res_sel['kappa_mean']:.4f}/{res_sel['kappa_q20']:.4f} acc(mean)={res_sel['acc_mean']:.4f}")
        print(f"[l2] saved: {out_path}")

        if args.l2_full:
            full_idx = list(range(22))
            res_full = evaluate_l2_deep_train_eval(
                subject_data=sd,
                sel_idx=full_idx,
                train_idx=train_idx,
                val_idx=val_idx,
                model_name=str(args.l2_model),
                seeds=seed_list,
                device=l2_device,
                epochs=int(args.l2_epochs),
                batch_size=int(args.l2_batch_size),
                lr=float(args.l2_lr),
                weight_decay=float(args.l2_weight_decay),
                patience=int(args.l2_patience),
            )
            out_path2 = eval_dir / f"subj{sd.subject:02d}_l2_full.json"
            out_path2.write_text(json.dumps(res_full, indent=2), encoding="utf-8")
            print(f"[l2] full kappa(mean/q20)={res_full['kappa_mean']:.4f}/{res_full['kappa_q20']:.4f} acc(mean)={res_full['acc_mean']:.4f}")
            print(f"[l2] saved: {out_path2}")


if __name__ == "__main__":
    main()
