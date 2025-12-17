from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eeg_channel_game.eeg.io import load_subject_data
from eeg_channel_game.eeg.variant import variant_from_cfg
from eeg_channel_game.eval.fbcsp import fit_fbcsp_ovr_filters, transform_fbcsp_features
from eeg_channel_game.eval.metrics import accuracy, cohen_kappa
from eeg_channel_game.utils.config import load_config, make_run_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run strong baselines on BCI-IV 2a (train session -> eval session)")
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--override",
        action="extend",
        nargs="+",
        default=[],
        help="YAML override(s) like project.out_dir=runs/x (repeatable)",
    )
    p.add_argument("--subjects", type=str, default=None, help="Comma-separated list, e.g. 1,2,3 (override)")
    p.add_argument("--model", type=str, default="eegnetv4", help="L2 model: eegnetv4 | shallowfbcspnet | vtransformer")
    p.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds for deep baseline")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--split-seed", type=int, default=42, help="Seed for train/val split within training session")
    p.add_argument("--device", type=str, default=None, help="cuda|cpu (default: from config)")
    return p.parse_args()


def _cov_path(data_root: Path, variant: str, subject: int, session: str) -> Path:
    if session not in {"T", "E"}:
        raise ValueError("session must be 'T' or 'E'")
    name = "sessionT_cov_fb.npz" if session == "T" else "sessionE_cov_fb.npz"
    return data_root / "cache" / variant / f"subj{subject:02d}" / name


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
        raise RuntimeError("scikit-learn is required for FBCSP baseline") from e

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
    return {"kappa": cohen_kappa(y_eval, y_pred), "acc": accuracy(y_eval, y_pred)}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    variant = variant_from_cfg(cfg)
    data_root = Path("eeg_channel_game") / "data"

    out_dir = Path(cfg["project"]["out_dir"])
    paths = make_run_paths(out_dir)
    base_dir = paths.out_dir / "baselines"
    base_dir.mkdir(parents=True, exist_ok=True)

    subjects = cfg["data"].get("subjects", [1])
    if args.subjects:
        subjects = [int(s.strip()) for s in args.subjects.split(",") if s.strip()]
    subjects = [int(s) for s in subjects]

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    device = str(args.device) if args.device else str(cfg["project"].get("device", "cuda"))

    # fixed train/val split inside training session for deep baseline
    try:
        from sklearn.model_selection import StratifiedShuffleSplit
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for baselines") from e

    all_rows = []
    for subj in subjects:
        sd = load_subject_data(subj, data_root=data_root, variant=variant, include_eval=True)
        if sd.y_eval is None:
            raise RuntimeError("Expected eval labels (include_eval=True)")

        cov_t = np.load(_cov_path(data_root, variant, subj, "T"))["cov_fb"].astype(np.float32, copy=False)
        cov_e = np.load(_cov_path(data_root, variant, subj, "E"))["cov_fb"].astype(np.float32, copy=False)

        full_sel = list(range(22))
        fbcsp = _eval_fbcsp_train_eval(
            cov_train=cov_t,
            y_train=sd.y_train,
            cov_eval=cov_e,
            y_eval=sd.y_eval,
            sel_idx=full_sel,
        )

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=int(args.split_seed))
        tr_idx, va_idx = next(splitter.split(np.zeros_like(sd.y_train), sd.y_train))

        from eeg_channel_game.eval.evaluator_l2_deep import evaluate_l2_deep_train_eval

        deep = evaluate_l2_deep_train_eval(
            subject_data=sd,
            sel_idx=full_sel,
            train_idx=tr_idx,
            val_idx=va_idx,
            model_name=str(args.model),
            seeds=seeds,
            device=device,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            patience=int(args.patience),
        )

        row = {
            "subject": subj,
            "variant": variant,
            "fbcsp_kappa": float(fbcsp["kappa"]),
            "fbcsp_acc": float(fbcsp["acc"]),
            "deep_model": deep["model"],
            "deep_kappa_mean": deep["kappa_mean"],
            "deep_kappa_std": deep["kappa_std"],
            "deep_kappa_q20": deep["kappa_q20"],
            "deep_acc_mean": deep["acc_mean"],
            "deep_acc_std": deep["acc_std"],
            "deep_train_time_s_sum": deep["train_time_s_sum"],
        }
        all_rows.append(row)
        print(
            f"[baseline] subj={subj:02d} FBCSP kappa/acc={row['fbcsp_kappa']:.4f}/{row['fbcsp_acc']:.4f} "
            f"DEEP({row['deep_model']}) kappa(mean/q20)={row['deep_kappa_mean']:.4f}/{row['deep_kappa_q20']:.4f}"
        )

    # save
    out_json = base_dir / f"baselines_{variant}.json"
    out_json.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    try:
        import pandas as pd

        df = pd.DataFrame(all_rows)
        out_csv = base_dir / f"baselines_{variant}.csv"
        df.to_csv(out_csv, index=False)
        print(f"[baseline] saved: {out_csv}")
    except Exception:
        pass
    print(f"[baseline] saved: {out_json}")


if __name__ == "__main__":
    main()
