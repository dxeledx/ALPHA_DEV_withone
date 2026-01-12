from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eeg_channel_game.eeg.covariance import compute_cov_fb
from eeg_channel_game.eeg.features import compute_bandpower_fft, compute_quality_features
from eeg_channel_game.eeg.preprocess import fit_eog_regression, mean_abs_eeg_eog_corr


LABEL_MAP = {
    "left_hand": 0,
    "right_hand": 1,
    "feet": 2,
    "tongue": 3,
}


@dataclass(frozen=True)
class PreparedSubjectPaths:
    processed_dir: Path
    cache_dir: Path

    train_epochs_path: Path
    eval_epochs_path: Path
    meta_path: Path

    sessionT_bp_path: Path
    sessionE_bp_path: Path
    sessionT_quality_path: Path
    sessionT_cov_fb_path: Path
    sessionE_cov_fb_path: Path


def _paths(data_root: Path, subject: int, variant: str) -> PreparedSubjectPaths:
    processed_dir = data_root / "processed" / variant / f"subj{subject:02d}"
    cache_dir = data_root / "cache" / variant / f"subj{subject:02d}"
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return PreparedSubjectPaths(
        processed_dir=processed_dir,
        cache_dir=cache_dir,
        train_epochs_path=processed_dir / "train_epochs.npz",
        eval_epochs_path=processed_dir / "eval_epochs.npz",
        meta_path=processed_dir / "meta.json",
        sessionT_bp_path=cache_dir / "sessionT_bp.npz",
        sessionE_bp_path=cache_dir / "sessionE_bp.npz",
        sessionT_quality_path=cache_dir / "sessionT_quality.npz",
        sessionT_cov_fb_path=cache_dir / "sessionT_cov_fb.npz",
        sessionE_cov_fb_path=cache_dir / "sessionE_cov_fb.npz",
    )


def prepare_subject_bciiv2a(
    *,
    subject: int,
    data_root: str | Path,
    variant: str,
    fmin: float,
    fmax: float,
    tmin_rel: float,
    tmax_rel: float,
    bands: list[tuple[float, float]],
    include_eog: bool = True,
    use_eog_regression: bool = True,
    compute_cov: bool = True,
) -> PreparedSubjectPaths:
    """
    Prepare one subject:
      - MOABB load epochs (EEG-only by default; optionally include EOG channels)
      - optional: fit EOG regression on training session only, then clean EEG for train/eval
      - save EEG epochs (22ch) + caches (bandpower/quality/cov_fb)
    """
    data_root = Path(data_root)
    paths = _paths(data_root, subject, str(variant))

    try:
        import mne
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependencies: mne/moabb") from e

    dataset = BNCI2014_001()

    raw0 = dataset.get_data(subjects=[subject])[subject]["0train"]["0"]
    include_eog = bool(include_eog)
    if include_eog:
        all_channels = [ch for ch in raw0.ch_names if ch != "stim"]
    else:
        eeg_picks_raw = mne.pick_types(raw0.info, eeg=True, eog=False, stim=False)
        all_channels = [raw0.ch_names[i] for i in eeg_picks_raw]
        if not all_channels:
            raise RuntimeError("No EEG channels found when include_eog=false")
        if use_eog_regression:
            # Enforce consistency: cannot do EOG regression without loading EOG channels.
            use_eog_regression = False

    paradigm = MotorImagery(
        n_classes=4,
        fmin=float(fmin),
        fmax=float(fmax),
        tmin=float(tmin_rel),
        tmax=float(tmax_rel),
        channels=all_channels,
    )

    epochs, y_raw, meta = paradigm.get_data(dataset=dataset, subjects=[subject], return_epochs=True)
    y = np.array([LABEL_MAP[str(lbl)] for lbl in y_raw], dtype=np.int64)

    eeg_picks = mne.pick_types(epochs.info, eeg=True, eog=False, stim=False)
    eog_picks = mne.pick_types(epochs.info, eeg=False, eog=True, stim=False)
    eeg_names = [epochs.ch_names[i] for i in eeg_picks]
    eog_names = [epochs.ch_names[i] for i in eog_picks]

    session_arr = meta["session"].to_numpy()
    train_idx = np.where(session_arr == "0train")[0]
    eval_idx = np.where(session_arr == "1test")[0]
    if train_idx.size == 0 or eval_idx.size == 0:
        raise RuntimeError(f"Unexpected sessions in meta: {sorted(set(session_arr.tolist()))}")

    X = epochs.get_data().astype(np.float32, copy=False)  # [N, 25, T]

    X_train = X[train_idx]
    X_eval = X[eval_idx]
    y_train = y[train_idx]
    y_eval = y[eval_idx]

    eeg_train = X_train[:, eeg_picks, :]
    eeg_eval = X_eval[:, eeg_picks, :]

    if include_eog and eog_picks.size:
        eog_train = X_train[:, eog_picks, :]
        eog_eval = X_eval[:, eog_picks, :]
        artifact_corr = mean_abs_eeg_eog_corr(eeg_train, eog_train)
    else:
        # Pure-EEG mode: no EOG channels are loaded, so EOG correlation features are undefined.
        # Keep a stable placeholder to preserve downstream shapes.
        artifact_corr = np.zeros((eeg_train.shape[1],), dtype=np.float32)
        eog_train = None
        eog_eval = None

    if use_eog_regression:
        assert eog_train is not None and eog_eval is not None
        reg = fit_eog_regression(eeg_train, eog_train, ridge=1e-3)
        eeg_train_clean = reg.apply(eeg_train, eog_train)
        eeg_eval_clean = reg.apply(eeg_eval, eog_eval)
        resid_ratio = (
            np.var(eeg_train_clean, axis=(0, 2)) / (np.var(eeg_train, axis=(0, 2)) + 1e-8)
        ).astype(np.float32)
        reg_meta: dict[str, Any] = {"ridge": 1e-3, "coef_shape": list(reg.coef.shape)}
    else:
        eeg_train_clean = eeg_train.astype(np.float32, copy=False)
        eeg_eval_clean = eeg_eval.astype(np.float32, copy=False)
        resid_ratio = np.ones((eeg_train_clean.shape[1],), dtype=np.float32)
        reg_meta = {"disabled": True, "reason": "include_eog=false" if not include_eog else "use_eog_regression=false"}

    meta_out: dict[str, Any] = {
        "subject": int(subject),
        "variant": str(variant),
        "sfreq": float(epochs.info["sfreq"]),
        "window_abs_s": [float(dataset.interval[0] + tmin_rel), float(dataset.interval[0] + tmax_rel)],
        "label_map": LABEL_MAP,
        "eeg_names": eeg_names,
        "eog_names": eog_names,
        "include_eog": bool(include_eog),
        "bands": [[float(a), float(b)] for a, b in bands],
        "eog_regression": reg_meta,
    }
    paths.meta_path.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")

    np.savez_compressed(
        paths.train_epochs_path,
        X=eeg_train_clean.astype(np.float32, copy=False),
        y=y_train,
    )
    np.savez_compressed(
        paths.eval_epochs_path,
        X=eeg_eval_clean.astype(np.float32, copy=False),
        y=y_eval,
    )

    sfreq = float(epochs.info["sfreq"])
    bp_train = compute_bandpower_fft(eeg_train_clean, sfreq=sfreq, bands=bands)
    bp_eval = compute_bandpower_fft(eeg_eval_clean, sfreq=sfreq, bands=bands)
    np.savez_compressed(paths.sessionT_bp_path, bp=bp_train.astype(np.float32, copy=False))
    np.savez_compressed(paths.sessionE_bp_path, bp=bp_eval.astype(np.float32, copy=False))

    q_train = compute_quality_features(eeg_train_clean)
    np.savez_compressed(
        paths.sessionT_quality_path,
        q=q_train.astype(np.float32, copy=False),
        artifact_corr_eog=artifact_corr,
        resid_ratio=resid_ratio,
    )

    if compute_cov:
        cov_fb_t = compute_cov_fb(eeg_train_clean, sfreq=sfreq, bands=bands)
        cov_fb_e = compute_cov_fb(eeg_eval_clean, sfreq=sfreq, bands=bands)
        np.savez_compressed(paths.sessionT_cov_fb_path, cov_fb=cov_fb_t.astype(np.float32, copy=False))
        np.savez_compressed(paths.sessionE_cov_fb_path, cov_fb=cov_fb_e.astype(np.float32, copy=False))

    return paths
