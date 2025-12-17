from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SubjectData:
    subject: int
    sfreq: float
    ch_names: list[str]
    bands: list[tuple[float, float]]

    X_train: np.ndarray  # [N_train, 22, T]
    y_train: np.ndarray  # [N_train]
    X_eval: Optional[np.ndarray]  # [N_eval, 22, T]
    y_eval: Optional[np.ndarray]  # [N_eval]

    bp_train: np.ndarray  # [N_train, 22, B]
    bp_eval: Optional[np.ndarray]  # [N_eval, 22, B]
    q_train: np.ndarray  # [N_train, 22, Q]
    artifact_corr_eog: np.ndarray  # [22]
    resid_ratio: np.ndarray  # [22]


def _resolve_variant_dirs(data_root: Path, subject: int, variant: str | None) -> tuple[Path, Path]:
    """
    Returns (processed_dir, cache_dir).
    Backward compatible:
      - if variant is None, first try legacy layout: processed/subjXX + cache/subjXX
      - else use processed/<variant>/subjXX + cache/<variant>/subjXX
    """
    if variant is None:
        legacy_processed = data_root / "processed" / f"subj{subject:02d}"
        legacy_cache = data_root / "cache" / f"subj{subject:02d}"
        if legacy_processed.exists():
            return legacy_processed, legacy_cache
        variant = "default"

    processed_dir = data_root / "processed" / str(variant) / f"subj{subject:02d}"
    cache_dir = data_root / "cache" / str(variant) / f"subj{subject:02d}"
    return processed_dir, cache_dir


def load_subject_data(
    subject: int,
    *,
    data_root: str | Path = Path("eeg_channel_game") / "data",
    variant: str | None = None,
    include_eval: bool = True,
) -> SubjectData:
    data_root = Path(data_root)
    processed_dir, cache_dir = _resolve_variant_dirs(data_root, int(subject), variant)

    meta = json.loads((processed_dir / "meta.json").read_text(encoding="utf-8"))
    sfreq = float(meta["sfreq"])
    ch_names = list(meta["eeg_names"])
    bands = [tuple(map(float, b)) for b in meta["bands"]]

    tr = np.load(processed_dir / "train_epochs.npz")
    X_train = tr["X"].astype(np.float32, copy=False)
    y_train = tr["y"].astype(np.int64, copy=False)
    if include_eval:
        ev = np.load(processed_dir / "eval_epochs.npz")
        X_eval = ev["X"].astype(np.float32, copy=False)
        y_eval = ev["y"].astype(np.int64, copy=False)
    else:
        X_eval = None
        y_eval = None

    bp_train = np.load(cache_dir / "sessionT_bp.npz")["bp"].astype(np.float32, copy=False)
    bp_eval = (
        np.load(cache_dir / "sessionE_bp.npz")["bp"].astype(np.float32, copy=False) if include_eval else None
    )

    q_npz = np.load(cache_dir / "sessionT_quality.npz")
    q_train = q_npz["q"].astype(np.float32, copy=False)
    artifact_corr_eog = q_npz["artifact_corr_eog"].astype(np.float32, copy=False)
    resid_ratio = q_npz["resid_ratio"].astype(np.float32, copy=False)

    return SubjectData(
        subject=int(subject),
        sfreq=sfreq,
        ch_names=ch_names,
        bands=bands,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        bp_train=bp_train,
        bp_eval=bp_eval,
        q_train=q_train,
        artifact_corr_eog=artifact_corr_eog,
        resid_ratio=resid_ratio,
    )
