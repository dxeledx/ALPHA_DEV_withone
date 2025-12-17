from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eeg_channel_game.eeg.fold_stats import FoldStats, compute_fold_stats
from eeg_channel_game.eeg.io import SubjectData, load_subject_data
from eeg_channel_game.eeg.splits import Split, make_stratified_splits


@dataclass(frozen=True)
class FoldData:
    subject: int
    split_id: int
    stats: FoldStats
    split: Split
    subject_data: SubjectData


def _save_fold_stats(path: Path, stats: FoldStats) -> None:
    np.savez_compressed(
        path,
        bp_mean=stats.bp_mean,
        bp_std=stats.bp_std,
        fisher=stats.fisher,
        redund_corr=stats.redund_corr,
        quality_mean=stats.quality_mean,
        artifact_corr_eog=stats.artifact_corr_eog,
        resid_ratio=stats.resid_ratio,
    )


def _load_fold_stats(path: Path) -> FoldStats:
    d = np.load(path)
    return FoldStats(
        bp_mean=d["bp_mean"].astype(np.float32, copy=False),
        bp_std=d["bp_std"].astype(np.float32, copy=False),
        fisher=d["fisher"].astype(np.float32, copy=False),
        redund_corr=d["redund_corr"].astype(np.float32, copy=False),
        quality_mean=d["quality_mean"].astype(np.float32, copy=False),
        artifact_corr_eog=d["artifact_corr_eog"].astype(np.float32, copy=False),
        resid_ratio=d["resid_ratio"].astype(np.float32, copy=False),
    )


class FoldSampler:
    def __init__(
        self,
        *,
        subjects: list[int],
        n_splits: int,
        seed: int,
        variant: str | None = None,
        include_eval: bool = False,
        data_root: str | Path = Path("eeg_channel_game") / "data",
    ):
        self.rng = np.random.default_rng(seed)
        self.n_splits = int(n_splits)
        self.seed = int(seed)
        self.variant = variant
        self.include_eval = bool(include_eval)
        self.data_root = Path(data_root)

        self.subject_data: dict[int, SubjectData] = {}
        self.splits: dict[int, list[Split]] = {}
        self.fold_stats: dict[tuple[int, int], FoldStats] = {}

        for subject in subjects:
            self._load_subject(int(subject))

    def _load_subject(self, subject: int) -> None:
        sd = load_subject_data(subject, data_root=self.data_root, variant=self.variant, include_eval=self.include_eval)
        self.subject_data[subject] = sd
        self.splits[subject] = make_stratified_splits(sd.y_train, n_splits=self.n_splits, seed=self.seed)

        if self.variant is None:
            subj_cache_dir = self.data_root / "cache" / f"subj{subject:02d}"
        else:
            subj_cache_dir = self.data_root / "cache" / str(self.variant) / f"subj{subject:02d}"
        subj_cache_dir.mkdir(parents=True, exist_ok=True)
        for split_id, split in enumerate(self.splits[subject]):
            path = subj_cache_dir / f"foldstats_split{split_id:02d}.npz"
            if path.exists():
                stats = _load_fold_stats(path)
            else:
                stats = compute_fold_stats(
                    bp=sd.bp_train[split.train_idx],
                    q=sd.q_train[split.train_idx],
                    y=sd.y_train[split.train_idx],
                    artifact_corr_eog=sd.artifact_corr_eog,
                    resid_ratio=sd.resid_ratio,
                )
                _save_fold_stats(path, stats)
            self.fold_stats[(subject, split_id)] = stats

    def sample_fold(self) -> FoldData:
        subject = int(self.rng.choice(list(self.subject_data.keys())))
        split_id = int(self.rng.integers(0, self.n_splits))
        stats = self.fold_stats[(subject, split_id)]
        split = self.splits[subject][split_id]
        sd = self.subject_data[subject]
        return FoldData(subject=subject, split_id=split_id, stats=stats, split=split, subject_data=sd)

    def get_fold(self, subject: int, split_id: int) -> FoldData:
        subject = int(subject)
        split_id = int(split_id)
        stats = self.fold_stats[(subject, split_id)]
        split = self.splits[subject][split_id]
        sd = self.subject_data[subject]
        return FoldData(subject=subject, split_id=split_id, stats=stats, split=split, subject_data=sd)
