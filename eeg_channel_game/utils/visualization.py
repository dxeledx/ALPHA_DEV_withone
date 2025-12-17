from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_channel_mask_topomap(
    *,
    ch_names: list[str],
    mask: np.ndarray,
    sfreq: float,
    title: str,
    save_path: str | Path,
    show_names: bool = False,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import mne
    except Exception as e:  # pragma: no cover
        raise RuntimeError("mne/matplotlib are required for topomap plotting") from e

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mask = np.asarray(mask).astype(np.float32, copy=False)
    info = mne.create_info(ch_names=ch_names, sfreq=float(sfreq), ch_types=["eeg"] * len(ch_names))
    info.set_montage("standard_1005")

    fig, ax = plt.subplots(figsize=(5, 4))
    mne.viz.plot_topomap(
        mask,
        info,
        axes=ax,
        show=False,
        contours=0,
        cmap="viridis",
        sphere=0.09,
        names=ch_names if show_names else None,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
