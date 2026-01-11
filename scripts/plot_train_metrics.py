#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training diagnostics from runs/<run>/train_metrics.csv")
    p.add_argument("--run-dir", type=str, required=True, help="Path like runs/<run_name>/")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Where to write figures (default: <run-dir>/figures/train_diagnostics/).",
    )
    return p.parse_args()


def _setup_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: F401

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _plot_curves(
    *,
    df: pd.DataFrame,
    x: str,
    series: list[tuple[str, str]],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    for col, label in series:
        if col not in df.columns:
            continue
        y = df[col].astype(float).to_numpy()
        if np.all(~np.isfinite(y)):
            continue
        ax.plot(df[x].astype(int).to_numpy(), y, label=label, linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    csv_path = run_dir / "train_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "figures" / "train_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    _setup_matplotlib()
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"Empty metrics file: {csv_path}")
    df = df.sort_values("iter").reset_index(drop=True)

    _plot_curves(
        df=df,
        x="iter",
        series=[
            ("reward_mean", "reward_mean"),
            ("reward_q20", "reward_q20"),
            ("reward_best", "reward_best"),
        ],
        title="Reward over training",
        ylabel="reward",
        out_path=out_dir / "reward.png",
    )

    _plot_curves(
        df=df,
        x="iter",
        series=[
            ("leaf_value_mix_alpha", "leaf_value_mix_alpha"),
            ("policy_prior_eta", "policy_prior_eta"),
            ("teacher_weight", "teacher_weight"),
        ],
        title="Training schedules (bootstrap/prior/teacher)",
        ylabel="value",
        out_path=out_dir / "schedules.png",
    )

    _plot_curves(
        df=df,
        x="iter",
        series=[
            ("kappa_robust_mean", "kappa_robust_mean"),
            ("kappa_mean_mean", "kappa_mean_mean"),
            ("kappa_q20_mean", "kappa_q20_mean"),
            ("acc_mean_mean", "acc_mean_mean"),
        ],
        title="Classifier metrics proxy over training (from evaluator info)",
        ylabel="metric",
        out_path=out_dir / "metrics_proxy.png",
    )

    _plot_curves(
        df=df,
        x="iter",
        series=[
            ("train_loss_total", "loss_total"),
            ("train_loss_pi", "loss_pi"),
            ("train_loss_v", "loss_v"),
            ("train_loss_teacher", "loss_teacher"),
        ],
        title="Training losses",
        ylabel="loss",
        out_path=out_dir / "losses.png",
    )

    _plot_curves(
        df=df,
        x="iter",
        series=[
            ("train_policy_entropy", "policy_entropy"),
            ("train_grad_norm", "grad_norm"),
            ("pi_entropy_mean", "selfplay_pi_entropy_mean"),
        ],
        title="Optimization + search entropy",
        ylabel="value",
        out_path=out_dir / "entropy_and_grad.png",
    )

    _plot_curves(
        df=df,
        x="iter",
        series=[
            ("n_ch_mean", "n_ch_mean"),
            ("stop_frac", "stop_frac"),
            ("b_max_mean", "b_max_mean"),
        ],
        title="Self-play behavior",
        ylabel="value",
        out_path=out_dir / "selfplay_behavior.png",
    )

    _plot_curves(
        df=df,
        x="iter",
        series=[
            ("time_selfplay_s", "time_selfplay_s"),
            ("time_train_s", "time_train_s"),
            ("time_iter_s", "time_iter_s"),
        ],
        title="Iteration wall-time breakdown",
        ylabel="seconds",
        out_path=out_dir / "time_breakdown.png",
    )

    print(f"[ok] wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
