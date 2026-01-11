#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two pareto eval dirs (e.g., before vs after expanded search).")
    p.add_argument("--before", type=str, required=True, help="Pareto dir (before), contains pareto_by_subject.csv")
    p.add_argument("--after", type=str, required=True, help="Pareto dir (after), contains pareto_by_subject.csv")
    p.add_argument("--out-dir", type=str, required=True, help="Where to write figures")
    p.add_argument("--method", type=str, default="ours", help="Method to compare (default: ours)")
    p.add_argument("--subjects", type=str, default=None, help="Comma-separated subjects to include (default: all)")
    p.add_argument("--k", type=str, default=None, help="Comma-separated K list to include (default: all)")
    p.add_argument("--title", type=str, default=None, help="Optional figure title prefix")
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


def _split_int_list(x: str | None) -> list[int] | None:
    if x is None:
        return None
    items = [s.strip() for s in str(x).split(",") if s.strip()]
    return [int(s) for s in items] if items else None


def _load(pareto_dir: Path) -> pd.DataFrame:
    path = pareto_dir / "pareto_by_subject.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    # Normalize column names across versions.
    if "fbcsp_kappa" in df.columns:
        df = df.rename(columns={"fbcsp_kappa": "kappa"})
    if "fbcsp_acc" in df.columns:
        df = df.rename(columns={"fbcsp_acc": "acc"})
    return df


def _pivot(df: pd.DataFrame, *, method: str) -> pd.DataFrame:
    df = df[df["method"] == method].copy()
    if df.empty:
        raise RuntimeError(f"No rows for method={method!r}")
    # Each cell should be unique per (subject, k). Keep first robustly.
    out = df.pivot_table(index=["subject"], columns=["k"], values=["kappa", "acc"], aggfunc="first")
    # Flatten columns: kappa_10, acc_10, ...
    flat = {}
    for metric in ("kappa", "acc"):
        if metric not in out.index.get_level_values(0) and metric not in out.columns.get_level_values(0):
            continue
    out.columns = [f"{metric}_{int(k)}" for metric, k in out.columns]
    out = out.reset_index()
    return out


def _heatmap(
    *,
    mat: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    cbar_label: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(7.8, 3.2 + 0.25 * len(row_labels)))
    sns.heatmap(
        mat,
        ax=ax,
        cmap="RdBu_r",
        center=0.0,
        annot=True,
        fmt=".3f",
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={"label": cbar_label},
    )
    ax.set_title(title)
    ax.set_xlabel("K")
    ax.set_ylabel("Subject")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _lineplot(
    *,
    ks: list[int],
    y_before: list[float],
    y_after: list[float],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    ax.plot(ks, y_before, marker="o", linewidth=2.2, label="before (default search)")
    ax.plot(ks, y_after, marker="o", linewidth=2.2, label="after (expanded search)")
    ax.axhline(0.0, color="#000000", linewidth=1.0, alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    before_dir = Path(args.before)
    after_dir = Path(args.after)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    method = str(args.method)
    keep_subjects = _split_int_list(args.subjects)
    keep_k = _split_int_list(args.k)
    title_prefix = str(args.title) if args.title else ""

    _setup_matplotlib()

    df_b = _load(before_dir)
    df_a = _load(after_dir)

    # Filter to method and common rows.
    df_b = df_b[df_b["method"] == method].copy()
    df_a = df_a[df_a["method"] == method].copy()
    if keep_subjects is not None:
        df_b = df_b[df_b["subject"].isin(keep_subjects)]
        df_a = df_a[df_a["subject"].isin(keep_subjects)]
    if keep_k is not None:
        df_b = df_b[df_b["k"].isin(keep_k)]
        df_a = df_a[df_a["k"].isin(keep_k)]

    # Pivot wide: subject × K.
    wide_b = _pivot(df_b, method=method)
    wide_a = _pivot(df_a, method=method)
    merged = wide_b.merge(wide_a, on="subject", how="inner", suffixes=("_before", "_after"))
    if merged.empty:
        raise RuntimeError("No overlapping (subject,k) cells between before/after after filtering.")

    # Determine Ks from columns.
    ks = sorted(
        {
            int(c.split("_")[1])
            for c in merged.columns
            if c.startswith("kappa_") and (c.endswith("_before") or c.endswith("_after")) is False
        }
    )
    # The above may fail because we flatten without suffix. Use original keep_k when provided.
    if keep_k is not None:
        ks = sorted(int(x) for x in keep_k)
    else:
        ks = sorted(int(k) for k in df_b["k"].unique().tolist() if int(k) in set(df_a["k"].unique().tolist()))

    subjects = sorted(int(s) for s in merged["subject"].tolist())
    row_labels = [f"S{int(s):02d}" for s in subjects]
    col_labels = [str(int(k)) for k in ks]

    # Build matrices: after - before.
    mat_kappa = []
    mat_acc = []
    for _, r in merged.sort_values("subject").iterrows():
        dk = []
        da = []
        for k in ks:
            kb = float(r.get(f"kappa_{k}_before", np.nan))
            ka = float(r.get(f"kappa_{k}_after", np.nan))
            ab = float(r.get(f"acc_{k}_before", np.nan))
            aa = float(r.get(f"acc_{k}_after", np.nan))
            dk.append(ka - kb)
            da.append(aa - ab)
        mat_kappa.append(dk)
        mat_acc.append(da)
    mat_kappa = np.asarray(mat_kappa, dtype=np.float64)
    mat_acc = np.asarray(mat_acc, dtype=np.float64)

    # Mean curves.
    mean_before_kappa = []
    mean_after_kappa = []
    mean_before_acc = []
    mean_after_acc = []
    for k in ks:
        mean_before_kappa.append(float(df_b[df_b["k"] == k]["kappa"].mean()))
        mean_after_kappa.append(float(df_a[df_a["k"] == k]["kappa"].mean()))
        mean_before_acc.append(float(df_b[df_b["k"] == k]["acc"].mean()))
        mean_after_acc.append(float(df_a[df_a["k"] == k]["acc"].mean()))

    # Save raw comparison table for reproducibility.
    meta = {
        "before": str(before_dir),
        "after": str(after_dir),
        "method": method,
        "subjects": subjects,
        "k": ks,
        "title": title_prefix,
    }
    (out_dir / "compare_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    t0 = (title_prefix + " — " if title_prefix else "")
    _heatmap(
        mat=mat_kappa,
        row_labels=row_labels,
        col_labels=col_labels,
        title=f"{t0}Δkappa (after - before) for {method}",
        cbar_label="Δkappa",
        out_path=out_dir / "delta_kappa_heatmap.png",
    )
    _heatmap(
        mat=mat_acc,
        row_labels=row_labels,
        col_labels=col_labels,
        title=f"{t0}Δacc (after - before) for {method}",
        cbar_label="Δacc",
        out_path=out_dir / "delta_acc_heatmap.png",
    )

    _lineplot(
        ks=ks,
        y_before=mean_before_kappa,
        y_after=mean_after_kappa,
        title=f"{t0}{method}: mean kappa vs K (before/after expanded search)",
        ylabel="kappa (mean over subjects)",
        out_path=out_dir / "kappa_mean_before_after.png",
    )
    _lineplot(
        ks=ks,
        y_before=mean_before_acc,
        y_after=mean_after_acc,
        title=f"{t0}{method}: mean acc vs K (before/after expanded search)",
        ylabel="acc (mean over subjects)",
        out_path=out_dir / "acc_mean_before_after.png",
    )

    print(f"[ok] wrote: {out_dir}")


if __name__ == "__main__":
    main()

