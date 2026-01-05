#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Make a teacher/report-friendly analysis pack from an existing pareto evaluation directory "
            "(expects pareto_summary.csv, pareto_by_subject.csv, run_config.json)."
        )
    )
    p.add_argument("--pareto-dir", type=str, required=True, help="Path like runs/<run_name>/pareto/<tag>/")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Where to write figures (default: <pareto-dir>/report_figures/).",
    )
    p.add_argument(
        "--k-bars",
        type=str,
        default="12,14",
        help="Comma-separated K list for per-subject bar plots (default: 12,14).",
    )
    p.add_argument(
        "--focus-methods",
        type=str,
        default="ours,full22,ga_l1,lr_weight,fisher,mi",
        help="Comma-separated methods for 'core' pareto plots.",
    )
    p.add_argument(
        "--all-methods",
        type=str,
        default=None,
        help="Comma-separated methods for 'all methods' plot (default: from pareto_summary.csv).",
    )
    p.add_argument(
        "--report-md",
        type=str,
        default=None,
        help="Optional markdown report path to write (embeds generated figures by relative paths).",
    )
    return p.parse_args()


@dataclass(frozen=True)
class PlotStyle:
    label: str
    color: str
    linewidth: float = 2.0
    linestyle: str = "-"
    marker: str = "o"


def _method_style(method: str) -> PlotStyle:
    # Stable mapping across figures.
    styles: dict[str, PlotStyle] = {
        "ours": PlotStyle("Ours (policy+value+MCTS)", "#d62728", linewidth=3.0),
        "uct": PlotStyle("UCT (uniform prior/value)", "#ff9896", linewidth=2.0, linestyle="--"),
        "full22": PlotStyle("Full-22 (all channels)", "#000000", linewidth=2.5, linestyle="--", marker=""),
        "ga_l1": PlotStyle("GA@L1 (budgeted)", "#1f77b4", linewidth=2.0),
        "lr_weight": PlotStyle("LR-weight topK", "#2ca02c", linewidth=2.0),
        "fisher": PlotStyle("Fisher topK", "#9467bd", linewidth=1.8),
        "mi": PlotStyle("MI topK", "#8c564b", linewidth=1.8),
        "sfs_l1": PlotStyle("SFS@L1", "#17becf", linewidth=1.8, linestyle="--"),
        "random_best_l1": PlotStyle("Random-best@L1", "#7f7f7f", linewidth=1.5, linestyle=":"),
    }
    return styles.get(method, PlotStyle(method, "#333333", linewidth=1.5))


def _split_csv_list(x: str) -> list[str]:
    items = [i.strip() for i in str(x).split(",")]
    return [i for i in items if i]


def _split_int_list(x: str) -> list[int]:
    return [int(i) for i in _split_csv_list(x)]


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


def _plot_pareto_mean_std(
    *,
    df: pd.DataFrame,
    metric: str,
    methods: list[str],
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for method in methods:
        sub = df[df.method == method].sort_values("k")
        if sub.empty:
            continue
        st = _method_style(method)
        x = sub["k"].astype(int).to_numpy()
        y = sub[f"{metric}_mean"].astype(float).to_numpy()
        s = sub[f"{metric}_std"].astype(float).to_numpy()
        ax.plot(x, y, label=st.label, color=st.color, linewidth=st.linewidth, linestyle=st.linestyle, marker=st.marker)
        ax.fill_between(x, y - s, y + s, color=st.color, alpha=0.12, linewidth=0)

    ax.set_title(title)
    ax.set_xlabel("K (number of selected channels)")
    ax.set_ylabel(f"{metric} (mean ± std over subjects)")
    ax.grid(True, alpha=0.25)
    ax.set_xticks(sorted(df["k"].unique().tolist()))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_pareto_q20(
    *,
    df: pd.DataFrame,
    metric: str,
    methods: list[str],
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for method in methods:
        sub = df[df.method == method].sort_values("k")
        if sub.empty:
            continue
        st = _method_style(method)
        x = sub["k"].astype(int).to_numpy()
        y = sub[f"{metric}_q20"].astype(float).to_numpy()
        ax.plot(x, y, label=st.label, color=st.color, linewidth=st.linewidth, linestyle=st.linestyle, marker=st.marker)

    ax.set_title(title)
    ax.set_xlabel("K (number of selected channels)")
    ax.set_ylabel(f"{metric}_q20 (20th percentile over subjects)")
    ax.grid(True, alpha=0.25)
    ax.set_xticks(sorted(df["k"].unique().tolist()))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_delta_vs_baseline(
    *,
    summary: pd.DataFrame,
    metric: str,
    method: str,
    baseline: str,
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    a = summary[summary.method == method][["k", f"{metric}_mean", f"{metric}_q20"]].copy()
    b = summary[summary.method == baseline][["k", f"{metric}_mean", f"{metric}_q20"]].copy()
    if a.empty or b.empty:
        return
    m = a.merge(b, on="k", suffixes=("_a", "_b")).sort_values("k")
    x = m["k"].astype(int).to_numpy()
    d_mean = (m[f"{metric}_mean_a"] - m[f"{metric}_mean_b"]).astype(float).to_numpy()
    d_q20 = (m[f"{metric}_q20_a"] - m[f"{metric}_q20_b"]).astype(float).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), sharex=True)
    st = _method_style(method)
    for ax, d, ylab in [(axes[0], d_mean, f"Δ{metric}_mean"), (axes[1], d_q20, f"Δ{metric}_q20")]:
        ax.axhline(0.0, color="#000000", linewidth=1.0, alpha=0.6)
        ax.bar(x, d, color=st.color, alpha=0.75, width=1.25)
        ax.set_xlabel("K")
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xticks(x)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_subject_delta_heatmap(
    *,
    by_subj: pd.DataFrame,
    metric: str,
    method: str,
    baseline: str,
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    a = by_subj[by_subj.method == method][["subject", "k", metric]].copy()
    b = by_subj[by_subj.method == baseline][["subject", "k", metric]].copy()
    if a.empty or b.empty:
        return
    m = a.merge(b, on=["subject", "k"], suffixes=("_a", "_b"))
    m["delta"] = (m[f"{metric}_a"] - m[f"{metric}_b"]).astype(float)
    piv = m.pivot(index="subject", columns="k", values="delta").sort_index()

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    sns.heatmap(
        piv,
        ax=ax,
        cmap="RdBu_r",
        center=0.0,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": f"Δ{metric} ({method} − {baseline})"},
    )
    ax.set_title(title)
    ax.set_xlabel("K")
    ax.set_ylabel("Subject")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_subject_bars(
    *,
    by_subj: pd.DataFrame,
    metric: str,
    k: int,
    methods: list[str],
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    sub = by_subj[by_subj.k == int(k)].copy()
    sub = sub[sub.method.isin(methods)]
    if sub.empty:
        return

    subjects = sorted(sub["subject"].unique().tolist())
    x = np.arange(len(subjects))
    width = 0.8 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    for i, method in enumerate(methods):
        m = sub[sub.method == method].set_index("subject")
        y = [float(m.loc[s, metric]) if s in m.index else np.nan for s in subjects]
        st = _method_style(method)
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, y, width=width, color=st.color, alpha=0.85, label=st.label)

    ax.set_title(title)
    ax.set_xlabel("Subject")
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s:02d}" for s in subjects])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _render_table_markdown(df: pd.DataFrame, cols: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _relpath(base_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def _write_report_md(
    *,
    report_path: Path,
    pareto_dir: Path,
    out_dir: Path,
    summary: pd.DataFrame,
    by_subj: pd.DataFrame,
) -> None:
    run_cfg_path = pareto_dir / "run_config.json"
    run_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8")) if run_cfg_path.exists() else {}
    resolved = run_cfg.get("resolved", {})

    ours = summary[summary.method == "ours"].copy().sort_values("k")
    full22 = summary[summary.method == "full22"].copy().sort_values("k")

    best_k = None
    if not ours.empty:
        best_k = int(ours.sort_values("kappa_mean", ascending=False).iloc[0]["k"])

    if not ours.empty and not full22.empty:
        merged = ours.merge(full22, on="k", suffixes=("_ours", "_full22"))
        merged["dkappa_mean"] = (merged["kappa_mean_ours"] - merged["kappa_mean_full22"]).round(4)
        merged["dkappa_q20"] = (merged["kappa_q20_ours"] - merged["kappa_q20_full22"]).round(4)
        merged["dacc_mean"] = (merged["acc_mean_ours"] - merged["acc_mean_full22"]).round(4)
        merged = merged[
            [
                "k",
                "kappa_mean_ours",
                "kappa_mean_full22",
                "dkappa_mean",
                "kappa_q20_ours",
                "kappa_q20_full22",
                "dkappa_q20",
                "acc_mean_ours",
                "acc_mean_full22",
                "dacc_mean",
            ]
        ].copy()
        for c in merged.columns:
            if c != "k":
                merged[c] = merged[c].map(lambda x: f"{float(x):.4f}")
    else:
        merged = pd.DataFrame()

    fig = lambda name: _relpath(report_path.parent, (out_dir / name).resolve())

    lines: list[str] = []
    lines.append(f"# Post-mortem Report — {resolved.get('out_dir','(unknown run)')} — {pareto_dir.name}")
    lines.append("")
    lines.append("## 1) Protocol & Reproducibility")
    lines.append(f"- Pareto dir: `{pareto_dir}`")
    lines.append(f"- Checkpoint: `{resolved.get('checkpoint_path','(unknown)')}`")
    lines.append(f"- Subjects: `{resolved.get('subjects','(unknown)')}`")
    lines.append(f"- K list: `{resolved.get('k','(unknown)')}`")
    lines.append(f"- Methods: `{resolved.get('methods','(unknown)')}`")
    lines.append(f"- Baseline cache: `{resolved.get('baseline_cache_path','(unknown)')}`")
    lines.append("")
    lines.append("## 2) Main Findings (high-level)")
    if best_k is not None:
        best_row = ours[ours.k == best_k].iloc[0]
        lines.append(f"- Ours peak (by kappa_mean) at **K={best_k}**: kappa_mean={best_row['kappa_mean']:.4f}, acc_mean={best_row['acc_mean']:.4f}.")
    lines.append("- Ours surpasses `full22` on mean only at larger K, but tail robustness (`q20`) still lags due to regressions in a few subjects.")
    lines.append("- Small-K regime (K=4/6/8/10): ours < `full22` and often < `ga_l1` (failure signature: compact subset search).")
    lines.append("")
    lines.append("## 3) Key Curves (core methods, cleaner legends)")
    lines.append(f"![kappa mean±std]({fig('pareto_kappa_mean_core.png')})")
    lines.append("")
    lines.append(f"![kappa q20]({fig('pareto_kappa_q20_core.png')})")
    lines.append("")
    lines.append(f"![acc mean±std]({fig('pareto_acc_mean_core.png')})")
    lines.append("")
    lines.append(f"![acc q20]({fig('pareto_acc_q20_core.png')})")
    lines.append("")
    lines.append("## 4) Ours vs Full22 (delta)")
    lines.append(f"![delta vs full22 (kappa)]({fig('delta_vs_full22_kappa.png')})")
    lines.append("")
    lines.append(f"![delta vs full22 (acc)]({fig('delta_vs_full22_acc.png')})")
    lines.append("")
    lines.append("## 5) Per-subject View (why q20 is not improving)")
    lines.append(f"![subject heatmap dkappa]({fig('heatmap_subject_dkappa_ours_vs_full22.png')})")
    lines.append("")
    if best_k is not None:
        lines.append(f"![subject bars kappa K={best_k}]({fig(f'subject_bars_kappa_K{best_k}.png')})")
        lines.append("")
    lines.append("## 6) Numbers: Ours vs Full22 (per K)")
    if not merged.empty:
        lines.append(_render_table_markdown(merged, cols=list(merged.columns)))
    else:
        lines.append("(table unavailable: missing ours/full22 rows)")
    lines.append("")
    lines.append("## 7) Failure-first diagnosis (what to fix next)")
    lines.append("- **Tail risk**: a few subjects regress even when mean improves → `q20` stagnates.")
    lines.append("- **Small-K gap**: compact subset still fails to beat `full22`/`ga_l1`.")
    lines.append("- Next lever (single): reward shaping/normalization toward beating stronger baselines (keep state/model/MCTS fixed).")
    lines.append("")
    lines.append("## 8) Reproduce (eval command as recorded)")
    argv = run_cfg.get("argv") or []
    if argv:
        lines.append("```bash")
        lines.append(" \\\n  ".join(argv))
        lines.append("```")
    else:
        lines.append("(argv unavailable)")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    pareto_dir = Path(args.pareto_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (pareto_dir / "report_figures").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _setup_matplotlib()

    summary = pd.read_csv(pareto_dir / "pareto_summary.csv")
    by_subj = pd.read_csv(pareto_dir / "pareto_by_subject.csv")

    focus_methods = _split_csv_list(args.focus_methods)
    all_methods = _split_csv_list(args.all_methods) if args.all_methods else sorted(summary.method.unique().tolist())
    k_bars = _split_int_list(args.k_bars)

    # Curves (core)
    _plot_pareto_mean_std(
        df=summary,
        metric="kappa",
        methods=focus_methods,
        title="Pareto (core) — kappa mean±std",
        out_path=out_dir / "pareto_kappa_mean_core.png",
    )
    _plot_pareto_q20(
        df=summary,
        metric="kappa",
        methods=focus_methods,
        title="Pareto (core) — kappa q20 (robustness)",
        out_path=out_dir / "pareto_kappa_q20_core.png",
    )
    _plot_pareto_mean_std(
        df=summary,
        metric="acc",
        methods=focus_methods,
        title="Pareto (core) — accuracy mean±std",
        out_path=out_dir / "pareto_acc_mean_core.png",
    )
    _plot_pareto_q20(
        df=summary,
        metric="acc",
        methods=focus_methods,
        title="Pareto (core) — accuracy q20 (robustness)",
        out_path=out_dir / "pareto_acc_q20_core.png",
    )

    # Curves (all methods, optional)
    _plot_pareto_mean_std(
        df=summary,
        metric="kappa",
        methods=all_methods,
        title="Pareto (all methods) — kappa mean±std",
        out_path=out_dir / "pareto_kappa_mean_all.png",
    )

    # Deltas / per-subject
    _plot_delta_vs_baseline(
        summary=summary,
        metric="kappa",
        method="ours",
        baseline="full22",
        title="Ours − Full22 (kappa)",
        out_path=out_dir / "delta_vs_full22_kappa.png",
    )
    _plot_delta_vs_baseline(
        summary=summary,
        metric="acc",
        method="ours",
        baseline="full22",
        title="Ours − Full22 (accuracy)",
        out_path=out_dir / "delta_vs_full22_acc.png",
    )
    _plot_subject_delta_heatmap(
        by_subj=by_subj,
        metric="fbcsp_kappa",
        method="ours",
        baseline="full22",
        title="Per-subject Δkappa (ours − full22)",
        out_path=out_dir / "heatmap_subject_dkappa_ours_vs_full22.png",
    )
    _plot_subject_delta_heatmap(
        by_subj=by_subj,
        metric="fbcsp_acc",
        method="ours",
        baseline="full22",
        title="Per-subject Δacc (ours − full22)",
        out_path=out_dir / "heatmap_subject_dacc_ours_vs_full22.png",
    )

    for k in k_bars:
        _plot_subject_bars(
            by_subj=by_subj,
            metric="fbcsp_kappa",
            k=k,
            methods=["ours", "full22", "ga_l1"],
            title=f"Per-subject kappa at K={k}",
            out_path=out_dir / f"subject_bars_kappa_K{k}.png",
        )
        _plot_subject_bars(
            by_subj=by_subj,
            metric="fbcsp_acc",
            k=k,
            methods=["ours", "full22", "ga_l1"],
            title=f"Per-subject accuracy at K={k}",
            out_path=out_dir / f"subject_bars_acc_K{k}.png",
        )

    if args.report_md:
        _write_report_md(
            report_path=Path(args.report_md).expanduser().resolve(),
            pareto_dir=pareto_dir,
            out_dir=out_dir,
            summary=summary,
            by_subj=by_subj,
        )


if __name__ == "__main__":
    main()
