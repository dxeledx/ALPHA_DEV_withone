from __future__ import annotations

import argparse
from pathlib import Path

from eeg_channel_game.eeg.prepare_bciiv2a import prepare_subject_bciiv2a
from eeg_channel_game.eeg.variant import variant_from_cfg
from eeg_channel_game.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare BCI-IV 2a (MOABB BNCI2014_001) cached data")
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--override",
        action="extend",
        nargs="+",
        default=[],
        help="YAML override(s) like mcts.n_sim=256 (repeatable)",
    )
    p.add_argument("--subjects", type=str, default=None, help="Comma-separated list, e.g. 1,2,3 (override)")
    p.add_argument("--no-cov", action="store_true", help="Skip cov_fb computation (faster)")
    p.add_argument("--variant", type=str, default=None, help="Dataset/cache variant name (override)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)

    data_cfg = cfg["data"]
    bands = [tuple(map(float, b)) for b in cfg["features"]["bands"]]
    subjects = data_cfg.get("subjects", [1])
    if args.subjects:
        subjects = [int(s.strip()) for s in args.subjects.split(",") if s.strip()]

    variant = args.variant or variant_from_cfg(cfg)
    data_root = Path("eeg_channel_game") / "data"
    for subject in subjects:
        print(f"[prepare] subject={subject:02d} variant={variant}")
        prepare_subject_bciiv2a(
            subject=int(subject),
            data_root=data_root,
            variant=str(variant),
            fmin=float(data_cfg["fmin"]),
            fmax=float(data_cfg["fmax"]),
            tmin_rel=float(data_cfg["tmin_rel"]),
            tmax_rel=float(data_cfg["tmax_rel"]),
            bands=bands,
            use_eog_regression=bool(data_cfg.get("use_eog_regression", True)),
            compute_cov=(not args.no_cov),
        )

    print("[prepare] done")


if __name__ == "__main__":
    main()
