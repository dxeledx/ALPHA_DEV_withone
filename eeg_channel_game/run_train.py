from __future__ import annotations

import argparse
import sys

from eeg_channel_game.rl.train_loop import train
from eeg_channel_game.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AlphaZero-style channel selection (L0 proxy)")
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--override",
        action="extend",
        nargs="+",
        default=[],
        help="YAML override(s) like train.num_iters=100 (repeatable)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    cfg.setdefault("_meta", {})
    cfg["_meta"].update(
        {
            "argv": [str(x) for x in sys.argv],
            "config_path": str(args.config),
            "overrides": list(args.override or []),
        }
    )
    train(cfg)


if __name__ == "__main__":
    main()
