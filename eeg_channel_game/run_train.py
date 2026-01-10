from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
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


def _read_git_commit(repo_root: Path) -> str | None:
    git_dir = repo_root / ".git"
    head = git_dir / "HEAD"
    if not head.exists():
        return None
    ref = head.read_text(encoding="utf-8").strip()
    if not ref:
        return None
    if not ref.startswith("ref:"):
        return ref
    ref_name = ref.split(":", 1)[1].strip()
    ref_path = git_dir / ref_name
    if ref_path.exists():
        sha = ref_path.read_text(encoding="utf-8").strip()
        return sha or None
    packed = git_dir / "packed-refs"
    if not packed.exists():
        return None
    for line in packed.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if (not line) or line.startswith("#") or line.startswith("^"):
            continue
        try:
            sha, name = line.split(" ", 1)
        except ValueError:
            continue
        if name.strip() == ref_name and sha:
            return sha.strip()
    return None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    cfg.setdefault("_meta", {})
    cfg["_meta"].update(
        {
            "argv": [str(x) for x in sys.argv],
            "config_path": str(args.config),
            "overrides": list(args.override or []),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _read_git_commit(Path(__file__).resolve().parents[1]),
        }
    )
    train(cfg)


if __name__ == "__main__":
    main()
