from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {} if data is None else data


def load_yaml_with_base(path: str | Path, *, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """
    Load a YAML config with optional base/include support.

    A config file may specify:
      base: other.yaml
    or:
      base: [a.yaml, b.yaml]

    Base paths are resolved relative to the current file. Later bases override earlier bases,
    and the current file overrides all bases.
    """
    path = Path(path)
    resolved = path.resolve()
    if resolved in _stack:
        chain = " -> ".join([str(p) for p in _stack + (resolved,)])
        raise ValueError(f"Config base/include cycle detected: {chain}")

    data = load_yaml(path)
    base_spec = data.pop("base", None)
    if base_spec is None:
        return data

    if isinstance(base_spec, (str, Path)):
        base_list = [base_spec]
    elif isinstance(base_spec, list):
        base_list = base_spec
    else:
        raise TypeError(f"{path}: expected 'base' to be a string or list, got {type(base_spec).__name__}")

    merged: dict[str, Any] = {}
    for item in base_list:
        base_path = Path(item)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = deep_update(merged, load_yaml_with_base(base_path, _stack=_stack + (resolved,)))

    merged = deep_update(merged, data)
    return merged


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = load_yaml_with_base(config_path)
    if overrides:
        cfg = deep_update(cfg, parse_overrides(overrides))
    return cfg


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """
    Parse CLI overrides like:
      project.seed=123
      mcts.n_sim=256
      data.subjects=[1,2,3]
    """
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override (missing '='): {item}")
        path, raw = item.split("=", 1)
        keys = path.strip().split(".")
        value = yaml.safe_load(raw)
        cur = out
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = value
    return out


@dataclass(frozen=True)
class RunPaths:
    out_dir: Path
    ckpt_dir: Path
    fig_dir: Path
    log_path: Path


def make_run_paths(out_dir: str | Path) -> RunPaths:
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        out_dir=out_dir,
        ckpt_dir=ckpt_dir,
        fig_dir=fig_dir,
        log_path=out_dir / "train.log",
    )
