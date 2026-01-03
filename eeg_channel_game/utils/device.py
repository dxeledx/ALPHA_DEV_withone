from __future__ import annotations

from typing import Any


def resolve_device(device: str | None) -> str:
    """
    Resolve a requested device string to a safe runtime device.

    - If CUDA is requested but unavailable, fall back to "cpu".
    - If "auto" is requested, pick "cuda" when available, else "cpu".

    This makes configs portable across GPU/CPU machines without changing results logic
    (only the execution device changes).
    """
    if device is None:
        requested = "auto"
    else:
        requested = str(device)
    req_l = requested.strip().lower()

    if req_l in {"", "none", "null"}:
        req_l = "auto"

    if req_l == "cpu":
        return "cpu"

    # NOTE: keep "cuda:0" etc when available.
    if req_l.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return requested
        except Exception:
            pass
        return "cpu"

    if req_l in {"auto", "gpu_if_available", "cuda_if_available"}:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    # Unknown device string: return as-is (caller may handle).
    return requested


def normalize_devices_in_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize well-known device fields in a config dict in-place.

    This is intentionally conservative (only touches the project's standard keys).
    """
    project = cfg.setdefault("project", {})
    project["device"] = resolve_device(project.get("device", None))

    eval_cfg = cfg.get("eval", {}) or {}
    pareto_cfg = eval_cfg.get("pareto", {}) or {}
    if "device" in pareto_cfg:
        pareto_cfg["device"] = resolve_device(pareto_cfg.get("device"))

    train_cfg = cfg.get("train", {}) or {}
    sp_cfg = train_cfg.get("selfplay", {}) or {}
    if "device" in sp_cfg and sp_cfg.get("device") not in (None, "null", "None", ""):
        sp_cfg["device"] = resolve_device(sp_cfg.get("device"))

    return cfg

