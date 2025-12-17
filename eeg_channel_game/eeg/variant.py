from __future__ import annotations


def make_default_variant(*, fmin: float, fmax: float, tmin_rel: float, tmax_rel: float, use_eog_regression: bool) -> str:
    # BNCI2014_001 interval is [2, 6]; convert to absolute seconds for readability.
    abs_tmin = 2.0 + float(tmin_rel)
    abs_tmax = 2.0 + float(tmax_rel)
    use_eog = int(bool(use_eog_regression))
    return f"f{float(fmin):g}-{float(fmax):g}_t{abs_tmin:g}-{abs_tmax:g}_eog{use_eog}"


def variant_from_cfg(cfg: dict) -> str:
    data_cfg = cfg["data"]
    if "variant" in data_cfg and data_cfg["variant"]:
        return str(data_cfg["variant"])
    return make_default_variant(
        fmin=float(data_cfg["fmin"]),
        fmax=float(data_cfg["fmax"]),
        tmin_rel=float(data_cfg["tmin_rel"]),
        tmax_rel=float(data_cfg["tmax_rel"]),
        use_eog_regression=bool(data_cfg.get("use_eog_regression", True)),
    )

