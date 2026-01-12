from __future__ import annotations


def make_default_variant(
    *,
    fmin: float,
    fmax: float,
    tmin_rel: float,
    tmax_rel: float,
    use_eog_regression: bool,
    include_eog: bool,
) -> str:
    # BNCI2014_001 interval is [2, 6]; convert to absolute seconds for readability.
    abs_tmin = 2.0 + float(tmin_rel)
    abs_tmax = 2.0 + float(tmax_rel)

    include_eog = bool(include_eog)
    use_eog_regression = bool(use_eog_regression) and include_eog

    use_eog = int(use_eog_regression)
    variant = f"f{float(fmin):g}-{float(fmax):g}_t{abs_tmin:g}-{abs_tmax:g}_eog{use_eog}"

    # Backward compatible: only add a suffix when we explicitly exclude EOG channels.
    # (Existing cached variants like "..._eog0" must keep their original names.)
    if not include_eog:
        variant += "_eogch0"
    return variant


def variant_from_cfg(cfg: dict) -> str:
    data_cfg = cfg["data"]
    if "variant" in data_cfg and data_cfg["variant"]:
        return str(data_cfg["variant"])
    include_eog = bool(data_cfg.get("include_eog", True))
    return make_default_variant(
        fmin=float(data_cfg["fmin"]),
        fmax=float(data_cfg["fmax"]),
        tmin_rel=float(data_cfg["tmin_rel"]),
        tmax_rel=float(data_cfg["tmax_rel"]),
        use_eog_regression=bool(data_cfg.get("use_eog_regression", True)),
        include_eog=include_eog,
    )
