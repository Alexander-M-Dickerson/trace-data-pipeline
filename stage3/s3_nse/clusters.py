"""clusters.py -- the 108-signal to 9-cluster map for Section 5, and the grid grammars.

❗The cluster map here is NOT the same as `zoo_engine.CLUSTERS`. They differ on
`b_rvol`, deliberately: Section 5's exhibits are built on this grouping and reassigning
that one signal moves four of its cells. Keep them separate; do not "unify" them.

Also owns the two spec-grammar parsers, because both grids' exhibits need them:

  * the MUA spec-id grammar, {EW|VW}_{Tp|Qp|Dp}_Q_{bp}_{rating}_{maturity}, with the
    full redundancy rule -- the twin specs AND the infeasible investment-grade-
    breakpoint x high-yield set. Both halves matter: excluding only the twins
    overcounts the grid.
  * the DUA filter_config grammar, where a price screen encodes its tail by the VALUE
    RANGE rather than by a sign, unlike trim and bounce.
"""
from __future__ import annotations

import pandas as pd


SIGNAL_GROUPS = {
    1: {"name": "Spreads, Yields, Size",
        "signals": ["tmat", "age", "ytm", "cs", "md_dur", "convx", "sze", "dcs6", "cs_mu12_1"]},
    2: {"name": "Value",
        "signals": ["bbtm", "val_hz", "val_hz_dts", "val_ipr", "val_ipr_dts"]},
    3: {"name": "Momentum & Reversal",
        "signals": ["mom3_1", "mom6_1", "mom9_1", "mom12_1", "mom12_7",
                    "sysmom3_1", "sysmom6_1", "sysmom12_1",
                    "idimom3_1", "idimom6_1", "idimom12_1",
                    "imom1", "imom3_1", "imom12_1",
                    "ltr24_3", "ltr30_6", "ltr48_12",
                    "iltr24_3", "iltr30_6", "iltr48_12", "str"]},
    4: {"name": "Illiquidity",
        "signals": ["pi", "ami", "ami_v", "roll", "ilq", "spd_rel", "spd_abs",
                    "cs_sprd", "ar_sprd", "p_zro", "p_fht", "vov", "lix"]},
    5: {"name": "Volatility & Risk",
        "signals": ["dvol", "dskew", "dkurt", "rvol", "rsj", "rsk", "rkt",
                    "var_90", "var_95", "es_90",
                    "dvol_sys", "dvol_idio", "ivol_mkt", "ivol_bbw", "ivol_vp", "iskew"]},
    6: {"name": "Market Risk",
        "signals": ["b_mktrf_mkt", "b_mktb_mkt", "b_mktb", "b_mktbx_dcapm", "b_term_dcapm",
                    "b_mktb_dn", "b_mktb_up", "b_termb", "db_mkt"]},
    7: {"name": "Credit & Default Risk",
        "signals": ["b_drf", "b_crf", "b_lrf", "b_defb"]},
    8: {"name": "Volatility & Liquidity Risk",
        "signals": ["b_dvix", "b_dvix_va", "b_dvix_vp", "b_dvix_dn", "b_dvix_up",
                    "b_psb", "b_psb_m", "b_amd_m", "b_amd",
                    "b_coskew", "b_vix", "b_dvixd", "b_illiq"]},
    9: {"name": "Macro & Other Risk",
        "signals": ["b_dunc", "b_duncr", "b_duncf", "b_unc", "b_dunc3", "b_dunc6",
                    "b_dcpi", "b_cpi_vol6", "b_dcredit", "b_credit",
                    "b_cptlt", "b_rvol", "b_rsj", "b_lvl", "b_ysp",
                    "b_epu", "b_epum", "b_eput"]},
}

ALL_SIGNALS = [s for g in SIGNAL_GROUPS.values() for s in g["signals"]]
CLUSTER_SIZES = tuple(len(g["signals"]) for g in SIGNAL_GROUPS.values())
GROUP_NAMES = [SIGNAL_GROUPS[i]["name"] for i in range(1, 10)]
_SIGNAL_TO_GROUP = {s: num for num, g in SIGNAL_GROUPS.items() for s in g["signals"]}

# derived, never typed: Table 5's printed N_paths are these x 648
assert len(ALL_SIGNALS) == 108, f"expected 108 signals, got {len(ALL_SIGNALS)}"
assert len(set(ALL_SIGNALS)) == 108, "duplicate signal in the cluster map"
assert CLUSTER_SIZES == (9, 5, 21, 13, 16, 9, 4, 13, 18), CLUSTER_SIZES


def get_signal_group(signal: str) -> int | None:
    return _SIGNAL_TO_GROUP.get(signal)


def get_group_name(num: int) -> str:
    return SIGNAL_GROUPS.get(num, {}).get("name", f"Group {num}")


def add_groups(df: pd.DataFrame, signal_col: str = "signal") -> pd.DataFrame:
    df = df.copy()
    df["group"] = df[signal_col].map(_SIGNAL_TO_GROUP)
    df["group_name"] = df["group"].map(lambda g: get_group_name(g) if pd.notna(g) else "Unknown")
    return df


# ---------------------------------------------------------------------------
# the MUA spec-id grammar
# ---------------------------------------------------------------------------
def exclude_redundant(df: pd.DataFrame, twin: str, col: str = "spec_id") -> pd.DataFrame:
    """The FULL redundancy rule: drop one member of the all_ig == ig_bp_ig twin pair
    AND the 24 infeasible ig_bp x hy specs (PyBondLab emits those as all-NaN rows,
    invisible to the degenerate check -- this mask is the only thing removing them).

    The twin pair is the same portfolio reached two ways: filtering to investment
    grade, or drawing the breakpoints on an investment-grade universe that is then
    filtered to investment grade. Which member to keep is a LABELLING choice and
    nothing else -- both give 168 specs per signal with identical content.
    `twin='mar14'` keeps ig_bp_ig; `twin='feb'` keeps all_ig, which is the labelling
    the paper prints.
    """
    s = df[col]
    if twin == "mar14":
        mask = s.str.contains("_all_ig_", regex=False)
    elif twin == "feb":
        mask = s.str.contains("_ig_bp_ig_", regex=False)
    else:
        raise ValueError(f"unknown twin convention: {twin!r} (mar14|feb)")
    mask = mask | s.str.contains("_ig_bp_hy_", regex=False)
    return df[~mask]


def parse_spec_id_cols(df: pd.DataFrame, col: str = "spec_id") -> pd.DataFrame:
    """Add weighting_p / nport / bp_scheme / bp_universe / rating / maturity columns."""
    df = df.copy()

    def one(spec: str):
        p = spec.split("_")
        if len(p) == 6:
            bp, rating, mat = p[3], p[4], p[5]
        elif len(p) == 7:                       # bp_universe is 'ig_bp' or 'lg_bp'
            bp, rating, mat = p[3] + "_" + p[4], p[5], p[6]
        else:
            bp = rating = mat = None
        return p[0], p[1], p[2], bp, rating, mat

    parsed = df[col].map(one)
    for i, name in enumerate(["weighting_p", "nport", "bp_scheme",
                              "bp_universe", "rating", "maturity"]):
        df[name] = parsed.map(lambda t, i=i: t[i])
    return df


def apply_sign_correction(df: pd.DataFrame, baseline_spec: str,
                          cols_to_flip: tuple = ("mean_ret", "t_stat", "alpha", "tstat_alpha"),
                          ) -> pd.DataFrame:
    """Flip ALL statistics of signals whose baseline-spec mean return is negative.

    ❗There are TWO baseline specs in this section and that is deliberate, not a slip:
    the premium tables sign off VW_Qp_Q_all_all_all (42 signals flip), while the
    figures and the sign-flip discussion sign off VW_Dp_Q_all_all_all. Pass the one
    the exhibit uses; do not assume a default.
    """
    df = df.copy()
    base = df[df["spec_id"] == baseline_spec].set_index("signal")["mean_ret"]
    flip = base[base.notna() & (base < 0)].index
    m = df["signal"].isin(flip)
    for c in cols_to_flip:
        if c in df.columns:
            df.loc[m, c] *= -1
    return df


# ---------------------------------------------------------------------------
# the DUA filter_config grammar
# ---------------------------------------------------------------------------
def parse_filter_config(config: str) -> tuple[str, str | None]:
    """(filter_type, location) from a DUA filter_config string.

    Rating suffixes _ALL/_IG/_NIG are stripped first. trim/bounce encode the tail
    by SIGN (negative = left, positive = right, two values = both); price encodes
    it by VALUE RANGE (2..20 are lower bounds = left, 150..285 upper = right) --
    a sign-based parser silently misclassifies all 20 single-bound price configs.
    """
    for suf in ("_ALL", "_IG", "_NIG"):
        if config.endswith(suf):
            config = config[: -len(suf)]
            break
    if config == "baseline" or config.startswith("baseline"):
        return "baseline", None
    for ftype in ("trim", "price", "bounce", "wins"):
        if config.startswith(f"{ftype}_"):
            vals = [float(x) for x in config[len(ftype) + 1:].split("_")]
            if len(vals) == 2:
                return ftype, "both"
            v = vals[0]
            if ftype == "price":                # tail encoded by the value range
                return ftype, "left" if v < 100 else "right"
            return ftype, "left" if v < 0 else "right"
    return "other", None
