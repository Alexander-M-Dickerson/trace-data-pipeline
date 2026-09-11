"""contract.py -- the panel's published column list, frozen.

`main_panel_<mode>.parquet` is a public artifact. Its column NAMES and their ORDER are part
of what is published: people index into it positionally, diff it against last year's vintage,
and write code against the names. Nothing in this repository pinned either of those --
`wrangle.reorder_panel_cols` fixes a ~31-name prefix and everything after it fell out of
merge order, so a change to the beta model list could silently permute the panel.

That is not hypothetical. Collapsing the two degenerate DEF and TERM regressions into the
single two-factor model that produces a real default beta also swapped `b_defb` and `b_termb`
in the output -- emission follows the model's `keep` order. Same 140 columns, different file.
Nobody would have noticed until a downstream positional read broke.

So the list below is the contract, and `assert_panel_contract` enforces it at the end of the
build. It is deliberately a literal: a list generated from the panel could not catch the case
it exists for.

CHANGING THIS LIST IS A PUBLIC API CHANGE. Adding a column, removing one, or moving one is a
CHANGELOG entry and a new vintage -- not a silent edit. Update it in the same commit as the
code that changes the panel, and say why in the commit message.

What this does NOT check: dtypes, row counts, the index, or whether any column is populated.
`validate_coverage.py` covers the last of those.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

PANEL_COLUMNS: tuple[str, ...] = (
    "cusip", "date", "issuer_cusip", "permno", "permco", "gvkey", "144a", "country",
    "call", "ret_vw", "ret_vw_bgn", "hprd", "lib", "libd", "ret_type", "spc_rat",
    "mdc_rat", "ff17num", "ff30num", "fce_val", "mcap_s", "mcap_e", "tret", "rfret",
    "dt_s", "dt_e", "dt_s_bgn", "dt_e_bgn", "hprd_bgn", "igap_bgn", "sig_dt", "sig_gap",
    "tmat", "age", "ytm", "cs", "md_dur", "convx", "bbtm", "sze", "val_hz", "val_hz_dts",
    "val_ipr", "val_ipr_dts", "dcs6", "cs_mu12_1", "pi", "ami", "ami_v", "lix", "ilq",
    "roll", "spd_abs", "spd_rel", "cs_sprd", "ar_sprd", "p_zro", "p_fht", "vov", "dvol",
    "dskew", "dkurt", "db_mkt", "dvol_sys", "dvol_idio", "rvol", "rsj", "rsk", "rkt",
    "b_vix", "b_dvixd", "b_mktrf_mkt", "b_mktb_mkt", "ivol_mkt", "ivol_bbw",
    "b_mktbx_dcapm", "b_term_dcapm", "b_dvix_va", "b_dvix_vp", "ivol_vp", "b_psb_m",
    "b_amd_m", "b_dvix", "b_cpi_vol6", "b_dunc", "b_unc", "b_dunc3", "b_dunc6", "b_duncr",
    "b_duncf", "b_dcredit", "b_credit", "b_dcpi", "b_cptlt", "b_rvol", "b_rsj", "b_psb",
    "b_amd", "b_illiq", "b_dvix_dn", "b_dvix_up", "b_coskew", "iskew", "b_defb", "b_termb",
    "b_drf", "b_crf", "b_lrf", "b_mktb", "b_lvl", "b_ysp", "b_mktb_dn", "b_mktb_up",
    "b_epu", "b_epum", "b_eput", "sysmom3_1", "sysmom6_1", "sysmom12_1", "idimom3_1",
    "idimom6_1", "idimom12_1", "mom3_1", "mom6_1", "mom9_1", "mom12_1", "mom12_7",
    "ltr48_12", "ltr30_6", "ltr24_3", "imom1", "imom3_1", "imom12_1", "iltr48_12",
    "iltr30_6", "iltr24_3", "var_90", "es_90", "var_95", "str",
)


# ---------------------------------------------------------------------------
# The three shipped panels: what each is called, and where each one ENDS
# ---------------------------------------------------------------------------
# Three panels are produced from this contract. They do not share a frontier, and treating
# them as if they did is a mistake with a specific shape: panel 3 ends when ICE ends, so
# waiting for it to extend with a new TRACE vintage means waiting for ever.
#
#   tracks_trace_vintage -- moves every year with the WRDS pull (panels 1 and 2)
#   fixed_source_end     -- pinned; its source stopped and is not coming back (panel 3)
#
# ❗A `fixed_source_end` panel still REBUILDS. Its trigger is a change to the factor panel or
# to its own inputs, never a new TRACE vintage. Those are different questions and conflating
# them is how a panel gets rebuilt for no reason, or not rebuilt when it should be.

PRE_TRACE_FIXED_END = "2023-01-31"      # ICE/BAML ends here. Moving this is a deliberate act.

TRACKS_TRACE_VINTAGE = "tracks_trace_vintage"
FIXED_SOURCE_END = "fixed_source_end"

DATASETS: dict[str, dict] = {
    "trace": dict(
        n=1,
        what="TRACE-era monthly panel",
        stem="osbap_trace",
        published=True,                 # OSBAP, and REDACTED -- see make_release.redact_for_publication
        frontier=TRACKS_TRACE_VINTAGE,
        fixed_end=None,
    ),
    "combined": dict(
        n=2,
        what="Lehman-ICE joined to TRACE",
        stem="lehman_ice_trace",
        published=False,                # collaborators only: its firm ids derive from private data
        frontier=TRACKS_TRACE_VINTAGE,
        fixed_end=None,
    ),
    "pre_trace": dict(
        n=3,
        what="Lehman-ICE only",
        stem="lehman_ice",
        published=False,
        frontier=FIXED_SOURCE_END,
        fixed_end=PRE_TRACE_FIXED_END,
    ),
}

RETURN_TAGS = ("std", "dur_adj")


def external_name(dataset: str, return_type: str, first, last) -> str:
    """The delivered filename: dataset, coverage, return type.

    The span comes from the DATA, so it rolls on its own each vintage. Before this, the two
    private panels' external names existed only as labels in make_data_dictionary.py and no
    file with either name was ever written -- the dictionary described files nobody received.

        external_name("combined", "standard", 1973-01-31, 2025-11-30)
            -> "lehman_ice_trace_1973_2025_std.parquet"
    """
    import pandas as pd

    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; known: {sorted(DATASETS)}")
    tag = "dur_adj" if return_type in ("duration_adj", "dur_adj") else "std"
    y0, y1 = pd.Timestamp(first).year, pd.Timestamp(last).year
    return f"{DATASETS[dataset]['stem']}_{y0}_{y1}_{tag}.parquet"


def assert_frontier_policy(df, dataset: str, *, date_col: str = "date") -> None:
    """A fixed-end panel must still end where its source stopped."""
    import pandas as pd

    spec = DATASETS[dataset]
    if spec["frontier"] != FIXED_SOURCE_END:
        return
    got = pd.Timestamp(pd.to_datetime(df[date_col]).max()).normalize()
    want = pd.Timestamp(spec["fixed_end"])
    if got != want:
        raise AssertionError(
            f"{dataset}: the frontier moved to {got.date()}, expected {want.date()}.\n"
            "  This panel ends where its source ends; it does not extend with a TRACE\n"
            "  vintage. If a genuinely new source vintage extends it, change\n"
            "  contract.PRE_TRACE_FIXED_END deliberately, in the same commit, and say why."
        )


def assert_panel_contract(df, *, what: str = "main panel") -> None:
    """Raise unless `df` carries exactly PANEL_COLUMNS, in exactly that order."""
    actual = list(df.columns)
    expected = list(PANEL_COLUMNS)
    if actual == expected:
        return

    missing = [c for c in expected if c not in actual]
    added = [c for c in actual if c not in expected]
    problems = []
    if missing:
        problems.append(f"  {len(missing)} column(s) MISSING: {missing}")
    if added:
        problems.append(f"  {len(added)} column(s) ADDED: {added}")
    if not missing and not added:
        moved = [(i, e, a) for i, (e, a) in enumerate(zip(expected, actual)) if e != a]
        shown = "; ".join(f"position {i}: expected {e!r}, got {a!r}" for i, e, a in moved[:6])
        problems.append(
            f"  the same {len(expected)} columns, but {len(moved)} are REORDERED -- {shown}")

    detail = "\n".join(problems)
    raise AssertionError(
        f"{what} does not match the published column contract "
        f"({len(actual)} columns vs {len(expected)} expected).\n"
        f"{detail}\n\n"
        "  The panel is a published artifact: the names AND their order are part of it.\n"
        "  If this change is intended, edit lib/contract.py in the same commit and record\n"
        "  it in the CHANGELOG. If it is not, something reordered the panel by accident."
    )
