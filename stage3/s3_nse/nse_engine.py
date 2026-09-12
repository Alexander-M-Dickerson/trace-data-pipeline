"""nse_engine.py -- the Section-5 (non-standard errors) engine.

One load per grid, one statistics pass per exhibit family; the drivers are formatters
and never recompute. The inputs are the two grids Stage 3 produces itself:
`run_dua_grid.py --stats` for data uncertainty, `run_mua_grid.py` then
`mua_summarize.py` for method uncertainty.

The grain here is deliberately ONE ROW PER PATH -- (signal, rating, weighting,
filter_config) for DUA, (signal, spec_id) for MUA -- rather than the (factor, quantity)
shape the other sections use. Collapsing to a per-factor statistic would destroy the
dispersion across paths, which is the entire quantity this section measures.

Four conventions cause most mistakes here, and each is marked at its use:

  1. The DUA statistics are ALREADY sign-corrected at source. `sign_mult_premia` is a
     receipt saying what was done, NOT an instruction to do it again.
  2. The two cluster tables compute Ratio by OPPOSITE rules -- DUA on the pairwise
     matched sample, MUA with independent skipna. They are not interchangeable.
  3. Units: the DUA statistics are already in percent; the MUA summary stores decimals
     and is scaled x100 here, once, at load.
  4. There are TWO sign-correction baselines -- the tables use VW_Qp_Q_all_all_all,
     the figures VW_Dp_Q_all_all_all. Pass the one the exhibit wants.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _p in (str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths                    # noqa: E402
import clusters as C            # noqa: E402

TABLE_BASELINE = "VW_Qp_Q_all_all_all"   # Table 6 / IA.XIX sign + improvement baseline
FLIP_BASELINE = "VW_Dp_Q_all_all_all"    # figures / sign-flip prose baseline (convention 4)
MIN_TSTAT = 1.96
FILTER_TYPES = ("trim", "price", "bounce")
LOCATIONS = ("left", "right", "both")
N_DUA_PATHS = 69_984


# ---------------------------------------------------------------------------
# DUA -- data uncertainty (69,984 filter paths; stored stats are PERCENT)
# ---------------------------------------------------------------------------
DUA_WINDOWS = ("full", "paper")
DUA_GRID_DIR = paths.GRIDS / "dua"


def _dua_file(name: str, window: str) -> Path:
    """One statistics parquet from the DUA grid, for one sample window.

    `paper` truncates each SERIES at the sample end and then computes; `full` leaves
    it untruncated. The two differ by more than a row filter: a sign correction
    decided on a shorter baseline mean can flip.
    """
    if window not in DUA_WINDOWS:
        raise ValueError(f"unknown DUA window {window!r} (full|paper)")
    f = DUA_GRID_DIR / f"dua_{name}_{window}.parquet"
    if not f.exists():
        raise SystemExit(
            f"the DUA statistics layer is not at {f}.\n"
            "  Run `python s3_nse/run_dua_grid.py` then the same file with --stats.")
    return f


def load_dua_paths(window: str = "full") -> pd.DataFrame:
    """One row per (signal, rating, weighting, filter_config): premia + alpha stats.

    ❗The frame INCLUDES the baseline rows (70,632 = 108 signals x 654). Exclude them
    by `filter_type`, never by matching the string 'baseline': there are THREE
    baseline labels (baseline, baseline_IG, baseline_NIG), and a string equality
    leaves 70,416 rows instead of 69,984 -- 432 baselines silently counted as paths.

    Tail location is read from the fit's own configs (dua_config_locations.parquet)
    when that file exists, with the filter_config grammar asserted to agree on every
    filter row; otherwise it is derived from the grammar alone.
    `df.attrs['location_source']` records which.
    """
    prem = pd.read_parquet(_dua_file("premia", window))
    alph = pd.read_parquet(_dua_file("alpha", window))
    key = ["signal", "rating", "weighting", "filter_config"]
    df = prem.merge(alph, on=key, how="outer", validate="1:1")
    # A path whose alpha has fewer than 12 MKTB-joined months drops its alpha row, so
    # the count is bounded rather than pinned -- but it must not be far short.
    assert 69_000 < len(df) <= 70_632, f"DUA rows {len(df)}"
    parsed = df["filter_config"].map(C.parse_filter_config)
    df["filter_type"] = parsed.map(lambda t: t[0])
    df["location"] = parsed.map(lambda t: t[1])
    df = C.add_groups(df)
    n_grid = (df["filter_type"].isin(FILTER_TYPES)).sum()
    df.attrs["n_grid_paths"] = int(n_grid)

    loc_file = DUA_GRID_DIR / "dua_config_locations.parquet"
    if loc_file.exists():
        cfg = pd.read_parquet(loc_file)
        m = cfg.groupby("filter_config")[["filter_type", "location"]].nunique()
        assert (m <= 1).all().all(), "filter_config -> (type, location) not unique"
        lut = cfg.drop_duplicates("filter_config").set_index("filter_config")
        grid = df["filter_type"].isin(FILTER_TYPES)
        for col in ("filter_type", "location"):
            from_cfg = df.loc[grid, "filter_config"].map(lut[col])
            same = (from_cfg == df.loc[grid, col])
            assert same.all(), (
                f"configs vs grammar disagree on {col} for "
                f"{df.loc[grid][~same]['filter_config'].unique()[:5]}")
        df.attrs["location_source"] = "fit configs, grammar asserted equal"
    else:
        df.attrs["location_source"] = "grammar only"
    return df


def load_dua_baselines(window: str = "full") -> pd.DataFrame:
    """648 rows: per (signal, rating, weighting) baseline premia/alpha statistics.

    ❗ALREADY sign-corrected at source (convention 1). `sign_mult_premia` is a receipt
    saying which way it went, not an instruction to flip again. Under window='paper'
    the sign was decided on the truncated baseline mean, so a handful of units can
    carry a different sign from the full-window file -- that is correct, not a bug.
    """
    df = pd.read_parquet(_dua_file("baselines", window))
    assert len(df) == 648, (
        f"the DUA baselines frame has {len(df)} rows, expected 648 "
        "(108 signals x 2 weightings x 3 ratings). Re-run "
        "`python s3_nse/run_dua_grid.py --stats` -- a partial grid gives a short frame.")
    return df


def dua_cluster_summary(paths_df: pd.DataFrame) -> pd.DataFrame:
    """Table 5: NSE by cluster over the 69,984 non-baseline paths.

    Ratio = std(value) / mean(SE), with SE = |value|/|t| guarded at |t| > 0.01, and --
    ❗convention 2 -- computed on the PAIRWISE-dropna matched sample. The MUA cluster
    table uses the opposite rule (independent skipna). Do not unify them.
    """
    df = paths_df[paths_df["filter_type"].isin(FILTER_TYPES)].copy()
    df["se_premia"] = np.where(df["tstat_premia"].abs() > 0.01,
                               df["premia"].abs() / df["tstat_premia"].abs(), np.nan)
    df["se_alpha"] = np.where(df["tstat_alpha"].abs() > 0.01,
                              df["alpha"].abs() / df["tstat_alpha"].abs(), np.nan)
    rows = []
    for cl in list(range(1, 10)) + [None]:
        sub = df if cl is None else df[df["group"] == cl]
        p_vals = sub["premia"].dropna()
        if len(p_vals) < 10:
            continue
        p_matched = sub[["premia", "se_premia"]].dropna()
        ratio_mu = (p_matched["premia"].std() / p_matched["se_premia"].mean()
                    if len(p_matched) >= 10 and p_matched["se_premia"].mean() > 0 else np.nan)
        a_vals = sub["alpha"].dropna()
        a_matched = sub[["alpha", "se_alpha"]].dropna()
        ratio_alpha = (a_matched["alpha"].std() / a_matched["se_alpha"].mean()
                       if len(a_matched) >= 10 and a_matched["se_alpha"].mean() > 0 else np.nan)
        rows.append({
            "cluster_name": "All" if cl is None else C.get_group_name(cl),
            "mu_mean": p_vals.mean(), "mu_median": p_vals.median(),
            "nse_mu": p_vals.quantile(.75) - p_vals.quantile(.25), "ratio_mu": ratio_mu,
            "alpha_mean": a_vals.mean(), "alpha_median": a_vals.median(),
            "nse_alpha": a_vals.quantile(.75) - a_vals.quantile(.25), "ratio_alpha": ratio_alpha,
            "n_paths": len(p_vals),
        })
    return pd.DataFrame(rows)


def dua_improvements(paths_df: pd.DataFrame, baselines_df: pd.DataFrame,
                     min_tstat: float = MIN_TSTAT) -> pd.DataFrame:
    """IA.XVII's improving-path flags, per non-baseline path.

    A filter improves iff t(alpha) > 1.96 AND t(alpha) > the baseline t AND
    alpha > the baseline alpha, per (signal, rating, weighting). All three, not any:
    a filter that raises alpha while lowering its t has not improved the result.
    """
    df = paths_df[paths_df["filter_type"].isin(FILTER_TYPES)].copy()
    df = df.merge(baselines_df[["signal", "rating", "weighting",
                                "baseline_alpha", "baseline_alpha_tstat"]],
                  on=["signal", "rating", "weighting"], how="left", validate="m:1")
    df["improving"] = ((df["tstat_alpha"] > min_tstat)
                       & (df["tstat_alpha"] > df["baseline_alpha_tstat"])
                       & (df["alpha"] > df["baseline_alpha"]))
    return df


def dua_filter_paths_counts(improv_df: pd.DataFrame) -> pd.DataFrame:
    """IA.XVII cells: per (cluster, location, filter_type) the total path count
    (Panel B) and the improving count (Panel C); Panel A is n and round(100n/N)."""
    rows = []
    for gname in C.GROUP_NAMES:
        g = improv_df[improv_df["group_name"] == gname]
        for loc in LOCATIONS:
            for ft in FILTER_TYPES:
                cell = g[(g["location"] == loc) & (g["filter_type"] == ft)]
                n_tot = len(cell)
                n_imp = int(cell["improving"].sum())
                rows.append({"cluster_name": gname, "location": loc, "filter_type": ft,
                             "n_total": n_tot, "n_improving": n_imp,
                             "pct": (100.0 * n_imp / n_tot) if n_tot else np.nan})
    return pd.DataFrame(rows)


def dua_signal_stats(paths_df: pd.DataFrame, value: str) -> pd.DataFrame:
    """Per-signal distribution statistics of `value` over all rows for that signal
    (654 = 648 filter paths + 6 baselines; the baselines are included here).

    `value`: premia | alpha | tstat_premia | tstat_alpha. ❗The 'mean' column is what
    the DUA figures select their top four on -- the MEAN of the LEVEL frame, not the
    t-statistic that is then plotted. The MUA figures select differently again.
    """
    rows = []
    for sig, sub in paths_df.groupby("signal", sort=False):
        v = sub[value].dropna()
        if len(v) < 10:
            continue
        rows.append({
            "signal": sig, "group": C.get_signal_group(sig),
            "group_name": C.get_group_name(C.get_signal_group(sig)),
            "mean": v.mean(), "median": v.median(),
            "q25": v.quantile(.25), "q75": v.quantile(.75),
            "min": v.min(), "max": v.max(),
            "p05": v.quantile(.05), "p95": v.quantile(.95),
            "nse": v.quantile(.75) - v.quantile(.25), "n_paths": len(v),
        })
    return pd.DataFrame(rows)


def dua_baseline_values(baselines_df: pd.DataFrame, col: str) -> dict[str, float]:
    """{signal: mean of `col` across the 6 (rating, weighting) baselines} -- the
    red line in the DUA figures ('averaged across ratings and weightings')."""
    return baselines_df.groupby("signal")[col].mean().to_dict()


# ---------------------------------------------------------------------------
# MUA -- method uncertainty. The analysis set's size depends on how many
# strategies are degenerate in THIS data, so it is reported, never pinned.
# ---------------------------------------------------------------------------
MUA_SUMMARY_DIR = paths.DATA / "s3_nse" / "mua_summary"

# 108 signals x 168 economically distinct constructions: the grid every MUA denominator
# is a subset of. Derived, so a change to the spec grid cannot leave a footnote behind.
GRID_STRATEGIES = 108 * (216 - 24 - 24)


def window_span(window: str) -> tuple[str, str]:
    """(first, last) of a Section-5 reporting window.

    ❗Unlike Sections 3 and 4, this is NOT read off an index: Section 5 summarises
    across construction paths, and the window is a truncation applied to each series at
    the statistics layer. The span is therefore a property of the WINDOW, and
    `mua_summarize` has already truncated to it.
    """
    import _stage3_settings as _S
    if window == "paper":
        return _S.SAMPLE["lib"]["start"], _S.SAMPLE["lib"]["end"]
    lo = _S.SAMPLE["lib"]["start"]
    led = load_ledger(window)
    hi = str(led["last_month"].dropna().max())[:10] or _S.SAMPLE["lib"]["end"]
    return lo, hi


def load_ledger(window: str) -> pd.DataFrame:
    """The status ledger: one row per (signal, spec_id), all 23,328 of them.

    ❗THE single source of truth about which construction paths exist. Every
    denominator Section 5 prints is derived from this frame and nowhere else, because
    the alternative -- each exhibit inferring the answer from whichever artifact it
    happened to read -- is what let Table 6 print 18,064, Table IA.XVIII print 18,038,
    and Table IA.XIX build a pool on the first while IA.XVIII used the second.

    Written by `mua_summarize.py`. See its STATUSES for the vocabulary.
    """
    f = MUA_SUMMARY_DIR / f"mua_status_{window}.parquet"
    if not f.exists():
        raise SystemExit(
            f"the MUA status ledger is not at {f}.\n"
            "  Run `python s3_nse/mua_summarize.py` (it writes the ledger beside the "
            "summary).")
    # ❗Refuse a ledger older than the grid it claims to describe. Re-running the grid
    # does NOT invalidate the summarizer's completion marker, so the orchestrator will
    # happily skip it and leave every Section-5 exhibit built on a statistics layer
    # derived from a different grid. With the engine's cells flipping between runs the
    # two can disagree about which strategies exist at all -- which is exactly what was
    # found on 2026-09-11: six cells with a full 268-month series in the grid and
    # `n_obs = 0` in the summary, seven minutes apart.
    import os
    if os.environ.get("STAGE3_ALLOW_STALE_LEDGER") != "1":
        grids = sorted((paths.GRIDS / "mua").glob("*.parquet"))
        newest = max((g.stat().st_mtime for g in grids), default=0.0)
        if newest > f.stat().st_mtime + 1:
            raise SystemExit(
                f"the MUA status ledger is OLDER than the grid it describes.\n"
                f"  ledger : {f.name}\n"
                f"  grid   : {len(grids)} parquets, newest {newest - f.stat().st_mtime:.0f}s "
                "later\n"
                "  Re-run `python s3_nse/mua_summarize.py`. (Set "
                "STAGE3_ALLOW_STALE_LEDGER=1 to proceed anyway, knowing the exhibits "
                "will describe a grid that is no longer on disk.)")
    led = pd.read_parquet(f)
    assert len(led) == 23_328, (
        f"the ledger has {len(led):,} rows, expected 23,328 (108 signals x 216 specs). "
        "Every cell of the grid gets exactly one status; a short ledger means the "
        "summarizer did not see the whole grid.")
    return led


def usable(window: str, twin: str) -> pd.Index:
    """The (signal, spec_id) pairs that are well-defined factor return series.

    The twin pair is a LABELLING choice -- `feb` keeps `all_ig` (the paper's printed
    labels), `mar14` keeps `ig_bp_ig` -- so the ledger stores both an applied `status`
    (feb) and an `intrinsic_status` (twin-agnostic), and this picks the right one.
    """
    led = load_ledger(window)
    if twin == "feb":
        keep = led["status"] == "ok"
    elif twin == "mar14":
        keep = (led["intrinsic_status"] == "ok") & (led["twin_member"] != "all_ig")
    else:
        raise ValueError(f"unknown twin convention: {twin!r} (mar14|feb)")
    return pd.MultiIndex.from_frame(led.loc[keep, ["signal", "spec_id"]])


def twin_asymmetry(window: str) -> list[str]:
    """Twin pairs where one member is usable and the other is not.

    The two conventions are supposed to be a relabelling: `all_ig` and `ig_bp_ig` are
    the same portfolio reached two ways, so which one you keep cannot move a number.
    That holds only while BOTH members exist. When the engine forms one and not the
    other -- the unstable empty cell, see README_stage3.md -- the conventions select
    different data and every cluster statistic
    moves -- which is what `t06_mua_nse.py`'s invariance check detects.

    Returns the `signal__spec_id` of the usable member of each broken pair, so a failure
    can name the cause instead of just a tolerance.
    """
    led = load_ledger(window).set_index(["signal", "spec_id"])
    out = []
    for (sig, spec), row in led[led["twin_member"] == "all_ig"].iterrows():
        partner = spec.replace("_all_ig_", "_ig_bp_ig_")
        if (sig, partner) not in led.index:
            continue
        a_ok = row["intrinsic_status"] == "ok"
        b_ok = led.loc[(sig, partner), "intrinsic_status"] == "ok"
        if a_ok != b_ok:
            out.append(f"{sig}__{spec if a_ok else partner}")
    return sorted(out)


def load_mua_paths(twin: str, window: str = "paper") -> pd.DataFrame:
    """The MUA analysis set: one row per (signal, spec_id), statistics in PERCENT,
    spec dimensions parsed, clusters attached.

    ❗Sign correction is NOT applied here. Each exhibit applies its own baseline
    (convention 4), and applying one here would make the other exhibit wrong.
    """
    f = MUA_SUMMARY_DIR / f"mua_summary_{window}.parquet"
    if not f.exists():
        raise SystemExit(
            f"the MUA summary is not at {f}.\n"
            "  Run `python s3_nse/run_mua_grid.py` then "
            "`python s3_nse/mua_summarize.py`.")
    df = pd.read_parquet(f)
    assert len(df) == 23_328, (
        f"the MUA summary has {len(df)} rows, expected 23,328 (108 signals x 216 "
        "specs). Re-run `python s3_nse/mua_summarize.py`; a short frame means the "
        "grid behind it is incomplete.")
    led = load_ledger(window)
    keep = usable(window, twin)
    df = df.set_index(["signal", "spec_id"])
    df = df.loc[df.index.isin(keep)].reset_index()
    assert 17_000 < len(df) <= 18_144, f"analysis set {len(df)}"
    # ❗Nothing without a series may be counted as a construction path. The old rule
    # could not see these at all: a cell with no return never reached the bond-count
    # frame the check read, so 26 all-NaN cells sat inside n_paths, contributing nothing
    # to any statistic while inflating every denominator built on it.
    assert int((df["n_obs"] == 0).sum()) == 0, (
        f"{int((df['n_obs'] == 0).sum())} cells with no observations are in the analysis "
        "set. The ledger classifies those `no_series`; they must never reach here.")
    df = df.copy()
    for c in ("mean_ret", "alpha"):
        df[c] = df[c] * 100.0                      # decimals -> percent, once
    df = C.parse_spec_id_cols(df)
    df = C.add_groups(df)
    hist = led["status"].value_counts().astype(int).to_dict()
    df.attrs.update({"window": window, "twin": twin,
                     "n_degenerate": hist.get("empty_leg", 0),
                     "n_no_series": hist.get("no_series", 0),
                     "n_by_status": hist, "n_paths": len(df)})
    return df


def mua_cluster_summary(paths_df: pd.DataFrame) -> pd.DataFrame:
    """Table 6: NSE by cluster over every usable path (baseline specs INCLUDED).

    Sign-corrected on VW_Qp_Q_all_all_all. ❗Ratio uses INDEPENDENT skipna on
    std(value) and mean(SE) -- the OPPOSITE of the DUA cluster table's pairwise
    rule (convention 2).

    ❗`n_paths` here is `len(sub)`, while the DUA table's is `len(premia.dropna())`.
    Both print under the same `$N$` header and they now agree, because the analysis set
    can no longer contain a path with no series -- the ledger classifies those
    `no_series` and they never arrive. Before that they did, and this column printed 26
    more than the sample its own means and quantiles were computed over.

    The path count is NOT pinned to a constant. It follows the data, and while the
    unstable empty cell is
    open it moves between runs; `df.attrs["n_by_status"]` records the whole histogram so
    two runs can be compared.
    """
    signed = C.apply_sign_correction(paths_df, TABLE_BASELINE)
    signed = signed[signed["group"].notna()].copy()
    signed["se_premia"] = np.where(signed["t_stat"].abs() > 0.01,
                                   signed["mean_ret"].abs() / signed["t_stat"].abs(), np.nan)
    signed["se_alpha"] = np.where(signed["tstat_alpha"].abs() > 0.01,
                                  signed["alpha"].abs() / signed["tstat_alpha"].abs(), np.nan)
    rows = []
    for cl in list(range(1, 10)) + [None]:
        sub = signed if cl is None else signed[signed["group"] == cl]
        rows.append({
            "cluster_name": "All" if cl is None else C.get_group_name(cl),
            "mu_mean": sub["mean_ret"].mean(), "mu_median": sub["mean_ret"].median(),
            "nse_mu": sub["mean_ret"].quantile(.75) - sub["mean_ret"].quantile(.25),
            "ratio_mu": sub["mean_ret"].std() / sub["se_premia"].mean(),
            "alpha_mean": sub["alpha"].mean(), "alpha_median": sub["alpha"].median(),
            "nse_alpha": sub["alpha"].quantile(.75) - sub["alpha"].quantile(.25),
            "ratio_alpha": sub["alpha"].std() / sub["se_alpha"].mean(),
            "n_paths": len(sub),
        })
    return pd.DataFrame(rows)


MUA_COLUMN_GROUPS = [
    ("rating", [("All", "all"), ("IG", "ig"), ("NIG", "hy")]),
    ("maturity", [("All", "all"), ("Short", "short"), ("Mid", "mid"), ("Long", "long")]),
    ("bp_universe", [("All", "all"), ("IG", "ig_bp"), ("Large", "lg_bp")]),
]


def mua_improvement_counts(paths_df: pd.DataFrame,
                           expected_denominator: int | None = None) -> pd.DataFrame:
    """The improvement-count cells.

    ❗The two pools are NOT the same, deliberately. The NUMERATOR excludes only the
    single baseline spec, so the other five *_all_all_all twins can count as
    improvements; the DENOMINATOR excludes all six per signal. Using one pool for
    both undercounts.

    ❗The pool is whatever the ledger says is usable, less the six `*_all_all_all`
    per signal. It is DERIVED, not pinned: while the engine's restricted-universe cells
    flip between runs the absolute size moves, but the relationship to Table 6's
    N does not -- and that relationship is what the tests check.

    `expected_denominator` pins the pool size when you know it; pass None to record
    it instead, which is what you want whenever the degenerate set can differ.
    """
    signed = C.apply_sign_correction(paths_df, TABLE_BASELINE)
    base = signed[signed["spec_id"] == TABLE_BASELINE][["signal", "alpha", "tstat_alpha"]]
    base = base.rename(columns={"alpha": "b_alpha", "tstat_alpha": "b_t"})
    cand = signed[signed["spec_id"] != TABLE_BASELINE].merge(base, on="signal",
                                                             validate="m:1")
    improving = cand[(cand["tstat_alpha"] > MIN_TSTAT)
                     & (cand["tstat_alpha"] > cand["b_t"])
                     & (cand["alpha"] > cand["b_alpha"])]
    total = signed[signed["spec_id"] != TABLE_BASELINE]
    total = total[~total["spec_id"].str.endswith("_all_all_all")]
    if expected_denominator is not None:
        assert len(total) == expected_denominator, (
            f"denominator pool {len(total)} != {expected_denominator}")

    rows = []
    for gname in C.GROUP_NAMES + ["Total"]:
        im = improving if gname == "Total" else improving[improving["group_name"] == gname]
        tm = total if gname == "Total" else total[total["group_name"] == gname]
        row = {"cluster_name": gname}
        for dim, vals in MUA_COLUMN_GROUPS:
            for header, val in vals:
                n = int((im[dim] == val).sum())
                t = int((tm[dim] == val).sum())
                key = f"{dim}:{val}"
                row[f"{key}_n"], row[f"{key}_N"] = n, t
                row[f"{key}_pct"] = round(100 * n / t) if t else None
        rows.append(row)
    return pd.DataFrame(rows)


def load_mua_nbonds(twin: str, window: str = "paper") -> pd.DataFrame:
    """The long-short strategy rows of the portfolio-size exhibit, from the per-leg
    bond-count summary.

    ❗108 x 216 x 3 legs is 69,984, which is also the DUA path count. They are
    unrelated numbers that happen to coincide; do not read one as a check on the other.

    ❗Its strategy set is THE SAME SET Table 6 reports, taken from the same ledger.
    It used to re-derive the degeneracy rule from this frame -- a second copy of the
    same nine lines -- which is how the two tables came to print different totals for
    the same quantity (18,038 here against 18,064 there). The frame the rule was
    derived from could not contain a cell with no series, so it silently agreed with
    the right answer for the wrong reason.
    """
    nb = pd.read_parquet(MUA_SUMMARY_DIR / f"mua_nbonds_{window}.parquet")
    keep = usable(window, twin)
    ls = nb[nb["leg"] == "LS"].set_index(["signal", "spec_id"])
    ls = ls.loc[ls.index.isin(keep)].reset_index()
    assert 17_000 < len(ls) <= 18_144, f"strategy set {len(ls)}"
    ls = C.parse_spec_id_cols(ls)
    ls["tailbin"] = pd.cut(ls["mean"], bins=[-np.inf, 200, 600, np.inf],
                           labels=["<200", "200-600", ">600"])
    hist = load_ledger(window)["status"].value_counts().astype(int).to_dict()
    ls.attrs.update({"window": window, "n_degenerate": hist.get("empty_leg", 0),
                     "n_no_series": hist.get("no_series", 0),
                     "n_strategies": len(ls)})
    return ls


def mua_portfolio_size(ls: pd.DataFrame) -> pd.DataFrame:
    """IA.XVIII: 17 rows (3 nport + 3 rating + 4 maturity + 3 BP + 3 tail bins +
    All), each averaging per-strategy time-series stats within the dimension."""
    def agg(sub: pd.DataFrame, label: str) -> dict:
        return {"row": label, "avg": sub["mean"].mean(), "med": sub["median"].mean(),
                "min": sub["min"].mean(), "p05": sub["p05"].mean(),
                "pct_low": sub["pct_low"].mean(), "n_spec": len(sub)}

    rows = [agg(ls[ls["nport"] == v], lbl) for v, lbl in
            [("Tp", "Terciles"), ("Qp", "Quintiles"), ("Dp", "Deciles")]]
    rows += [agg(ls[ls["rating"] == v], lbl) for v, lbl in
             [("all", "Rating: All"), ("ig", "Rating: IG"), ("hy", "Rating: NIG")]]
    rows += [agg(ls[ls["maturity"] == v], lbl) for v, lbl in
             [("all", "Maturity: All"), ("short", "Maturity: Short"),
              ("mid", "Maturity: Intermediate"), ("long", "Maturity: Long")]]
    rows += [agg(ls[ls["bp_universe"] == v], lbl) for v, lbl in
             [("all", "BP: Full universe"), ("ig_bp", "BP: IG bonds"),
              ("lg_bp", "BP: Large bonds")]]
    rows += [agg(ls[ls["tailbin"] == v], f"Tail bin: {v}")
             for v in ["<200", "200-600", ">600"]]
    rows.append(agg(ls, "All specifications"))
    return pd.DataFrame(rows)


def mua_signal_stats(paths_df: pd.DataFrame, value: str) -> pd.DataFrame:
    """Per-signal statistics of `value` over the 168 constructions, under the FIGURE
    sign convention (the Dp baseline -- convention 4). ❗The 'median' column is what
    the MUA figures select their top four on, and it is the median of the
    T-STATISTIC frame -- not the mean of the level, which is what the DUA figures
    use. Four figures, and the two pairs select differently."""
    signed = C.apply_sign_correction(paths_df, FLIP_BASELINE)
    signed = signed[signed["group"].notna()]
    rows = []
    for sig, sub in signed.groupby("signal", sort=False):
        v = sub[value].dropna()
        if len(v) < 10:
            continue
        rows.append({
            "signal": sig, "group": C.get_signal_group(sig),
            "group_name": C.get_group_name(C.get_signal_group(sig)),
            "mean": v.mean(), "median": v.median(),
            "q25": v.quantile(.25), "q75": v.quantile(.75),
            "min": v.min(), "max": v.max(),
            "p05": v.quantile(.05), "p95": v.quantile(.95),
            "nse": v.quantile(.75) - v.quantile(.25), "n_paths": len(v),
        })
    return pd.DataFrame(rows)


def top4_per_cluster(stats_df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Top 4 signals per cluster, sorted descending on `by`.

    ❗The selection rule differs by figure: DUA sorts on the MEAN of the LEVEL frame,
    MUA on the MEDIAN of the T frame. Never reuse one figure's selection for another.
    """
    out = []
    for grp in sorted(stats_df["group"].dropna().unique()):
        sub = stats_df[stats_df["group"] == grp].sort_values(by, ascending=False)
        out.append(sub.head(4))
    return pd.concat(out, ignore_index=True)
