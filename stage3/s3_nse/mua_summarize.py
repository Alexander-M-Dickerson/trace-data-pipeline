r"""mua_summarize.py -- per-(signal, spec) statistics from the stored MUA grid.

The grid (`run_mua_grid.py`) saves UNTRUNCATED monthly series. This applies the sample
window and computes the statistics every Section-5 exhibit reads.

For each of the 108 x 216 long-short series: drop missing months, index-join with MKTB,
then compute BOTH the mean with its Newey-West t (OLS on a constant) and the CAPM_B
alpha with its t (OLS on [1, MKTB]) ON THAT JOINED SAMPLE, lags = int(T**0.25), at
least 12 observations. `n_obs` is the joined T and is re-derived from the window each
time -- never inherited, because joining against a shorter MKTB series silently
shortens it.

All 216 specs per signal get a row. The 24 infeasible investment-grade-breakpoint x
high-yield cells are all-NaN with n_obs 0 rather than absent, so the frame is a
rectangle and a missing spec is a defect rather than a convention. Values stay in
DECIMALS; the annualized columns are the ones in percent.

Also writes the per-leg bond-count summary, in the REALISED convention: how many bonds
the portfolio actually held in the return month. Counts are taken over each series'
ACTIVE window -- from its first non-missing return onward -- so the months before a
late-starting signal exists do not count as months when its portfolio was empty. An
empty month INSIDE the active window does count: that is the degeneracy being measured.

Outputs under data/s3_nse/mua_summary/:
    mua_summary_{window}.parquet   one row per (signal, spec_id)
    mua_nbonds_{window}.parquet    per (signal, spec_id, leg): mean/median/min/p05/pct_low

    python s3_nse/mua_summarize.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

for _p in (str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _stage3_settings as S   # noqa: E402
import clusters as C        # noqa: E402
import paths                # noqa: E402

MIN_OBS = 12
LOW_BOND_THRESHOLD = 20     # a leg holding fewer than this counts as "low" in the
                            # portfolio-size exhibit. ❗THE one definition --
                            # `t18_portfolio_size.py` imports it rather than restating it,
                            # so the printed footnote cannot drift from the column.
SUMMARY_COLS = ["mean_ret", "t_stat", "p_value", "alpha", "tstat_alpha", "n_obs",
                "signal", "spec_id", "group", "group_name", "mean_ret_ann", "alpha_ann"]

# The ledger's vocabulary, in PRECEDENCE order -- first match wins, so the counts are
# disjoint and sum to the full 23,328-cell grid. The first, second and fifth are the
# paper's own words (main.tex, the MUA grid paragraph):
#
#   "Not all are admissible. Breakpoints computed on IG bonds cannot sort NIG bonds
#    (the universes do not overlap). This eliminates 24 specifications per signal.
#    When portfolios are formed, e.g., exclusively within IG bonds, IG-based breakpoints
#    coincide with full-universe breakpoints. This eliminates a further 24 redundant
#    specifications. ... Sixteen specifications (0.088%) produce months with empty long
#    or short legs ... Excluding these leaves 18,128 well-defined factor return series."
#
# `no_series` and `short_sample` are OURS -- the paper's rule presupposes a series, and
# says nothing about one that was never formed or is too short to fit.
STATUSES = ("inadmissible", "redundant", "no_series", "short_sample", "empty_leg", "ok")
STATUS_COLS = ["signal", "spec_id", "status", "intrinsic_status", "n_obs",
               "n_months_grid", "first_month", "last_month", "n_months_active",
               "min_long", "min_short", "pct_low", "twin_member"]


def grid_dir() -> Path:
    return paths.GRIDS / "mua"


def out_dir() -> Path:
    d = paths.section_results("s3_nse") / "mua_summary"
    d.mkdir(parents=True, exist_ok=True)
    return d


def window_end(window: str) -> str | None:
    import drrlib as D
    if window == "paper":
        return D.SAMPLE_END_PAPER
    if window == "full":
        return None
    raise ValueError(f"unknown window {window!r} (paper|full)")


def signal_last_return(d: Path, signals: list[str], end: str | None) -> dict:
    """The last month in which EACH SIGNAL produced a portfolio return, over all its specs.

    ❗Per SIGNAL, never per cell. This is the upper bound of the window inside which a
    strategy is judged, and it has to come from outside the cell being judged: a cell that
    sets its own upper bound can erase the very months that prove it is degenerate.

    Signals do not all span the panel -- some stop early because the underlying data does
    (`b_cptlt` ends 2025-05 against a 2025-11 frontier). Every month after that has no
    bonds, so every leg's minimum is zero, so the whole signal used to be flagged
    degenerate: 168 of its 168 strategies on the full window. A signal's end date is not a
    defect, and this is what stops it being read as one.
    """
    import duckdb
    files = [(d / f"{s_}.parquet").as_posix() for s_ in signals]
    where = f"WHERE date <= DATE '{end}'" if end else ""
    df = duckdb.sql(
        f'SELECT signal, MAX(date) AS last_return FROM read_parquet({files!r}) '
        f'{where} {"AND" if where else "WHERE"} "return" IS NOT NULL '
        "GROUP BY 1").df()
    return {r.signal: pd.Timestamp(r.last_return) for r in df.itertuples()}


def load_mktb() -> pd.Series:
    """MKTB in decimals, tolerant of its being stored as an index or a column."""
    df = pd.read_parquet(paths.BBW)
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    return df["MKTB"]


def _stats(y: pd.Series, mktb: pd.Series) -> dict:
    """Mean + NW t, and CAPM_B alpha + t, both on the MKTB-joined sample."""
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    df = pd.DataFrame({"y": y, "mktb": mktb}).dropna()
    if len(df) < MIN_OBS:
        return {"mean_ret": np.nan, "t_stat": np.nan, "p_value": np.nan,
                "alpha": np.nan, "tstat_alpha": np.nan, "n_obs": 0}
    T = len(df)
    lag = int(T ** 0.25)
    y_vals = df["y"].values
    try:
        m = OLS(y_vals, np.ones((T, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
        mean_ret, t_stat, p_value = m.params[0], m.tvalues[0], m.pvalues[0]
    except Exception:
        mean_ret = t_stat = p_value = np.nan
    try:
        m2 = OLS(y_vals, add_constant(df["mktb"].values)).fit(
            cov_type="HAC", cov_kwds={"maxlags": lag})
        alpha, tstat_alpha = m2.params[0], m2.tvalues[0]
    except Exception:
        alpha = tstat_alpha = np.nan
    return {"mean_ret": mean_ret, "t_stat": t_stat, "p_value": p_value,
            "alpha": alpha, "tstat_alpha": tstat_alpha, "n_obs": T}


def classify(spec_id: str, n_obs: int, n_months_grid: int,
             min_long, min_short, *, apply_redundant: bool = True) -> str:
    """The one place a cell's status is decided. Precedence order; first match wins.

    ❗`inadmissible` and `redundant` come FIRST, so the counts below them describe only
    cells the paper would have expected to exist. That is why `no_series` is 26 and not 42:
    sixteen of the empty cells are the redundant twin, already accounted for.

    ❗`apply_redundant=False` gives the INTRINSIC status -- what the cell is on its own
    merits, ignoring which member of the twin pair a convention happens to keep. The ledger
    stores both, because which twin is redundant is a LABELLING choice: `feb` keeps
    `all_ig` (the paper's printed labels), `mar14` keeps `ig_bp_ig`. Without the intrinsic
    column the second convention cannot be derived, and the twin-invariance check that
    compares them has nothing to compare.
    """
    if "_ig_bp_hy_" in spec_id:
        return "inadmissible"        # IG breakpoints cannot sort NIG -- disjoint universes
    if apply_redundant and "_ig_bp_ig_" in spec_id:
        return "redundant"           # == its _all_ig_ twin; the paper drops one of the pair
    if n_months_grid == 0:
        return "no_series"           # no portfolio was ever formed
    if n_obs == 0:
        return "short_sample"        # a series exists but is too short to fit
    if min_long == 0 or min_short == 0:
        return "empty_leg"           # the paper's rule: a leg empty in some month
    if pd.isna(min_long) or pd.isna(min_short):
        return "no_series"           # a leg that never appears at all
    return "ok"


def summarize_chunk(item) -> dict:
    """TOP-LEVEL worker: statistics, bond-count aggregates and the status ledger."""
    signals, window, sig_last = item
    end = window_end(window)
    mktb = load_mktb()
    d = grid_dir()

    import mua_engines as ME
    specs = ME.all_spec_ids()

    stat_rows, nb_frames, status_rows = [], [], []
    for signal in signals:
        df = pd.read_parquet(d / f"{signal}.parquet")
        df["date"] = pd.to_datetime(df["date"])
        if end:
            df = df[df["date"] <= pd.Timestamp(end)]
        last = pd.Timestamp(sig_last[signal])

        # ---- statistics, over every spec in the fixed grid --------------------
        ls = df[df["leg"] == "LS"]
        grid_months, seen = {}, set()
        for spec, g in ls.groupby("spec_id", sort=True):
            ser = g.set_index("date")["return"].dropna()
            grid_months[spec] = len(ser)
            row = _stats(ser, mktb)
            row["signal"], row["spec_id"] = signal, spec
            stat_rows.append(row)
            seen.add(spec)
        for spec in specs:                      # the grid is a rectangle by construction
            if spec not in seen:
                grid_months[spec] = 0
                stat_rows.append(
                    {"mean_ret": np.nan, "t_stat": np.nan, "p_value": np.nan,
                     "alpha": np.nan, "tstat_alpha": np.nan, "n_obs": 0,
                     "signal": signal, "spec_id": spec})

        # ---- the ACTIVE WINDOW: two-sided -------------------------------------
        # Head: each cell's own first return. A portfolio may legitimately start after
        # its signal does, and the kernel emits a zero count for every month before it
        # exists -- 91,948 of them across the 108 signals. Counting those would make
        # every signal degenerate.
        # Tail: the SIGNAL's last return, never the cell's own. See
        # `signal_last_return`. A cell that sets its own upper bound can delete the
        # months that prove it empty, which is exactly how an all-zero portfolio used
        # to escape this check.
        first = (df[df["return"].notna()]
                 .groupby(["signal", "spec_id", "leg"])["date"].min()
                 .rename("first_valid"))
        act = df.join(first, on=["signal", "spec_id", "leg"])
        # ❗A cell with no return at all has first_valid = NaT and would drop out
        # entirely -- taking its counts, which are present and say zero, with it. Fall
        # back to the signal's own window so the evidence survives.
        act["first_valid"] = act["first_valid"].fillna(act["date"].min())
        act = act[(act["date"] >= act["first_valid"]) & (act["date"] <= last)].copy()

        act["_low"] = act["nbonds"].lt(LOW_BOND_THRESHOLD).where(act["nbonds"].notna())
        g = act.groupby(["signal", "spec_id", "leg"])
        nb = g["nbonds"].agg(["mean", "median", "min"])
        nb["p05"] = g["nbonds"].quantile(0.05)
        nb["pct_low"] = g["_low"].mean() * 100
        nb = nb.reset_index()
        nb_frames.append(nb)

        # ---- the ledger -------------------------------------------------------
        legs = nb.pivot_table(index="spec_id", columns="leg", values="min",
                              aggfunc="first")
        span = (act.groupby("spec_id")["date"].agg(["min", "max", "count"])
                if len(act) else None)
        n_by_spec = {r.spec_id: r.n_obs for r in
                     pd.DataFrame(stat_rows[-len(specs):]).itertuples()}
        for spec in specs:
            ml = legs["L"].get(spec, np.nan) if "L" in legs.columns else np.nan
            ms = legs["S"].get(spec, np.nan) if "S" in legs.columns else np.nan
            n_obs = int(n_by_spec.get(spec, 0))
            n_grid = int(grid_months.get(spec, 0))
            fm = lm = pd.NaT
            n_act = 0
            if span is not None and spec in span.index:
                fm, lm = span.loc[spec, "min"], span.loc[spec, "max"]
                n_act = int(span.loc[spec, "count"] // 3) if span.loc[spec, "count"] else 0
            status_rows.append({
                "signal": signal, "spec_id": spec,
                "status": classify(spec, n_obs, n_grid, ml, ms),
                "intrinsic_status": classify(spec, n_obs, n_grid, ml, ms,
                                             apply_redundant=False),
                "n_obs": n_obs, "n_months_grid": n_grid,
                "first_month": fm, "last_month": lm, "n_months_active": n_act,
                "min_long": ml, "min_short": ms,
                "pct_low": (nb[(nb["spec_id"] == spec) & (nb["leg"] == "LS")]["pct_low"]
                            .iloc[0] if ((nb["spec_id"] == spec)
                                         & (nb["leg"] == "LS")).any() else np.nan),
                "twin_member": ("ig_bp_ig" if "_ig_bp_ig_" in spec
                                else "all_ig" if "_all_ig_" in spec else ""),
            })

    summary = pd.DataFrame(stat_rows)
    summary = C.add_groups(summary)
    summary["mean_ret_ann"] = summary["mean_ret"] * 12 * 100
    summary["alpha_ann"] = summary["alpha"] * 12 * 100
    return {"summary": summary[SUMMARY_COLS],
            "nbonds": pd.concat(nb_frames, ignore_index=True),
            "status": pd.DataFrame(status_rows)[STATUS_COLS]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", nargs="+", default=["paper", "full"],
                    choices=("paper", "full"))
    ap.add_argument("--workers", type=int, default=S.N_WORKERS,
                    help="fresh processes to fan out over (default: from cpu_count)")
    ap.add_argument("--chunk", type=int, default=9, help="signals per worker process")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()

    import drrlib as D          # noqa: E402
    from bench import Bench     # noqa: E402
    from fastrun import pmap, sized    # noqa: E402

    d = grid_dir()
    missing = [s_ for s_ in C.ALL_SIGNALS if not (d / f"{s_}.parquet").exists()]
    if missing:
        raise SystemExit(
            f"ABORT: {len(missing)} signal grids are missing under {d}\n"
            f"  first few: {missing[:5]}\n"
            "  Run `python s3_nse/run_mua_grid.py` first.")
    chunks = [C.ALL_SIGNALS[i_:i_ + args.chunk]
              for i_ in range(0, len(C.ALL_SIGNALS), args.chunk)]
    out = out_dir()

    with Bench("mua-summarize", section="s3_nse") as b:
        results, ledgers = {}, {}
        for w in args.windows:
            with b.phase(f"stats-{w}"):
                # The per-signal upper bound, computed ONCE and handed to every worker --
                # it must not be derived inside the cell being judged. See
                # `signal_last_return`.
                sig_last = signal_last_return(d, list(C.ALL_SIGNALS), window_end(w))
                assert len(sig_last) == len(C.ALL_SIGNALS), (
                    f"{w}: window bounds for {len(sig_last)} of {len(C.ALL_SIGNALS)} "
                    "signals -- a signal with no return anywhere would be judged against "
                    "no window at all.")
                n_w, n_t = sized(len(chunks), args.workers, None, min_threads=2)
                parts = pmap(summarize_chunk, [(c, w, sig_last) for c in chunks],
                             workers=min(n_w, len(chunks)), threads=n_t)
                summary = pd.concat([p["summary"] for p in parts], ignore_index=True)
                nbonds = pd.concat([p["nbonds"] for p in parts], ignore_index=True)
                ledger = pd.concat([p["status"] for p in parts], ignore_index=True)
                assert len(summary) == 108 * 216, (
                    f"{w} window: {len(summary):,} rows, expected "
                    f"{108 * 216:,} (108 signals x 216 specs). The frame is padded to "
                    "the full grid, so a short one means either a worker returned "
                    "fewer signals than it was given, or `mua_engines.all_spec_ids()` "
                    "and the grid's spec grammar have drifted apart.")
                assert len(ledger) == 108 * 216, (
                    f"{w} window: ledger has {len(ledger):,} rows, expected "
                    f"{108 * 216:,}. Every cell of the grid gets exactly one status.")
                summary.to_parquet(out / f"mua_summary_{w}.parquet", index=False)
                nbonds.to_parquet(out / f"mua_nbonds_{w}.parquet", index=False)
                ledger.to_parquet(out / f"mua_status_{w}.parquet", index=False)
                results[w] = summary
                ledgers[w] = ledger
        t_by_window = {w: int(s_["n_obs"].max()) for w, s_ in results.items()}
        hist = {w: l_["status"].value_counts().reindex(STATUSES, fill_value=0)
                            .astype(int).to_dict()
                for w, l_ in ledgers.items()}
        b.note(windows=list(args.windows), n_rows=108 * 216,
               T_max_by_window=t_by_window, status_histogram=hist)
        ok = b.check(
            all(len(s_) == 108 * 216 and s_["mean_ret"].notna().sum() > 18_000
                for s_ in results.values())
            and all(sum(h.values()) == 108 * 216 for h in hist.values()),
            f"{len(results)} window(s) x 23,328 rows, T_max {t_by_window}; "
            + "; ".join(f"{w}: {h['ok']:,} ok, {h['empty_leg']} empty-leg, "
                        f"{h['no_series']} no-series" for w, h in hist.items()))
        D.write_result(
            "mua_summary",
            {"summary": {"windows": list(args.windows),
                         "T_max_by_window": t_by_window},
             "status_histogram": hist},
            section="s3_nse",
            # ❗The GRID is an input, and recording it is what lets a consumer refuse a
            # summary computed from a different one. Re-running the grid does not
            # invalidate this step's completion marker, so without the fingerprint a
            # stale statistics layer is invisible.
            inputs=[paths.BBW] + [d / f"{s_}.parquet" for s_ in C.ALL_SIGNALS],
            t0=t0, extra={"nbonds_convention": "realised"})

    D.mark_complete(out / "_complete.json", ok,
                    {"windows": list(args.windows), "rows_per_window": 108 * 216})
    for w, l_ in ledgers.items():
        h = l_["status"].value_counts().reindex(STATUSES, fill_value=0).astype(int)
        print(f"\n  {w} window, {len(l_):,} cells:")
        for st in STATUSES:
            print(f"     {st:14s} {h[st]:7,}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
