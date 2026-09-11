"""mua_engines.py -- the method-uncertainty (MUA) grid, in both engines.

Section 5 asks how much of a published bond-factor premium is a choice rather than a
finding. The MUA grid answers it by re-running one signal under every defensible
combination of method choices and reading off the spread of premia:

    weighting            EW, VW                                   2
    portfolios           terciles, quintiles, deciles             3
    breakpoint universe  all bonds, large bonds only, IG only     3
    rating filter        all, IG, HY                              3
    maturity bucket      all, 0-5y, 5-10y, 10y+                   4
                                                              = 216 specs

times 108 signals -- ignoring the 24 infeasible cells per signal where an
investment-grade breakpoint universe is crossed with a high-yield rating filter.

Two engines produce it:

  `run_slow`  the definition: per breakpoint universe, SingleSort +
              AssayAnomalyRunner, save_idx=True, turnover=True,
              dynamic_weights=True. Correct, and slow enough that the full grid
              is a multi-hour job.
  `run_fast`  `assay_anomaly_fast` handed the identical grid, which returns the
              same surface in seconds.

Both emit CANONICAL spec coordinates (weighting, nport, bp_universe, rating,
maturity), so the two are comparable without either grammar leaking outward.

Three rules are enforced here rather than assumed:
  * `pblenv.use()` runs BEFORE any PyBondLab import, so the engine is known;
  * `spc_rat` is cast to float64 -- a nullable Int breaks numba;
  * (date, cusip) uniqueness is checked before any fit -- a duplicate silently
    corrupts the fast path instead of raising.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _p in (str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths                # noqa: E402
import pblenv               # noqa: E402

DATE_START, DATE_END = "2002-08-01", "2024-12-31"
PANEL_COLS = ["cusip", "date", "ret_vw", "mcap_e", "spc_rat", "tmat"]
NPORT_CODE = {3: "Tp", 5: "Qp", 10: "Dp"}


def bp_universe_funcs() -> dict:
    """The three breakpoint universes: every bond, large bonds only, IG only."""
    def large_bonds_only(df: pd.DataFrame) -> pd.Series:
        col = "VW" if "VW" in df.columns else "mcap_e"
        return df[col] > 100

    def ig_bonds_only(df: pd.DataFrame) -> pd.Series:
        col = "RATING_NUM" if "RATING_NUM" in df.columns else "spc_rat"
        return (df[col] >= 1) & (df[col] <= 10)

    return {"all": None, "lg_bp": large_bonds_only, "ig_bp": ig_bonds_only}


def load_panel(signals: list[str], end: str | None = None) -> pd.DataFrame:
    """The bond-month panel, restricted to the MUA sample and the columns needed.

    Read through DuckDB and column-projected, so a worker process pulls its own six
    columns plus one signal rather than inheriting the whole panel.

    `end=None` leaves the series untruncated, which is the default everywhere in
    Stage 3: the window is applied at the statistics layer. Sorting is per-date, so a
    month inside the window is unaffected by panel months after it.
    """
    import duckdb

    p = Path(paths.PANEL).as_posix()
    cols = ", ".join(f'"{c}"' for c in PANEL_COLS + list(signals))
    where = f"WHERE date >= DATE '{DATE_START}'"
    if end:
        where += f" AND date <= DATE '{end}'"
    df = duckdb.sql(
        f"SELECT {cols} FROM read_parquet('{p}') {where}").df()
    df["date"] = pd.to_datetime(df["date"])
    df["spc_rat"] = df["spc_rat"].astype("float64")        # nullable Int breaks numba
    dup = df.duplicated(["date", "cusip"]).sum()
    if dup:
        raise AssertionError(f"{dup} duplicate (date, cusip) rows -- PyBondLab's "
                             f"fast path silently corrupts on these (open bug D1)")
    return df


def run_slow(data: pd.DataFrame, signal: str,
             universes: tuple[str, ...] = ("all", "lg_bp", "ig_bp")) -> pd.DataFrame:
    """The definitional slow path. Returns the long frame: one row per
    (date, leg, canonical spec coordinates) with return / nbonds / turnover."""
    pblenv.use()
    from PyBondLab.StrategyClass import SingleSort
    from PyBondLab.AnomalyAssayer import AssayAnomalyRunner

    funcs = bp_universe_funcs()
    frames = []
    for bp_name in universes:
        strategy = SingleSort(sort_var=signal, holding_period=1, num_portfolios=5,
                              breakpoint_universe_func=funcs[bp_name], verbose=False)
        runner = AssayAnomalyRunner(
            strategy=strategy, data=data, holding_periods=[1],
            nport=[3, 5, 10], ratings=[None, "IG", "NIG"],
            subset_filter={"tmat": [(0, float("inf")), (0, 5), (5, 10),
                                    (10, float("inf"))]},
            breakpoint_universe_func=funcs[bp_name],
            save_idx=True, turnover=True, dynamic_weights=True,
            verbose=False, n_jobs=1,
            IDvar="cusip", DATEvar="date", RETvar="ret_vw", Wvar="mcap_e",
            RATINGvar="spc_rat")
        results = runner.run()
        out = results.runs.reset_index().rename(columns={"index": "date"})
        out["bp_universe"] = bp_name
        frames.append(out)
    long = pd.concat(frames, ignore_index=True)
    long["weighting"] = long["weight"]
    long["nport"] = long["Sort"].map(_nport_code)
    long["rating"] = long["Rating"].map(_rating_code)
    long["maturity"] = long["Subset"].map(_maturity_code)
    long["leg"] = long["type"]
    return long


def _nport_code(sort_val) -> str:
    """'10'/'10p' -> 'Dp'; the runner's Sort column holds 'T'/'Q'/'D' letters."""
    s = str(sort_val).replace("_", "").rstrip("p")
    if s.isdigit():
        return NPORT_CODE[int(s)]
    if s in ("T", "Q", "D"):
        return s + "p"
    raise ValueError(f"unrecognised Sort value {sort_val!r}")


def _rating_code(r) -> str:
    if r is None or (isinstance(r, float) and np.isnan(r)) or str(r).lower() in ("none", "all"):
        return "all"
    return {"IG": "ig", "NIG": "hy", "HY": "hy"}.get(str(r), str(r).lower())


def _maturity_code(subset) -> str:
    s = str(subset)
    if "0-5" in s:
        return "short"
    if "5-10" in s:
        return "mid"
    if "10-inf" in s:
        return "long"
    return "all"


def fast_specs() -> dict:
    """The same 216-cell grid, in assay_anomaly_fast's grammar."""
    return {
        "weighting": ["EW", "VW"],
        "portfolio_structures": [(3, "Q", None), (5, "Q", None), (10, "Q", None)],
        "rating_filters": {"all": None, "ig": "IG", "hy": "NIG"},
        "bp_universes": bp_universe_funcs(),
        # ❗maturity "all" is (0, inf), NOT None. Even the "all" bucket excludes a
        # bond whose time to maturity is non-positive or missing at formation, and
        # passing None would quietly include them in one cell of 216.
        "maturity_filters": {"all": (0, float("inf")), "short": (0, 5),
                             "mid": (5, 10), "long": (10, float("inf"))},
    }


def run_fast(data: pd.DataFrame, signal: str, verbose: bool = False,
             return_legs: bool = False, return_counts: bool = False,
             return_formation_counts: bool = False):
    """assay_anomaly_fast over the identical grid.

    Returns (wide LS frame with canonical tuple columns, canon map, the result
    object). `return_legs` / `return_counts` add the long and short legs and their
    realised bond counts; `return_formation_counts` adds the counts as of FORMATION,
    so formation minus realised is the month's attrition -- which is what Table IA
    on portfolio size reports.
    """
    pblenv.use()
    from PyBondLab.anomaly_assay_fast import assay_anomaly_fast

    extra = {}
    if return_legs or return_counts:
        extra = {"return_legs": return_legs, "return_counts": return_counts}
    if return_formation_counts:
        extra["return_formation_counts"] = True
    res = assay_anomaly_fast(
        data, signal, fast_specs(), holding_period=1, dynamic_weights=True,
        validate=False, skip_invalid=True, verbose=verbose,
        IDvar="cusip", DATEvar="date", RETvar="ret_vw", VWvar="mcap_e",
        RATINGvar="spc_rat", **extra)
    wide = res.returns_df.copy()
    canon = {}
    for col in wide.columns:                    # {W}_{n}p_Q_{bp}_{rat}_{mat}
        w, npt, _q, rest = col.split("_", 3)
        for bp in ("ig_bp", "lg_bp", "all"):
            if rest.startswith(bp + "_"):
                rat, mat = rest[len(bp) + 1:].split("_", 1)
                canon[col] = (w, NPORT_CODE[int(npt.rstrip('p'))], bp, rat, mat)
                break
    return wide, canon, res
