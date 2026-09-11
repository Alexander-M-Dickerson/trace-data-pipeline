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
                            # portfolio-size exhibit
SUMMARY_COLS = ["mean_ret", "t_stat", "p_value", "alpha", "tstat_alpha", "n_obs",
                "signal", "spec_id", "group", "group_name", "mean_ret_ann", "alpha_ann"]


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


def summarize_chunk(item) -> dict:
    """TOP-LEVEL worker: stats + nbonds aggregates for a signal chunk."""
    signals, window = item
    end = window_end(window)
    mktb = load_mktb()
    d = grid_dir()

    stat_rows, nb_frames = [], []
    for signal in signals:
        df = pd.read_parquet(d / f"{signal}.parquet")
        df["date"] = pd.to_datetime(df["date"])
        if end:
            df = df[df["date"] <= pd.Timestamp(end)]

        ls = df[df["leg"] == "LS"]
        for spec, g in ls.groupby("spec_id", sort=True):
            row = _stats(g.set_index("date")["return"].dropna(), mktb)
            row["signal"], row["spec_id"] = signal, spec
            stat_rows.append(row)

        # ❗Count only over each series' ACTIVE window -- from its first non-missing
        # return onward. The kernel emits a count of 0 for every month BEFORE a
        # late-starting signal exists, and treating those as empty portfolios would
        # make every late signal look degenerate. An empty month INSIDE the window
        # survives this filter, which is the degeneracy the exhibit reports.
        first = (df[df["return"].notna()]
                 .groupby(["signal", "spec_id", "leg"])["date"].min()
                 .rename("first_valid"))
        act = df.join(first, on=["signal", "spec_id", "leg"])
        act = act[act["date"] >= act["first_valid"]].copy()
        act["_low"] = act["nbonds"].lt(LOW_BOND_THRESHOLD).where(act["nbonds"].notna())
        g = act.groupby(["signal", "spec_id", "leg"])
        nb = g["nbonds"].agg(["mean", "median", "min"])
        nb["p05"] = g["nbonds"].quantile(0.05)
        nb["pct_low"] = g["_low"].mean() * 100
        nb_frames.append(nb.reset_index())

    summary = pd.DataFrame(stat_rows)
    summary = C.add_groups(summary)
    summary["mean_ret_ann"] = summary["mean_ret"] * 12 * 100
    summary["alpha_ann"] = summary["alpha"] * 12 * 100
    return {"summary": summary[SUMMARY_COLS],
            "nbonds": pd.concat(nb_frames, ignore_index=True)}


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
        results = {}
        for w in args.windows:
            with b.phase(f"stats-{w}"):
                n_w, n_t = sized(len(chunks), args.workers, None, min_threads=2)
                parts = pmap(summarize_chunk, [(c, w) for c in chunks],
                             workers=min(n_w, len(chunks)), threads=n_t)
                summary = pd.concat([p["summary"] for p in parts], ignore_index=True)
                nbonds = pd.concat([p["nbonds"] for p in parts], ignore_index=True)
                assert len(summary) == 108 * 216, len(summary)
                summary.to_parquet(out / f"mua_summary_{w}.parquet", index=False)
                nbonds.to_parquet(out / f"mua_nbonds_{w}.parquet", index=False)
                results[w] = summary
        t_by_window = {w: int(s_["n_obs"].max()) for w, s_ in results.items()}
        b.note(windows=list(args.windows), n_rows=108 * 216,
               T_max_by_window=t_by_window)
        ok = b.check(
            all(len(s_) == 108 * 216 and s_["mean_ret"].notna().sum() > 18_000
                for s_ in results.values()),
            f"{len(results)} window(s) x 23,328 rows, T_max {t_by_window}")
        D.write_result(
            "mua_summary",
            {"summary": {"windows": list(args.windows),
                         "T_max_by_window": t_by_window}},
            section="s3_nse", inputs=[paths.BBW], t0=t0,
            extra={"nbonds_convention": "realised"})

    D.mark_complete(out / "_complete.json", ok,
                    {"windows": list(args.windows), "rows_per_window": 108 * 216})
    for w, s_ in results.items():
        print(f"  {w}: T_max={int(s_['n_obs'].max())}, "
              f"{int(s_['mean_ret'].notna().sum()):,} non-NaN of {len(s_):,}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
