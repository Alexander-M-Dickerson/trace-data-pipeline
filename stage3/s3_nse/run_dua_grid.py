r"""run_dua_grid.py -- the data-uncertainty (DUA) grid: 108 signals x 120 filters x 3 ratings.

Where the MUA grid varies the METHOD, this varies the DATA: every signal is re-sorted
after each of 120 defensible ways of cleaning the return panel, and the spread of the
resulting premia is the data-uncertainty the section reports.

The 120 filters (see `filters_all`): 48 return-trim thresholds, 30 price screens,
30 bounce-back screens, and 12 winsorization levels. Crossed with 3 rating universes
and 108 signals, with both weightings saved, that is 69,984 filter paths plus their
baselines.

A unit of work is one (rating x signal-chunk). Each worker reads its OWN column slice
of the panel through DuckDB -- the panel is never pickled into a worker -- fits
`DataUncertaintyAnalysis(use_fast_path=True)`, and saves the UNTRUNCATED monthly
ex-ante return series for every filter column, plus the fit's `result.configs` rows
(which carry each filter's tail location, and are what the tail-location exhibit reads).

The fit: num_portfolios=10, holding_period=1, dynamic_weights=True,
include_baseline=True, PRICE = 100/bbtm.

`--stats` then recomputes the per-path statistics from those saved series, on BOTH
windows -- `paper` truncates the SERIES at the sample end and then computes; `full`
leaves it untruncated -- with MKTB from Stage 2's factor file. It writes
dua_{premia,alpha,baselines}_{window}.parquet + dua_config_locations.parquet, which is
what every Section-5 exhibit reads.

❗Needs a PyBondLab build with the fast path. Two passes, in this order:

    python s3_nse/run_dua_grid.py --signals cs      # one signal, to smoke it
    python s3_nse/run_dua_grid.py                   # the grid (hours)
    python s3_nse/run_dua_grid.py --stats           # the statistics layer
"""
from __future__ import annotations

import argparse
import os
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

RATINGS = ("ALL", "IG", "NIG")
WEIGHTINGS = ("ew", "vw")
PANEL_BASE_COLS = ["cusip", "date", "ret_vw", "mcap_e", "spc_rat", "bbtm"]


def filters_all() -> dict:
    """The 120-filter grid: 48 trim + 30 price + 30 bounce + 12 winsorization."""
    wins_levels = [(w, loc) for w in np.arange(98.0, 99.8, 0.5)
                   for loc in ["right", "left", "both"]]
    price_up = list(np.arange(150, 300, 15))
    price_down = list(np.arange(20, 0, -2))
    trim_up = list(np.arange(0.20, 1.00, 0.05))
    trim_down = list(np.arange(-0.95, -0.15, 0.05))
    trim_levels = [round(x, 4) for x in trim_up + trim_down]
    trim_sym = [[round(-w, 4), round(w, 4)] for w in np.arange(0.20, 1.00, 0.05)]
    bounce_up = list(np.arange(-0.10, 0, 0.01))
    bounce_down = list(np.arange(0.01, 0.11, 0.01))
    bounce_levels = [round(x, 4) for x in bounce_up + bounce_down]
    bounce_sym = [[round(-w, 4), round(w, 4)] for w in np.arange(0.01, 0.11, 0.01)]
    return {"trim": trim_levels + trim_sym,
            "price": [price_down, price_up, "zip"],
            "bounce": bounce_levels + bounce_sym,
            "wins": wins_levels}


def out_root() -> Path:
    d = paths.GRIDS / "dua"
    (d / "configs").mkdir(parents=True, exist_ok=True)
    for r in RATINGS:
        (d / "series" / r).mkdir(parents=True, exist_ok=True)
    return d


def run_one(item) -> dict:
    """TOP-LEVEL worker: one (rating x signal-chunk) fit in a fresh process."""
    rating_label, signals, root_s = item
    os.environ.setdefault("NUMBA_NUM_THREADS",
                          os.environ.get("STAGE3_WORKER_THREADS", "4"))
    t0 = time.perf_counter()
    import duckdb
    import pblenv
    pblenv.use()
    from PyBondLab import DataUncertaintyAnalysis

    root = Path(root_s)
    p = Path(paths.PANEL).as_posix()
    cols = ", ".join(f'"{c}"' for c in PANEL_BASE_COLS + list(signals))
    data = duckdb.sql(f"SELECT {cols} FROM read_parquet('{p}')").df()
    data["date"] = pd.to_datetime(data["date"])
    # the price the filters screen on. `bbtm` is 100/price, so this inverts it --
    # NOT a multiply, which would give a price of 100*100/price and screen nothing.
    data["PRICE"] = 100 / data["bbtm"]
    data["spc_rat"] = data["spc_rat"].astype("float64")
    dup = data.duplicated(["cusip", "date"]).sum()
    assert dup == 0, f"{dup} duplicate (cusip, date) rows"
    data = data.sort_values(["cusip", "date"]).reset_index(drop=True)

    rating = None if rating_label == "ALL" else rating_label
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dua = DataUncertaintyAnalysis(
            data=data, signals=list(signals), holding_periods=[1],
            num_portfolios=10, filters=filters_all(), include_baseline=True,
            rating=rating, dynamic_weights=True, verbose=False,
            use_fast_path=True)
        result = dua.fit(IDvar="cusip", RETvar="ret_vw", VWvar="mcap_e",
                         RATINGvar="spc_rat", PRICEvar="PRICE")

    n_cols_saved = 0
    for sig in signals:
        frames = []
        for w in WEIGHTINGS:
            f = getattr(result, f"{w}_ex_ante")
            sig_cols = [c for c in f.columns if c.startswith(f"{sig}_hp1_")]
            long = (f[sig_cols].reset_index(names="date")
                    .melt("date", var_name="col", value_name="ret"))
            long["filter_config"] = long["col"].str.replace(
                f"{sig}_hp1_", "", regex=False)
            long["weighting"] = w
            frames.append(long[["date", "weighting", "filter_config", "ret"]])
            n_cols_saved += len(sig_cols)
        pd.concat(frames, ignore_index=True).to_parquet(
            root / "series" / rating_label / f"{sig}.parquet", index=False)

    cfg = result.configs.copy()
    cfg["filter_config"] = [cn.replace(f"{s}_hp1_", "")
                            for cn, s in zip(cfg["column_name"], cfg["signal"])]
    cfg["rating"] = rating_label
    cfg = cfg[cfg["signal"].isin(signals)]
    cfg["level"] = cfg["level"].astype(str)         # two-sided filters store a list
    cfg["location"] = cfg["location"].astype(str)
    cfg[["rating", "signal", "filter_config", "filter_type", "level", "location"]] \
        .to_parquet(root / "configs" / f"{rating_label}_{signals[0]}.parquet",
                    index=False)
    try:
        import psutil
        rss_gb = round(psutil.Process().memory_info().rss / 2**30, 1)
    except Exception:  # noqa: BLE001
        rss_gb = None
    return {"rating": rating_label, "signals": list(signals),
            "n_series_cols": n_cols_saved, "rss_gb": rss_gb,
            "wall_s": round(time.perf_counter() - t0, 1)}


# ---------------------------------------------------------------------------
# the statistics layer, computed from OUR saved series
# ---------------------------------------------------------------------------
MIN_OBS = 12            # fewer observations than this and the path is skipped


def _nw_mean(returns: pd.Series) -> tuple[float, float]:
    """Mean and Newey-West t: OLS on a constant, HAC lags = floor(T**0.25)."""
    import statsmodels.api as sm
    T = len(returns)
    n_lags = int(np.floor(T ** 0.25))
    res = sm.OLS(returns.values, np.ones(T)).fit(cov_type="HAC",
                                                 cov_kwds={"maxlags": n_lags})
    return float(res.params[0]), float(res.tvalues[0])


def _alpha(y: pd.Series, mktb: pd.Series) -> tuple[float, float]:
    """CAPM_B alpha and its HAC t, on the overlapping sample."""
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    df = pd.DataFrame({"y": y, "mktb": mktb}).dropna()
    if len(df) < MIN_OBS:
        return float("nan"), float("nan")
    lag = int(len(df) ** 0.25)
    try:
        X = add_constant(df["mktb"].values)
        m = OLS(df["y"].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
        return float(m.params[0]), float(m.tvalues[0])
    except Exception:       # noqa: BLE001 -- a degenerate path returns NaN, not a crash
        return float("nan"), float("nan")


def load_mktb() -> pd.Series:
    """MKTB in decimals, tolerant of its being stored as an index or a column."""
    df = pd.read_parquet(paths.BBW)
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    return df["MKTB"]


def stats_chunk(item) -> dict:
    """TOP-LEVEL worker: per-path premia/alphas/baselines for a signal chunk, on one
    window, from the saved series."""
    signals, window, root_s = item
    import drrlib as D
    root = Path(root_s)
    end = None if window == "full" else D.SAMPLE_END_PAPER
    mktb = load_mktb()

    prem_rows, alpha_rows, base_rows = [], [], []
    for signal in signals:
        for r in RATINGS:
            suffix = "" if r == "ALL" else f"_{r}"
            baseline_fc = f"baseline{suffix}"
            long = pd.read_parquet(root / "series" / r / f"{signal}.parquet")
            long["date"] = pd.to_datetime(long["date"])
            for w in WEIGHTINGS:
                sub = (long[long["weighting"] == w]
                       .pivot(index="date", columns="filter_config", values="ret")
                       .sort_index())
                if end:
                    sub = sub.loc[:end]
                assert baseline_fc in sub.columns, (signal, r, w, baseline_fc)
                sign_mult = -1 if sub[baseline_fc].mean() < 0 else 1
                for fc in sub.columns:
                    if "_wins_" in fc:
                        continue
                    series = sub[fc]
                    returns = series.dropna()
                    if len(returns) >= MIN_OBS:
                        mu, t = _nw_mean(returns)
                        prem_rows.append((signal, r, w, fc,
                                          mu * sign_mult * 100, t * sign_mult))
                    a, ta = _alpha(series, mktb)
                    if not np.isnan(a):
                        alpha_rows.append((signal, r, w, fc,
                                           a * sign_mult * 100, ta * sign_mult))
                returns = sub[baseline_fc].dropna()
                if len(returns) >= MIN_OBS:
                    bm = returns.mean()
                    sm_ = -1 if bm < 0 else 1
                    _, bt = _nw_mean(returns * sm_)
                    ar, tar = _alpha(sub[baseline_fc], mktb)
                    base_rows.append(
                        (signal, r, w, bm * sm_ * 100, bt,
                         ar * sm_ * 100 if not np.isnan(ar) else np.nan,
                         tar * sm_ if not np.isnan(ar) else np.nan, sm_))

    return {
        "premia": pd.DataFrame(prem_rows, columns=[
            "signal", "rating", "weighting", "filter_config", "premia", "tstat_premia"]),
        "alpha": pd.DataFrame(alpha_rows, columns=[
            "signal", "rating", "weighting", "filter_config", "alpha", "tstat_alpha"]),
        "baselines": pd.DataFrame(base_rows, columns=[
            "signal", "rating", "weighting", "baseline_premia", "baseline_tstat",
            "baseline_alpha", "baseline_alpha_tstat", "sign_mult_premia"]),
    }


def run_stats(args, root: Path, signals: list[str], b) -> dict:
    """Both windows from the saved series + the concatenated configs table."""
    from fastrun import pmap

    cfg_parts = sorted((root / "configs").glob("*.parquet"))
    cfg = pd.concat([pd.read_parquet(p) for p in cfg_parts], ignore_index=True)
    cfg = cfg.drop_duplicates(["rating", "signal", "filter_config"])
    cfg["level"] = cfg["level"].astype(str)
    cfg["location"] = cfg["location"].astype(str)
    cfg.to_parquet(root / "dua_config_locations.parquet", index=False)

    sigs = sorted(signals)
    chunks = [sigs[i:i + args.chunk] for i in range(0, len(sigs), args.chunk)]
    out = {}
    for window in ("paper", "full"):
        with b.phase(f"stats-{window}"):
            s_w, s_t = sized(len(chunks), args.workers, None, min_threads=2)
            parts = pmap(stats_chunk, [(c, window, str(root)) for c in chunks],
                         workers=min(s_w, len(chunks)), threads=s_t)
            for name in ("premia", "alpha", "baselines"):
                df = pd.concat([p[name] for p in parts], ignore_index=True)
                df.to_parquet(root / f"dua_{name}_{window}.parquet", index=False)
                out[f"{name}_{window}"] = len(df)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--signals", nargs="+", default=None, help="default: all 108")
    ap.add_argument("--ratings", nargs="+", default=list(RATINGS), choices=RATINGS)
    ap.add_argument("--workers", type=int, default=S.N_WORKERS,
                    help="fresh processes to fan out over (default: from cpu_count). "
                         "Measured peak is well under 1 GB per worker, so cores rather "
                         "than RAM is normally the limit")
    ap.add_argument("--threads", type=int, default=None,
                    help="numba threads per worker (default: from cpu_count, keeping "
                         "workers*threads within the core count)")
    ap.add_argument("--chunk", type=int, default=4, help="signals per fit")
    ap.add_argument("--stats", action="store_true",
                    help="recompute the per-path statistics (both windows) from the "
                         "saved series; needs a complete grid")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()

    # ❗--stats writes the FULL dua_{premia,alpha,baselines,config_locations} parquets.
    # Run with a --signals subset it would OVERWRITE them with subset-only frames, and
    # nothing downstream would notice: every exhibit would simply report on however
    # many signals happened to be in the file. Refuse instead.
    if args.stats and args.signals is not None:
        raise SystemExit(
            "ABORT: --stats with a --signals subset would overwrite the full statistics\n"
            "  parquets with subset-only frames, and no exhibit downstream can tell the\n"
            "  difference. Run --stats over the whole signal set.")

    import pblenv
    prov = pblenv.use()
    pblenv.require_fast("the DUA grid")

    import drrlib as D          # noqa: E402
    from bench import Bench     # noqa: E402
    from fastrun import pmap, sized    # noqa: E402

    signals = args.signals or list(C.ALL_SIGNALS)
    root = out_root()
    label = f"dua-grid-{len(signals)}sig" + ("-stats" if args.stats else "")

    with Bench(label, section="s3_nse") as b:
        if args.stats:
            missing = [(r, s_) for r in args.ratings for s_ in signals
                       if not (root / "series" / r / f"{s_}.parquet").exists()]
            if missing:
                raise SystemExit(
                    f"ABORT: --stats needs a complete grid; {len(missing)} series are\n"
                    f"  missing, first: {missing[:4]}\n"
                    "  Run the grid first (this file with no --stats).")
            counts = run_stats(args, root, signals, b)
            # 654 = (120 filters x 2 weightings x 3 ratings) minus the wins columns,
            # plus baselines -- pinned so a short grid cannot pass quietly.
            exp = {"premia": len(signals) * 654, "baselines": len(signals) * 6}
            ok = all(counts[f"premia_{w}"] == exp["premia"]
                     and counts[f"baselines_{w}"] == exp["baselines"]
                     for w in ("paper", "full"))
            b.note(mode="stats", **counts)
            ok = b.check(bool(ok),
                         f"per-path statistics, both windows: {counts} "
                         f"(expect premia {exp['premia']}, baselines {exp['baselines']})")
        else:
            sigs = sorted(signals)
            chunks = [sigs[i_:i_ + args.chunk] for i_ in range(0, len(sigs), args.chunk)]
            items = [(r, c, str(root)) for r in args.ratings for c in chunks
                     if args.force or not all(
                         (root / "series" / r / f"{s_}.parquet").exists() for s_ in c)]
            n_w, n_t = sized(len(items) or 1, args.workers, args.threads,
                             min_threads=2)
            with b.phase("grid"):
                if items:
                    results = pmap(run_one, items,
                                   workers=min(n_w, len(items)), threads=n_t)
                else:
                    results = []
                    print("all series parquets present -- skipping (--force to redo)")
            n_expected = len(args.ratings) * len(signals)
            present = sum((root / "series" / r / f"{s_}.parquet").exists()
                          for r in args.ratings for s_ in signals)
            slow = max((r["wall_s"] for r in results), default=0.0)
            rss = max((r["rss_gb"] or 0 for r in results), default=0.0)
            b.note(n_signals=len(signals), n_ratings=len(args.ratings),
                   n_units=len(items), workers=n_w, threads=n_t,
                   chunk=args.chunk, max_unit_wall_s=slow, max_worker_rss_gb=rss)
            ok = b.check(present == n_expected,
                         f"{present}/{n_expected} (rating, signal) series present")
        D.write_result(
            "dua_grid" + ("_stats" if args.stats else ""),
            {"summary": {"n_signals": len(signals), "stats": args.stats}},
            section="s3_nse", inputs=[Path(paths.PANEL), Path(paths.BBW)],
            t0=t0, extra={"pybondlab": prov})
    # The grid and its statistics layer each get their own marker: they are separate
    # orchestrator steps, and one can be complete while the other is not.
    D.mark_complete(root / ("_stats_complete.json" if args.stats else "_complete.json"),
                    ok, {"n_signals": len(signals), "ratings": list(args.ratings),
                         "stats": args.stats})
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
