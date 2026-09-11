r"""run_lab.py -- the Section-4 (LAB) winsorization sweep, through PyBondLab.

Section 4 asks what ex-post filtering does to a published result: take a sort, then
winsorize its returns at a percentile of the FULL sample -- a threshold nobody could
have known at formation -- and read off how much the anomaly improves.

  data   the monthly panel, loaded raw, with `ret_vw` put in excess of the PANEL'S
         OWN `rfret`.

         ❗This is NOT how `run_sorts.py` does it. There the risk-free rate is taken
         from the factor file, because the three approaches must sit on one common
         series. Here every comparison is a signal against ITSELF under a different
         filter, so the panel's own rate is the right one and swapping in another
         would put the two sides of the paired difference on different rates.

         Window: [2002-08-31, 2024-12-31] -- one month earlier than the Section-3
         window, T = 269.
  sweep  per (tail, rating): DataUncertaintyAnalysis(signals, holding_periods=[1],
         filters={'wins': [(99.5, tail)]}, include_baseline=True, num_portfolios=10,
         dynamic_weights=True, rating=rating).fit()
  out    the nine time-series frames per cell, as parquet under data/s2_lab/series/.

❗The winsorization threshold is a full-sample quantile BY CONSTRUCTION -- that is the
look-ahead the section measures. So the sample window is a PRODUCER argument here, not
something applied later at the statistics layer: changing `--date-end` changes the
threshold and therefore the series, which is not true anywhere else in Stage 3.

Run it as a file -- on Windows the entry module is re-imported in every worker:

    python s2_lab/run_lab.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # stage3/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _stage3_settings as S   # noqa: E402
import drrlib as D          # noqa: E402
import lab_engine as E      # noqa: E402
import paths                # noqa: E402
import pblenv               # noqa: E402
from bench import Bench     # noqa: E402

DATE_START = S.SAMPLE["lab"]["start"]
DATE_END = S.SAMPLE["lab"]["end"]
WINS_LEVEL = 99.5


def series_root() -> Path:
    d = paths.section_results("s2_lab") / "series"
    d.mkdir(parents=True, exist_ok=True)
    return d


def prepare_data(*, date_start: str = DATE_START, date_end: str = DATE_END,
                 verbose: bool = True) -> pd.DataFrame:
    """The LAB data prep: the panel's own risk-free rate, then the window."""
    if paths.PANEL is None or not Path(paths.PANEL).exists():
        raise SystemExit(f"Stage 3 needs the Stage 2 panel and it is not at {paths.PANEL}."
                         "\n  Run `python tools/check_inputs.py` for the full list.")
    data = pd.read_parquet(paths.PANEL)
    data["date"] = pd.to_datetime(data["date"])
    if "rfret" in data.columns:
        data["ret_vw"] = data["ret_vw"] - data["rfret"]
    data = data[(data["date"] >= date_start) & (data["date"] <= date_end)].copy()

    # PyBondLab traps: a nullable-Int rating breaks numba, and a duplicate (cusip,date)
    # silently corrupts the sort rather than raising.
    if not pd.api.types.is_float_dtype(data["spc_rat"]):
        data["spc_rat"] = data["spc_rat"].astype("float64")
    dup = data.duplicated(["cusip", "date"]).sum()
    assert dup == 0, f"{dup} duplicate (cusip,date) rows would corrupt PyBondLab"
    if verbose:
        print(f"[data] {len(data):,} rows  {data['date'].min():%Y-%m-%d}"
              f" .. {data['date'].max():%Y-%m-%d}")
    return data


def run_cell(data: pd.DataFrame, *, tail: str, rating: str | None,
             signals: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """One (tail, rating) cell -> the nine time-series frames."""
    import warnings

    from PyBondLab import DataUncertaintyAnalysis

    mapped = data.rename(columns={"cusip": "ID", "mcap_e": "VW",
                                  "spc_rat": "RATING_NUM", "ret_vw": "ret"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = DataUncertaintyAnalysis(
            data=mapped, signals=list(signals), holding_periods=[1],
            filters={"wins": [(WINS_LEVEL, tail)]}, include_baseline=True,
            num_portfolios=10, dynamic_weights=True, rating=rating,
            verbose=False,
        ).fit()

    suffix = f"_{rating}" if rating else ""
    ts: dict[str, dict] = {k: {} for k in E.TS_KEYS}
    for s in signals:
        sig = results.filter(signal=s)
        base_col = f"{s}_hp1_baseline{suffix}"
        wins_col = f"{s}_hp1_wins_{WINS_LEVEL}_{tail}{suffix}"
        # for the baseline there is no look-ahead, so ex ante == ex post
        ls_b, lo_b, sh_b = (sig.vw_ex_ante[base_col], sig.vw_long_ex_ante[base_col],
                            sig.vw_short_ex_ante[base_col])
        ls_w, lo_w, sh_w = (sig.vw_ex_post[wins_col], sig.vw_long_ex_post[wins_col],
                            sig.vw_short_ex_post[wins_col])
        ts["ts_long_wins"][s], ts["ts_long_base"][s] = lo_w, lo_b
        ts["ts_short_wins"][s], ts["ts_short_base"][s] = sh_w, sh_b
        ts["ts_ls_wins"][s], ts["ts_ls_base"][s] = ls_w, ls_b
        ts["ts_bias_long"][s] = lo_w - lo_b
        ts["ts_bias_short"][s] = sh_w - sh_b
        ts["ts_bias_ls"][s] = ls_w - ls_b
    return {k: pd.DataFrame(v) for k, v in ts.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tails", nargs="+", default=["left", "right"],
                    choices=["left", "right"])
    ap.add_argument("--ratings", nargs="+", default=["All", "IG", "NIG"],
                    choices=["All", "IG", "NIG"])
    ap.add_argument("--date-start", default=DATE_START)
    ap.add_argument("--date-end", default=DATE_END,
                    help="❗the winsorization threshold is a quantile of THIS window, "
                         "so moving it changes the series -- see the module docstring")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    prov = pblenv.use()
    root = series_root()
    tag = "lab-series"

    data = None
    with Bench(tag, section="s2_lab", echo=True) as bench:
        for tail in args.tails:
            for rating in args.ratings:
                cell_files = [root / f"standard__{rating}__{tail}__{k}.parquet"
                              for k in E.TS_KEYS]
                with bench.phase(f"{tail}.{rating}"):
                    if all(p.exists() for p in cell_files) and not args.force:
                        print(f"[skip] {tail}/{rating} exists")
                        continue
                    if data is None:
                        data = prepare_data(date_start=args.date_start,
                                            date_end=args.date_end)
                    t0 = time.perf_counter()
                    ours = run_cell(data, tail=tail,
                                    rating=None if rating == "All" else rating,
                                    signals=E.TAIL_SIGNALS[tail])
                    for k, p in zip(E.TS_KEYS, cell_files):
                        ours[k].to_parquet(p)
                    print(f"[done] {tail}/{rating}  {time.perf_counter()-t0:.1f}s")
        bench.note(tails=args.tails, ratings=args.ratings, wins_level=WINS_LEVEL,
                   date_start=args.date_start, date_end=args.date_end)
        n_cells = len(args.tails) * len(args.ratings)
        n_have = sum(all((root / f"standard__{r}__{t}__{k}.parquet").exists()
                         for k in E.TS_KEYS)
                     for t in args.tails for r in args.ratings)
        ok = bench.check(n_have == n_cells,
                         f"{n_have}/{n_cells} LAB cells complete (9 frames each)")

    manifest = {"tails": args.tails, "ratings": args.ratings,
                "date_start": args.date_start, "date_end": args.date_end,
                "wins_level": WINS_LEVEL,
                "manifest": {
                    "name": tag, "section": "s2_lab",
                    "written_utc": pd.Timestamp.utcnow().isoformat(),
                    "git_commit": D._git("rev-parse", "HEAD"),
                    "git_branch": D._git("rev-parse", "--abbrev-ref", "HEAD"),
                    "pybondlab": prov, "python": sys.version.split()[0],
                    "inputs": [D.fingerprint(paths.PANEL)],
                }}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str),
                                        encoding="utf-8")
    print(f"\nwrote {root / 'manifest.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
