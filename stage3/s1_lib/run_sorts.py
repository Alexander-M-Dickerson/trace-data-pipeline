r"""run_sorts.py -- the Section-3 sort panels, produced through PyBondLab.

Two steps, matching the paper:

  step 1  the monthly panel, left-merged with its unadjusted `*_mmn` signal twins on
          (cusip, date), with the risk-free rate taken from the factor file -- giving
          ret_vw_exc / ret_vw_bgn_exc (and the duration-adjusted twins) -- then cut to
          dates on or after the formation start.
  step 2  per result set: BatchStrategyFormation (single sorts, deciles for the whole
          universe and quintiles for a rating split) or BatchWithinFirmSortFormation
          (within-firm, at least 2 bonds per firm), holding_period=1, turnover on, then
          extract_panel with sign correction. The month-begin set also carries the `lib`
          and `ilq` portfolio characteristics, which Table 2's decomposition needs.

Output CSVs use the file grammar the rest of Stage 3 reads:

    {ret_type}_{sort}_{rating}_{signal_type}_{timing}_p{N}.csv

Run it as a file -- on Windows the entry module is re-imported in every worker:

    python s1_lib/run_sorts.py --sets baseline_end mmn_end mmn_bgn --sorts single wf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # stage3/

import _stage3_settings as S   # noqa: E402
import drrlib as D          # noqa: E402
import lib_engine as E      # noqa: E402
import paths                # noqa: E402
import pblenv               # noqa: E402
from bench import Bench     # noqa: E402

DATE_CUTOFF = S.FORMATION_START     # step 1 keeps data['date'] >= this
LIB7 = list(D.LIB_FACTORS)          # ytm cs bbtm dcs6 val_ipr val_hz str
ILLIQ5 = list(D.ILLIQ_FACTORS)      # ami ar_sprd cs_sprd ilq spd_rel
# One file per (sort, set, rating) carries EVERY signal, and each exhibit selects the
# ones it prints. The filename does not encode the signal list, so running this twice
# with different --signals would silently leave whichever ran last -- and an exhibit
# would then fail on a missing factor rather than read a stale one.
DEFAULT_SIGNALS = LIB7 + ILLIQ5

# the three Table-1 result sets: (signal_type, timing) -> which signals + return + chars
SETS = {
    "baseline_end": dict(signal_type="baseline", timing="end", mmn=False, chars=None),
    "mmn_end":      dict(signal_type="mmn",      timing="end", mmn=True,  chars=None),
    "mmn_bgn":      dict(signal_type="mmn",      timing="bgn", mmn=True,  chars=["lib", "ilq"]),
}

INPUT_PATHS = {"panel": paths.PANEL, "mmn": paths.MMN, "factors": paths.FACTORS}


def sorts_root() -> Path:
    """data/sorts/ -- the one place every sort CSV lands."""
    paths.SORTS.mkdir(parents=True, exist_ok=True)
    return paths.SORTS


# --------------------------------------------------------------------------
# step 1 -- data prep
# --------------------------------------------------------------------------
def prepare_data(signals: list[str], *, ret_types: tuple[str, ...] = ("exc",),
                 verbose: bool = True) -> pd.DataFrame:
    panel_p, mmn_p, fac_p = paths.PANEL, paths.MMN, paths.FACTORS
    for what, f in (("panel", panel_p), ("mmn twins", mmn_p), ("factors", fac_p)):
        if f is None or not Path(f).exists():
            raise SystemExit(
                f"Stage 3 needs the Stage 2 {what} file and it is not at {f}."
                "\n  Run `python tools/check_inputs.py` for the full list.")

    data = pd.read_parquet(panel_p)
    data["date"] = pd.to_datetime(data["date"])

    mmn = pd.read_parquet(mmn_p, columns=["cusip", "date"] + [f"{s}_mmn" for s in signals])
    mmn["date"] = pd.to_datetime(mmn["date"])

    n0 = len(data)
    data = data.merge(mmn, on=["cusip", "date"], how="left")
    assert len(data) == n0, f"mmn merge changed rows {n0} -> {len(data)}: duplicate (cusip,date) in mmn file"

    # the panel carries its own rfret; it is REPLACED with the factor file's rf, so
    # every excess return in Stage 3 rests on one risk-free series
    fac = pd.read_parquet(fac_p, columns=["date", "rf"]).rename(columns={"rf": "rfret"})
    fac["date"] = pd.to_datetime(fac["date"])
    data = data.drop(columns=["rfret"], errors="ignore").merge(fac, on="date", how="left")
    assert len(data) == n0, "rf merge changed rows: duplicate dates in factors file"

    if "exc" in ret_types:
        data["ret_vw_exc"] = data["ret_vw"] - data["rfret"]
        data["ret_vw_bgn_exc"] = data["ret_vw_bgn"] - data["rfret"]
    if "dur" in ret_types:
        data["ret_vw_dur"] = data["ret_vw"] - data["tret"]
        data["ret_vw_bgn_dur"] = data["ret_vw_bgn"] - data["tret"]

    data = data[data["date"] >= DATE_CUTOFF].reset_index(drop=True)

    # PyBondLab trap: a nullable-Int rating breaks numba. The parquet SMALLINT
    # arrives float64 via numpy-backed read; assert rather than assume.
    if not pd.api.types.is_float_dtype(data["spc_rat"]):
        data["spc_rat"] = data["spc_rat"].astype("float64")
    dup = data.duplicated(["cusip", "date"]).sum()
    assert dup == 0, f"{dup} duplicate (cusip,date) rows would silently corrupt PyBondLab (bug D1)"

    if verbose:
        print(f"[data] {len(data):,} rows  {data['date'].min():%Y-%m-%d}"
              f" .. {data['date'].max():%Y-%m-%d}  ({len(data.columns)} cols)")
    return data


# --------------------------------------------------------------------------
# step 2 -- one result set through PyBondLab
# --------------------------------------------------------------------------
def run_set(data: pd.DataFrame, *, sort: str, set_name: str, signals: list[str],
            ret_type: str = "exc", rating: str | None = None, n_jobs: int = 1,
            fast: bool = False) -> pd.DataFrame:
    """One (sort, result-set) batch -> the extract_panel frame (sign_correct=True)."""
    from PyBondLab import (BatchStrategyFormation, BatchWithinFirmSortFormation,
                           NamingConfig, extract_panel)

    cfg = SETS[set_name]
    sig_cols = [f"{s}_mmn" for s in signals] if cfg["mmn"] else list(signals)
    ret_col = {"end": f"ret_vw_{ret_type}", "bgn": f"ret_vw_bgn_{ret_type}"}[cfg["timing"]]

    # trim: PyBondLab renames but never subsets, so pass only what the fit reads
    keep = ["cusip", "date", "mcap_e", "spc_rat", "permno", ret_col] + sig_cols
    keep += cfg["chars"] or []
    df = data[list(dict.fromkeys(keep))]

    if fast and sort == "single":
        # the kernelized single sort -- the same output as the batch path, over one
        # shared set of panel arrays instead of one set per signal
        from PyBondLab.fast_sorts import fast_single_sorts
        res = fast_single_sorts(
            df, sig_cols,
            columns={"ID": "cusip", "ret": ret_col, "VW": "mcap_e",
                     "RATING_NUM": "spc_rat"},
            num_portfolios=E.n_portfolios("single",
                                          "all" if rating is None else rating.lower()),
            rating=rating, dynamic_weights=True, chars=cfg["chars"])
        return extract_panel(res, naming=NamingConfig(sign_correct=True))

    if fast and sort == "wf":
        # the kernelized within-firm pipeline. The rating filter restricts the FORMATION
        # universe INSIDE the engine -- never by subsetting the panel first, which would
        # change which bonds each firm has and so change the sort itself.
        from PyBondLab.fast_sorts import fast_within_firm_sorts
        res = fast_within_firm_sorts(
            df, sig_cols,
            columns={"ID": "cusip", "ret": ret_col, "VW": "mcap_e",
                     "RATING_NUM": "spc_rat"},
            firm_id_col=S.FIRM_ID_COL, chars=cfg["chars"], rating=rating)
        return extract_panel(res, naming=NamingConfig(sign_correct=True))

    kw: dict = dict(signals=sig_cols, turnover=True, n_jobs=n_jobs)
    if rating is not None:
        kw["rating"] = rating
    if cfg["chars"] is not None:
        kw["chars"] = cfg["chars"]

    if sort == "single":
        num_ports = E.n_portfolios("single", "all" if rating is None else rating.lower())
        batch = BatchStrategyFormation(
            data=df, columns={"ID": "cusip", "ret": ret_col, "VW": "mcap_e",
                              "RATING_NUM": "spc_rat"},
            holding_period=S.HOLDING_PERIOD, num_portfolios=num_ports,
            verbose=False, **kw)
    elif sort == "wf":
        batch = BatchWithinFirmSortFormation(
            data=df, firm_id_col=S.FIRM_ID_COL,
            columns={"ID": "cusip", "VW": "mcap_e", "RATING_NUM": "spc_rat", "ret": ret_col},
            min_bonds_per_firm=S.MIN_BONDS_PER_FIRM, verbose=False, **kw)
    else:
        raise ValueError(f"sort={sort!r}")

    results = batch.fit()
    return extract_panel(results, naming=NamingConfig(sign_correct=True))


def out_name(*, ret_type: str, sort: str, rating: str | None, set_name: str) -> str:
    cfg = SETS[set_name]
    rating_str = (rating or "all").lower()
    n = E.n_portfolios(sort, rating_str)
    return f"{ret_type}_{sort}_{rating_str}_{cfg['signal_type']}_{cfg['timing']}_p{n}.csv"


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    ap.add_argument("--sorts", nargs="+", default=["single", "wf"], choices=["single", "wf"])
    ap.add_argument("--signals", nargs="+", default=DEFAULT_SIGNALS,
                    help="base mnemonics (default: the 7 LIB factors plus the 5 "
                         "illiquidity ones, which is what the exhibits need)")
    ap.add_argument("--ret", default="exc", choices=["exc", "dur"],
                    help="exc = in excess of the one-month bill; dur = duration-adjusted")
    ap.add_argument("--rating", default=None, choices=[None, "IG", "NIG"],
                    help="restrict the formation universe to one rating class")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--fast", action="store_true",
                    help="use PyBondLab's sort kernels (needs a build that has them)")
    ap.add_argument("--force", action="store_true", help="recompute even if the CSV exists")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    prov = pblenv.use()
    if args.fast:
        pblenv.require_fast("--fast")
    root = sorts_root()
    tag = "sorts" + (f"-{args.rating.lower()}" if args.rating else "")

    todo = [(s_, n_) for s_ in args.sorts for n_ in args.sets
            if args.force or not (root / out_name(ret_type=args.ret, sort=s_,
                                                  rating=args.rating, set_name=n_)).exists()]
    written, data = [], None
    with Bench(tag, section="s1_lib", echo=True) as bench:
        with bench.phase("load"):
            if todo:                    # nothing to do means nothing to load
                data = prepare_data(args.signals, ret_types=(args.ret,))
        for sort in args.sorts:
            for set_name in args.sets:
                name = out_name(ret_type=args.ret, sort=sort, rating=args.rating,
                                set_name=set_name)
                out = root / name
                with bench.phase(f"{sort}.{set_name}"):
                    if out.exists() and not args.force:
                        print(f"[skip] {name} exists")
                        continue
                    t0 = time.perf_counter()
                    panel = run_set(data, sort=sort, set_name=set_name,
                                    signals=args.signals, ret_type=args.ret,
                                    rating=args.rating, n_jobs=args.n_jobs,
                                    fast=args.fast)
                    panel.to_csv(out, index=False)
                    written.append(name)
                    print(f"[done] {name}  {len(panel):,} rows  "
                          f"{time.perf_counter() - t0:.1f}s")
        bench.note(signals=args.signals, ret=args.ret, rating=args.rating,
                   sets=args.sets, sorts=args.sorts, fast=args.fast,
                   n_written=len(written))
        # the run's own check: every (sort, set) asked for exists afterwards, whether
        # this run made it or a previous one did. A partial sort set is the failure
        # mode that matters -- every later exhibit reads these by name.
        want = [out_name(ret_type=args.ret, sort=s_, rating=args.rating, set_name=n_)
                for s_ in args.sorts for n_ in args.sets]
        have = [n_ for n_ in want if (root / n_).exists()]
        ok = bench.check(len(have) == len(want),
                         f"{len(have)}/{len(want)} sort CSVs present in {root.name}/")

    manifest = {"signals": args.signals, "ret": args.ret, "rating": args.rating,
                "sets": args.sets, "sorts": args.sorts, "date_cutoff": DATE_CUTOFF,
                "files": want,
                "manifest": {
                    "name": tag,
                    "section": "s1_lib",
                    "written_utc": pd.Timestamp.utcnow().isoformat(),
                    "git_commit": D._git("rev-parse", "HEAD"),
                    "git_branch": D._git("rev-parse", "--abbrev-ref", "HEAD"),
                    "pybondlab": prov,
                    "python": sys.version.split()[0],
                    "inputs": [D.fingerprint(v) for v in INPUT_PATHS.values()],
                }}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str),
                                        encoding="utf-8")
    print(f"\nwrote {root / 'manifest.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
