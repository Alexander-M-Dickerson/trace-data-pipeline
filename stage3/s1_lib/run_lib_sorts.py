r"""run_lib_sorts.py -- the 108-signal month-end / month-begin sorts (Table B.1).

Where `run_sorts.py` produces the seven LIB factors in the three approaches, this
produces the WHOLE signal set on both return windows, which is what the bias census in
Table B.1 and the Internet Appendix counts over.

The conventions, each one different from `run_sorts.py` and each one deliberate:

  * data      the monthly panel from 2002-08-31 -- one month EARLIER than the three-
              approach runs, because this design has no formation-month gap to absorb.
  * signals   every panel column that is not an identifier, with the unadjusted `_mmn`
              twin swapped in wherever one exists (30 of the 108).
  * returns   RAW `ret_vw` / `ret_vw_bgn` -- NOT excess of the bill. The census compares
              a signal against ITSELF on the other return window, so the risk-free rate
              is common to both sides and cancels; subtracting it would only add a step.
  * sorts     single p10 or within-firm p2 (at least 2 bonds per firm), holding_period=1,
              turnover on, no rating restriction, extract_panel with sign correction.

One (sort, timing) per invocation, so the four can run concurrently:

    python s1_lib/run_lib_sorts.py --sort single --timing end --n-jobs 6
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
import paths                # noqa: E402
import pblenv               # noqa: E402
from bench import Bench     # noqa: E402

DATE_CUTOFF = "2002-08-31"          # one month EARLIER than the three-approach runs

# Everything that identifies a bond-month, prices it, or dates it -- never a signal.
ID_COLUMNS = [
    "cusip", "date", "issuer_cusip", "permno", "permco", "gvkey",
    "144a", "country", "call",
    "ret_vw", "ret_vw_bgn", "ret_vwx", "ret_vwx_bgn", "ret_type",
    "hprd", "lib", "libd", "hprd_bgn", "igap_bgn",
    "dt_s", "dt_e", "dt_s_bgn", "dt_e_bgn",
    "spc_rat", "mdc_rat",
    "ff17num", "ff30num",
    "mcap_s", "mcap_e", "fce_val",
    "tret", "rfret",
    "sig_dt", "sig_gap",
]


def lib_csv_name(sort: str, timing: str) -> str:
    prefix = "bond_single_sort" if sort == "single" else "bond_within_firm"
    p = 10 if sort == "single" else 2
    return f"{prefix}_lib_all_{timing}_p{p}_h1.csv"


def lib_root() -> Path:
    """data/sorts/lib/ -- kept apart from the three-approach CSVs, different grammar."""
    d = paths.SORTS / "lib"
    d.mkdir(parents=True, exist_ok=True)
    return d


def prepare(verbose: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """The panel plus its 30 unadjusted twins, and the signal column list."""
    for what, f in (("panel", paths.PANEL), ("mmn twins", paths.MMN)):
        if f is None or not Path(f).exists():
            raise SystemExit(f"Stage 3 needs the Stage 2 {what} file and it is not at {f}."
                             "\n  Run `python tools/check_inputs.py` for the full list.")

    data = pd.read_parquet(paths.PANEL)
    data["date"] = pd.to_datetime(data["date"])
    data = data[data["date"] >= DATE_CUTOFF].reset_index(drop=True)

    # Only the 30 price-based twins are swapped in. The sidecar carries more (`md_dur`,
    # `convx` and others have twins too); those signals stay on their month-end form
    # here, because Table B.1's census is defined over this list.
    mmn = pd.read_parquet(paths.MMN,
                          columns=["cusip", "date"] + [f"{s}_mmn" for s in D.PRICE_BASED])
    mmn["date"] = pd.to_datetime(mmn["date"])
    n0 = len(data)
    data = data.merge(mmn, on=["cusip", "date"], how="left")
    assert len(data) == n0, f"mmn merge changed rows {n0} -> {len(data)}"

    base = [c for c in data.columns if c not in ID_COLUMNS and not c.endswith("_mmn")]
    signal_cols = [f"{c}_mmn" if f"{c}_mmn" in data.columns else c for c in base]
    n_swap = sum(c.endswith("_mmn") for c in signal_cols)
    assert n_swap == len(D.PRICE_BASED), \
        f"expected {len(D.PRICE_BASED)} _mmn swaps, got {n_swap}"

    # PyBondLab trap: a nullable-Int rating breaks numba. Assert rather than assume.
    if not pd.api.types.is_float_dtype(data["spc_rat"]):
        data["spc_rat"] = data["spc_rat"].astype("float64")
    dup = data.duplicated(["cusip", "date"]).sum()
    assert dup == 0, f"{dup} duplicate (cusip,date) rows would silently corrupt the sort"

    if verbose:
        print(f"[data] {len(data):,} rows {data['date'].min():%Y-%m-%d}"
              f" .. {data['date'].max():%Y-%m-%d}; {len(signal_cols)} signals"
              f" ({n_swap} swapped to _mmn)")
    return data, signal_cols


def run_one(sort: str, timing: str, n_jobs: int, *, fast: bool = False) -> tuple[Path, dict]:
    from PyBondLab import (BatchStrategyFormation, BatchWithinFirmSortFormation,
                           NamingConfig, extract_panel)

    data, signal_cols = prepare()
    ret_col = "ret_vw_bgn" if timing == "bgn" else "ret_vw"
    out = lib_root() / lib_csv_name(sort, timing)
    cols = {"ID": "cusip", "ret": ret_col, "VW": "mcap_e", "RATING_NUM": "spc_rat"}

    t0 = time.perf_counter()
    if fast and sort == "single":
        from PyBondLab.fast_sorts import fast_single_sorts
        res = fast_single_sorts(data, signal_cols, columns=cols, num_portfolios=10,
                                dynamic_weights=True)
        panel = extract_panel(res, naming=NamingConfig(sign_correct=True))
        how = "kernel"
    elif fast and sort == "wf":
        from PyBondLab.fast_sorts import fast_within_firm_sorts
        res = fast_within_firm_sorts(data, signal_cols, columns=cols,
                                     firm_id_col=S.FIRM_ID_COL)
        panel = extract_panel(res, naming=NamingConfig(sign_correct=True))
        how = "kernel"
    else:
        if sort == "single":
            batch = BatchStrategyFormation(
                data=data, columns=cols, signals=signal_cols,
                holding_period=S.HOLDING_PERIOD, num_portfolios=10,
                turnover=True, n_jobs=n_jobs, chunk_size="auto", verbose=False)
        else:
            batch = BatchWithinFirmSortFormation(
                data=data, signals=signal_cols, firm_id_col=S.FIRM_ID_COL,
                columns={"ID": "cusip", "VW": "mcap_e", "RATING_NUM": "spc_rat",
                         "ret": ret_col},
                min_bonds_per_firm=S.MIN_BONDS_PER_FIRM, turnover=True,
                n_jobs=n_jobs, chunk_size="auto", verbose=False)
        panel = extract_panel(batch.fit(), naming=NamingConfig(sign_correct=True))
        how = f"n_jobs={n_jobs}"

    D.write_atomic(panel, out, index=False)
    wall = time.perf_counter() - t0
    print(f"[done] {out.name}  {len(panel):,} rows  {wall:.0f}s  {how}")
    return out, {"rows": len(panel), "n_signals": len(signal_cols),
                 "wall_s": round(wall, 1), "how": how}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sort", required=True, choices=["single", "wf"])
    ap.add_argument("--timing", required=True, choices=["end", "bgn"])
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--fast", action="store_true",
                    help="use PyBondLab's sort kernels (needs a build that has them)")
    ap.add_argument("--force", action="store_true",
                    help="accepted for symmetry with the other producers. This one has "
                         "no skip-if-exists branch, so it always recomputes anyway")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    prov = pblenv.use()
    if args.fast:
        pblenv.require_fast("--fast")
    tag = f"lib-sorts-{args.sort}-{args.timing}"
    with Bench(tag, section="s1_lib", echo=True) as b:
        with b.phase(f"{args.sort}.{args.timing}"):
            out, rep = run_one(args.sort, args.timing, args.n_jobs, fast=args.fast)
        b.note(**rep)
        # Table B.1 counts over the whole signal set; a short run would silently
        # shrink the census rather than fail it.
        ok = b.check(rep["n_signals"] >= 100,
                     f"{rep['n_signals']} signals sorted into {out.name}")

    out.with_suffix(".json").write_text(json.dumps({
        "sort": args.sort, "timing": args.timing, "n_jobs": args.n_jobs,
        "date_cutoff": DATE_CUTOFF, "result": rep,
        "manifest": {
            "name": tag, "section": "s1_lib",
            "written_utc": pd.Timestamp.utcnow().isoformat(),
            "git_commit": D._git("rev-parse", "HEAD"),
            "pybondlab": prov, "python": sys.version.split()[0],
            "inputs": [D.fingerprint(paths.PANEL), D.fingerprint(paths.MMN)],
        }}, indent=2, default=str), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
