r"""run_zoo_sorts.py -- the factor-zoo sorts: all 108 signals, single and within-firm.

This is the widest producer in Stage 3. Everything the Internet Appendix says about
how many bond factors survive, and which ones, is counted off these two CSVs.

  step 1  the monthly panel, loaded raw, from 2002-08-31. Signals = every column that
          is not an identifier, restricted to the 108 in the cluster map. Returns are
          RAW `ret_vw`, not excess: every printed statistic is on the LONG-SHORT leg,
          which the risk-free rate cancels out of. `--excess` subtracts it anyway if
          you want the legs on an excess basis; the long-short series does not move.
  step 2  single sort: BatchStrategyFormation, deciles, holding_period=1, turnover on.
  step 3  within-firm: BatchWithinFirmSortFormation, high/low, at least 2 bonds/firm.
  both    extract_panel(NamingConfig(sign_correct=True)).

❗Sign correction is decided on THE SAMPLE THAT WAS SORTED. Run to a different end date
and the flip set can differ -- a factor whose full-sample mean is barely positive can
be barely negative on a shorter window. So the window here is a producer argument, and
the `*` in the factor name is the receipt saying which way this run decided. Do not
compare a starred series from one run against an unstarred series from another without
re-orienting both.

By default the whole panel is sorted and the exhibits truncate to their own window at
the statistics layer.

Run it as a file -- on Windows the entry module is re-imported in every worker:

    python s4_zoo/run_zoo_sorts.py --n-jobs 6
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
import zoo_engine as Z      # noqa: E402
from bench import Bench     # noqa: E402

DATE_CUTOFF = "2002-08-31"

# Everything that identifies a bond-month, prices it, or dates it -- never a signal.
ID_COLUMNS = [
    "cusip", "date", "issuer_cusip", "permno", "permco", "gvkey",
    "144a", "country", "call",
    "ret_vw", "ret_vw_bgn", "ret_vwx", "ret_vwx_bgn", "ret_type",
    "hprd", "lib", "libd", "hprd_bgn", "igap_bgn",
    "dt_s", "dt_e", "dt_s_bgn", "dt_e_bgn",
    "spc_rat", "mdc_rat", "ff17num", "ff30num",
    "mcap_s", "mcap_e", "fce_val", "tret", "rfret", "sig_dt", "sig_gap",
]


def sorts_root() -> Path:
    d = paths.SORTS / "zoo"
    d.mkdir(parents=True, exist_ok=True)
    return d


def prepare_data(*, end: str | None = None,
                 excess: bool = False) -> tuple[pd.DataFrame, list[str]]:
    if paths.PANEL is None or not Path(paths.PANEL).exists():
        raise SystemExit(f"Stage 3 needs the Stage 2 panel and it is not at {paths.PANEL}."
                         "\n  Run `python tools/check_inputs.py` for the full list.")
    data = pd.read_parquet(paths.PANEL)
    data["date"] = pd.to_datetime(data["date"])
    if excess:
        fac = pd.read_parquet(paths.FACTORS, columns=["date", "rf"])
        fac["date"] = pd.to_datetime(fac["date"])
        n0 = len(data)
        data = (data.drop(columns=["rfret"], errors="ignore")
                .merge(fac.rename(columns={"rf": "rfret"}), on="date", how="left"))
        assert len(data) == n0, "rf merge changed rows: duplicate dates in the factor file"
        data["ret_vw"] = data["ret_vw"] - data["rfret"]
    data = data[data["date"] >= DATE_CUTOFF]
    if end:
        data = data[data["date"] <= end]
    data = data.reset_index(drop=True)

    if not pd.api.types.is_float_dtype(data["spc_rat"]):
        data["spc_rat"] = data["spc_rat"].astype("float64")
    dup = data.duplicated(["cusip", "date"]).sum()
    assert dup == 0, f"{dup} duplicate (cusip,date) rows would corrupt PyBondLab"

    signals = [c for c in data.columns if c not in ID_COLUMNS]
    # A panel carrying columns beyond the published 140 would otherwise be sorted as
    # extra "signals", and every false-discovery threshold in Section IA is computed
    # over m = the number of factors. Restrict to the cluster map, and say so.
    extras = sorted(c for c in signals if c not in Z.CLUSTER_OF)
    if extras:
        print(f"[data] dropping {len(extras)} columns that are not zoo signals: {extras}")
        signals = [c for c in signals if c in Z.CLUSTER_OF]
    assert len(signals) == Z.N_FACTORS, \
        f"{len(signals)} signals after the cluster-map restriction != {Z.N_FACTORS}"
    print(f"[data] {len(data):,} rows  {data['date'].min():%Y-%m-%d}"
          f" .. {data['date'].max():%Y-%m-%d}  {len(signals)} signals")
    return data, signals


def run_batch(data: pd.DataFrame, signals: list[str], *, sort: str,
              n_jobs: int = 1, fast: bool = False) -> pd.DataFrame:
    from PyBondLab import (BatchStrategyFormation, BatchWithinFirmSortFormation,
                           NamingConfig, extract_panel)

    cols = {"ID": "cusip", "ret": "ret_vw", "VW": "mcap_e", "RATING_NUM": "spc_rat"}

    if fast and sort == "single":
        from PyBondLab.fast_sorts import fast_single_sorts
        res = fast_single_sorts(data, signals, columns=cols, num_portfolios=10,
                                dynamic_weights=True)
        return extract_panel(res, naming=NamingConfig(sign_correct=True))

    if fast and sort == "wf":
        from PyBondLab.fast_sorts import fast_within_firm_sorts
        res = fast_within_firm_sorts(data, signals, columns=cols,
                                     firm_id_col=S.FIRM_ID_COL)
        return extract_panel(res, naming=NamingConfig(sign_correct=True))

    keep = ["cusip", "date", "mcap_e", "spc_rat", S.FIRM_ID_COL, "ret_vw"] + signals
    df = data[list(dict.fromkeys(keep))]
    kw: dict = dict(signals=signals, turnover=True)
    if n_jobs > 1:
        kw["n_jobs"] = n_jobs
    if sort == "single":
        batch = BatchStrategyFormation(
            data=df, columns=cols, holding_period=S.HOLDING_PERIOD,
            num_portfolios=10, verbose=False, **kw)
    else:
        batch = BatchWithinFirmSortFormation(
            data=df, firm_id_col=S.FIRM_ID_COL,
            columns={"ID": "cusip", "VW": "mcap_e", "RATING_NUM": "spc_rat",
                     "ret": "ret_vw"},
            min_bonds_per_firm=S.MIN_BONDS_PER_FIRM, verbose=False, **kw)
    return extract_panel(batch.fit(), naming=NamingConfig(sign_correct=True))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sorts", nargs="+", default=["single", "wf"],
                    choices=["single", "wf"])
    ap.add_argument("--end", default=None,
                    help="optional end date; default is the panel's own frontier, and "
                         "the exhibits truncate to their window afterwards")
    ap.add_argument("--excess", action="store_true",
                    help="subtract the factor file's rf from ret_vw (the long-short "
                         "series does not move; the legs do)")
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--fast", action="store_true",
                    help="use PyBondLab's sort kernels (needs a build that has them)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    prov = pblenv.use()
    if args.fast:
        pblenv.require_fast("--fast")
    root = sorts_root()
    tag = "zoo-sorts"

    data, signals = None, None
    with Bench(tag, section="s4_zoo", echo=True) as bench:
        for sort in args.sorts:
            out = root / Z.CSV[sort]
            with bench.phase(sort):
                if out.exists() and not args.force:
                    print(f"[skip] {out.name} exists")
                    continue
                if data is None:
                    with bench.phase("load"):
                        data, signals = prepare_data(end=args.end, excess=args.excess)
                t0 = time.perf_counter()
                panel = run_batch(data, signals, sort=sort, n_jobs=args.n_jobs,
                                  fast=args.fast)
                panel.to_csv(out, index=False)
                print(f"[done] {out.name}  {len(panel):,} rows  "
                      f"{time.perf_counter() - t0:.1f}s")
        bench.note(sorts=args.sorts, end=args.end, excess=args.excess,
                   n_signals=len(signals or []), fast=args.fast)
        have = [s_ for s_ in args.sorts if (root / Z.CSV[s_]).exists()]
        ok = bench.check(len(have) == len(args.sorts),
                         f"{len(have)}/{len(args.sorts)} zoo sort CSVs present")

    manifest = {"sorts": args.sorts, "end": args.end, "excess": args.excess,
                "date_cutoff": DATE_CUTOFF, "n_signals": len(signals or []),
                "manifest": {
                    "name": tag, "section": "s4_zoo",
                    "written_utc": pd.Timestamp.utcnow().isoformat(),
                    "git_commit": D._git("rev-parse", "HEAD"),
                    "git_branch": D._git("rev-parse", "--abbrev-ref", "HEAD"),
                    "pybondlab": prov, "python": sys.version.split()[0],
                    "inputs": [D.fingerprint(paths.PANEL), D.fingerprint(paths.FACTORS)],
                }}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str),
                                        encoding="utf-8")
    print(f"\nwrote {root / 'manifest.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
