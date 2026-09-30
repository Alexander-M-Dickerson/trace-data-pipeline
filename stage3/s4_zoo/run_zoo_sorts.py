r"""run_zoo_sorts.py -- the factor-zoo sorts: all 108 signals, single and within-firm.

This is the widest producer in Stage 3. Everything the Internet Appendix says about
how many bond factors survive, and which ones, is counted off these two CSVs.

  step 1  the monthly panel, loaded raw, from 2002-08-31. Signals = every column that
          is not an identifier, restricted to the 108 in the cluster map. Returns are
          RAW `ret_vw`, not excess: every printed statistic is on the LONG-SHORT leg,
          which the risk-free rate cancels out of. `--excess` subtracts it anyway if
          you want the legs on an excess basis; the long-short series does not move.
  step 2  single sort: PyBondLab's fast_single_sorts, deciles, holding_period=1, turnover on.
  step 3  within-firm: fast_within_firm_sorts, high/low, at least 2 bonds/firm.
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

    python s4_zoo/run_zoo_sorts.py
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
import returns as R         # noqa: E402
import zoo_engine as Z      # noqa: E402
import signal_set as SIG    # noqa: E402
from bench import Bench     # noqa: E402

DATE_CUTOFF = "2002-08-31"


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
    # [ref:rule.return_types] the run's signals; the standard run's are left as read
    data = R.signals(data)
    if excess:
        fac = pd.read_parquet(paths.FACTORS, columns=["date", "rf"])
        fac["date"] = pd.to_datetime(fac["date"])
        n0 = len(data)
        data = (data.drop(columns=["rfret"], errors="ignore")
                .merge(fac.rename(columns={"rf": "rfret"}), on="date", how="left"))
        assert len(data) == n0, "rf merge changed rows: duplicate dates in the factor file"
    # the run's return: raw `ret_vw` in the standard run, less the factor file's rf with
    # --excess; in a duration-adjusted run less the Treasury column, never less rf
    R.set_returns(data, ("ret_vw",), rf="rfret" if excess else None)
    data = data[data["date"] >= DATE_CUTOFF]
    if end:
        data = data[data["date"] <= end]
    data = data.reset_index(drop=True)

    if not pd.api.types.is_float_dtype(data["spc_rat"]):
        data["spc_rat"] = data["spc_rat"].astype("float64")
    dup = data.duplicated(["cusip", "date"]).sum()
    assert dup == 0, f"{dup} duplicate (cusip,date) rows would corrupt PyBondLab"

    # [ref:rule.signal_set] every false-discovery threshold in Section IA is computed over
    # m = the number of factors, so a column that is not a signal must never be sorted here.
    signals = SIG.select(data.columns, what="the Stage 2 panel")
    assert set(signals) == set(Z.CLUSTER_OF) and len(signals) == Z.N_FACTORS, (
        "the zoo's cluster map and the spec's signals differ")
    print(f"[data] {len(data):,} rows  {data['date'].min():%Y-%m-%d}"
          f" .. {data['date'].max():%Y-%m-%d}  {len(signals)} signals")
    return data, signals


def run_batch(data: pd.DataFrame, signals: list[str], *, sort: str) -> pd.DataFrame:
    from PyBondLab import NamingConfig, extract_panel
    from PyBondLab.fast_sorts import fast_single_sorts, fast_within_firm_sorts

    cols = {"ID": "cusip", "ret": "ret_vw", "VW": "mcap_e", "RATING_NUM": "spc_rat"}
    if sort == "single":
        res = fast_single_sorts(data, signals, columns=cols, num_portfolios=10,
                                dynamic_weights=True)
    else:
        res = fast_within_firm_sorts(data, signals, columns=cols,
                                     firm_id_col=S.FIRM_ID_COL)
    return extract_panel(res, naming=NamingConfig(sign_correct=True))


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
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    prov = pblenv.use()
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
                panel = run_batch(data, signals, sort=sort)
                D.write_atomic(panel, out, index=False)
                print(f"[done] {out.name}  {len(panel):,} rows  "
                      f"{time.perf_counter() - t0:.1f}s")
        bench.note(sorts=args.sorts, end=args.end, excess=args.excess,
                   n_signals=len(signals or []))
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
                    "inputs": [D.fingerprint(p) for p in
                               [paths.PANEL, paths.FACTORS, *R.input_paths()]],
                    **R.manifest(),
                }}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str),
                                        encoding="utf-8")
    print(f"\nwrote {root / 'manifest.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
