"""make_excess_blocks.py -- the 68 beta and momentum columns, for any Treasury benchmark.

    python make_excess_blocks.py --mode stage1 --benchmark bns
    python make_excess_blocks.py --mode stage1 --benchmark all      # bns + cls
    python make_excess_blocks.py --mode stage1 --verify             # the tret path, bit-identity

WHAT THIS IS FOR. The panel ships five Treasury benchmarks beside `tret`, but every
duration-adjusted quantity in it -- the 68 beta and momentum columns, plus `ret_vwx` and `str` --
is built from `ret_vw - tret`. So the panel offers alternative BENCHMARKS and not alternative
SYSTEMS: nothing can be sorted on a `tret_bns`-adjusted beta.

This closes that, as an OPTIONAL step run after a stage-2 build. It writes

    blocks/<mode>/betas_<bm>.parquet        the 45 beta + 6 sysmom columns
    blocks/<mode>/mom_retx_<bm>.parquet     the 14 momentum/LTR + 3 VaR columns

with the CANONICAL column names inside -- `b_amd`, not `b_amd_bns` -- so a block swaps in
wholesale, exactly as `betas_x` / `mom_retx` already do for `tret`.

WHY IT IS NOT A NEW RETURN TYPE. `RETURN_TYPES` is a hardcoded binary re-expressed at ten sites as
`"dur_adj" if ... else "std"`. Shipping blocks avoids all ten, and matches how the x-blocks already
travel: step 7 never reads them, they go out in the additional-data bundle, and the panel variant is
assembled downstream.

WHAT IT DOES NOT RECOMPUTE. The raw (std) side, `value_signals_*` (deliberately shared across
return types), and DEFB/TERMB (total-return legs). Only what actually moves with the benchmark.

❗THE BENCHMARK MUST REACH THE FACTORS TOO. A duration-adjusted beta is not the same regression on
a different y: the four bond-market factors AND `term` are swapped for twins estimated on the same
excess return. `compute_all_betas` raises if a variant's twins are absent rather than falling back
to the `tret` ones, because that failure produces b_* columns that look entirely normal and mean
nothing. Build them first with `step3_bbw.build(benchmark=...)`.

Author: Open Source Bond Asset Pricing
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "steps"))

import _stage2_settings as cfg                      # noqa: E402
from lib import betas as betalib                    # noqa: E402
from lib import momentum as momlib                  # noqa: E402
from lib import var_es                              # noqa: E402
import step4_betas                                  # noqa: E402
import step6_momentum                               # noqa: E402

BENCHMARKS = ("bns", "cls")
VAR_COLS = step6_momentum.VAR_COLS


def tret_col(bm: str) -> str:
    """The benchmark's column. The incumbent is plain `tret`, not `tret_tret` -- which is what
    lets --verify run the SAME code path as a benchmark build."""
    return "tret" if bm == "tret" else f"tret_{bm}"


def _betas(blocks_dir: Path, factors: pd.DataFrame, bm: str, verbose: bool) -> pd.DataFrame:
    """step 4's half: rolling betas + systematic momentum, for one benchmark."""
    combined = step4_betas.build_combined_returns(blocks_dir, tret_col=tret_col(bm))
    variants = ((bm, "ret_vwx", bm),)
    (b,) = betalib.compute_all_betas(
        combined_returns=combined, factors=factors,
        window=cfg.BETA_WINDOW, min_obs=cfg.BETA_MIN_OBS, verbose=verbose, variants=variants)
    (m,) = betalib.compute_sys_momentum(
        combined_returns=combined, factors=factors, factor_cols=["mktb"],
        window=cfg.BETA_WINDOW, min_obs=cfg.BETA_MIN_OBS, verbose=verbose, variants=variants)
    return b.merge(m, on=["cusip", "date"], how="left")


def _momentum(blocks_dir: Path, bm: str) -> pd.DataFrame:
    """step 6's half: momentum/LTR/industry + rolling VaR, for one benchmark."""
    are = pd.read_parquet(blocks_dir / "all_returns_ext.parquet")
    are["cusip"] = are["cusip"].astype(str)
    are["date"] = pd.to_datetime(are["date"])

    fisd = pd.read_parquet(cfg.AUX["fisd"], columns=["complete_cusip", "sic_code"])
    fisd = fisd.drop_duplicates(subset=["complete_cusip"]).rename(
        columns={"complete_cusip": "cusip"})
    are = are.merge(fisd[["cusip", "sic_code"]], on="cusip", how="left")
    are["ret_vwx"] = are["ret_vw"] - are[tret_col(bm)]

    _, mom_x = momlib.build_mom_ltr_and_industry(are)
    v = var_es.compute_rolling_var_es(are, id_col="cusip", ret_col="ret_vwx")[VAR_COLS]
    return mom_x.merge(v, on=["cusip", "date"], how="left")


def _compare(got: pd.DataFrame, want_path: Path, label: str) -> bool:
    """Bit-identity against a shipped block. The whole basis for trusting the benchmark runs."""
    want = pd.read_parquet(want_path)
    if list(got.columns) != list(want.columns):
        only_w = [c for c in want.columns if c not in got.columns]
        only_g = [c for c in got.columns if c not in want.columns]
        print(f"  {label}: COLUMNS MOVED  only-shipped={only_w}  only-rebuilt={only_g}")
        return False
    if len(got) != len(want):
        print(f"  {label}: rows {len(want):,} -> {len(got):,}")
        return False
    g = got.sort_values(["cusip", "date"]).reset_index(drop=True)
    w = want.sort_values(["cusip", "date"]).reset_index(drop=True)
    worst, wc, nulls = 0.0, None, []
    for c in w.columns:
        if c in ("cusip", "date"):
            continue
        a, b = g[c].to_numpy("float64"), w[c].to_numpy("float64")
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            nulls.append(c)
        m = ~np.isnan(a) & ~np.isnan(b)
        d = float(np.abs(a[m] - b[m]).max()) if m.any() else 0.0
        if d > worst:
            worst, wc = d, c
    ok = worst == 0.0 and not nulls
    print(f"  {label}: {len(w):,} x {len(w.columns)}  "
          f"{'BIT-IDENTICAL' if ok else 'DIFFERS'}  worst |d| {worst:.3e}"
          + (f" on {wc}" if worst else "")
          + (f"  null-mask moved: {nulls}" if nulls else ""))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default=None, help=f"block vintage (default {cfg.INPUT_MODE})")
    ap.add_argument("--benchmark", default="all",
                    help="bns | cls | all  (the benchmarks to build)")
    ap.add_argument("--verify", action="store_true",
                    help="run the TRET path instead and require bit-identity with the shipped "
                         "betas_x / mom_retx. Writes nothing.")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    mode = a.mode or cfg.INPUT_MODE
    blocks = cfg.BLOCKS_DIR / mode
    verbose = not a.quiet

    print(f"factor matrix from {blocks} ...")
    factors = step4_betas.build_factor_matrix(blocks)
    print(f"  {factors.shape[0]} months x {factors.shape[1]} columns")

    if a.verify:
        # The gate. Same code path, `tret` as the benchmark, against the shipped blocks -- so a
        # refactor that moved something is caught before any benchmark output is believed.
        print("\nVERIFY: rebuilding the tret path through this runner ...")
        t0 = time.time()
        b = _betas(blocks, factors, "tret", verbose)
        ok = _compare(b, blocks / "betas_x.parquet", "betas_x")
        m = _momentum(blocks, "tret")
        ok &= _compare(m, blocks / "mom_retx.parquet", "mom_retx")
        print(f"\n{'PASS' if ok else 'FAIL'} in {time.time() - t0:.0f}s")
        return 0 if ok else 1

    todo = list(BENCHMARKS) if a.benchmark == "all" else [a.benchmark]
    unknown = [b for b in todo if b not in BENCHMARKS]
    if unknown:
        raise SystemExit(f"unknown benchmark(s) {unknown}; known: {list(BENCHMARKS)}")

    report = {"mode": mode, "benchmarks": {}}
    for bm in todo:
        print(f"\n=== {bm} ===")
        t0 = time.time()
        b = _betas(blocks, factors, bm, verbose)
        m = _momentum(blocks, bm)
        pb = blocks / f"betas_{bm}.parquet"
        pm = blocks / f"mom_retx_{bm}.parquet"
        b.to_parquet(pb, index=False)
        m.to_parquet(pm, index=False)
        wall = round(time.time() - t0, 1)
        span = (str(pd.to_datetime(b["date"]).min())[:10], str(pd.to_datetime(b["date"]).max())[:10])
        print(f"  betas_{bm}      {len(b):,} x {len(b.columns)}   {span[0]} .. {span[1]}")
        print(f"  mom_retx_{bm}   {len(m):,} x {len(m.columns)}")
        print(f"  {wall}s")
        report["benchmarks"][bm] = {
            "wall_s": wall, "betas_rows": int(len(b)), "betas_cols": int(len(b.columns)),
            "mom_rows": int(len(m)), "mom_cols": int(len(m.columns)), "span": list(span)}

    (blocks / "excess_blocks_meta.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nmeta -> {blocks / 'excess_blocks_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
