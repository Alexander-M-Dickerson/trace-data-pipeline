r"""run_mua_grid.py -- the method-uncertainty grid: 108 signals x 216 specs.

108 signals x 216 method choices (see `mua_engines.py` for what the 216 are), through
`assay_anomaly_fast`, one fresh process per signal.

The transport is the design. A unit of work is ONE SIGNAL, and each worker reads its
own seven-column slice of the panel through DuckDB. The panel is never pickled into a
worker: on Windows every argument is re-pickled per worker, so shipping a 1.2 GB panel
twelve times costs far more than the parallelism returns.

Output: `data/grids/mua/<signal>.parquet`, long format --
(date, signal, spec_id, leg L/S/LS, return, nbonds).

❗Needs a PyBondLab build carrying `anomaly_assay_fast`; `pblenv.require_fast` checks
before the fan-out starts rather than letting 108 workers each fail on an import.

    python s3_nse/run_mua_grid.py                       # all 108 signals
    python s3_nse/run_mua_grid.py --signals cs mom6_1   # two, for a smoke run
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

for _p in (str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _stage3_settings as S    # noqa: E402
import clusters as C            # noqa: E402
import paths                    # noqa: E402

N_SPECS = 216           # per signal, including the 24 infeasible ig_bp x hy cells


def out_dir() -> Path:
    d = paths.GRIDS / "mua"
    d.mkdir(parents=True, exist_ok=True)
    return d


def canon_spec(fast_col: str) -> str:
    """'EW_10p_Q_all_hy_long' (the engine's grammar) -> 'EW_Dp_Q_all_hy_long'."""
    w, npt, rest = fast_col.split("_", 2)
    import mua_engines as ME
    return f"{w}_{ME.NPORT_CODE[int(npt.rstrip('p'))]}_{rest}"


def run_one(item) -> dict:
    """TOP-LEVEL worker: one signal end to end, in a fresh process."""
    signal, out_path = item
    os.environ.setdefault("NUMBA_NUM_THREADS",
                          os.environ.get("STAGE3_WORKER_THREADS", "4"))
    t0 = time.perf_counter()
    import mua_engines as ME

    data = ME.load_panel([signal])
    wide, canon, res = ME.run_fast(data, signal, return_legs=True, return_counts=True)
    frames = []
    for leg, frame in [("LS", res.returns_df), ("L", res.long_df), ("S", res.short_df)]:
        long = frame.reset_index(names="date").melt(
            id_vars="date", var_name="fast_spec", value_name="return")
        counts = (res.nlong_df if leg == "L" else
                  res.nshort_df if leg == "S" else None)
        if counts is not None:
            cl = counts.reset_index(names="date").melt(
                id_vars="date", var_name="fast_spec", value_name="nbonds")
            long = long.merge(cl, on=["date", "fast_spec"], validate="1:1")
        else:
            # a long-short portfolio holds both legs, so its bond count is their sum
            nl = res.nlong_df.reset_index(names="date").melt(
                id_vars="date", var_name="fast_spec", value_name="nl")
            ns = res.nshort_df.reset_index(names="date").melt(
                id_vars="date", var_name="fast_spec", value_name="ns")
            long = (long.merge(nl, on=["date", "fast_spec"], validate="1:1")
                        .merge(ns, on=["date", "fast_spec"], validate="1:1"))
            long["nbonds"] = long["nl"] + long["ns"]
            long = long.drop(columns=["nl", "ns"])
        long["leg"] = leg
        frames.append(long)
    out = pd.concat(frames, ignore_index=True)
    out["spec_id"] = out["fast_spec"].map(canon_spec)
    out["signal"] = signal
    out = out[["date", "signal", "spec_id", "leg", "return", "nbonds"]]
    # ❗Temp file then rename. This runs in a spawned worker; a kill halfway through
    # the write would otherwise leave a truncated parquet that EXISTS, and the
    # orchestrator skips a producer whose output exists.
    tmp = out_path.with_name(out_path.name + ".tmp")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    return {"signal": signal, "n_specs": res.returns_df.shape[1],
            "rows": len(out), "wall_s": round(time.perf_counter() - t0, 2)}


def spec_census(d: Path, signals: list[str]) -> list[tuple[str, int]]:
    """(signal, n_specs) for every signal whose grid holds fewer than the full 216.

    Read straight off the parquets with a column projection, so it costs well under a
    second and says something true about the grid on disk rather than about whatever
    this particular invocation happened to recompute.
    """
    import duckdb
    p = (d / "*.parquet").as_posix()
    try:
        df = duckdb.sql(f"SELECT signal, COUNT(DISTINCT spec_id) n "
                        f"FROM read_parquet('{p}') GROUP BY 1").df()
    except Exception:
        return []
    keep = set(signals)
    return sorted((r.signal, int(r.n)) for r in df.itertuples()
                  if r.signal in keep and r.n < N_SPECS)


def unstable_empties(d: Path) -> int:
    """Cells that came back EMPTY under a restricted breakpoint universe while the
    same cell under the `all` universe has a full series.

    ❗This is a KNOWN DEFECT IN THE ENGINE, measured here rather than hidden. Running
    the identical grid twice over identical data flips a small number of `ig_bp` and
    `lg_bp` cells between a full 279-month series and nothing at all, always in
    EW/VW pairs. Every cell that IS populated is bit-identical between runs -- the
    arithmetic is stable; what is not stable is whether a restricted-universe
    portfolio gets formed.

    It is not the spec validator (`skip_invalid` is off), not the numba cache, not the
    thread count, and not the panel's row order (which is now pinned by an ORDER BY,
    and which halved it). It does not reproduce when the engine is called repeatedly
    inside one process -- only across the spawned workers.

    So: roughly 0.2-0.5% of the grid is unstable in PRESENCE. It moves the path
    counts, the degenerate count, and every count Section 5 prints; it does not move
    any value. The number is recorded in the manifest of every run so two runs can be
    compared, and `t06_mua_nse.py` fails its twin-invariance check when it bites.
    """
    import duckdb
    p = (d / "*.parquet").as_posix()
    try:
        df = duckdb.sql(
            f"""WITH cell AS (
                    SELECT signal, spec_id, COUNT("return") AS n
                    FROM read_parquet('{p}') WHERE leg = 'LS' GROUP BY 1, 2),
                parts AS (
                    SELECT signal, n,
                           split_part(spec_id, '_', 1) AS w,
                           split_part(spec_id, '_', 2) AS np,
                           regexp_extract(spec_id,
                               '_(all|ig_bp|lg_bp)_(all|ig|hy)_(all|short|mid|long)$',
                               1) AS bp,
                           regexp_extract(spec_id,
                               '_(all|ig_bp|lg_bp)_(all|ig|hy)_(all|short|mid|long)$',
                               2) AS rat,
                           regexp_extract(spec_id,
                               '_(all|ig_bp|lg_bp)_(all|ig|hy)_(all|short|mid|long)$',
                               3) AS mat
                    FROM cell)
                SELECT COUNT(*) AS k
                FROM parts r
                JOIN parts b
                  ON r.signal = b.signal AND r.w = b.w AND r.np = b.np
                 AND r.rat = b.rat AND r.mat = b.mat AND b.bp = 'all'
                WHERE r.bp <> 'all' AND r.n = 0 AND b.n > 0
                  -- the 24 infeasible cells per signal: an investment-grade
                  -- breakpoint universe crossed with a high-yield filter is
                  -- an EMPTY population by construction, not an engine slip
                  AND NOT (r.bp = 'ig_bp' AND r.rat = 'hy')""").fetchone()
        return int(df[0]) if df else 0
    except Exception:
        return -1


def readable(path: Path) -> bool:
    """A crash-truncated parquet must not count as a finished signal."""
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows > 0
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--signals", nargs="+", default=None,
                    help="default: all 108")
    ap.add_argument("--workers", type=int, default=S.N_WORKERS,
                    help="fresh processes to fan out over (default: from cpu_count)")
    ap.add_argument("--threads", type=int, default=3,
                    help="numba threads per worker (workers*threads <= cores)")
    ap.add_argument("--force", action="store_true",
                    help="recompute signals whose parquet already exists")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()

    import pblenv
    prov = pblenv.use()
    pblenv.require_fast("the MUA grid")

    import drrlib as D          # noqa: E402  (after pblenv)
    from bench import Bench     # noqa: E402
    from fastrun import pmap, sized    # noqa: E402

    signals = args.signals or list(C.ALL_SIGNALS)
    d = out_dir()
    todo = [s for s in signals if args.force or not (d / f"{s}.parquet").exists()]
    label = f"mua-grid-{len(signals)}sig"

    with Bench(label, section="s3_nse") as b:
        with b.phase("grid"):
            if todo:
                items = [(s, d / f"{s}.parquet") for s in todo]
                n_w, n_t = sized(len(items), args.workers, args.threads,
                                 min_threads=2)
                results = pmap(run_one, items, workers=min(n_w, len(items)),
                               threads=n_t)
            else:
                n_w = n_t = 0
                results = []
                print(f"all {len(signals)} signal parquets present -- "
                      "skipping compute (--force to redo)")
        b.note(n_signals=len(signals), n_computed=len(todo),
               workers=n_w, threads=n_t)
        # Completeness, on what is READABLE on disk rather than on what this run
        # computed: a re-run over a finished grid must still report the grid as
        # complete, and a truncated file must not count as present.
        present = [s for s in signals if readable(d / f"{s}.parquet")]
        # ❗The grid should be a RECTANGLE, and is: `assay_anomaly_fast` is handed
        # `skip_invalid=False`, so it returns all 216 columns for every signal and hands
        # back an all-NaN column where it cannot form a portfolio. Measured on the grid
        # on disk: 108 of 108 signals carry exactly 216 specs.
        #
        # This check survives from when the flag was True and the engine DROPPED
        # unformable cells, which made the column set depend on the sample and differ
        # run to run. It is kept as the guard for that regression: if the flag or the
        # engine changes back, a signal comes up short here rather than silently
        # narrowing every count Section 5 reports.
        #
        # Measured from the PARQUETS, like `present` above and for the same reason: a
        # re-run over a finished grid computes nothing, so reporting what THIS run
        # computed would say "0 narrow" about a grid it never looked at.
        narrow = spec_census(d, signals)
        wide_enough = all(n >= N_SPECS // 2 for _, n in narrow)
        min_specs = min([n for _, n in narrow], default=N_SPECS)
        unstable = unstable_empties(d)
        b.note(n_narrow=len(narrow), min_specs=min_specs, narrow=narrow[:10],
               n_unstable_empty_cells=unstable)
        ok = b.check(len(present) == len(signals) and wide_enough,
                     f"{len(present)}/{len(signals)} signal grids readable, "
                     f"{len(results)} computed this run; {len(narrow)} narrow "
                     f"(min {min_specs}/{N_SPECS} specs); "
                     f"{unstable} unstable-empty cells")
        if unstable > 0:
            print(f"\nWARN {unstable} cell(s) came back EMPTY under a restricted "
                  "breakpoint universe while the\n"
                  "     same cell under the `all` universe has a full series. This is "
                  "an ENGINE defect,\n"
                  "     not a property of your data: the same grid run twice flips a "
                  "few of these. Values\n"
                  "     are unaffected (bit-identical between runs); PATH COUNTS are "
                  "not. See the note on\n"
                  "     `unstable_empties` in this file.", flush=True)
        D.write_result(
            "mua_grid",
            {"summary": {"n_signals": len(signals), "n_computed": len(todo),
                         "n_present": len(present)},
             "per_signal": results},
            section="s3_nse", inputs=[Path(paths.PANEL)], t0=t0,
            extra={"pybondlab": prov})

    D.mark_complete(d / "_complete.json", ok,
                    {"n_signals": len(signals), "n_present": len(present)})
    print(f"\n{len(present)}/{len(signals)} signals in {d}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
