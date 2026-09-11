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
        wide_enough = all(r["n_specs"] >= N_SPECS - 24 for r in results) if results else True
        ok = b.check(len(present) == len(signals) and wide_enough,
                     f"{len(present)}/{len(signals)} signal grids readable, "
                     f"{len(results)} computed this run")
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
