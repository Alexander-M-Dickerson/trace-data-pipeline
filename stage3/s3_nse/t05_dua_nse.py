r"""t05_dua_nse.py -- Table 5: non-standard errors from DATA uncertainty, by cluster.

Ten rows (nine factor clusters plus All) x nine columns, over the 69,984 filter paths
of the DUA grid: how far a factor's premium and alpha move when the return panel is
cleaned in each of 120 defensible ways.

  NSE    the interquartile range of the estimate ACROSS paths
  Ratio  that spread divided by the average conventional standard error. Above 1 means
         the cleaning choice moves the answer more than sampling noise does.

❗Ratio here uses the PAIRWISE-dropna matched sample. Table 6 uses independent skipna.
The two rules give different numbers and each table wants its own -- do not unify them.

❗The grid's statistics are ALREADY sign-corrected and ALREADY in percent. Nothing here
scales or flips anything; `sign_mult_premia` in the source is a receipt of what was
done, not an instruction to do it again.

`--window paper` truncates each SERIES at the paper's sample end and recomputes;
`--window full` leaves it untruncated. The window can change a sign correction, because
the correction is decided on the baseline mean of whatever window was used.

    python s3_nse/run_dua_grid.py            # the grid (hours)
    python s3_nse/run_dua_grid.py --stats    # the statistics layer
    python s3_nse/t05_dua_nse.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # stage3/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cluster_table as CT  # noqa: E402
import drrlib as D          # noqa: E402
import nse_engine as E      # noqa: E402
import paths                # noqa: E402
from bench import Bench     # noqa: E402

LABEL = "tab:nse_by_cluster"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", choices=E.DUA_WINDOWS, default="paper")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()
    with Bench(f"table05-{args.window}", section="s3_nse",
               sample=not args.no_bench, echo=True) as b:
        with b.phase("load"):
            dua = E.load_dua_paths(window=args.window)
        with b.phase("stats"):
            ours = E.dua_cluster_summary(dua)
        with b.phase("render"):
            out = paths.section_results("s3_nse")
            ours.to_csv(out / f"table05_nse_by_cluster_{args.window}.csv", index=False)
            tex = paths.TABLES / f"table05_{args.window}.tex"
            note = (f"Ratio computed on the pairwise-matched sample; "
                    f"{dua.attrs['n_grid_paths']:,} filter paths; tail location from "
                    f"{dua.attrs['location_source']}.")
            lo, hi = E.window_span(args.window)
            sample = D.sample_block(first=lo, last=hi, window=args.window,
                                    n_paths=dua.attrs["n_grid_paths"],
                            paths_label="filter paths",
                                    basis="filter paths, not a time series")
            tex.write_text(CT.render_latex(ours, LABEL, note=note, sample=sample),
                           encoding="utf-8")
            D.write_result(
                f"table05_{args.window}",
                {"summary": {"exhibit": "Table 5", "tex_label": LABEL,
                             "sample": sample,
                             "window": args.window,
                             "n_grid_paths": dua.attrs["n_grid_paths"],
                             "location_source": dua.attrs["location_source"],
                             "n_cells": len(ours) * len(CT.COLS)},
                 "cells": CT.as_rows(ours)},
                section="s3_nse",
                inputs=[E._dua_file(n, args.window) for n in ("premia", "alpha")],
                t0=t0, extra={"exhibit": "Table 5", "tex_label": LABEL,
                              "window": args.window})
        b.note(window=args.window, n_rows=len(ours),
               n_grid_paths=dua.attrs["n_grid_paths"])
        # Ten rows, and the path pool must be the whole grid: a short grid would
        # narrow every NSE in the table without anything looking wrong.
        ok = b.check(len(ours) == 10 and dua.attrs["n_grid_paths"] == E.N_DUA_PATHS,
                     f"{len(ours)} rows over {dua.attrs['n_grid_paths']:,} filter "
                     f"paths (expect 10 over {E.N_DUA_PATHS:,})")

    print(f"\nTable 5 ({LABEL}), window={args.window}:")
    print(ours[["cluster_name", "nse_mu", "ratio_mu", "nse_alpha",
                "ratio_alpha", "n_paths"]].round(2).to_string(index=False))
    print(f"\nwrote {tex}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
