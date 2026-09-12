r"""t06_mua_nse.py -- Table 6: non-standard errors from METHOD uncertainty, by cluster.

Table 5's twin, over construction choices instead of data cleaning. Each path is one
defensible way of building the same factor -- weighting, portfolio count, breakpoint
universe, rating filter, maturity bucket -- and the analysis set is what survives after
the exclusions below. Its size is reported by the run rather than assumed, because how
many strategies are degenerate depends on the data.

❗Ratio here uses INDEPENDENT skipna on std(value) and mean(SE). Table 5 uses the
pairwise-matched sample. Each table wants its own rule.

❗Sign correction is applied HERE, on the VW_Qp_Q_all_all_all baseline -- the tables'
baseline. The figures sign off VW_Dp_Q_all_all_all. Two baselines, deliberately, and
using one where the other belongs silently flips a subset of the factors.

The analysis set drops degenerate strategies (a leg empty in some month within its
active window) and the redundant specs: one member of the all_ig / ig_bp_ig twin pair,
which is the same portfolio labelled two ways, and the 24 infeasible
investment-grade-breakpoint x high-yield cells per signal. Excluding only the twins
overcounts. Which twin member is kept is a LABEL and cannot move a cell here, which
this run asserts. ❗It is NOT label-only in Table IA.XVIII, where the two members carry
different breakpoint-universe labels and the rows re-attribute.

    python s3_nse/run_mua_grid.py      # the grid
    python s3_nse/mua_summarize.py     # the statistics layer
    python s3_nse/t06_mua_nse.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # stage3/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cluster_table as CT  # noqa: E402
import drrlib as D          # noqa: E402
import nse_engine as E      # noqa: E402
import paths                # noqa: E402
from bench import Bench     # noqa: E402

LABEL = "tab:mu_nse_by_cluster"
TWIN_TOL = 1e-12   # relabelling is float-exact up to reassociation, nothing more


def flip_set(mua) -> set[str]:
    """The signals the table's baseline sign-corrects."""
    base = mua[mua["spec_id"] == E.TABLE_BASELINE].set_index("signal")["mean_ret"]
    return set(base[base.notna() & (base < 0)].index)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", choices=("paper", "full"), default="paper")
    ap.add_argument("--twin", choices=("mar14", "feb"), default="feb",
                    help="which member of the redundant twin pair to keep; a LABEL "
                         "choice that cannot move a cell, asserted each run")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()
    with Bench(f"table06-{args.window}", section="s3_nse",
               sample=not args.no_bench, echo=True) as b:
        with b.phase("load"):
            mua = E.load_mua_paths(twin=args.twin, window=args.window)
        with b.phase("stats"):
            ours = E.mua_cluster_summary(mua)
            # The twin convention is a relabelling. If a cell moves, the exclusion
            # rule is wrong rather than the label.
            other = "mar14" if args.twin == "feb" else "feb"
            alt = E.mua_cluster_summary(
                E.load_mua_paths(twin=other, window=args.window))
            twin_max_d = float(np.nanmax(np.abs(
                ours[CT.COLS[:-1]].to_numpy(float)
                - alt[CT.COLS[:-1]].to_numpy(float))))
            flips = flip_set(mua)
        with b.phase("render"):
            out = paths.section_results("s3_nse")
            ours.to_csv(out / f"table06_mu_nse_by_cluster_{args.window}.csv",
                        index=False)
            tex = paths.TABLES / f"table06_{args.window}.tex"
            # ❗Both exclusions, so the arithmetic closes: the footnote used to
            # report the empty-leg count alone, and a reader subtracting it from the
            # 18,144-strategy grid landed 32 short with nothing to explain the gap.
            note = (f"Ratio computed with independent skipna; "
                    f"{mua.attrs['n_paths']:,} construction paths of "
                    f"{E.GRID_STRATEGIES:,}, after excluding "
                    f"{mua.attrs['n_degenerate']} with an empty leg and "
                    f"{mua.attrs['n_no_series']} never formed; "
                    f"{len(flips)} signals sign-corrected.")
            lo, hi = E.window_span(args.window)
            sample = D.sample_block(first=lo, last=hi, window=args.window,
                                    n_paths=mua.attrs["n_paths"],
                                    
                                    paths_label="construction paths",basis="construction paths, not a time series")
            tex.write_text(CT.render_latex(ours, LABEL, note=note, sample=sample),
                           encoding="utf-8")
            D.write_result(
                f"table06_{args.window}",
                {"summary": {"exhibit": "Table 6", "tex_label": LABEL,
                             "window": args.window, "twin": args.twin,
                             "n_paths": mua.attrs["n_paths"], "sample": sample,
                             "n_degenerate": mua.attrs["n_degenerate"],
                             "n_signals_flipped": len(flips),
                             "twin_invariance_max_abs_diff": twin_max_d,
                             "n_cells": len(ours) * len(CT.COLS)},
                 "signals_flipped": sorted(flips),
                 "cells": CT.as_rows(ours)},
                section="s3_nse",
                inputs=[E.MUA_SUMMARY_DIR / f"mua_summary_{args.window}.parquet"],
                t0=t0, extra={"exhibit": "Table 6", "tex_label": LABEL,
                              "window": args.window, "twin": args.twin})
        b.note(window=args.window, twin=args.twin, n_paths=mua.attrs["n_paths"],
               n_signals_flipped=len(flips), twin_invariance=twin_max_d)
        # The two label conventions are supposed to cover the same portfolios, so a
        # cell could only differ by float reassociation. That holds while BOTH members
        # of each twin pair exist; when the engine forms one and not the other, the
        # conventions select different DATA and the statistics genuinely move.
        broken = E.twin_asymmetry(args.window)
        if broken:
            print(f"\nWARN the twin conventions disagree, and the reason is in the data:"
                  f"\n     {len(broken)} twin pair(s) have one member usable and the "
                  "other with no series at all,"
                  "\n     so `feb` and `mar14` select different data, not different "
                  "labels."
                  "\n     Examples: " + ", ".join(broken[:3])
                  + "\n     This is the sort engine's unstable empty cell, "
                    "described under"
                    "\n     'What is not reproducible' in README_stage3.md. No VALUE "
                    "is wrong; the"
                    "\n     cluster statistics move because the two sets are not the "
                    "same set.", flush=True)
        ok = b.check(len(ours) == 10 and twin_max_d <= TWIN_TOL,
                     f"{len(ours)} rows over {mua.attrs['n_paths']:,} paths; "
                     f"twin convention moves no cell (max|d|={twin_max_d:.2e}, "
                     f"tolerance {TWIN_TOL:g})"
                     + (f"; {len(broken)} twin pair(s) asymmetric" if broken
                        else ""))

    print(f"\nTable 6 ({LABEL}), window={args.window}: "
          f"{mua.attrs['n_paths']:,} paths, {len(flips)} signals flipped")
    print(ours[["cluster_name", "nse_mu", "ratio_mu", "nse_alpha",
                "ratio_alpha", "n_paths"]].round(2).to_string(index=False))
    print(f"\nwrote {tex}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
