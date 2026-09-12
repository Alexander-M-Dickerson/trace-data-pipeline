r"""t17_filter_paths.py -- Table IA.XVII: filter-path improvement by cluster, filter
type and tail location.

Three panels x 9 clusters x 9 columns -- (left, right, both) tail x (trim, price,
bounce) filter:

  Panel A  how many paths improve the result, and what share of that cell's paths
  Panel B  the total path count per cell, which is fixed by the grid's design
  Panel C  the improving count alone

A path IMPROVES iff t(alpha) > 1.96 AND t(alpha) > the baseline t AND alpha > the
baseline alpha, per (signal, rating, weighting). All three conditions, not any: a
filter that lifts alpha while widening its standard error has not improved anything.

❗Panel B is a design invariant, not a result: per signal each tail column holds 96
trim, 60 price and 60 bounce paths, so each cell is that times the cluster's size. It
is checked every run -- if it fails, the grid is short and every percentage in Panel A
is computed against the wrong denominator.

    python s3_nse/run_dua_grid.py --stats
    python s3_nse/t17_filter_paths.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # stage3/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import captions             # noqa: E402
import clusters as C        # noqa: E402
import drrlib as D          # noqa: E402
import latex_format as F    # noqa: E402
import nse_engine as E      # noqa: E402
import paths                # noqa: E402

# ❗Set by main() before rendering, from the data this driver actually used.
# The caption TITLE is fixed in `captions.py`; this is the sentence that has to
# track the sample, so it is never written by hand.
_sample: dict = {}
_note = ""
from bench import Bench     # noqa: E402

LABEL = "tab:filter_paths"
# per signal, per tail column, by design
PER_SIGNAL = {"trim": 96, "price": 60, "bounce": 60}
N_SIGNALS = 108
PANELS = [("Panel A", "Improving paths, count (share)"),
          ("Panel B", "Total paths"),
          ("Panel C", "Improving paths")]


def render_latex(counts: pd.DataFrame) -> str:
    cols = [(loc, ft) for loc in E.LOCATIONS for ft in E.FILTER_TYPES]
    L = [r"\begin{table}[!ht]", r"\caption{" + captions.caption(LABEL, note=_note) + "}",
         r"\begin{center}", r"\label{" + LABEL + r"}", r"\scalebox{0.72}{%",
         r"\begin{tabular}{l " + " ".join("r" * 3 for _ in E.LOCATIONS) + "}",
         r"\toprule"]
    L.append(" & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{loc.capitalize()}}}"
                                for loc in E.LOCATIONS) + r" \\")
    L.append(" ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}"
                      for i in range(len(E.LOCATIONS))))
    L.append("Cluster & " + " & ".join(ft.capitalize() for _, ft in cols) + r" \\")
    idx = counts.set_index(["cluster_name", "location", "filter_type"])
    for panel, subtitle in PANELS:
        L += [r"\midrule",
              rf"\multicolumn{{{len(cols) + 1}}}{{l}}{{\textbf{{{panel}:}} "
              rf"{subtitle}}} \\", r"\midrule"]
        for gname in C.GROUP_NAMES:
            cells = [F.escape(gname)]
            for loc, ft in cols:
                r = idx.loc[(gname, loc, ft)]
                if panel == "Panel A":
                    pct = "" if pd.isna(r["pct"]) else f" ({r['pct']:.0f}\\%)"
                    cells.append(f"{int(r['n_improving'])}{pct}")
                elif panel == "Panel B":
                    cells.append(F.thousands(r["n_total"]))
                else:
                    cells.append(str(int(r["n_improving"])))
            L.append(F.row(cells))
    L += [r"\bottomrule", r"\end{tabular}", "}", r"\end{center}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", choices=E.DUA_WINDOWS, default="paper")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()

    with Bench(f"table-ia17-{args.window}", section="s3_nse",
               sample=not args.no_bench, echo=True) as b:
        with b.phase("load"):
            dua = E.load_dua_paths(window=args.window)
            baselines = E.load_dua_baselines(window=args.window)
        with b.phase("stats"):
            improv = E.dua_improvements(dua, baselines)
            counts = E.dua_filter_paths_counts(improv)
            per = counts.groupby(["location", "filter_type"])["n_total"].sum()
            design_ok = all(int(per[(loc, ft)]) == exp * N_SIGNALS
                            for loc in E.LOCATIONS
                            for ft, exp in PER_SIGNAL.items())
        global _sample, _note
        lo, hi = E.window_span(args.window)
        _sample = D.sample_block(
            first=lo, last=hi, window=args.window,
            n_paths=int(counts["n_total"].sum()), paths_label="filter paths",
            basis="filter paths, not a time series")
        _note = D.sample_sentence(_sample)
        with b.phase("render"):
            out = paths.section_results("s3_nse")
            counts.to_csv(out / f"table_ia17_filter_paths_{args.window}.csv",
                          index=False)
            tex = paths.TABLES / f"table_ia17_{args.window}.tex"
            tex.write_text(render_latex(counts), encoding="utf-8")
            D.write_result(
                f"table_ia17_{args.window}",
                {"summary": {"exhibit": "Table IA.XVII", "tex_label": LABEL,
                             "sample": _sample,
                             "window": args.window,
                             "n_improving_total": int(counts["n_improving"].sum()),
                             "n_paths_total": int(counts["n_total"].sum()),
                             "panel_b_design_ok": bool(design_ok)},
                 "cells": counts.to_dict("records")},
                section="s3_nse",
                inputs=[E._dua_file(n, args.window) for n in ("premia", "alpha")],
                t0=t0, extra={"exhibit": "Table IA.XVII", "tex_label": LABEL,
                              "window": args.window})
        b.note(window=args.window,
               n_improving=int(counts["n_improving"].sum()),
               n_paths=int(counts["n_total"].sum()))
        ok = b.check(design_ok,
                     "Panel B recovers the grid design "
                     f"({', '.join(f'{k} {v}' for k, v in PER_SIGNAL.items())} per "
                     f"signal per tail x {N_SIGNALS} signals)")

    print(f"\nTable IA.XVII ({LABEL}), window={args.window}: "
          f"{int(counts['n_improving'].sum()):,} improving of "
          f"{int(counts['n_total'].sum()):,} paths")
    if not ok:
        print("Panel B totals per (location, filter_type):")
        print(per.to_string())
    print(f"wrote {tex}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
