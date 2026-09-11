r"""t19_mua_improvement.py -- Table IA.XIX: construction improvement by cluster and
methodology dimension.

Three panels x 9 clusters plus Total x 10 columns -- 3 rating universes, 4 maturity
buckets, 3 breakpoint universes:

  Panel A  improving specs, count and share
  Panel B  the pool size in that cell
  Panel C  the improving count alone

A spec IMPROVES iff its alpha beats the baseline spec's AND its t beats the baseline's
AND its t clears 1.96.

❗The two pools are NOT the same, deliberately. The NUMERATOR excludes only the single
baseline spec, so the other five *_all_all_all twins can themselves count as
improvements. The DENOMINATOR excludes all six per signal. Using one pool for both
undercounts Panel C.

    python s3_nse/mua_summarize.py
    python s3_nse/t19_mua_improvement.py
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
from bench import Bench     # noqa: E402

LABEL = "tab:mua_improvement"
KEYS = [f"{dim}:{val}" for dim, vals in E.MUA_COLUMN_GROUPS for _, val in vals]
HEADERS = [h for _, vals in E.MUA_COLUMN_GROUPS for h, _ in vals]
GROUP_SPANS = [(dim.capitalize(), len(vals)) for dim, vals in E.MUA_COLUMN_GROUPS]
PANELS = [("Panel A", "np", "Improving specifications, count (share)"),
          ("Panel B", "N", "Specifications in the pool"),
          ("Panel C", "n", "Improving specifications")]


def render_latex(df: pd.DataFrame) -> str:
    ncol = len(KEYS) + 1
    L = [r"\begin{table}[!ht]", r"\caption{" + captions.caption(LABEL) + "}",
         r"\begin{center}", r"\label{" + LABEL + r"}", r"\scalebox{0.68}{%",
         r"\begin{tabular}{l " + "r" * len(KEYS) + "}", r"\toprule"]
    L.append(" & " + " & ".join(rf"\multicolumn{{{n}}}{{c}}{{{name}}}"
                                for name, n in GROUP_SPANS) + r" \\")
    start = 2
    rules = []
    for _, n in GROUP_SPANS:
        rules.append(rf"\cmidrule(lr){{{start}-{start + n - 1}}}")
        start += n
    L.append(" ".join(rules))
    L.append("Cluster & " + " & ".join(HEADERS) + r" \\")
    t = df.set_index("cluster_name")
    for panel, kind, subtitle in PANELS:
        L += [r"\midrule",
              rf"\multicolumn{{{ncol}}}{{l}}{{\textbf{{{panel}:}} {subtitle}}} \\",
              r"\midrule"]
        for gname in C.GROUP_NAMES + ["Total"]:
            if gname == "Total":
                L.append(r"\midrule")
            r = t.loc[gname]
            cells = [F.escape(gname)]
            for k in KEYS:
                if kind == "np":
                    pct = r[f"{k}_pct"]
                    cells.append(f"{int(r[f'{k}_n'])}"
                                 + ("" if pd.isna(pct) else f" ({int(pct)}\\%)"))
                elif kind == "N":
                    cells.append(F.thousands(r[f"{k}_N"]))
                else:
                    cells.append(str(int(r[f"{k}_n"])))
            L.append(F.row(cells))
    L += [r"\bottomrule", r"\end{tabular}", "}", r"\end{center}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", choices=("paper", "full"), default="paper")
    ap.add_argument("--twin", choices=("feb", "mar14"), default="feb")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()

    with Bench(f"table-ia19-{args.window}", section="s3_nse",
               sample=not args.no_bench, echo=True) as b:
        with b.phase("load"):
            mua = E.load_mua_paths(twin=args.twin, window=args.window)
        with b.phase("stats"):
            # the pool size follows the data, so it is recorded rather than pinned
            ours = E.mua_improvement_counts(mua, expected_denominator=None)
            tot = ours[ours["cluster_name"] == "Total"].iloc[0]
            n_improving = int(sum(tot[f"rating:{v}_n"] for v in ("all", "ig", "hy")))
            n_pool = int(sum(tot[f"rating:{v}_N"] for v in ("all", "ig", "hy")))
        with b.phase("render"):
            out = paths.section_results("s3_nse")
            ours.to_csv(out / f"table_ia19_mua_improvement_{args.window}.csv",
                        index=False)
            tex = paths.TABLES / f"table_ia19_{args.window}.tex"
            tex.write_text(render_latex(ours), encoding="utf-8")
            D.write_result(
                f"table_ia19_{args.window}",
                {"summary": {"exhibit": "Table IA.XIX", "tex_label": LABEL,
                             "window": args.window, "twin": args.twin,
                             "n_improving": n_improving,
                             "n_denominator_pool": n_pool,
                             "n_paths": mua.attrs["n_paths"]},
                 "cells": ours.to_dict("records")},
                section="s3_nse",
                inputs=[E.MUA_SUMMARY_DIR / f"mua_summary_{args.window}.parquet"],
                t0=t0, extra={"exhibit": "Table IA.XIX", "tex_label": LABEL,
                              "window": args.window, "twin": args.twin})
        b.note(window=args.window, twin=args.twin, n_improving=n_improving,
               n_pool=n_pool, n_paths=mua.attrs["n_paths"])
        # Each dimension partitions the same pool: the three rating columns, the four
        # maturity columns and the three breakpoint columns must each sum to it.
        sums = {dim: int(sum(tot[f"{dim}:{val}_N"] for _, val in vals))
                for dim, vals in E.MUA_COLUMN_GROUPS}
        ok = b.check(len(ours) == len(C.GROUP_NAMES) + 1
                     and len(set(sums.values())) == 1,
                     f"{len(ours)} rows; every dimension partitions the same pool "
                     f"({sums})")

    print(f"\nTable IA.XIX ({LABEL}), window={args.window}: "
          f"{n_improving:,} improving of a {n_pool:,} pool")
    print(f"wrote {tex}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
