r"""t18_portfolio_size.py -- Table IA.XVIII: long-short portfolio size by design dimension.

Seventeen rows -- 3 portfolio counts, 3 rating universes, 4 maturity buckets, 3
breakpoint universes, 3 tail bins, and All -- x six columns: the average, median,
minimum and 5th percentile of how many bonds a strategy actually held, the share of
months holding fewer than 20, and the number of specs averaged over.

The point of the table is that some defensible constructions produce portfolios far too
small to be real: a decile of high-yield bonds in a narrow maturity bucket can be a
handful of names.

❗Counts are REALISED -- how many bonds the portfolio held in the return month, not how
many were selected at formation. A bond selected at formation that has no return the
next month is not in the portfolio that earned the return.

❗Two rows are both naturally labelled "All" (rating-All and maturity-All). They are
distinct rows and are prefixed here so they cannot be confused.

    python s3_nse/mua_summarize.py
    python s3_nse/t18_portfolio_size.py
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
import drrlib as D          # noqa: E402
import latex_format as F    # noqa: E402
import nse_engine as E      # noqa: E402
import paths                # noqa: E402
from bench import Bench     # noqa: E402

LABEL = "tab:mua_portfolio_size"
COLS = [("avg", "Average", 1), ("med", "Median", 1), ("min", "Min", 1),
        ("p05", "5th pct.", 1), ("pct_low", r"\%Low", 1), ("n_spec", r"$N$", 0)]
from mua_summarize import LOW_BOND_THRESHOLD as LOW_THRESHOLD   # noqa: E402
# ❗Imported, not restated. The printed footnote below quotes this number and the
# `pct_low` column is COMPUTED from it -- two copies could disagree silently.


def render_latex(df: pd.DataFrame) -> str:
    L = [r"\begin{table}[!ht]", r"\caption{" + captions.caption(LABEL) + "}",
         r"\begin{center}", r"\label{" + LABEL + r"}", r"\scalebox{0.88}{%",
         r"\begin{tabular}{l rrrrrr}", r"\toprule",
         "Design dimension & " + " & ".join(h for _, h, _ in COLS) + r" \\",
         r"\midrule"]
    for _, r in df.iterrows():
        if r["row"] == "All specifications":
            L.append(r"\midrule")
        cells = [F.escape(r["row"])]
        for key, _, dec in COLS:
            cells.append(F.thousands(r[key]) if key == "n_spec"
                         else F.num(r[key], dec))
        L.append(F.row(cells))
    L += [r"\bottomrule", r"\end{tabular}", "}",
          rf"\\[0.5em]\footnotesize \%Low is the share of months holding fewer than "
          rf"{LOW_THRESHOLD} bonds. Counts are realised, not formation.",
          r"\end{center}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", choices=("paper", "full"), default="paper")
    ap.add_argument("--twin", choices=("feb", "mar14"), default="feb",
                    help="which member of the redundant twin pair to keep. ❗Unlike "
                         "Table 6, this DOES move cells here: the two members carry "
                         "different breakpoint-universe labels, so the BP rows "
                         "re-attribute. ❗The strategy count is NOT conserved "
                         "when the engine forms one twin member and not the "
                         "other: feb 18,038 vs mar14 18,024 on the 2026-09-11 "
                         "build. See nse_engine.twin_asymmetry.")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()

    with Bench(f"table-ia18-{args.window}", section="s3_nse",
               sample=not args.no_bench, echo=True) as b:
        with b.phase("load"):
            ls = E.load_mua_nbonds(twin=args.twin, window=args.window)
        with b.phase("stats"):
            ours = E.mua_portfolio_size(ls)
        with b.phase("render"):
            out = paths.section_results("s3_nse")
            ours.to_csv(out / f"table_ia18_portfolio_size_{args.window}.csv",
                        index=False)
            tex = paths.TABLES / f"table_ia18_{args.window}.tex"
            tex.write_text(render_latex(ours), encoding="utf-8")
            D.write_result(
                f"table_ia18_{args.window}",
                {"summary": {"exhibit": "Table IA.XVIII", "tex_label": LABEL,
                             "window": args.window, "twin": args.twin,
                             "n_rows": len(ours),
                             "n_strategies": ls.attrs["n_strategies"],
                             "n_degenerate": ls.attrs["n_degenerate"]},
                 "rows": ours.to_dict("records")},
                section="s3_nse",
                inputs=[E.MUA_SUMMARY_DIR / f"mua_nbonds_{args.window}.parquet"],
                t0=t0, extra={"exhibit": "Table IA.XVIII", "tex_label": LABEL,
                              "window": args.window, "twin": args.twin})
        b.note(window=args.window, twin=args.twin, n_rows=len(ours),
               n_strategies=ls.attrs["n_strategies"],
               n_degenerate=ls.attrs["n_degenerate"])
        # Each of the four partitions must cover every strategy exactly once; a row
        # set that does not sum to the total means a spec fell outside its dimension.
        total = int(ours[ours["row"] == "All specifications"]["n_spec"].iloc[0])
        partitions = {
            "nport": ["Terciles", "Quintiles", "Deciles"],
            "rating": ["Rating: All", "Rating: IG", "Rating: NIG"],
            "maturity": ["Maturity: All", "Maturity: Short",
                         "Maturity: Intermediate", "Maturity: Long"],
            "bp": ["BP: Full universe", "BP: IG bonds", "BP: Large bonds"],
            # ❗The tail bins partition the strategies too, and were the one grouping
            # nothing checked. In the printed paper all five groupings sum to the same
            # 18,128; that identity is the property worth enforcing.
            "tailbin": ["Tail bin: <200", "Tail bin: 200-600", "Tail bin: >600"],
        }
        sums = {k: int(ours[ours["row"].isin(v)]["n_spec"].sum())
                for k, v in partitions.items()}
        ok = b.check(len(ours) == 17 and all(v == total for v in sums.values()),
                     f"{len(ours)} rows; each partition sums to {total:,} "
                     f"({sums})")

    print(f"\nTable IA.XVIII ({LABEL}), window={args.window}: "
          f"{ls.attrs['n_strategies']:,} strategies, "
          f"{ls.attrs['n_degenerate']} degenerate excluded")
    print(ours[["row", "avg", "med", "min", "pct_low", "n_spec"]]
          .round(1).to_string(index=False))
    print(f"\nwrote {tex}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
