r"""t_ia_daily.py -- Tables IA.I and IA.II: the DAILY panel's coverage and descriptives.

  IA.I   per variable and rating bucket: how many observations exist, and what share
         of that bucket's bond-days are missing it
  IA.II  per variable: mean, median, SD and the 1/5/95/99th percentiles, pooled over
         every bond-day (Panel A) and averaged across daily cross-sections (Panel B)

❗The two panels are not two views of one number. Panel A pools every observation, so a
heavily traded bond counts once per trade day. Panel B computes each day's statistic
first and then averages, so every day counts equally. They answer different questions
and it is normal for them to differ.

❗Percent-missing is measured against the BUCKET's row count, not the panel's -- so the
default bucket's missingness is about defaulted bonds, not about how few of them there
are.

Everything heavy runs as one grouped pass in DuckDB rather than a 30-million-row pandas
groupby, which is the difference between seconds and not finishing.

    python s0_data/t_ia_daily.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

for _p in (str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import captions             # noqa: E402
import data_engine as E     # noqa: E402
import drrlib as D          # noqa: E402
import latex_format as F    # noqa: E402
import paths                # noqa: E402
from bench import Bench     # noqa: E402

LAB1, LAB2 = "tab:data_availability", "tab:descriptive_stats"


def render_availability(avail: pd.DataFrame, label: str, vars_: list) -> str:
    piv = avail.set_index(["variable", "bucket"])
    L = [r"\begin{table}[!ht]", r"\caption{" + captions.caption(label) + "}",
         r"\begin{center}", r"\label{" + label + r"}", r"\scalebox{0.88}{%",
         r"\begin{tabular}{l " + " ".join("rr" for _ in E.RATING_BUCKETS) + "}",
         r"\toprule"]
    L.append(" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{b}}}"
                                for b in E.RATING_BUCKETS) + r" \\")
    L.append(" ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}"
                      for i in range(len(E.RATING_BUCKETS))))
    L.append("Variable & " + " & ".join(r"Obs. & \%Miss." for _ in E.RATING_BUCKETS)
             + r" \\")
    L.append(r"\midrule")
    for _, disp in vars_:
        cells = [F.escape(disp)]
        for b in E.RATING_BUCKETS:
            r = piv.loc[(disp, b)]
            cells += [F.thousands(int(r.observations)), F.num(float(r.pct_missing), 2)]
        L.append(F.row(cells))
    L += [r"\bottomrule", r"\end{tabular}", "}", r"\end{center}", r"\end{table}"]
    return "\n".join(L) + "\n"


def render_stats(pooled: pd.DataFrame, cross: pd.DataFrame, label: str) -> str:
    L = [r"\begin{table}[!ht]", r"\caption{" + captions.caption(label) + "}",
         r"\begin{center}", r"\label{" + label + r"}", r"\scalebox{0.85}{%",
         r"\begin{tabular}{l " + "r" * len(E.STAT_COLS) + "}", r"\toprule",
         "Variable & " + " & ".join(E.STAT_COLS) + r" \\"]
    for panel, df, subtitle in (("Panel A", pooled, "Pooled across all bond-days"),
                                ("Panel B", cross,
                                 "Daily cross-sections, averaged over time")):
        L += [r"\midrule",
              rf"\multicolumn{{{len(E.STAT_COLS) + 1}}}{{l}}{{\textbf{{{panel}:}} "
              rf"{subtitle}}} \\", r"\midrule"]
        for _, r in df.iterrows():
            L.append(F.row([F.escape(r["Variable"])]
                           + [F.num(r[c], 2) for c in E.STAT_COLS]))
    L += [r"\bottomrule", r"\end{tabular}", "}", r"\end{center}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--end", default=None,
                    help="optional sample end; default is the daily panel's frontier")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()

    if paths.DAILY is None or not Path(paths.DAILY).exists():
        raise SystemExit(
            f"the data appendix needs Stage 1's daily panel and it is not at "
            f"{paths.DAILY}.\n  Set STAGE1_DAILY, or run `python tools/check_inputs.py`.")

    with Bench("tables-ia1-ia2", section="s0_data", sample=not args.no_bench,
               echo=True) as b:
        with b.phase("dates"):
            span = E.daily_date_range(end=args.end)
        with b.phase("availability"):
            avail = E.daily_availability(end=args.end)
        with b.phase("pooled"):
            pooled = E.daily_pooled(end=args.end)
        with b.phase("cross"):
            cross = E.daily_cross_sectional(end=args.end)
        with b.phase("render"):
            out = paths.section_results("s0_data")
            avail.to_csv(out / "table_ia1_availability.csv", index=False)
            pooled.to_csv(out / "table_ia2_pooled.csv", index=False)
            cross.to_csv(out / "table_ia2_cross_sectional.csv", index=False)
            (paths.TABLES / "table_ia1.tex").write_text(
                render_availability(avail, LAB1, E.DAILY_AVAIL_VARS), encoding="utf-8")
            (paths.TABLES / "table_ia2.tex").write_text(
                render_stats(pooled, cross, LAB2), encoding="utf-8")
            D.write_result(
                "tables_ia1_ia2",
                {"summary": {"exhibit": "Tables IA.I-IA.II",
                             "tex_labels": [LAB1, LAB2],
                             "sample": list(span),
                             "n_availability_rows": len(avail),
                             "n_pooled_vars": len(pooled),
                             "n_cross_vars": len(cross)},
                 "availability": avail.to_dict("records"),
                 "pooled": pooled.to_dict("records"),
                 "cross_sectional": cross.to_dict("records")},
                section="s0_data", inputs=[paths.DAILY], t0=t0,
                extra={"exhibit": "Tables IA.I-IA.II"})
        b.note(sample=list(span), n_availability_rows=len(avail))
        # Both panels of IA.II must cover the same variables, or the table prints two
        # different sets of rows under one header.
        ok = b.check(
            len(avail) == len(E.DAILY_AVAIL_VARS) * len(E.RATING_BUCKETS)
            and list(pooled["Variable"]) == list(cross["Variable"]),
            f"{len(avail)} availability rows "
            f"({len(E.DAILY_AVAIL_VARS)} vars x {len(E.RATING_BUCKETS)} buckets); "
            f"{len(pooled)} variables in both IA.II panels")

    print(f"\nTables IA.I-IA.II: daily panel {span[0]} to {span[1]}")
    print(f"wrote {paths.TABLES / 'table_ia1.tex'}\n      {paths.TABLES / 'table_ia2.tex'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
