r"""t_ia_monthly.py -- Tables IA.III to IA.VII: the MONTHLY panel's descriptives.

  IA.III  coverage per variable and rating bucket
  IA.IV   descriptive statistics, pooled and cross-sectional
  IA.V    the return distribution's tails: far percentiles, skewness and excess
          kurtosis, and counts beyond +/-20%, +/-50%, +/-100% and +500%
  IA.VI   when those extreme returns happened, by year
  IA.VII  per-year return statistics, and the correlation between the month-end and
          month-begin return

Every table reports BOTH return conventions side by side -- month-end (`ret_vw`) and
month-begin (`ret_vw_bgn`) -- because the gap between them is what Section 3 is about.
IA.VII's rho is the matched-pair correlation within the year: how closely the two agree
on the same bond-months.

❗IA.IV prints 17 variables, not the 20 it lists as candidates. `pr`, `mod_dur` and
`conv` are not columns of the monthly panel, and an absent variable is SKIPPED rather
than printed as zero.

❗Percent-missing is against the BUCKET's row count, not the panel's.

    python s0_data/t_ia_monthly.py
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

# ❗Set by main() before anything renders, from the data this driver actually
# used. The caption TITLE is fixed in `captions.py`; this is the sentence that has
# to track the sample, so it is never written by hand.
_note = ""
import data_engine as E     # noqa: E402
import drrlib as D          # noqa: E402
import latex_format as F    # noqa: E402
import paths                # noqa: E402
from bench import Bench     # noqa: E402

LABELS = {"IA3": "tab:monthly_data_availability", "IA4": "tab:desc_stats_all",
          "IA5": "tab:extreme_returns", "IA6": "tab:time_concentration_extremes",
          "IA7": "tab:annual_return_stats"}
RET_COLS = [("End", "ret_vw"), ("Begin", "ret_vw_bgn")]


def _table(label: str, body: list[str], colspec: str, scale: str = "0.85") -> str:
    return "\n".join(
        [r"\begin{table}[!ht]", r"\caption{" + captions.caption(label, note=_note) + "}",
         r"\begin{center}", r"\label{" + label + r"}", r"\scalebox{" + scale + r"}{%",
         r"\begin{tabular}{" + colspec + "}", r"\toprule", *body,
         r"\bottomrule", r"\end{tabular}", "}", r"\end{center}", r"\end{table}"]) + "\n"


def render_ia3(avail: pd.DataFrame) -> str:
    body = [" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{b}}}"
                               for b in E.RATING_BUCKETS) + r" \\",
            " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}"
                     for i in range(len(E.RATING_BUCKETS))),
            "Variable & " + " & ".join(r"Obs. & \%Miss." for _ in E.RATING_BUCKETS)
            + r" \\", r"\midrule"]
    piv = avail.set_index(["variable", "bucket"])
    for v in avail["variable"].drop_duplicates():
        cells = [F.escape(str(v))]
        for b in E.RATING_BUCKETS:
            try:
                r = piv.loc[(v, b)]
                cells += [F.thousands(int(r.observations)),
                          F.num(float(r.pct_missing), 2)]
            except KeyError:
                cells += ["", ""]
        body.append(F.row(cells))
    return _table(LABELS["IA3"], body,
                  "l " + " ".join("rr" for _ in E.RATING_BUCKETS))


def render_ia4(pooled: pd.DataFrame, cross: pd.DataFrame) -> str:
    body = ["Variable & " + " & ".join(E.STAT_COLS) + r" \\"]
    for panel, df, subtitle in (("Panel A", pooled, "Pooled across all bond-months"),
                                ("Panel B", cross,
                                 "Monthly cross-sections, averaged over time")):
        body += [r"\midrule",
                 rf"\multicolumn{{{len(E.STAT_COLS) + 1}}}{{l}}{{\textbf{{{panel}:}} "
                 rf"{subtitle}}} \\", r"\midrule"]
        for _, r in df.iterrows():
            body.append(F.row([F.escape(r["Variable"])]
                              + [F.num(r[c], 2) for c in E.STAT_COLS]))
    return _table(LABELS["IA4"], body, "l " + "r" * len(E.STAT_COLS))


def _ret_bucket_cols() -> list[str]:
    return [f"{d}_{b}" for d, _ in RET_COLS for b in E.RATING_BUCKETS]


def render_ia5(ext: dict) -> str:
    cols = _ret_bucket_cols()
    body = [" & " + " & ".join(
        rf"\multicolumn{{{len(E.RATING_BUCKETS)}}}{{c}}{{{d}-of-month return}}"
        for d, _ in RET_COLS) + r" \\",
        " ".join(rf"\cmidrule(lr){{{2 + len(E.RATING_BUCKETS) * i}-"
                 rf"{1 + len(E.RATING_BUCKETS) * (i + 1)}}}"
                 for i in range(len(RET_COLS))),
        "Statistic & " + " & ".join(b for _ in RET_COLS for b in E.RATING_BUCKETS)
        + r" \\"]
    for title, df, key, dec in (("Tail percentiles (\\%)", ext["tail"], "Statistic", 2),
                                ("Moments", ext["moments"], "Statistic", 2),
                                ("Counts", ext["counts"], "Direction", 0)):
        body += [r"\midrule",
                 rf"\multicolumn{{{len(cols) + 1}}}{{l}}{{\textit{{{title}}}}} \\",
                 r"\midrule"]
        for _, r in df.iterrows():
            body.append(F.row([F.escape(str(r[key]))]
                              + [(F.thousands(r[c]) if dec == 0 else F.num(r[c], dec))
                                 for c in cols]))
    return _table(LABELS["IA5"], body, "l " + "r" * len(cols), scale="0.72")


def render_ia6(tc: pd.DataFrame) -> str:
    cols = [c for c in tc.columns if c != "Year"]
    body = ["Year & " + " & ".join(F.escape(c.replace("_", " ")) for c in cols) + r" \\",
            r"\midrule"]
    for _, r in tc.iterrows():
        body.append(F.row([str(int(r["Year"]))] + [F.thousands(r[c]) for c in cols]))
    return _table(LABELS["IA6"], body, "l " + "r" * len(cols), scale="0.75")


def render_ia7(an: pd.DataFrame, total: dict) -> str:
    cols = [c for c in an.columns if c != "Year"]
    body = ["Year & " + " & ".join(F.escape(c.replace("_", " ")) for c in cols) + r" \\",
            r"\midrule"]

    def fmt(c, v):
        return F.thousands(v) if c.startswith("N_") else F.num(v, 2)

    for _, r in an.iterrows():
        body.append(F.row([str(int(r["Year"]))] + [fmt(c, r[c]) for c in cols]))
    body += [r"\midrule",
             F.row([r"\textbf{Total}"] + [fmt(c, total.get(c)) for c in cols])]
    return _table(LABELS["IA7"], body, "l " + "r" * len(cols), scale="0.70")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--end", default=None,
                    help="optional sample end; default is the panel's frontier")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()

    with Bench("tables-ia3-ia7", section="s0_data", sample=not args.no_bench,
               echo=True) as b:
        with b.phase("load"):
            df = E.load_monthly(end=args.end)
            res = E.resample_monthly(df)
        with b.phase("stats"):
            avail = E.monthly_availability(res)
            pooled = E.monthly_pooled(df)
            cross = E.monthly_cross_sectional(df)
            ext = E.extreme_stats(df)
            tc = E.time_concentration(df)
            an = E.annual_stats(df)
            total = E.annual_total_row(an)
        global _note
        _note = D.sample_sentence(D.sample_block(
            first=str(df["date"].min())[:10], last=str(df["date"].max())[:10],
            basis="the monthly panel, to its own frontier"))
        with b.phase("render"):
            out = paths.section_results("s0_data")
            for name, frame in (("table_ia3_availability", avail),
                                ("table_ia4_pooled", pooled),
                                ("table_ia4_cross_sectional", cross),
                                ("table_ia5_tail", ext["tail"]),
                                ("table_ia5_moments", ext["moments"]),
                                ("table_ia5_counts", ext["counts"]),
                                ("table_ia6_time_concentration", tc),
                                ("table_ia7_annual", an)):
                frame.to_csv(out / f"{name}.csv", index=False)
            (paths.TABLES / "table_ia3.tex").write_text(render_ia3(avail), encoding="utf-8")
            (paths.TABLES / "table_ia4.tex").write_text(
                render_ia4(pooled, cross), encoding="utf-8")
            (paths.TABLES / "table_ia5.tex").write_text(render_ia5(ext), encoding="utf-8")
            (paths.TABLES / "table_ia6.tex").write_text(render_ia6(tc), encoding="utf-8")
            (paths.TABLES / "table_ia7.tex").write_text(
                render_ia7(an, total), encoding="utf-8")
            D.write_result(
                "tables_ia3_ia7",
                {"summary": {"exhibit": "Tables IA.III-IA.VII",
                             "tex_labels": list(LABELS.values()),
                             "n_rows": len(df), "n_years": len(an),
                             "n_printed_vars_ia4": len(pooled),
                             "sample": D.sample_block(
                                 first=str(df["date"].min().date()),
                                 last=str(df["date"].max().date()),
                                 basis="the monthly panel, to its own frontier")},
                 "availability": avail.to_dict("records"),
                 "pooled": pooled.to_dict("records"),
                 "cross_sectional": cross.to_dict("records"),
                 "annual": an.to_dict("records"), "annual_total": total},
                section="s0_data", inputs=[paths.PANEL], t0=t0,
                extra={"exhibit": "Tables IA.III-IA.VII"})
        b.note(n_rows=len(df), n_years=len(an), n_printed_vars_ia4=len(pooled))
        # IA.VI and IA.VII cover the same years, and IA.IV's two panels the same
        # variables. A mismatch means one table quietly dropped a year or a row.
        ok = b.check(
            list(tc["Year"]) == list(an["Year"])
            and list(pooled["Variable"]) == list(cross["Variable"]),
            f"{len(an)} years in both IA.VI and IA.VII; "
            f"{len(pooled)} variables in both IA.IV panels")

    print(f"\nTables IA.III-IA.VII: {len(df):,} bond-months, "
          f"{df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}, {len(an)} years")
    print(f"wrote {paths.TABLES / 'table_ia3.tex'} (and IA.4 to IA.7)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
