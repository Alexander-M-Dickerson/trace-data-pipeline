r"""t02_validation.py -- Table 2, "Latent implementation bias validation".

Table 1 measures the bias. This tests the EXPLANATION for it, by decomposition:

    r_End  ~=  LIB  +  r_Bgn

If the gap between the month-end and month-begin return really is the portfolio's
latent implementation bias, then the difference of the two return series should equal
the portfolio-level `lib` characteristic -- not merely correlate with it, but match in
level, leaving a residual near zero.

Eight numeric cells per factor: mu_End, mu_Bgn, d_mu, t(d_mu), mu_LIB, t(mu_LIB), the
correlation rho, and the Residual. Two panels x 7 factors = 112 cells.

`mu_LIB` comes from the portfolio-level `lib` characteristic carried in the month-begin
sort CSVs, which is why `run_sorts.py` runs that set with chars=['lib','ilq'].
❗Its sign-flip undo is the SAME as the return series': when PyBondLab sign-corrects it
swaps the legs, which negates the characteristic spread exactly as it negates the
return.

    python s1_lib/t02_validation.py
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
import lib_engine as E      # noqa: E402
import paths                # noqa: E402
from bench import Bench     # noqa: E402
from two_row import DEFAULT_PANEL_SPECS, DEFAULT_SUBTITLES, _fmt  # noqa: E402

LABEL = "tab:mmn_2"
FACTORS = D.LIB_FACTORS
COLUMNS = ["mu_end", "mu_bgn", "dmu", "t_dmu", "mu_lib", "t_mu_lib", "rho", "resid"]
HEADERS = [r"$\mu_{\text{End}}$", r"$\mu_{\text{Bgn}}$", r"$\Delta\mu$",
           r"$t(\Delta\mu)$", r"$\mu_{\text{LIB}}$", r"$t(\mu_{\text{LIB}})$",
           r"$\rho$", "Residual"]
DEC = 2


def compute(*, end: str | None = None) -> pd.DataFrame:
    import dataclasses

    specs = list(DEFAULT_PANEL_SPECS.values())
    expected_T: int | None = 268
    if end:
        specs = [dataclasses.replace(sp, end=end) for sp in specs]
        expected_T = None
    return pd.concat([E.validation_stats(sp, FACTORS, root=paths.SORTS,
                                         expected_T=expected_T) for sp in specs],
                     ignore_index=True)


def as_cells(stats: pd.DataFrame) -> dict:
    out = {}
    for panel, sp in DEFAULT_PANEL_SPECS.items():
        sub = stats[stats["sort"] == sp.sort].set_index("factor")
        out[panel] = {f: [round(float(sub.loc[f, c]), 10) for c in COLUMNS]
                      for f in FACTORS}
    return out


def as_rows(ours: dict) -> list[dict]:
    return [{"panel": panel, "factor": f, "column": col,
             "value": round(ours[panel][f][i], 6)}
            for panel in ours for f in FACTORS for i, col in enumerate(COLUMNS)]


def render_latex(ours: dict, caption: str, sample: dict) -> str:
    """❗Took `T` and never referenced it. It now takes the sample block and the
    caption carries the sentence."""
    L = [r"\begin{table}[!ht]",
         r"\caption{" + caption + " " + D.sample_sentence(sample) + "}",
         r"\begin{center}", r"\label{" + LABEL + r"}", r"\scalebox{0.85}{%",
         r"\begin{tabular}{l rrrrrrrr}", r"\toprule",
         "Factor & " + " & ".join(HEADERS) + r" \\"]
    for panel, rows in ours.items():
        L += [r"\midrule",
              r"\multicolumn{9}{c}{\textbf{" + panel + r":} "
              + DEFAULT_SUBTITLES.get(panel, "") + r"} \\",
              r"\midrule"]
        for f in FACTORS:
            name = r"\texttt{" + f.replace("_", r"\_") + "}"
            L.append(name + " & " + " & ".join(_fmt(v) for v in rows[f]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", "}", r"\end{center}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--end", default=None, help="override the sample end")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()
    with Bench("table02", section="s1_lib", sample=not args.no_bench, echo=True) as b:
        with b.phase("stats"):
            stats = compute(end=args.end)
        with b.phase("render"):
            ours = as_cells(stats)
            rows = as_rows(ours)
            T = int(stats["T"].iloc[0])
            lags = int(stats["nw_lags"].iloc[0])
            out = paths.section_results("s1_lib")
            pd.DataFrame(rows).to_csv(out / "table02_cells.csv", index=False)
            stats.to_csv(out / "table02_stats.csv", index=False)
            tex = paths.TABLES / "table02.tex"
            sample = D.sample_block(
                first=stats["first"].iloc[0], last=stats["last"].iloc[0], T=T,
                basis="the LIB window; T is asserted, so every row shares it")
            tex.write_text(render_latex(ours, captions.caption(LABEL), sample),
                           encoding="utf-8")
            D.write_result(
                "table02",
                {"summary": {"exhibit": "Table 2", "tex_label": LABEL,
                             "sample": sample,
                             "n_cells": len(rows), "T": T, "nw_lags": lags},
                 "cells": rows},
                section="s1_lib",
                inputs=[E.sort_csv(a, sort=sp.sort) for sp in DEFAULT_PANEL_SPECS.values()
                        for a in ("unadjusted", "adj_return")],
                t0=t0, extra={"exhibit": "Table 2", "tex_label": LABEL})
        b.note(n_cells=len(rows), T=T, nw_lags=lags)
        n_finite = sum(1 for r in rows if pd.notna(r["value"]))
        ok = b.check(n_finite == len(rows),
                     f"{n_finite}/{len(rows)} cells populated, T={T}")

    print(f"\nTable 2 ({LABEL}): {len(rows)} cells, T={T}, NW lags={lags}")
    print(f"wrote {out / 'table02_cells.csv'}\n      {tex}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
