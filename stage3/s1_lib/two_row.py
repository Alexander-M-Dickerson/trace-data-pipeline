r"""two_row.py -- the shared renderer for every two-row LIB exhibit.

Table 1, Table IA.XII, Table IA.XIII and Table IA.XIV differ only in their factor
list, their panel definitions and their caption. They all print, per factor, a row of
coefficients above a row of t-statistics in parentheses, across the same ten columns:

    Unadjusted (1)   mu  alpha
    Adj. Signal (2)  mu  alpha
    Adj. Return (3)  mu  alpha
    Bias (1)-(2)     dmu dalpha
    Bias (1)-(3)     dmu dalpha

So the shape lives here once, and each driver is a few constants plus a call.

Two conventions worth stating, because both are easy to get backwards:

  * Bias (1)-(2) varies the portfolio WEIGHTS while holding the return fixed;
    Bias (1)-(3) varies the RETURN WINDOW while holding the weights fixed. The
    second is the one the paper calls latent implementation bias.
  * Each bias is tested on the DIFFERENCE SERIES, not by differencing two separately
    estimated means. The point estimate is the same either way; the standard error is
    not, because the two series are highly correlated.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # stage3/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _stage3_settings as S    # noqa: E402
import drrlib as D              # noqa: E402
import lib_engine as E          # noqa: E402
import paths                    # noqa: E402
from bench import Bench         # noqa: E402

FACTORS = D.LIB_FACTORS                       # ytm cs bbtm dcs6 val_ipr val_hz str
DEFAULT_PANEL_SPECS = {"Panel A": E.LibSpec(sort="single"),
                       "Panel B": E.LibSpec(sort="wf")}
DEFAULT_SUBTITLES = {"Panel A": "Single-Sort", "Panel B": "Within-Firm Sort"}

# the column order, left to right, as the exhibits print it
COLUMNS = ["unadjusted.mu", "unadjusted.alpha",
           "adj_signal.mu", "adj_signal.alpha",
           "adj_return.mu", "adj_return.alpha",
           "bias_1_2.d_mu", "bias_1_2.d_alpha",
           "bias_1_3.d_mu", "bias_1_3.d_alpha"]
HEAD_GROUPS = [("Unadjusted (1)", r"$\mu$", r"$\alpha$"),
               (r"Adj.\ Signal (2)", r"$\mu$", r"$\alpha$"),
               (r"Adj.\ Return (3)", r"$\mu$", r"$\alpha$"),
               ("Bias (1)$-$(2)", r"$\Delta\mu$", r"$\Delta\alpha$"),
               ("Bias (1)$-$(3)", r"$\Delta\mu$", r"$\Delta\alpha$")]
DEC = 2                                       # two decimals, everywhere


# --------------------------------------------------------------------------
def compute(*, mktb: pd.Series, factors: tuple[str, ...] = FACTORS,
            panel_specs: dict | None = None, source_root: Path | None = None,
            expected_T: int | None = 268) -> pd.DataFrame:
    """The tidy statistics frame, one LibSpec per printed panel."""
    specs = list((panel_specs or DEFAULT_PANEL_SPECS).values())
    root = source_root or paths.SORTS
    missing = sorted({n for sp in specs for a in E.APPROACHES
                      if not (root / (n := E.sort_csv(a, sort=sp.sort,
                                                      rating=sp.rating).name)).exists()})
    if missing:
        raise SystemExit(
            f"this exhibit needs sort CSVs under {root}, and these are missing:\n"
            + "".join(f"    {m}\n" for m in missing)
            + "  Produce them with `python s1_lib/run_sorts.py`"
            + (" --rating IG (and NIG) for the rating splits" if any(
                sp.rating != "all" for sp in specs) else "") + ".")
    return E.build(specs, factors, root=root, mktb=mktb, expected_T=expected_T)


def as_cells(stats: pd.DataFrame, factors: tuple[str, ...] = FACTORS,
             panel_specs: dict | None = None) -> dict:
    """{panel: {factor: {'coef': [...], 't': [...]}}}, in the printed column order."""
    out = {}
    for panel, sp in (panel_specs or DEFAULT_PANEL_SPECS).items():
        sub = stats[(stats["sort"] == sp.sort) & (stats["rating"] == sp.rating)]
        val = sub.pivot_table(index="factor", columns="quantity", values="value")
        tst = sub.pivot_table(index="factor", columns="quantity", values="tstat")
        out[panel] = {f: {"coef": [round(float(val.loc[f, c]), 10) for c in COLUMNS],
                          "t": [round(float(tst.loc[f, c]), 10) for c in COLUMNS]}
                      for f in factors}
    return out


def as_rows(ours: dict, factors: tuple[str, ...] = FACTORS) -> list[dict]:
    """The long-format cell frame that ships beside the rendered table."""
    return [{"panel": panel, "factor": f, "column": col, "kind": kind,
             "value": round(ours[panel][f][kind][i], 6)}
            for panel in ours for f in factors for kind in ("coef", "t")
            for i, col in enumerate(COLUMNS)]


# --------------------------------------------------------------------------
def _fmt(v: float) -> str:
    """Two decimals, with LaTeX's math minus."""
    s = f"{v:.{DEC}f}"
    return s.replace("-", "$-$", 1) if s.startswith("-") else s


def render_latex(ours: dict, caption: str, T: int, *, label: str,
                 factors: tuple[str, ...] = FACTORS,
                 subtitles: dict[str, str] | None = None) -> str:
    subtitles = subtitles or DEFAULT_SUBTITLES
    L = [r"\begin{table}[!ht]", r"\caption{" + caption + "}",
         r"\begin{center}", r"\label{" + label + r"}", r"\scalebox{0.75}{%",
         r"\begin{tabular}{l rr rr rr rr rr}", r"\toprule"]
    L.append(" & " + " & ".join(r"\multicolumn{2}{c}{" + g[0] + "}"
                                for g in HEAD_GROUPS) + r" \\")
    L.append(" ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}"
                      for i in range(len(HEAD_GROUPS))))
    L.append("Factor & " + " & ".join(f"{g[1]} & {g[2]}" for g in HEAD_GROUPS) + r" \\")
    for panel, rows in ours.items():
        L += [r"\midrule",
              r"\multicolumn{11}{c}{\textbf{" + panel + r":} "
              + subtitles.get(panel, "") + r"} \\",
              r"\midrule"]
        for f in factors:
            name = r"\texttt{" + f.replace("_", r"\_") + "}"
            L.append(name + " & " + " & ".join(_fmt(v) for v in rows[f]["coef"]) + r" \\")
            L.append(" & " + " & ".join("(" + _fmt(v) + ")" for v in rows[f]["t"]) + r" \\")
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", "}", r"\end{center}", r"\end{table}"]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
def run_two_row_exhibit(*, exhibit: str, label: str, caption: str,
                        factors: tuple[str, ...], stem: str,
                        panel_specs: dict | None = None,
                        subtitles: dict[str, str] | None = None,
                        end: str | None = None, no_bench: bool = False) -> int:
    """One two-row LIB exhibit: statistics, cells, LaTeX.

    `end` overrides the sample end. The default is the paper's window, which fixes
    T = 268 and therefore the Newey-West lag count; passing a later end date derives
    T from the data instead, and the exhibit records what it used.
    """
    import dataclasses

    panel_specs = dict(panel_specs or DEFAULT_PANEL_SPECS)
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()
    expected_T: int | None = 268
    if end:
        panel_specs = {p: dataclasses.replace(sp, end=end)
                       for p, sp in panel_specs.items()}
        expected_T = None            # derived from the data, and recorded below

    with Bench(stem, section="s1_lib", sample=not no_bench, echo=True) as b:
        with b.phase("load"):
            any_spec = next(iter(panel_specs.values()))
            mktb = D.load_mktb(paths.BBW, start=any_spec.start, end=any_spec.end)
        with b.phase("stats"):
            stats = compute(mktb=mktb, factors=factors, panel_specs=panel_specs,
                            expected_T=expected_T)
        with b.phase("render"):
            ours = as_cells(stats, factors, panel_specs)
            T = int(stats["T"].iloc[0])
            lags = int(stats["nw_lags"].iloc[0])
            out = paths.section_results("s1_lib")
            pd.DataFrame(as_rows(ours, factors)).to_csv(
                out / f"{stem}_cells.csv", index=False)
            stats.to_csv(out / f"{stem}_stats.csv", index=False)
            tex = paths.TABLES / f"{stem}.tex"
            tex.write_text(render_latex(ours, caption, T, label=label,
                                        factors=factors, subtitles=subtitles),
                           encoding="utf-8")
            D.write_result(
                stem,
                {"summary": {"exhibit": exhibit, "tex_label": label,
                             "n_factors": len(factors),
                             "n_panels": len(panel_specs),
                             "n_cells": len(factors) * len(panel_specs)
                                        * len(COLUMNS) * 2,
                             "T": T, "nw_lags": lags,
                             "sample": [any_spec.start, any_spec.end]},
                 "cells": as_rows(ours, factors)},
                section="s1_lib",
                inputs=[paths.BBW] + [paths.SORTS / E.sort_csv(
                    a, sort=sp.sort, rating=sp.rating).name
                    for sp in panel_specs.values() for a in E.APPROACHES],
                t0=t0, extra={"exhibit": exhibit, "tex_label": label})
        b.note(n_factors=len(factors), n_panels=len(panel_specs), T=T, nw_lags=lags)
        # Every printed position must carry a number. A NaN here means a factor is
        # absent from one approach, which prints as a blank cell rather than failing.
        n_cells = len(factors) * len(panel_specs) * len(COLUMNS) * 2
        n_finite = sum(1 for panel in ours for f in factors for kind in ("coef", "t")
                       for v in ours[panel][f][kind] if pd.notna(v))
        ok = b.check(n_finite == n_cells, f"{n_finite}/{n_cells} cells populated, T={T}")

    print(f"\n{exhibit} ({label}): {n_cells} cells, T={T}, NW lags={lags}")
    print(f"wrote {out / f'{stem}_cells.csv'}\n      {tex}")
    return 0 if ok else 1
