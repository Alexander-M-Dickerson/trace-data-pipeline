r"""lab_exhibit.py -- the shared renderer for every Section-4 (look-ahead bias) table.

Table 4 and Tables IA.XV/IA.XVI differ only in which columns they print. All of them
report, per factor, a coefficient row above a t-statistic row, comparing:

    mu-tilde   the winsorized result -- returns clipped at a percentile of the FULL
               sample, a threshold nobody could have known at formation
    mu         the baseline, with no ex-post filtering
    Bias       the difference, tested on the PAIRED series

❗The alpha Bias is asymmetric on purpose, and it is easy to "fix" it into being wrong:
the POINT ESTIMATE is the difference of the two separately estimated alphas, while the
T-STATISTIC comes from regressing the paired bias series on MKTB. The point estimate is
a difference of levels; the test is on the difference series, which is far less noisy
because the two alphas share almost all of their variation.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # stage3/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import captions             # noqa: E402
import drrlib as D          # noqa: E402
import lab_engine as E      # noqa: E402
import paths                # noqa: E402
from bench import Bench     # noqa: E402

PANELS = {"Panel A": "left", "Panel B": "right"}     # the tail each panel reports
DEC = 2


def load_source() -> dict:
    """The LAB series `run_lab.py` produced."""
    root = paths.section_results("s2_lab") / "series"
    if not any(root.glob("*.parquet")):
        raise SystemExit(
            f"the LAB series are not under {root}.\n"
            "  Produce them with `python s2_lab/run_lab.py`.")
    return E.load_series(root)


def as_cells(stats: pd.DataFrame, columns: list[tuple], *,
             rating: str = "All", panels: dict[str, str] | None = None) -> dict:
    """{panel: {factor: {'coef': [...], 't': [...]}}}, in the printed column order."""
    out = {}
    for panel, tail in (panels or PANELS).items():
        sub = stats[(stats["tail"] == tail) & (stats["rating"] == rating)]
        val = sub.pivot_table(index="factor", columns=["leg", "variant", "stat"],
                              values="value")
        tst = sub.pivot_table(index="factor", columns=["leg", "variant", "stat"],
                              values="tstat")
        out[panel] = {f: {"coef": [round(float(val.loc[f, c]), 10) for c in columns],
                          "t": [round(float(tst.loc[f, c]), 10) for c in columns]}
                      for f in E.TAIL_SIGNALS[tail]}
    return out


def as_rows(ours: dict, columns: list[tuple]) -> list[dict]:
    names = [".".join(c) for c in columns]
    return [{"panel": panel, "factor": f, "column": cname, "kind": kind,
             "value": round(kinds[kind][i], 6)}
            for panel, facs in ours.items() for f, kinds in facs.items()
            for kind in ("coef", "t") for i, cname in enumerate(names)]


def _fmt(v: float) -> str:
    s = f"{v:.{DEC}f}"
    return s.replace("-", "$-$", 1) if s.startswith("-") else s


def render_latex(ours: dict, caption: str, columns: list[tuple], label: str,
                 head_groups: list[tuple[str, list[str]]],
                 sample: dict | None = None) -> str:
    ncol = len(columns) + 1
    L = [r"\begin{table}[!ht]",
         r"\caption{" + caption + " " + D.sample_sentence(sample) + "}",
         r"\begin{center}",
         r"\label{" + label + r"}", r"\scalebox{0.78}{%",
         r"\begin{tabular}{l " + " ".join("c" * len(g[1]) for g in head_groups) + "}",
         r"\toprule"]
    L.append(" & " + " & ".join(rf"\multicolumn{{{len(g[1])}}}{{c}}{{{g[0]}}}"
                                for g in head_groups) + r" \\")
    L.append(" & " + " & ".join(h for g in head_groups for h in g[1]) + r" \\")
    for panel, facs in ours.items():
        L += [r"\midrule", rf"\multicolumn{{{ncol}}}{{l}}{{\textbf{{{panel}}}}} \\",
              r"\midrule"]
        for f, kinds in facs.items():
            name = r"\texttt{" + f.replace("_", r"\_") + "}"
            L.append(name + " & " + " & ".join(_fmt(v) for v in kinds["coef"]) + r" \\")
            L.append(" & " + " & ".join("(" + _fmt(v) + ")" for v in kinds["t"]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", "}", r"\end{center}", r"\end{table}"]
    return "\n".join(L) + "\n"


def run_lab_exhibit(*, exhibit: str, label: str, columns: list[tuple], stem: str,
                    head_groups: list[tuple[str, list[str]]],
                    no_bench: bool = False) -> int:
    """One Section-4 table: statistics, cells, LaTeX."""
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()
    with Bench(stem, section="s2_lab", sample=not no_bench, echo=True) as b:
        with b.phase("load"):
            source = load_source()
            mktb = E.load_mktb_lab()
        with b.phase("stats"):
            specs = [E.LabSpec(tail=t) for t in PANELS.values()]
            stats = E.build(source, specs, mktb=mktb)
        with b.phase("render"):
            ours = as_cells(stats, columns)
            rows = as_rows(ours, columns)
            # ❗T is per series here, not one shared number: a LAB signal can be
            # missing months its neighbours have, and each t-statistic uses its own.
            T_range = (int(stats["T"].min()), int(stats["T"].max()))
            # ❗T is per series here, so the sentence says a RANGE. Section 4's
            # winsorization threshold is a full-sample quantile, so the window is a
            # producer argument and the series inside it are each their own length.
            sample = D.sample_block(
                first=str(stats["first"].min())[:10] if "first" in stats else None,
                last=str(stats["last"].max())[:10] if "last" in stats else None,
                T_min=T_range[0], T_max=T_range[1],
                basis="each series' own length inside the LAB window")
            out = paths.section_results("s2_lab")
            pd.DataFrame(rows).to_csv(out / f"{stem}_cells.csv", index=False)
            stats.to_csv(out / f"{stem}_stats.csv", index=False)
            tex = paths.TABLES / f"{stem}.tex"
            tex.write_text(render_latex(ours, captions.caption(label), columns,
                                        label, head_groups, sample=sample),
                           encoding="utf-8")
            D.write_result(
                stem,
                {"summary": {"exhibit": exhibit, "tex_label": label,
                             "n_cells": len(rows), "T_min": T_range[0],
                             "T_max": T_range[1], "sample": sample},
                 "cells": rows},
                section="s2_lab", inputs=[paths.BBW], t0=t0,
                extra={"exhibit": exhibit, "tex_label": label})
        b.note(n_cells=len(rows), T_min=T_range[0], T_max=T_range[1])
        n_finite = sum(1 for r in rows if pd.notna(r["value"]))
        ok = b.check(n_finite == len(rows),
                     f"{n_finite}/{len(rows)} cells populated, T in {T_range}")

    print(f"\n{exhibit} ({label}): {len(rows)} cells, T ranges {T_range}")
    print(f"wrote {out / f'{stem}_cells.csv'}\n      {tex}")
    return 0 if ok else 1
