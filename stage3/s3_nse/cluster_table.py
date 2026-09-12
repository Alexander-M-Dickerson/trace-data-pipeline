r"""cluster_table.py -- the shared renderer for the two non-standard-error tables.

Table 5 (data uncertainty) and Table 6 (method uncertainty) print the same 10 x 9
shape: nine factor clusters plus All, against premium mean / median / NSE / Ratio, the
same four for alpha, and the path count.

❗The two tables compute Ratio by OPPOSITE rules, and this is not an oversight to tidy
up. Table 5 takes std(value) and mean(SE) on the PAIRWISE-dropna matched sample; Table 6
takes them with INDEPENDENT skipna. The engine applies each rule where it belongs
(`nse_engine.dua_cluster_summary` and `mua_cluster_summary`); this module only prints.

NSE is the interquartile range of the estimate across paths -- how much the answer moves
when a defensible choice is made differently. Ratio compares that spread against the
conventional standard error: above 1 means the choices matter more than the sampling
noise the paper reports.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # stage3/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import captions             # noqa: E402
import drrlib as D          # noqa: E402
import latex_format as F    # noqa: E402

COLS = ["mu_mean", "mu_median", "nse_mu", "ratio_mu",
        "alpha_mean", "alpha_median", "nse_alpha", "ratio_alpha", "n_paths"]
HEAD = [(r"Premium (\%)", [r"Mean", r"Median", r"NSE", r"Ratio"]),
        (r"Alpha (\%)", [r"Mean", r"Median", r"NSE", r"Ratio"]),
        ("", [r"$N$"])]


def as_rows(df: pd.DataFrame) -> list[dict]:
    return [{"cluster": r.cluster_name, "column": c, "value": float(r[c])}
            for _, r in df.iterrows() for c in COLS]


def render_latex(df: pd.DataFrame, label: str, *, note: str = "",
                 sample: dict | None = None) -> str:
    L = [r"\begin{table}[!ht]",
         r"\caption{" + captions.caption(label) + " "
         + D.sample_sentence(sample) + "}",
         r"\begin{center}", r"\label{" + label + r"}", r"\scalebox{0.85}{%",
         r"\begin{tabular}{l rrrr rrrr r}", r"\toprule"]
    L.append(" & " + " & ".join(
        (rf"\multicolumn{{{len(g[1])}}}{{c}}{{{g[0]}}}" if g[0] else g[1][0])
        for g in HEAD) + r" \\")
    L.append(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}")
    L.append("Cluster & " + " & ".join(
        h for g in HEAD[:2] for h in g[1]) + " & " + r" \\")
    L.append(r"\midrule")
    for _, r in df.iterrows():
        if r.cluster_name == "All":
            L.append(r"\midrule")
        cells = [F.escape(r.cluster_name)]
        cells += [F.num(r[c], 2) for c in COLS[:-1]]
        cells.append(F.thousands(r["n_paths"]))
        L.append(F.row(cells))
    L += [r"\bottomrule", r"\end{tabular}", "}"]
    if note:
        L.append(r"\\[0.5em]\footnotesize " + note)
    L += [r"\end{center}", r"\end{table}"]
    return "\n".join(L) + "\n"
