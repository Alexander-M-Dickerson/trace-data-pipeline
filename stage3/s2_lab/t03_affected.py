r"""t03_affected.py -- Table 3, "Corporate bond factors sensitive to ex-post filtering".

A classification table rather than an estimate: which factors improve when the return
distribution is clipped, and in which tail. It is generated from the same signal lists
that drive Table 4 and the decomposition tables, so the two cannot describe different
sets of factors.

  left-tail factors   winsorized at the 0.50th percentile -- their result improves
                      when the worst returns are clipped
  right-tail factors  winsorized at the 99.50th -- momentum, where the best returns
                      are the ones a filter removes

❗One extra row. `b_dunc6` is listed here but is NOT quantified in Table 4, which
reports 15 factors to this table's 16. That is the published paper's own
inconsistency, reproduced rather than silently corrected: this file emits the 16 rows
and marks which one the later tables do not carry. Set --quantified-only to print the
15 the rest of Section 4 actually estimates.

    python s2_lab/t03_affected.py
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
import lab_engine as E      # noqa: E402
import paths                # noqa: E402
from bench import Bench     # noqa: E402

LABEL = "tab:lab_affected_factors"
THRESHOLD = {"left": "0.50", "right": "99.50"}
# listed in this table, absent from Table 4's 15 -- the paper's own 16-vs-15 gap.
# It is a left-tail uncertainty beta, so its threshold is still determinate.
LISTED_ONLY_HERE = {"b_dunc6": "left"}

DESCRIPTION = {
    "left": "Result improves when the left tail of returns is clipped",
    "right": "Result improves when the right tail of returns is clipped",
}


def rows(*, quantified_only: bool = False) -> pd.DataFrame:
    out = []
    for tail, sigs in (("left", E.LEFT_SIGNALS), ("right", E.RIGHT_SIGNALS)):
        for s in sigs:
            out.append({"mnemonic": s, "tail": tail, "threshold": THRESHOLD[tail],
                        "quantified_in_table4": True})
    if not quantified_only:
        for s, tail in LISTED_ONLY_HERE.items():
            out.append({"mnemonic": s, "tail": tail, "threshold": THRESHOLD[tail],
                        "quantified_in_table4": False})
    return pd.DataFrame(out)


def render_latex(df: pd.DataFrame) -> str:
    unquantified = df[~df["quantified_in_table4"]]["mnemonic"].tolist()
    L = []
    if unquantified:
        L += [rf"% {', '.join(unquantified)} appears in this table but is not",
              r"% quantified in Table 4, which reports one fewer factor. The paper's",
              r"% own inconsistency, reproduced here rather than corrected."]
    L += [r"\begin{table}[!ht]", r"\caption{" + captions.caption(LABEL) + "}",
          r"\begin{center}", r"\label{" + LABEL + r"}",
          r"\begin{tabular}{l l r}", r"\toprule",
          r"Factor & Sensitivity & Winsorization pct. \\", r"\midrule"]
    for tail in ("left", "right"):
        sub = df[df["tail"] == tail]
        if sub.empty:
            continue
        L.append(r"\multicolumn{3}{l}{\textit{" + DESCRIPTION[tail] + r"}} \\")
        for _, r in sub.iterrows():
            name = r"\texttt{" + r["mnemonic"].replace("_", r"\_") + "}"
            mark = "" if r["quantified_in_table4"] else r"$^{\dagger}$"
            L.append(f"{name}{mark} & {tail.capitalize()} tail & "
                     f"{r['threshold']}\\% \\\\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    if unquantified:
        L.append(r"\\[0.5em]\footnotesize$^{\dagger}$ Listed here but not quantified "
                 r"in Table~\ref{tab:lab_ls_1}.")
    L += [r"\end{center}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quantified-only", action="store_true",
                    help="print only the factors Table 4 also estimates")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()
    with Bench("table03", section="s2_lab") as b:
        with b.phase("render"):
            df = rows(quantified_only=args.quantified_only)
            out = paths.section_results("s2_lab")
            df.to_csv(out / "table03_cells.csv", index=False)
            tex = paths.TABLES / "table03.tex"
            tex.write_text(render_latex(df), encoding="utf-8")
            D.write_result(
                "table03",
                {"summary": {"exhibit": "Table 3", "tex_label": LABEL,
                             "n_rows": len(df),
                             "n_quantified": int(df["quantified_in_table4"].sum())},
                 "rows": df.to_dict("records")},
                section="s2_lab", inputs=[], t0=t0,
                extra={"exhibit": "Table 3", "tex_label": LABEL})
        b.note(n_rows=len(df))
        # The classification here and the factors Section 4 estimates must be the same
        # set, apart from the documented extra row. A silent divergence would make two
        # tables in the same section describe different universes.
        listed = set(df[df["quantified_in_table4"]]["mnemonic"])
        quantified = set(E.LEFT_SIGNALS) | set(E.RIGHT_SIGNALS)
        ok = b.check(listed == quantified,
                     f"{len(listed)} classified factors == the {len(quantified)} "
                     "Section 4 estimates"
                     + (f"; plus {sorted(set(df['mnemonic']) - quantified)} listed only here"
                        if not args.quantified_only else ""))

    print(f"\nTable 3 ({LABEL}): {len(df)} rows "
          f"({int(df['quantified_in_table4'].sum())} also in Table 4)")
    print(f"wrote {tex}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
