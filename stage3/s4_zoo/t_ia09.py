r"""t_ia09.py -- Table IA.IX: false-discovery survivors by factor cluster.

Panel A counts, per cluster and per specification, how many factors clear t(alpha) >
1.96 AND survive Benjamini-Hochberg. Panel B names them.

❗TWO VARIANTS ARE WRITTEN, because the published table and the paper's own signal
dictionary disagree about one factor.

`table_ia09.tex` is AS PUBLISHED: it places `b_rvol` in the "Vol. & Liq. Betas"
cluster, which is where Table IA.IX prints it. `table_ia09_dictionary.tex` follows the
paper's own signal dictionary (and `zoo_engine.CLUSTERS`), which puts `b_rvol` in
"Macro & Other Betas". Nothing else moves between them. Reproduce, state, do not
silently correct.

    python s4_zoo/run_zoo_sorts.py
    python s4_zoo/t_ia09.py
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
import paths                # noqa: E402
import zoo_engine as Z      # noqa: E402
import zoo_frames as ZF     # noqa: E402
from bench import Bench     # noqa: E402

LABEL = "tab:fdr_by_cluster"
# printed cluster row labels -> zoo_engine.CLUSTERS keys, in printed order
ROW_TO_CLUSTER = [
    ("I", "Spreads, Yields, Size"), ("II", "Value"), ("III", "Momentum & Reversal"),
    ("IV", "Illiquidity"), ("V", "Volatility & Risk"), ("VI", "Market Risk"),
    ("VII", "Credit & Default Betas"), ("VIII", "Vol. & Liq. Betas"),
    ("IX", "Macro & Other Betas"),
]
SPEC_HEADERS = {("vw", "single"): "VW Single", ("vw", "wf"): "VW Within-Firm",
                ("ew", "single"): "EW Single", ("ew", "wf"): "EW Within-Firm"}
# where the printed table puts a factor that its own dictionary places elsewhere
PAPER_CLUSTER_OVERRIDES = {"b_rvol": "Vol. & Liq. Betas"}


def survivors(frames: dict) -> dict:
    """{(weighting, sort): [decorated surviving factor, ...]}"""
    return {key: list(df[df["sig_196"] & df["bh_pass"]]["factor"])
            for key, df in frames.items()}


def counts_and_names(surv: dict, *, as_printed: bool = True) -> tuple[dict, dict]:
    """as_printed=True places b_rvol where the published table does."""
    counts = {c: {k: 0 for k in ZF.SPEC_COLS} for _, c in ROW_TO_CLUSTER}
    names: dict[str, dict[str, str]] = {c: {} for _, c in ROW_TO_CLUSTER}
    for key, facs in surv.items():
        for f in facs:
            base = Z.base_of(f)
            cl = Z.CLUSTER_OF[base]
            if as_printed and base in PAPER_CLUSTER_OVERRIDES:
                cl = PAPER_CLUSTER_OVERRIDES[base]
            counts[cl][key] += 1
            star = "*" if f.endswith("*") else ""
            prev = names[cl].get(base)
            if prev is not None and prev != star:
                # the same factor sign-corrected one way in one specification and the
                # other way in another: worth seeing, not worth hiding
                star = prev + "!CONFLICT"
            names[cl][base] = star
    return counts, names


def render_latex(counts: dict, names: dict, *, as_printed: bool) -> str:
    note = ("b_rvol is placed as the published table places it."
            if as_printed else
            "b_rvol is placed as the paper's own signal dictionary places it.")
    L = [r"% " + note,
         r"\begin{table}[!ht]", r"\caption{" + captions.caption(LABEL) + "}",
         r"\begin{center}",
         r"\label{" + LABEL + ("}" if as_printed else "-dictionary}"),
         r"\scalebox{0.85}{%", r"\begin{tabular}{l l rrrr}", r"\toprule",
         " & Cluster & " + " & ".join(SPEC_HEADERS[k] for k in ZF.SPEC_COLS) + r" \\",
         r"\midrule"]
    totals = {k: 0 for k in ZF.SPEC_COLS}
    for numeral, cl in ROW_TO_CLUSTER:
        cells = [numeral, F.escape(cl)]
        for k in ZF.SPEC_COLS:
            cells.append(str(counts[cl][k]))
            totals[k] += counts[cl][k]
        L.append(F.row(cells))
    L += [r"\midrule",
          F.row(["", r"\textbf{Total}"] + [str(totals[k]) for k in ZF.SPEC_COLS]),
          r"\midrule",
          r"\multicolumn{6}{l}{\textbf{Panel B:} Surviving factors} \\", r"\midrule"]
    for numeral, cl in ROW_TO_CLUSTER:
        listed = ", ".join(
            r"\texttt{" + b.replace("_", r"\_") + "}" + ("$^{*}$" if s else "")
            for b, s in sorted(names[cl].items())) or "---"
        L.append(rf"{numeral} & \multicolumn{{5}}{{p{{0.72\textwidth}}}}{{{listed}}} \\")
    L += [r"\bottomrule", r"\end{tabular}", "}", r"\end{center}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--end", default=Z.DATE_END)
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()
    with Bench("table_ia09", section="s4_zoo", sample=not args.no_bench,
               echo=True) as b:
        with b.phase("stats"):
            frames = ZF.load_all(end=args.end)
            surv = survivors(frames)
            printed_c, printed_n = counts_and_names(surv, as_printed=True)
            dict_c, dict_n = counts_and_names(surv, as_printed=False)
        with b.phase("render"):
            out = paths.section_results("s4_zoo")
            pd.DataFrame([{"cluster": cl, "spec": f"{w}_{s}", "n": printed_c[cl][(w, s)],
                           "n_dictionary": dict_c[cl][(w, s)]}
                          for _, cl in ROW_TO_CLUSTER
                          for w, s in ZF.SPEC_COLS]).to_csv(
                out / "table_ia09_cells.csv", index=False)
            for c, n, stem, ap_ in ((printed_c, printed_n, "table_ia09", True),
                                    (dict_c, dict_n, "table_ia09_dictionary", False)):
                (paths.TABLES / f"{stem}.tex").write_text(
                    render_latex(c, n, as_printed=ap_), encoding="utf-8")
            n_total = {f"{w}_{s}": sum(printed_c[cl][(w, s)] for _, cl in ROW_TO_CLUSTER)
                       for w, s in ZF.SPEC_COLS}
            conflicts = sorted(b_ for cl in printed_n for b_, s_ in printed_n[cl].items()
                               if "CONFLICT" in s_)
            D.write_result(
                "table_ia09",
                {"summary": {"exhibit": "Table IA.IX", "tex_label": LABEL,
                             "end": args.end, "survivors_by_spec": n_total,
                             "sign_conflicts": conflicts},
                 "as_printed": {cl: {f"{w}_{s_}": printed_c[cl][(w, s_)]
                                     for w, s_ in ZF.SPEC_COLS}
                                for _, cl in ROW_TO_CLUSTER},
                 "by_dictionary": {cl: {f"{w}_{s_}": dict_c[cl][(w, s_)]
                                        for w, s_ in ZF.SPEC_COLS}
                                   for _, cl in ROW_TO_CLUSTER},
                 "names": {cl: printed_n[cl] for _, cl in ROW_TO_CLUSTER}},
                section="s4_zoo",
                inputs=[Z.zoo_csv(s_) for s_ in ("single", "wf")] + [paths.BBW],
                t0=t0, extra={"exhibit": "Table IA.IX", "tex_label": LABEL})
        b.note(survivors_by_spec=n_total, n_sign_conflicts=len(conflicts))
        # Every survivor must land in exactly one cluster; the two variants differ
        # only in WHERE b_rvol goes, never in HOW MANY there are.
        tot_p = sum(sum(v.values()) for v in printed_c.values())
        tot_d = sum(sum(v.values()) for v in dict_c.values())
        ok = b.check(tot_p == tot_d == sum(len(v) for v in surv.values()),
                     f"{tot_p} survivor placements, identical under both cluster "
                     f"variants")

    print(f"\nTable IA.IX ({LABEL}): survivors per specification {n_total}")
    if conflicts:
        print(f"  factors sign-corrected inconsistently across specs: {conflicts}")
    print(f"wrote {paths.TABLES / 'table_ia09.tex'}"
          f"\n      {paths.TABLES / 'table_ia09_dictionary.tex'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
