r"""t_inline.py -- the two specification-count tables that appear inline in Section IA.3.

Neither carries a caption in the paper. Both count, per specification and in total, how
many of the 108 factors clear t > 1.96 and how many of those additionally survive
Benjamini-Hochberg -- one table for the CAPM_B alpha, one for the mean premium.

    alpha    t(alpha) > 1.96, BH over the panel's 108 alpha p-values
    premium  t(mu)    > 1.96, BH over the panel's 108 mean p-values

They are the compact statement of the paper's headline: the count that clears the
conventional threshold is several times the count that survives the correction.

    python s4_zoo/run_zoo_sorts.py
    python s4_zoo/t_inline.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # stage3/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import drrlib as D          # noqa: E402
import latex_format as F    # noqa: E402
import paths                # noqa: E402
import zoo_engine as Z      # noqa: E402
import zoo_frames as ZF     # noqa: E402
from bench import Bench     # noqa: E402

ROWS = [("VW, Single-Sort", ("vw", "single")), ("VW, Within-Firm", ("vw", "wf")),
        ("EW, Single-Sort", ("ew", "single")), ("EW, Within-Firm", ("ew", "wf"))]
STATS = {"alpha": (r"CAPM$_{B}$ alpha", "sig_196", "bh_pass"),
         "premium": ("Mean premium", "sig_196_mu", "bh_pass_mu")}
TOTAL_N = 4 * Z.N_FACTORS       # 432 across the four specifications


def counts(frames: dict, stat: str) -> dict:
    """{row label: [count, pct, fdr, fdr_pct]}, plus a Total row."""
    _, sig_col, bh_col = STATS[stat]
    rows, tot_n, tot_f = {}, 0, 0
    for label, key in ROWS:
        df = frames[key]
        n = int(df[sig_col].sum())
        f = int((df[sig_col] & df[bh_col]).sum())
        rows[label] = [n, round(100 * n / Z.N_FACTORS, 1),
                       f, round(100 * f / Z.N_FACTORS, 1)]
        tot_n += n
        tot_f += f
    rows["Total"] = [tot_n, round(100 * tot_n / TOTAL_N, 1),
                     tot_f, round(100 * tot_f / TOTAL_N, 1)]
    return rows


def render_latex(rows: dict, stat: str) -> str:
    title, _, _ = STATS[stat]
    L = [rf"% inline specification counts: {title}",
         r"\begin{center}", r"\begin{tabular}{l rr rr}", r"\toprule",
         r"Specification & \multicolumn{2}{c}{$t > 1.96$} & "
         r"\multicolumn{2}{c}{FDR survivors} \\",
         r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
         r" & $N$ & \% & $N$ & \% \\", r"\midrule"]
    for label, _ in ROWS:
        c = rows[label]
        L.append(F.row([label, str(c[0]), F.pct(c[1]), str(c[2]), F.pct(c[3])]))
    c = rows["Total"]
    L += [r"\midrule",
          F.row([r"\textbf{Total}", str(c[0]), F.pct(c[1]), str(c[2]), F.pct(c[3])]),
          r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--end", default=Z.DATE_END)
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()
    with Bench("inline-counts", section="s4_zoo", sample=not args.no_bench,
               echo=True) as b:
        with b.phase("stats"):
            frames = ZF.load_all(end=args.end)
            out_counts = {stat: counts(frames, stat) for stat in STATS}
        with b.phase("render"):
            out = paths.section_results("s4_zoo")
            flat = [{"stat": stat, "row": label, "count": c[0], "pct": c[1],
                     "fdr": c[2], "fdr_pct": c[3]}
                    for stat, rows in out_counts.items()
                    for label, c in rows.items()]
            pd.DataFrame(flat).to_csv(out / "inline_counts.csv", index=False)
            for stat in STATS:
                (paths.TABLES / f"inline_counts_{stat}.tex").write_text(
                    render_latex(out_counts[stat], stat), encoding="utf-8")
            D.write_result(
                "inline_counts",
                {"summary": {"exhibit": "Inline specification counts",
                             "tex_label": None, "end": args.end,
                             "n_factors_per_spec": Z.N_FACTORS,
                             "totals": {s: out_counts[s]["Total"] for s in STATS}},
                 "rows": flat},
                section="s4_zoo",
                inputs=[Z.zoo_csv(s) for s in ("single", "wf")] + [paths.BBW],
                t0=t0, extra={"exhibit": "Inline specification counts"})
        b.note(totals={s: out_counts[s]["Total"][0] for s in STATS},
               fdr={s: out_counts[s]["Total"][2] for s in STATS})
        # FDR selects a subset of what clears 1.96, per row and in total. If it does
        # not, the p-values and the t-statistics came from different fits.
        bad = [(stat, label) for stat, rows in out_counts.items()
               for label, c in rows.items() if c[2] > c[0]]
        ok = b.check(not bad,
                     "FDR survivors are a subset of the significant set in every row"
                     + (f"; violations: {bad}" if bad else ""))

    for stat, rows in out_counts.items():
        c = rows["Total"]
        print(f"\n{STATS[stat][0]}: {c[0]}/{TOTAL_N} clear t>1.96, "
              f"{c[2]} survive FDR")
    print(f"\nwrote {paths.TABLES / 'inline_counts_alpha.tex'}"
          f"\n      {paths.TABLES / 'inline_counts_premium.tex'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
