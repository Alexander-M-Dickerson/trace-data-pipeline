r"""t_ia10_11.py -- Tables IA.X and IA.XI: factors with significant alphas.

One row per factor whose CAPM_B alpha clears t > 1.96, ranked by alpha, in two panels
(single sort, within-firm) for each weighting:

    T, Date_Start, Date_End, mu, SD, t(mu), SR, alpha, t(alpha), IR

Means, standard deviations and alphas are annualized and in percent; SR and IR are
annualized ratios. A SHADED row additionally survives Benjamini-Hochberg at 5%.

❗The t > 1.96 cut and the Benjamini-Hochberg pass are different tests, and the shading
is the point of the table: many factors clear the conventional threshold, far fewer
survive the multiple-testing correction over all 108.

❗BH is computed over the panel's OWN 108 p-values. Running with a short frame would
loosen the threshold silently, which is why `zoo_frames` asserts the count.

    python s4_zoo/run_zoo_sorts.py
    python s4_zoo/t_ia10_11.py --which vw
    python s4_zoo/t_ia10_11.py --which ew
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

CONFIG = {"vw": dict(exhibit="Table IA.X", label="tab:vw_results",
                     stem="table_ia10", what="Value-weighted"),
          "ew": dict(exhibit="Table IA.XI", label="tab:ew_results",
                     stem="table_ia11", what="Equal-weighted")}
HEADERS = [r"$T$", "Start", "End", r"$\mu$", "SD", r"$t(\mu)$", "SR",
           r"$\alpha$", r"$t(\alpha)$", "IR"]
DEC = 2


def as_rows(frames: dict, weighting: str) -> dict:
    """{panel: {factor: {'cells': [...], 'shaded': bool}}}."""
    out = {}
    for panel, sort in ZF.PANELS.items():
        pv = Z.printed_view(frames[(weighting, sort)])
        rows = {}
        for _, r in pv.iterrows():
            rows[r["factor"]] = {
                "cells": [int(r["n"]),
                          r["date_start"].strftime("%Y-%m"),
                          r["date_end"].strftime("%Y-%m"),
                          round(float(r["mean"]) * 100, DEC),
                          round(float(r["std"]) * 100, DEC),
                          round(float(r["t_stat"]), DEC),
                          round(float(r["sr"]), DEC),
                          round(float(r["alpha"]) * 100, DEC),
                          round(float(r["alpha_t"]), DEC),
                          round(float(r["ir"]), DEC)],
                "shaded": bool(r["bh_pass"])}
        out[panel] = rows
    return out


def render_latex(ours: dict, label: str, what: str) -> str:
    L = [r"\begin{longtable}{l rrr rrr r rrr}",
         r"\caption{" + captions.caption(label) + r"}\label{" + label + r"}\\",
         r"\toprule",
         "Factor & " + " & ".join(HEADERS) + r" \\", r"\midrule", r"\endfirsthead",
         r"\toprule", "Factor & " + " & ".join(HEADERS) + r" \\", r"\midrule",
         r"\endhead"]
    for panel, rows in ours.items():
        sort_name = "Single-Sort" if ZF.PANELS[panel] == "single" else "Within-Firm"
        L += [r"\midrule",
              rf"\multicolumn{{11}}{{l}}{{\textbf{{{panel}:}} {what}, {sort_name} "
              rf"({len(rows)} of {Z.N_FACTORS} factors)}} \\", r"\midrule"]
        for factor, r in rows.items():
            name = r"\texttt{" + factor.replace("_", r"\_").replace("*", "$^{*}$") + "}"
            c = r["cells"]
            cells = [name, F.thousands(c[0]), c[1], c[2]] + [F.num(v, DEC) for v in c[3:]]
            line = F.row(cells)
            L.append(r"\rowcolor{gray!20} " + line if r["shaded"] else line)
    L += [r"\bottomrule", r"\end{longtable}"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--which", required=True, choices=("vw", "ew"))
    ap.add_argument("--end", default=Z.DATE_END, help="sample end for the statistics")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    cfg = CONFIG[args.which]

    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()
    with Bench(cfg["stem"], section="s4_zoo", sample=not args.no_bench,
               echo=True) as b:
        with b.phase("stats"):
            frames = ZF.load_frames(args.which, end=args.end)
            ours = as_rows(frames, args.which)
        with b.phase("render"):
            out = paths.section_results("s4_zoo")
            flat = [{"panel": p, "factor": f, "shaded": r["shaded"],
                     **{h: v for h, v in zip(
                         ["T", "start", "end", "mu", "sd", "t_mu", "sr",
                          "alpha", "t_alpha", "ir"], r["cells"])}}
                    for p, rows in ours.items() for f, r in rows.items()]
            pd.DataFrame(flat).to_csv(out / f"{cfg['stem']}_cells.csv", index=False)
            tex = paths.TABLES / f"{cfg['stem']}.tex"
            tex.write_text(render_latex(ours, cfg["label"], cfg["what"]),
                           encoding="utf-8")
            n_sig = {p: len(r) for p, r in ours.items()}
            n_bh = {p: sum(v["shaded"] for v in r.values()) for p, r in ours.items()}
            D.write_result(
                cfg["stem"],
                {"summary": {"exhibit": cfg["exhibit"], "tex_label": cfg["label"],
                             "weighting": args.which, "end": args.end,
                             "n_factors": Z.N_FACTORS,
                             "n_significant": n_sig, "n_fdr_survivors": n_bh},
                 "rows": flat},
                section="s4_zoo",
                inputs=[Z.zoo_csv(s) for s in ("single", "wf")] + [paths.BBW],
                t0=t0, extra={"exhibit": cfg["exhibit"], "tex_label": cfg["label"]})
        b.note(weighting=args.which, n_significant=n_sig, n_fdr_survivors=n_bh)
        # BH can only ever select a subset of what clears 1.96; if it does not, the
        # p-values and the t-statistics have come from different fits.
        ok = b.check(all(n_bh[p] <= n_sig[p] for p in ours),
                     f"significant {n_sig}, FDR survivors {n_bh} "
                     f"(of {Z.N_FACTORS} factors per panel)")

    print(f"\n{cfg['exhibit']} ({cfg['label']}), {args.which}: "
          f"significant {n_sig}, FDR survivors {n_bh}")
    print(f"wrote {tex}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
