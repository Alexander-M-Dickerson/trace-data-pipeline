r"""tB1_lib_summary.py -- Table B.1, "Latent implementation bias across 108 factors".

The census: across every factor, how many show a statistically significant gap between
the month-end and month-begin return, split by whether the signal is price-based.
Four rows -- single/within-firm x value/equal-weighted -- and two column groups.

❗TWO VARIANTS ARE WRITTEN, and the difference between them is a real finding.

`tableB1.tex` is the AS-PUBLISHED computation: it differences the two stored series
without undoing PyBondLab's extract-time sign corrections. Those corrections are
decided per file, on each extraction's own full-sample mean, so a factor whose
month-end mean is positive and whose month-begin mean is negative is stored starred in
one file and unstarred in the other. Differencing those two gives -(r_End + r_Bgn),
which is not a bias estimate at all: its volatility runs several times the true gap's
and its t-statistic collapses toward zero. On this data the mismatch count is reported
each run.

`tableB1_corrected.tex` undoes every flip first, so Delta = End_true - Bgn_true
throughout. It is the number the decomposition actually intends.

The mechanism, checkable on the output: a factor is starred in a file exactly when that
file's untruncated mean is negative, and a pair is mismatched exactly when the two
files' true means straddle zero.

    python s1_lib/run_lib_sorts.py --sort single --timing end     # x4, see that file
    python s1_lib/tB1_lib_summary.py
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
import paths                # noqa: E402
import run_lib_sorts as RL  # noqa: E402
from bench import Bench     # noqa: E402

LABEL = "tab:lib_summary"
DATE_CUTOFF_END = "2024-12-31"
PRICE_BASED = list(D.PRICE_BASED)
ROWS = [("single", "vw"), ("single", "ew"), ("within", "vw"), ("within", "ew")]
ROW_LABELS = {("single", "vw"): "Single-Sorted (VW)", ("single", "ew"): "Single-Sorted (EW)",
              ("within", "vw"): "Within-Firm (VW)", ("within", "ew"): "Within-Firm (EW)"}
COLS = ["pb_sig", "pb_tot", "pb_pct", "pb_avg_delta", "pb_avg_t",
        "npb_sig", "npb_tot", "npb_pct", "npb_avg_delta", "npb_avg_t"]


def _lib_csv(sort: str, timing: str, *, root: Path | None = None) -> Path:
    name = RL.lib_csv_name("single" if sort == "single" else "wf", timing)
    return (root or RL.lib_root()) / name


def compute_lib(*, undo_flips: bool = False, root: Path | None = None,
                cutoff_end: str = DATE_CUTOFF_END) -> tuple[dict[str, pd.DataFrame], dict]:
    """Per-factor End-minus-Bgn gaps. `undo_flips=True` gives the corrected variant.

    Also returns the flip-mismatch census, which does not depend on `undo_flips`.
    """
    results, flips = {}, {"mismatched_pairs": [], "n_pairs": 0}
    for sort in ("single", "within"):
        for timing, key in (("end", "e"), ("bgn", "b")):
            f = _lib_csv(sort, timing, root=root)
            if not f.exists():
                raise SystemExit(
                    f"Table B.1 needs {f.name} and it is not under {f.parent}.\n"
                    "  Produce the four files with `python s1_lib/run_lib_sorts.py "
                    "--sort {single,wf} --timing {end,bgn}`.")
        df_end = pd.read_csv(_lib_csv(sort, "end", root=root), parse_dates=["date"])
        df_bgn = pd.read_csv(_lib_csv(sort, "bgn", root=root), parse_dates=["date"])
        df_end = df_end[df_end["date"] <= pd.Timestamp(cutoff_end)]
        df_bgn = df_bgn[df_bgn["date"] <= pd.Timestamp(cutoff_end)]
        for weighting in ("vw", "ew"):
            ls_e = df_end[(df_end["leg"] == "ls") & (df_end["weighting"] == weighting)]
            ls_b = df_bgn[(df_bgn["leg"] == "ls") & (df_bgn["weighting"] == weighting)]
            we = ls_e.pivot_table(index="date", columns="factor", values="return")
            wb = ls_b.pivot_table(index="date", columns="factor", values="return")
            emap = {c.rstrip("*"): c for c in we.columns}
            bmap = {c.rstrip("*"): c for c in wb.columns}
            stats = []
            for base in sorted(set(emap) & set(bmap)):
                ec, bc = emap[base], bmap[base]
                se_, sb_ = we[ec].dropna(), wb[bc].dropna()
                common = se_.index.intersection(sb_.index)
                if len(common) < 12:
                    continue
                e, b = se_.loc[common], sb_.loc[common]
                if undo_flips:
                    e = -e if ec.endswith("*") else e
                    b = -b if bc.endswith("*") else b
                diff = e - b
                _, t_diff = D.nw_mean(diff, D.nw_lags(len(diff)))
                base_clean = base.replace("_mmn", "")
                base_clean = base_clean[:-3] if base_clean.endswith("_wf") else base_clean
                flips["n_pairs"] += 1
                if ec.endswith("*") != bc.endswith("*"):
                    flips["mismatched_pairs"].append(
                        {"sort": sort, "weighting": weighting, "end": ec, "bgn": bc})
                stats.append({"factor": ec.replace("_mmn", ""),
                              "price_based": base_clean in PRICE_BASED,
                              "T": len(common),
                              "delta": float(diff.mean()) * 12 * 100,   # annualized %
                              "delta_t": float(t_diff)})
            results[f"{weighting}_{sort}"] = pd.DataFrame(stats)
    return results, flips


def summarize(lib: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sort, wt in ROWS:
        df = lib[f"{wt}_{sort}"]
        pb, npb = df[df["price_based"]], df[~df["price_based"]]
        rows.append({
            "sort": sort, "weighting": wt, "label": ROW_LABELS[(sort, wt)],
            "pb_sig": int((pb["delta_t"].abs() > 1.96).sum()), "pb_tot": len(pb),
            "pb_pct": 100 * (pb["delta_t"].abs() > 1.96).mean(),
            "pb_avg_delta": float(pb["delta"].abs().mean() / 12),   # back to monthly %
            "pb_avg_t": float(pb["delta_t"].abs().mean()),
            "npb_sig": int((npb["delta_t"].abs() > 1.96).sum()), "npb_tot": len(npb),
            "npb_pct": 100 * (npb["delta_t"].abs() > 1.96).mean(),
            "npb_avg_delta": float(npb["delta"].abs().mean() / 12),
            "npb_avg_t": float(npb["delta_t"].abs().mean()),
        })
    return pd.DataFrame(rows)


def render_tex(summ: pd.DataFrame, *, corrected: bool, n_mismatch: int,
               n_pairs: int) -> str:
    note = ([rf"% CORRECTED: extract-time sign flips undone before differencing, so",
             rf"% Delta = End_true - Bgn_true. {n_mismatch} of {n_pairs} end/bgn pairs",
             rf"% were flip-mismatched; in the as-published variant their Delta is",
             rf"% -(r_End + r_Bgn) rather than a bias estimate."]
            if corrected else
            [rf"% AS PUBLISHED: stored orientations differenced without undoing the",
             rf"% extract-time sign flips. {n_mismatch} of {n_pairs} end/bgn pairs are",
             rf"% flip-mismatched -- see tableB1_corrected.tex."])
    L = note + [
        r"\begin{table}[!ht]",
        r"\caption{" + captions.caption(LABEL) + "}",
        r"\label{" + LABEL + ("-corrected}" if corrected else "}"),
        r"\begin{center}",
        r"\scalebox{0.85}{%",
        r"\begin{tabular}{l cccc cccc}",
        r"\toprule",
        r" & \multicolumn{4}{c}{\textbf{Price-Based}} & "
        r"\multicolumn{4}{c}{\textbf{Non-Price-Based}} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r"\textbf{Sort} & \textbf{Sig./Total} & \textbf{\%} & "
        r"\textbf{$\overline{|\Delta|}$} & \textbf{$\overline{|t|}$} & "
        r"\textbf{Sig./Total} & \textbf{\%} & \textbf{$\overline{|\Delta|}$} & "
        r"\textbf{$\overline{|t|}$} \\",
        r"\midrule"]
    for _, r in summ.iterrows():
        L.append(f"{r['label']} & {r['pb_sig']}/{r['pb_tot']} & {r['pb_pct']:.0f}\\% "
                 f"& {r['pb_avg_delta']:.2f} & {r['pb_avg_t']:.2f} "
                 f"& {r['npb_sig']}/{r['npb_tot']} & {r['npb_pct']:.0f}\\% "
                 f"& {r['npb_avg_delta']:.2f} & {r['npb_avg_t']:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", "}", r"\end{center}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--end", default=DATE_CUTOFF_END,
                    help="sample end for the census (default: the paper's)")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()
    out = paths.section_results("s1_lib")
    with Bench("tableB1", section="s1_lib", sample=not args.no_bench, echo=True) as b:
        with b.phase("stats"):
            lib, flips = compute_lib(cutoff_end=args.end)
            lib_c, _ = compute_lib(undo_flips=True, cutoff_end=args.end)
            summ, summ_c = summarize(lib), summarize(lib_c)
        with b.phase("render"):
            n_mm, n_pairs = len(flips["mismatched_pairs"]), flips["n_pairs"]
            summ.to_csv(out / "tableB1_cells.csv", index=False)
            summ_c.to_csv(out / "tableB1_corrected_cells.csv", index=False)
            pd.DataFrame(flips["mismatched_pairs"]).to_csv(
                out / "tableB1_flip_mismatches.csv", index=False)
            for frame, corrected, stem in ((summ, False, "tableB1"),
                                           (summ_c, True, "tableB1_corrected")):
                (paths.TABLES / f"{stem}.tex").write_text(
                    render_tex(frame, corrected=corrected, n_mismatch=n_mm,
                               n_pairs=n_pairs), encoding="utf-8")
            D.write_result(
                "tableB1",
                {"summary": {"exhibit": "Table B.1", "tex_label": LABEL,
                             "n_flip_mismatched_pairs": n_mm, "n_pairs": n_pairs,
                             "cutoff_end": args.end},
                 "as_published": summ.to_dict("records"),
                 "corrected": summ_c.to_dict("records"),
                 "flip_mismatches": flips["mismatched_pairs"]},
                section="s1_lib",
                inputs=[_lib_csv(s_, t_) for s_ in ("single", "within")
                        for t_ in ("end", "bgn")],
                t0=t0, extra={"exhibit": "Table B.1", "tex_label": LABEL})
        b.note(n_rows=len(summ), n_flip_mismatched=n_mm, n_pairs=n_pairs)
        # Every row must cover the whole factor set; a short census is the failure
        # that would otherwise read as a finding.
        totals = summ["pb_tot"] + summ["npb_tot"]
        ok = b.check(bool((totals >= 100).all()),
                     f"factors per row: {sorted(totals.tolist())}")

    print(f"\nTable B.1: {n_mm} of {n_pairs} end/bgn pairs are flip-mismatched")
    print(summ[["label", "pb_sig", "pb_tot", "npb_sig", "npb_tot"]].to_string(index=False))
    print(f"\nwrote {paths.TABLES / 'tableB1.tex'}"
          f"\n      {paths.TABLES / 'tableB1_corrected.tex'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
