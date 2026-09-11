r"""f_nse_figures.py -- Figures IA.3 to IA.6: t-statistic boxes across the two grids.

Four figures, each showing the top four signals per cluster and, for each, the spread of
its t-statistic across every path of one uncertainty grid. A box straddling 1.96 is a
factor whose significance is a choice.

    IA.3  alpha t, DATA uncertainty     selected by mean(alpha)
    IA.4  premium t, DATA uncertainty   selected by mean(premia)
    IA.5  alpha t, METHOD uncertainty   selected by median(tstat_alpha), Dp-signed
    IA.6  premium t, METHOD uncertainty selected by median(t_stat),      Dp-signed

❗FOUR FIGURES, FOUR SELECTION RULES, and they are not interchangeable. The DUA pair
selects on the MEAN of the LEVEL frame; the MUA pair on the MEDIAN of the T frame. IA.5
and IA.6 therefore choose DIFFERENT 36-signal sets from each other. Reusing one
figure's selection for another silently changes which factors the reader sees.

❗The MUA half is sign-corrected on the FIGURE baseline (VW_Dp_Q_all_all_all), not the
tables' VW_Qp baseline.

Each figure carries an IDENTITY CHECK: every plotted box statistic is recomputed
independently here and compared at 1e-9. It catches a selection or sign-correction
applied on one path and not the other, which the picture cannot show you.

Also reported: how many of the 36 signals have a premium (and an alpha) whose sign
FLIPS somewhere across the grid -- the section's headline count, under its own
selection rule (top four by median alpha level, Dp-signed). It is reported, not gated:
the count is a finding about the data, not an invariant.

    python s3_nse/f_nse_figures.py
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

import clusters as C        # noqa: E402
import drrlib as D          # noqa: E402
import nse_engine as E      # noqa: E402
import paths                # noqa: E402
from bench import Bench     # noqa: E402

FIGS = [
    # (exhibit, tex_label, output stem, grid, plotted col, selection col, select by)
    ("Figure IA.3", "fig:dua_figure_4", "figIA3_nse_alpha_tstat_dua",
     "dua", "tstat_alpha", "alpha", "mean"),
    ("Figure IA.4", "fig:dua_figure_2", "figIA4_nse_tstat_dua",
     "dua", "tstat_premia", "premia", "mean"),
    ("Figure IA.5", "fig:mua_figure_3", "figIA5_nse_alpha_tstat_mua",
     "mua", "tstat_alpha", "tstat_alpha", "median"),
    ("Figure IA.6", "fig:mua_figure_5", "figIA6_nse_tstat_mua",
     "mua", "t_stat", "t_stat", "median"),
]
BOX_COLS = ["median", "q25", "q75", "min", "max"]
IDENT_TOL = 1e-9


def recompute_box(df, value: str, signals: list[str], signed_baseline: str | None) -> dict:
    """Independent recomputation of the box stats (the 1e-9 identity gate)."""
    sub = df
    if signed_baseline is not None:
        sub = C.apply_sign_correction(df, signed_baseline)
        sub = sub[sub["group"].notna()]
    out = {}
    for sig in signals:
        v = sub.loc[sub["signal"] == sig, value].dropna()
        out[sig] = {"median": float(v.median()), "q25": float(v.quantile(.25)),
                    "q75": float(v.quantile(.75)), "min": float(v.min()),
                    "max": float(v.max())}
    return out


def render(sel, baseline_vals, out_pdf: Path, xlabel: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = sel.sort_values(["group", "median"], ascending=[False, True]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 10))
    for i, row in df.iterrows():
        ax.add_patch(plt.Rectangle((row["q25"], i - 0.3), row["q75"] - row["q25"], 0.6,
                                   facecolor="#9ecae1", edgecolor="black", linewidth=0.5))
        ax.vlines(row["median"], i - 0.3, i + 0.3, color="black", linewidth=1.5)
        if baseline_vals and row["signal"] in baseline_vals:
            ax.vlines(baseline_vals[row["signal"]], i - 0.3, i + 0.3,
                      color="red", linewidth=1.5)
        ax.hlines(i, row["min"], row["q25"], color="black", linewidth=0.8)
        ax.hlines(i, row["q75"], row["max"], color="black", linewidth=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["signal"], fontsize=8, style="italic")
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.axvline(1.96, color="red", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.5, len(df) - 0.5)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", choices=("paper", "full"), default="paper")
    ap.add_argument("--twin", choices=("feb", "mar14"), default="feb")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()

    with Bench(f"nse-figures-{args.window}", section="s3_nse",
               sample=not args.no_bench, echo=True) as b:
        with b.phase("load"):
            dua = E.load_dua_paths(window=args.window)
            baselines = E.load_dua_baselines(window=args.window)
            mua = E.load_mua_paths(twin=args.twin, window=args.window)
        fig_dir = paths.FIGURES
        fig_dir.mkdir(parents=True, exist_ok=True)
        checks = []
        with b.phase("figures"):
            for exhibit, tex_label, stem, grid, plot_col, sel_col, by in FIGS:
                if grid == "dua":
                    sel_stats = E.dua_signal_stats(dua, sel_col)
                    plot_stats = E.dua_signal_stats(dua, plot_col)
                    signed_baseline = None
                    base_col = {"tstat_alpha": "baseline_alpha_tstat",
                                "tstat_premia": "baseline_tstat"}[plot_col]
                    baseline_vals = E.dua_baseline_values(baselines, base_col)
                else:
                    sel_stats = E.mua_signal_stats(mua, sel_col)
                    plot_stats = E.mua_signal_stats(mua, plot_col)
                    signed_baseline = E.FLIP_BASELINE
                    baseline_vals = None
                chosen = E.top4_per_cluster(sel_stats, by=by)
                sel = plot_stats[plot_stats["signal"].isin(chosen["signal"])].copy()
                assert len(sel) == 36, f"{exhibit}: selected {len(sel)} signals"

                indep = recompute_box(dua if grid == "dua" else mua, plot_col,
                                      list(sel["signal"]), signed_baseline)
                max_d = max(abs(float(r[c]) - indep[r["signal"]][c])
                            for _, r in sel.iterrows() for c in BOX_COLS)

                render(sel, baseline_vals, fig_dir / f"{stem}_{args.window}.pdf",
                       xlabel="$t$-statistic")
                checks.append({
                    "exhibit": exhibit, "tex_label": tex_label,
                    "grid": grid,
                    "selection_rule": f"top4 per cluster by {by}({sel_col})",
                    "n_selected": len(sel),
                    "signals": sorted(sel["signal"]),
                    "identity_max_abs_diff": max_d,
                    "identity_ok": bool(max_d < IDENT_TOL),
                    "box_stats": {r["signal"]: {c: float(r[c]) for c in BOX_COLS}
                                  for _, r in sel.iterrows()}})
        with b.phase("flips"):
            # the section's headline: among the top four per cluster by median alpha
            # LEVEL (Dp-signed), how many have a sign that flips somewhere on the grid
            signed = C.apply_sign_correction(mua, E.FLIP_BASELINE)
            valid = signed.dropna(subset=["mean_ret"])
            valid = valid[valid["group"].notna()]
            med = valid.groupby(["group", "signal"])["alpha"].median().reset_index()
            top4 = [s_ for g in sorted(med["group"].unique())
                    for s_ in med[med["group"] == g]
                    .sort_values("alpha", ascending=False)["signal"].head(4)]
            rng = valid[valid["signal"].isin(top4)].groupby("signal").agg(
                pmin=("mean_ret", "min"), pmax=("mean_ret", "max"),
                amin=("alpha", "min"), amax=("alpha", "max"))
            premia_flips = int(((rng.pmin < 0) & (rng.pmax > 0)).sum())
            alpha_flips = int(((rng.amin < 0) & (rng.amax > 0)).sum())
        with b.phase("record"):
            out = paths.section_results("s3_nse")
            pd.DataFrame([{"exhibit": c["exhibit"], "signal": sig, **stats}
                          for c in checks
                          for sig, stats in c["box_stats"].items()]).to_csv(
                out / f"figures_nse_boxes_{args.window}.csv", index=False)
            D.write_result(
                f"figures_nse_{args.window}",
                {"summary": {"exhibit": "Figures IA.3-IA.6",
                             "window": args.window, "twin": args.twin,
                             "n_figures": len(checks),
                             "identity_ok": all(c["identity_ok"] for c in checks),
                             "sign_flips_premia_of_36": premia_flips,
                             "sign_flips_alpha_of_36": alpha_flips},
                 "figures": checks},
                section="s3_nse",
                inputs=[E._dua_file(n, args.window)
                        for n in ("premia", "alpha", "baselines")]
                + [E.MUA_SUMMARY_DIR / f"mua_summary_{args.window}.parquet"],
                t0=t0, extra={"exhibit": "Figures IA.3-IA.6",
                              "window": args.window})
        b.note(window=args.window, twin=args.twin,
               sign_flips_premia=premia_flips, sign_flips_alpha=alpha_flips,
               identity_max=max(c["identity_max_abs_diff"] for c in checks))
        ok = b.check(all(c["identity_ok"] for c in checks),
                     f"{sum(c['identity_ok'] for c in checks)}/{len(checks)} figures "
                     f"identical to an independent recomputation at {IDENT_TOL:g}")

    print(f"\nFigures IA.3-IA.6, window={args.window}:")
    for c in checks:
        print(f"  {c['exhibit']}: {c['n_selected']} signals, "
              f"identity max|d|={c['identity_max_abs_diff']:.1e}")
    print(f"\nsign flips among the 36 selected: premium {premia_flips}, "
          f"alpha {alpha_flips}")
    print(f"wrote {fig_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
