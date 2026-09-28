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

Drawn in each look `--style` asks for (figstyle.py): each box in its cluster's colour, with
a legend of the clusters and the baseline mark under the plot, top to bottom in selection
order (cluster I first).

Each figure carries an IDENTITY CHECK: every plotted box statistic is recomputed
independently here and compared at 1e-9. It catches a selection or sign-correction
applied on one path and not the other, which the picture cannot show you.

Also reported: how many of the 36 signals have a premium (and an alpha) whose sign
FLIPS somewhere across the grid -- the section's headline count, under its own
selection rule (top four by median alpha level, Dp-signed). It is reported, not gated:
the count is a finding about the data, not an invariant.

    python s3_nse/f_nse_figures.py [--window paper|full] [--style paper|house|both]
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


def render(sel, order: list[str], baseline_vals, out_pdf: Path, xlabel: str, st) -> None:
    """One box per signal, top to bottom in selection order (cluster I first): the IQR box
    filled in its cluster's colour, the median, the min-max whiskers, and the baseline t
    (averaged across ratings and weightings) as a mark across the box."""
    import figstyle
    from matplotlib.lines import Line2D

    paper = st.name == "paper"
    df = sel.set_index("signal").loc[order].reset_index().iloc[::-1].reset_index(drop=True)
    ink = st.c("ink")
    with figstyle.use(st) as plt:
        fig, ax = plt.subplots(figsize=((10, 14) if paper else (6.5, 7.2)))
        h = 0.6
        drawn_baseline = False
        for i, row in df.iterrows():
            colour = st.clusters[int(row["group"]) - 1]
            ax.add_patch(plt.Rectangle((row["q25"], i - h / 2), row["q75"] - row["q25"], h,
                                       facecolor=colour, edgecolor=ink, linewidth=0.5))
            ax.vlines(row["median"], i - h / 2, i + h / 2, color=ink, linewidth=(1.5 if paper else 1.1))
            if baseline_vals and row["signal"] in baseline_vals:
                ax.vlines(baseline_vals[row["signal"]], i - h / 2, i + h / 2,
                          color=st.c("baseline"), linewidth=(1.5 if paper else 1.3))
                drawn_baseline = True
            lw = 0.8 if paper else 0.7
            ax.hlines(i, row["min"], row["q25"], color=ink, linewidth=lw)
            ax.hlines(i, row["q75"], row["max"], color=ink, linewidth=lw)
            for end in (row["min"], row["max"]):
                ax.vlines(end, i - h / 4, i + h / 4, color=ink, linewidth=lw)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(list(df["signal"]), fontsize=(9 if paper else 6.5),
                           style=("italic" if paper else "normal"))
        ax.axvline(0, color=st.c("zero"), linewidth=0.5)
        ax.axvline(1.96, color=st.c("threshold"), linewidth=(1.0 if paper else 0.8),
                   linestyle="--", alpha=(0.7 if paper else 1.0))
        ax.set_xlabel(xlabel)
        lo = min(float(df["min"].min()), 0.0) - 0.2
        hi = max(float(df["max"].max()), 1.96) + 0.2
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.5, len(df) - 0.5)
        st.grid(ax, axis=("both" if paper else "x"))
        if not paper:
            ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(True)
        # the clusters' colours, and the baseline mark, in a legend under the plot
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=st.clusters[g - 1], edgecolor=ink,
                                 linewidth=0.5) for g in sorted(int(g) for g in df["group"].unique())]
        labels = [C.get_group_name(g) for g in sorted(int(g) for g in df["group"].unique())]
        if drawn_baseline:
            handles.append(Line2D([0], [0], color=st.c("baseline"), linewidth=1.5))
            labels.append("Baseline")
        ax.legend(handles=handles, labels=labels, loc="upper center",
                  bbox_to_anchor=(0.5, (-0.04 if paper else -0.07)),
                  ncol=(4 if drawn_baseline else 3) if paper else 4, fontsize=(8 if paper else 6.5),
                  columnspacing=1.0, handletextpad=0.5, **st.legend)
        fig.tight_layout()
        fig.subplots_adjust(bottom=(0.12 if paper else 0.16))
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        figstyle.save(fig, out_pdf, dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", choices=("paper", "full"), default="paper")
    ap.add_argument("--twin", choices=("feb", "mar14"), default="feb")
    ap.add_argument("--no-bench", action="store_true")
    import figstyle
    figstyle.add_argument(ap)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()
    looks = figstyle.styles(args.style)

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
                else:
                    sel_stats = E.mua_signal_stats(mua, sel_col)
                    plot_stats = E.mua_signal_stats(mua, plot_col)
                    signed_baseline = E.FLIP_BASELINE
                # the mark across each box: the signal's unfiltered baseline t, averaged across
                # ratings and weightings, in all four figures (the published captions define it
                # so for Figures IA.3-IA.5, and IA.6 drew it too)
                base_col = {"tstat_alpha": "baseline_alpha_tstat", "tstat_premia": "baseline_tstat",
                            "t_stat": "baseline_tstat"}[plot_col]
                baseline_vals = E.dua_baseline_values(baselines, base_col)
                chosen = E.top4_per_cluster(sel_stats, by=by)
                sel = plot_stats[plot_stats["signal"].isin(chosen["signal"])].copy()
                assert len(sel) == 36, f"{exhibit}: selected {len(sel)} signals"

                indep = recompute_box(dua if grid == "dua" else mua, plot_col,
                                      list(sel["signal"]), signed_baseline)
                max_d = max(abs(float(r[c]) - indep[r["signal"]][c])
                            for _, r in sel.iterrows() for c in BOX_COLS)

                xlabel = "$t$-statistic (Alpha)" if "alpha" in plot_col else "$t$-statistic"
                for st in looks:
                    render(sel, list(chosen["signal"]), baseline_vals,
                           figstyle.out_path(fig_dir, f"{stem}_{args.window}.pdf", st), xlabel, st)
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
                             "sample": D.sample_block(
                                 first=E.window_span(args.window)[0],
                                 last=E.window_span(args.window)[1],
                                 window=args.window,
                                 basis="box statistics over paths, not a time series"),
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
