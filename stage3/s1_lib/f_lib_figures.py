r"""f_lib_figures.py -- the three Section-3 figures.

  Figure 3   cumulative value of $1 under each of the three approaches, 2x2 over
             (short-term reversal, credit spread) x (single, within-firm). The gap
             between the lines IS the bias, compounded.
  Figure 4   the bias magnitude per factor, with 1.96 x Newey-West whiskers, beside
             the share of it the LIB characteristic accounts for.
  Figure IA.1  the same bias averaged over the seven factors, split by rating.

Every series is loaded flip-undone, so a sign-corrected factor is plotted in its true
orientation. Within each panel of Figure 3 all four series are flipped together when
the unadjusted mean is negative, so the panel reads as a long-short strategy someone
would actually run.

Figures 4 and IA.1 carry an IDENTITY CHECK: each plotted bar is compared against the
same quantity computed by the statistics engine behind Tables 1 and 2, at 1e-9. A
figure that disagrees with its own table is the failure worth catching, and the picture
itself cannot show you that it has happened.

The data is computed once and drawn in each look `--style` asks for (figstyle.py): the
house look (the default, the paper's) or the submitted look. Figure 3 shades the NBER recessions.

    python s1_lib/f_lib_figures.py [--style house|submitted|both]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import drrlib as D          # noqa: E402
import figstyle             # noqa: E402
import lib_engine as E      # noqa: E402
import paths                # noqa: E402
from bench import Bench     # noqa: E402

SIGNALS = list(D.LIB_FACTORS)
FIG_DIR = paths.FIGURES
IDENT_TOL = 1e-9        # a plotted bar vs the statistics engine, both in percent

# the sample the figures cover; --end moves it, and then T comes from the data
CTX = {"start": D.SAMPLE_START, "end": D.SAMPLE_END, "expected_T": 268}


def _mktb():
    return D.load_mktb(paths.BBW, start=CTX["start"], end=CTX["end"])


def _spec(**kw) -> E.LibSpec:
    return E.LibSpec(start=CTX["start"], end=CTX["end"], **kw)


def _series(sort: str, sig: str, root, rating: str = "all"):
    kw = dict(root=root or paths.SORTS)
    win = dict(start=CTX["start"], end=CTX["end"])
    f = lambda a: E.sort_csv(a, sort=sort, rating=rating, **kw)  # noqa: E731
    r1 = D.load_sort_panel(f("unadjusted"), **win)[sig]
    r2 = D.load_sort_panel(f("adj_signal"), **win)[sig]
    r3 = D.load_sort_panel(f("adj_return"), **win)[sig]
    lib = D.load_sort_panel(f("adj_return"), value_col="lib", **win)[sig]
    return r1, r2, r3, lib


# --------------------------------------------------------------- Figure 3
def fig3_data(root: Path | None = None) -> dict:
    out = {}
    for key, (sig, sort) in {"A": ("str", "single"), "B": ("str", "wf"),
                             "C": ("cs", "single"), "D": ("cs", "wf")}.items():
        r1, r2, r3, lib = _series(sort, sig, root)
        df = pd.concat({"r1": r1, "r2": r2, "r3": r3, "lib": lib}, axis=1).dropna()
        if df["r1"].mean() < 0:
            df = -df
        out[key] = {"sig": sig, "sort": sort, "cum": (1 + df).cumprod()}
    return out


# the series of each panel: (column, legend label, colour role, line style)
FIG3_SERIES = [("r1", "With Noise", "unadjusted", "-"), ("r2", "Adj. Signal", "adj_signal", "--"),
               ("r3", "Adj. Return", "adj_return", "-"), ("lib", "Cumulative LIB", "lib_line", ":")]


def fig3_finals(data: dict) -> dict:
    """The growth of $1 at the end of the sample, per panel and series (printed in the prose)."""
    return {key: {c: float(d["cum"][c].iloc[-1]) for c, *_ in FIG3_SERIES}
            for key, d in data.items()}


def build_fig3(data: dict, out_pdf: Path, st) -> None:
    with figstyle.use(st) as plt:
        fig, axes = plt.subplots(2, 2, figsize=st.size(12, 8))
        order = {"A": (0, 0), "B": (0, 1), "C": (1, 0), "D": (1, 1)}
        finals = fig3_finals(data)
        for key, d in data.items():
            ax = axes[order[key]]
            cum = d["cum"]
            st.recessions(ax, cum.index)
            for c, lab, role, ls in FIG3_SERIES:
                ax.plot(cum.index, cum[c], color=st.c(role), ls=ls,
                        lw=(1.0 if c == "lib" else 1.2), label=lab)
            sort = "Single-Sort" if d["sort"] == "single" else "Within-Firm"
            st.panel(ax, key, f"{st.factor(d['sig'])}: {sort}")
            ax.axhline(1, color=st.c("zero"), lw=0.5)
            st.grid(ax, axis="both")
            figstyle.date_axis(ax)
            figstyle.dollar_axes(ax, cum, [finals[key][c] for c, *_ in FIG3_SERIES],
                                 [st.c(role) for _, _, role, _ in FIG3_SERIES], st)
        h, lab = axes[0, 0].get_legend_handles_labels()
        if st.name == "submitted":
            st.add_legend(axes[0, 0], h, lab, loc="upper left", fontsize=8)
            fig.text(0.02, 0.5, "Dollar Value", va="center", rotation="vertical", fontsize=12)
            fig.tight_layout(rect=(0.03, 0, 1, 1))
        else:
            fig.supylabel("Dollar value", fontsize=8.5)
            st.legends(fig, [(axes[0, 0], h, lab, {})])
        figstyle.save(fig, out_pdf, bbox_inches="tight")
        plt.close(fig)


# --------------------------------------------------------------- Figure 4
def fig4_data(root: Path | None = None) -> dict[str, pd.DataFrame]:
    out = {}
    for sort in ("single", "wf"):
        rows = []
        for sig in SIGNALS:
            r1, r2, r3, lib = _series(sort, sig, root)
            lags = D.nw_lags(len(r1.dropna()))
            b12, t12 = D.paired_diff_mean(r1, r2, lags)
            b13, t13 = D.paired_diff_mean(r1, r3, lags)
            mu1 = float(r1.mean())
            mul = float(lib.mean())
            lib_pct = min(abs(mul) / abs(mu1) * 100, 100)
            rows.append({"signal": sig,
                         "bias_1_2": abs(b12) * D.PCT, "se_1_2": abs(b12 / t12) * D.PCT,
                         "bias_1_3": abs(b13) * D.PCT, "se_1_3": abs(b13 / t13) * D.PCT,
                         "lib_pct": lib_pct, "actual_pct": 100 - lib_pct})
        out[sort] = pd.DataFrame(rows)
    return out


def check_fig4(data: dict, root: Path | None = None) -> dict:
    """Identity: every plotted bar equals the statistics engine's own number."""
    rep = {"checks": [], "pass": True}
    for sort, df in data.items():
        spec = _spec(sort=sort)
        stats = E.lib_stats(E.load_series(spec, root=root), _mktb(), spec, SIGNALS,
                            expected_T=CTX["expected_T"])
        piv = stats.pivot_table(index="factor", columns="quantity", values="value")
        val = E.validation_stats(spec, tuple(SIGNALS), root=root,
                                 expected_T=CTX["expected_T"]).set_index("factor")
        for _, r in df.iterrows():
            f = r["signal"]
            d12 = abs(r["bias_1_2"] - abs(piv.loc[f, "bias_1_2.d_mu"]))
            d13 = abs(r["bias_1_3"] - abs(piv.loc[f, "bias_1_3.d_mu"]))
            # decomposition vs Table 2's engine: same mu_lib / mu_end, but the figure
            # uses the PLAIN mean of r1 while validation_stats uses the NW mean --
            # identical values (OLS on a constant IS the mean), so tolerance holds
            pct = min(abs(val.loc[f, "mu_lib"]) / abs(val.loc[f, "mu_end"]) * 100, 100)
            dp = abs(r["lib_pct"] - pct)
            ok = max(d12, d13, dp) <= IDENT_TOL
            rep["checks"].append({"sort": sort, "signal": f, "d_bias12": d12,
                                  "d_bias13": d13, "d_libpct": dp, "ok": bool(ok)})
            rep["pass"] &= bool(ok)
    return rep


def bias_bars(ax, x, df, st, capsize: float) -> None:
    """The two biases side by side with 1.96 x Newey-West whiskers (Figures 4 A-B and IA.1)."""
    alpha = 0.8 if st.name == "submitted" else 1.0
    err = dict(ecolor=st.c("whisker"), elinewidth=(1.0 if st.name == "submitted" else 0.7))
    ax.bar(x - 0.175, df["bias_1_2"], 0.35, yerr=1.96 * df["se_1_2"], capsize=capsize,
           label=r"Bias (1)$-$(2): Signal Adj.", color=st.c("bias12"), alpha=alpha, error_kw=err)
    ax.bar(x + 0.175, df["bias_1_3"], 0.35, yerr=1.96 * df["se_1_3"], capsize=capsize,
           label=r"Bias (1)$-$(3): Return Adj.", color=st.c("bias13"), alpha=alpha, error_kw=err)
    ax.axhline(0, color=st.c("zero"), lw=0.5)
    st.grid(ax)


def build_fig4(data: dict, out_pdf: Path, st) -> None:
    with figstyle.use(st) as plt:
        fig, axes = plt.subplots(2, 2, figsize=st.size(12, 9))
        # the house panels are narrower: val_ipr and val_hz would run together unrotated
        ticks = {"fontsize": 10} if st.name == "submitted" else {"rotation": 30, "ha": "right"}
        for j, (sort, title) in enumerate([("single", "Single-Sort"), ("wf", "Within-Firm Sort")]):
            df = data[sort]
            x = np.arange(len(df))
            names = [st.factor(s_) for s_ in df["signal"]]
            ax = axes[0, j]
            bias_bars(ax, x, df, st, capsize=3)
            ax.set_ylabel("Bias (% monthly)")
            ax.set_xticks(x, names, **ticks)
            st.panel(ax, "AB"[j], title)
            ax = axes[1, j]
            alpha = 0.9 if st.name == "submitted" else 1.0
            ax.bar(x, df["lib_pct"], 0.6, label="LIB", color=st.c("lib_bar"), alpha=alpha)
            # Referee 1 (round 2, point B): the implementable part is the ADJUSTED return
            ax.bar(x, df["actual_pct"], 0.6, bottom=df["lib_pct"], label="Adjusted Return",
                   color=st.c("implementable"), alpha=alpha)
            ax.set_ylabel("Decomposition (%)")
            ax.set_ylim(0, 105)
            ax.axhline(100, color=st.c("zero"), lw=0.5, ls="--")
            ax.set_xticks(x, names, **ticks)
            st.panel(ax, "CD"[j], f"{title} - Decomposition")
            st.grid(ax)
        hb, lb = axes[0, 1].get_legend_handles_labels()
        # the stacked bars read top-down, so the legend does too: adjusted return, then LIB
        hd, ld = axes[1, 1].get_legend_handles_labels()
        st.legends(fig, [(axes[0, 1], hb, lb, {"loc": "upper right"}),
                         (axes[1, 1], hd[::-1], ld[::-1], {"loc": "upper right"})])
        figstyle.save(fig, out_pdf, bbox_inches="tight")
        plt.close(fig)


# --------------------------------------------------------------- Figure IA.1
def figia1_data(root: Path | None = None) -> dict[str, pd.DataFrame]:
    out = {}
    for sort in ("single", "wf"):
        rows = []
        for rating in ("all", "ig", "nig"):
            b12, b13 = [], []
            for sig in SIGNALS:
                r1, r2, r3, _ = _series(sort, sig, root, rating=rating)
                b12.append(abs(float((r1 - r2).dropna().mean())) * D.PCT)
                b13.append(abs(float((r1 - r3).dropna().mean())) * D.PCT)
            rows.append({"rating": rating,
                         "bias_1_2": float(np.nanmean(b12)),
                         "se_1_2": float(np.nanstd(b12) / np.sqrt(len(b12))),
                         "bias_1_3": float(np.nanmean(b13)),
                         "se_1_3": float(np.nanstd(b13) / np.sqrt(len(b13)))})
        out[sort] = pd.DataFrame(rows)
    return out


def check_figia1(data: dict, root: Path | None = None) -> dict:
    """Identity: each bar equals the mean of |d_mu| from the stats engine."""
    rep = {"checks": [], "pass": True}
    mktb = _mktb()
    for sort, df in data.items():
        for _, r in df.iterrows():
            spec = _spec(sort=sort, rating=r["rating"])
            stats = E.lib_stats(E.load_series(spec, root=root), mktb, spec, SIGNALS,
                                expected_T=CTX["expected_T"])
            piv = stats.pivot_table(index="factor", columns="quantity", values="value")
            m12 = float(piv["bias_1_2.d_mu"].abs().mean())
            m13 = float(piv["bias_1_3.d_mu"].abs().mean())
            d = max(abs(r["bias_1_2"] - m12), abs(r["bias_1_3"] - m13))
            rep["checks"].append({"sort": sort, "rating": r["rating"],
                                  "max_d": d, "ok": bool(d <= IDENT_TOL)})
            rep["pass"] &= bool(d <= IDENT_TOL)
    return rep


def build_figia1(data: dict, out_pdf: Path, st) -> None:
    labels = {"all": "All Bonds", "ig": "Inv. Grade", "nig": "Non-Inv. Grade"}
    with figstyle.use(st) as plt:
        fig, axes = plt.subplots(1, 2, figsize=st.size(10, 5))
        for j, (sort, title) in enumerate([("single", "Single-Sort"), ("wf", "Within-Firm Sort")]):
            df = data[sort]
            x = np.arange(len(df))
            ax = axes[j]
            bias_bars(ax, x, df, st, capsize=4)
            ax.set_ylabel("Average Bias (% monthly)")
            ax.set_xticks(x, [labels[r] for r in df["rating"]])
            st.panel(ax, "AB"[j], title)
        h, lab = axes[1].get_legend_handles_labels()
        st.legends(fig, [(axes[1], h, lab, {"loc": "upper right"})])
        figstyle.save(fig, out_pdf, bbox_inches="tight")
        plt.close(fig)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--end", default=None,
                    help="override the sample end; T is then derived from the data")
    ap.add_argument("--no-bench", action="store_true")
    figstyle.add_argument(ap)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()
    if args.end:
        CTX["end"] = args.end
        CTX["expected_T"] = None
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    looks = figstyle.styles(args.style)

    with Bench("lib-figures", section="s1_lib", sample=not args.no_bench,
               echo=True) as b:
        with b.phase("fig3"):
            d3 = fig3_data()
            finals = fig3_finals(d3)
            for st in looks:
                build_fig3(d3, figstyle.out_path(FIG_DIR, "fig03_cumret.pdf", st), st)
        with b.phase("fig4"):
            d4 = fig4_data()
            g4 = check_fig4(d4)
            for st in looks:
                build_fig4(d4, figstyle.out_path(FIG_DIR, "fig04_bias.pdf", st), st)
        with b.phase("figia1"):
            dia = figia1_data()
            gia = check_figia1(dia)
            for st in looks:
                build_figia1(dia, figstyle.out_path(FIG_DIR, "figIA1_bias_by_rating.pdf", st), st)
        with b.phase("record"):
            out = paths.section_results("s1_lib")
            for sort, df in d4.items():
                df.to_csv(out / f"fig04_{sort}.csv", index=False)
            for sort, df in dia.items():
                df.to_csv(out / f"figIA1_{sort}.csv", index=False)
            pd.DataFrame([{"panel": k, "series": s_, "final_value": v}
                          for k, vals in finals.items() for s_, v in vals.items()]
                         ).to_csv(out / "fig03_finals.csv", index=False)
            D.write_result(
                "lib_figures",
                {"summary": {"exhibit": "Figures 3, 4, IA.1",
                             "identity_fig4": bool(g4["pass"]),
                             "identity_figIA1": bool(gia["pass"]),
                             "sample": D.sample_block(
                                 first=CTX["start"], last=CTX["end"],
                                 T=CTX["expected_T"],
                                 basis="the LIB window; T is asserted")},
                 "fig03_finals": finals,
                 "fig04": {k: v.to_dict("records") for k, v in d4.items()},
                 "figIA1": {k: v.to_dict("records") for k, v in dia.items()},
                 "identity_checks": {"fig04": g4, "figIA1": gia}},
                section="s1_lib",
                inputs=[paths.BBW], t0=t0,
                extra={"exhibit": "Figures 3, 4, IA.1"})
        b.note(n_signals=len(SIGNALS), sample_end=CTX["end"])
        # A figure that disagrees with the table behind it is the defect this catches;
        # nothing in the rendered PDF would show it.
        ok = b.check(bool(g4["pass"] and gia["pass"]),
                     f"identity vs the statistics engine at {IDENT_TOL:g}: "
                     f"fig4 {'ok' if g4['pass'] else 'FAIL'}, "
                     f"figIA.1 {'ok' if gia['pass'] else 'FAIL'}")

    print(f"\nwrote {FIG_DIR / 'fig03_cumret.pdf'}"
          f"\n      {FIG_DIR / 'fig04_bias.pdf'}"
          f"\n      {FIG_DIR / 'figIA1_bias_by_rating.pdf'}")
    if not ok:
        for c in g4["checks"] + gia["checks"]:
            if not c["ok"]:
                print(f"  IDENTITY FAIL {c}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
