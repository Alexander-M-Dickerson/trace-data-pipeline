r"""f_lab_figures.py -- the three Section-4 figures.

  Figure 7    the look-ahead bias over time (Panels A/B) and against the VIX
              (Panels C/D, with a quadratic fit). The bias is largest when markets
              are most volatile, which is when the clipped returns actually happen.
  Figure 8    growth of $1 in the infeasible (winsorized) series, the feasible
              (baseline) series, and the cumulative bias between them.
  Figure IA.2 the bias split by leg, and again by rating.

❗Two display transforms, applied for readability and stated here because they change
what the picture shows:

  * `b_dunc3` and `ltr48_12` are plotted x(-1) throughout, so every panel reads as a
    strategy someone would hold long rather than short.
  * In Figure IA.2 those same two factors have their LEGS SWAPPED, and the short bar
    is always scaled by -1 so both bars point the way the reader expects.

Figure IA.2 carries an IDENTITY CHECK: every plotted bar is compared against the same
quantity from the statistics engine behind Table 4, at 1e-9. A figure that disagrees
with its own table is what this catches, and the rendered PDF cannot show it to you.

    python s2_lab/run_lab.py        # produce the series
    python s2_lab/f_lab_figures.py
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

import _stage3_settings as S   # noqa: E402
import drrlib as D          # noqa: E402
import lab_engine as E      # noqa: E402
import paths                # noqa: E402
from bench import Bench     # noqa: E402

FIG_DIR = paths.FIGURES
IDENT_TOL = 1e-9


def load_series_source() -> dict:
    import run_lab
    root = run_lab.series_root()
    if not any(root.glob("*.parquet")):
        raise SystemExit(f"the LAB series are not under {root}.\n"
                         "  Produce them with `python s2_lab/run_lab.py`.")
    return E.load_series(root)


def load_vix() -> pd.Series:
    fac = pd.read_parquet(paths.FACTORS, columns=["date", "vix"])
    fac["date"] = pd.to_datetime(fac["date"])
    return fac.set_index("date")["vix"].sort_index().astype(float)


# ------------------------------------------------------------------ Figure 7
FIG7_SCATTER = {"C": ("ivol_bbw", "left"), "D": ("mom3_1", "right")}


def fig7_stats(source: dict, vix: pd.Series) -> dict:
    """R^2 of the quadratic LAB-on-VIX fit and the correlation, panels C and D."""
    import statsmodels.api as sm
    out = {}
    for panel, (sig, tail) in FIG7_SCATTER.items():
        bias = source[("standard", "All", tail)]["ts_bias_ls"][sig] * 100
        df = pd.DataFrame({"lab": bias, "vix": vix}).dropna()
        X = df["vix"].to_numpy(float)
        y = df["lab"].to_numpy(float)
        model = sm.OLS(y, np.column_stack([np.ones_like(X), X, X ** 2])).fit()
        out[panel] = {"r2": float(model.rsquared),
                      "rho": float(np.corrcoef(y, X)[0, 1]), "n": len(df)}
    return out


def build_fig7(source: dict, vix: pd.Series, out_pdf: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    left = source[("standard", "All", "left")]["ts_bias_ls"]
    right = source[("standard", "All", "right")]["ts_bias_ls"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, (sigs, signs, title) in zip(
            axes[0], [(["b_dunc3", "ltr48_12", "ivol_bbw"], [-1, -1, 1],
                       "(A) Left-Tail Affected Factors"),
                      (["mom3_1", "mom6_1", "mom12_1"], [1, 1, 1],
                       "(B) Right-Tail Affected Factors (Momentum)")]):
        src = left if "b_dunc3" in sigs else right
        for sig, sg, color, ls in zip(sigs, signs,
                                      ["#08306b", "#2171b5", "#6baed6"],
                                      ["-", "--", ":"]):
            s = src[sig] * sg * 100
            ax.plot(s.index, s.values, color=color, lw=1.2, ls=ls, label=sig)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(title)
        ax.set_ylabel("LAB (%)")
        ax.legend(loc="upper left", fontsize=8)

    for ax, panel in zip(axes[1], ("C", "D")):
        sig, tail = FIG7_SCATTER[panel]
        src = left if tail == "left" else right
        df = pd.DataFrame({"lab": src[sig] * 100, "vix": vix}).dropna()
        X, y = df["vix"].to_numpy(float), df["lab"].to_numpy(float)
        ax.scatter(X, y, alpha=0.4, s=15, color="#1f78b4")
        model = sm.OLS(y, np.column_stack([np.ones_like(X), X, X ** 2])).fit()
        xl = np.linspace(X.min(), X.max(), 100)
        ax.plot(xl, model.params[0] + model.params[1] * xl + model.params[2] * xl ** 2,
                color="#e31a1c", lw=2)
        ax.text(0.95, 0.95,
                f"$R^2$ = {model.rsquared:.3f}\n$\\rho$ = {np.corrcoef(y, X)[0, 1]:.3f}",
                transform=ax.transAxes, va="top", ha="right", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(f"({panel}) LAB({sig}) vs VIX")
        ax.set_xlabel("VIX")
        ax.set_ylabel("LAB (%)")

    plt.tight_layout()
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Figure 8
FIG8_PANELS = [("A", "ltr48_12", "left", -1), ("B", "ivol_bbw", "left", 1),
               ("C", "b_dunc3", "left", -1), ("D", "mom3_1", "right", 1)]


def _fmt_final(v: float) -> str:
    return f"${v:.1f}" if v < 10 else f"${v:.0f}"


def fig8_data(source: dict) -> dict:
    out = {}
    for panel, sig, tail, sg in FIG8_PANELS:
        cell = source[("standard", "All", tail)]
        df = pd.DataFrame({"wins": cell["ts_ls_wins"][sig],
                           "base": cell["ts_ls_base"][sig],
                           "bias": cell["ts_bias_ls"][sig]}).dropna() * sg
        cum = (1 + df).cumprod()
        out[panel] = {"sig": sig, "cum": cum,
                      "finals": {k: float(cum[k].iloc[-1]) for k in df.columns}}
    return out


def build_fig8(data: dict, out_pdf: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    order = {"A": (0, 0), "B": (0, 1), "C": (1, 0), "D": (1, 1)}
    for panel, d in data.items():
        ax = axes[order[panel]]
        cum = d["cum"]
        ax.plot(cum.index, cum["wins"], color="#a6cee3", lw=1.2, label="Infeasible (Wins.)")
        ax.plot(cum.index, cum["base"], color="#08306b", lw=1.2, ls="--", label="Feasible")
        ax.plot(cum.index, cum["bias"], color="gray", lw=1.0, ls=":", label="Cumulative LAB")
        ax.set_yscale("log")
        ax.set_title(f"({panel}) {d['sig']}")
        ax.axhline(1, color="gray", lw=0.5)
        for k, color in zip(("wins", "base", "bias"), ("#a6cee3", "#08306b", "gray")):
            v = d["finals"][k]
            ax.annotate(_fmt_final(v), xy=(cum.index[-1], v), xytext=(4, 0),
                        textcoords="offset points", color=color, fontsize=8)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Figure IA.2
IA2_LEFT = ["b_dunc3", "ltr48_12", "ivol_bbw"]
IA2_RIGHT = ["mom3_1", "mom6_1", "mom12_1"]
IA2_SWAP = {"b_dunc3", "ltr48_12"}
IA2_SIGN = {"b_dunc3": -1, "ltr48_12": -1}


def ia2_data(source: dict) -> dict:
    """The bar values: leg decomposition (All) + ls bias by rating, all x100."""
    out = {"legs": {}, "rating": {}}
    for tail, sigs in (("left", IA2_LEFT), ("right", IA2_RIGHT)):
        cell = source[("standard", "All", tail)]
        for f in sigs:
            bl = float(cell["ts_bias_long"][f].mean() * 100)
            bs = float(cell["ts_bias_short"][f].mean() * 100)
            if f in IA2_SWAP:
                out["legs"][f] = {"long_bar": bs, "short_bar": -bl}
            else:
                out["legs"][f] = {"long_bar": bl, "short_bar": -bs}
    for rating in E.RATINGS:
        for tail, sigs in (("left", IA2_LEFT), ("right", IA2_RIGHT)):
            cell = source[("standard", rating, tail)]
            for f in sigs:
                v = float(cell["ts_bias_ls"][f].mean() * IA2_SIGN.get(f, 1) * 100)
                out["rating"][f"{f}|{rating}"] = v
    return out


def ia2_identity(data: dict, stats: pd.DataFrame) -> float:
    """Max |figure value - engine value| across every bar (both in %)."""
    worst = 0.0
    b = stats[(stats["variant"] == "bias") & (stats["stat"] == "mu")]
    for f, bars in data["legs"].items():
        sub = b[(b["factor"] == f) & (b["rating"] == "All")]
        bl = float(sub[sub["leg"] == "long"]["value"].iloc[0])
        bs = float(sub[sub["leg"] == "short"]["value"].iloc[0])
        exp_long, exp_short = ((bs, -bl) if f in IA2_SWAP else (bl, -bs))
        worst = max(worst, abs(bars["long_bar"] - exp_long),
                    abs(bars["short_bar"] - exp_short))
    for key, v in data["rating"].items():
        f, rating = key.split("|")
        sub = b[(b["factor"] == f) & (b["rating"] == rating) & (b["leg"] == "ls")]
        exp = float(sub["value"].iloc[0]) * IA2_SIGN.get(f, 1)
        worst = max(worst, abs(v - exp))
    return worst


def build_ia2(data: dict, out_pdf: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.25)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    width = 0.35
    for ax, sigs, title in ((ax_a, IA2_LEFT, "(A) Left-Tail Factors"),
                            (ax_b, IA2_RIGHT, "(B) Right-Tail Factors (Momentum)")):
        x = np.arange(len(sigs))
        ax.bar(x - width / 2, [data["legs"][f]["long_bar"] for f in sigs], width,
               label="Long Leg", color="#08306b")
        ax.bar(x + width / 2, [data["legs"][f]["short_bar"] for f in sigs], width,
               label="Short Leg", color="#6baed6")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(title)
        ax.set_ylabel("Average LAB (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(sigs, fontsize=8)
        ax.legend(fontsize=8)
    sigs = IA2_LEFT + IA2_RIGHT
    x = np.arange(len(sigs))
    wc = 0.25
    for rating, off, color in (("All", -wc, "#08306b"), ("IG", 0, "#2171b5"),
                               ("NIG", wc, "#6baed6")):
        ax_c.bar(x + off, [data["rating"][f"{f}|{rating}"] for f in sigs], wc,
                 label=rating, color=color)
    ax_c.axhline(0, color="gray", lw=0.5)
    ax_c.set_title("(C) LAB by Rating Category")
    ax_c.set_ylabel("Average LAB (%)")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(sigs, fontsize=8)
    ax_c.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    with Bench("lab-figures", section="s2_lab", sample=not args.no_bench,
               echo=True) as b:
        with b.phase("load"):
            source = load_series_source()
            vix = load_vix()
            mktb = E.load_mktb_lab()
            specs = [E.LabSpec(tail=t, rating=r)
                     for t in ("left", "right") for r in E.RATINGS]
            stats = E.build(source, specs, mktb=mktb)
        with b.phase("fig7"):
            f7 = fig7_stats(source, vix)
            build_fig7(source, vix, FIG_DIR / "fig07_lab_bias_2x2.pdf")
        with b.phase("fig8"):
            f8 = fig8_data(source)
            build_fig8(f8, FIG_DIR / "fig08_lab_cumret_2x2.pdf")
        with b.phase("figIA2"):
            ia2 = ia2_data(source)
            ident = ia2_identity(ia2, stats)
            build_ia2(ia2, FIG_DIR / "figIA2_lab_decomposition_rating.pdf")
        with b.phase("record"):
            out = paths.section_results("s2_lab")
            pd.DataFrame([{"panel": k, **v} for k, v in f7.items()]).to_csv(
                out / "fig07_vix_fit.csv", index=False)
            pd.DataFrame([{"panel": k, "signal": d["sig"], "series": s_,
                           "final_value": v}
                          for k, d in f8.items()
                          for s_, v in d["finals"].items()]).to_csv(
                out / "fig08_finals.csv", index=False)
            pd.DataFrame(
                [{"kind": "leg", "factor": f, **bars}
                 for f, bars in ia2["legs"].items()]
                + [{"kind": "rating", "factor": k.split("|")[0],
                    "rating": k.split("|")[1], "value": v}
                   for k, v in ia2["rating"].items()]).to_csv(
                out / "figIA2_bars.csv", index=False)
            D.write_result(
                "lab_figures",
                {"summary": {"exhibit": "Figures 7, 8, IA.2",
                             "figIA2_identity_max_abs_diff": ident,
                             "identity_ok": bool(ident <= IDENT_TOL),
                             "sample": D.sample_block(
                                 first=S.SAMPLE["lab"]["start"],
                                 last=S.SAMPLE["lab"]["end"],
                                 basis="the LAB window; each series is its own length")},
                 "fig07_vix_fit": f7,
                 "fig08_finals": {k: d["finals"] for k, d in f8.items()},
                 "figIA2_bars": ia2},
                section="s2_lab", inputs=[paths.FACTORS, paths.BBW], t0=t0,
                extra={"exhibit": "Figures 7, 8, IA.2"})
        b.note(figIA2_identity=ident,
               fig07_r2={k: round(v["r2"], 4) for k, v in f7.items()})
        ok = b.check(ident <= IDENT_TOL,
                     f"Figure IA.2 identity vs the statistics engine: "
                     f"max|d|={ident:.2e} (tolerance {IDENT_TOL:g})")

    print("\nFigure 7 VIX fits:")
    for panel, v in f7.items():
        print(f"  panel {panel}: R2={v['r2']:.3f}  rho={v['rho']:.3f}  n={v['n']}")
    print(f"\nwrote {FIG_DIR / 'fig07_lab_bias_2x2.pdf'}"
          f"\n      {FIG_DIR / 'fig08_lab_cumret_2x2.pdf'}"
          f"\n      {FIG_DIR / 'figIA2_lab_decomposition_rating.pdf'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
