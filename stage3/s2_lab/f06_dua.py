r"""f06_dua.py -- Figure 6: momentum's alpha against the return-trimming threshold.

The figure that separates the two kinds of filtering. Momentum(6,6) is sorted under
nine trim thresholds in each tail, twice over:

  ex ante   the threshold is applied to the returns used to FORM the portfolio, which
            an investor could have done in real time
  ex post   the threshold is applied to the returns the portfolio EARNED, which nobody
            could have done

Four panels: (left, right) tail x (ex ante, ex post). Each bar is the CAPM_B alpha with
a 1.96 x Newey-West whisker.

❗The claim this figure supports is about the EX-ANTE panels: under ex-ante trimming the
alpha is statistically indistinguishable from zero at every threshold in both tails. So
this driver checks exactly that -- all 18 ex-ante |t| below 1.96 -- and reports any bar
that breaks it rather than leaving the reader to squint at a whisker.

Unlike the rest of Section 4, this runs its own small sweep: momentum at a SIX-month
holding period is not one of the 108 signals, so no stored grid covers it.

    python s2_lab/f06_dua.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # stage3/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _stage3_settings as S    # noqa: E402
import drrlib as D              # noqa: E402
import paths                    # noqa: E402
import pblenv                   # noqa: E402
from bench import Bench         # noqa: E402

SIGNAL = "mom6_1"           # 6-month formation, skipping the most recent month
HOLDING = 6                 # ... held for six months: "Momentum(6,6)"
TRIM_PCTS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
MIN_T = 200                 # a 6-month hold starts the series late; below this, stop
PANELS = [("left", "ex_ante", "(A) Ex-Ante, Left-Tail"),
          ("left", "ex_post", "(B) Ex-Post, Left-Tail"),
          ("right", "ex_ante", "(C) Ex-Ante, Right-Tail"),
          ("right", "ex_post", "(D) Ex-Post, Right-Tail")]


def trim_grid() -> dict:
    """Nine thresholds per tail, as signed return cut-offs."""
    return {"trim": [round(-p / 100, 4) for p in TRIM_PCTS]
                    + [round(p / 100, 4) for p in TRIM_PCTS]}


def run_sweep():
    """The DataUncertaintyAnalysis fit behind all four panels."""
    import warnings

    from PyBondLab import DataUncertaintyAnalysis

    if paths.PANEL is None or not Path(paths.PANEL).exists():
        raise SystemExit(f"Figure 6 needs the Stage 2 panel and it is not at "
                         f"{paths.PANEL}.")
    cols = ["cusip", "date", "ret_vw", "mcap_e", "spc_rat", "rfret", SIGNAL]
    data = pd.read_parquet(paths.PANEL, columns=cols)
    data["date"] = pd.to_datetime(data["date"])
    data["ret_vw"] = data["ret_vw"] - data["rfret"]
    data = data[(data["date"] >= S.SAMPLE["lab"]["start"])
                & (data["date"] <= S.SAMPLE["lab"]["end"])].copy()
    data["spc_rat"] = data["spc_rat"].astype("float64")
    assert data.duplicated(["cusip", "date"]).sum() == 0, "duplicate (cusip, date)"
    mapped = data.rename(columns={"cusip": "ID", "mcap_e": "VW",
                                  "spc_rat": "RATING_NUM", "ret_vw": "ret"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DataUncertaintyAnalysis(
            data=mapped, signals=[SIGNAL], holding_periods=[HOLDING],
            num_portfolios=10, filters=trim_grid(), include_baseline=True,
            dynamic_weights=True, verbose=False,
        ).fit()


def compute_panels(result, mktb: pd.Series) -> dict:
    """{panel title: DataFrame(col, alpha, se, t, T)} -- the plotted bars."""
    import statsmodels.api as sm

    out = {}
    for location, timing, title in PANELS:
        filtered = result.filter(filter_type="trim", location=location)
        df_returns = getattr(filtered, f"vw_{timing}", None)
        if df_returns is None or df_returns.empty:
            raise ValueError(f"no vw_{timing} returns for location={location}")
        rows = []
        for col in df_returns.columns:
            r = df_returns[col].dropna()
            df = pd.DataFrame({"ret": r, "mktb": mktb}).dropna()
            T = len(df)
            n_lags = int(np.floor(T ** 0.25))
            res = sm.OLS(df["ret"].to_numpy(float),
                         sm.add_constant(df["mktb"].to_numpy(float))
                         ).fit(cov_type="HAC", cov_kwds={"maxlags": n_lags})
            a, se = float(res.params[0] * 100), float(res.bse[0] * 100)
            rows.append({"panel": title, "location": location, "timing": timing,
                         "col": col, "alpha": a, "se": se,
                         "t": a / se if se > 0 else np.nan, "T": T,
                         "start": str(df.index.min().date()),
                         "end": str(df.index.max().date())})
        out[title] = pd.DataFrame(rows)
    return out


def build_figure(panels: dict, out_pdf: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (location, timing, title) in zip(axes.flatten(), PANELS):
        df = panels[title]
        x = np.arange(len(df))
        ax.bar(x, df["alpha"], yerr=1.96 * df["se"], capsize=3,
               color="#2171b5", alpha=0.85, edgecolor="gray", linewidth=0.5)
        labels = ([f"<-{p}%" for p in TRIM_PCTS] if location == "left"
                  else [f">{p}%" for p in TRIM_PCTS])[:len(df)]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(title)
        ax.set_ylabel(r"Alpha ($\alpha$, % monthly)")
    plt.tight_layout()
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    pblenv.use()
    t0 = time.perf_counter()

    with Bench("fig06-dua", section="s2_lab", sample=not args.no_bench,
               echo=True) as b:
        with b.phase("sweep"):
            result = run_sweep()
        with b.phase("stats"):
            mktb = D.load_mktb(paths.BBW, start=S.SAMPLE["lab"]["start"],
                               end=S.SAMPLE["lab"]["end"])
            panels = compute_panels(result, mktb)
            allp = pd.concat(panels.values(), ignore_index=True)
        with b.phase("render"):
            out_pdf = paths.FIGURES / "fig06_momentum_trim.pdf"
            build_figure(panels, out_pdf)
            out = paths.section_results("s2_lab")
            allp.to_csv(out / "fig06_cells.csv", index=False)
            ex_ante = allp[allp["timing"] == "ex_ante"]
            significant = ex_ante[ex_ante["t"].abs() >= 1.96]
            D.write_result(
                "fig06_dua",
                {"summary": {"exhibit": "Figure 6", "signal": SIGNAL,
                             "holding_period": HOLDING,
                             "n_bars": len(allp), "T_min": int(allp["T"].min()),
                             "T_max": int(allp["T"].max()),
                             "sample": D.sample_block(
                                 first=str(allp["start"].min())[:10],
                                 last=str(allp["end"].max())[:10],
                                 T_min=int(allp["T"].min()),
                                 T_max=int(allp["T"].max()),
                                 basis="the ex-ante momentum sweep, per bar"),
                             "n_ex_ante": len(ex_ante),
                             "n_ex_ante_significant": len(significant)},
                 "bars": allp.to_dict("records")},
                section="s2_lab", inputs=[paths.PANEL, paths.BBW], t0=t0,
                extra={"exhibit": "Figure 6"})
        b.note(n_bars=len(allp), T_min=int(allp["T"].min()),
               T_max=int(allp["T"].max()))
        # The figure's claim, checked rather than eyeballed.
        ok = b.check(len(significant) == 0 and int(allp["T"].min()) >= MIN_T,
                     f"{len(ex_ante) - len(significant)}/{len(ex_ante)} ex-ante alphas "
                     f"indistinguishable from zero, T in "
                     f"({int(allp['T'].min())}, {int(allp['T'].max())})")

    if len(significant):
        print("\nex-ante bars with |t| >= 1.96:")
        print(significant[["panel", "col", "alpha", "t", "T"]].to_string(index=False))
    print(f"\nwrote {out_pdf}\n      {out / 'fig06_cells.csv'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
