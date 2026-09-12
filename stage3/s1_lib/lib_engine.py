"""lib_engine.py -- the ONE computation behind every Section-3 (LIB) exhibit.

Table 1, Table 2, Figures 3 and 4, Table B.1 and Tables IA.XII-XIV all read the same
long-short return series and ask slightly different questions of them. So this module
computes the series and the statistics ONCE into a tidy frame, and every exhibit is a
formatter over that frame.

That is deliberate, and it is why the section is fast: the naive shape refits four HAC
regressions per factor, per table and per figure, over the same series each time.

The three approaches (the paper's Section 3):

    (1) unadjusted   noisy month-end signal  + month-end return
    (2) adj_signal   gapped signal (>=1 BD before month-end) + month-end return
    (3) adj_return   noisy month-end signal  + month-begin return

Bias (1)-(2) varies the portfolio weights holding the return fixed; Bias (1)-(3)
varies the return window holding the weights fixed -- the latter is LIB.

The series come from `run_sorts.py`, which writes them into `data/sorts/`.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # stage3/

import drrlib as D          # noqa: E402
import paths                # noqa: E402

APPROACHES = ("unadjusted", "adj_signal", "adj_return")
BIAS_PAIRS = {"bias_1_2": ("unadjusted", "adj_signal"),
              "bias_1_3": ("unadjusted", "adj_return")}

# the file grammar: {ret}_{sort}_{rating}_{signal_type}_{timing}_p{N}.csv
_SIGNAL_TYPE = {"unadjusted": "mmn", "adj_signal": "baseline", "adj_return": "mmn"}
_TIMING = {"unadjusted": "end", "adj_signal": "end", "adj_return": "bgn"}


def n_portfolios(sort: str, rating: str) -> int:
    """Deciles for all-bond single sorts, quintiles for the rating splits, 2 within-firm.

    Matches the paper: "Panel A sorts bonds into deciles"; the public zoo uses quintiles
    for IG and NIG; within-firm is a high/low split.
    """
    if sort == "wf":
        return 2
    return 10 if rating == "all" else 5


def sort_csv(approach: str, *, sort: str = "single", rating: str = "all",
             ret_type: str = "exc", root: Path | None = None) -> Path:
    """The sort CSV holding one approach's series, by the grammar run_sorts writes."""
    root = Path(root or paths.SORTS)
    name = (f"{ret_type}_{sort}_{rating}_{_SIGNAL_TYPE[approach]}_"
            f"{_TIMING[approach]}_p{n_portfolios(sort, rating)}.csv")
    return root / name


@dataclass(frozen=True)
class LibSpec:
    """One cell of the LIB design: which sort, which universe, which weighting."""
    sort: str = "single"        # 'single' | 'wf'
    rating: str = "all"         # 'all' | 'ig' | 'nig'
    weighting: str = "vw"       # 'vw' | 'ew'
    ret_type: str = "exc"       # 'exc' (excess) | 'dur' (duration-adjusted)
    start: str = D.SAMPLE_START
    end: str = D.SAMPLE_END

    @property
    def key(self) -> str:
        return f"{self.ret_type}_{self.sort}_{self.rating}_{self.weighting}"


def load_series(spec: LibSpec, *, root: Path | None = None) -> dict[str, pd.DataFrame]:
    """{approach: wide date x mnemonic frame of long-short returns}, sign flips undone."""
    out = {}
    for a in APPROACHES:
        out[a] = D.load_sort_panel(
            sort_csv(a, sort=spec.sort, rating=spec.rating,
                       ret_type=spec.ret_type, root=root),
            leg="ls", weighting=spec.weighting, start=spec.start, end=spec.end)
    return out


def lib_stats(series: dict[str, pd.DataFrame], mktb: pd.Series, spec: LibSpec,
              factors: tuple[str, ...] | None = None,
              *, expected_T: int | None = 268) -> pd.DataFrame:
    """The tidy stats frame: one row per (factor, quantity), in PERCENT.

    Columns: spec fields, factor, quantity, value, tstat, T. `quantity` is one of the
    three approaches (mu/alpha) or one of the two bias pairs, encoded as
    '<approach>.mu', '<approach>.alpha', '<pair>.d_mu', '<pair>.d_alpha'.
    """
    common = sorted(set.intersection(*(set(df.columns) for df in series.values())))
    factors = tuple(factors) if factors else tuple(common)
    missing = [f for f in factors if f not in common]
    if missing:
        raise KeyError(f"factors absent from at least one approach: {missing}")

    idx = series["unadjusted"].index
    if expected_T is not None:
        D.assert_sample(idx, expected_T, what=f"{spec.key} LIB sample")
    lags = D.nw_lags(len(idx))

    rows = []

    # ❗The REALISED span, carried beside T. The exhibits used to record the window
    # they ASKED for -- the LibSpec constants -- while the index they actually got sat
    # here unread, so a caption built on it would have described the intention rather
    # than the result.
    first, last = str(idx.min())[:10], str(idx.max())[:10]

    def add(factor: str, quantity: str, value: float, tstat: float):
        rows.append({"ret_type": spec.ret_type, "sort": spec.sort, "rating": spec.rating,
                     "weighting": spec.weighting, "factor": factor, "quantity": quantity,
                     "value": value, "tstat": tstat, "T": len(idx), "nw_lags": lags,
                     "first": first, "last": last})

    for f in factors:
        for a in APPROACHES:
            s = series[a][f]
            mu, t_mu = D.nw_mean(s, lags)
            al, t_al = D.capm_alpha(s, mktb, lags)
            add(f, f"{a}.mu", mu * D.PCT, t_mu)
            add(f, f"{a}.alpha", al * D.PCT, t_al)
        for pair, (a, b) in BIAS_PAIRS.items():
            dmu, t_dmu = D.paired_diff_mean(series[a][f], series[b][f], lags)
            dal, t_dal = D.paired_diff_alpha(series[a][f], series[b][f], mktb, lags)
            add(f, f"{pair}.d_mu", dmu * D.PCT, t_dmu)
            add(f, f"{pair}.d_alpha", dal * D.PCT, t_dal)

    return pd.DataFrame(rows)


def validation_stats(spec: LibSpec, factors: tuple[str, ...], *,
                     root: Path | None = None,
                     expected_T: int | None = 268) -> pd.DataFrame:
    """Table 2's decomposition test: r_End ~= LIB + r_Bgn, one row per factor.

    Sources: the unadjusted (End) and adj_return (Bgn) L-S return series, plus the
    portfolio-level LIB characteristic stored in the SAME adj_return CSV -- the
    month-begin set is run with chars=['lib','ilq'] and extract_panel writes the
    long-short spread, sign flip handled exactly as for returns. Values in PERCENT
    except rho.
    """
    f_end = sort_csv("unadjusted", sort=spec.sort, rating=spec.rating,
                       ret_type=spec.ret_type, root=root)
    f_bgn = sort_csv("adj_return", sort=spec.sort, rating=spec.rating,
                       ret_type=spec.ret_type, root=root)
    kw = dict(leg="ls", weighting=spec.weighting, start=spec.start, end=spec.end)
    end = D.load_sort_panel(f_end, **kw)
    bgn = D.load_sort_panel(f_bgn, **kw)
    lib = D.load_sort_panel(f_bgn, value_col="lib", **kw)

    if expected_T is not None:
        D.assert_sample(end.index, expected_T, what=f"{spec.key} End sample")
    lags = D.nw_lags(len(end.index))
    # the realised span, carried on every row -- same reason as in `lib_stats`
    _first, _last = str(end.index.min())[:10], str(end.index.max())[:10]

    rows = []
    for f in factors:
        e, b, l = end[f], bgn[f], lib[f]
        mu_end, _ = D.nw_mean(e, lags)
        mu_bgn, _ = D.nw_mean(b, lags)
        dmu, t_dmu = D.paired_diff_mean(e, b, lags)
        mu_lib, t_lib = D.nw_mean(l, lags)
        aligned = pd.concat([(e - b).rename("d"), l.rename("l")], axis=1).dropna()
        rho = float(aligned["d"].corr(aligned["l"]))
        rows.append({"ret_type": spec.ret_type, "sort": spec.sort,
                     "rating": spec.rating, "weighting": spec.weighting, "factor": f,
                     "mu_end": mu_end * D.PCT, "mu_bgn": mu_bgn * D.PCT,
                     "dmu": dmu * D.PCT, "t_dmu": t_dmu,
                     "mu_lib": mu_lib * D.PCT, "t_mu_lib": t_lib,
                     "rho": rho, "resid": (dmu - mu_lib) * D.PCT,
                     "first": _first, "last": _last,
                     "T": len(end.index), "nw_lags": lags})
    return pd.DataFrame(rows)


def build(specs: list[LibSpec], factors: tuple[str, ...] | None = None,
          *, root: Path | None = None, mktb: pd.Series | None = None,
          expected_T: int | None = 268) -> pd.DataFrame:
    """Run `lib_stats` over several design cells and stack them."""
    m = mktb if mktb is not None else D.load_mktb()
    frames = []
    for spec in specs:
        series = load_series(spec, root=root)
        frames.append(lib_stats(series, m, spec, factors, expected_T=expected_T))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    spec = LibSpec()
    stats = build([spec], D.LIB_FACTORS)
    wide = stats.pivot_table(index="factor", columns="quantity", values="value")
    print(wide.round(2).to_string())
