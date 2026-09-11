"""lab_engine.py -- the ONE computation behind every Section-4 (LAB) exhibit.

Table 4, Tables IA.XV/XVI and Figures 7/8/IA.2 all read the same winsorized-vs-base
portfolio return series and ask slightly different questions of them. This module
computes the statistics ONCE into a tidy frame; every exhibit is a formatter.

The series, produced by `run_lab.py`:

    base  = DataUncertaintyAnalysis baseline (ex-ante == ex-post for baseline)
    wins  = the SAME sort with returns winsorized ex post at the 0.50th (left) or
            99.50th (right) percentile of the FULL sample -- the look-ahead
    bias  = wins - base, per (leg, month) -- the paired series

Statistics:

    mu     = series.mean() * 100; t = Newey-West (constant-only HAC), lags =
             int(T**0.25) on the series' OWN dropna T. T varies by signal here
             (257..268), unlike Section 3 where one window covers every series.
    alpha  = HAC intercept on MKTB (aligned by dropna), * 100
    bias   = mean of the PAIRED series. The alpha bias POINT ESTIMATE is the
             difference of the two separately estimated alphas, while its T-STAT
             comes from regressing the paired bias series on MKTB. That asymmetry
             is the paper's and is reproduced deliberately: the point estimate is
             a difference of levels, the test is on the difference series.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # stage3/

import drrlib as D          # noqa: E402
import paths                # noqa: E402

# the 15 LAB signals, in the paper's printed order (asserted vs LAB_*_TAIL_SIGNALS
# in the upstream helpers AND the printed rows of tab:lab_ls_1)
LEFT_SIGNALS = ("b_dunc", "b_dunc3", "b_unc", "ltr48_12", "ltr30_6",
                "ivol_bbw", "ivol_vp", "b_dvix_vp", "b_psb_m", "b_amd_m",
                "var_95", "es_90")
RIGHT_SIGNALS = ("mom3_1", "mom6_1", "mom12_1")
TAIL_SIGNALS = {"left": LEFT_SIGNALS, "right": RIGHT_SIGNALS}

RATINGS = ("All", "IG", "NIG")
LEGS = ("long", "short", "ls")
VARIANTS = ("wins", "base", "bias")

# the nine per-(tail,rating) series frames run_lab.py writes
TS_KEYS = ("ts_long_wins", "ts_long_base", "ts_short_wins", "ts_short_base",
           "ts_ls_wins", "ts_ls_base", "ts_bias_long", "ts_bias_short", "ts_bias_ls")


@dataclass(frozen=True)
class LabSpec:
    """One cell of the LAB design."""
    tail: str                    # 'left' | 'right'
    rating: str = "All"          # 'All' | 'IG' | 'NIG'
    return_type: str = "standard"

    @property
    def key(self) -> str:
        return f"{self.return_type}_{self.rating}_{self.tail}"


def _series_key(leg: str, variant: str) -> str:
    """('ls','bias') -> 'ts_bias_ls'; ('long','wins') -> 'ts_long_wins'."""
    return f"ts_bias_{leg}" if variant == "bias" else f"ts_{leg}_{variant}"


def load_series(root: Path | None = None) -> dict:
    """The LAB series -> {(return_type, rating, tail): {ts_key: DataFrame}}.

    `run_lab.py` writes one parquet per (return_type, rating, tail, ts_key) under
    `root`, named '{return_type}__{rating}__{tail}__{ts_key}.parquet'.
    """
    root = Path(root) if root else paths.section_results("s2_lab") / "series"
    out: dict = {}
    for p in sorted(Path(root).glob("*.parquet")):
        rt, rating, tail, ts_key = p.stem.split("__")
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        out.setdefault((rt, rating, tail), {})[ts_key] = df
    if not out:
        raise FileNotFoundError(f"no LAB series parquets under {root}")
    return out


def load_mktb_lab() -> pd.Series:
    """MKTB over the LAB window (2002-08 .. 2024-12), from Stage 2's factor file."""
    return D.load_mktb(paths.BBW, start=D.SAMPLE_START_LAB, end=D.SAMPLE_END)


def lab_stats(cell: dict, mktb: pd.Series, spec: LabSpec,
              factors: tuple[str, ...] | None = None) -> pd.DataFrame:
    """The tidy stats frame for one (tail, rating, return_type) cell, in PERCENT.

    One row per (factor, leg, variant, stat): value, tstat, T, nw_lags. T is each
    series' own dropna length -- NOT one shared T, because a LAB signal can be
    missing months that its neighbours have.
    """
    factors = tuple(factors) if factors else TAIL_SIGNALS[spec.tail]
    rows = []

    def add(factor, leg, variant, stat, value, tstat, T):
        rows.append({"return_type": spec.return_type, "rating": spec.rating,
                     "tail": spec.tail, "factor": factor, "leg": leg,
                     "variant": variant, "stat": stat, "value": value,
                     "tstat": tstat, "T": T, "nw_lags": D.nw_lags(T) if T else None})

    for f in factors:
        series = {(leg, var): cell[_series_key(leg, var)][f]
                  for leg in LEGS for var in ("wins", "base")}
        for leg in LEGS:
            w, b = series[(leg, "wins")], series[(leg, "base")]
            bias = cell[_series_key(leg, "bias")][f]

            # -- means --------------------------------------------------------
            for var, s in (("wins", w), ("base", b), ("bias", bias)):
                v = s.dropna()
                mu, t_mu = D.nw_mean(v)
                add(f, leg, var, "mu", mu * D.PCT, t_mu, len(v))

            # -- alphas -------------------------------------------------------
            a_w, t_aw = D.capm_alpha(w, mktb)
            a_b, t_ab = D.capm_alpha(b, mktb)
            _, t_abias = D.capm_alpha(bias, mktb)     # paired t; point est is a_w - a_b
            Tw = len(pd.concat([w.rename("y"), mktb.rename("m")], axis=1).dropna())
            Tb = len(pd.concat([b.rename("y"), mktb.rename("m")], axis=1).dropna())
            add(f, leg, "wins", "alpha", a_w * D.PCT, t_aw, Tw)
            add(f, leg, "base", "alpha", a_b * D.PCT, t_ab, Tb)
            add(f, leg, "bias", "alpha", (a_w - a_b) * D.PCT, t_abias, Tw)

    return pd.DataFrame(rows)


def build(source: dict, specs: list[LabSpec], mktb: pd.Series | None = None,
          factors: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Run lab_stats over several design cells and stack them."""
    m = mktb if mktb is not None else load_mktb_lab()
    frames = []
    for spec in specs:
        key = (spec.return_type, spec.rating, spec.tail)
        if key not in source:
            raise KeyError(f"cell {key} absent from the series source")
        frames.append(lab_stats(source[key], m, spec, factors))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    src = load_series()
    stats = build(src, [LabSpec(tail="left"), LabSpec(tail="right")])
    ls = stats[(stats["leg"] == "ls")]
    print(ls.pivot_table(index="factor", columns=["variant", "stat"], values="value")
            .round(2).to_string())
