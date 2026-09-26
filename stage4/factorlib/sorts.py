"""sorts.py -- one return type's panel through PyBondLab, for every rating band and sort.

Single sorts: deciles over all bonds, quintiles within the investment-grade and the
non-investment-grade band. Within-firm sorts: high minus low within each firm (`permno`),
firms with at least two bonds. Holding period one month, value weights from the month
before, rebalanced monthly -- `spec/factors.json` declares all of it.

❗No sign correction. PyBondLab can flip a factor whose full-sample mean is negative, but that
decision moves when the sample grows, so the series are published RAW and the flips it WOULD
make are recorded as data (`flip_set`), keyed per return type.
"""
from __future__ import annotations

import time

import pandas as pd

import _stage4_settings as S

FLIP_KEY = ["factor", "weighting", "return_type"]


def run_one(data: pd.DataFrame, signals: list[str], *, sort: str, band: dict,
            return_type: str) -> pd.DataFrame:
    """One (sort, rating band, return type) cell of the grid, as a long factor panel."""
    from PyBondLab import extract_panel
    from PyBondLab.fast_sorts import fast_single_sorts, fast_within_firm_sorts

    c = S.SPEC["columns"]
    cols = {"ID": c["ID"], "VW": c["VW"], "RATING_NUM": c["RATING_NUM"],
            "ret": S.RETURN_TYPES[return_type]["ret"]}
    t0 = time.perf_counter()
    if sort == "single":
        spec = S.SPEC["sorts"]["single"]
        nport = spec["num_portfolios"][band["label"]]
        res = fast_single_sorts(data, signals, columns=cols, num_portfolios=nport,
                                rating=band["rating"], holding_period=spec["holding_period"],
                                dynamic_weights=S.SPEC["dynamic_weights"],
                                rating_suffix=band["suffix"])
    else:
        spec = S.SPEC["sorts"]["within_firm"]
        res = fast_within_firm_sorts(data, signals, columns=cols, firm_id_col=c["FIRM"],
                                     min_bonds_per_firm=spec["min_bonds_per_firm"],
                                     rating=band["rating"],
                                     dynamic_weights=S.SPEC["dynamic_weights"],
                                     rating_suffix=band["suffix"])

    panel = extract_panel(res)          # no NamingConfig: the series stay unflipped
    panel["rating_type"] = band["label"]
    panel["return_type"] = return_type
    if sort == "single":
        panel["num_portfolios"] = nport
    print(f"     {sort:<11} {band['label']:<4} {return_type:<4} {len(panel):>9,} rows  "
          f"[{time.perf_counter() - t0:.0f}s]", flush=True)
    return panel


def run_bands(data: pd.DataFrame, signals: list[str], *, sort: str,
              return_type: str) -> list[pd.DataFrame]:
    """Every rating band for one sort and return type. The band is PyBondLab's `rating`
    argument, which restricts the FORMATION universe; the panel itself is never subset."""
    return [run_one(data, signals, sort=sort, band=b, return_type=return_type)
            for b in S.SPEC["rating_bands"]]


def assert_one_cell_per_key(ls: pd.DataFrame) -> None:
    """Every flip-set key must name exactly one grid cell per month; otherwise the
    full-sample mean below would average two different series."""
    n = ls.groupby(FLIP_KEY + ["date"]).size()
    bad = n[n > 1]
    if len(bad):
        raise AssertionError(
            f"{len(bad):,} (factor, weighting, return_type, date) cells appear more than "
            f"once, e.g. {bad.head(3).index.tolist()}: two grid cells share one key.")


def flip_set(panel: pd.DataFrame) -> dict:
    """What the full-sample sign rule WOULD choose: True where the long-short mean is
    negative. Per (factor, weighting, return_type); the rating band is in the factor name."""
    ls = panel[panel["leg"] == "ls"]
    assert_one_cell_per_key(ls)
    g = ls.groupby(FLIP_KEY)["return"].mean()
    return {"|".join(k): bool(m < 0) for k, m in g.items()}
