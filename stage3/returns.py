"""returns.py -- the return each section sorts, in the run's return type.

[ref:rule.return_types] Stage 3 forms its return in six places, each by the convention its
exhibits were built with: the raw return, the return less the factor file's rf, or less the
panel's own `rfret`. In the standard run (`exc`, the default) every function here evaluates
exactly that expression, so the standard exhibits cannot move. In a duration-adjusted run
(STAGE3_RETURNS = dur, dbns or dcls, see return_types.py at the repository root):

  * the return is the bond's return less the type's Treasury column. The month-end and the
    month-begin return both end at the same month-end trade, so both take the same column.
    No risk-free rate is subtracted: the Treasury return already makes it an excess return;
  * the 68 beta and momentum signals are replaced by stage 2's blocks, estimated on that
    same return;
  * the signals that are themselves returns are adjusted the same way: short-term reversal
    `str`, and its unadjusted twin `str_mmn`, which is month t's own return.

Alphas do not change: every section regresses on MKTB, in every run.

Every section calls `signals()` once, then `ret()` or `set_returns()` where it used to
subtract. `tests/test_returns.py` refuses a new site that forms a return without them.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import _stage3_settings as S
import return_types as RT   # at the repository root, which _stage3_settings puts on the path

# the signals that are themselves returns, so they move with the return
RETURN_SIGNALS = ("str", "str_mmn")


def adjusted() -> bool:
    """Is this a duration-adjusted run?"""
    return S.RETURNS != RT.DEFAULT


def benchmark() -> str | None:
    """The Treasury column this run subtracts (None in the standard run)."""
    return RT.benchmark(S.RETURNS)


def load_columns() -> list[str]:
    """The extra panel columns a loader that picks its columns must read: the Treasury column,
    in a duration-adjusted run only, so the standard run reads exactly what it did."""
    return [benchmark()] if adjusted() else []


def signals(df: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    """The frame with the run's return-based signals. The standard run returns it untouched.

    names=None  a full panel: all 68 beta and momentum columns are swapped.
    a list      a loader that read only these signals: only those of them are swapped.
    Either way `str` and `str_mmn`, where present, are adjusted by the Treasury column.
    """
    if not adjusted():
        return df
    df = RT.swap(df, RT.blocks(S.RETURNS), S.block_path, names=names)
    t = RT.treasury(df, benchmark())
    for c in RETURN_SIGNALS:
        if c in df.columns:
            df[c] = df[c] - t
    return df


def ret(df: pd.DataFrame, col: str, rf: str | None = None) -> pd.Series:
    """The return `col` as a section sorts it.

    Standard run: `df[col] - df[rf]` where the section subtracts a risk-free rate, `df[col]`
    where it sorts the raw return -- each exactly the expression it used before.
    Duration-adjusted: `df[col]` less the Treasury column, and never less `rf`.
    """
    if not adjusted():
        return df[col] - df[rf] if rf else df[col]
    return df[col] - RT.treasury(df, benchmark())


def set_returns(df: pd.DataFrame, cols: tuple[str, ...], rf: str | None = None) -> None:
    """In place: each of `cols` becomes the run's return. A raw return in the standard run is
    left exactly as it is."""
    if not adjusted() and not rf:
        return
    for c in cols:
        df[c] = ret(df, c, rf)


def input_paths() -> list[Path]:
    """The block files this run reads, for the manifests' input fingerprints."""
    return [S.block_path(b) for b in RT.blocks(S.RETURNS)]


def caption_note() -> str:
    """The sentence a caption adds in a duration-adjusted run ("" in the standard run)."""
    if not adjusted():
        return ""
    long = RT.RETURN_TYPES[S.RETURNS]["long"]
    return (f"Returns are {long[0].lower()}{long[1:]}. The beta and momentum signals are "
            "estimated on the same return; alphas are on MKTB.")


def manifest() -> dict:
    """What a result's manifest records about the return (nothing in the standard run, so its
    manifests are unchanged)."""
    return {"returns": S.RETURNS} if adjusted() else {}
