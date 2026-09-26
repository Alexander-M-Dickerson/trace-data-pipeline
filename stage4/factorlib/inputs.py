"""inputs.py -- Stage 2's panel, in the form each return type sorts.

Four return types, from one panel:

  exc    the panel as built, with `ret_vw` made EXCESS of the one-month T-bill (Stage 2's
         factor file, column `rf`). The published leg levels are excess returns; the
         long-short leg would not notice the difference, the long and short legs would.
  dur    duration-adjusted: `ret_vwx = ret_vw - tret`, `str = str - tret` (short-term
  dbns   reversal is a return, so it moves too), and the 68 beta and momentum columns
  dcls   replaced by Stage 2's blocks estimated on that same adjusted return. dbns and dcls
         do the same against `tret_bns` and `tret_cls`.

❗A duration-adjusted return type is a different PANEL, not the same panel with another
return column: sorting `ret_vwx` on betas estimated on `ret_vw` would give plausible
numbers for the wrong question. The swap below is checked in both directions.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import _stage4_settings as S


def missing_inputs(return_types) -> list[str]:
    """Every input the requested return types need that is not on disk, with its maker."""
    need = [(S.PANEL, "python _run_stage2.py (in stage2/)"),
            (S.RISK_FREE, "python _run_stage2.py (in stage2/)")]
    for rt in return_types:
        for b in S.RETURN_TYPES[rt]["blocks"] or []:
            tret = S.RETURN_TYPES[rt]["tret"]
            maker = ("python _run_stage2.py (in stage2/)" if tret == "tret" else
                     f"python make_excess_blocks.py --benchmark {tret.split('_')[1]} (in stage2/)")
            need.append((S.block_path(b), maker))
    return [f"{p}\n      made by: {how}" for p, how in need if not Path(p).exists()]


def swap(df: pd.DataFrame, blocks: list[str], tret_col: str) -> pd.DataFrame:
    """Replace the 68 beta and momentum columns with the blocks' versions; adjust the returns.

    The blocks carry the same 68 names (`b_amd`, not `b_amd_bns`), so the swap is by name.
    Anything dropped and not added back, or added and never dropped, stops the build.
    """
    names = S.SPEC["swap_columns"]["names"]
    absent = [c for c in names if c not in df.columns]
    if absent:
        raise AssertionError(f"the panel lacks {len(absent)} of the 68 columns to swap: {absent}")
    df = df.drop(columns=names)
    added: list[str] = []
    for name in blocks:
        blk = pd.read_parquet(S.block_path(name))
        blk["date"] = pd.to_datetime(blk["date"])
        cols = [c for c in blk.columns if c not in ("cusip", "date")]
        added += cols
        n0 = len(df)
        df = df.merge(blk[["cusip", "date"] + cols], on=["cusip", "date"], how="left")
        if len(df) != n0:
            raise AssertionError(f"{name} changed the row count {n0:,} -> {len(df):,}: it "
                                 "has duplicate (cusip, date) keys")
    if set(names) - set(added):
        raise AssertionError(f"dropped but not added back: {sorted(set(names) - set(added))}")
    if set(added) - set(names):
        raise AssertionError(f"added but never dropped: {sorted(set(added) - set(names))}")

    # At ret_vw's precision, whatever precision the panel stores the Treasury column in.
    t = df[tret_col].astype(df["ret_vw"].dtype)
    df["ret_vwx"] = df["ret_vw"] - t
    df["str"] = df["str"] - t
    return df


def add_risk_free(df: pd.DataFrame, excess: bool) -> pd.DataFrame:
    """Merge the one-month T-bill by date, replacing the panel's own `rfret`; `excess`
    subtracts it from `ret_vw` (the exc return type only)."""
    rf = pd.read_parquet(S.RISK_FREE, columns=["date", S.RISK_FREE_COLUMN])
    rf["date"] = pd.to_datetime(rf["date"])
    rf = rf.rename(columns={S.RISK_FREE_COLUMN: "rfret"})
    if "rfret" in df.columns:
        df = df.drop(columns=["rfret"])
    n0 = len(df)
    df = df.merge(rf, on="date", how="left")
    if len(df) != n0:
        raise AssertionError(f"{S.RISK_FREE.name} has duplicate dates")
    miss = int(df["rfret"].isna().sum())
    if miss:
        raise AssertionError(f"{miss:,} panel rows have no risk-free rate in {S.RISK_FREE}")
    if excess:
        df["ret_vw"] = df["ret_vw"] - df["rfret"]
    return df


def load(return_type: str) -> pd.DataFrame:
    """The panel for one return type, ready to sort."""
    spec = S.RETURN_TYPES[return_type]
    df = pd.read_parquet(S.PANEL)
    df["date"] = pd.to_datetime(df["date"])
    if spec["blocks"]:
        df = swap(df, spec["blocks"], spec["tret"])
    df = add_risk_free(df, excess=spec["tret"] is None)

    # PyBondLab needs a float rating (a nullable integer breaks numba) and one row per
    # bond-month.
    if not pd.api.types.is_float_dtype(df["spc_rat"]):
        df["spc_rat"] = df["spc_rat"].astype("float64")
    dup = int(df.duplicated(["cusip", "date"]).sum())
    if dup:
        raise AssertionError(f"{dup:,} duplicate (cusip, date) rows in {S.PANEL}")
    return df.reset_index(drop=True)
