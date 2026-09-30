"""return_types.py -- what a return type is, and the swap that builds its panel, defined once.

Stage 3 (the exhibits) and stage 4 (the factors) both sort a return, and both can sort it in
four forms. The forms are declared in `stage4/spec/factors.json` (`return_types`), read here:

  exc    the return as built, in excess of the one-month T-bill;
  dur    duration-adjusted: the return minus `tret`, the duration-matched Treasury return;
  dbns   the same, against `tret_bns` (the Treasury return of the bond's own cash flows,
         van Binsbergen, Nozawa and Schwert);
  dcls   the same, against `tret_cls` (Cui, Lu and Song).

A duration-adjusted return type is a different PANEL, not the same panel with another
return column: the 68 beta and momentum signals (`swap_columns`) are
estimated on returns, so they are replaced by stage 2's blocks estimated on that same adjusted
return, and any signal that is itself a return (short-term reversal) is adjusted too. Sorting
an adjusted return on signals estimated on the unadjusted one gives plausible numbers for the
wrong question.

Each stage decides which of its columns are returns; this module gives the swap and the
Treasury series, so the two stages cannot disagree about either.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd

SPEC_PATH = Path(__file__).resolve().parent / "stage4" / "spec" / "factors.json"
_SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

RETURN_TYPES: dict = {k: v for k, v in _SPEC["return_types"].items() if not k.startswith("_")}
SWAP_COLUMNS: list[str] = list(_SPEC["swap_columns"]["names"])
DEFAULT = "exc"


def check(rt: str) -> str:
    """`rt` if it is a declared return type; otherwise stop, naming the ones there are."""
    if rt not in RETURN_TYPES:
        raise SystemExit(f"unknown return type {rt!r}: the declared ones are "
                         f"{', '.join(RETURN_TYPES)} (stage4/spec/factors.json)")
    return rt


def benchmark(rt: str) -> str | None:
    """The Treasury column the type subtracts, or None for `exc`."""
    return RETURN_TYPES[check(rt)]["tret"]


def blocks(rt: str) -> list[str]:
    """The stage 2 block files whose columns replace the 68 signals (none for `exc`)."""
    return list(RETURN_TYPES[check(rt)]["blocks"] or [])


def block_maker(rt: str) -> str:
    """The command that writes the type's blocks: stage 2's own run for `dur`, the
    benchmark blocks for the others."""
    tret = benchmark(rt)
    return ("python _run_stage2.py (in stage2/)" if tret == "tret" else
            f"python make_excess_blocks.py --benchmark {tret.split('_')[1]} (in stage2/)")


def all_blocks() -> list[str]:
    """Every block any return type reads, in declaration order."""
    out: list[str] = []
    for rt in RETURN_TYPES:
        out += [b for b in blocks(rt) if b not in out]
    return out


def treasury(df: pd.DataFrame, col: str) -> pd.Series:
    """The Treasury column at `ret_vw`'s precision, whatever precision the panel stores it in."""
    return df[col].astype(df["ret_vw"].dtype)


# [tag:rule.return_types] a duration-adjusted return type swaps the return-based signals for ones estimated on that return; one definition, stages 3 and 4
def swap(df: pd.DataFrame, block_names: list[str], block_path: Callable[[str], Path],
         names: list[str] | None = None) -> pd.DataFrame:
    """Replace the beta and momentum signals with the blocks' versions.

    The blocks carry the same names as the panel (`b_amd`, not `b_amd_bns`), so the swap is by
    name.

    names=None  all 68: the frame must hold every one, they are dropped, and each block's
                columns are appended in block order. Anything dropped and not added back, or
                added and never dropped, stops the build.
    a list      only those of the 68 the frame holds (a loader that reads a few columns): each
                is replaced in place, so the frame keeps its columns in order.
    """
    if names is None:
        absent = [c for c in SWAP_COLUMNS if c not in df.columns]
        if absent:
            raise AssertionError(f"the panel lacks {len(absent)} of the 68 columns to swap: {absent}")
        df = df.drop(columns=SWAP_COLUMNS)
        added: list[str] = []
        for name in block_names:
            blk = pd.read_parquet(block_path(name))
            blk["date"] = pd.to_datetime(blk["date"])
            cols = [c for c in blk.columns if c not in ("cusip", "date")]
            added += cols
            n0 = len(df)
            df = df.merge(blk[["cusip", "date"] + cols], on=["cusip", "date"], how="left")
            if len(df) != n0:
                raise AssertionError(f"{name} changed the row count {n0:,} -> {len(df):,}: it "
                                     "has duplicate (cusip, date) keys")
        if set(SWAP_COLUMNS) - set(added):
            raise AssertionError(f"dropped but not added back: {sorted(set(SWAP_COLUMNS) - set(added))}")
        if set(added) - set(SWAP_COLUMNS):
            raise AssertionError(f"added but never dropped: {sorted(set(added) - set(SWAP_COLUMNS))}")
        return df

    want = [c for c in dict.fromkeys(names) if c in SWAP_COLUMNS and c in df.columns]
    if not want:
        return df
    import pyarrow.parquet as pq
    found: dict[str, str] = {}
    for name in block_names:
        for c in pq.ParquetFile(block_path(name)).schema.names:
            if c in want:
                found[c] = name
    missing = [c for c in want if c not in found]
    if missing:
        raise AssertionError(f"no block of {block_names} carries {missing}")
    keys = df[["cusip", "date"]]
    for name in block_names:
        cols = [c for c in want if found[c] == name]
        if not cols:
            continue
        blk = pd.read_parquet(block_path(name), columns=["cusip", "date"] + cols)
        blk["date"] = pd.to_datetime(blk["date"]).astype(keys["date"].dtype)
        if blk.duplicated(["cusip", "date"]).any():
            raise AssertionError(f"{name} has duplicate (cusip, date) keys")
        m = keys.merge(blk, on=["cusip", "date"], how="left")
        for c in cols:
            df[c] = m[c].to_numpy()
    return df
