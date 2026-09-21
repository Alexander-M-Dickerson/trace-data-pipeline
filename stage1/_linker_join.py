# -*- coding: utf-8 -*-
"""
_linker_join.py
===============
Attach permno / permco / gvkey to the daily panel from the published bond-firm linker.

The linker ships TWO dated windows and they answer different questions:

    w0 / w1   EVIDENCE -- when is the mapping provable. The bond's life intersected with
              the firm's CRSP listing.
    i0 / i1   IDENTITY -- whose bond is this. Contains the evidence window and stays open
              where no dated successor or acquisition contradicts it.

A bond panel LABELS the issuer, so both the daily panel (here) and the monthly panel
(stage2/lib/linker.py) join the IDENTITY window. A bond does not stop being a firm's bond
because the firm's CRSP listing ended.

Until 2026-09 this stage joined the evidence window while Stage 2 joined the identity
window, and the two panels disagreed about the firm on about 2% of bond-months. Measured on
the 2026-09-10 run before the change: moving this join to the identity window gave 644,754
bond-days a firm id they did not have, removed none and changed none, because on every row
of the linker the evidence window lies inside the identity window with the same firm.

This module is pure pandas and imports nothing else from the pipeline, so
tests/test_linker_window.py can exercise it without WRDS.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import pandas as pd

ID_COLS = ("permno", "permco", "gvkey")


def prepare_linker(dfl: pd.DataFrame, window: tuple[str, str]) -> pd.DataFrame:
    """Validate and type the linker. Refuses a file that lacks the declared window.

    A linker without `i0`/`i1` predates the identity window. Falling back to `w0`/`w1`
    would silently bring back the daily/monthly disagreement this module exists to end, so
    it is an error, not a warning.
    """
    lo, hi = window
    dfl = dfl.copy()
    dfl.columns = dfl.columns.str.lower()
    dfl = dfl.rename(columns={"cusip9": "cusip_id"})
    missing = {"cusip_id", lo, hi, "permno"} - set(dfl.columns)
    if missing:
        raise ValueError(
            f"The bond-firm linker is missing column(s) {sorted(missing)}. This stage joins "
            f"the {lo}/{hi} window. A linker without it was built before the identity "
            f"window shipped (2026-09) -- download the current bundle with "
            f"`bash download_inputs.sh`. Columns found: {sorted(dfl.columns)}")

    # confidence / window_src / rung describe HOW each link was earned. They stay in the
    # published file for anyone auditing a link and are not carried into the daily panel.
    keep = ["cusip_id", lo, hi] + [c for c in ID_COLS if c in dfl.columns]
    dfl = dfl[keep].copy()
    for c in (lo, hi):
        dfl[c] = pd.to_datetime(dfl[c], errors="coerce")
    dfl["permno"] = pd.to_numeric(dfl["permno"], errors="coerce").astype("Int64")
    if "permco" in dfl.columns:
        dfl["permco"] = pd.to_numeric(dfl["permco"], errors="coerce").astype("Int64")
    if "gvkey" in dfl.columns:
        # GVKEY ships as a zero-padded string ('013557'). Kept numeric for schema continuity
        # with previous releases -- re-pad to 6 characters before joining to Compustat.
        dfl["gvkey"] = pd.to_numeric(dfl["gvkey"], errors="coerce").astype("Int32")
    dfl["cusip_id"] = dfl["cusip_id"].astype(str)
    dfl = dfl.dropna(subset=["cusip_id", lo, hi])

    dup = int(dfl.duplicated(subset=["cusip_id", lo]).sum())
    if dup:
        raise ValueError(
            f"The linker has {dup} duplicate (cusip_id, {lo}) row(s). Windows must not "
            f"overlap within a bond, or the join below would pick one arbitrarily.")
    return dfl


def attach_firm_ids(panel: pd.DataFrame, dfl: pd.DataFrame,
                    window: tuple[str, str], date_col: str = "trd_exctn_dt"
                    ) -> tuple[pd.DataFrame, int]:
    """Left-join the firm ids onto `panel`, one linker window per bond-day at most.

    `dfl` is the output of `prepare_linker`. Returns (panel_with_ids, n_cleared) where
    n_cleared counts rows whose matched window had already closed.

    A bond with two owners over its life has two rows in the linker. A plain merge on
    cusip_id would fan every trade day of such a bond out to two rows before the window
    could be applied -- millions of transient rows on a small node. merge_asof takes the
    latest window that OPENED at or before the trade date and cannot fan out; rows dated
    after that window CLOSED are then cleared. The windows of one bond do not overlap, so
    this is the same answer as `date BETWEEN lo AND hi`.
    """
    lo, hi = window
    out = panel.copy()
    # cusip_id is a category upstream; merge_asof matches `by` keys by value, and a
    # category against a str silently matches NOTHING -- cast first.
    out["cusip_id"] = out["cusip_id"].astype(str)
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    # merge_asof requires both `on` keys at the SAME datetime resolution and raises
    # MergeError otherwise. The panel comes back from parquet as datetime64[us] and the
    # linker as [ns], so align the linker to the panel rather than assuming either.
    ts_dtype = out[date_col].dtype
    dfl = dfl.copy()
    for c in (lo, hi):
        dfl[c] = dfl[c].astype(ts_dtype)

    before = len(out)
    dfl = dfl.sort_values(lo).reset_index(drop=True)
    out = out.sort_values([date_col]).reset_index(drop=True)
    out = pd.merge_asof(out, dfl, left_on=date_col, right_on=lo, by="cusip_id",
                        direction="backward")
    if len(out) != before:
        raise RuntimeError(
            f"The linker merge changed the row count ({before} -> {len(out)}). There must "
            f"be at most one open window per bond per date.")

    # merge_asof only enforces the window's LEFT edge. Clear the ids where the trade falls
    # after the window closed. A missing link is the intended answer there.
    id_cols = [c for c in ID_COLS if c in out.columns]
    past = out[hi].notna() & (out[date_col] > out[hi])
    n_past = int(past.sum())
    if n_past:
        out.loc[past, id_cols] = pd.NA
    out = out.drop(columns=[lo, hi], errors="ignore")
    return out, n_past
