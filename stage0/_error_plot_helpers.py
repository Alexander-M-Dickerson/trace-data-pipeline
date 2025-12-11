# -*- coding: utf-8 -*-
"""
error_filters.py
========================
Shared plotting utilities and error-filter functions for TRACE panels.

Author : Alex Dickerson
Created: 2025-10-20
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, List, Iterable, Union

import numpy as np
import pandas as pd
import matplotlib as mpl
import os
if not os.environ.get("DISPLAY"):
    os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import AutoDateLocator, DateFormatter
from pathlib import Path
import logging

# ---------------------------------------------------------------------
# Plotting params
# ---------------------------------------------------------------------
@dataclass
class PlotParams:
    # Fonts & LaTeX
    use_latex: bool = True
    base_font: int = 9
    title_size: int = 10
    label_size: int = 9
    tick_size: int = 8
    legend_size: int = 8
    figure_dpi: int = 300
    
    export_format: str = "pdf"      # "pdf" | "png" | "jpg"
    jpeg_quality: int = 85          # only used when export_format is jpg
    transparent: bool = False       # force opaque output for speed

    # Orientation: "auto" | "landscape" | "portrait"
    orientation: str = "auto"

    # Overall title (optional)
    suptitle: str = ""

    # X spacing mode: "time" (true datetimes) or "rank" (0..n-1)
    x_spacing: str = "time"

    # Colors / lines
    all_color: str = "orange"
    all_alpha: float = 0.7
    all_lw: float = 1.0

    filtered_color: str = "0.05"    
    filtered_alpha: float = 1.0
    filtered_lw: float = 1.25

    # Flagged markers
    show_flagged: bool = True
    flagged_marker: str = "o"
    flagged_size: float = 14.0
    flagged_edgecolor: str = "red"
    flagged_facecolor: str = "none"
    flagged_linewidth: float = 0.9

    # Grid & legend
    grid_alpha: float = 0.25
    grid_lw: float = 0.6
    legend_loc: str = "upper left"


def _apply_rcparams(p: PlotParams) -> None:
    """Apply LaTeX + font sizes via rcParams, with safe fallback."""
    try:
        mpl.rcParams.update({
            "text.usetex": bool(p.use_latex),
            "font.family": "serif",
            "font.size": p.base_font,
            "axes.titlesize": p.title_size,
            "axes.labelsize": p.label_size,
            "xtick.labelsize": p.tick_size,
            "ytick.labelsize": p.tick_size,
            "legend.fontsize": p.legend_size,
            "figure.dpi": p.figure_dpi,
        })
    except Exception:
        mpl.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.size": p.base_font,
            "figure.dpi": p.figure_dpi,
        })


def _format_time_date_axis(ax, tick_size: int) -> None:
    locator = AutoDateLocator(minticks=3, maxticks=8, interval_multiples=True)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelsize=tick_size, pad=1)
    ax.tick_params(axis="y", labelsize=tick_size)
    ax.margins(x=0.01)


def _format_rank_date_axis(ax, dates: pd.Series, tick_size: int) -> None:
    """For 'rank' spacing: ticks at integer positions, labels are formatted dates."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, prune=None))
    ticks = [int(t) for t in ax.get_xticks() if 0 <= int(t) < len(dates)]
    ax.set_xticks(ticks)

    dts = pd.to_datetime(dates, errors="coerce")
    labels = dts.iloc[ticks].dt.strftime("%Y-%m")
    ax.set_xticklabels(labels)

    ax.tick_params(axis="x", labelsize=tick_size, pad=1)
    ax.tick_params(axis="y", labelsize=tick_size)
    ax.margins(x=0.01)


def _plot_panel(ax,
                dfID: pd.DataFrame,
                cusip: str,
                p: PlotParams,
                *,
                show_legend: bool = True,
                error_type: str = "bounce_back") -> None:
    """
    One CUSIP panel.

    Parameters
    ----------
    ax : matplotlib Axes
    dfID : DataFrame filtered to a single cusip_id
        Must contain at least:
          - Common: 'trd_exctn_dt', 'rptd_pr'
          - For bounce_back: 'filtered_error' (0 kept, 1 eliminated)
          - For decimal_shift: 'dec_shift_flag' (0/1), 'suggested_price'
          - For init_price: 'initial_error_flag' (0 kept, 1 eliminated)
    cusip : str
    p : PlotParams
    error_type : {'bounce_back','decimal_shift','init_price'}
        Controls which columns are used for the "Filtered" series and flags.
    """
    
    d = dfID if dfID.index.is_monotonic_increasing else dfID.sort_values("trd_exctn_dt", ignore_index=True)
    
    dates = d["trd_exctn_dt"]
    y_all = d["rptd_pr"].astype(float)

    # Branch by error type
    if error_type == "bounce_back":
        # "Filtered" line is the kept trades (filtered_error == 0)
        kept_mask = d["filtered_error"].fillna(0).astype(int) == 0
        kept = d.loc[kept_mask]
        y_filtered = kept["rptd_pr"].astype(float)

        flagged_mask = d["filtered_error"].fillna(0).astype(int) == 1
        flagged = d.loc[flagged_mask]
        y_flagged = flagged["rptd_pr"].astype(float)

        if p.x_spacing == "rank":
            pos = np.arange(len(d))
            km  = kept_mask.to_numpy()
            fm  = flagged_mask.to_numpy()
            x_all  = pos
            x_filt = pos[km]
            x_flag = pos[fm]
        else:
            x_all  = dates
            x_filt = kept["trd_exctn_dt"]
            x_flag = flagged["trd_exctn_dt"]

        legend_flag_label = "Eliminated"

    elif error_type == "decimal_shift":
        # "Filtered" line is the corrected/suggested price series
        # Flags show original uncorrected trades where dec_shift_flag == 1
        if "suggested_price" not in d.columns or "dec_shift_flag" not in d.columns:
            ax.text(0.5, 0.5,
                    f"Missing columns for decimal_shift\n{cusip}",
                    ha="center", va="center")
            ax.axis("off")
            return

        y_filtered = d["suggested_price"].astype(float)

        flagged_mask = d["dec_shift_flag"].fillna(0).astype(int) == 1
        flagged = d.loc[flagged_mask]
        y_flagged = flagged["rptd_pr"].astype(float)

        if p.x_spacing == "rank":
            pos = np.arange(len(d))
            fm  = flagged_mask.to_numpy()
            x_all  = pos
            x_filt = pos                # suggested_price defined for each row
            x_flag = pos[fm]
        else:
            x_all  = dates
            x_filt = dates
            x_flag = flagged["trd_exctn_dt"]


        legend_flag_label = "Corrected"

    elif error_type == "init_price":
        # "Filtered" line is the kept trades (initial_error_flag == 0)
        # Flags show eliminated trades where initial_error_flag == 1
        if "initial_error_flag" not in d.columns:
            ax.text(0.5, 0.5,
                    f"Missing initial_error_flag for init_price\n{cusip}",
                    ha="center", va="center")
            ax.axis("off")
            return

        kept_mask = d["initial_error_flag"].fillna(0).astype(int) == 0
        kept = d.loc[kept_mask]
        y_filtered = kept["rptd_pr"].astype(float)

        flagged_mask = d["initial_error_flag"].fillna(0).astype(int) == 1
        flagged = d.loc[flagged_mask]
        y_flagged = flagged["rptd_pr"].astype(float)

        if p.x_spacing == "rank":
            pos = np.arange(len(d))
            km  = kept_mask.to_numpy()
            fm  = flagged_mask.to_numpy()
            x_all  = pos
            x_filt = pos[km]
            x_flag = pos[fm]
        else:
            x_all  = dates
            x_filt = kept["trd_exctn_dt"]
            x_flag = flagged["trd_exctn_dt"]

        legend_flag_label = "Eliminated"

    else:
        ax.text(0.5, 0.5, f"Unknown error_type: {error_type}", ha="center", va="center")
        ax.axis("off")
        return

    # Plot: All vs Filtered
    ax.plot(x_all,  y_all,
            color=p.all_color, alpha=p.all_alpha, lw=p.all_lw, label="All")
    ax.plot(x_filt, y_filtered,
            color=p.filtered_color, alpha=p.filtered_alpha, lw=p.filtered_lw, label="Filtered")

    # Flagged markers
    if p.show_flagged and not flagged.empty:
        ax.scatter(
            x_flag, y_flagged,
            s=p.flagged_size,
            marker=p.flagged_marker,
            facecolors=p.flagged_facecolor,
            edgecolors=p.flagged_edgecolor,
            linewidths=p.flagged_linewidth,
            zorder=3,
            label=legend_flag_label,
            rasterized=True
        )

    ax.set_title(f"{cusip}", pad=2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, alpha=p.grid_alpha, linewidth=p.grid_lw)
    if show_legend and p.legend_loc:
        ax.legend(frameon=False, ncols=3, handlelength=2.5, borderaxespad=0.2, loc=p.legend_loc)


    if p.x_spacing == "rank":
        _format_rank_date_axis(ax, dates, p.tick_size)
    else:
        _format_time_date_axis(ax, p.tick_size)

def _choose_orientation(rows: int, p: PlotParams) -> str:
    if p.orientation.lower() in {"landscape", "portrait"}:
        return p.orientation.lower()
    return "portrait" if rows >= 4 else "landscape"


def _a4_figsize(orientation: str) -> Tuple[float, float]:
    # A4 inches: 8.27 x 11.69
    return (8.27, 11.69) if orientation == "portrait" else (11.69, 8.27)


def _grid_margins(rows: int, cols: int) -> dict:
    if rows == 2 and cols == 2:
        return dict(left=0.06, right=0.995, bottom=0.065, top=0.93, wspace=0.12, hspace=0.26)
    if rows == 3 and cols == 2:
        return dict(left=0.06, right=0.995, bottom=0.055, top=0.93, wspace=0.12, hspace=0.18)
    if rows == 4 and cols == 2:
        return dict(left=0.07, right=0.995, bottom=0.045, top=0.93, wspace=0.12, hspace=0.14)
    return dict(left=0.06, right=0.995, bottom=0.06, top=0.93, wspace=0.12, hspace=0.20)
# -------------------------------------------------------------------------
def make_panel(df_out: pd.DataFrame,
               error_cusips: List[str],
               subplot_dim: Tuple[int, int] = (2, 2),
               export_dir: pd.Path = None,
               filename_stub: str | None = None,
               params: PlotParams | None = None,
               idx_map=None,
               *,
               error_type: str = "bounce_back"):
    """
    Create an A4 figure with (rows x cols) subplots (one CUSIP per panel) and save to PNG.

    Parameters
    ----------
    df_out : DataFrame
        For error_type='bounce_back', must contain:
          ['cusip_id','trd_exctn_dt','rptd_pr','filtered_error'].
        For error_type='decimal_shift', must contain:
          ['cusip_id','trd_exctn_dt','rptd_pr','dec_shift_flag','suggested_price'].
        For error_type='init_price', must contain:
          ['cusip_id','trd_exctn_dt','rptd_pr','initial_error_flag'].
    error_cusips : list[str]
    subplot_dim : (rows, cols)
    export_dir : path-like
    filename_stub : optional custom filename root (suffix is auto-appended)
    params : PlotParams
    error_type : {'bounce_back','decimal_shift','init_price'}
        Controls plotting logic and filename suffix ('_bb', '_ds', or '_ie').

    Returns
    -------
    (png_path) : tuple[pathlib.Path, pathlib.Path]
    """
    if export_dir is None:
        from pathlib import Path
        export_dir = Path(".")

    rows, cols = subplot_dim
    n_panels = rows * cols

    p = params or PlotParams()
    export_dir.mkdir(parents=True, exist_ok=True)
    _apply_rcparams(p)

    orientation = _choose_orientation(rows, p)
    fig_w, fig_h = _a4_figsize(orientation)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs_kwargs = _grid_margins(rows, cols)
    gs = fig.add_gridspec(nrows=rows, ncols=cols, **gs_kwargs)
    axes = gs.subplots().ravel()

    # Column subsets by mode (kept minimal to what _plot_panel needs)
    if error_type == "bounce_back":
        needed = ["cusip_id", "trd_exctn_dt", "rptd_pr", "filtered_error"]
    elif error_type == "decimal_shift":
        needed = ["cusip_id", "trd_exctn_dt", "rptd_pr", "dec_shift_flag", "suggested_price"]
    elif error_type == "init_price":
        needed = ["cusip_id", "trd_exctn_dt", "rptd_pr", "initial_error_flag"]
    else:
        raise ValueError(f"Unknown error_type: {error_type}")

    missing = [c for c in needed if c not in df_out.columns]
    if missing:
        raise KeyError(f"df_out missing required columns for {error_type}: {missing}")

    # Draw each CUSIP panel
    # ----------------------------------------------------------- #
    page_cusips = list(error_cusips[:n_panels])
    
    if len(page_cusips) == 0:
        page_groups = {}
    else:
        if idx_map is not None:
            # Gather row indices for this page only (skips any missing cusips)
            idx_lists = [idx_map.get(c, None) for c in page_cusips]
            idx_lists = [ix for ix in idx_lists if ix is not None and len(ix)]
            if len(idx_lists):
                import numpy as np
                take_idx = np.concatenate(idx_lists)
                # One take, then column-narrow
                df_sub = df_out.take(take_idx)[needed]
                # Preserve page order; build groups without an expensive groupby over the whole df
                # (We do small groupbys only on df_sub filtered to the page)
                df_sub = df_sub[df_sub["cusip_id"].isin(page_cusips)]
                gb = df_sub.groupby("cusip_id", sort=False, observed=True)
                page_groups = {k: v for k, v in gb}
            else:
                page_groups = {}
        else:
            # Fallback: old categorical slice (used only if caller didn't pass idx_map)
            cats = pd.Categorical(df_out["cusip_id"], categories=page_cusips, ordered=False)
            mask = cats.notna()
            df_sub = df_out.loc[mask, needed].copy()
            df_sub["cusip_id"] = cats[mask]
            if not pd.api.types.is_datetime64_any_dtype(df_sub["trd_exctn_dt"].dtype):
                df_sub["trd_exctn_dt"] = pd.to_datetime(df_sub["trd_exctn_dt"], errors="coerce")
            gb = df_sub.groupby("cusip_id", sort=False, observed=True)
            page_groups = {k: v for k, v in gb}
    
    # Draw each CUSIP panel using pre-sliced groups
    for i, (ax, cusip) in enumerate(zip(axes, page_cusips)):
        dfi = page_groups.get(cusip)
        if dfi is None or dfi.empty:
            ax.text(0.5, 0.5, f"No data for {cusip}", ha="center", va="center")
            ax.axis("off")
            continue
        _plot_panel(ax, dfi, cusip, p, show_legend=(i == 0), error_type=error_type)
    # ----------------------------------------------------------- #
    # Fill any remaining empty slots
    if len(error_cusips) < n_panels:
        for ax in axes[len(error_cusips):]:
            ax.text(0.5, 0.5, "No CUSIP provided", ha="center", va="center")
            ax.axis("off")

    if p.suptitle:
        fig.suptitle(p.suptitle, y=0.97)

    from pathlib import Path
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows_cols = f"{rows}x{cols}"
    stub_base = filename_stub or f"prices_{rows_cols}_{ts}"
    suffix_map = {"bounce_back": "_bb", "decimal_shift": "_ds", "init_price": "_ie"}
    suffix = suffix_map.get(error_type, "_bb")
    stub = f"{stub_base}{suffix}"

    # ----- choose format + build path -----
    fmt = (params.export_format if params else "pdf").lower()
    ext = "pdf" if fmt == "pdf" else ("jpg" if fmt in ("jpg", "jpeg") else "png")
    out_path = Path(export_dir) / f"{stub}.{ext}"
    
    savefig_kwargs = dict(
        format=ext,
        bbox_inches=None,        # faster than "tight" at save time
        facecolor="white",
        edgecolor="none",
        transparent=(params.transparent if params else False),
    )
    
    # Raster outputs benefit from lower dpi; PDF ignores dpi for vectors
    if ext in ("png", "jpg", "jpeg"):
        savefig_kwargs["dpi"] = (params.figure_dpi if params else 150)
    
    # Pillow options for smaller files
    if ext in ("png",):
        savefig_kwargs["pil_kwargs"] = {"optimize": True, "compress_level": 6}
    elif ext in ("jpg", "jpeg"):
        q = max(60, min(95, getattr(params, "jpeg_quality", 85)))
        savefig_kwargs["pil_kwargs"] = {"quality": q, "optimize": True, "progressive": True}
    
    fig.savefig(out_path, **savefig_kwargs)
    plt.close(fig)
    return out_path

# -------------------------------------------------------------------------
def _ordered_aggregate(df: pd.DataFrame, stage_col: str = "stage", chunk_col: str = "chunk"):
    """
    Preserve global stage order based on within-chunk sequence, then aggregate.
    Expects columns: stage, chunk, rows_before, rows_after, removed
    """
    df = df.copy()
    df["_seq"] = df.groupby(chunk_col).cumcount()

    # Canonical (stable) order: median position across chunks
    order = (
        df.groupby(stage_col)["_seq"]
          .median()
          .sort_values(kind="mergesort")
          .index
          .tolist()
    )

    agg = (
        df.groupby(stage_col, sort=False)
          .agg(
              Npre=("rows_before", "sum"),
              Npost=("rows_after", "sum"),
              Removed=("removed", "sum"),  # raw count from logs
          )
          .rename_axis(None)
    )

    return agg.reindex(order)
# -------------------------------------------------------------------------
def build_filter_summary(
    dr: pd.DataFrame,
    *,
    decimal_stage: str = "decimal_shift",
) -> pd.DataFrame:
    """
    Filters summary with an appended 'overall' row.

    Columns:
        Npre, Npost, Removed/Corrected, %removed_start

    - Excludes 'start' from the table body.
    - Treats `decimal_stage` as corrections (not deletions), merges into
      'Removed/Corrected'.
    - %removed_start = (Removed/Corrected) / total_start, but note that the
      'overall' row uses (total_start - final_Npost) / total_start.
    - Uses a dense 'true_chunk' to handle non-contiguous chunk ids.
    """
    # ---- global start ----
    total_start = int(dr.loc[dr["stage"] == "start", "rows_before"].sum())
    
    if total_start == 0:
        total_start = int(dr.loc[dr["stage"] == "dick_nielsen_filter",
                                 "rows_before"].sum())

    # ---- only actual filters ----
    drf = dr.loc[dr["stage"] != "start"].copy()

    # ---- contiguous chunk ids to avoid gaps ----
    codes, uniques = pd.factorize(drf["chunk"], sort=True)  # NaNs -> -1

    # Try nullable Int64 if available; else fall back to numpy int64
    try:
        int_dtype = pd.Int64Dtype()  # may not exist on older pandas
        drf["true_chunk"] = pd.Series(codes + 1, index=drf.index, dtype=int_dtype)
        # preserve NA semantics only when nullable dtype is supported
        drf.loc[drf["chunk"].isna(), "true_chunk"] = pd.NA
    except AttributeError:
        # older pandas: keep dense ints; NaNs stayed as code -1 -> 0 after +1
        drf["true_chunk"] = pd.Series(codes + 1, index=drf.index).astype(np.int64)

    # ---- canonical order via median step index (by true_chunk) ----
    drf["_seq"] = drf.groupby("true_chunk", dropna=False).cumcount()
    order = (
        drf.groupby("stage", dropna=False)["_seq"].median()
           .sort_values(kind="mergesort")
           .index.tolist()
    )

    # ---- aggregate across chunks ----
    agg = (
        drf.groupby("stage", sort=False)
           .agg(Npre=("rows_before", "sum"),
                Npost=("rows_after",  "sum"),
                Removed=("removed",    "sum"))
           .reindex(order)
    )

    # ---- decimal -> corrections; unify as Removed/Corrected ----
    is_decimal = (agg.index == decimal_stage)
    removed = agg["Removed"].astype("int64").copy()
    corrected = pd.Series(0, index=agg.index, dtype="int64")
    corrected[is_decimal] = removed[is_decimal]
    removed[is_decimal] = 0

    agg["Removed/Corrected"] = (removed + corrected).astype("int64")

    # ---- % of global start (per stage) ----
    agg["%removed_start"] = np.where(
        total_start > 0,
        100.0 * agg["Removed/Corrected"] / total_start,
        np.nan
    ).round(3)

    # ---- keep only requested columns ----
    out = agg.loc[:, ["Npre", "Npost", "Removed/Corrected", "%removed_start"]].copy()
    out["Npre"]  = out["Npre"].astype("int64")
    out["Npost"] = out["Npost"].astype("int64")

    # ---- append 'overall' row ----
    final_npost = int(out["Npost"].iloc[-1]) if len(out) else total_start
    overall_pct_start = (
        round(100.0 * (total_start - final_npost) / total_start, 3)
        if total_start > 0 else np.nan
    )

    overall = pd.DataFrame(
        {
            "Npre": [total_start],
            "Npost": [final_npost],
            "Removed/Corrected": [int(out["Removed/Corrected"].sum())],
            "%removed_start": [overall_pct_start],
        },
        index=["overall"],
    )

    out = pd.concat([out, overall], axis=0)
    out.index.name = "Filter"
    out.attrs["total_start"] = total_start
    return out
# -------------------------------------------------------------------------
def build_dn_summary(dn: pd.DataFrame, *, total_start: int) -> pd.DataFrame:
    """
    Sequential Dick-Nielsen summary with an appended 'overall' row.

    Output columns:
        Npre, Npost, Removed, %_start
    Index:
        'Filter' (the DN step)
    """
    dn = dn.copy()

    # Establish DN step order via within-chunk sequence
    dn["_seq"] = dn.groupby("chunk").cumcount()
    step_order = (
        dn.groupby("stage")["_seq"]
          .median()
          .sort_values(kind="mergesort")
          .index
          .tolist()
    )

    # Total removed per step across chunks
    removed_by_step = (
        dn.groupby("stage", sort=False)["removed"]
          .sum()
          .reindex(step_order)
          .fillna(0)
          .astype("int64")
    )

    # Walk sequentially
    rows = []
    npre = int(total_start)
    for step in step_order:
        removed = int(max(0, removed_by_step.loc[step]))
        npost = max(0, npre - removed)
        pct_start = (100.0 * removed / total_start) if total_start > 0 else np.nan
        rows.append((step, npre, npost, removed, round(pct_start, 3)))
        npre = npost

    out = pd.DataFrame(
        rows, columns=["Filter", "Npre", "Npost", "Removed", "%_start"]
    ).set_index("Filter")

    # Append 'overall' row
    overall_removed = int(out["Removed"].sum())
    overall_npost = int(out["Npost"].iloc[-1]) if len(out) else int(total_start)
    overall_pct_start = (
        round(100.0 * (total_start - overall_npost) / total_start, 3)
        if total_start > 0 else np.nan
    )

    overall = pd.DataFrame(
        {
            "Npre": [int(total_start)],
            "Npost": [overall_npost],
            "Removed": [overall_removed],
            "%_start": [overall_pct_start],
        },
        index=["overall"],
    )

    out = pd.concat([out, overall], axis=0)

    out.index.name = "Filter"

    out.attrs["total_start"] = int(total_start)
    return out
# -------------------------------------------------------------------------
def filters_df_to_summary(df: pd.DataFrame, start_label: str = "start") -> pd.DataFrame:
    d = df.copy()

    # Determine the starting total from the 'start' row's rows_before
    if (d["stage"] == start_label).any():
        total_start = int(d.loc[d["stage"] == start_label, "rows_before"].iloc[0])
    else:
        # Fallback: first row's rows_before
        total_start = int(d["rows_before"].iloc[0])

    # Build output
    out = (
        d.loc[:, ["stage", "rows_before", "rows_after", "removed"]]
         .rename(columns={
             "stage": "Filter",
             "rows_before": "Npre",
             "rows_after": "Npost",
             "removed": "Removed",
         })
    )

    # Ensure integer counts
    for c in ["Npre", "Npost", "Removed"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int64")

    # % of starting sample
    if total_start > 0:
        out["%_start"] = (100.0 * out["Removed"] / total_start).round(3)
    else:
        out["%_start"] = np.nan

    # Set index to Filter and name it explicitly
    out = out.set_index("Filter")
    out.index.name = "Filter"

    return out
# -------------------------------------------------------------------------
# Latex #
# -------------------------------------------------------------------------
def _rows_to_latex_generic(df):
    """Assumes df cells are preformatted for LaTeX. Emits 'c1 & c2 & ... \\\\' per row."""
    return "\n".join(" & ".join(map(str, row)) + r" \\"
                     for row in df.to_numpy())

def _header_to_latex(df):
    """Build the header line from df.columns (already LaTeX-ready)."""
    return " & ".join(map(str, df.columns)) + r" \\"

def _rows_to_latex(df):
    """
    Convert a 2-col DataFrame to LaTeX rows: 'key & value \\\\' per row.
    Assumes keys/values are already preformatted (e.g., \\texttt{...}).
    """
    return "\n".join(f"{k} & {v} \\\\" for k, v in df.iloc[:, :2].to_numpy())


def _rows_to_latex_wrapped(df):
    """
    Convert a 2-col DataFrame to LaTeX rows with wrapping on the Value column.
    Assumes the df already contains LaTeX (e.g., \\texttt{...}).
    """
    out_lines = []
    for key, val in df.iloc[:, :2].to_numpy():
        out_lines.append(f"{key} & {val} \\\\")
    return "\n".join(out_lines)


def make_parameters_table(
    filters_df,
    ds_df,
    bb_df,
    ie_df=None,
    caption="Error-Correction Filters and Module Parameters",
    label="tab:parameters",
    note=(
        "This table provides user-defined inputs related to the error correction "
        "functionality designed by \\citet*{DickersonRobottiRossetti_2024}. "
        "Panel A lists overall options for correcting the TRACE transaction data for "
        "potential errors, "
        "Panel B lists decimal-shift corrector settings, Panel C the bounce-back filter settings, "
        "and Panel D the initial price error filter settings, "
        "all described in \\citet*{DickersonRobottiRossetti_2024}. "
        "Additional information can be found on the "
        "\\href{https://github.com/Alexander-M-Dickerson/trace-data-pipeline/tree/main/stage0}{\\texttt{trace-data-pipeline}} GitHub repository."
    ),

):
    panel_a = _rows_to_latex(filters_df)
    panel_b = _rows_to_latex(ds_df)
    panel_c = _rows_to_latex(bb_df)
    panel_d = _rows_to_latex(ie_df) if ie_df is not None and len(ie_df) > 0 else ""

    # Build Panel D section if we have init_error params
    panel_d_section = ""
    if panel_d:
        panel_d_section = r"""
            \midrule
            \multicolumn{2}{c}{\textbf{Panel D: Initial Price Error Parameters}} \\
            \midrule
            """ + panel_d

    latex = r"""
            \begin{table}[!ht]
            \begin{center}
            \footnotesize
            \caption{""" + caption + r"""}
            \label{""" + label + r"""}\vspace{2mm}
            \begin{tabular}{lc}
            \midrule
            Parameter & Value \\
            \midrule
            \multicolumn{2}{c}{\textbf{Panel A: Error-Correction and Filtering Toggles}} \\
            \midrule
            """ + panel_a + r"""
            \midrule
            \multicolumn{2}{c}{\textbf{Panel B: Decimal-Shift Parameters}} \\
            \midrule
            """ + panel_b + r"""
            \midrule
            \multicolumn{2}{c}{\textbf{Panel C: Bounce-Back Parameters}} \\
            \midrule
            """ + panel_c + panel_d_section + r"""
            \bottomrule
            \end{tabular}
            \end{center}
            \begin{spacing}{1}
            \footnotesize{
            """ + note + r"""
            }
            \end{spacing}
            \vspace{-2mm}
            \end{table}
            """.strip()

    return latex

# FISD NOTE
fisd_note = (
    "This table summarises the FISD universe-construction filters and their values. "
    "These values are set in the \\texttt{\\_trace\\_settings.py} Python script in the "
    "\\texttt{FISD\\_PARAMS} dictionary input. "
    "Additional information can be found on the "
    "\\href{https://github.com/Alexander-M-Dickerson/trace-data-pipeline/tree/main/stage0}{\\texttt{trace-data-pipeline}} GitHub repository."
)

def make_fisd_table(
    fi_df,
    caption="FISD Parameter Settings",
    label="tab:fisd_params",
    note=fisd_note
):
    # Use a paragraph column for Value to allow wrapping
    latex = r"""
                \begin{table}[!ht]
                \begin{center}
                \footnotesize
                \caption{""" + caption + r"""}
                \label{""" + label + r"""}\vspace{2mm}
                \begin{tabular}{l p{0.62\textwidth}}
                \midrule
                FISD Parameters & Value \\
                \midrule
                """ + _rows_to_latex_wrapped(fi_df) + r"""
                \bottomrule
                \end{tabular}
                \end{center}
                \begin{spacing}{1}
                {\footnotesize """ + note + r"""}
                \end{spacing}
                \vspace{-2mm}
                \end{table}
                """.strip()
    return latex


def make_filters_counts_table(
    df_fmt,   
    dr_fmt,   
    dn_fmt,   
    caption="TRACE Transaction-Level Filter Records",
    label="tab:filters_counts",
):
    # All three should share the same columns/ordering
    ncols   = df_fmt.shape[1]
    colspec = "l" + "r" * (ncols - 1)

    header  = _header_to_latex(df_fmt)  # use Panel A header (identical across)
    rows_A  = _rows_to_latex_generic(df_fmt)
    rows_B  = _rows_to_latex_generic(dr_fmt)
    header_B = _header_to_latex(dr_fmt)
    rows_C  = _rows_to_latex_generic(dn_fmt)
    header_C = _header_to_latex(dn_fmt)

    note_block = r"""
    \begin{spacing}{1}
    {\footnotesize 
    This table presents the sequential application of data filters to the TRACE enhanced dataset. 
    Panel A shows the FISD corporate bond universe-construction filters. 
    Panel B shows filters from \citet*{DickersonRobottiRossetti_2024}, DRR. 
    Panel C shows cleaning steps from \citet*{van2025duration}, BNS and \citet{dick2009liquidity,dick2014clean}, DN,
    which breaks down the detailed steps in the \texttt{dick\_nielsen\_filter} filter in the first row
    of Panel B.
    The filters \texttt{pre\_settle\_<=2d} to \texttt{pre\_exclude\_special\_cond} are from BNS, the remainder are from DN.
    N$_{pre}$ and N$_{post}$ indicate row counts before and after each filter.
    \% Removed is always recorded as the number of transactions removed divided by the total number
    of transactions.
    Additional information can be found on the 
    \href{https://github.com/Alexander-M-Dickerson/trace-data-pipeline/tree/main/stage0}{\texttt{trace-data-pipeline}} GitHub repository.
    }
    \end{spacing}
    """.strip()


    latex = r"""
            \begin{table}[!ht]
            \begin{center}
            \footnotesize
            \caption{""" + caption + r"""}
            \label{""" + label + r"""}\vspace{2mm}
            \begin{tabular}{""" + colspec + r"""}
            \midrule
            """ + header + r"""
            \midrule
            \multicolumn{""" + str(ncols) + r"""}{c}{\textbf{Panel A: FISD Universe Construction Filters}} \\[2pt]
            \midrule
            """ + rows_A + r"""
            \midrule
            \multicolumn{""" + str(ncols) + r"""}{c}{\textbf{Panel B: Dickerson--Rossetti--Robotti Filters}} \\[2pt]
            \midrule
            """ + header_B + r"""
            \midrule
            """ + rows_B + r"""
            \midrule
            \multicolumn{""" + str(ncols) + r"""}{c}{\textbf{Panel C: van Binsbergen et al. and Dick--Nielsen Cleaning Steps}} \\[2pt]
            \midrule
            """ + header_C + r"""
            \midrule
            """ + rows_C + r"""
            \bottomrule
            \end{tabular}
            \end{center}
            """ + note_block + r"""
            \vspace{-2mm}
            \end{table}
            """.strip()

    return latex

def _escape_filter_value(s: str) -> str:
    # Escape underscores and wrap in \texttt{...}
    return r"\texttt{" + str(s).replace("_", r"\_") + "}"

def _format_table(df: pd.DataFrame, percent_col: str) -> pd.DataFrame:
    df_fmt = df.copy().reset_index().rename(columns={"index": "Filter"})
    # Standardize headers
    col_map = {
        "Npre": "N$_{pre}$",
        "Npost": "N$_{post}$",
        percent_col: r"\% Removed",
    }
    # Numeric formatting
    if "Npre" in df_fmt.columns:
        df_fmt["Npre"] = df_fmt["Npre"].map(lambda x: f"{int(x):,}")
    if "Npost" in df_fmt.columns:
        df_fmt["Npost"] = df_fmt["Npost"].map(lambda x: f"{int(x):,}")
    rem_col = "Removed/Corrected" if "Removed/Corrected" in df_fmt.columns else "Removed"
    if rem_col in df_fmt.columns:
        df_fmt[rem_col] = df_fmt[rem_col].map(lambda x: f"{int(x):,}")
    df_fmt[percent_col] = df_fmt[percent_col].map(lambda x: f"{x:.3f}")
    # Escape filter names and wrap in \texttt{}
    df_fmt["Filter"] = df_fmt["Filter"].map(_escape_filter_value)
    # Rename headers for LaTeX
    df_fmt = df_fmt.rename(columns=col_map)
    return df_fmt

def _escape_header(h: str) -> str:
    """
    Escape function
    """
    if r"\%" in h:
        return h  # already properly escaped (prevents \\% issue)
    return h.replace("%", r"\%")

def _fmt_scalar_value(v):
    """Pretty-print for LaTeX table cells."""
    import numpy as np
    # Booleans -> On/Off in typewriter font
    if isinstance(v, bool):
        return r"\texttt{On}" if v else r"\texttt{Off}"
    # Numbers: int or float
    if isinstance(v, (int, np.integer)):
        return f"{int(v)}"
    if isinstance(v, (float, np.floating)):
        # compact but readable; 1e-08 -> 1e-08, 0.25 -> 0.25
        return f"{v:.8g}"
    # Tuples/lists -> literal tuple/list in \texttt{}
    if isinstance(v, (tuple, list)):
        inside = ", ".join(_fmt_scalar_value(x) for x in v)
        # strip any nested \texttt{} inside lists to avoid nesting
        inside = inside.replace(r"\texttt{", "").replace("}", "")
        return r"\texttt{(" + inside + r")}" if isinstance(v, tuple) else r"\texttt{[" + inside + r"]}"
    # Strings -> escape underscores and wrap in \texttt{}
    s = str(v).replace("_", r"\_")
    return r"\texttt{" + s + "}"


def _dict_to_df(dct, *, key_header="Parameter", val_header="Value"):
    """Turn a (possibly None) dict into a two-column DataFrame, preserving insertion order."""
    import pandas as pd
    dct = dct or {}
    rows = []
    for k, v in dct.items():
        key_tex = _escape_filter_value(k)   # wraps and escapes underscores
        val_tex = _fmt_scalar_value(v)
        rows.append((key_tex, val_tex))
    df = pd.DataFrame(rows, columns=[key_header, val_header])
    return df


def _filters_to_df(filters: dict):
    """Format the boolean filter toggles as Filter / Setting."""
    import pandas as pd
    filters = filters or {}
    rows = []
    for k, v in filters.items():
        k_tex = _escape_filter_value(k)
        v_tex = r"\texttt{On}" if bool(v) else r"\texttt{Off}"
        rows.append((k_tex, v_tex))
    return pd.DataFrame(rows, columns=["Filter", "Setting"])

# --- BibTeX helpers -----------------------------------------------------------
def default_references_bib() -> str:
    """
    Returns the project's default BibTeX references as a single string.
    Keep all canonical entries here so multiple reports can reuse them.
    """
    return r"""
                @article{van2025duration,
                  title={Duration-based valuation of corporate bonds},
                  author={van Binsbergen, Jules H and Nozawa, Yoshio and Schwert, Michael},
                  journal={The Review of Financial Studies},
                  volume={38},
                  number={1},
                  pages={158--191},
                  year={2025},
                  publisher={Oxford University Press}
                }
                
                @unpublished{DickersonRobottiRossetti_2024,
                  author = {Alexander Dickerson and Cesare Robotti and Giulio Rossetti},
                  note = {Working Paper},
                  title={Common pitfalls in the evaluation of corporate bond strategies},
                  year = {2024}
                }
                
                @article{dick2009liquidity,
                  title={Liquidity biases in TRACE},
                  author={Dick-Nielsen, Jens},
                  journal={The Journal of Fixed Income},
                  volume={19},
                  number={2},
                  pages={43},
                  year={2009},
                  publisher={Pageant Media}
                }
                
                @unpublished{dick2014clean,
                  title={How to clean enhanced TRACE data},
                  author={Dick-Nielsen, Jens},
                  note={Working Paper},
                  year={2014}
                }

                @article{rossi2014realized,
                  title={Realized volatility, liquidity, and corporate yield spreads},
                  author={Rossi, Marco},
                  journal={The Quarterly Journal of Finance},
                  volume={4},
                  number={01},
                  pages={1450004},
                  year={2014},
                  publisher={World Scientific}
                }
""".strip()


def write_references_bib(out_dir: Union[str, Path], *, overwrite: bool = True) -> Path:
    """
    Write references.bib into out_dir using the default entries.
    Returns the full Path to the written (or existing) file.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    bib_path = out_path / "references.bib"

    if bib_path.exists() and not overwrite:
        return bib_path

    with open(bib_path, "w", encoding="utf-8") as f:
        f.write(default_references_bib())
    return bib_path

# -------------------------------------------------------------------------
def build_data_report_tex(
    *,
    out_dir: Path,
    data_type: str,      # 'standard', 'enhanced', or '144a'
    filters_df,
    ds_df,
    bb_df,
    ie_df=None,
    fisd_params: dict,
    filters_table_dr,
    filters_table_dn,
    filters_table_fi,
    output_figures: bool = False,
    pages_made_ds: Optional[Iterable[Union[str, Path]]] = None,
    pages_made_bb: Optional[Iterable[Union[str, Path]]] = None,
    pages_made_ie: Optional[Iterable[Union[str, Path]]] = None,
    author: str = None,
) -> Path:
    """
    Assemble and write data_report.tex in out_dir, including natbib + apalike bibliography.
    If output_figures=True, figures are included from pages_made_ds / pages_made_bb / pages_made_ie (if provided).
    """
    # Enforce pages lists only when figures are requested
    if output_figures and (pages_made_ds is None or pages_made_bb is None):
        raise ValueError(
            "output_figures=True requires pages_made_ds and pages_made_bb "
            "(can be empty lists if no pages were produced)."
        )

    
    
    _title_map = {"standard": "Standard", "enhanced": "Enhanced", "144a": "144A"}
    dtype_title = _title_map.get(str(data_type).lower().strip(), str(data_type))

    # Build author line if provided
    author_line = rf"\author{{{author}}}" if author else ""

    tex_lines = []
    tex_lines.append(r"\documentclass[11pt]{article}")
    tex_lines.append(r"\usepackage{graphicx,booktabs,geometry,ragged2e,setspace}")
    tex_lines.append(r"\usepackage{amsmath,amssymb}")
    tex_lines.append(r"\usepackage[round,authoryear]{natbib}")
    tex_lines.append(r"\usepackage{hyperref}")
    tex_lines.append(r"\geometry{margin=1in}")
    tex_lines.append(rf"\title{{Stage 0 {dtype_title} TRACE Daily Data Report}}")
    if author_line:
        tex_lines.append(author_line)
    tex_lines.append(rf"\date{{{datetime.now().strftime('%Y-%m-%d')}}}")
    tex_lines.append(r"\begin{document}")
    tex_lines.append(r"\maketitle")
    tex_lines.append(r"""
\begin{abstract}
This document presents the data processing pipeline for """ + dtype_title + r""" Trade Reporting
and Compliance Engine (TRACE) data, converting intraday transaction-level data to a daily format.
We apply new error filters by \citet{DickersonRobottiRossetti_2024} which are more
conservative than \citet{rossi2014realized}. The new filters first attempt to correct for decimal
shift errors and then apply a rules-based approach to eliminate data entry errors. As a result,
less data is discarded, and the potential for false-positive data entry errors is
reduced. In addition, for each and every bond \texttt{cusip\_id} that is impacted
by \textit{any} of the \citet{DickersonRobottiRossetti_2024} filters, the time-series of its
transaction-level price series is plotted and retained in this report. The data processing
pipeline is part of the \href{https://openbondassetpricing.com/}{Open Source Bond Asset Pricing}
initiative \citep{DickersonRobottiRossetti_2024}, which aims to provide transparent and
reproducible methods for corporate bond research.
\end{abstract}
""")
    tex_lines.append(r"\section{Configured Filters and Parameters}")

    # Table 1
    tex_lines.append(make_parameters_table(filters_df, ds_df, bb_df, ie_df=ie_df))

    # Table 2
    fi_df = _dict_to_df(fisd_params, key_header="FISD Parameters", val_header="Value")
    tex_lines.append(r"\clearpage")
    tex_lines.append(r"\section{Filter Tables}")
    tex_lines.append(make_fisd_table(fi_df))

    # Table 3
    dr_fmt = _format_table(filters_table_dr, percent_col="%removed_start")
    dn_fmt = _format_table(filters_table_dn, percent_col="%_start")
    df_fmt = _format_table(filters_table_fi, percent_col="%_start")
    tex_lines.append(make_filters_counts_table(df_fmt, dr_fmt, dn_fmt))

    # Figures (optional)
    if output_figures:
        ds_list = list(pages_made_ds or [])
        bb_list = list(pages_made_bb or [])
        ie_list = list(pages_made_ie or [])

        if not ds_list and not bb_list and not ie_list:
            logging.warning("output_figures=True but no figure pages provided.")

        if ds_list:
            tex_lines.append(r"\clearpage")
            tex_lines.append(r"\section{Decimal Shift Corrections}")
            for png_name in ds_list:
                tex_lines.append(r"\begin{figure}[h!]\centering")
                tex_lines.append(
                    rf"\includegraphics[width=\textwidth,height=\textheight,keepaspectratio]{{{png_name}}}"
                )
                tex_lines.append(r"\end{figure}")
                tex_lines.append(r"\clearpage")

        if bb_list:
            tex_lines.append(r"\clearpage")
            tex_lines.append(r"\section{Bounce Back Corrections}")
            for png_name in bb_list:
                tex_lines.append(r"\begin{figure}[h!]\centering")
                tex_lines.append(
                    rf"\includegraphics[width=\textwidth,height=\textheight,keepaspectratio]{{{png_name}}}"
                )
                tex_lines.append(r"\end{figure}")
                tex_lines.append(r"\clearpage")

        if ie_list:
            tex_lines.append(r"\clearpage")
            tex_lines.append(r"\section{Initial Price Error Corrections}")
            for png_name in ie_list:
                tex_lines.append(r"\begin{figure}[h!]\centering")
                tex_lines.append(
                    rf"\includegraphics[width=\textwidth,height=\textheight,keepaspectratio]{{{png_name}}}"
                )
                tex_lines.append(r"\end{figure}")
                tex_lines.append(r"\clearpage")

    # Bibliography
    tex_lines.append(r"\clearpage")
    tex_lines.append(r"\bibliographystyle{apalike}")
    tex_lines.append(r"\bibliography{references}")
    tex_lines.append(r"\end{document}")

    # Ensure references.bib exists
    bib_path = write_references_bib(out_dir, overwrite=True)
    logging.info("Wrote/confirmed BibTeX at: %s", bib_path)

    # Write the .tex file
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = Path(out_dir) / f"{data_type}_data_report.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines))

    logging.info("Wrote LaTeX: %s", tex_path)
    logging.info("=== Done ===")
    return tex_path