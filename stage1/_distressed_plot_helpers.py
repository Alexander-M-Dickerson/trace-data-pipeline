# -*- coding: utf-8 -*-
"""
_distressed_plot_helpers.py
===========================
Plotting utilities and LaTeX document generation for Stage 1 Ultra Distressed filter reports.

Reuses plot styling from stage0/_error_plot_helpers.py for consistency.

Author : Alex Dickerson
Created: 2025-12-03
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, List, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import os
if not os.environ.get("DISPLAY"):
    os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import AutoDateLocator, DateFormatter
import logging

# ---------------------------------------------------------------------
# Plotting params (identical to stage0/_error_plot_helpers.py)
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

    export_format: str = "pdf"
    jpeg_quality: int = 85
    transparent: bool = False

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


# ---------------------------------------------------------------------
# Panel plotting for ultra distressed filter
# ---------------------------------------------------------------------
def _plot_panel_distressed(ax,
                           dfID: pd.DataFrame,
                           cusip: str,
                           p: PlotParams,
                           *,
                           show_legend: bool = True,
                           flag_col: str = "flag_refined_any",
                           price_col: str = "pr") -> None:
    """
    Plot one CUSIP panel for ultra distressed filter.

    Parameters
    ----------
    ax : matplotlib Axes
    dfID : DataFrame filtered to a single cusip_id
        Must contain: 'trd_exctn_dt', price_col, flag_col
    cusip : str
    p : PlotParams
    flag_col : str
        Column indicating flagged observations (1 = flagged/eliminated, 0 = kept)
    price_col : str
        Column containing prices to plot
    """
    d = dfID if dfID.index.is_monotonic_increasing else dfID.sort_values("trd_exctn_dt", ignore_index=True)

    dates = d["trd_exctn_dt"]
    y_all = d[price_col].astype(float)

    # "Filtered" line is the kept trades (flag == 0)
    kept_mask = d[flag_col].fillna(0).astype(int) == 0
    kept = d.loc[kept_mask]
    y_filtered = kept[price_col].astype(float)

    flagged_mask = d[flag_col].fillna(0).astype(int) == 1
    flagged = d.loc[flagged_mask]
    y_flagged = flagged[price_col].astype(float)

    if p.x_spacing == "rank":
        pos = np.arange(len(d))
        km = kept_mask.to_numpy()
        fm = flagged_mask.to_numpy()
        x_all = pos
        x_filt = pos[km]
        x_flag = pos[fm]
    else:
        x_all = dates
        x_filt = kept["trd_exctn_dt"]
        x_flag = flagged["trd_exctn_dt"]

    # Plot: All vs Filtered
    ax.plot(x_all, y_all,
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
            label="Eliminated",
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


# ---------------------------------------------------------------------
# Multi-panel figure creation
# ---------------------------------------------------------------------
def make_distressed_panel(df_out: pd.DataFrame,
                          error_cusips: List[str],
                          subplot_dim: Tuple[int, int] = (4, 2),
                          export_dir: Path = None,
                          filename_stub: str = None,
                          params: PlotParams = None,
                          idx_map: dict = None,
                          *,
                          flag_col: str = "flag_refined_any",
                          price_col: str = "pr") -> Path:
    """
    Create an A4 figure with (rows x cols) subplots (one CUSIP per panel) and save.

    Parameters
    ----------
    df_out : DataFrame
        Must contain: ['cusip_id', 'trd_exctn_dt', price_col, flag_col]
    error_cusips : list[str]
        CUSIPs to plot on this page
    subplot_dim : (rows, cols)
    export_dir : path-like
    filename_stub : optional custom filename root
    params : PlotParams
    idx_map : dict mapping cusip_id -> row indices (for fast slicing)
    flag_col : str
        Column indicating flagged observations
    price_col : str
        Column containing prices

    Returns
    -------
    Path to saved figure
    """
    if export_dir is None:
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

    needed = ["cusip_id", "trd_exctn_dt", price_col, flag_col]
    missing = [c for c in needed if c not in df_out.columns]
    if missing:
        raise KeyError(f"df_out missing required columns: {missing}")

    # Draw each CUSIP panel
    page_cusips = list(error_cusips[:n_panels])

    if len(page_cusips) == 0:
        page_groups = {}
    else:
        if idx_map is not None:
            idx_lists = [idx_map.get(c, None) for c in page_cusips]
            idx_lists = [ix for ix in idx_lists if ix is not None and len(ix)]
            if len(idx_lists):
                take_idx = np.concatenate(idx_lists)
                df_sub = df_out.take(take_idx)[needed]
                df_sub = df_sub[df_sub["cusip_id"].isin(page_cusips)]
                gb = df_sub.groupby("cusip_id", sort=False, observed=True)
                page_groups = {k: v for k, v in gb}
            else:
                page_groups = {}
        else:
            cats = pd.Categorical(df_out["cusip_id"], categories=page_cusips, ordered=False)
            mask = cats.notna()
            df_sub = df_out.loc[mask, needed].copy()
            df_sub["cusip_id"] = cats[mask]
            if not pd.api.types.is_datetime64_any_dtype(df_sub["trd_exctn_dt"].dtype):
                df_sub["trd_exctn_dt"] = pd.to_datetime(df_sub["trd_exctn_dt"], errors="coerce")
            gb = df_sub.groupby("cusip_id", sort=False, observed=True)
            page_groups = {k: v for k, v in gb}

    for i, (ax, cusip) in enumerate(zip(axes, page_cusips)):
        dfi = page_groups.get(cusip)
        if dfi is None or dfi.empty:
            ax.text(0.5, 0.5, f"No data for {cusip}", ha="center", va="center")
            ax.axis("off")
            continue
        _plot_panel_distressed(ax, dfi, cusip, p, show_legend=(i == 0),
                               flag_col=flag_col, price_col=price_col)

    # Fill empty slots
    if len(error_cusips) < n_panels:
        for ax in axes[len(error_cusips):]:
            ax.text(0.5, 0.5, "No CUSIP provided", ha="center", va="center")
            ax.axis("off")

    if p.suptitle:
        fig.suptitle(p.suptitle, y=0.97)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows_cols = f"{rows}x{cols}"
    stub = filename_stub or f"distressed_{rows_cols}_{ts}"

    fmt = (params.export_format if params else "pdf").lower()
    ext = "pdf" if fmt == "pdf" else ("jpg" if fmt in ("jpg", "jpeg") else "png")
    out_path = Path(export_dir) / f"{stub}.{ext}"

    savefig_kwargs = dict(
        format=ext,
        bbox_inches=None,
        facecolor="white",
        edgecolor="none",
        transparent=(params.transparent if params else False),
    )

    if ext in ("png", "jpg", "jpeg"):
        savefig_kwargs["dpi"] = (params.figure_dpi if params else 150)

    if ext in ("png",):
        savefig_kwargs["pil_kwargs"] = {"optimize": True, "compress_level": 6}
    elif ext in ("jpg", "jpeg"):
        q = max(60, min(95, getattr(params, "jpeg_quality", 85)))
        savefig_kwargs["pil_kwargs"] = {"quality": q, "optimize": True, "progressive": True}

    fig.savefig(out_path, **savefig_kwargs)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------
def _escape_latex(s: str) -> str:
    """Escape underscores for LaTeX."""
    return str(s).replace("_", r"\_")


def build_distressed_summary_table(
    total_rows: int,
    total_cusips: int,
    flagged_rows: int,
    flagged_cusips: int,
    flag_breakdown: dict = None,
    caption: str = "Ultra Distressed Filter Summary",
    label: str = "tab:distressed_summary",
) -> str:
    """
    Build a LaTeX table summarizing the ultra distressed filter impact.

    Parameters
    ----------
    total_rows : int
        Total rows in the dataset
    total_cusips : int
        Total unique cusip_id in dataset
    flagged_rows : int
        Number of rows flagged by filter
    flagged_cusips : int
        Number of unique cusip_id impacted
    flag_breakdown : dict, optional
        Breakdown by individual flag columns {col_name: count}
    caption : str
    label : str

    Returns
    -------
    LaTeX table string
    """
    pct_rows = 100 * flagged_rows / total_rows if total_rows > 0 else 0
    pct_cusips = 100 * flagged_cusips / total_cusips if total_cusips > 0 else 0

    rows_tex = []
    rows_tex.append(r"Total Observations & " + f"{total_rows:,}" + r" \\")
    rows_tex.append(r"Total CUSIPs & " + f"{total_cusips:,}" + r" \\")
    rows_tex.append(r"\midrule")
    rows_tex.append(r"Flagged Observations & " + f"{flagged_rows:,}" + f" ({pct_rows:.2f}\\%)" + r" \\")
    rows_tex.append(r"Flagged CUSIPs & " + f"{flagged_cusips:,}" + f" ({pct_cusips:.2f}\\%)" + r" \\")

    if flag_breakdown:
        rows_tex.append(r"\midrule")
        rows_tex.append(r"\multicolumn{2}{c}{\textbf{Breakdown by Flag Type}} \\")
        rows_tex.append(r"\midrule")
        for col, cnt in flag_breakdown.items():
            col_esc = _escape_latex(col)
            pct = 100 * cnt / total_rows if total_rows > 0 else 0
            rows_tex.append(rf"\texttt{{{col_esc}}} & {cnt:,} ({pct:.2f}\%) \\")

    body = "\n".join(rows_tex)

    latex = rf"""
\begin{{table}}[!ht]
\begin{{center}}
\footnotesize
\caption{{{caption}}}
\label{{{label}}}\vspace{{2mm}}
\begin{{tabular}}{{lr}}
\midrule
Metric & Value \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{center}}
\begin{{spacing}}{{1}}
{{\footnotesize
This table summarizes the impact of the ultra distressed filter applied in Stage 1.
The filter identifies observations with anomalous price behavior including ultra-low prices,
upward price spikes, plateau sequences, and intraday price inconsistencies.
Flagged observations are candidates for exclusion from downstream analysis.
}}
\end{{spacing}}
\vspace{{-2mm}}
\end{{table}}
""".strip()

    return latex


def default_references_bib() -> str:
    """Returns the project's default BibTeX references."""
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
    """Write references.bib into out_dir."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    bib_path = out_path / "references.bib"

    if bib_path.exists() and not overwrite:
        return bib_path

    with open(bib_path, "w", encoding="utf-8") as f:
        f.write(default_references_bib())
    return bib_path


def build_distressed_report_tex(
    *,
    out_dir: Path,
    total_rows: int,
    total_cusips: int,
    flagged_rows: int,
    flagged_cusips: int,
    flag_breakdown: dict = None,
    pages_made: List[str] = None,
    author: str = None,
) -> Path:
    """
    Assemble and write the Stage 1 Ultra Distressed Filter report LaTeX file.

    Parameters
    ----------
    out_dir : Path
        Output directory for .tex and .bib files
    total_rows : int
    total_cusips : int
    flagged_rows : int
    flagged_cusips : int
    flag_breakdown : dict, optional
    pages_made : list of figure filenames
    author : str, optional

    Returns
    -------
    Path to written .tex file
    """
    author_line = rf"\author{{{author}}}" if author else ""

    tex_lines = []
    tex_lines.append(r"\documentclass[11pt]{article}")
    tex_lines.append(r"\usepackage{graphicx,booktabs,geometry,ragged2e,setspace}")
    tex_lines.append(r"\usepackage{amsmath,amssymb}")
    tex_lines.append(r"\usepackage[round,authoryear]{natbib}")
    tex_lines.append(r"\usepackage{hyperref}")
    tex_lines.append(r"\geometry{margin=1in}")
    tex_lines.append(r"\title{Stage 1 Ultra Distressed Filters}")
    if author_line:
        tex_lines.append(author_line)
    tex_lines.append(rf"\date{{{datetime.now().strftime('%Y-%m-%d')}}}")
    tex_lines.append(r"\begin{document}")
    tex_lines.append(r"\maketitle")

    # Abstract
    tex_lines.append(r"""
\begin{abstract}
This document presents the results of the Stage 1 Ultra Distressed Filter applied to daily
corporate bond price data. The filter identifies observations exhibiting anomalous price
behavior that may indicate data errors. Specifically, the filter
flags: (1) ultra-low prices inconsistent with historical trading patterns, (2) upward price
spikes followed by immediate reversals, (3) extended plateau sequences at suspiciously round
price levels, and (4) intraday price inconsistencies. For each bond \texttt{cusip\_id}
impacted by the filter, the time-series of its price is plotted and retained in this report
for manual inspection. The filtering methodology follows
\citet{DickersonRobottiRossetti_2024} and is part of the
\href{https://openbondassetpricing.com/}{Open Source Bond Asset Pricing} initiative.
\end{abstract}
""")

    # Summary section
    tex_lines.append(r"\section{Filter Summary}")
    tex_lines.append(build_distressed_summary_table(
        total_rows=total_rows,
        total_cusips=total_cusips,
        flagged_rows=flagged_rows,
        flagged_cusips=flagged_cusips,
        flag_breakdown=flag_breakdown,
    ))

    # Figures section
    if pages_made and len(pages_made) > 0:
        tex_lines.append(r"\clearpage")
        tex_lines.append(r"\section{Flagged CUSIP Price Series}")
        tex_lines.append(r"""
Each figure below displays the price time-series for a bond flagged by the ultra distressed
filter. The orange line shows all observations, the dark line shows retained (non-flagged)
observations, and red circles indicate flagged observations that are candidates for exclusion.
Note that not all flagged candidates will have extremely low prices. Part of 
the filtering protocol identifies within-day price and return movements that could be anomalous. 
The default setting flags candidates where intraday price movements exceed \$200 or where
the within-day return using min and max transaction prices exceeds $\lvert 75\% \rvert$.
""")
        for fig_name in pages_made:
            tex_lines.append(r"\begin{figure}[h!]\centering")
            tex_lines.append(
                rf"\includegraphics[width=\textwidth,height=\textheight,keepaspectratio]{{{fig_name}}}"
            )
            tex_lines.append(r"\end{figure}")
            tex_lines.append(r"\clearpage")

    # Bibliography
    tex_lines.append(r"\clearpage")
    tex_lines.append(r"\bibliographystyle{apalike}")
    tex_lines.append(r"\bibliography{references}")
    tex_lines.append(r"\end{document}")

    # Ensure references.bib exists
    out_dir.mkdir(parents=True, exist_ok=True)
    bib_path = write_references_bib(out_dir, overwrite=True)
    logging.info("Wrote/confirmed BibTeX at: %s", bib_path)

    # Write .tex file
    tex_path = Path(out_dir) / "stage1_distressed_report.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines))

    logging.info("Wrote LaTeX: %s", tex_path)
    return tex_path
