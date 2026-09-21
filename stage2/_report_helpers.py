# -*- coding: utf-8 -*-
"""
stage2_data_report_helpers.py
=============================
Helper functions for generating Stage 2 data reports.

Includes plotting utilities, LaTeX document generation, and table formatting
functions for the monthly asset pricing panel documentation.

Author: Alex Dickerson
Created: 2025-12
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import gc
from pandas.tseries.offsets import MonthEnd
from typing import Union


# ============================================================================
# PLOTTING STYLE (UNIFIED WITH STAGE 0/1)
# ============================================================================

@dataclass
class PlotParams:
    """Unified plotting parameters matching stage0/stage1 style."""
    use_latex: bool = False
    base_font: int = 10
    title_size: int = 10
    label_size: int = 10
    tick_size: int = 9
    legend_size: int = 9
    figure_dpi: int = 150
    export_format: str = "pdf"
    transparent: bool = False
    grid_alpha: float = 0.25
    grid_lw: float = 0.6
    line_color: str = "0.05"
    line_alpha: float = 1.0
    line_lw: float = 1.25


def apply_plot_params(params: PlotParams):
    """Apply plotting parameters to matplotlib."""
    matplotlib.rcParams.update({
        "text.usetex": params.use_latex,
        "font.family": "serif",
        "font.size": params.base_font,
        "axes.titlesize": params.title_size,
        "axes.labelsize": params.label_size,
        "xtick.labelsize": params.tick_size,
        "ytick.labelsize": params.tick_size,
        "legend.fontsize": params.legend_size,
        "figure.dpi": params.figure_dpi,
    })


# ============================================================================
# LATEX DOCUMENT GENERATION
# ============================================================================

# The released panel artifacts are named for their vintage year (main_panel_2026.parquet
# and friends), so the prose below has to move with the release rather than naming a
# frozen year. The document is written against 2025 and re-stamped on assembly.
_VINTAGE_IN_SOURCE = "2025"


def build_latex_document(
    tables: dict = None,
    fig_filenames: list = None,
    author: str = None,
    vintage: str = None,
    facts: dict = None,
) -> str:
    """
    Build complete LaTeX document for Stage 2 monthly panel report.

    `facts` fills the @@NAME@@ tokens in the prose with numbers computed from the panel the
    report describes (_build_data_report.window_facts). They used to be typed in, and a typed
    number outlives the data it came from. A token left unfilled is refused.

    Parameters
    ----------
    tables : dict, optional
        Dictionary of table name -> LaTeX table string.
        Keys can include: 'config', 'panel_summary', 'factor_summary', etc.
    fig_filenames : list of tuples, optional
        List of (filename, caption) tuples for figures.
    author : str, optional
        Author name for the document.

    Returns
    -------
    str
        Complete LaTeX document.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d')

    # Build tables section
    tables_section = ""
    if tables:
        for name, table_str in tables.items():
            if table_str:
                tables_section += f"\n\\clearpage\n{table_str}\n"

    # Build figures section if provided
    figures_section = ""
    if fig_filenames:
        figures_section = r"""

\clearpage
\section{Figures}
"""
        for i, (filename, caption) in enumerate(fig_filenames, start=1):
            if i > 1:
                figures_section += r"\clearpage" + "\n"

            figures_section += r"""
\begin{figure}[h!]
\centering
\includegraphics[width=\textwidth,height=0.85\textheight,keepaspectratio]{""" + filename + r"""}
\caption{""" + caption + r"""}
\label{fig:figure_""" + str(i) + r"""}
\end{figure}
"""

    # Data description section
    data_section = r"""
\section{Monthly Open Source Bond Asset Pricing TRACE Data}

This section describes the construction of monthly corporate bond returns from the Trade Reporting and Compliance Engine (TRACE) database. The panel is designed to facilitate cross-sectional asset pricing tests.

\subsection{How to Use the Monthly Data}

All data in the panel is sampled at the end of month $t$; no variables have a lead or a lag.

Several datasets are available on the \href{https://openbondassetpricing.com/}{Open Bond Asset Pricing} website.

\subsubsection{Market Microstructure Adjusted Signals and Returns}

The main panel includes several identifiers, additional variables, and signals. As standard, we provide market microstructure-adjusted (MMN) price-based signals, which includes all variables related to bond price, yield, spread (including spread-based value signals), and prior 1-month return. The methodology is outlined in Section~\ref{sec:signal_gap} and graphically presented in Panel B: Adjusted Signal in Figure~\ref{fig:return_timeline}.

As an alternative, researchers may access and download the unadjusted price-based signals from \href{https://openbondassetpricing.com/}{openbondassetpricing.com}. The file is named \texttt{mmn\_price\_based\_signals\_2025.parquet}; all variables in this file have the suffix \texttt{\_mmn}. These can be merged to the main panel to use unadjusted signals, but researchers must then use \texttt{ret\_vw\_bgn} to adjust returns, as illustrated in Panel C of Figure~\ref{fig:return_timeline}.

Our recommendation is to use the main panel data as provided. However, results are extremely similar using either method.

\subsubsection{Excess and Duration-Adjusted Return-Based Signals}

The main panel assumes researchers will form factors with excess returns, $r - r^{f}$, which are computed from the panel as \texttt{ret\_vw - rfret}. If research designs instead rely on duration-adjusted returns, $r^{x} = r - r^{\text{Tsy}}$ --- this is what \texttt{ret\_vwx} means throughout, namely \texttt{ret\_vw - tret} --- we provide signals specifically computed with duration-adjusted returns. These include all factor betas in \texttt{betas\_x\_2025.parquet} and momentum/long-term reversal variables in \texttt{mom\_retx\_2025.parquet}, where the ``x'' suffix indicates signals computed with duration-adjusted returns.

The short-term reversal signal can be computed as the current \texttt{str} variable minus \texttt{tret}. This signal is already MMN-adjusted in the main panel.

\subsubsection{Alternative Month-End Returns}

For researchers requiring alternative return measures, we provide \texttt{returns\_alt\_2025.parquet}, which includes:
\begin{itemize}
\item \texttt{ret\_vwp}: Returns computed using par-weighted prices on day $d$
\item \texttt{ret\_ew}: Returns computed using equal-weighted prices on day $d$
\item \texttt{ret\_1st}: Returns computed using the first available trade price on day $d$
\item \texttt{ret\_lst}: Returns computed using the last available trade price on day $d$
\item \texttt{ret\_bid}: Returns computed using the volume-weighted average bid price on day $d$
\end{itemize}
\noindent where day $d$ is in the last 5 business days of months $t$ and $t{+}1$.

\subsection{Return Computation}

We compute monthly holding-period returns using two measurement windows. The \emph{month-end return} measures performance from the end of month $t$ to the end of month $t+1$:
\begin{equation}
r_{i,t+1}^{\text{End}} = \frac{P_{i,t+1}^{\text{end}} + AI_{i,t+1}^{\text{end}} + C_{i,t+1}}{P_{i,t}^{\text{end}} + AI_{i,t}^{\text{end}}} - 1,
\label{eq:ret_end}
\end{equation}
where $P_{i,t+1}^{\text{end}}$ is the clean price, $AI_{i,t+1}^{\text{end}}$ is accrued interest, and $C_{i,t+1}$ is any coupon payment during the month. The \emph{month-begin return} measures performance from the beginning to the end of month $t+1$:
\begin{equation}
r_{i,t+1}^{\text{Bgn}} = \frac{P_{i,t+1}^{\text{end}} + AI_{i,t+1}^{\text{end}} + C_{i,t+1}}{P_{i,t+1}^{\text{bgn}} + AI_{i,t+1}^{\text{bgn}}} - 1,
\label{eq:ret_bgn}
\end{equation}
where all prices are observed within the same calendar month. The month-begin return is relevant for implementable trading strategies based on price-based signals, as a trader observing a signal at month-end cannot transact at that exact price.

Following \citet{AndreaniPalharesRichardson_2023}, we also compute duration-adjusted returns to remove exposure to interest rate movements:
\begin{equation}
r_{i,t+1}^{x} = r_{i,t+1} - r_{i,t+1}^{\text{Tsy}},
\label{eq:ret_x}
\end{equation}
where $r_{i,t+1}^{\text{Tsy}}$ is the return on a duration-matched Treasury portfolio. This adjustment isolates credit-specific performance from parallel shifts in the yield curve.

Figure~\ref{fig:return_timeline} illustrates the timing of price observations for month-end and month-begin returns. In the data, the month-end return is stored in \texttt{ret\_vw}, with trade dates \texttt{dt\_s} (month $t$) and \texttt{dt\_e} (month $t{+}1$), and holding period \texttt{hprd}. The month-begin return is stored in \texttt{ret\_vw\_bgn}, with trade dates \texttt{dt\_s\_bgn} and \texttt{dt\_e\_bgn} within month $t{+}1$, and holding period \texttt{hprd\_bgn}. Both holding periods count NYSE trading sessions between the two trades. The matched Treasury return $r_{i,t+1}^{\text{Tsy}}$ is stored in \texttt{tret}; duration-adjusted returns can be computed by subtracting \texttt{tret} from \texttt{ret\_vw} or \texttt{ret\_vw\_bgn}.

\paragraph{The 5-session rule and the length of a month-end return.} A month-end return needs a trade in the last 5 NYSE sessions of month $t$ and another in the last 5 sessions of month $t{+}1$. A bond that misses either window has no return for that month. Each trade may fall on any of its 5 sessions, so the window between them, \texttt{hprd}, runs from @@HPRD_MIN@@ to @@HPRD_MAX@@ sessions, with a mean of @@HPRD_MEAN@@. A month has at most 23 sessions. A window longer than that is a return whose start trade fell before the last session of its month and whose end trade fell late, which the rule allows. @@HPRD_GT23@@\% of returns are of that kind. The distribution across all @@N_RET@@ month-end returns is below.

@@WINDOW_TABLE@@

\paragraph{Two prices.} The tables report two prices. \textit{Price (VW)} is the volume-weighted month-end price that the return is computed from, and every return has one. \textit{Price at Signal Date} is the price on the signal date described in Section~\ref{sec:signal_gap}, which the price-based signals in the main panel use. @@N_RET_NO_SIGNAL_PRICE@@ returns have no signal-date price, because the bond had no earlier trade in the same month within @@ADJ_WINDOW@@ sessions of its month-end trade.

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
    >=Stealth,
    month/.style={draw, minimum width=4.2cm, minimum height=0.8cm, font=\small},
    timepoint/.style={circle, fill=black, inner sep=1.5pt},
    signalpoint/.style={circle, fill=blue!70!black, inner sep=1.5pt},
    brace/.style={decorate, decoration={brace, amplitude=4pt, raise=1pt}},
    bracebelow/.style={decorate, decoration={brace, amplitude=4pt, raise=1pt, mirror}},
    smallbrace/.style={decorate, decoration={brace, amplitude=3pt, raise=1pt, mirror}},
    lbl/.style={font=\footnotesize},
    panel/.style={font=\bfseries\small},
    paneldesc/.style={font=\small}
]

% ============ PANEL A: Month-End Return ============
\node[panel, anchor=west] at (-5.2, 4.6) {Panel A:};
\node[paneldesc, anchor=west] at (-3.4, 4.6) {Month-End Return};

% Month boxes
\node[month] (m1a) at (-2.5, 3.9) {Month $t$};
\node[month] (m2a) at (2.5, 3.9) {Month $t{+}1$};

% Timeline
\draw[thick, ->] (-5.2, 2.4) -- (5.2, 2.4) node[right, font=\footnotesize] {time};

% Month boundaries on timeline
\draw[gray, dashed] (-0.4, 3.4) -- (-0.4, 1.8);
\draw[gray, dashed] (4.6, 3.4) -- (4.6, 1.8);

% Price points
\node[timepoint] (p1a) at (-0.8, 2.4) {};
\node[timepoint] (p2a) at (4.2, 2.4) {};

% Labels above points
\node[lbl, above=2pt] at (p1a) {$P_{t}^{\text{end}}$};
\node[lbl, above=2pt] at (p2a) {$P_{t+1}^{\text{end}}$};

% 5BD underbraces
\draw[smallbrace] (-1.2, 2.25) -- node[below=4pt, font=\scriptsize] {5 BD} (-0.4, 2.25);
\draw[smallbrace] (3.8, 2.25) -- node[below=4pt, font=\scriptsize] {5 BD} (4.6, 2.25);


% ============ PANEL B: Adjusted Signal ============
\node[panel, anchor=west] at (-5.2, 1.4) {Panel B:};
\node[paneldesc, anchor=west] at (-3.4, 1.4) {Adjusted Signal};

% Month boxes
\node[month] (m1b) at (-2.5, 0.7) {Month $t$};
\node[month] (m2b) at (2.5, 0.7) {Month $t{+}1$};

% Timeline
\draw[thick, ->] (-5.2, -0.8) -- (5.2, -0.8) node[right, font=\footnotesize] {time};

% Month boundaries on timeline
\draw[gray, dashed] (-0.4, 0.2) -- (-0.4, -1.4);
\draw[gray, dashed] (4.6, 0.2) -- (4.6, -1.4);

% Price points (black - month-end prices)
\node[timepoint] (p1b) at (-0.8, -0.8) {};
\node[timepoint] (p2b) at (4.2, -0.8) {};

% Signal points (blue - observed before month-end)
\node[signalpoint] (s1b) at (-1.8, -0.8) {};
\node[signalpoint] (s2b) at (3.2, -0.8) {};

% Labels above points
\node[lbl, above=2pt] at (p1b) {$P_{t}^{\text{end}}$};
\node[lbl, above=2pt] at (p2b) {$P_{t+1}^{\text{end}}$};
\node[lbl, above=2pt, blue!70!black] at (s1b) {$P^{s}_{t-\Delta}$};
\node[lbl, above=2pt, blue!70!black] at (s2b) {$P^{s}_{t+1-\Delta}$};

% Signal gap arrows
\draw[<->, thick, blue!70!black] (-1.8, -1.15) -- (-0.8, -1.15);
\node[lbl, blue!70!black] at (-1.3, -1.45) {$\leq$10 BD};
\draw[<->, thick, blue!70!black] (3.2, -1.15) -- (4.2, -1.15);
\node[lbl, blue!70!black] at (3.7, -1.45) {$\leq$10 BD};


% ============ PANEL C: Month-Begin Return ============
\node[panel, anchor=west] at (-5.2, -2.2) {Panel C:};
\node[paneldesc, anchor=west] at (-3.4, -2.2) {Month-Begin Return};

% Month box (single month spanning wider)
\node[month, minimum width=8.4cm] (m1c) at (0, -2.9) {Month $t{+}1$};

% Timeline
\draw[thick, ->] (-5.2, -4.6) -- (5.2, -4.6) node[right, font=\footnotesize] {time};

% Month boundaries on timeline
\draw[gray, dashed] (-4.2, -3.5) -- (-4.2, -5.2);
\draw[gray, dashed] (4.2, -3.5) -- (4.2, -5.2);

% Price points
\node[timepoint] (p1c) at (-3.8, -4.6) {};
\node[timepoint] (p2c) at (3.8, -4.6) {};

% Labels above points
\node[lbl, above=2pt] at (p1c) {$P_{t+1}^{\text{bgn}}$};
\node[lbl, above=2pt] at (p2c) {$P_{t+1}^{\text{end}}$};

% 5BD underbraces
\draw[smallbrace] (-4.2, -4.75) -- node[below=4pt, font=\scriptsize] {5 BD} (-3.4, -4.75);
\draw[smallbrace] (3.4, -4.75) -- node[below=4pt, font=\scriptsize] {5 BD} (4.2, -4.75);

% Return brace (above)
\draw[brace] ($(p1c.north)+(0,0.45)$) -- node[above=5pt, lbl] {$r_{t+1}^{\text{Bgn}}$} ($(p2c.north)+(0,0.45)$);


% ============ PANEL D: LIB Decomposition ============
\node[panel, anchor=west] at (-5.2, -5.8) {Panel D:};
\node[paneldesc, anchor=west] at (-3.4, -5.8) {Latent Implementation Bias};

% Month boxes - nudged up very slightly
\node[month] (m1d) at (-2.5, -6.35) {Month $t$};
\node[month] (m2d) at (2.5, -6.35) {Month $t{+}1$};

% Timeline - pushed down a little
\draw[thick, ->] (-5.2, -8.1) -- (5.2, -8.1) node[right, font=\footnotesize] {time};

% Month boundaries on timeline
\draw[gray, dashed] (-0.4, -6.95) -- (-0.4, -9.3);
\draw[gray, dashed] (4.6, -6.95) -- (4.6, -9.3);

% Price points - P_{t+1}^{bgn} centered at x=0.0
\node[timepoint] (pe) at (-0.8, -8.1) {};
\node[timepoint] (pb) at (0.0, -8.1) {};
\node[timepoint] (pf) at (4.2, -8.1) {};

% Labels above points
\node[lbl, above=2pt] at (pe) {$P_{t}^{\text{end}}$};
\node[lbl, above=2pt] at (pb) {$P_{t+1}^{\text{bgn}}$};
\node[lbl, above=2pt] at (pf) {$P_{t+1}^{\text{end}}$};

% 5BD underbraces - all same width (0.8)
\draw[smallbrace] (-1.2, -8.25) -- node[below=4pt, font=\scriptsize] {5 BD} (-0.4, -8.25);
\draw[smallbrace] (-0.4, -8.25) -- node[below=4pt, font=\scriptsize] {5 BD} (0.4, -8.25);
\draw[smallbrace] (3.8, -8.25) -- node[below=4pt, font=\scriptsize] {5 BD} (4.6, -8.25);

% LIB brace (above, between pe and pb)
\draw[brace] ($(pe.north)+(0,0.45)$) -- node[above=5pt, lbl] {LIB} ($(pb.north)+(0,0.45)$);

% r_bgn brace (above, between pb and pf)
\draw[brace] ($(pb.north)+(0,0.45)$) -- node[above=5pt, lbl] {$r_{t+1}^{\text{Bgn}}$} ($(pf.north)+(0,0.45)$);

% r_end brace (below)
\draw[bracebelow] ($(pe.south)+(0,-0.55)$) -- node[below=5pt, lbl] {$r_{t+1}^{\text{End}}$} ($(pf.south)+(0,-0.55)$);

% IGAP annotation (from dashed line to pb) - thinner arrow
\draw[<->, thin, blue!70!black] (-0.4, -9.0) -- (0.0, -9.0);
\node[lbl, blue!70!black] at (-0.2, -9.3) {IGAP$\leq$5 BD};

\end{tikzpicture}
\caption{Return measurement windows and decomposition. Panel A shows the month-end return, measured from the last 5 business days of month $t$ to the last 5 business days of month $t{+}1$. Panel B shows the adjusted signal timing, where the representative investor observes signals at $P^{s}_{t-\Delta}$ and $P^{s}_{t+1-\Delta}$, up to 10 business days before the month-end prices. Panel C shows the month-begin return, measured entirely within month $t{+}1$ from the first 5 to last 5 business days. Panel D shows the Latent Implementation Bias decomposition: the month-end return spans from $P_{t}^{\text{end}}$ to $P_{t+1}^{\text{end}}$ and decomposes into LIB plus the month-begin return. The implementation gap (IGAP) measures the delay between signal observation and execution. The NYSE trading calendar is used to compute business days. The representative factor investor forms their position at the end of month $t$, with weights $\omega$, and holds the position---in this example for 1-month---rebalancing again at the end of month $t{+}1$. The main panel by default includes price-based signals with a minimum 1-business-day gap before the month-end price used for returns.}
\label{fig:return_timeline}
\end{figure}

\subsubsection{Latent Implementation Bias}

A trader observing a price-based signal at month-end $t$ cannot execute at that price. The earliest execution occurs at month-begin $t{+}1$. The \emph{Latent Implementation Bias} (LIB) captures this cost:
\begin{equation}
\text{LIB}_{i,t+1} = \frac{P_{i,t+1}^{\text{bgn}}}{P_{i,t}^{\text{end}}} - 1,
\label{eq:lib}
\end{equation}
which represents the clean price return between signal observation and trade execution. In the data, LIB is stored in \texttt{lib} (clean prices) and \texttt{libd} (dirty prices). The implementation gap \texttt{igap\_bgn} measures the business days between \texttt{dt\_e} of month $t$ and \texttt{dt\_s\_bgn} of month $t{+}1$, capped at 5.

\subsubsection{Signal Gap for Price-Based Signals}
\label{sec:signal_gap}

All price-based signals---those involving bond price, yield, spread, or prior 1-month return---are adjusted such that they are observed with at least a 1-business day gap (using the NYSE trading calendar) before the $P^{\text{end}}_{i,t}$ used for month-end return computation. We allow a maximum gap of @@ADJ_WINDOW@@ business days, within the same month. This adjustment removes the mechanical bid-ask bias that can inflate factor performance when sorting on a price-based signal that is also used in the return computation. Panel B: Adjusted Signal in Figure~\ref{fig:return_timeline} graphically illustrates this adjustment. In our sample, the average (median) signal gap is @@SIGGAP_MEAN@@ (@@SIGGAP_MEDIAN@@) business days and the largest is @@SIGGAP_MAX@@.

\subsection{Rule 144a Bonds}

Rule 144a bonds are privately placed securities that can be traded among qualified institutional buyers. Despite representing a substantial portion of corporate debt issuance, these securities have received limited attention in the academic literature. \citet{choi2025private} examine 144a bonds in the context of mutual fund holdings, but most monthly asset pricing studies exclude them entirely.

In the TRACE data available from WRDS (as opposed to academic TRACE), comprehensive coverage of 144a bonds begins only on 2014-07-31. Prior to this date, the number of 144a bonds in the database is sparse. By the end of our sample period, 144a bonds comprise approximately 25\% of all bond-month observations with valid returns. Given their economic importance as a source of firm financing and their active secondary market trading, we include 144a bonds in our sample. Tables~\ref{tab:desc_stats_144a} and Figure~\ref{fig:figure_3} document the characteristics and time-series evolution of 144a bond coverage.

\subsection{Defaulted Bonds}

We include bonds that have entered default in our sample. Recent evidence suggests that defaulted bonds remain actively traded in secondary markets and exhibit distinct pricing dynamics. \citet{baumann2025life} document how dealer intermediation affects recovery rates for defaulted bonds, while \citet{baumann2025defaulted} show that defaulted bonds represent a hybrid asset class with characteristics of both fixed income and equity securities. Figure~\ref{fig:figure_2} shows the time-series evolution of defaulted bond coverage. A bond is classified as defaulted if the S\&P rating (\texttt{sp\_rat}) equals 22 or the Moody's rating (\texttt{mdy\_rat}) equals 21 (S\&P's D and Moody's C, the lowest grade on each scale).

\subsubsection{Return Calculation for Defaulted Bonds}

Because defaulted bonds cease coupon payments, the standard total return formula must be adjusted. The variable \texttt{ret\_type} indicates which return formula was applied, taking one of three values:

\paragraph{Standard Return (\texttt{ret\_type = `standard'})} For bonds not in default, we use the standard total return formula as in Equation~\eqref{eq:ret_end}:
\begin{equation}
r_{i,t+1}^{\text{standard}} = \frac{P_{i,t+1}^{\text{end}} + AI_{i,t+1}^{\text{end}} + C_{i,t+1}}{P_{i,t}^{\text{end}} + AI_{i,t}^{\text{end}}} - 1.
\end{equation}

\paragraph{Default Event Return (\texttt{ret\_type = `default\_evnt'})} When a bond transitions into default at $t{+}1$ (i.e., not in default at $t$ but in default at $t{+}1$), coupon payments cease immediately. The return reflects only the price change, comparing the clean price at default to the dirty price before default:
\begin{equation}
r_{i,t+1}^{\text{default}} = \frac{P_{i,t+1}^{\text{end}}}{P_{i,t}^{\text{end}} + AI_{i,t}^{\text{end}}} - 1.
\label{eq:ret_default}
\end{equation}

\paragraph{Trading Under Default (\texttt{ret\_type = `trad\_in\_def'})} When a bond remains in default from $t$ to $t{+}1$, returns are based solely on clean price changes with no accrued interest component:
\begin{equation}
r_{i,t+1}^{\text{flat}} = \frac{P_{i,t+1}^{\text{end}}}{P_{i,t}^{\text{end}}} - 1.
\label{eq:ret_flat}
\end{equation}
We impose the constraint that the trading-under-default return cannot exceed the standard return. This affects a small number of monthly observations where the flat price change would otherwise produce an artificially higher return than the total return calculation.

\subsection{Sample Overview}

Table~\ref{tab:monthly_data_availability} reports the overall data availability across our sample period. Tables~\ref{tab:desc_stats_all}--\ref{tab:desc_stats_def} present descriptive statistics for all bonds, investment grade bonds (S\&P ratings AAA to BBB-), non-investment grade bonds (BB+ to C), and defaulted bonds, respectively. Table~\ref{tab:desc_stats_144a} presents descriptive statistics for Rule 144a bonds, which make up about a fifth of all bond-month observations at sample end. Figure~\ref{fig:figure_1} compares market-level returns computed using month-end versus month-begin pricing conventions, illustrating the economic magnitude of the implementation gap for price-based strategies. Figure~\ref{fig:figure_2} shows the time-series evolution of defaulted bond coverage, and Figure~\ref{fig:figure_3} documents the dynamics of 144a bond coverage.
"""

    doc = r"""\documentclass[11pt]{article}
\usepackage{graphicx,booktabs,geometry,ragged2e,setspace}
\usepackage{amsmath,amssymb}
\usepackage[round,authoryear]{natbib}
\usepackage{hyperref}
\usepackage{tikz}
\usepackage{longtable}
\usepackage{array}
\usetikzlibrary{arrows.meta,positioning,decorations.pathreplacing,calc}
\geometry{margin=1in}
\title{The Corporate Bond Factor Replication Crisis:\\ New Protocols}
\author{\href{https://openbondassetpricing.com/}{Open Source Bond Asset Pricing}}
\date{""" + timestamp + r"""}
\begin{document}
\maketitle

\begin{abstract}
This document presents an analysis of the monthly corporate bond asset pricing
panel constructed from Stage 1 daily data output of the \href{https://openbondassetpricing.com/}{Open Source Bond Asset Pricing} (OSBAP) pipeline, using the Trade Reporting and Compliance Engine (TRACE) database. We
aggregate daily bond data into a monthly panel suitable for cross-sectional asset pricing
tests, computing returns, risk measures, liquidity signals, and factor exposures.
This work is part of the OSBAP initiative \citep{DickersonRobottiRossetti_2024}, which aims to provide transparent
and reproducible methods for corporate bond research.
\end{abstract}
""" + data_section + tables_section + figures_section + r"""

\clearpage
\bibliographystyle{apalike}
\bibliography{references}
\end{document}
"""

    # Re-stamp the released-artifact filenames for this vintage. The escaped form
    # `\_2025.parquet` appears only in those names, so the substitution cannot reach
    # the DFPS constants (trace_alternate_2025_12_2024) or anything else.
    if vintage and vintage != _VINTAGE_IN_SOURCE:
        doc = doc.replace(rf"\_{_VINTAGE_IN_SOURCE}.parquet", rf"\_{vintage}.parquet")

    for name, value in (facts or {}).items():
        doc = doc.replace(f"@@{name}@@", str(value))
    import re as _re
    unfilled = sorted(set(_re.findall(r"@@([A-Z0-9_]+)@@", doc)))
    if unfilled:
        raise KeyError(
            f"the report's prose asks for {unfilled} and build_latex_document was not given "
            "them. Pass facts=_build_data_report.window_facts(panel).")

    return doc


def get_references_bib() -> str:
    """Return bibliography content for Stage 2 report."""
    return r"""
@article{AndreaniPalharesRichardson_2023,
  title={Computing Corporate Bond Returns: A Word (or Two) of Caution},
  author={Andreani, Martina and Palhares, Diogo and Richardson, Scott},
  journal={Review of Accounting Studies},
  volume={29},
  number={4},
  pages={3887--3906},
  year={2024}
}

@unpublished{DickersonRobottiRossetti_2024,
  author = {Alexander Dickerson and Cesare Robotti and Giulio Rossetti},
  note = {Working Paper. Earlier versions circulated as ``Common Pitfalls in the Evaluation of Corporate Bond Strategies''},
  title = {The Corporate Bond Factor Replication Crisis},
  year = {2026}
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

@article{AM2002,
  title={Illiquidity and stock returns: cross-section and time-series effects},
  author={Amihud, Yakov},
  journal={Journal of Financial Markets},
  volume={5},
  number={1},
  pages={31--56},
  year={2002},
  publisher={Elsevier}
}

@article{roll1984simple,
  title={A simple implicit measure of the effective bid-ask spread in an efficient market},
  author={Roll, Richard},
  journal={The Journal of Finance},
  volume={39},
  number={4},
  pages={1127--1139},
  year={1984},
  publisher={Wiley}
}

@article{harris1990statistical,
  title={Statistical properties of the Roll serial covariance bid/ask spread estimator},
  author={Harris, Lawrence},
  journal={The Journal of Finance},
  volume={45},
  number={2},
  pages={579--590},
  year={1990},
  publisher={Wiley Online Library}
}

@article{bao2011illiquidity,
  title={The illiquidity of corporate bonds},
  author={Bao, Jack and Pan, Jun and Wang, Jiang},
  journal={The Journal of Finance},
  volume={66},
  number={3},
  pages={911--946},
  year={2011},
  publisher={Wiley Online Library}
}

@article{dick2012corporate,
  title={Corporate bond liquidity before and after the onset of the subprime crisis},
  author={Dick-Nielsen, Jens and Feldh{\"u}tter, Peter and Lando, David},
  journal={Journal of Financial Economics},
  volume={103},
  number={3},
  pages={471--492},
  year={2012},
  publisher={Elsevier}
}

@article{danyliv2014convenient,
  title={A practical approach to liquidity calculation},
  author={Danyliv, Oleh and Bland, Bruce and Nicholass, Daniel},
  journal={The Journal of Trading},
  volume={9},
  number={3},
  doi={10.3905/jot.2014.9.3.057},
  year={2014},
  publisher={Pageant Media}
}

@article{PS2003,
  title={Liquidity risk and expected stock returns},
  author={P{\'a}stor, L'ubos and Stambaugh, Robert F},
  journal={Journal of Political Economy},
  volume={111},
  number={3},
  pages={642--685},
  year={2003},
  publisher={University of Chicago Press}
}

@article{he2017intermediary,
  title={Intermediary asset pricing: New evidence from many asset classes},
  author={He, Zhiguo and Kelly, Bryan and Manela, Asaf},
  journal={Journal of Financial Economics},
  volume={126},
  number={1},
  pages={1--35},
  year={2017},
  publisher={Elsevier}
}

@article{fama1993common,
  title={Common risk factors in the returns on stocks and bonds},
  author={Fama, Eugene F and French, Kenneth R},
  journal={Journal of Financial Economics},
  volume={33},
  number={1},
  pages={3--56},
  year={1993},
  publisher={Elsevier}
}

@article{choi2025private,
  title={Private Investments of Corporate Bond Mutual Funds},
  author={Choi, Jaewon and Qin, Nan and Zhu, Qifei},
  journal={Available at SSRN 5134832},
  year={2025}
}

@article{baumann2025life,
  title={Life after default: How dealer intermediation improves default recovery},
  author={Baumann, Friedrich and Kakhbod, Ali and Livdan, Dmitry and Nazemi, Abdolreza and Sch{\"u}rhoff, Norman},
  journal={Available at SSRN 4579966},
  year={2025}
}

@article{baumann2025defaulted,
  title={Defaulted Bonds: A Hybrid Asset Priced by Bond and Equity Markets},
  author={Baumann, Friedrich and Nazemi, Abdolreza},
  journal={Available at SSRN 5322649},
  year={2025}
}

@article{Bartram-Grinblatt-Nozawa-2021,
  title={Book-to-market, mispricing, and the cross section of corporate bond returns},
  author={Bartram, S{\"o}hnke M and Grinblatt, Mark and Nozawa, Yoshio},
  journal={Journal of Financial and Quantitative Analysis},
  volume={60},
  number={3},
  pages={1185--1233},
  year={2025},
  publisher={Cambridge University Press}
}

@article{houweling2017factor,
  title={Factor investing in the corporate bond market},
  author={Houweling, Patrick and {Van Zundert}, Jeroen},
  journal={Financial Analysts Journal},
  volume={73},
  number={2},
  pages={100--115},
  year={2017},
  publisher={Taylor \& Francis}
}

@article{KPP2023,
  title={Modeling corporate bond returns},
  author={Kelly, Bryan and Palhares, Diogo and Pruitt, Seth},
  journal={The Journal of Finance},
  volume={78},
  number={4},
  pages={1967--2008},
  year={2023},
  publisher={Wiley Online Library}
}

@article{elkamhi2022one,
  title={A one-factor model of corporate bond premia},
  author={Elkamhi, Redouane and Jo, Chanik and Nozawa, Yoshio},
  journal={Management Science},
  volume={70},
  number={3},
  pages={1875--1900},
  year={2024},
  publisher={INFORMS}
}

@article{IPR2018,
  title={Common factors in corporate bond returns},
  author={Israel, Ronen and Palhares, Diogo and Richardson, Scott},
  journal={Journal of Investment Management},
  volume={16},
  number={2},
  pages={17--46},
  year={2018}
}

@article{gebhardt2005cross,
  title={The cross-section of expected corporate bond returns: Betas or characteristics?},
  author={Gebhardt, William R and Hvidkjaer, Soeren and Swaminathan, Bhaskaran},
  journal={Journal of Financial Economics},
  volume={75},
  number={1},
  pages={85--114},
  year={2005},
  publisher={Elsevier}
}

@article{gebhardt2005stock,
  title={Stock and bond market interaction: {D}oes momentum spill over?},
  author={Gebhardt, William R. and Hvidkjaer, Soeren and Swaminathan, Bhaskaran},
  journal={Journal of Financial Economics},
  volume={75},
  pages={651--690},
  year={2005},
  publisher={Elsevier}
}

@article{novy2012momentum,
  title={Is momentum really momentum?},
  author={Novy-Marx, Robert},
  journal={Journal of Financial Economics},
  volume={103},
  number={3},
  pages={429--453},
  year={2012},
  publisher={Elsevier}
}

@article{blitz2011residual,
  title={Residual momentum},
  author={Blitz, David and Huij, Joop and Martens, Martin},
  journal={Journal of Empirical Finance},
  volume={18},
  number={3},
  pages={506--521},
  year={2011},
  publisher={Elsevier}
}

@unpublished{wang2024industry,
  title={Cross-bond momentum spillovers},
  author={Wang, Junbo and Wu, Dexin and Yang, Lu},
  note={Working Paper},
  year={2024}
}

@article{bali2021long,
  title={Long-term reversals in the corporate bond market},
  author={Bali, Turan G and Subrahmanyam, Avanidhar and Wen, Quan},
  journal={Journal of Financial Economics},
  volume={139},
  number={2},
  pages={656--677},
  year={2021},
  publisher={Elsevier}
}

@unpublished{subrahmanyam2023corporatebonddata,
  title={Corporate bond data projects: Some clarifications},
  author={Subrahmanyam, Avanidhar},
  note={Working Paper},
  year={2023}
}

@article{BaiBaliWen_2019,
  title={Common risk factors in the cross-section of corporate bond returns},
  author={Bai, Jennie and Bali, Turan G and Wen, Quan},
  journal={Journal of Financial Economics},
  volume={131},
  number={3},
  pages={619--642},
  year={2019},
  publisher={Elsevier}
}

@article{hong2000empirical,
  title={An empirical study of bond market transactions},
  author={Hong, Gwangheon and Warga, Arthur},
  journal={Financial Analysts Journal},
  volume={56},
  number={2},
  pages={32--46},
  year={2000},
  publisher={Taylor \& Francis}
}

@article{corwin2012simple,
  title={A simple way to estimate bid-ask spreads from daily high and low prices},
  author={Corwin, Shane A and Schultz, Paul},
  journal={The Journal of Finance},
  volume={67},
  number={2},
  pages={719--760},
  year={2012},
  publisher={Wiley}
}

@article{abdi2017simple,
  title={A simple estimation of bid-ask spreads from daily close, high, and low prices},
  author={Abdi, Farshid and Ranaldo, Angelo},
  journal={The Review of Financial Studies},
  volume={30},
  number={12},
  pages={4437--4480},
  year={2017},
  publisher={Oxford University Press}
}

@article{fong2017,
  title={What are the best liquidity proxies for global research?},
  author={Fong, Kingsley YL and Holden, Craig W and Trzcinka, Charles A},
  journal={Review of Finance},
  volume={21},
  number={4},
  pages={1355--1401},
  year={2017},
  publisher={Oxford University Press}
}

@unpublished{tobek2016,
  title={Liquidity proxies using daily trading volume},
  author={Tobek, Ondrej},
  note={Working Paper},
  year={2016}
}

@article{bollerslev2020,
  title={Good volatility, bad volatility, and the cross section of stock returns},
  author={Bollerslev, Tim and Li, Sophia Zhengzi and Zhao, Bingzhi},
  journal={Journal of Financial and Quantitative Analysis},
  volume={55},
  number={3},
  pages={751--781},
  year={2020},
  publisher={Cambridge University Press}
}

@article{dickerson2023priced,
  title={Priced risk in corporate bonds},
  author={Dickerson, Alexander and Mueller, Philippe and Robotti, Cesare},
  journal={Journal of Financial Economics},
  volume={150},
  number={2},
  pages={103707},
  year={2023},
  publisher={Elsevier}
}

@article{dickerson-bayesian,
  title={The co-pricing factor zoo},
  author={Dickerson, Alexander and Julliard, Christian and Mueller, Philippe},
  journal={Journal of Financial Economics},
  volume={182},
  pages={104295},
  year={2026}
}

@article{Binsbergen-Schwert-Nozawa-2023,
  title={Duration-based valuation of corporate bonds},
  author={{van Binsbergen}, Jules H and Nozawa, Yoshio and Schwert, Michael},
  journal={Review of Financial Studies},
  volume={38},
  number={1},
  pages={158--191},
  year={2025},
  publisher={Oxford University Press}
}

@article{ang2006downside,
  title={Downside risk},
  author={Ang, Andrew and Chen, Joseph and Xing, Yuhang},
  journal={The Review of Financial Studies},
  volume={19},
  number={4},
  pages={1191--1239},
  year={2006},
  publisher={Oxford University Press}
}

@article{ChungWangWu_2019,
  title={Volatility and the cross-section of corporate bond returns},
  author={Chung, Kee H and Wang, Junbo and Wu, Chunchi},
  journal={Journal of Financial Economics},
  volume={133},
  number={2},
  pages={397--417},
  year={2019},
  publisher={Elsevier}
}

@article{LinWangWu_2011,
  title={Liquidity risk and expected corporate bond returns},
  author={Lin, Hai and Wang, Junbo and Wu, Chunchi},
  journal={Journal of Financial Economics},
  volume={99},
  number={3},
  pages={628--650},
  year={2011},
  publisher={Elsevier}
}

@article{harvey2000conditional,
  title={Conditional skewness in asset pricing tests},
  author={Harvey, Campbell R and Siddique, Akhtar},
  journal={The Journal of Finance},
  volume={55},
  number={3},
  pages={1263--1295},
  year={2000},
  publisher={Wiley}
}

@article{bali2021macroeconomic,
  title={The macroeconomic uncertainty premium in the corporate bond market},
  author={Bali, Turan G and Subrahmanyam, Avanidhar and Wen, Quan},
  journal={Journal of Financial and Quantitative Analysis},
  volume={56},
  number={5},
  pages={1653--1678},
  year={2021},
  publisher={Cambridge University Press}
}

@unpublished{ceballos2021inflation,
  title={Inflation volatility risk and the cross-section of corporate bond returns},
  author={Ceballos, Luis},
  note={Working Paper},
  year={2021}
}

@article{koijen2017,
  title={The cross-section and time series of stock and bond returns},
  author={Koijen, Ralph SJ and Lustig, Hanno and {Van Nieuwerburgh}, Stijn},
  journal={Journal of Monetary Economics},
  volume={88},
  pages={50--69},
  year={2017},
  publisher={Elsevier}
}

@article{baker2016measuring,
  title={Measuring economic policy uncertainty},
  author={Baker, Scott R and Bloom, Nick and Davis, Steven J},
  journal={The Quarterly Journal of Economics},
  volume={131},
  number={4},
  pages={1593--1636},
  year={2016},
  publisher={Oxford University Press}
}
""".strip()


# ============================================================================
# SIGNAL DEFINITIONS TABLE
# ============================================================================

SIGNAL_SPEC = Path(__file__).resolve().parent.parent / "stage3" / "spec" / "signal_definitions.json"

# The spec cross-references the paper's timeline figure. This report prints the same figure
# under the same label, so the reference resolves here too.
_MACRO_ESCAPES = {"_": r"\_", "&": r"\&", "%": r"\%"}


def _tex_escape(x: str) -> str:
    """Escape _ & % in plain text and leave LaTeX alone (macros, their braced arguments, math).

    The same rule stage3/s4_zoo/t_ia08.py applies to the same spec, so Table IA.VIII and this
    table print identical text. tests/test_signal_definitions.py holds the two in step.
    """
    out, i, in_math = [], 0, False
    while i < len(x):
        c = x[i]
        if c == "$":
            in_math = not in_math
            out.append(c)
            i += 1
        elif c == "\\":
            j = i + 1
            while j < len(x) and x[j].isalpha():
                j += 1
            if j == i + 1 and j < len(x):          # an escaped character, e.g. \% \& \_
                j += 1
            while j < len(x) and x[j] == "{":      # ...and every braced argument
                depth = 0
                while j < len(x):
                    if x[j] == "{":
                        depth += 1
                    elif x[j] == "}":
                        depth -= 1
                        if depth == 0:
                            j += 1
                            break
                    j += 1
            out.append(x[i:j])
            i = j
        elif in_math:
            out.append(c)
            i += 1
        else:
            out.append(_MACRO_ESCAPES.get(c, c))
            i += 1
    return "".join(out)


def _backticks_to_texttt(x: str) -> str:
    """The spec marks a few column names with backticks. LaTeX wants a texttt macro."""
    import re
    return re.sub(r"`([^`]+)`", lambda m: r"\texttt{" + m.group(1).replace("_", r"\_") + "}", x)


def _cite(row: dict, known: set) -> str:
    """\\citet{...} for a spec row. Refuses a key this report's bibliography lacks.

    A row with words and no key (the five tret_* benchmarks cite an appendix or a working
    paper) prints its words. A row with neither prints '--'.
    """
    keys = row.get("citation_keys") or []
    if not keys:
        return _tex_escape(row.get("citation_text") or "--")
    missing = [k for k in keys if k not in known]
    if missing:
        raise KeyError(
            f"{row['mnemonic']}: the spec cites {missing}, which get_references_bib() does not "
            "define. LaTeX would print a question mark and carry on. Add the entry, copied from "
            "the paper's references.bib, under the same key.")
    return r"\citet{" + ",".join(keys) + "}"


def get_signal_definitions() -> list:
    """The panel's column definitions, grouped for the appendix table.

    ONE source. This used to be a 950-line literal, a third hand-kept copy beside
    DATA_DICTIONARY.md and stage3/spec/signal_definitions.json. It drifted: when five names
    were corrected elsewhere (b_eput is TAX policy, lix is negated, ...) this copy kept the old
    ones, and the published report printed them. It now reads the spec Table IA.VIII is built
    from, so the two tables cannot disagree, and tests/test_signal_definitions.py holds the
    dictionary to the same text.

    Returns a list of {'panel', 'panel_name', 'signals'}; each signal is
    {'mnemonic', 'name', 'description', 'citation'}, LaTeX-ready. `citation` is a citet of the
    spec's own keys, so every cited work reaches this report's reference list. The keys are the
    paper's, and get_references_bib() uses the same ones; a key it lacks is refused here rather
    than printed as a question mark.
    """
    import json
    import re

    if not SIGNAL_SPEC.exists():
        raise FileNotFoundError(
            f"the signal definitions are read from {SIGNAL_SPEC}, which is missing.\n"
            "    It ships in this repository under stage3/spec/. Restore it rather than "
            "typing definitions here: a second copy is how the report came to print stale names.")
    rows = json.loads(SIGNAL_SPEC.read_text(encoding="utf-8"))["rows"]
    known = set(re.findall(r"@\w+\{([^,\s]+),", get_references_bib()))

    panels: list[dict] = []
    for r in rows:
        m = re.match(r"Cluster ([IVX]+): (.*)", r["group"])
        label, name = (m.group(1), m.group(2)) if m else ("", r["group"])
        if not panels or panels[-1]["_group"] != r["group"]:
            panels.append({"panel": label, "panel_name": _tex_escape(name),
                           "_group": r["group"], "signals": []})
        panels[-1]["signals"].append({
            "mnemonic": r["mnemonic"],
            "name": _tex_escape(r["name"]),
            "description": _tex_escape(_backticks_to_texttt(r["description"])),
            "citation": _cite(r, known),
        })
    for p in panels:
        del p["_group"]
    return panels


def build_signal_definitions_table() -> str:
    """
    Build LaTeX longtable for signal definitions.

    Returns
    -------
    str
        LaTeX longtable code.
    """
    panels = get_signal_definitions()

    # Preamble explaining duration-adjusted and price-based signal variants
    preamble = r"""
\subsection{Signal Definitions}

Table~\ref{tab:signal_definitions} provides definitions for all signals in the database.

\paragraph{Return-based signals} (betas, momentum, reversals, VaR, ES): We compute both standard and duration-adjusted versions. The duration-adjusted variant uses $r^x = r - r^{Tsy}$ and is stored in \texttt{betas\_x\_2025.parquet} (for all factor betas) and \texttt{mom\_retx\_2025.parquet} (for momentum and long-term reversal signals).

\paragraph{Price-based signals} (yields, spreads, value, book-to-market, prior 1-month return): All price-based signals in the main panel are market microstructure adjusted (MMN) by default, observed with a minimum 1-business-day gap before the month-end price used for returns. This adjustment is outlined in Section~\ref{sec:signal_gap} and graphically presented in Panel B of Figure~\ref{fig:return_timeline}. Researchers preferring unadjusted signals can download \texttt{mmn\_price\_based\_signals\_2025.parquet} from \href{https://openbondassetpricing.com/}{openbondassetpricing.com}; all variables in this file have the suffix \texttt{\_mmn}. These can be merged to the main panel to use unadjusted signals, but researchers must then use \texttt{ret\_vw\_bgn} to adjust returns, as illustrated in Panel C of Figure~\ref{fig:return_timeline}. The short-term reversal signal (\texttt{str}) can be made duration-adjusted by subtracting \texttt{tret}; this signal is already MMN-adjusted in the main panel.

For all signals requiring a rolling window (betas, VaR, ES), we use a 36-month rolling window with a minimum of 12 observations, denoted 36(12). For long-term reversal signals requiring $>$12 months of history, we use an expanding window that starts at 12-3 and ramps up to the target horizon (e.g., 48-12 or 30-6) to preserve sample coverage.

"""

    # Table header
    table = r"""
\begin{longtable}{>{\ttfamily}p{2.2cm} p{3.8cm} p{7.5cm} p{2.8cm}}
\caption{Signal Definitions and Citations} \label{tab:signal_definitions} \\
\toprule
\normalfont\textbf{Mnemonic} & \textbf{Name} & \textbf{Description} & \textbf{Citation} \\
\midrule
\endfirsthead

\multicolumn{4}{c}{\textit{Table \ref{tab:signal_definitions} continued from previous page}} \\
\toprule
\normalfont\textbf{Mnemonic} & \textbf{Name} & \textbf{Description} & \textbf{Citation} \\
\midrule
\endhead

\midrule
\multicolumn{4}{r}{\textit{Continued on next page}} \\
\endfoot

\bottomrule
\endlastfoot

"""

    # Add panels/clusters
    for panel in panels:
        if panel['panel']:
            # Use "Cluster" for Roman numeral labels (I, II, III, etc.)
            panel_header = (
                f"\\multicolumn{{4}}{{l}}{{\\textbf{{Cluster {panel['panel']}: "
                f"{panel['panel_name']}}}}} \\\\\n\\midrule\n"
            )
        else:
            # Unlabelled section (no "Cluster X:" prefix) - for Bond Identifiers and Bond Characteristics
            panel_header = (
                f"\\multicolumn{{4}}{{l}}{{\\textbf{{{panel['panel_name']}}}}} \\\\\n\\midrule\n"
            )
        table += panel_header

        for sig in panel['signals']:
            # Escape any special characters and format row
            mnemonic = sig['mnemonic'].replace('_', '\\_')
            name = sig['name']
            desc = sig['description']
            cite = sig['citation']

            row = f"{mnemonic} & {name} & {desc} & {cite} \\\\\n"
            table += row

        # Add small vertical space between panels
        table += "\\addlinespace[0.5em]\n"

    table += r"\end{longtable}"

    return preamble + table


# ============================================================================
# TABLE GENERATION HELPERS (PLACEHOLDERS FOR FUTURE IMPLEMENTATION)
# ============================================================================

def create_config_table(config: dict) -> str:
    """
    Create LaTeX table of configuration parameters.

    Parameters
    ----------
    config : dict
        Dictionary of parameter name -> value.

    Returns
    -------
    str
        LaTeX table string.
    """
    # Placeholder - to be implemented
    pass


def create_panel_summary_table(df: pd.DataFrame) -> str:
    """
    Create LaTeX table of panel summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Main panel DataFrame.

    Returns
    -------
    str
        LaTeX table string.
    """
    # Placeholder - to be implemented
    pass


# ============================================================================
# FIGURE GENERATION HELPERS (PLACEHOLDERS FOR FUTURE IMPLEMENTATION)
# ============================================================================

def create_coverage_plot(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "coverage_plot",
    params: PlotParams = None,
) -> Path:
    """
    Create time-series plot of panel coverage.

    Parameters
    ----------
    df : pd.DataFrame
        Main panel DataFrame with 'date' column.
    output_dir : Path
        Directory to save the figure.
    filename : str
        Base filename (extension added based on params.export_format).
    params : PlotParams
        Plotting parameters.

    Returns
    -------
    Path
        Path to saved figure.
    """
    # Placeholder - to be implemented
    pass


def compute_market_returns(
    df: pd.DataFrame,
    date_col: str = 'date',
    ret_end_col: str = 'ret_vw',
    ret_bgn_col: str = 'ret_vw_bgn',
    weight_col: str = 'mcap_s',
) -> pd.DataFrame:
    """
    Compute monthly market returns (value-weighted and equal-weighted).

    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame with returns and weights.
    date_col : str
        Date column name.
    ret_end_col : str
        End-of-month return column.
    ret_bgn_col : str
        Beginning-of-month return column.
    weight_col : str
        Market cap weight column.

    Returns
    -------
    pd.DataFrame
        Monthly market returns with columns:
        - date
        - mktb_end_vw, mktb_bgn_vw (value-weighted)
        - mktb_end_ew, mktb_bgn_ew (equal-weighted)
    """
    # Filter to valid observations (non-nan in all required columns)
    w = df[[date_col, ret_end_col, ret_bgn_col, weight_col]].copy()
    w = w.dropna(subset=[ret_end_col, ret_bgn_col, weight_col])

    # Compute value-weighted returns by month
    def vw_return(g, ret_col, wgt_col):
        weights = g[wgt_col]
        returns = g[ret_col]
        total_weight = weights.sum()
        if total_weight > 0:
            return (returns * weights).sum() / total_weight
        return np.nan

    # Group by month
    monthly = w.groupby(date_col, observed=True).apply(
        lambda g: pd.Series({
            'mktb_end_vw': vw_return(g, ret_end_col, weight_col),
            'mktb_bgn_vw': vw_return(g, ret_bgn_col, weight_col),
            'mktb_end_ew': g[ret_end_col].mean(),
            'mktb_bgn_ew': g[ret_bgn_col].mean(),
            'n_bonds': len(g),
        }),
        include_groups=False,
    ).reset_index()

    return monthly


def run_regression_nw(
    y: np.ndarray,
    x: np.ndarray,
    lag_rule: str = 'T025',
) -> dict:
    """
    Run OLS regression with Newey-West standard errors.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable.
    x : np.ndarray
        Independent variable.
    lag_rule : str
        Lag selection rule: 'T025' for T^0.25.

    Returns
    -------
    dict
        Dictionary with keys: alpha, beta, alpha_tstat (H0: alpha=0),
        beta_tstat (H0: beta=1), r2, rho (correlation)
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    # Remove any NaN pairs
    valid = ~(np.isnan(y) | np.isnan(x))
    y_clean = y[valid]
    x_clean = x[valid]

    T = len(y_clean)
    if T < 3:
        return {'alpha': np.nan, 'beta': np.nan,
                'alpha_tstat': np.nan, 'beta_tstat': np.nan,
                'r2': np.nan, 'rho': np.nan}

    # Determine lag
    if lag_rule == 'T025':
        n_lags = int(np.floor(T ** 0.25))
    else:
        n_lags = 0

    # Run OLS with Newey-West
    X = add_constant(x_clean)
    model = OLS(y_clean, X).fit(cov_type='HAC', cov_kwds={'maxlags': n_lags})

    # t-stat for alpha: H0: alpha = 0
    alpha_tstat = model.tvalues[0]

    # t-stat for beta: H0: beta = 1 (not zero)
    beta = model.params[1]
    beta_se = model.bse[1]
    beta_tstat = (beta - 1.0) / beta_se

    # Correlation coefficient
    rho = np.corrcoef(x_clean, y_clean)[0, 1]

    return {
        'alpha': model.params[0],
        'beta': beta,
        'alpha_tstat': alpha_tstat,
        'beta_tstat': beta_tstat,
        'r2': model.rsquared,
        'rho': rho,
        'n_obs': T,
        'n_lags': n_lags,
    }


def paired_tstat_means(
    y: np.ndarray,
    x: np.ndarray,
    lag_rule: str = 'T025',
) -> float:
    """
    Compute t-statistic for equality of means (paired) with Newey-West SE.

    H0: mean(y) = mean(x)

    Equivalent to regressing diff = y - x on a constant and testing H0: alpha = 0
    using HAC (Newey-West) standard errors.

    Parameters
    ----------
    y : np.ndarray
        First series.
    x : np.ndarray
        Second series (same length as y).
    lag_rule : str
        Lag selection rule: 'T025' for int(T^0.25), 'none' for no adjustment.

    Returns
    -------
    float
        t-statistic for the paired difference (Newey-West adjusted).
    """
    from statsmodels.regression.linear_model import OLS

    # Convert to float64 numpy arrays (handles pandas nullable floats)
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    valid = ~(np.isnan(y) | np.isnan(x))
    y_c = y[valid]
    x_c = x[valid]
    T = len(y_c)
    if T < 2:
        return np.nan

    diff = y_c - x_c

    # Determine lag
    if lag_rule == 'T025':
        n_lags = int(np.floor(T ** 0.25))
    else:
        n_lags = 0

    # Regress diff on constant, test H0: alpha = 0
    X = np.ones((T, 1))  # Just constant
    model = OLS(diff, X).fit(cov_type='HAC', cov_kwds={'maxlags': n_lags})

    return model.tvalues[0]


def create_market_return_scatter(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "market_return_scatter",
    params: PlotParams = None,
    date_col: str = 'date',
    ret_end_col: str = 'ret_vw',
    ret_bgn_col: str = 'ret_vw_bgn',
    weight_col: str = 'mcap_s',
) -> tuple:
    """
    Create scatter plot of market returns: MKTB_End vs MKTB_Bgn.

    Panel A: Value-weighted returns
    Panel B: Equal-weighted returns

    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame with returns and weights.
    output_dir : Path
        Directory to save the figure.
    filename : str
        Base filename (extension added based on params.export_format).
    params : PlotParams
        Plotting parameters.
    date_col : str
        Date column name.
    ret_end_col : str
        End-of-month return column.
    ret_bgn_col : str
        Beginning-of-month return column.
    weight_col : str
        Market cap weight column.

    Returns
    -------
    tuple
        (Path to saved figure, caption text).
    """
    if params is None:
        params = PlotParams()

    apply_plot_params(params)

    # Compute market returns
    mkt = compute_market_returns(
        df, date_col=date_col,
        ret_end_col=ret_end_col, ret_bgn_col=ret_bgn_col,
        weight_col=weight_col,
    )

    # Scale to percentage
    for col in ['mktb_end_vw', 'mktb_bgn_vw', 'mktb_end_ew', 'mktb_bgn_ew']:
        mkt[col] = mkt[col] * 100

    # Run regressions
    reg_vw = run_regression_nw(
        y=mkt['mktb_end_vw'].values,
        x=mkt['mktb_bgn_vw'].values,
    )
    reg_ew = run_regression_nw(
        y=mkt['mktb_end_ew'].values,
        x=mkt['mktb_bgn_ew'].values,
    )

    # t-stat for equality of means (paired)
    tstat_vw = paired_tstat_means(mkt['mktb_end_vw'].values, mkt['mktb_bgn_vw'].values)
    tstat_ew = paired_tstat_means(mkt['mktb_end_ew'].values, mkt['mktb_bgn_ew'].values)

    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: Value-weighted
    ax = axes[0]
    x_vw = mkt['mktb_bgn_vw'].values
    y_vw = mkt['mktb_end_vw'].values

    ax.scatter(x_vw, y_vw, s=20, alpha=0.6, color='steelblue', edgecolor='none')

    # Add regression line
    x_line = np.linspace(np.nanmin(x_vw), np.nanmax(x_vw), 100)
    y_line = reg_vw['alpha'] + reg_vw['beta'] * x_line
    ax.plot(x_line, y_line, color='darkred', lw=1.5, ls='--')

    # Annotation with t-stat for equality of means
    annot_text = (
        f"$\\alpha$ = {reg_vw['alpha']:.3f} ({reg_vw['alpha_tstat']:.2f})\n"
        f"$\\beta$ = {reg_vw['beta']:.3f} ({reg_vw['beta_tstat']:.2f})\n"
        f"$R^2$ = {reg_vw['r2']:.3f}\n"
        f"$\\rho$ = {reg_vw['rho']:.3f}\n"
        f"$t_{{\\mu}}$ = {tstat_vw:.2f}"
    )
    ax.text(0.05, 0.95, annot_text, transform=ax.transAxes,
            fontsize=params.legend_size, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_xlabel(r'$\mathrm{MKTB}_{\mathrm{Bgn}}$ (%)')
    ax.set_ylabel(r'$\mathrm{MKTB}_{\mathrm{End}}$ (%)')
    ax.set_title('Panel A: Value-Weighted')
    ax.axhline(0, color='gray', lw=0.5, ls='-')
    ax.axvline(0, color='gray', lw=0.5, ls='-')
    ax.grid(True, alpha=params.grid_alpha, lw=params.grid_lw)

    # Panel B: Equal-weighted
    ax = axes[1]
    x_ew = mkt['mktb_bgn_ew'].values
    y_ew = mkt['mktb_end_ew'].values

    ax.scatter(x_ew, y_ew, s=20, alpha=0.6, color='steelblue', edgecolor='none')

    # Add regression line
    x_line = np.linspace(np.nanmin(x_ew), np.nanmax(x_ew), 100)
    y_line = reg_ew['alpha'] + reg_ew['beta'] * x_line
    ax.plot(x_line, y_line, color='darkred', lw=1.5, ls='--')

    # Annotation with t-stat for equality of means
    annot_text = (
        f"$\\alpha$ = {reg_ew['alpha']:.3f} ({reg_ew['alpha_tstat']:.2f})\n"
        f"$\\beta$ = {reg_ew['beta']:.3f} ({reg_ew['beta_tstat']:.2f})\n"
        f"$R^2$ = {reg_ew['r2']:.3f}\n"
        f"$\\rho$ = {reg_ew['rho']:.3f}\n"
        f"$t_{{\\mu}}$ = {tstat_ew:.2f}"
    )
    ax.text(0.05, 0.95, annot_text, transform=ax.transAxes,
            fontsize=params.legend_size, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_xlabel(r'$\mathrm{MKTB}_{\mathrm{Bgn}}$ (%)')
    ax.set_ylabel(r'$\mathrm{MKTB}_{\mathrm{End}}$ (%)')
    ax.set_title('Panel B: Equal-Weighted')
    ax.axhline(0, color='gray', lw=0.5, ls='-')
    ax.axvline(0, color='gray', lw=0.5, ls='-')
    ax.grid(True, alpha=params.grid_alpha, lw=params.grid_lw)

    plt.tight_layout()

    # Save figure
    ext = params.export_format
    fig_path = output_dir / f"{filename}.{ext}"
    fig.savefig(fig_path, dpi=params.figure_dpi, transparent=params.transparent,
                bbox_inches='tight')
    plt.close(fig)

    # Caption text
    caption = (
        r"Scatter plot of monthly bond market returns. "
        r"Panel A shows value-weighted returns using market capitalization weights (mcap\_s). "
        r"Panel B shows equal-weighted returns. "
        r"The $x$-axis is $\mathrm{MKTB}_{\mathrm{Bgn}}$ (beginning-of-month pricing) "
        r"and $y$-axis is $\mathrm{MKTB}_{\mathrm{End}}$ (end-of-month pricing). "
        r"All $t$-statistics use Newey-West standard errors (lags = $\lfloor T^{0.25} \rfloor$). "
        r"The $t$-statistic for $\alpha$ tests $H_0$: $\alpha = 0$; "
        r"the $t$-statistic for $\beta$ tests $H_0$: $\beta = 1$; "
        r"$t_{\mu}$ tests $H_0$: $\mu_{\mathrm{End}} = \mu_{\mathrm{Bgn}}$."
    )

    return fig_path, caption


def create_dynamics_of_default_plot(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "dynamics_of_default",
    params: PlotParams = None,
) -> tuple:
    """
    Create 2x1 plot showing dynamics of defaulted bonds over time.

    Panel A: Count of Defaulted Bonds (monthly)
    Panel B: Defaulted Bonds (%) - percentage of total bonds

    A bond is identified as defaulted if EITHER spc_rat == 22 OR mdc_rat == 22.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'date', 'cusip', 'spc_rat', 'mdc_rat'
    output_dir : Path
        Directory to save the figure
    filename : str
        Base filename (extension will be added)
    params : PlotParams
        Plotting parameters

    Returns
    -------
    tuple
        (Path to saved figure, caption text).
    """
    from matplotlib.dates import AutoDateLocator, DateFormatter
    from matplotlib.ticker import FormatStrFormatter
    import gc

    if params is None:
        params = PlotParams()

    apply_plot_params(params)

    # Check required columns
    required_cols = ['cusip', 'date', 'spc_rat', 'mdc_rat']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Prepare data
    df_temp = df[['cusip', 'date', 'spc_rat', 'mdc_rat']].copy()
    df_temp = df_temp.sort_values(['cusip', 'date'])

    # Step 1: Get bond lifespans for ALL bonds
    bond_life = df_temp.groupby('cusip', observed=True)['date'].agg(['min', 'max']).reset_index()
    bond_life.columns = ['cusip', 'first_trade', 'last_trade']

    # Step 2: For each bond, find first default date (first date where spc_rat == 22 OR mdc_rat == 22)
    defaulted = df_temp[(df_temp['spc_rat'] == 22) | (df_temp['mdc_rat'] == 22)].copy()
    first_default = defaulted.groupby('cusip', observed=True)['date'].min().reset_index()
    first_default.columns = ['cusip', 'first_default_date']
    del defaulted
    gc.collect()

    # Step 3: For each defaulted bond, find first upgrade date (vectorized)
    df_with_default = df_temp.merge(first_default, on='cusip', how='inner')
    del df_temp
    gc.collect()

    after_default = df_with_default[df_with_default['date'] > df_with_default['first_default_date']]
    # Upgraded if both ratings are < 22 (or NA)
    upgrades = after_default[
        (after_default['spc_rat'] < 22) & (after_default['mdc_rat'] < 22) |
        (after_default['spc_rat'].isna() & (after_default['mdc_rat'] < 22)) |
        ((after_default['spc_rat'] < 22) & after_default['mdc_rat'].isna())
    ]
    del df_with_default, after_default
    gc.collect()

    if len(upgrades) > 0:
        first_upgrade = upgrades.groupby('cusip', observed=True)['date'].min().reset_index()
        first_upgrade.columns = ['cusip', 'first_upgrade_date']
    else:
        first_upgrade = pd.DataFrame(columns=['cusip', 'first_upgrade_date'])
    del upgrades
    gc.collect()

    # Step 4: Only keep bonds that were ever defaulted
    defaulted_cusips = first_default['cusip'].unique()
    bond_info_defaulted = bond_life[bond_life['cusip'].isin(defaulted_cusips)].copy()

    # Merge default and upgrade dates only for defaulted bonds
    bond_info_defaulted = bond_info_defaulted.merge(first_default, on='cusip', how='left')
    if len(first_upgrade) > 0:
        bond_info_defaulted = bond_info_defaulted.merge(first_upgrade, on='cusip', how='left')
    else:
        bond_info_defaulted['first_upgrade_date'] = pd.NaT
    del first_default, first_upgrade
    gc.collect()

    # Step 5: Get all unique months
    all_dates = pd.date_range(
        start=bond_life['first_trade'].min(),
        end=bond_life['last_trade'].max(),
        freq='ME'  # Month-end frequency
    )

    # Step 6: Cross-join ONLY for defaulted bonds
    df_dates = pd.DataFrame({'date': all_dates})
    df_dates['_key'] = 1
    bond_info_temp = bond_info_defaulted.copy()
    bond_info_temp['_key'] = 1

    df_expanded_defaulted = df_dates.merge(bond_info_temp, on='_key').drop('_key', axis=1)
    del bond_info_temp, bond_info_defaulted
    gc.collect()

    # Filter to alive defaulted bonds
    alive_mask = (
        (df_expanded_defaulted['first_trade'] <= df_expanded_defaulted['date']) &
        (df_expanded_defaulted['last_trade'] >= df_expanded_defaulted['date'])
    )
    df_alive_defaulted = df_expanded_defaulted[alive_mask].copy()
    del df_expanded_defaulted, alive_mask
    gc.collect()

    # Apply defaulted mask (currently defaulted, not yet upgraded)
    defaulted_mask = (
        df_alive_defaulted['first_default_date'].notna() &
        (df_alive_defaulted['first_default_date'] <= df_alive_defaulted['date']) &
        (df_alive_defaulted['first_upgrade_date'].isna() |
         (df_alive_defaulted['date'] < df_alive_defaulted['first_upgrade_date']))
    )
    df_alive_defaulted = df_alive_defaulted[defaulted_mask].copy()
    del defaulted_mask
    gc.collect()

    # Count defaulted bonds per month
    df_defaulted_monthly = df_alive_defaulted.groupby('date').size().reset_index(name='defaulted_bonds')
    del df_alive_defaulted
    gc.collect()

    # Step 7: Count total alive bonds per month
    all_dates_list = list(all_dates)
    total_results = []

    bond_life_np = bond_life[['cusip', 'first_trade', 'last_trade']].copy()

    for date in all_dates_list:
        alive_count = (
            (bond_life_np['first_trade'] <= date) &
            (bond_life_np['last_trade'] >= date)
        ).sum()
        total_results.append({'date': date, 'total_bonds': alive_count})

    df_total_monthly = pd.DataFrame(total_results)
    del bond_life_np, total_results, all_dates_list, bond_life
    gc.collect()

    # Merge total and defaulted counts
    df_monthly = df_total_monthly.merge(df_defaulted_monthly, on='date', how='left')
    df_monthly['defaulted_bonds'] = df_monthly['defaulted_bonds'].fillna(0).astype(int)
    del df_total_monthly, df_defaulted_monthly
    gc.collect()

    # Compute percentage
    df_monthly['defaulted_pct'] = (df_monthly['defaulted_bonds'] / df_monthly['total_bonds']) * 100

    # A4 portrait dimensions - adjust for 2x1 layout
    fig_w, fig_h = 8.27, 5.85

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Grid margins for 2x1
    gs = fig.add_gridspec(
        nrows=2, ncols=1,
        left=0.10, right=0.95, bottom=0.10, top=0.93,
        hspace=0.18
    )

    axes = gs.subplots().ravel()

    # Panel A: Count of Defaulted Bonds
    ax = axes[0]
    dates = df_monthly['date']
    values = df_monthly['defaulted_bonds']

    ax.plot(
        dates, values,
        color=params.line_color,
        alpha=params.line_alpha,
        lw=params.line_lw,
    )

    ax.set_title('A: Count of Defaulted Bonds', pad=2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, alpha=params.grid_alpha, linewidth=params.grid_lw)

    # Format date axis
    locator = AutoDateLocator(minticks=3, maxticks=8, interval_multiples=True)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelsize=params.tick_size, pad=1)
    ax.tick_params(axis="y", labelsize=params.tick_size)
    ax.margins(x=0.01)

    # Panel B: Defaulted Bonds (%)
    ax = axes[1]
    values = df_monthly['defaulted_pct']

    ax.plot(
        dates, values,
        color=params.line_color,
        alpha=params.line_alpha,
        lw=params.line_lw,
    )

    ax.set_title('B: Defaulted Bonds (%)', pad=2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, alpha=params.grid_alpha, linewidth=params.grid_lw)

    # Format y-axis with 1 decimal place
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Format date axis
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelsize=params.tick_size, pad=1)
    ax.tick_params(axis="y", labelsize=params.tick_size)
    ax.margins(x=0.01)

    # Save figure
    ext = params.export_format.lower()
    if ext not in ["pdf", "png", "jpg", "jpeg"]:
        ext = "pdf"

    out_path = output_dir / f"{filename}.{ext}"

    savefig_kwargs = dict(
        format=ext,
        bbox_inches='tight',
        facecolor="white",
        edgecolor="none",
        transparent=params.transparent,
    )

    if ext in ("png", "jpg", "jpeg"):
        savefig_kwargs["dpi"] = params.figure_dpi

    if ext == "png":
        savefig_kwargs["pil_kwargs"] = {"optimize": True, "compress_level": 6}
    elif ext in ("jpg", "jpeg"):
        savefig_kwargs["pil_kwargs"] = {"quality": 85, "optimize": True, "progressive": True}

    fig.savefig(out_path, **savefig_kwargs)
    plt.close(fig)

    # Clean up memory before return
    del df_monthly
    gc.collect()

    # Caption text
    caption = (
        r"Dynamics of defaulted bonds over time. "
        r"A bond is identified as defaulted if either the S\&P composite rating (\texttt{spc\_rat}) "
        r"or the Moody's composite rating (\texttt{mdc\_rat}) equals 22 (D - Default). "
        r"Panel A shows the count of defaulted bonds each month. "
        r"Panel B shows the percentage of defaulted bonds relative to the total number of bonds."
    )

    return out_path, caption


def create_dynamics_of_144a_plot(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "dynamics_of_144a",
    params: PlotParams = None,
    ret_col: str = 'ret_vw',
    rule_144a_col: str = '144a',
) -> tuple:
    """
    Create 2x1 plot showing dynamics of 144a bonds over time.

    Panel A: Count of 144a Bonds (monthly)
    Panel B: 144a Bonds (%) - percentage of total bonds

    Uses non-nan ret_vw observations.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'date', 'cusip', ret_col, rule_144a_col
    output_dir : Path
        Directory to save the figure
    filename : str
        Base filename (extension will be added)
    params : PlotParams
        Plotting parameters
    ret_col : str
        Return column to use for filtering (default: 'ret_vw')
    rule_144a_col : str
        Column name for 144a indicator (default: '144a')

    Returns
    -------
    tuple
        (Path to saved figure, caption text).
    """
    from matplotlib.dates import AutoDateLocator, DateFormatter
    from matplotlib.ticker import FormatStrFormatter
    import gc

    if params is None:
        params = PlotParams()

    apply_plot_params(params)

    # Check required columns
    required_cols = ['cusip', 'date', ret_col, rule_144a_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Filter to non-nan returns
    df_valid = df[df[ret_col].notna()].copy()

    # Count 144a and total by date (vectorized, no lambda)
    df_valid['_is_144a'] = (df_valid[rule_144a_col] == 1).astype('int8')
    monthly = df_valid.groupby('date', observed=True).agg(
        n_144a=('_is_144a', 'sum'),
        n_total=('_is_144a', 'size'),
    ).reset_index()
    monthly['pct_144a'] = (monthly['n_144a'] / monthly['n_total']) * 100

    del df_valid
    gc.collect()

    # A4 portrait dimensions - adjust for 2x1 layout
    fig_w, fig_h = 8.27, 5.85

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Grid margins for 2x1
    gs = fig.add_gridspec(
        nrows=2, ncols=1,
        left=0.10, right=0.95, bottom=0.10, top=0.93,
        hspace=0.18
    )

    axes = gs.subplots().ravel()

    # Panel A: Count of 144a Bonds
    ax = axes[0]
    dates = monthly['date']
    values = monthly['n_144a']

    ax.plot(
        dates, values,
        color=params.line_color,
        alpha=params.line_alpha,
        lw=params.line_lw,
    )

    ax.set_title('A: Count of 144a Bonds', pad=2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, alpha=params.grid_alpha, linewidth=params.grid_lw)

    # Format date axis
    locator = AutoDateLocator(minticks=3, maxticks=8, interval_multiples=True)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelsize=params.tick_size, pad=1)
    ax.tick_params(axis="y", labelsize=params.tick_size)
    ax.margins(x=0.01)

    # Panel B: 144a Bonds (%)
    ax = axes[1]
    values = monthly['pct_144a']

    ax.plot(
        dates, values,
        color=params.line_color,
        alpha=params.line_alpha,
        lw=params.line_lw,
    )

    ax.set_title('B: 144a Bonds (%)', pad=2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, alpha=params.grid_alpha, linewidth=params.grid_lw)

    # Format y-axis with 1 decimal place
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Format date axis
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelsize=params.tick_size, pad=1)
    ax.tick_params(axis="y", labelsize=params.tick_size)
    ax.margins(x=0.01)

    # Save figure
    ext = params.export_format.lower()
    if ext not in ["pdf", "png", "jpg", "jpeg"]:
        ext = "pdf"

    out_path = output_dir / f"{filename}.{ext}"

    savefig_kwargs = dict(
        format=ext,
        bbox_inches='tight',
        facecolor="white",
        edgecolor="none",
        transparent=params.transparent,
    )

    if ext in ("png", "jpg", "jpeg"):
        savefig_kwargs["dpi"] = params.figure_dpi

    if ext == "png":
        savefig_kwargs["pil_kwargs"] = {"optimize": True, "compress_level": 6}
    elif ext in ("jpg", "jpeg"):
        savefig_kwargs["pil_kwargs"] = {"quality": 85, "optimize": True, "progressive": True}

    fig.savefig(out_path, **savefig_kwargs)
    plt.close(fig)

    # Clean up memory before return
    del monthly
    gc.collect()

    # Caption text
    caption = (
        r"Dynamics of Rule 144a bonds over time. "
        r"Uses bond-month observations with non-missing month-end returns (\texttt{ret\_vw}). "
        r"Panel A shows the count of 144a bonds each month. "
        r"Panel B shows the percentage of 144a bonds relative to the total number of bonds."
    )

    return out_path, caption


def build_month_end_panel(
    df: pd.DataFrame,
    id_col: str = "cusip",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Build a contiguous month-end panel for each security between its first and last observed month,
    then left-merge the original data onto that skeleton.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least `id_col` and `date_col`.
    id_col : str, default "cusip_id"
        Column identifying each bond/security.
    date_col : str, default "date"
        Date column (any dtype convertible to datetime).

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with one row per (id_col, month-end) between first and last observation,
        original columns merged in, sorted by [id_col, date_col].
    """
    if df.empty:
        return df.copy()

    # Work on a copy
    out = df.copy()

    # Ensure datetime and snap to month-end
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce") + MonthEnd(0)
    out = out[out[date_col].notna()].copy()

    # First/last observed month per id
    bounds = (
        out.groupby(id_col, as_index=False, observed=True)[date_col]
           .agg(first_month="min", last_month="max")
    )

    # Build month-end skeleton for each id
    skeleton_frames = []
    for row in bounds.itertuples(index=False):
        id_val: Union[str, int] = getattr(row, id_col)
        all_months = pd.date_range(
            start=row.first_month, end=row.last_month, freq=MonthEnd(1)
        )
        skeleton_frames.append(pd.DataFrame({id_col: id_val, date_col: all_months}))

    skeleton = pd.concat(skeleton_frames, ignore_index=True)

    # Merge original data back onto the skeleton
    resampled = (
        skeleton.merge(out, on=[id_col, date_col], how="left")
                .sort_values([id_col, date_col], kind="stable")
                .reset_index(drop=True)
    )
    return resampled


# ============================================================================
# DATA AVAILABILITY TABLE
# ============================================================================

def escape_latex(s: str) -> str:
    """Escape special LaTeX characters in a string."""
    if not isinstance(s, str):
        return str(s)
    replacements = [
        ('&', r'\&'),
        ('%', r'\%'),
        ('$', r'\$'),
        ('#', r'\#'),
        ('_', r'\_'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('~', r'\textasciitilde{}'),
        ('^', r'\textasciicircum{}'),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s


def make_data_availability_table(
    df: pd.DataFrame,
    min_date: str,
    max_date: str,
    variables: list = None,
    id_col: str = "cusip",
    date_col: str = "date",
    rating_col: str = "spc_rat",
) -> str:
    """
    Generate Data Availability table by Rating Category for monthly panel.

    Shows the number of observations and % missing for key variables across
    four rating categories: All Bonds, Investment Grade, Non-Investment Grade,
    and Defaulted bonds.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly panel DataFrame (already resampled via build_month_end_panel).
    min_date : str
        Minimum date in data (YYYY-MM-DD).
    max_date : str
        Maximum date in data (YYYY-MM-DD).
    variables : list of tuples, optional
        List of (column_name, display_name) tuples.
    id_col : str
        Bond identifier column.
    date_col : str
        Date column.
    rating_col : str
        Rating column for splits (default: 'spc_rat').

    Returns
    -------
    str
        LaTeX table code.
    """
    # Default variables
    if variables is None:
        variables = [
            ('pr', 'Price (VW)'),
            ('ret_vw', 'Month-End Return'),
            ('ret_vw_bgn', 'Month-Begin Return'),
            ('ytm', 'YTM'),
            ('cs', 'Spread'),
            ('spc_rat', 'Composite Rating (SP)'),
            ('mdc_rat', 'Composite Rating (MD)'),
            ('permno', 'PERMNO'),
        ]

    # Define rating categories
    df_all = df
    df_ig = df[(df[rating_col] >= 1) & (df[rating_col] <= 10)]
    df_nig = df[(df[rating_col] > 10) & (df[rating_col] <= 21)]
    df_def = df[df[rating_col] == 22]

    def format_with_commas(n):
        return f"{int(n):,}"

    def compute_stats(df_panel, vars_list):
        total_rows = len(df_panel)
        stats = []
        for var_name, display_name in vars_list:
            if var_name in df_panel.columns:
                non_null_count = df_panel[var_name].notna().sum()
                null_count = df_panel[var_name].isna().sum()
                pct_missing = (null_count / total_rows * 100) if total_rows > 0 else 0.0
            else:
                non_null_count = 0
                pct_missing = 100.0
            stats.append({
                'variable': display_name,
                'observations': non_null_count,
                'pct_missing': pct_missing
            })
        return stats

    stats_all = compute_stats(df_all, variables)
    stats_ig = compute_stats(df_ig, variables)
    stats_nig = compute_stats(df_nig, variables)
    stats_def = compute_stats(df_def, variables)

    # Build LaTeX table rows
    rows = []

    # Add Total row first
    total_row = (f"Total & {format_with_commas(len(df_all))} & -- & "
                 f"{format_with_commas(len(df_ig))} & -- & "
                 f"{format_with_commas(len(df_nig))} & -- & "
                 f"{format_with_commas(len(df_def))} & -- " + r"\\")
    rows.append(total_row)
    rows.append(r"\midrule")

    for i in range(len(variables)):
        var_name = escape_latex(stats_all[i]['variable'])

        obs_all = format_with_commas(stats_all[i]['observations'])
        pct_all = f"{stats_all[i]['pct_missing']:.2f}"

        obs_ig = format_with_commas(stats_ig[i]['observations'])
        pct_ig = f"{stats_ig[i]['pct_missing']:.2f}"

        obs_nig = format_with_commas(stats_nig[i]['observations'])
        pct_nig = f"{stats_nig[i]['pct_missing']:.2f}"

        obs_def = format_with_commas(stats_def[i]['observations'])
        pct_def = f"{stats_def[i]['pct_missing']:.2f}"

        row = (f"{var_name} & {obs_all} & {pct_all} & "
               f"{obs_ig} & {pct_ig} & "
               f"{obs_nig} & {pct_nig} & "
               f"{obs_def} & {pct_def} " + r"\\")
        rows.append(row)

    rows_tex = "\n".join(rows)

    note_text = (
        r"This table reports data availability for key variables in the monthly panel across rating categories. "
        r"For each panel, we report the number of non-missing observations and the "
        r"percentage of missing values after resampling each bond to a contiguous monthly time-series. "
        r"Panel A includes all bonds in the sample. "
        r"Panel B includes investment grade bonds (S\&P ratings 1--10, AAA to BBB$-$). "
        r"Panel C includes non-investment grade bonds (S\&P ratings 11--21, BB+ to CCC$-$). "
        r"Panel D includes defaulted bonds (S\&P rating 22, D). "
        r"The sample spans the period " + min_date + r" to " + max_date + r"."
    )

    latex = r"""
\begin{table}[!ht]
\begin{center}
\footnotesize
\caption{Monthly Panel Data Availability by Rating Category}
\label{tab:monthly_data_availability}\vspace{2mm}
\scalebox{0.85}{%
\begin{tabular}{lrrrrrrrr}
\toprule
& \multicolumn{2}{c}{\textbf{Panel A: All}} & \multicolumn{2}{c}{\textbf{Panel B: Inv. Grade}} & \multicolumn{2}{c}{\textbf{Panel C: Non-Inv. Grade}} & \multicolumn{2}{c}{\textbf{Panel D: Defaulted}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
Variable & Obs. & \% Missing & Obs. & \% Missing & Obs. & \% Missing & Obs. & \% Missing \\
\midrule
""" + rows_tex + r"""
\bottomrule
\end{tabular}
}
\end{center}
\begin{spacing}{1}
\footnotesize{
""" + note_text + r"""
}
\end{spacing}
\vspace{-2mm}
\end{table}
""".strip()

    return latex


# ============================================================================
# DESCRIPTIVE STATISTICS FUNCTIONS
# ============================================================================

def compute_pooled_stats(
    df: pd.DataFrame,
    stat_vars: list,
    scale_vars: dict = None,
) -> pd.DataFrame:
    """
    Compute pooled descriptive statistics (Panel A) for monthly panel.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly panel DataFrame.
    stat_vars : list of tuples
        List of (column_name, display_name) tuples.
    scale_vars : dict, optional
        Dictionary of {column_name: scale_factor} for scaling (e.g., {'ytm': 100}).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Variable, Mean, Median, SD, P1, P5, P95, P99.
    """
    if scale_vars is None:
        scale_vars = {
            'ret_vw': 100, 'ret_vw_bgn': 100,  # Returns in percent
            'ytm': 100, 'cs': 100,  # Yields/spreads in percent
        }

    stats_list = []

    for var_name, label in stat_vars:
        if var_name not in df.columns:
            continue

        series = df[var_name].dropna()
        if len(series) == 0:
            continue

        # Scale if needed
        scale = scale_vars.get(var_name, 1)
        if scale != 1:
            series = series * scale

        stats = {
            'Variable': label,
            'Mean': round(series.mean(), 2),
            'Median': round(series.median(), 2),
            'SD': round(series.std(), 2),
            'P1': round(series.quantile(0.01), 2),
            'P5': round(series.quantile(0.05), 2),
            'P95': round(series.quantile(0.95), 2),
            'P99': round(series.quantile(0.99), 2),
        }
        stats_list.append(stats)

    return pd.DataFrame(stats_list)


def compute_cross_sectional_stats(
    df: pd.DataFrame,
    stat_vars: list,
    date_col: str = "date",
    scale_vars: dict = None,
) -> pd.DataFrame:
    """
    Compute cross-sectional statistics (Panel B) - time-series averages of monthly stats.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly panel DataFrame.
    stat_vars : list of tuples
        List of (column_name, display_name) tuples.
    date_col : str
        Date column for grouping.
    scale_vars : dict, optional
        Dictionary of {column_name: scale_factor} for scaling.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Variable, Mean, Median, SD, P1, P5, P95, P99.
    """
    if scale_vars is None:
        scale_vars = {
            'ret_vw': 100, 'ret_vw_bgn': 100,  # Returns in percent
            'ytm': 100, 'cs': 100,  # Yields/spreads in percent
        }

    stats_list = []

    for var_name, label in stat_vars:
        if var_name not in df.columns:
            continue

        # Compute monthly statistics using groupby with named agg
        g = df.groupby(date_col, observed=True)[var_name]

        monthly_mean = g.mean()
        monthly_median = g.median()
        monthly_std = g.std()
        monthly_p1 = g.quantile(0.01)
        monthly_p5 = g.quantile(0.05)
        monthly_p95 = g.quantile(0.95)
        monthly_p99 = g.quantile(0.99)

        # Time-series average of monthly stats
        stats = {
            'Variable': label,
            'Mean': monthly_mean.mean(),
            'Median': monthly_median.mean(),
            'SD': monthly_std.mean(),
            'P1': monthly_p1.mean(),
            'P5': monthly_p5.mean(),
            'P95': monthly_p95.mean(),
            'P99': monthly_p99.mean(),
        }

        # Scale AFTER computation
        scale = scale_vars.get(var_name, 1)
        if scale != 1:
            for key in ['Mean', 'Median', 'SD', 'P1', 'P5', 'P95', 'P99']:
                stats[key] = stats[key] * scale

        # Round to 2 decimals
        for key in ['Mean', 'Median', 'SD', 'P1', 'P5', 'P95', 'P99']:
            stats[key] = round(stats[key], 2)

        stats_list.append(stats)

    return pd.DataFrame(stats_list)


def make_descriptive_stats_table(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    min_date: str,
    max_date: str,
    title: str = "Monthly Panel Descriptive Statistics",
    label: str = "tab:monthly_descriptive_stats",
) -> str:
    """
    Generate descriptive statistics table for monthly panel.

    Parameters
    ----------
    panel_a : pd.DataFrame
        Pooled statistics from compute_pooled_stats().
    panel_b : pd.DataFrame
        Cross-sectional statistics from compute_cross_sectional_stats().
    min_date : str
        Minimum date in data (YYYY-MM-DD).
    max_date : str
        Maximum date in data (YYYY-MM-DD).
    title : str
        Table title.
    label : str
        LaTeX label.

    Returns
    -------
    str
        LaTeX table code.
    """
    def format_number(val: float) -> str:
        """Format number with proper LaTeX minus sign for negatives."""
        if val < 0:
            return f"$-${abs(val):.2f}"
        return f"{val:.2f}"

    def format_stats_df(df: pd.DataFrame) -> str:
        rows = []
        for _, row in df.iterrows():
            var = escape_latex(row['Variable'])
            vals = [
                format_number(row['Mean']),
                format_number(row['Median']),
                format_number(row['SD']),
                format_number(row['P1']),
                format_number(row['P5']),
                format_number(row['P95']),
                format_number(row['P99']),
            ]
            rows.append(f"{var} & " + " & ".join(vals) + r" \\")
        return "\n".join(rows)

    panel_a_tex = format_stats_df(panel_a)
    panel_b_tex = format_stats_df(panel_b)

    note_text = (
        r"This table presents descriptive statistics for the monthly corporate bond panel. "
        r"Panel A shows statistics pooled across all cusip-month observations. "
        r"Panel B shows time-series averages of monthly cross-sectional statistics. "
        r"The sample spans the period " + min_date + r" to " + max_date + r". "
        r"All prices are in percentage of par, 100\% implies a dollar value of \$1000. "
        r"Yield to maturity (\texttt{ytm}) and Spread (\texttt{cs}) are in percentage points. "
        r"Duration, Bond Maturity and Age are in years. "
        r"Ratings are in numeric format (AAA = 1, ..., D = 22). "
        r"Returns are monthly decimal returns."
    )

    latex = r"""
\begin{table}[!ht]
\begin{center}
\footnotesize
\caption{""" + title + r"""}
\label{""" + label + r"""}\vspace{2mm}
\scalebox{1}{%
\begin{tabular}{lrrrrrrr}
\midrule
Variable & Mean & Median & SD & P1 & P5 & P95 & P99 \\
\midrule
\multicolumn{8}{c}{\textbf{Panel A: Pooled}} \\
\midrule
""" + panel_a_tex + r"""
\midrule
\multicolumn{8}{c}{\textbf{Panel B: Cross-sectional}} \\
\midrule
""" + panel_b_tex + r"""
\bottomrule
\end{tabular}
}
\end{center}
\begin{spacing}{1}
\footnotesize{
""" + note_text + r"""
}
\end{spacing}
\vspace{-2mm}
\end{table}
""".strip()

    return latex


def make_descriptive_stats_table_by_rating(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    min_date: str,
    max_date: str,
    table_number: int,
    title: str,
    rating_range_text: str,
    label: str = None,
    n_obs: int = None,
    n_total: int = None,
) -> str:
    """
    Generate descriptive statistics table for a specific rating category.

    Parameters
    ----------
    panel_a : pd.DataFrame
        Pooled statistics from compute_pooled_stats().
    panel_b : pd.DataFrame
        Cross-sectional statistics from compute_cross_sectional_stats().
    min_date : str
        Minimum date in data (YYYY-MM-DD).
    max_date : str
        Maximum date in data (YYYY-MM-DD).
    table_number : int
        Table number for caption (e.g., 3, 4, 5).
    title : str
        Rating category title (e.g., "Investment Grade Corporate Bonds").
    rating_range_text : str
        Text describing rating range (e.g., "Ratings 1-10 (AAA to BBB-)").
    label : str, optional
        LaTeX label. If None, derived from title.
    n_obs : int, optional
        Number of observations with non-nan ret_vw. Included in caption if provided.
    n_total : int, optional
        Total observations (for computing percentage). Required if n_obs provided.

    Returns
    -------
    str
        LaTeX table code.
    """
    if label is None:
        label = f"tab:monthly_stats_{title.lower().replace(' ', '_').replace('-', '_')}"

    def format_number(val: float) -> str:
        """Format number with proper LaTeX minus sign for negatives."""
        if val < 0:
            return f"$-${abs(val):.2f}"
        return f"{val:.2f}"

    def format_stats_df(df: pd.DataFrame) -> str:
        rows = []
        for _, row in df.iterrows():
            var = escape_latex(row['Variable'])
            vals = [
                format_number(row['Mean']),
                format_number(row['Median']),
                format_number(row['SD']),
                format_number(row['P1']),
                format_number(row['P5']),
                format_number(row['P95']),
                format_number(row['P99']),
            ]
            rows.append(f"{var} & " + " & ".join(vals) + r" \\")
        return "\n".join(rows)

    panel_a_tex = format_stats_df(panel_a)
    panel_b_tex = format_stats_df(panel_b)

    # Build observation count text if provided
    obs_text = ""
    if n_obs is not None:
        obs_text = f"The sample contains {n_obs:,} bond-month observations with non-missing returns"
        if n_total is not None and n_total > 0:
            pct = 100 * n_obs / n_total
            obs_text += f" ({pct:.2f}\\% of total)"
        obs_text += ". "

    note_text = (
        r"This table presents descriptive statistics for the monthly corporate bond panel. "
        r"The sample includes " + escape_latex(title.lower()) + r" (" + rating_range_text + r"). "
        + obs_text +
        r"Panel A shows statistics pooled across all cusip-month observations. "
        r"Panel B shows time-series averages of monthly cross-sectional statistics. "
        r"The sample spans the period " + min_date + r" to " + max_date + r". "
        r"All prices are in percentage of par. "
        r"Yield to maturity, spread, and returns are in percentage points. "
        r"Duration, bond maturity and age are in years. "
        r"Ratings are in numeric format (AAA = 1, ..., D = 22)."
    )

    latex = r"""
\begin{table}[!ht]
\begin{center}
\footnotesize
\caption{Monthly Panel Descriptive Statistics --- """ + title + r"""}
\label{""" + label + r"""}\vspace{2mm}
\scalebox{1}{%
\begin{tabular}{lrrrrrrr}
\midrule
Variable & Mean & Median & SD & P1 & P5 & P95 & P99 \\
\midrule
\multicolumn{8}{c}{\textbf{Panel A: Pooled}} \\
\midrule
""" + panel_a_tex + r"""
\midrule
\multicolumn{8}{c}{\textbf{Panel B: Cross-sectional}} \\
\midrule
""" + panel_b_tex + r"""
\bottomrule
\end{tabular}
}
\end{center}
\begin{spacing}{1}
\footnotesize{
""" + note_text + r"""
}
\end{spacing}
\vspace{-2mm}
\end{table}
""".strip()

    return latex


# ============================================================================
# EXTREME RETURNS / OUTLIER ANALYSIS
# ============================================================================

def compute_combined_extreme_stats(
    dfs: dict,
    ret_cols: list = None,
    thresholds: list = None,
) -> dict:
    """
    Compute extreme returns statistics for multiple rating categories.

    Parameters
    ----------
    dfs : dict
        Dictionary of {label: DataFrame} for each rating category.
        E.g., {'All': df_all, 'IG': df_ig, 'NIG': df_nig}
    ret_cols : list, optional
        List of (column_name, display_name) tuples for return columns.
        Default: [('ret_vw', 'End'), ('ret_vw_bgn', 'Begin')]
    thresholds : list, optional
        List of absolute thresholds (in decimal) to count exceedances.
        Default: [0.20, 0.50, 1.00] (20%, 50%, 100%)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'tail_stats': DataFrame with tail percentiles
        - 'threshold_counts': DataFrame with exceedance counts
        - 'moments': DataFrame with skewness, kurtosis, and N(>500%)
        - 'n_obs': dict of observation counts per category
    """
    from scipy import stats as scipy_stats

    if ret_cols is None:
        ret_cols = [('ret_vw', 'End'), ('ret_vw_bgn', 'Begin')]

    if thresholds is None:
        thresholds = [0.20, 0.50, 1.00]

    labels = list(dfs.keys())
    n_obs = {lbl: len(df) for lbl, df in dfs.items()}

    # --- Panel A: Tail Percentiles (richer array) ---
    percentiles = [
        ('P0.01', lambda s: s.quantile(0.0001)),
        ('P0.05', lambda s: s.quantile(0.0005)),
        ('P0.10', lambda s: s.quantile(0.001)),
        ('P99.90', lambda s: s.quantile(0.999)),
        ('P99.95', lambda s: s.quantile(0.9995)),
        ('P99.99', lambda s: s.quantile(0.9999)),
    ]

    tail_rows = []
    for stat_name, stat_func in percentiles:
        row = {'Statistic': stat_name}
        for col_name, display_name in ret_cols:
            for lbl, df in dfs.items():
                col_key = f'{display_name}_{lbl}'
                if col_name in df.columns:
                    series = df[col_name].dropna()
                    row[col_key] = stat_func(series) * 100 if len(series) > 0 else np.nan
                else:
                    row[col_key] = np.nan
        tail_rows.append(row)

    tail_stats = pd.DataFrame(tail_rows)

    # --- Panel B: Higher Moments (no scaling) ---
    moment_rows = []
    for stat_name, stat_func in [
        ('Skewness', scipy_stats.skew),
        ('Excess Kurtosis', lambda x: scipy_stats.kurtosis(x, fisher=True)),
    ]:
        row = {'Statistic': stat_name}
        for col_name, display_name in ret_cols:
            for lbl, df in dfs.items():
                col_key = f'{display_name}_{lbl}'
                if col_name in df.columns:
                    series = df[col_name].dropna()
                    row[col_key] = stat_func(series) if len(series) > 0 else np.nan
                else:
                    row[col_key] = np.nan
        moment_rows.append(row)

    moments = pd.DataFrame(moment_rows)

    # --- Panel C: Extreme Return Counts ---
    count_rows = []
    for thresh in thresholds:
        thresh_pct = int(thresh * 100)
        # Negative extremes
        row_neg = {'Direction': f'$<$$-${thresh_pct}\\%'}
        for col_name, display_name in ret_cols:
            for lbl, df in dfs.items():
                col_key = f'{display_name}_{lbl}'
                if col_name in df.columns:
                    series = df[col_name].dropna()
                    row_neg[col_key] = int((series < -thresh).sum())
                else:
                    row_neg[col_key] = 0
        count_rows.append(row_neg)
        # Positive extremes
        row_pos = {'Direction': f'$>$+{thresh_pct}\\%'}
        for col_name, display_name in ret_cols:
            for lbl, df in dfs.items():
                col_key = f'{display_name}_{lbl}'
                if col_name in df.columns:
                    series = df[col_name].dropna()
                    row_pos[col_key] = int((series > thresh).sum())
                else:
                    row_pos[col_key] = 0
        count_rows.append(row_pos)

    # Add N(>+500%) row as last row in Panel C
    row_500 = {'Direction': 'N($>$+500\\%)'}
    for col_name, display_name in ret_cols:
        for lbl, df in dfs.items():
            col_key = f'{display_name}_{lbl}'
            if col_name in df.columns:
                series = df[col_name].dropna()
                row_500[col_key] = int((series > 5.00).sum())
            else:
                row_500[col_key] = 0
    count_rows.append(row_500)

    threshold_counts = pd.DataFrame(count_rows)

    return {
        'tail_stats': tail_stats,
        'threshold_counts': threshold_counts,
        'moments': moments,
        'n_obs': n_obs,
        'labels': labels,
        'ret_cols': ret_cols,
    }


def make_combined_extreme_table(
    stats: dict,
    min_date: str,
    max_date: str,
    label: str = "tab:extreme_returns",
) -> str:
    """
    Generate combined LaTeX table with Panels A, B, C for extreme returns analysis.

    Panel A: Tail Percentiles (P0.01, P0.05, P0.10, P99.90, P99.95, P99.99)
    Panel B: Higher Moments (Skewness, Excess Kurtosis)
    Panel C: Extreme Return Counts (including N(>+500%))

    Parameters
    ----------
    stats : dict
        Output from compute_combined_extreme_stats().
    min_date : str
        Minimum date in data.
    max_date : str
        Maximum date in data.
    label : str
        LaTeX label.

    Returns
    -------
    str
        LaTeX table code.
    """
    def format_number(val, decimals=2) -> str:
        if pd.isna(val):
            return "--"
        if val < 0:
            return f"$-${abs(val):.{decimals}f}"
        return f"{val:.{decimals}f}"

    def format_int(val) -> str:
        return f"{int(val):,}"

    def format_mixed(val, is_count=False) -> str:
        """Format either as int (for counts) or float (for moments)."""
        if pd.isna(val):
            return "--"
        if is_count or (isinstance(val, (int, np.integer)) and val == int(val)):
            return f"{int(val):,}"
        if val < 0:
            return f"$-${abs(val):.2f}"
        return f"{val:.2f}"

    tail_df = stats['tail_stats']
    moments_df = stats['moments']
    counts_df = stats['threshold_counts']
    labels = stats['labels']
    ret_cols = stats['ret_cols']
    n_obs = stats['n_obs']

    # Build column keys in order: End_All, End_IG, End_NIG, Begin_All, Begin_IG, Begin_NIG
    col_keys = []
    for _, display_name in ret_cols:
        for lbl in labels:
            col_keys.append(f'{display_name}_{lbl}')

    n_cols = len(col_keys)
    n_per_ret = len(labels)
    total_cols = n_cols + 1  # +1 for row label column

    # Build header
    header1_parts = [""]
    cmidrule_parts = []
    col_idx = 2
    for _, display_name in ret_cols:
        header1_parts.append(f"\\multicolumn{{{n_per_ret}}}{{c}}{{{display_name} Return (\\%)}}")
        cmidrule_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + n_per_ret - 1}}}")
        col_idx += n_per_ret
    header1 = " & ".join(header1_parts) + r" \\"
    cmidrule = " ".join(cmidrule_parts)

    header2_parts = [""]
    for _ in ret_cols:
        header2_parts.extend(labels)
    header2 = " & ".join(header2_parts) + r" \\"

    # Panel A rows
    panel_a_rows = []
    for _, row in tail_df.iterrows():
        vals = [format_number(row[k]) for k in col_keys]
        panel_a_rows.append(f"{row['Statistic']} & " + " & ".join(vals) + r" \\")
    panel_a_tex = "\n".join(panel_a_rows)

    # Panel B rows (moments + N(>500%))
    panel_b_rows = []
    for _, row in moments_df.iterrows():
        stat_name = row['Statistic']
        is_count = stat_name.startswith('N(')
        vals = [format_mixed(row[k], is_count=is_count) for k in col_keys]
        panel_b_rows.append(f"{stat_name} & " + " & ".join(vals) + r" \\")
    panel_b_tex = "\n".join(panel_b_rows)

    # Panel C rows
    panel_c_rows = []
    for _, row in counts_df.iterrows():
        vals = [format_int(row[k]) for k in col_keys]
        panel_c_rows.append(f"{row['Direction']} & " + " & ".join(vals) + r" \\")
    panel_c_tex = "\n".join(panel_c_rows)

    # Observation counts note
    obs_parts = [f"{lbl}: {n_obs[lbl]:,}" for lbl in labels]
    obs_text = "; ".join(obs_parts)

    note_text = (
        r"This table presents extreme returns analysis for the monthly corporate bond panel. "
        r"Values are pooled over all bonds and months within each rating category. "
        r"Panel A reports tail percentiles of the return distribution (in \%). "
        r"Panel B reports skewness and excess kurtosis. "
        r"Panel C reports the count of observations exceeding the specified thresholds by direction, "
        r"including returns exceeding +500\%. "
        r"Observations: " + obs_text + r". "
        r"Sample period: " + min_date + r" to " + max_date + r"."
    )

    latex = r"""
\begin{table}[!ht]
\begin{center}
\footnotesize
\caption{Extreme Returns Analysis}
\label{""" + label + r"""}\vspace{2mm}
\scalebox{1}{%
\begin{tabular}{l""" + "r" * n_cols + r"""}
\toprule
""" + header1 + r"""
""" + cmidrule + r"""
""" + header2 + r"""
\midrule
\multicolumn{""" + str(total_cols) + r"""}{c}{\textbf{Panel A: Tail Percentiles}} \\
\midrule
""" + panel_a_tex + r"""
\midrule
\multicolumn{""" + str(total_cols) + r"""}{c}{\textbf{Panel B: Higher Moments}} \\
\midrule
""" + panel_b_tex + r"""
\midrule
\multicolumn{""" + str(total_cols) + r"""}{c}{\textbf{Panel C: Extreme Return Counts}} \\
\midrule
""" + panel_c_tex + r"""
\bottomrule
\end{tabular}
}
\end{center}
\begin{spacing}{1}
\footnotesize{
""" + note_text + r"""
}
\end{spacing}
\vspace{-2mm}
\end{table}
""".strip()

    return latex


def compute_time_concentration_stats(
    df: pd.DataFrame,
    date_col: str = 'date',
    ret_cols: list = None,
    thresholds: list = None,
) -> pd.DataFrame:
    """
    Compute annual counts of extreme returns for time concentration analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly panel DataFrame.
    date_col : str
        Date column name.
    ret_cols : list, optional
        List of (column_name, display_name) tuples.
    thresholds : list, optional
        List of absolute thresholds (default: [0.20, 0.95]).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Year, N, and counts for each return column/direction/threshold.
    """
    if ret_cols is None:
        ret_cols = [('ret_vw', 'End'), ('ret_vw_bgn', 'Begin')]

    if thresholds is None:
        thresholds = [0.20, 0.95]

    # Extract year
    w = df.copy()
    w['_year'] = pd.to_datetime(w[date_col]).dt.year

    # Group by year
    years = sorted(w['_year'].dropna().unique())

    rows = []
    for year in years:
        mask = w['_year'] == year
        year_df = w[mask]
        row = {'Year': int(year)}

        for col_name, display_name in ret_cols:
            if col_name in year_df.columns:
                series = year_df[col_name].dropna()
                # Add N count for this return type (non-null count)
                row[f'N_{display_name}'] = len(series)
                for thresh in thresholds:
                    thresh_pct = int(thresh * 100)
                    row[f'{display_name}_neg_{thresh_pct}'] = int((series < -thresh).sum())
                    row[f'{display_name}_pos_{thresh_pct}'] = int((series > thresh).sum())
            else:
                row[f'N_{display_name}'] = 0
                for thresh in thresholds:
                    thresh_pct = int(thresh * 100)
                    row[f'{display_name}_neg_{thresh_pct}'] = 0
                    row[f'{display_name}_pos_{thresh_pct}'] = 0

        rows.append(row)

    return pd.DataFrame(rows)


def make_time_concentration_table(
    stats_df: pd.DataFrame,
    min_date: str,
    max_date: str,
    thresholds: list = None,
    label: str = "tab:time_concentration_extremes",
) -> str:
    """
    Generate LaTeX table for time concentration of extreme returns.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Output from compute_time_concentration_stats().
    min_date : str
        Minimum date in data.
    max_date : str
        Maximum date in data.
    thresholds : list, optional
        List of thresholds used (default: [0.20, 0.95]).
    label : str
        LaTeX label.

    Returns
    -------
    str
        LaTeX table code.
    """
    if thresholds is None:
        thresholds = [0.20, 0.95]

    def format_int(val) -> str:
        return f"{int(val):,}"

    ret_types = ['End', 'Begin']

    # Build column structure: Year | N_End | thresholds... | N_Bgn | thresholds...
    # Each return type has: N, then threshold columns
    col_keys_per_ret = []
    for thresh in thresholds:
        thresh_pct = int(thresh * 100)
        col_keys_per_ret.append((thresh_pct, 'neg'))
        col_keys_per_ret.append((thresh_pct, 'pos'))

    # Build data rows
    rows = []
    for _, row in stats_df.iterrows():
        year = int(row['Year'])
        vals = [str(year)]
        for ret_type in ret_types:
            # Add N for this return type
            n_col = f'N_{ret_type}'
            vals.append(format_int(row[n_col]))
            # Add threshold columns
            for thresh_pct, direction in col_keys_per_ret:
                col_name = f'{ret_type}_{direction}_{thresh_pct}'
                vals.append(format_int(row[col_name]))
        rows.append(" & ".join(vals) + r" \\")
    rows_tex = "\n".join(rows)

    # Build total row
    total_vals = ["Total"]
    for ret_type in ret_types:
        n_col = f'N_{ret_type}'
        total_vals.append(format_int(stats_df[n_col].sum()))
        for thresh_pct, direction in col_keys_per_ret:
            col_name = f'{ret_type}_{direction}_{thresh_pct}'
            total_vals.append(format_int(stats_df[col_name].sum()))
    total_row = " & ".join(total_vals) + r" \\"

    # Build header
    # Columns per return type: N + 2*len(thresholds)
    n_per_ret = 1 + len(thresholds) * 2  # N + 2 directions per threshold
    header1 = f" & \\multicolumn{{{n_per_ret}}}{{c}}{{End Return}} & \\multicolumn{{{n_per_ret}}}{{c}}{{Begin Return}} \\\\"

    # cmidrule positions
    end_start = 2
    end_end = end_start + n_per_ret - 1
    begin_start = end_end + 1
    begin_end = begin_start + n_per_ret - 1
    cmidrule = f"\\cmidrule(lr){{{end_start}-{end_end}}} \\cmidrule(lr){{{begin_start}-{begin_end}}}"

    # Level 2: Year, then for each ret_type: N, threshold headers
    header2_parts = ["Year"]
    for ret_type in ret_types:
        header2_parts.append("N")
        for thresh in thresholds:
            thresh_pct = int(thresh * 100)
            header2_parts.append(f"$<$$-${thresh_pct}\\%")
            header2_parts.append(f"$>$+{thresh_pct}\\%")
    header2 = " & ".join(header2_parts) + r" \\"

    # Number of columns
    n_cols = 1 + 2 * n_per_ret  # Year + 2 return types
    col_spec = "l" + "r" * (n_cols - 1)

    thresh_pct_list = [int(t * 100) for t in thresholds]
    thresh_str = ", ".join([f"{t}\\%" for t in thresh_pct_list])

    note_text = (
        r"This table reports the annual frequency of extreme monthly returns "
        r"(absolute returns exceeding " + thresh_str + r") for the monthly corporate bond panel. "
        r"For each year, we report the number of non-missing observations (N) for each return type "
        r"and the count of returns falling below or above the specified thresholds. "
        r"Sample period: " + min_date + r" to " + max_date + r"."
    )

    latex = r"""
\begin{table}[!ht]
\begin{center}
\footnotesize
\caption{Time Concentration of Extreme Returns}
\label{""" + label + r"""}\vspace{2mm}
\scalebox{0.85}{%
\begin{tabular}{""" + col_spec + r"""}
\toprule
""" + header1 + r"""
""" + cmidrule + r"""
""" + header2 + r"""
\midrule
""" + rows_tex + r"""
\midrule
""" + total_row + r"""
\bottomrule
\end{tabular}
}
\end{center}
\begin{spacing}{1}
\footnotesize{
""" + note_text + r"""
}
\end{spacing}
\vspace{-2mm}
\end{table}
""".strip()

    return latex


def compute_annual_return_stats(
    df: pd.DataFrame,
    date_col: str = 'date',
    ret_cols: list = None,
) -> pd.DataFrame:
    """
    Compute annual summary statistics for returns.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly panel DataFrame.
    date_col : str
        Date column name.
    ret_cols : list, optional
        List of (column_name, display_name) tuples.
        Default: [('ret_vw', 'End'), ('ret_vw_bgn', 'Begin')]

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Year, N, and stats for each return column,
        plus correlation between ret_vw and ret_vw_bgn.
    """
    if ret_cols is None:
        ret_cols = [('ret_vw', 'End'), ('ret_vw_bgn', 'Begin')]

    # Extract year
    w = df.copy()
    w['_year'] = pd.to_datetime(w[date_col]).dt.year

    # Group by year
    years = sorted(w['_year'].dropna().unique())

    rows = []
    for year in years:
        mask = w['_year'] == year
        year_df = w[mask]
        row = {'Year': int(year)}

        for col_name, display_name in ret_cols:
            if col_name in year_df.columns:
                series = year_df[col_name].dropna()
                # Add N count for this return type (non-null count)
                row[f'N_{display_name}'] = len(series)
                if len(series) > 0:
                    row[f'{display_name}_mean'] = series.mean() * 100
                    row[f'{display_name}_std'] = series.std() * 100
                    row[f'{display_name}_med'] = series.median() * 100
                    row[f'{display_name}_min'] = series.min() * 100
                    row[f'{display_name}_max'] = series.max() * 100
                else:
                    row[f'{display_name}_mean'] = np.nan
                    row[f'{display_name}_std'] = np.nan
                    row[f'{display_name}_med'] = np.nan
                    row[f'{display_name}_min'] = np.nan
                    row[f'{display_name}_max'] = np.nan
            else:
                row[f'N_{display_name}'] = 0
                row[f'{display_name}_mean'] = np.nan
                row[f'{display_name}_std'] = np.nan
                row[f'{display_name}_med'] = np.nan
                row[f'{display_name}_min'] = np.nan
                row[f'{display_name}_max'] = np.nan

        # Correlation between ret_vw and ret_vw_bgn (using matched non-null pairs)
        if 'ret_vw' in year_df.columns and 'ret_vw_bgn' in year_df.columns:
            valid = year_df[['ret_vw', 'ret_vw_bgn']].dropna()
            row['N_corr'] = len(valid)
            if len(valid) > 1:
                row['rho'] = valid['ret_vw'].corr(valid['ret_vw_bgn'])
            else:
                row['rho'] = np.nan
        else:
            row['N_corr'] = 0
            row['rho'] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def make_annual_return_stats_table(
    stats_df: pd.DataFrame,
    min_date: str,
    max_date: str,
    label: str = "tab:annual_return_stats",
) -> str:
    """
    Generate LaTeX table for annual return summary statistics.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Output from compute_annual_return_stats().
    min_date : str
        Minimum date in data.
    max_date : str
        Maximum date in data.
    label : str
        LaTeX label.

    Returns
    -------
    str
        LaTeX table code.
    """
    def format_int(val) -> str:
        return f"{int(val):,}"

    def format_number(val, decimals=2) -> str:
        if pd.isna(val):
            return "--"
        if val < 0:
            return f"$-${abs(val):.{decimals}f}"
        return f"{val:.{decimals}f}"

    # Build data rows
    # Columns: Year | N_End | End stats | N_Bgn | Begin stats | rho
    rows = []
    for _, row in stats_df.iterrows():
        year = int(row['Year'])
        n_end = format_int(row['N_End'])
        # End stats
        end_mean = format_number(row['End_mean'])
        end_std = format_number(row['End_std'])
        end_med = format_number(row['End_med'])
        end_min = format_number(row['End_min'])
        end_max = format_number(row['End_max'])
        # Begin stats
        n_bgn = format_int(row['N_Begin'])
        bgn_mean = format_number(row['Begin_mean'])
        bgn_std = format_number(row['Begin_std'])
        bgn_med = format_number(row['Begin_med'])
        bgn_min = format_number(row['Begin_min'])
        bgn_max = format_number(row['Begin_max'])
        # Correlation
        rho = format_number(row['rho'])

        vals = [str(year), n_end,
                end_mean, end_std, end_med, end_min, end_max,
                n_bgn, bgn_mean, bgn_std, bgn_med, bgn_min, bgn_max,
                rho]
        rows.append(" & ".join(vals) + r" \\")
    rows_tex = "\n".join(rows)

    # Build total/overall row
    total_n_end = stats_df['N_End'].sum()
    total_n_bgn = stats_df['N_Begin'].sum()

    def weighted_avg(col, weight_col):
        valid = stats_df[[col, weight_col]].dropna()
        if len(valid) == 0 or valid[weight_col].sum() == 0:
            return np.nan
        return (valid[col] * valid[weight_col]).sum() / valid[weight_col].sum()

    # For SD, we use pooled standard deviation approximation
    # For simplicity, just show overall stats from the full period
    # We'll compute simple averages of the annual values
    total_end_mean = format_number(weighted_avg('End_mean', 'N_End'))
    total_end_std = format_number(stats_df['End_std'].mean())
    total_end_med = format_number(stats_df['End_med'].median())
    total_end_min = format_number(stats_df['End_min'].min())
    total_end_max = format_number(stats_df['End_max'].max())

    total_bgn_mean = format_number(weighted_avg('Begin_mean', 'N_Begin'))
    total_bgn_std = format_number(stats_df['Begin_std'].mean())
    total_bgn_med = format_number(stats_df['Begin_med'].median())
    total_bgn_min = format_number(stats_df['Begin_min'].min())
    total_bgn_max = format_number(stats_df['Begin_max'].max())

    total_rho = format_number(stats_df['rho'].mean())

    total_vals = ["Total", format_int(total_n_end),
                  total_end_mean, total_end_std, total_end_med, total_end_min, total_end_max,
                  format_int(total_n_bgn), total_bgn_mean, total_bgn_std, total_bgn_med, total_bgn_min, total_bgn_max,
                  total_rho]
    total_row = " & ".join(total_vals) + r" \\"

    note_text = (
        r"This table reports annual summary statistics for monthly returns (in \%). "
        r"For each year, we report the number of non-missing observations (N), mean, standard deviation (SD), "
        r"median (Med.), minimum (Min), and maximum (Max) of returns. "
        r"$\rho$ is the within-year correlation between End and Begin returns using matched non-missing pairs. "
        r"The Total row shows weighted averages for means, simple averages for SD and $\rho$, "
        r"median of medians, and overall min/max. "
        r"Sample period: " + min_date + r" to " + max_date + r"."
    )

    latex = r"""
\begin{table}[!ht]
\begin{center}
\footnotesize
\caption{Annual Return Summary Statistics}
\label{""" + label + r"""}\vspace{2mm}
\scalebox{0.85}{%
\begin{tabular}{lrrrrrrrrrrrrr}
\toprule
& \multicolumn{6}{c}{End Return (\%)} & \multicolumn{6}{c}{Begin Return (\%)} & \\
\cmidrule(lr){2-7} \cmidrule(lr){8-13}
Year & N & Mean & SD & Med. & Min & Max & N & Mean & SD & Med. & Min & Max & $\rho$ \\
\midrule
""" + rows_tex + r"""
\midrule
""" + total_row + r"""
\bottomrule
\end{tabular}
}
\end{center}
\begin{spacing}{1}
\footnotesize{
""" + note_text + r"""
}
\end{spacing}
\vspace{-2mm}
\end{table}
""".strip()

    return latex


# ============================================================================
# MAIN PANEL LOADING UTILITY
# ============================================================================

def load_main_panel(
    data_dir: Path,
    date_stamp: str,
    columns: list = None,
) -> pd.DataFrame:
    """
    Load main_panel parquet file with optional column selection.

    Parameters
    ----------
    data_dir : Path
        Directory containing main_panel parquet file.
    date_stamp : str
        Date stamp suffix (e.g., '20251126').
    columns : list, optional
        List of columns to load. If None, loads all columns.

    Returns
    -------
    pd.DataFrame
        Main panel DataFrame.
    """
    path = data_dir / f"main_panel_{date_stamp}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"main_panel not found: {path}")

    if columns is not None:
        # Add required columns
        required = ['cusip', 'date', 'spc_rat']
        cols = list(set(columns + required))
        return pd.read_parquet(path, columns=cols)
    else:
        return pd.read_parquet(path)


# ============================================================================
# DFPS DATABASE COMPARISON
# ============================================================================

DFPS_URL = "https://openbondassetpricing.com/wp-content/uploads/2025/12/trace_alternate_2025_12_2024.zip"
DFPS_FILENAME = "trace_alternate_2025_12_2024.parquet"


def _check_internet_connectivity(host: str = "8.8.8.8", port: int = 53, timeout: int = 3) -> bool:
    """
    Check if internet connectivity is available.

    Uses Google's public DNS as default check target.

    Parameters
    ----------
    host : str
        Host to connect to (default: Google DNS 8.8.8.8)
    port : int
        Port to connect to (default: 53 for DNS)
    timeout : int
        Connection timeout in seconds

    Returns
    -------
    bool
        True if internet is available, False otherwise
    """
    import socket
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False


def load_dfps_data(
    cache_dir: Path = None,
    local_path: Path = None,
    url: str = DFPS_URL,
    filename: str = DFPS_FILENAME,
) -> pd.DataFrame:
    """
    Download and load DFPS TRACE data from openbondassetpricing.com.

    Checks internet connectivity first. If no internet (e.g., WRDS cloud),
    loads from local_path. If internet available, downloads from URL.

    Parameters
    ----------
    cache_dir : Path, optional
        Directory to cache downloaded file. If None, uses tempfile.
    local_path : Path, optional
        Local path to pre-downloaded DFPS parquet file (used when no internet).
    url : str
        URL to the zip file containing DFPS data.
    filename : str
        Name of the parquet file inside the zip.

    Returns
    -------
    pd.DataFrame
        DFPS data with columns: date, cusip, ret_vw_dfps
    """
    import zipfile
    import urllib.request
    import io

    # Check cache first
    if cache_dir is not None:
        cache_path = Path(cache_dir) / filename
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            df['date'] = pd.to_datetime(df['date'])
            return df

    # Check internet connectivity
    has_internet = _check_internet_connectivity()

    if not has_internet:
        # No internet (WRDS cloud) - load from local path
        if local_path is not None and Path(local_path).exists():
            df = pd.read_parquet(local_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            raise FileNotFoundError(
                f"No internet connection and local DFPS file not found at: {local_path}"
            )

    # Has internet - download zip file
    with urllib.request.urlopen(url) as response:
        zip_data = io.BytesIO(response.read())

    # Extract parquet from zip
    with zipfile.ZipFile(zip_data, 'r') as zf:
        with zf.open(filename) as f:
            df = pd.read_parquet(io.BytesIO(f.read()))

    # Cache if directory provided
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_dir / filename, index=False)

    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])

    return df


def load_wrds_data(
    wrds_username: str,
    cache_dir: Path = None,
    cache_filename: str = "wrds_bondret.parquet",
) -> pd.DataFrame:
    """
    Load WRDS Bond Returns data from wrdsapps.bondret.

    Downloads data via WRDS connection, converts columns to lowercase,
    and adjusts date to month-end.

    Parameters
    ----------
    wrds_username : str
        WRDS username for connection.
    cache_dir : Path, optional
        Directory to cache downloaded file. If cached file exists, loads from cache.
    cache_filename : str
        Filename for cached parquet file.

    Returns
    -------
    pd.DataFrame
        WRDS Bond Returns with columns: date, cusip, ret_l5m
        - date: Month-end datetime (MonthEnd offset applied)
        - cusip: Bond identifier (lowercase)
        - ret_l5m: WRDS last-5-day monthly return (equivalent to OSBAP ret_vw)
    """
    import wrds

    # Check cache first
    if cache_dir is not None:
        cache_path = Path(cache_dir) / cache_filename
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            df['date'] = pd.to_datetime(df['date'])
            return df

    # Connect to WRDS and download data
    conn = wrds.Connection(wrds_username=wrds_username)

    query = """
        SELECT date, cusip, ret_l5m
        FROM wrdsapps.bondret
    """
    df = conn.raw_sql(query)
    conn.close()

    # Convert columns to lowercase (WRDS returns uppercase)
    df.columns = df.columns.str.lower()

    # Convert date to month-end
    df['date'] = pd.to_datetime(df['date']) + pd.offsets.MonthEnd(0)

    # Cache if directory provided
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_dir / cache_filename, index=False)

    return df


def make_database_comparison_table(
    df_osbap: pd.DataFrame,
    df_alt: pd.DataFrame,
    min_date: str,
    max_date: str,
    date_col: str = 'date',
    id_col: str = 'cusip',
    ret_col_osbap: str = 'ret_vw',
    ret_col_alt: str = 'ret_vw',
    alt_name: str = 'DFPS',
    alt_description: str = None,
    alt_url: str = None,
    table_label: str = None,
) -> str:
    """
    Generate database coverage comparison table (OSBAP vs alternative database).

    Parameters
    ----------
    df_osbap : pd.DataFrame
        OSBAP panel with returns.
    df_alt : pd.DataFrame
        Alternative database with returns (e.g., DFPS, WRDS).
    min_date : str
        Minimum date for table note.
    max_date : str
        Maximum date for table note.
    date_col : str
        Date column name.
    id_col : str
        Bond identifier column.
    ret_col_osbap : str
        Return column in OSBAP data.
    ret_col_alt : str
        Return column in alternative data.
    alt_name : str
        Short name for alternative database (e.g., 'DFPS', 'WRDS').
    alt_description : str, optional
        Full description for the note (e.g., 'Dick-Nielsen, Feldhütter...').
        If None, uses alt_name.
    alt_url : str, optional
        URL for hyperlink in caption. If provided, alt_name becomes a hyperlink.
    table_label : str, optional
        LaTeX label for table. If None, generates from alt_name.

    Returns
    -------
    str
        LaTeX table code.
    """
    # Prepare data
    osbap = df_osbap[[date_col, id_col, ret_col_osbap]].copy()
    osbap = osbap.dropna(subset=[ret_col_osbap])
    osbap['_year'] = pd.to_datetime(osbap[date_col]).dt.year

    alt = df_alt[[date_col, id_col, ret_col_alt]].copy()
    alt = alt.dropna(subset=[ret_col_alt])
    alt['_year'] = pd.to_datetime(alt[date_col]).dt.year

    # Get unique years
    all_years = sorted(set(osbap['_year'].unique()) | set(alt['_year'].unique()))

    def format_int(val):
        return f"{int(val):,}"

    rows = []
    total_osbap_bonds = 0
    total_osbap_obs = 0
    total_alt_bonds = 0
    total_alt_obs = 0
    total_both_obs = 0

    for year in all_years:
        # OSBAP stats
        osbap_year = osbap[osbap['_year'] == year]
        n_osbap_obs = len(osbap_year)
        n_osbap_bonds = osbap_year[id_col].nunique()

        # Alternative stats
        alt_year = alt[alt['_year'] == year]
        n_alt_obs = len(alt_year)
        n_alt_bonds = alt_year[id_col].nunique()

        # Intersection (matched cusip-month pairs)
        osbap_keys = set(zip(osbap_year[date_col], osbap_year[id_col]))
        alt_keys = set(zip(alt_year[date_col], alt_year[id_col]))
        n_both = len(osbap_keys & alt_keys)

        total_osbap_bonds += n_osbap_bonds
        total_osbap_obs += n_osbap_obs
        total_alt_bonds += n_alt_bonds
        total_alt_obs += n_alt_obs
        total_both_obs += n_both

        row = (f"{year} & {format_int(n_osbap_bonds)} & {format_int(n_osbap_obs)} & "
               f"{format_int(n_alt_bonds)} & {format_int(n_alt_obs)} & "
               f"{format_int(n_both)} " + r"\\")
        rows.append(row)

    rows_tex = "\n".join(rows)

    # Total row (unique bonds across years)
    total_osbap_unique = osbap[id_col].nunique()
    total_alt_unique = alt[id_col].nunique()

    total_row = (f"Total & {format_int(total_osbap_unique)} & {format_int(total_osbap_obs)} & "
                 f"{format_int(total_alt_unique)} & {format_int(total_alt_obs)} & "
                 f"{format_int(total_both_obs)} " + r"\\")

    # Build description for note
    if alt_description is None:
        alt_desc = alt_name
    else:
        alt_desc = alt_description

    note_text = (
        r"This table compares database coverage between OSBAP (Open Source Bond Asset Pricing) "
        f"and {alt_desc}. "
        r"For each year, we report the number of unique bonds and bond-month observations with "
        r"non-missing returns. The `Both' column shows the number of matched cusip-month pairs "
        r"present in both databases. "
        r"Sample period: " + min_date + r" to " + max_date + r"."
    )

    # Build caption with optional hyperlink
    if alt_url is not None:
        caption_alt = r"\href{" + alt_url + r"}{" + alt_name + r"}"
    else:
        caption_alt = alt_name

    # Generate table label
    if table_label is None:
        table_label = f"tab:database_coverage_{alt_name.lower()}"

    latex = r"""
\begin{table}[!ht]
\begin{center}
\footnotesize
\caption{Database Coverage Comparison: OSBAP vs """ + caption_alt + r"""}
\label{""" + table_label + r"""}\vspace{2mm}
\scalebox{0.85}{%
\begin{tabular}{lrrrrr}
\toprule
& \multicolumn{2}{c}{OSBAP} & \multicolumn{2}{c}{""" + alt_name + r"""} & \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Year & Bonds & Obs. & Bonds & Obs. & Both \\
\midrule
""" + rows_tex + r"""
\midrule
""" + total_row + r"""
\bottomrule
\end{tabular}
}
\end{center}
\begin{spacing}{1}
\footnotesize{
""" + note_text + r"""
}
\end{spacing}
\vspace{-2mm}
\end{table}
""".strip()

    return latex


def make_return_comparison_table(
    df_osbap: pd.DataFrame,
    df_alt: pd.DataFrame,
    min_date: str,
    max_date: str,
    date_col: str = 'date',
    id_col: str = 'cusip',
    ret_col_osbap: str = 'ret_vw',
    ret_col_alt: str = 'ret_vw',
    ret_type_col: str = 'ret_type',
    rule_144a_col: str = '144a',
    alt_name: str = 'DFPS',
    alt_url: str = None,
    table_label: str = None,
) -> str:
    """
    Generate return comparison statistics table (OSBAP vs alternative database).

    Panel A: Bond-level return statistics (each database uses own non-NaN values)
    Panel B: EW market return statistics (equal-weighted average per date)

    Includes OSBAP exclusion variants:
    - No Def: excludes ret_type == 'trad_in_def'
    - No 144a: excludes rule_144a == 1
    - No Def & 144a: excludes both

    Parameters
    ----------
    df_osbap : pd.DataFrame
        OSBAP panel with returns.
    df_alt : pd.DataFrame
        Alternative database with returns (e.g., DFPS, WRDS).
    min_date : str
        Minimum date for table note.
    max_date : str
        Maximum date for table note.
    date_col : str
        Date column name.
    id_col : str
        Bond identifier column.
    ret_col_osbap : str
        Return column in OSBAP data.
    ret_col_alt : str
        Return column in alternative data.
    ret_type_col : str
        Return type column in OSBAP (for excluding 'trad_in_def').
    rule_144a_col : str
        Rule 144a indicator column in OSBAP.
    alt_name : str
        Short name for alternative database (e.g., 'DFPS', 'WRDS').
    alt_url : str, optional
        URL for hyperlink in caption. If provided, alt_name becomes a hyperlink.
    table_label : str, optional
        LaTeX label for table. If None, generates from alt_name.

    Returns
    -------
    str
        LaTeX table code.
    """
    # Columns to keep from OSBAP
    osbap_cols = [date_col, id_col, ret_col_osbap]
    if ret_type_col in df_osbap.columns:
        osbap_cols.append(ret_type_col)
    if rule_144a_col in df_osbap.columns:
        osbap_cols.append(rule_144a_col)

    osbap_full = df_osbap[osbap_cols].copy()
    osbap_full[date_col] = pd.to_datetime(osbap_full[date_col])
    osbap_full = osbap_full.rename(columns={ret_col_osbap: 'ret_osbap'})

    alt = df_alt[[date_col, id_col, ret_col_alt]].copy()
    alt[date_col] = pd.to_datetime(alt[date_col])
    alt = alt.rename(columns={ret_col_alt: 'ret_alt'})

    # Filter to common date range (same start/end for both)
    common_min = max(osbap_full[date_col].min(), alt[date_col].min())
    common_max = min(osbap_full[date_col].max(), alt[date_col].max())
    osbap_full = osbap_full[(osbap_full[date_col] >= common_min) & (osbap_full[date_col] <= common_max)]
    alt = alt[(alt[date_col] >= common_min) & (alt[date_col] <= common_max)]

    # Create OSBAP variants
    osbap = osbap_full[[date_col, id_col, 'ret_osbap']].copy()

    # No Def: exclude ret_type == 'trad_in_def'
    if ret_type_col in osbap_full.columns:
        mask_no_def = osbap_full[ret_type_col] != 'trad_in_def'
        osbap_no_def = osbap_full.loc[mask_no_def, [date_col, id_col, 'ret_osbap']].copy()
    else:
        osbap_no_def = osbap.copy()

    # No 144a: keep only rule_144a == 0
    if rule_144a_col in osbap_full.columns:
        mask_no_144a = osbap_full[rule_144a_col] == 0
        osbap_no_144a = osbap_full.loc[mask_no_144a, [date_col, id_col, 'ret_osbap']].copy()
    else:
        osbap_no_144a = osbap.copy()

    # No Def & 144a: exclude both
    if ret_type_col in osbap_full.columns and rule_144a_col in osbap_full.columns:
        mask_no_both = (osbap_full[ret_type_col] != 'trad_in_def') & (osbap_full[rule_144a_col] == 0)
        osbap_no_both = osbap_full.loc[mask_no_both, [date_col, id_col, 'ret_osbap']].copy()
    else:
        osbap_no_both = osbap.copy()

    # Compute stats function with extended percentiles
    def compute_stats(series):
        s = series.dropna()
        if len(s) == 0:
            return {k: np.nan for k in ['N', 'Mean', 'SD', 'Min', 'P0.1', 'P0.5', 'P1', 'Median', 'P99', 'P99.5', 'P99.9', 'Max']}
        return {
            'N': len(s),
            'Mean': s.mean() * 100,
            'SD': s.std() * 100,
            'Min': s.min() * 100,
            'P0.1': s.quantile(0.001) * 100,
            'P0.5': s.quantile(0.005) * 100,
            'P1': s.quantile(0.01) * 100,
            'Median': s.median() * 100,
            'P99': s.quantile(0.99) * 100,
            'P99.5': s.quantile(0.995) * 100,
            'P99.9': s.quantile(0.999) * 100,
            'Max': s.max() * 100,
        }

    # Panel A: Bond-level statistics
    stats_osbap = compute_stats(osbap['ret_osbap'])
    stats_osbap_no_def = compute_stats(osbap_no_def['ret_osbap'])
    stats_osbap_no_144a = compute_stats(osbap_no_144a['ret_osbap'])
    stats_osbap_no_both = compute_stats(osbap_no_both['ret_osbap'])
    stats_alt = compute_stats(alt['ret_alt'])

    # For difference and correlation, need matched observations with both non-NaN
    merged = osbap.merge(alt, on=[date_col, id_col], how='inner')
    matched = merged.dropna(subset=['ret_osbap', 'ret_alt'])
    n_matched = len(matched)
    matched['diff'] = matched['ret_osbap'] - matched['ret_alt']
    stats_diff = compute_stats(matched['diff'])
    rho_bond = matched['ret_osbap'].corr(matched['ret_alt'])

    # Panel B: EW market returns (equal-weighted average per date)
    mkt_osbap = osbap.dropna(subset=['ret_osbap']).groupby(date_col, observed=True)['ret_osbap'].mean()
    mkt_osbap_no_def = osbap_no_def.dropna(subset=['ret_osbap']).groupby(date_col, observed=True)['ret_osbap'].mean()
    mkt_osbap_no_144a = osbap_no_144a.dropna(subset=['ret_osbap']).groupby(date_col, observed=True)['ret_osbap'].mean()
    mkt_osbap_no_both = osbap_no_both.dropna(subset=['ret_osbap']).groupby(date_col, observed=True)['ret_osbap'].mean()
    mkt_alt = alt.dropna(subset=['ret_alt']).groupby(date_col, observed=True)['ret_alt'].mean()

    stats_mkt_osbap = compute_stats(mkt_osbap)
    stats_mkt_osbap_no_def = compute_stats(mkt_osbap_no_def)
    stats_mkt_osbap_no_144a = compute_stats(mkt_osbap_no_144a)
    stats_mkt_osbap_no_both = compute_stats(mkt_osbap_no_both)
    stats_mkt_alt = compute_stats(mkt_alt)

    # For market difference and correlation, match on date
    mkt_merged = pd.DataFrame({'mkt_osbap': mkt_osbap, 'mkt_alt': mkt_alt}).dropna()
    mkt_merged['diff'] = mkt_merged['mkt_osbap'] - mkt_merged['mkt_alt']
    stats_mkt_diff = compute_stats(mkt_merged['diff'])
    rho_mkt = mkt_merged['mkt_osbap'].corr(mkt_merged['mkt_alt'])
    n_mkt_matched = len(mkt_merged)

    def format_number(val, decimals=2):
        if pd.isna(val):
            return "--"
        if val < 0:
            return f"$-${abs(val):.{decimals}f}"
        return f"{val:.{decimals}f}"

    def format_int(val):
        if pd.isna(val):
            return "--"
        return f"{int(val):,}"

    # Build Panel A rows
    cols_a = ['N', 'Mean', 'SD', 'Min', 'P0.1', 'P0.5', 'P1', 'Median', 'P99', 'P99.5', 'P99.9', 'Max']

    def build_row(name, stats):
        vals = [name, format_int(stats['N'])]
        for c in cols_a[1:]:
            vals.append(format_number(stats[c]))
        return " & ".join(vals) + r" \\"

    rows_a = []
    rows_a.append(build_row("OSBAP", stats_osbap))
    rows_a.append(build_row("OSBAP (No Def)", stats_osbap_no_def))
    rows_a.append(build_row("OSBAP (No 144a)", stats_osbap_no_144a))
    rows_a.append(build_row("OSBAP (No Def \\& 144a)", stats_osbap_no_both))
    rows_a.append(build_row(alt_name, stats_alt))
    rows_a.append(r"\midrule")
    rows_a.append(build_row("Difference", stats_diff))

    # Build Panel B rows
    rows_b = []
    rows_b.append(build_row("OSBAP", stats_mkt_osbap))
    rows_b.append(build_row("OSBAP (No Def)", stats_mkt_osbap_no_def))
    rows_b.append(build_row("OSBAP (No 144a)", stats_mkt_osbap_no_144a))
    rows_b.append(build_row("OSBAP (No Def \\& 144a)", stats_mkt_osbap_no_both))
    rows_b.append(build_row(alt_name, stats_mkt_alt))
    rows_b.append(r"\midrule")
    rows_b.append(build_row("Difference", stats_mkt_diff))

    rows_a_tex = "\n".join(rows_a)
    rows_b_tex = "\n".join(rows_b)

    note_text = (
        f"This table compares monthly returns between OSBAP and {alt_name} databases. "
        r"Panel A reports bond-level return statistics; each database uses its own non-missing observations. "
        r"OSBAP (No Def) excludes bonds trading in default (\texttt{ret\_type} = `trad\_in\_def'). "
        r"OSBAP (No 144a) excludes Rule 144a bonds. "
        r"OSBAP (No Def \& 144a) excludes both. "
        r"Difference row uses " + format_int(n_matched) + r" matched cusip-month pairs with non-missing returns in both. "
        r"Bond-level correlation: $\rho$ = " + format_number(rho_bond, 3) + r". "
        r"Panel B reports equal-weighted (EW) market return statistics computed as the simple average across bonds each month. "
        r"Market correlation ($N$ = " + format_int(n_mkt_matched) + r" months): $\rho$ = " + format_number(rho_mkt, 3) + r". "
        r"Returns scaled to percent (\%). "
        r"Sample period: " + min_date + r" to " + max_date + r"."
    )

    # Build caption with optional hyperlink
    if alt_url is not None:
        caption_alt = r"\href{" + alt_url + r"}{" + alt_name + r"}"
    else:
        caption_alt = alt_name

    # Generate table label
    if table_label is None:
        table_label = f"tab:return_comparison_{alt_name.lower()}"

    header = r"& N & Mean & SD & Min & P0.1 & P0.5 & P1 & Median & P99 & P99.5 & P99.9 & Max \\"

    latex = r"""
\begin{table}[!ht]
\begin{center}
\footnotesize
\caption{Return Comparison Statistics: OSBAP vs """ + caption_alt + r"""}
\label{""" + table_label + r"""}\vspace{2mm}
\scalebox{0.75}{%
\begin{tabular}{lrrrrrrrrrrrr}
\toprule
\multicolumn{13}{l}{\textit{Panel A: Bond-Level Returns}} \\
\midrule
""" + header + r"""
\midrule
""" + rows_a_tex + r"""
\midrule
\multicolumn{13}{l}{\textit{Panel B: EW Market Returns}} \\
\midrule
""" + header + r"""
\midrule
""" + rows_b_tex + r"""
\bottomrule
\end{tabular}
}
\end{center}
\begin{spacing}{1}
\footnotesize{
""" + note_text + r"""
}
\end{spacing}
\vspace{-2mm}
\end{table}
""".strip()

    return latex


def create_coverage_comparison_plot(
    df_osbap: pd.DataFrame,
    df_alt: pd.DataFrame,
    output_dir: Path,
    filename: str = "coverage_comparison",
    params: PlotParams = None,
    date_col: str = 'date',
    id_col: str = 'cusip',
    ret_col_osbap: str = 'ret_vw',
    ret_col_alt: str = 'ret_vw',
    ret_type_col: str = 'ret_type',
    rule_144a_col: str = '144a',
    alt_name: str = 'DFPS',
    alt_url: str = None,
) -> tuple:
    """
    Create coverage comparison plot over time (OSBAP vs alternative database).

    Panel A: OSBAP vs alternative
    Panel B: OSBAP exclusions (No Def, No 144a, No Def & 144a) vs alternative

    Parameters
    ----------
    df_osbap : pd.DataFrame
        OSBAP panel with returns.
    df_alt : pd.DataFrame
        Alternative database with returns (e.g., DFPS, WRDS).
    output_dir : Path
        Directory to save the figure.
    filename : str
        Base filename.
    params : PlotParams
        Plotting parameters.
    date_col : str
        Date column name.
    id_col : str
        Bond identifier column.
    ret_col_osbap : str
        Return column in OSBAP data.
    ret_col_alt : str
        Return column in alternative data.
    ret_type_col : str
        Return type column in OSBAP (for excluding 'trad_in_def').
    rule_144a_col : str
        Rule 144a indicator column in OSBAP.
    alt_name : str
        Short name for alternative database (e.g., 'DFPS', 'WRDS').
    alt_url : str, optional
        URL for hyperlink in caption. If provided, alt_name becomes a hyperlink.

    Returns
    -------
    tuple
        (Path to saved figure, caption text).
    """
    from matplotlib.dates import AutoDateLocator, DateFormatter

    if params is None:
        params = PlotParams()

    apply_plot_params(params)

    # Columns to keep from OSBAP
    osbap_cols = [date_col, id_col, ret_col_osbap]
    if ret_type_col in df_osbap.columns:
        osbap_cols.append(ret_type_col)
    if rule_144a_col in df_osbap.columns:
        osbap_cols.append(rule_144a_col)

    osbap_full = df_osbap[osbap_cols].copy()
    osbap_full[date_col] = pd.to_datetime(osbap_full[date_col])

    alt = df_alt[[date_col, id_col, ret_col_alt]].copy()
    alt[date_col] = pd.to_datetime(alt[date_col])

    # Filter to common date range (same start/end for both)
    common_min = max(osbap_full[date_col].min(), alt[date_col].min())
    common_max = min(osbap_full[date_col].max(), alt[date_col].max())
    osbap_full = osbap_full[(osbap_full[date_col] >= common_min) & (osbap_full[date_col] <= common_max)]
    alt = alt[(alt[date_col] >= common_min) & (alt[date_col] <= common_max)]

    # Create OSBAP variants
    osbap = osbap_full[[date_col, id_col, ret_col_osbap]].copy()

    # No Def: exclude ret_type == 'trad_in_def'
    if ret_type_col in osbap_full.columns:
        mask_no_def = osbap_full[ret_type_col] != 'trad_in_def'
        osbap_no_def = osbap_full.loc[mask_no_def, [date_col, id_col, ret_col_osbap]].copy()
    else:
        osbap_no_def = osbap.copy()

    # No 144a: keep only rule_144a == 0
    if rule_144a_col in osbap_full.columns:
        mask_no_144a = osbap_full[rule_144a_col] == 0
        osbap_no_144a = osbap_full.loc[mask_no_144a, [date_col, id_col, ret_col_osbap]].copy()
    else:
        osbap_no_144a = osbap.copy()

    # No Def & 144a: exclude both
    if ret_type_col in osbap_full.columns and rule_144a_col in osbap_full.columns:
        mask_no_both = (osbap_full[ret_type_col] != 'trad_in_def') & (osbap_full[rule_144a_col] == 0)
        osbap_no_both = osbap_full.loc[mask_no_both, [date_col, id_col, ret_col_osbap]].copy()
    else:
        osbap_no_both = osbap.copy()

    # Count non-null returns per month for each variant
    def count_monthly(df, ret_col):
        return df.dropna(subset=[ret_col]).groupby(
            date_col, observed=True
        ).size().reset_index(name='n')

    osbap_monthly = count_monthly(osbap, ret_col_osbap)
    osbap_no_def_monthly = count_monthly(osbap_no_def, ret_col_osbap)
    osbap_no_144a_monthly = count_monthly(osbap_no_144a, ret_col_osbap)
    osbap_no_both_monthly = count_monthly(osbap_no_both, ret_col_osbap)
    alt_monthly = count_monthly(alt, ret_col_alt)

    # Merge all for Panel A
    monthly_a = osbap_monthly.rename(columns={'n': 'n_osbap'}).merge(
        alt_monthly.rename(columns={'n': 'n_alt'}), on=date_col, how='outer'
    ).sort_values(date_col).fillna(0)

    # Merge all for Panel B
    monthly_b = osbap_no_def_monthly.rename(columns={'n': 'n_no_def'}).merge(
        osbap_no_144a_monthly.rename(columns={'n': 'n_no_144a'}), on=date_col, how='outer'
    ).merge(
        osbap_no_both_monthly.rename(columns={'n': 'n_no_both'}), on=date_col, how='outer'
    ).merge(
        alt_monthly.rename(columns={'n': 'n_alt'}), on=date_col, how='outer'
    ).sort_values(date_col).fillna(0)

    # Create figure with two panels (2x1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    locator = AutoDateLocator(minticks=5, maxticks=10, interval_multiples=True)

    # Panel A: OSBAP vs alternative
    ax = axes[0]
    ax.plot(monthly_a[date_col], monthly_a['n_osbap'],
            color='steelblue', lw=1.2, label='OSBAP')
    ax.plot(monthly_a[date_col], monthly_a['n_alt'],
            color='darkred', lw=1.2, ls='--', label=alt_name)
    ax.set_ylabel('Bond-Month Observations')
    ax.set_title(f'Panel A: OSBAP vs {alt_name}', pad=5)
    ax.legend(loc='upper left', frameon=True, edgecolor='gray')
    ax.grid(True, alpha=params.grid_alpha, lw=params.grid_lw)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.margins(x=0.01)

    # Panel B: OSBAP exclusions vs alternative
    ax = axes[1]
    ax.plot(monthly_b[date_col], monthly_b['n_no_def'],
            color='steelblue', lw=1.2, label='OSBAP (No Def)')
    ax.plot(monthly_b[date_col], monthly_b['n_no_144a'],
            color='forestgreen', lw=1.2, label='OSBAP (No 144a)')
    ax.plot(monthly_b[date_col], monthly_b['n_no_both'],
            color='darkorange', lw=1.2, label='OSBAP (No Def & 144a)')
    ax.plot(monthly_b[date_col], monthly_b['n_alt'],
            color='darkred', lw=1.2, ls='--', label=alt_name)
    ax.set_ylabel('Bond-Month Observations')
    ax.set_title(f'Panel B: OSBAP Exclusions vs {alt_name}', pad=5)
    ax.legend(loc='upper left', frameon=True, edgecolor='gray')
    ax.grid(True, alpha=params.grid_alpha, lw=params.grid_lw)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.margins(x=0.01)

    plt.tight_layout()

    # Save figure
    ext = params.export_format
    fig_path = output_dir / f"{filename}.{ext}"
    fig.savefig(fig_path, dpi=params.figure_dpi, transparent=params.transparent,
                bbox_inches='tight')
    plt.close(fig)

    # Caption text with optional hyperlink
    if alt_url is not None:
        alt_ref = r"\href{" + alt_url + r"}{" + alt_name + r"}"
    else:
        alt_ref = alt_name

    caption = (
        f"Monthly coverage comparison between OSBAP and {alt_ref} TRACE databases. "
        f"Panel A compares full OSBAP vs {alt_name}. Panel B shows OSBAP exclusion variants: "
        r"(No Def) excludes trades in default, (No 144a) excludes Rule 144a bonds, "
        r"(No Def \& 144a) excludes both."
    )

    return fig_path, caption


def create_mktb_comparison_scatter(
    df_osbap: pd.DataFrame,
    df_alt: pd.DataFrame,
    output_dir: Path,
    filename: str = "mktb_comparison_scatter",
    params: PlotParams = None,
    date_col: str = 'date',
    id_col: str = 'cusip',
    ret_col_osbap: str = 'ret_vw',
    ret_col_alt: str = 'ret_vw',
    ret_type_col: str = 'ret_type',
    rule_144a_col: str = '144a',
    alt_name: str = 'DFPS',
    alt_url: str = None,
) -> tuple:
    """
    Create MKTB scatter comparison (OSBAP vs alternative equal-weighted market returns).

    Panel A: Full observations - MKTB computed using all non-NaN values in each database
    Panel B: Overlapping only - MKTB computed using matched cusip-month pairs
    Panel C: OSBAP (No Def) vs alternative - excludes trades in default
    Panel D: OSBAP (No Def & 144a) vs alternative - excludes both

    Parameters
    ----------
    df_osbap : pd.DataFrame
        OSBAP panel with returns.
    df_alt : pd.DataFrame
        Alternative database with returns (e.g., DFPS, WRDS).
    output_dir : Path
        Directory to save the figure.
    filename : str
        Base filename.
    params : PlotParams
        Plotting parameters.
    date_col : str
        Date column name.
    id_col : str
        Bond identifier column.
    ret_col_osbap : str
        Return column in OSBAP data.
    ret_col_alt : str
        Return column in alternative data.
    ret_type_col : str
        Return type column in OSBAP (for excluding 'trad_in_def').
    rule_144a_col : str
        Rule 144a indicator column in OSBAP.
    alt_name : str
        Short name for alternative database (e.g., 'DFPS', 'WRDS').
    alt_url : str, optional
        URL for hyperlink in caption. If provided, alt_name becomes a hyperlink.

    Returns
    -------
    tuple
        (Path to saved figure, caption text).
    """
    if params is None:
        params = PlotParams()

    apply_plot_params(params)

    # Columns to keep from OSBAP
    osbap_cols = [date_col, id_col, ret_col_osbap]
    if ret_type_col in df_osbap.columns:
        osbap_cols.append(ret_type_col)
    if rule_144a_col in df_osbap.columns:
        osbap_cols.append(rule_144a_col)

    osbap_full = df_osbap[osbap_cols].copy()
    osbap_full[date_col] = pd.to_datetime(osbap_full[date_col])
    osbap_full = osbap_full.rename(columns={ret_col_osbap: 'ret_osbap'})

    alt = df_alt[[date_col, id_col, ret_col_alt]].copy()
    alt[date_col] = pd.to_datetime(alt[date_col])
    alt = alt.rename(columns={ret_col_alt: 'ret_alt'})

    # Filter to common date range
    common_min = max(osbap_full[date_col].min(), alt[date_col].min())
    common_max = min(osbap_full[date_col].max(), alt[date_col].max())
    osbap_full = osbap_full[(osbap_full[date_col] >= common_min) & (osbap_full[date_col] <= common_max)]
    alt = alt[(alt[date_col] >= common_min) & (alt[date_col] <= common_max)]

    # Create OSBAP variants
    osbap = osbap_full[[date_col, id_col, 'ret_osbap']].copy()

    # No Def: exclude ret_type == 'trad_in_def'
    if ret_type_col in osbap_full.columns:
        mask_no_def = osbap_full[ret_type_col] != 'trad_in_def'
        osbap_no_def = osbap_full.loc[mask_no_def, [date_col, id_col, 'ret_osbap']].copy()
    else:
        osbap_no_def = osbap.copy()

    # No Def & 144a: exclude both
    if ret_type_col in osbap_full.columns and rule_144a_col in osbap_full.columns:
        mask_no_both = (osbap_full[ret_type_col] != 'trad_in_def') & (osbap_full[rule_144a_col] == 0)
        osbap_no_both = osbap_full.loc[mask_no_both, [date_col, id_col, 'ret_osbap']].copy()
    else:
        osbap_no_both = osbap.copy()

    # Helper function to compute market returns
    def compute_mkt(df_osbap_var, df_alt_var):
        mkt_o = df_osbap_var.dropna(subset=['ret_osbap']).groupby(
            date_col, observed=True
        )['ret_osbap'].mean().reset_index(name='mktb_osbap')
        mkt_a = df_alt_var.dropna(subset=['ret_alt']).groupby(
            date_col, observed=True
        )['ret_alt'].mean().reset_index(name='mktb_alt')
        mkt = mkt_o.merge(mkt_a, on=date_col, how='inner').dropna()
        mkt['mktb_osbap'] = mkt['mktb_osbap'] * 100
        mkt['mktb_alt'] = mkt['mktb_alt'] * 100
        return mkt

    # -------------------------------------------------------------------------
    # Panel A: Full observations - each database uses its own non-NaN values
    # -------------------------------------------------------------------------
    mkt_full = compute_mkt(osbap, alt)
    reg_full = run_regression_nw(y=mkt_full['mktb_osbap'].values, x=mkt_full['mktb_alt'].values)

    # -------------------------------------------------------------------------
    # Panel B: Overlapping only - matched cusip-month pairs
    # -------------------------------------------------------------------------
    merged = osbap.merge(alt, on=[date_col, id_col], how='inner')
    merged = merged.dropna(subset=['ret_osbap', 'ret_alt'])
    mkt_overlap = merged.groupby(date_col, observed=True).agg(
        mktb_osbap=('ret_osbap', 'mean'),
        mktb_alt=('ret_alt', 'mean'),
    ).reset_index().dropna()
    mkt_overlap['mktb_osbap'] = mkt_overlap['mktb_osbap'] * 100
    mkt_overlap['mktb_alt'] = mkt_overlap['mktb_alt'] * 100
    reg_overlap = run_regression_nw(y=mkt_overlap['mktb_osbap'].values, x=mkt_overlap['mktb_alt'].values)

    # -------------------------------------------------------------------------
    # Panel C: OSBAP (No Def) vs alternative
    # -------------------------------------------------------------------------
    mkt_no_def = compute_mkt(osbap_no_def, alt)
    reg_no_def = run_regression_nw(y=mkt_no_def['mktb_osbap'].values, x=mkt_no_def['mktb_alt'].values)
    tstat_no_def = paired_tstat_means(mkt_no_def['mktb_osbap'].values, mkt_no_def['mktb_alt'].values)

    # -------------------------------------------------------------------------
    # Panel D: OSBAP (No Def & 144a) vs alternative
    # -------------------------------------------------------------------------
    mkt_no_both = compute_mkt(osbap_no_both, alt)
    reg_no_both = run_regression_nw(y=mkt_no_both['mktb_osbap'].values, x=mkt_no_both['mktb_alt'].values)
    tstat_no_both = paired_tstat_means(mkt_no_both['mktb_osbap'].values, mkt_no_both['mktb_alt'].values)

    # t-stat for equality of means for Panel A and B
    tstat_full = paired_tstat_means(mkt_full['mktb_osbap'].values, mkt_full['mktb_alt'].values)
    tstat_overlap = paired_tstat_means(mkt_overlap['mktb_osbap'].values, mkt_overlap['mktb_alt'].values)

    # Helper function to plot a panel
    def plot_panel(ax, mkt_df, reg, tstat_mu, title, ylabel_label='OSBAP'):
        ax.scatter(mkt_df['mktb_alt'], mkt_df['mktb_osbap'],
                   s=20, alpha=0.6, color='steelblue', edgecolor='none')
        # 45-degree line
        lim_min = min(mkt_df['mktb_alt'].min(), mkt_df['mktb_osbap'].min())
        lim_max = max(mkt_df['mktb_alt'].max(), mkt_df['mktb_osbap'].max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], color='darkred', lw=1.5, ls='--')
        # Annotation with t-stat for equality of means
        annot = (
            f"$\\alpha$ = {reg['alpha']:.3f} ({reg['alpha_tstat']:.2f})\n"
            f"$\\beta$ = {reg['beta']:.3f} ({reg['beta_tstat']:.2f})\n"
            f"$R^2$ = {reg['r2']:.3f}\n"
            f"$\\rho$ = {reg['rho']:.3f}\n"
            f"$t_{{\\mu}}$ = {tstat_mu:.2f}"
        )
        ax.text(0.05, 0.95, annot, transform=ax.transAxes,
                fontsize=params.legend_size, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        ax.set_xlabel(f'{alt_name} ' + r'$\mathrm{MKTB}$ (%)')
        ax.set_ylabel(f'{ylabel_label} ' + r'$\mathrm{MKTB}$ (%)')
        ax.set_title(title)
        ax.axhline(0, color='gray', lw=0.5, ls='-')
        ax.axvline(0, color='gray', lw=0.5, ls='-')
        ax.grid(True, alpha=params.grid_alpha, lw=params.grid_lw)

    # Create figure with 2x2 panels
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    plot_panel(axes[0, 0], mkt_full, reg_full, tstat_full, 'Panel A: Full Observations', 'OSBAP')
    plot_panel(axes[0, 1], mkt_overlap, reg_overlap, tstat_overlap, 'Panel B: Overlapping Only', 'OSBAP')
    plot_panel(axes[1, 0], mkt_no_def, reg_no_def, tstat_no_def, 'Panel C: OSBAP (No Def)', 'OSBAP (No Def)')
    plot_panel(axes[1, 1], mkt_no_both, reg_no_both, tstat_no_both, 'Panel D: OSBAP (No Def & 144a)', 'OSBAP (No Def & 144a)')

    plt.tight_layout()

    # Save figure
    ext = params.export_format
    fig_path = output_dir / f"{filename}.{ext}"
    fig.savefig(fig_path, dpi=params.figure_dpi, transparent=params.transparent,
                bbox_inches='tight')
    plt.close(fig)

    # Caption text with optional hyperlink
    if alt_url is not None:
        alt_ref = r"\href{" + alt_url + r"}{" + alt_name + r"}"
    else:
        alt_ref = alt_name

    caption = (
        f"Scatter plot of monthly equal-weighted market returns (MKTB) comparing OSBAP "
        f"($y$-axis) and {alt_ref} ($x$-axis). Panel A: full observations. Panel B: overlapping cusip-month pairs. "
        r"Panel C: OSBAP (No Def) excludes trades in default. Panel D: OSBAP (No Def \& 144a) excludes both. "
        r"The dashed red line is the 45-degree line. "
        r"All $t$-statistics use Newey-West standard errors (lags = $\lfloor T^{0.25} \rfloor$). "
        r"The $t$-statistic for $\alpha$ tests $H_0$: $\alpha = 0$; "
        r"the $t$-statistic for $\beta$ tests $H_0$: $\beta = 1$; "
        f"$t_{{\\mu}}$ tests $H_0$: $\\mu_{{\\mathrm{{OSBAP}}}} = \\mu_{{\\mathrm{{{alt_name}}}}}$."
    )

    return fig_path, caption


def create_return_timeseries_plot(
    df_osbap: pd.DataFrame,
    df_alt: pd.DataFrame,
    output_dir: Path,
    filename: str = "return_timeseries_comparison",
    params: PlotParams = None,
    date_col: str = 'date',
    id_col: str = 'cusip',
    ret_col_osbap: str = 'ret_vw',
    ret_col_alt: str = 'ret_vw',
    alt_name: str = 'DFPS',
    alt_url: str = None,
) -> tuple:
    """
    Create return timeseries comparison plot (OSBAP vs alternative database).

    Panel A: Monthly EW market returns
    Panel B: Cumulative EW returns

    MKTB factors computed using respective non-NaN values for each database.

    Parameters
    ----------
    df_osbap : pd.DataFrame
        OSBAP panel with returns.
    df_alt : pd.DataFrame
        Alternative database with returns (e.g., DFPS, WRDS).
    output_dir : Path
        Directory to save the figure.
    filename : str
        Base filename.
    params : PlotParams
        Plotting parameters.
    date_col : str
        Date column name.
    id_col : str
        Bond identifier column.
    ret_col_osbap : str
        Return column in OSBAP data.
    ret_col_alt : str
        Return column in alternative data.
    alt_name : str
        Short name for alternative database (e.g., 'DFPS', 'WRDS').
    alt_url : str, optional
        URL for hyperlink in caption. If provided, alt_name becomes a hyperlink.

    Returns
    -------
    tuple
        (Path to saved figure, caption text).
    """
    from matplotlib.dates import AutoDateLocator, DateFormatter

    if params is None:
        params = PlotParams()

    apply_plot_params(params)

    # Prepare data
    osbap = df_osbap[[date_col, id_col, ret_col_osbap]].copy()
    osbap[date_col] = pd.to_datetime(osbap[date_col])

    alt = df_alt[[date_col, id_col, ret_col_alt]].copy()
    alt[date_col] = pd.to_datetime(alt[date_col])

    # Filter to common date range
    common_min = max(osbap[date_col].min(), alt[date_col].min())
    common_max = min(osbap[date_col].max(), alt[date_col].max())
    osbap = osbap[(osbap[date_col] >= common_min) & (osbap[date_col] <= common_max)]
    alt = alt[(alt[date_col] >= common_min) & (alt[date_col] <= common_max)]

    # Compute EW market returns using respective non-NaN values
    osbap_mkt = osbap.dropna(subset=[ret_col_osbap]).groupby(
        date_col, observed=True
    )[ret_col_osbap].mean().reset_index(name='mktb_osbap')

    alt_mkt = alt.dropna(subset=[ret_col_alt]).groupby(
        date_col, observed=True
    )[ret_col_alt].mean().reset_index(name='mktb_alt')

    # Merge on common dates and sort
    mkt = osbap_mkt.merge(alt_mkt, on=date_col, how='inner')
    mkt = mkt.sort_values(date_col).reset_index(drop=True)

    # Compute cumulative returns
    mkt['cum_osbap'] = (1 + mkt['mktb_osbap']).cumprod() - 1
    mkt['cum_alt'] = (1 + mkt['mktb_alt']).cumprod() - 1

    # Create figure with two panels
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    # Panel A: Monthly returns
    ax = axes[0]
    ax.plot(mkt[date_col], mkt['mktb_osbap'] * 100,
            color='steelblue', lw=1, alpha=0.8, label='OSBAP')
    ax.plot(mkt[date_col], mkt['mktb_alt'] * 100,
            color='darkred', lw=1, ls='--', alpha=0.8, label=alt_name)

    ax.set_ylabel('Monthly Return (%)')
    ax.set_title('Panel A: Monthly EW Market Returns', pad=5)
    ax.legend(loc='upper right', frameon=True, edgecolor='gray')
    ax.axhline(0, color='gray', lw=0.5)
    ax.grid(True, alpha=params.grid_alpha, lw=params.grid_lw)

    locator = AutoDateLocator(minticks=5, maxticks=10, interval_multiples=True)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.margins(x=0.01)

    # Panel B: Cumulative returns
    ax = axes[1]
    ax.plot(mkt[date_col], mkt['cum_osbap'] * 100,
            color='steelblue', lw=1.2, label='OSBAP')
    ax.plot(mkt[date_col], mkt['cum_alt'] * 100,
            color='darkred', lw=1.2, ls='--', label=alt_name)

    ax.set_ylabel('Cumulative Return (%)')
    ax.set_title('Panel B: Cumulative EW Returns', pad=5)
    ax.legend(loc='upper left', frameon=True, edgecolor='gray')
    ax.axhline(0, color='gray', lw=0.5)
    ax.grid(True, alpha=params.grid_alpha, lw=params.grid_lw)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.margins(x=0.01)

    plt.tight_layout()

    # Save figure
    ext = params.export_format
    fig_path = output_dir / f"{filename}.{ext}"
    fig.savefig(fig_path, dpi=params.figure_dpi, transparent=params.transparent,
                bbox_inches='tight')
    plt.close(fig)

    # Caption text with optional hyperlink
    if alt_url is not None:
        alt_ref = r"\href{" + alt_url + r"}{" + alt_name + r"}"
    else:
        alt_ref = alt_name

    caption = (
        f"Time-series comparison of equal-weighted market returns between OSBAP (solid blue) and {alt_ref} "
        r"(dashed red) databases. Each database computes MKTB using its own non-missing values. "
        r"Panel A shows monthly EW market returns. Panel B shows cumulative EW returns."
    )

    return fig_path, caption


def create_return_timeseries_exclusion_plot(
    df_osbap: pd.DataFrame,
    df_alt: pd.DataFrame,
    output_dir: Path,
    filename: str = "return_timeseries_exclusion",
    params: PlotParams = None,
    date_col: str = 'date',
    id_col: str = 'cusip',
    ret_col_osbap: str = 'ret_vw',
    ret_col_alt: str = 'ret_vw',
    ret_type_col: str = 'ret_type',
    rule_144a_col: str = '144a',
    alt_name: str = 'DFPS',
    alt_url: str = None,
) -> tuple:
    """
    Create return timeseries comparison plot: OSBAP (No Def & 144a) vs alternative.

    Panel A: Monthly EW market returns
    Panel B: Cumulative EW returns

    MKTB factors computed using respective non-NaN values for each database.
    OSBAP excludes both trades in default and 144a bonds.

    Parameters
    ----------
    df_osbap : pd.DataFrame
        OSBAP panel with returns.
    df_alt : pd.DataFrame
        Alternative database with returns (e.g., DFPS, WRDS).
    output_dir : Path
        Directory to save the figure.
    filename : str
        Base filename.
    params : PlotParams
        Plotting parameters.
    date_col : str
        Date column name.
    id_col : str
        Bond identifier column.
    ret_col_osbap : str
        Return column in OSBAP data.
    ret_col_alt : str
        Return column in alternative data.
    ret_type_col : str
        Return type column in OSBAP (for excluding 'trad_in_def').
    rule_144a_col : str
        Rule 144a indicator column in OSBAP.
    alt_name : str
        Short name for alternative database (e.g., 'DFPS', 'WRDS').
    alt_url : str, optional
        URL for hyperlink in caption. If provided, alt_name becomes a hyperlink.

    Returns
    -------
    tuple
        (Path to saved figure, caption text).
    """
    from matplotlib.dates import AutoDateLocator, DateFormatter

    if params is None:
        params = PlotParams()

    apply_plot_params(params)

    # Columns to keep from OSBAP
    osbap_cols = [date_col, id_col, ret_col_osbap]
    if ret_type_col in df_osbap.columns:
        osbap_cols.append(ret_type_col)
    if rule_144a_col in df_osbap.columns:
        osbap_cols.append(rule_144a_col)

    osbap_full = df_osbap[osbap_cols].copy()
    osbap_full[date_col] = pd.to_datetime(osbap_full[date_col])

    alt = df_alt[[date_col, id_col, ret_col_alt]].copy()
    alt[date_col] = pd.to_datetime(alt[date_col])

    # Filter to common date range
    common_min = max(osbap_full[date_col].min(), alt[date_col].min())
    common_max = min(osbap_full[date_col].max(), alt[date_col].max())
    osbap_full = osbap_full[(osbap_full[date_col] >= common_min) & (osbap_full[date_col] <= common_max)]
    alt = alt[(alt[date_col] >= common_min) & (alt[date_col] <= common_max)]

    # Apply exclusions: No Def & 144a
    if ret_type_col in osbap_full.columns and rule_144a_col in osbap_full.columns:
        mask = (osbap_full[ret_type_col] != 'trad_in_def') & (osbap_full[rule_144a_col] == 0)
        osbap = osbap_full.loc[mask, [date_col, id_col, ret_col_osbap]].copy()
    else:
        osbap = osbap_full[[date_col, id_col, ret_col_osbap]].copy()

    # Compute EW market returns using respective non-NaN values
    osbap_mkt = osbap.dropna(subset=[ret_col_osbap]).groupby(
        date_col, observed=True
    )[ret_col_osbap].mean().reset_index(name='mktb_osbap')

    alt_mkt = alt.dropna(subset=[ret_col_alt]).groupby(
        date_col, observed=True
    )[ret_col_alt].mean().reset_index(name='mktb_alt')

    # Merge on common dates and sort
    mkt = osbap_mkt.merge(alt_mkt, on=date_col, how='inner')
    mkt = mkt.sort_values(date_col).reset_index(drop=True)

    # Compute cumulative returns
    mkt['cum_osbap'] = (1 + mkt['mktb_osbap']).cumprod() - 1
    mkt['cum_alt'] = (1 + mkt['mktb_alt']).cumprod() - 1

    # Create figure with two panels
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    # Panel A: Monthly returns
    ax = axes[0]
    ax.plot(mkt[date_col], mkt['mktb_osbap'] * 100,
            color='steelblue', lw=1, alpha=0.8, label='OSBAP (No Def & 144a)')
    ax.plot(mkt[date_col], mkt['mktb_alt'] * 100,
            color='darkred', lw=1, ls='--', alpha=0.8, label=alt_name)

    ax.set_ylabel('Monthly Return (%)')
    ax.set_title('Panel A: Monthly EW Market Returns', pad=5)
    ax.legend(loc='upper right', frameon=True, edgecolor='gray')
    ax.axhline(0, color='gray', lw=0.5)
    ax.grid(True, alpha=params.grid_alpha, lw=params.grid_lw)

    locator = AutoDateLocator(minticks=5, maxticks=10, interval_multiples=True)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.margins(x=0.01)

    # Panel B: Cumulative returns
    ax = axes[1]
    ax.plot(mkt[date_col], mkt['cum_osbap'] * 100,
            color='steelblue', lw=1.2, label='OSBAP (No Def & 144a)')
    ax.plot(mkt[date_col], mkt['cum_alt'] * 100,
            color='darkred', lw=1.2, ls='--', label=alt_name)

    ax.set_ylabel('Cumulative Return (%)')
    ax.set_title('Panel B: Cumulative EW Returns', pad=5)
    ax.legend(loc='upper left', frameon=True, edgecolor='gray')
    ax.axhline(0, color='gray', lw=0.5)
    ax.grid(True, alpha=params.grid_alpha, lw=params.grid_lw)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.margins(x=0.01)

    plt.tight_layout()

    # Save figure
    ext = params.export_format
    fig_path = output_dir / f"{filename}.{ext}"
    fig.savefig(fig_path, dpi=params.figure_dpi, transparent=params.transparent,
                bbox_inches='tight')
    plt.close(fig)

    # Caption text with optional hyperlink
    if alt_url is not None:
        alt_ref = r"\href{" + alt_url + r"}{" + alt_name + r"}"
    else:
        alt_ref = alt_name

    caption = (
        f"Time-series comparison of equal-weighted market returns between OSBAP (No Def \\& 144a) "
        f"and {alt_ref} databases. OSBAP excludes trades in default and Rule 144a bonds. "
        r"Panel A shows monthly EW market returns. Panel B shows cumulative EW returns."
    )

    return fig_path, caption