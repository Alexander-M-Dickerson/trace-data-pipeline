# -*- coding: utf-8 -*-
"""
_build_data_report.py
=====================
Build the Stage 2 data report: a LaTeX document of tables and figures describing
the monthly asset-pricing panel, plus the PDF if a TeX toolchain is installed.

The report answers, for whoever downloads the panel: what is in it, how much of it
is populated, how the numbers are distributed, and how it compares with the other
TRACE-derived bond databases in circulation.

    python3 _build_data_report.py                    # the current build, everything
    python3 _build_data_report.py --no-external      # skip the DFPS/WRDS comparisons
    python3 _build_data_report.py --no-pdf           # emit .tex only
    python3 _build_data_report.py --mode prod_ext    # a specific build

Two comparison suites need the network (and WRDS credentials). They are ON by
default because a data report that does not compare against the alternatives is
not making the case; ``--no-external`` drops them and the report still builds.

The heavy lifting lives in ``_report_helpers.py``, ported unchanged from the
reference implementation. This file is the sequence, not the statistics.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import argparse
import gc
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _stage2_settings as cfg
import _report_helpers as rpt

WRDS_URL = ("https://wrds-www.wharton.upenn.edu/pages/get-data/wrds-bond-returns/"
            "wrds-bond-returns/")
DFPS_LONG = (r'DFPS (Dick-Nielsen, Feldh{\"u}tter, Pedersen, Schneider) TRACE database')


def _box(title: str, body: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}\n{body}\n{line}\n", flush=True)


def _say(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# The variable contract
# ---------------------------------------------------------------------------
# Every statistic below is requested by column name. The reference implementation
# skipped a name it could not find (`if var_name not in df.columns: continue`), so a
# renamed column removed a row from every table with no error anywhere -- three of
# them were missing from the shipped report and nobody noticed. Names are checked
# up front now, once, against the panel.

def require_columns(df: pd.DataFrame, requested: list[tuple[str, str]],
                    what: str, derived: set[str] = frozenset()) -> None:
    """Fail loudly if a requested variable is not in the panel."""
    missing = [(c, label) for c, label in requested
               if c not in df.columns and c not in derived]
    if not missing:
        return
    have = sorted(df.columns)
    lines = []
    for c, label in missing:
        near = [h for h in have if h.startswith(c[:3]) or c.startswith(h[:3])][:5]
        lines.append(f"    {c!r} ({label})" + (f"   did you mean: {near}" if near else ""))
    raise KeyError(
        f"{what}: {len(missing)} requested variable(s) are not columns of the panel.\n"
        + "\n".join(lines)
        + "\n\n  A missing name used to be skipped silently, which deleted the row from "
          "every\n  table. Fix the name (or the panel) rather than dropping the variable."
    )


def build_report(mode: str, report_dir: Path, external: bool, make_pdf: bool,
                 wrds_username: str) -> Path:
    t0 = time.time()
    report_dir.mkdir(parents=True, exist_ok=True)
    _say(f"  output directory: {report_dir}")

    # --- Load the panel -----------------------------------------------------
    panel_path = cfg.PANEL_DIR / f"main_panel_{mode}.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"no panel at {panel_path}.\n"
            f"    Run a build first, or pass --mode for a build you have.")
    _say(f"  loading {panel_path.name} ...")
    df = pd.read_parquet(panel_path)
    _say(f"    {len(df):,} rows, {len(df.columns)} cols")

    min_date = df["date"].min().strftime("%Y-%m-%d")
    max_date = df["date"].max().strftime("%Y-%m-%d")
    _say(f"    {min_date} -> {max_date}")

    # --- Contiguous month-end panel (for the availability table) ------------
    _say("  resampling to a contiguous monthly panel ...")
    df_resampled = rpt.build_month_end_panel(df, id_col="cusip", date_col="date")
    _say(f"    {len(df_resampled):,} rows")

    # --- Table 1: data availability -----------------------------------------
    _say("  table 1: data availability")
    # `bbtm` is book-to-market, not a price: step1_returns.py:251 builds it as 100/pr,
    # so the price is 100/bbtm. The reference multiplied instead of dividing, which
    # inverts the tails (bbtm p99 = 1.67 is a price of 60, not 167). The availability
    # table only counts non-nulls, so its numbers are unaffected either way.
    df_resampled["pr"] = 100 / df_resampled["bbtm"]
    avail_vars = [
        ("pr", "Price (VW)"),
        ("ret_vw", "Month-End Return"),
        ("ret_vw_bgn", "Month-Begin Return"),
        ("ytm", "YTM"),
        ("cs", "Spread"),
        ("spc_rat", "Composite Rating (SP)"),
        ("mdc_rat", "Composite Rating (MD)"),
        ("permno", "PERMNO"),
    ]
    require_columns(df_resampled, avail_vars, "Table 1 (data availability)")
    tbl_avail = rpt.make_data_availability_table(
        df=df_resampled, min_date=min_date, max_date=max_date,
        variables=avail_vars, id_col="cusip", date_col="date", rating_col="spc_rat")

    # --- Descriptive statistics ---------------------------------------------
    _say("  descriptive statistics by rating category")
    if "tret" in df.columns:
        df["ret_vwx"] = df["ret_vw"] - df["tret"]
        df["ret_vwx_bgn"] = df["ret_vw_bgn"] - df["tret"]
    df["pr"] = 100 / df["bbtm"]          # the reference never built this on the stats frame

    stat_vars = [
        ("ret_vw", "Total End Return (%)"),
        ("ret_vw_bgn", "Total Begin Return (%)"),
        ("ret_vwx", "Dur. Adj. End Return (%)"),
        ("ret_vwx_bgn", "Dur. Adj. Begin Return (%)"),
        ("lib", "Latent Imp. Bias"),
        ("hprd", "End Holding Period"),
        ("hprd_bgn", "Begin Holding Period"),
        ("igap_bgn", "Implementation Gap"),
        ("sig_gap", "Signal Gap"),
        ("pr", "Price (VW)"),
        ("ytm", "YTM (%)"),
        ("cs", "Spread (%)"),
        ("md_dur", "Duration (Modified)"),
        ("tmat", "Bond Maturity"),
        ("age", "Bond Age"),
        ("convx", "Convexity"),
        ("sze", "Market Cap."),
        ("spc_rat", "Composite Rating (SP)"),
        ("mdc_rat", "Composite Rating (MD)"),
        ("spd_rel", "Bid-Ask Spread (%)"),
    ]
    scale_vars = {
        "ret_vw": 100, "ret_vw_bgn": 100,
        "ret_vwx": 100, "ret_vwx_bgn": 100,
        "ytm": 100, "cs": 100,
        "spd_rel": 100,
    }

    require_columns(df, stat_vars, "Tables 2-6 (descriptive statistics)")

    df_all = df
    df_ig = df[(df["spc_rat"] >= 1) & (df["spc_rat"] <= 10)]
    df_nig = df[(df["spc_rat"] > 10) & (df["spc_rat"] <= 21)]
    df_def = df[df["spc_rat"] == 22]
    df_144a = df[df["144a"] == 1]

    n_obs_all = df_all["ret_vw"].notna().sum()
    n_obs_ig = df_ig["ret_vw"].notna().sum()
    n_obs_nig = df_nig["ret_vw"].notna().sum()
    n_obs_def = df_def["ret_vw"].notna().sum()
    n_obs_144a = df_144a["ret_vw"].notna().sum()
    for label, part in (("all bonds", df_all), ("investment grade", df_ig),
                        ("non-investment grade", df_nig), ("defaulted", df_def),
                        ("144a", df_144a)):
        _say(f"    {label:<22s} {len(part):>10,} rows ({100*len(part)/len(df_all):5.2f}%)")

    cuts = [
        ("stats_all", df_all, n_obs_all, 2, "All Corporate Bonds",
         "All ratings", "tab:desc_stats_all"),
        ("stats_ig", df_ig, n_obs_ig, 3, "Investment Grade Corporate Bonds",
         "Ratings 1-10 (AAA to BBB-)", "tab:desc_stats_ig"),
        ("stats_nig", df_nig, n_obs_nig, 4, "Non-Investment Grade Corporate Bonds",
         "Ratings 11-21 (BB+ to CCC-)", "tab:desc_stats_nig"),
        ("stats_def", df_def, n_obs_def, 5, "Defaulted Corporate Bonds",
         "Rating 22 (D)", "tab:desc_stats_def"),
        ("stats_144a", df_144a, n_obs_144a, 6, "144a Corporate Bonds",
         "All ratings, Rule 144a issues", "tab:desc_stats_144a"),
    ]
    stats_tables, panels = {}, []
    for key, part, n_obs, num, title, rating_text, label in cuts:
        _say(f"  table {num}: {title}")
        pa = rpt.compute_pooled_stats(part, stat_vars, scale_vars)
        pb = rpt.compute_cross_sectional_stats(part, stat_vars, "date", scale_vars)
        panels.append((pa, pb))
        stats_tables[key] = rpt.make_descriptive_stats_table_by_rating(
            panel_a=pa, panel_b=pb, min_date=min_date, max_date=max_date,
            table_number=num, title=title, rating_range_text=rating_text,
            label=label, n_obs=n_obs, n_total=n_obs_all)

    # --- Table 7: signal definitions ----------------------------------------
    _say("  table 7: signal definitions")
    tbl_signal_defs = rpt.build_signal_definitions_table()

    # --- Table 8: extreme returns -------------------------------------------
    _say("  table 8: extreme returns")
    extreme_stats = rpt.compute_combined_extreme_stats(
        dfs={"All": df_all, "IG": df_ig, "NIG": df_nig, "Def": df_def},
        ret_cols=[("ret_vw", "End"), ("ret_vw_bgn", "Begin")],
        thresholds=[0.20, 0.50, 1.00])
    outlier_section = r"""
\subsection{Outlier Analysis}

Tables~\ref{tab:extreme_returns}--\ref{tab:annual_return_stats} present analyses of extreme and outlier returns. Table~\ref{tab:extreme_returns} reports the frequency and characteristics of extreme monthly returns exceeding various thresholds. Table~\ref{tab:time_concentration_extremes} examines the time concentration of these extreme observations. Table~\ref{tab:annual_return_stats} provides annual summary statistics for bond returns.

"""
    tbl_extreme = outlier_section + rpt.make_combined_extreme_table(
        stats=extreme_stats, min_date=min_date, max_date=max_date)

    # --- Table 9: time concentration ----------------------------------------
    _say("  table 9: time concentration of extremes")
    time_thresholds = [0.20, 0.95]
    time_conc_stats = rpt.compute_time_concentration_stats(
        df_all, date_col="date",
        ret_cols=[("ret_vw", "End"), ("ret_vw_bgn", "Begin")],
        thresholds=time_thresholds)
    tbl_time_conc = rpt.make_time_concentration_table(
        stats_df=time_conc_stats, min_date=min_date, max_date=max_date,
        thresholds=time_thresholds)

    # --- Table 10: annual return statistics ---------------------------------
    _say("  table 10: annual return statistics")
    annual_stats = rpt.compute_annual_return_stats(
        df_all, date_col="date",
        ret_cols=[("ret_vw", "End"), ("ret_vw_bgn", "Begin")])
    tbl_annual_stats = rpt.make_annual_return_stats_table(
        stats_df=annual_stats, min_date=min_date, max_date=max_date)

    tables = {
        "data_availability": tbl_avail,
        **stats_tables,
        "signal_defs": tbl_signal_defs,
        "extreme_returns": tbl_extreme,
        "time_concentration": tbl_time_conc,
        "annual_return_stats": tbl_annual_stats,
    }

    # --- Figures 1-3 (core) --------------------------------------------------
    fig_filenames: list[tuple[str, str]] = []

    def _fig(n: int, blurb: str, fn, **kw):
        _say(f"  figure {n}: {blurb}")
        path, caption = fn(output_dir=report_dir, params=rpt.PlotParams(), **kw)
        fig_filenames.append((path.name, caption))
        return path

    _fig(1, "market return scatter", rpt.create_market_return_scatter,
         df=df_all, filename="fig1_market_return_scatter", date_col="date",
         ret_end_col="ret_vw", ret_bgn_col="ret_vw_bgn", weight_col="mcap_s")
    _fig(2, "dynamics of default", rpt.create_dynamics_of_default_plot,
         df=df_all, filename="fig2_dynamics_of_default")
    _fig(3, "dynamics of 144a", rpt.create_dynamics_of_144a_plot,
         df=df_all, filename="fig3_dynamics_of_144a")

    # --- External comparison suites -----------------------------------------
    if external:
        _say("\n  --- DFPS comparison ---")
        df_dfps = rpt.load_dfps_data(
            cache_dir=report_dir,
            local_path=cfg.STAGE2_DATA / "trace_alternate_2025_12_2024.parquet")
        _say(f"    DFPS: {len(df_dfps):,} rows")

        dfps_section = r"""
\subsection{Comparison with Alternative TRACE Data -- DFPS}

We compare our OSBAP monthly panel with the alternative TRACE dataset constructed by DFPS. Tables~\ref{tab:database_coverage_dfps} and \ref{tab:return_comparison_dfps} present coverage and return comparison statistics. Figures~\ref{fig:figure_4}--\ref{fig:figure_7} provide graphical comparisons of coverage, market returns, and return distributions across the two databases.

"""
        _say("  table 11: DFPS coverage comparison")
        tables["db_coverage"] = dfps_section + rpt.make_database_comparison_table(
            df_osbap=df_all, df_alt=df_dfps, min_date=min_date, max_date=max_date,
            ret_col_alt="ret_vw_dfps", alt_name="DFPS",
            alt_description=DFPS_LONG, table_label="tab:database_coverage_dfps")
        _say("  table 12: DFPS return comparison")
        tables["ret_compare"] = rpt.make_return_comparison_table(
            df_osbap=df_all, df_alt=df_dfps, min_date=min_date, max_date=max_date,
            ret_col_alt="ret_vw_dfps", alt_name="DFPS",
            table_label="tab:return_comparison_dfps")

        for n, blurb, fn, fname in (
                (4, "DFPS coverage", rpt.create_coverage_comparison_plot,
                 "fig4_coverage_comparison"),
                (5, "DFPS mktb scatter", rpt.create_mktb_comparison_scatter,
                 "fig5_mktb_comparison_scatter"),
                (6, "DFPS return timeseries", rpt.create_return_timeseries_plot,
                 "fig6_return_timeseries_comparison"),
                (7, "DFPS return timeseries (no def & 144a)",
                 rpt.create_return_timeseries_exclusion_plot,
                 "fig7_return_timeseries_exclusion")):
            _fig(n, blurb, fn, df_osbap=df_all, df_alt=df_dfps, filename=fname,
                 ret_col_alt="ret_vw_dfps", alt_name="DFPS")
        del df_dfps
        gc.collect()

        _say("\n  --- WRDS Bond Returns comparison ---")
        if not wrds_username:
            raise ValueError(
                "the WRDS comparison needs a username: set WRDS_USERNAME in the "
                "environment, or pass --wrds-username, or use --no-external.")
        df_wrds = rpt.load_wrds_data(wrds_username=wrds_username, cache_dir=report_dir,
                                     cache_filename="wrds_bondret.parquet")
        _say(f"    WRDS: {len(df_wrds):,} rows")

        wrds_section = (r"""
\subsection{Comparison with Alternative TRACE Data -- \href{""" + WRDS_URL + r"""}{WRDS Bond Returns}}

We compare our OSBAP monthly panel with the \href{""" + WRDS_URL + r"""}{WRDS Bond Returns} database. Tables~\ref{tab:database_coverage_wrds} and \ref{tab:return_comparison_wrds} present coverage and return comparison statistics. Figures~\ref{fig:figure_8}--\ref{fig:figure_11} provide graphical comparisons of coverage, market returns, and return distributions across the two databases.

""")
        _say("  table 13: WRDS coverage comparison")
        tables["db_coverage_wrds"] = wrds_section + rpt.make_database_comparison_table(
            df_osbap=df_all, df_alt=df_wrds, min_date=min_date, max_date=max_date,
            ret_col_alt="ret_l5m", alt_name="WRDS",
            alt_description=r"\href{" + WRDS_URL + r"}{WRDS Bond Returns} database",
            alt_url=WRDS_URL, table_label="tab:database_coverage_wrds")
        _say("  table 14: WRDS return comparison")
        tables["ret_compare_wrds"] = rpt.make_return_comparison_table(
            df_osbap=df_all, df_alt=df_wrds, min_date=min_date, max_date=max_date,
            ret_col_alt="ret_l5m", alt_name="WRDS", alt_url=WRDS_URL,
            table_label="tab:return_comparison_wrds")

        for n, blurb, fn, fname in (
                (8, "WRDS coverage", rpt.create_coverage_comparison_plot,
                 "fig8_wrds_coverage_comparison"),
                (9, "WRDS mktb scatter", rpt.create_mktb_comparison_scatter,
                 "fig9_wrds_mktb_comparison_scatter"),
                (10, "WRDS return timeseries", rpt.create_return_timeseries_plot,
                 "fig10_wrds_return_timeseries_comparison"),
                (11, "WRDS return timeseries (no def & 144a)",
                 rpt.create_return_timeseries_exclusion_plot,
                 "fig11_wrds_return_timeseries_exclusion")):
            _fig(n, blurb, fn, df_osbap=df_all, df_alt=df_wrds, filename=fname,
                 ret_col_alt="ret_l5m", alt_name="WRDS")
        del df_wrds
        gc.collect()
    else:
        _say("\n  external comparison suites skipped (--no-external)")

    del df, df_resampled, df_all, df_ig, df_nig, df_def, df_144a, panels
    del extreme_stats, time_conc_stats, annual_stats
    gc.collect()

    # --- Assemble ------------------------------------------------------------
    _say("\n  building the LaTeX document ...")
    doc = rpt.build_latex_document(tables=tables, fig_filenames=fig_filenames,
                                   author=cfg.AUTHOR, vintage=cfg.release_vintage())
    tex_path = report_dir / f"stage2_data_report_{cfg.DATE_STAMP}.tex"
    tex_path.write_text(doc, encoding="utf-8")
    _say(f"    wrote {tex_path.name}")

    bib_path = report_dir / "references.bib"
    bib_path.write_text(rpt.get_references_bib(), encoding="utf-8")
    _say(f"    wrote {bib_path.name}")

    if make_pdf:
        compile_pdf(tex_path)

    _say(f"\n  tables : {len(tables)}  ({', '.join(tables)})")
    _say(f"  figures: {len(fig_filenames)}")
    _say(f"  wall   : {time.time() - t0:.0f}s")
    return tex_path


def compile_pdf(tex_path: Path) -> Path | None:
    """Compile with pdflatex + bibtex. Absent toolchain is a warning, not a failure."""
    exe = shutil.which("pdflatex")
    if not exe:
        _say("  pdflatex not found -- .tex written, PDF skipped (install TeX, or --no-pdf)")
        return None
    d, stem = tex_path.parent, tex_path.stem
    _say("  compiling the PDF ...")
    for i, cmd in enumerate([[exe, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                             [shutil.which("bibtex") or "bibtex", stem],
                             [exe, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                             [exe, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]]):
        if cmd[0] is None:
            continue
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
        # bibtex fails harmlessly when nothing is cited; only the LaTeX passes matter
        if r.returncode != 0 and i != 1:
            tail = "\n".join((r.stdout or "").splitlines()[-25:])
            _say(f"  pdflatex pass {i} failed:\n{tail}")
            return None
    pdf = d / f"{stem}.pdf"
    if pdf.exists():
        _say(f"    wrote {pdf.name} ({pdf.stat().st_size/1e6:.1f} MB)")
        return pdf
    _say("  pdflatex reported success but produced no PDF")
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="_build_data_report.py",
        description="Build the Stage 2 data report (LaTeX + figures + PDF).")
    p.add_argument("--mode", default=None,
                   help=f"which build to report on (default {cfg.INPUT_MODE}).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="where to write (default stage2/data_reports/).")
    p.add_argument("--no-external", action="store_true",
                   help="skip the DFPS and WRDS comparison suites (no network needed).")
    p.add_argument("--no-pdf", action="store_true",
                   help="write the .tex but do not run pdflatex.")
    p.add_argument("--wrds-username", default=None,
                   help="WRDS username for the Bond Returns comparison.")
    args = p.parse_args(argv)

    mode = args.mode or cfg.INPUT_MODE
    report_dir = args.out_dir or cfg.REPORT_DIR
    _box("STAGE 2 DATA REPORT",
         f"  build    : {mode}\n"
         f"  output   : {report_dir}\n"
         f"  external : {'no' if args.no_external else 'DFPS + WRDS'}\n"
         f"  pdf      : {'no' if args.no_pdf else 'yes, if pdflatex is installed'}")
    try:
        build_report(mode=mode, report_dir=report_dir, external=not args.no_external,
                     make_pdf=not args.no_pdf,
                     wrds_username=args.wrds_username or cfg.WRDS_USERNAME)
    except Exception as exc:
        _box(f"DATA REPORT FAILED: {type(exc).__name__}", str(exc))
        raise
    finally:
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
