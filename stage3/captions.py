"""captions.py -- the caption printed above each table, keyed by its LaTeX label.

One place, so a driver is its design and nothing else, and so the full list of
exhibits Stage 3 produces can be read at a glance. Figures carry their titles in
their own plotting code, where the axes are.
"""
from __future__ import annotations

CAPTIONS: dict[str, str] = {
    # --- the data appendix (Section S0) ---------------------------------------
    "tab:data_availability":
        "Data Availability by Rating Category.",
    "tab:descriptive_stats":
        "TRACE daily descriptive statistics.",
    "tab:monthly_data_availability":
        "Monthly Panel Data Availability by Rating Category.",
    "tab:desc_stats_all":
        "Monthly Panel Descriptive Statistics --- All Corporate Bonds.",
    "tab:extreme_returns":
        "Extreme Returns Analysis.",
    "tab:time_concentration_extremes":
        "Time Concentration of Extreme Returns.",
    "tab:annual_return_stats":
        "Annual Return Summary Statistics.",
    "tab:bounce_params":
        "Bounce-back filter parameters.",
    "tab:decimal_params":
        "Decimal shift corrector parameters.",
    "tab:distressed_params":
        "Distressed bond filter parameters.",

    # --- latent implementation bias (Section 3) -------------------------------
    "tab:mmn_1":
        "Latent implementation bias in price-based factors.",
    "tab:mmn_2":
        "Latent implementation bias validation for price-based factors.",
    "tab:mmn_app_1":
        "Latent implementation bias in price-based factors (value-weighted, single-sort).",
    "tab:mmn_app_2":
        "Latent implementation bias in price-based factors (value-weighted, within-firm).",
    "tab:illiq_1":
        "Latent implementation bias in illiquidity factors.",
    "tab:lib_summary":
        "Latent implementation bias across 108 factors.",

    # --- look-ahead bias (Section 4) ------------------------------------------
    "tab:lab_ls_1":
        "Look-ahead bias by factor.",
    "tab:lab_affected_factors":
        "Corporate bond factors sensitive to ex-post return filtering.",
    "tab:lab_full_mean_8":
        "Look-ahead bias decomposition by leg (mean return).",
    "tab:lab_full_alpha_9":
        "Look-ahead bias decomposition by leg (CAPMB alpha).",

    # --- non-standard errors (Section 5) --------------------------------------
    "tab:nse_by_cluster":
        "Non-standard errors and data uncertainty by factor cluster.",
    "tab:mu_nse_by_cluster":
        "Non-standard errors and methodological uncertainty by factor cluster.",
    "tab:filter_paths":
        "Filter path improvement analysis by factor cluster, filter type, and tail location.",
    "tab:mua_portfolio_size":
        "Cross-sectional portfolio size by design dimension.",
    "tab:mua_improvement":
        "Portfolio construction improvement analysis by factor cluster and methodology "
        "dimension.",

    # --- the factor zoo (Section IA) ------------------------------------------
    "tab:vw_results":
        "Factors with significant alphas after bias correction (value-weighted).",
    "tab:ew_results":
        "Factors with significant alphas after bias correction (equal-weighted).",
    "tab:fdr_by_cluster":
        "FDR survivors by factor cluster.",
}


def caption(label: str, note: str = "") -> str:
    """The caption for a LaTeX label. Unknown labels fail here rather than printing
    an empty caption into a table nobody then notices is unlabelled.

    ❗`note` is appended verbatim and is where the SAMPLE SENTENCE goes -- render it
    with `drrlib.sample_sentence(block)` and pass it in. The titles above are the
    authors' and never change; what has to track the data is the sentence that says
    which months and how many observations produced the numbers. Keeping the two apart
    means this module stays free of dependencies and the note stays derived.
    """
    try:
        return CAPTIONS[label] + (" " + note if note else "")
    except KeyError:
        raise KeyError(
            f"no caption registered for {label!r}. Add it to captions.CAPTIONS -- "
            "an exhibit without one renders a table with an empty title."
        ) from None
