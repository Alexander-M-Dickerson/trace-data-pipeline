# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import os

# --- Change your username here please ----
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "")

# --- FISD universe build params --------------------------------------
FISD_PARAMS = {
    # Switches for each screen
    "currency_usd_only": True,                 # foreign_currency == 'N'
    "fixed_rate_only": True,                   # coupon_type != 'V'
    "non_convertible_only": True,              # convertible == 'N'
    "non_asset_backed_only": True,             # asset_backed == 'N'
    "exclude_bond_types": True,                # drop certain bond_type codes
    "valid_coupon_frequency_only": True,       # drop invalid interest_frequency
    "require_accrual_fields": True,            # offering_date/dated_date etc. non-null
    "principal_amt_eq_1000_only": True,        # principal_amt == 1000
    "exclude_equity_index_linked": True,       # name contains 'EQUITY-LINKED' / 'INDEX-LINKED'
    "enforce_tenor_min": True,                 # tenor >= tenor_min_years

    # Knobs/sets
    "invalid_coupon_freq": [-1, 13, 14, 15, 16],
    "excluded_bond_types": [
        "TXMU","CCOV","CPAS","MBS","FGOV","USTC","USBD","USNT","USSP","USSI",
        "FGS","USBL","ABS","O30Y","O10Y","O5Y","O3Y","O4W","O13W","O26W","O52W",
        "CCUR","ADEB","AMTN","ASPZ","EMTN","ADNT","ARNT","TPCS","CPIK","PS","PSTK"
    ],
    "tenor_min_years": 1.0,
}


# --- Filter switchboard (True = apply, False = skip) -----------------
FILTER_SWITCHES =  dict(
    dick_nielsen            = True,  # 1: clean_trace_chunk()
    decimal_shift_corrector = True,  # 2: decimal_shift_corrector() [note:see ds_params]
    trading_time            = False, # 3: filter_by_trade_time()
    trading_calendar        = True,  # 4: filter_by_calendar()
    price_filters           = True,  # 5: > 0 (not neg) & <= 1000 price screens
    volume_filter_toggle    = True,  # 6: dollar_vol >= threshold   [note: renamed key to disambiguate]
    bounce_back_filter      = True,  # 7: flag_price_change_errors() [note:see bb_params]
    yld_price_filter        = True,  # 8: rptd_pr != yld_pt
    amtout_volume_filter    = True,  # 9: entrd_vol_qt < 0.5*offamt*1000
    trd_exe_mat_filter      = True,  # 10: trd_exctn_dt <= maturity
)

# --- Decimal-shift corrector params ---------------------------------
DS_PARAMS = {
    "factors": (0.1, 0.01, 10.0, 100.0),
    "tol_pct_good": 0.02,
    "tol_abs_good": 8.0,
    "tol_pct_bad": 0.05,
    "low_pr": 5.0,
    "high_pr": 300.0,
    "anchor": "rolling",
    "window": 5,
    "improvement_frac": 0.2,
    "par_snap": True,
    "par_band": 15.0,
    "output_type": "cleaned",
}

# --- Bounce-back (price-change) filter params -------------------------
BB_PARAMS = {
    "threshold_abs": 35.0,
    "lookahead": 5,
    "max_span": 5,
    "window": 5,
    "back_to_anchor_tol": 0.25,
    "candidate_slack_abs": 1.0,
    "reassignment_margin_abs": 5.0,
    "use_unique_trailing_median": True,
    "par_spike_heuristic": True,
    "par_level": 100.0,
    "par_equal_tol": 1e-8,
    "par_min_run": 3,
    "par_cooldown_after_flag": 2,
}

# --- Arguments identical across all runners ---------------------------
COMMON_KWARGS = dict(
    wrds_username = WRDS_USERNAME,    
    output_format = "parquet",
    chunk_size    = 250,
    clean_agency  = True,
    out_dir       = "",
    volume_filter = ("dollar", 10000),
    trade_times   = ["00:00:00", "23:59:59"],  # Filter switched off as default
    calendar_name = "NYSE",
    ds_params     = DS_PARAMS,
    bb_params     = BB_PARAMS,
    filters       = FILTER_SWITCHES,
    fisd_params   = FISD_PARAMS
)


# --- Per-dataset overrides (only where needed) ------------------------
PER_DATASET = {
    "enhanced": dict(),  # no extra args required
    "standard": dict(start_date="2024-10-01", data_type="standard"),
    "144a":     dict(start_date="2002-07-01", data_type="144a"),
}

def get_config(kind: str) -> dict:
    """
    Returns a full kwargs dict for CreateDaily*TRACE calls.
    kind in {"enhanced", "standard", "144a"}.
    """
    overrides = PER_DATASET.get(kind, {})
    return {**COMMON_KWARGS, **overrides}
