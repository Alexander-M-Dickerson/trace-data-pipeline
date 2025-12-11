# ----------------------------------------
# Libraries
# ----------------------------------------
import pandas as pd
import numpy as np
import QuantLib as ql
from joblib import Parallel, delayed
import wrds
import zipfile
import csv
import gzip
import gc  # Added for garbage collection
gc.collect()
from tqdm import tqdm
import logging
# import pandas_datareader as pdr
import requests
from zipfile import ZipFile
import sys, gc, io
from pathlib import Path
from pandas.tseries.offsets import MonthEnd
from typing import Callable, Optional, List
from numba import jit
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')
tqdm.pandas()
import matplotlib
import os
if not os.environ.get("DISPLAY"):
    os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter
from dataclasses import dataclass

def load_and_process_trace_file(filepath, date_column='trd_exctn_dt'):
    """Load a TRACE parquet file and convert date columns to datetime."""
    df = pd.read_parquet(filepath)
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
    return df


def Timestamp2Date(ts):
    """Convert pandas Timestamp to QuantLib Date"""
    return ql.Date(ts.day, ts.month, min(2199, ts.year))


def Date2Timestamp(d):
    """Convert QuantLib Date to pandas Timestamp"""
    return pd.Timestamp(d.year(), d.month(), d.dayOfMonth())


def GetNewVarsPy(x):
    """
    Calculate bond analytics using QuantLib
    
    Args:
        x: Bond data object with required attributes
        
    Returns:
        Tuple of bond analytics including yields, prices, duration, convexity
    """
    # Extract market clean price
    MktCleanPrice = x.pr
    
    # Convert dates
    IssueDate = Timestamp2Date(x.offering_date)
    StartDate = Timestamp2Date(x.dated_date) if not pd.isna(x.dated_date) \
        else Timestamp2Date(x.offering_date)
    TransactionDate = Timestamp2Date(x.trd_exctn_dt)
    
    # Calculate settlement date (T+2)
    SettlementDate = ql.UnitedStates(ql.UnitedStates.NYSE).advance(
            TransactionDate, 2, ql.Days, ql.ModifiedFollowing
        )
    sttldt = Date2Timestamp(SettlementDate)
    
    # Convert maturity date
    MaturityDate = Timestamp2Date(x.maturity)
    
    # Set day count convention
    if x.day_count_basis in ["30/360", ""]:
        DayCountBasis = ql.Thirty360(ql.Thirty360.BondBasis)
    elif x.day_count_basis == "ACT/ACT":
        DayCountBasis = ql.ActualActual(ql.ActualActual.ISDA)
    elif x.day_count_basis == "ACT/360":
        DayCountBasis = ql.Actual360()
    elif x.day_count_basis in ["ACT/365", "ACT/366"]:
        DayCountBasis = ql.Actual365Fixed()
    else:
        raise ValueError("Invalid day_count_basis", x)
    
    # Set interest payment frequency
    if x.interest_frequency == '1':
        InterestFrequency = ql.Annual
    elif x.interest_frequency == '2':
        InterestFrequency = ql.Semiannual
    elif x.interest_frequency == '4':
        InterestFrequency = ql.Quarterly
    elif x.interest_frequency == '12':
        InterestFrequency = ql.Monthly
    elif x.interest_frequency in ['0', '99']:
        # Default to semiannual for coupon bonds with missing frequency
        if x.coupon > 0 and not np.isnan(x.coupon):
            InterestFrequency = ql.Semiannual
        else:
            InterestFrequency = ql.NoFrequency
    else:
        raise ValueError('Invalid interest_frequency', x)
    
    # Convert coupon rate from percentage to decimal
    Coupon = x.coupon / 100
    
    # Construct appropriate bond object based on type
    if x.coupon_type == 'Z' or (x.coupon_type == 'F'
                                and (x.coupon == 0 or np.isnan(x.coupon))
                                and MktCleanPrice < 100):
        # Zero Coupon Bond
        bond = ql.ZeroCouponBond(
            2,                                         # settlement days
            ql.UnitedStates(ql.UnitedStates.NYSE),     # calendar
            100,                                       # face amount
            MaturityDate,                              # maturity date
            ql.ModifiedFollowing,                      # payment convention
            100,                                       # redemption
            IssueDate                                  # issue date
        )
        # Set annual frequency for zero coupon bonds
        InterestFrequency = ql.Annual
        
    elif x.coupon_type == 'F' and x.coupon > 0 and not np.isnan(x.coupon):
        # Fixed Rate Bond - Create schedule first
        schedule = ql.Schedule(
            StartDate,                             # effective date
            MaturityDate,                          # termination date
            ql.Period(InterestFrequency),          # tenor
            ql.UnitedStates(ql.UnitedStates.NYSE), # calendar
            ql.ModifiedFollowing,                  # convention
            ql.ModifiedFollowing,                  # termination date convention
            ql.DateGeneration.Backward,            # rule - generate dates from maturity backward
            False                                  # end of month - don't force end-of-month payments
        )
        
        # Create fixed rate bond
        bond = ql.FixedRateBond(
            2,                       # settlement days
            100,                     # face amount
            schedule,                # payment schedule
            [Coupon],                # coupon rates (as list)
            DayCountBasis,           # accrual day counter
            ql.ModifiedFollowing,    # payment convention
            100,                     # redemption amount
            IssueDate                # issue date
        )
    else:
        bond = None
    
    # Initialize results
    ytm = np.nan
    prclean = np.nan
    prfull = np.nan
    acclast = np.nan
    accpmt = np.nan
    accall = np.nan
    mac_bond = np.nan
    dur_bond = np.nan
    conv_bond = np.nan
    
    # Calculate analytics if bond is valid and price is finite
    if bond is not None and sttldt < x.maturity and np.isfinite(MktCleanPrice):
        try:
            # Calculate yield to maturity using the bond's true frequency
            ytm = bond.bondYield(
                MktCleanPrice,             # clean price
                DayCountBasis,             # day counter
                ql.Compounded,             # compounding
                InterestFrequency,         # frequency
                SettlementDate             # settlement date
            )
            
            # Calculate clean price from yield
            prclean = bond.cleanPrice(
                ytm,                       # yield
                DayCountBasis,             # day counter
                ql.Compounded,             # compounding
                InterestFrequency,         # frequency
                SettlementDate             # settlement date
            )
            
            # Calculate dirty price (clean price + accrued interest)
            prfull = bond.dirtyPrice(
                ytm,                       # yield
                DayCountBasis,             # day counter
                ql.Compounded,             # compounding
                InterestFrequency,         # frequency
                SettlementDate             # settlement date
            )
            
            # Calculate duration        
            mac_bond = ql.BondFunctions.duration(
                bond,                      # bond
                ytm,                       # yield
                DayCountBasis,             # day counter
                ql.Compounded,             # compounding
                InterestFrequency,         # frequency
                ql.Duration.Macaulay,      # duration type
                SettlementDate             # settlement date
            )
            
            # Calculate modified duration  
            dur_bond = ql.BondFunctions.duration(
                bond,                      # bond
                ytm,                       # yield
                DayCountBasis,             # day counter
                ql.Compounded,             # compounding
                InterestFrequency,         # frequency
                ql.Duration.Modified,      # duration type
                SettlementDate             # settlement date
            )
            
            # Calculate convexity (second-order price sensitivity to yield changes)           
            conv_bond = ql.BondFunctions.convexity(
                bond,                      # bond
                ytm,                       # yield
                DayCountBasis,             # day counter
                ql.Compounded,             # compounding
                InterestFrequency,         # frequency
                SettlementDate             # settlement date
            )
                     
            # Calculate accrued interest since last payment
            acclast = bond.accruedAmount(SettlementDate)
            
            # Calculate sum of all payments made before settlement
            accpmt = sum(cf.amount() for cf in bond.cashflows()
                         if cf.date() <= SettlementDate)
                         
            # Total accumulated value
            accall = acclast + accpmt
            
        except RuntimeError as e:            
            pass
    
    # Return all calculated bond analytics    
    return (
        x.cusip_id, x.trd_exctn_dt, x.pr, prclean, prfull,
        acclast, accpmt, accall, ytm, 
        dur_bond, mac_bond, conv_bond, x.bond_maturity         
    )

def process_chunk(chunk,n_cores):
    return pd.DataFrame(
        Parallel(n_jobs=n_cores)(delayed(GetNewVarsPy)(x) for x in tqdm(chunk.itertuples(index=False)))
    )

def get_fred_yields(start_date: str | pd.Timestamp = "2000-01-31") -> pd.DataFrame:
    """
    Download key-rate Treasury yields from FRED without pandas_datareader.

    Returns columns:
      trd_exctn_dt, oneyr, twoyr, fiveyr, sevyr, tenyr, twentyr, thirtyr
    Yields are decimals (e.g., 0.045 for 4.5%).
    """
    start = pd.to_datetime(start_date)

    series_map = {
        "DGS1":  "oneyr",
        "DGS2":  "twoyr",
        "DGS5":  "fiveyr",
        "DGS7":  "sevyr",
        "DGS10": "tenyr",
        "DGS20": "twentyr",
        "DGS30": "thirtyr",
    }

    def _load_series_csv(series_id: str, out_col: str) -> pd.DataFrame:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        s = pd.read_csv(url)

        # find the date column (DATE vs observation_date)
        date_col = "DATE" if "DATE" in s.columns else "observation_date"
        s[date_col] = pd.to_datetime(s[date_col], errors="coerce")

        # numeric series column
        s[out_col] = pd.to_numeric(s[series_id], errors="coerce")

        return s[[date_col, out_col]].rename(columns={date_col: "trd_exctn_dt"})

    out = None
    for sid, colname in series_map.items():
        s = _load_series_csv(sid, colname)
        out = s if out is None else out.merge(s, on="trd_exctn_dt", how="outer")

    # tidy
    out = (
        out.sort_values("trd_exctn_dt")
           .loc[out["trd_exctn_dt"] >= start]
           .ffill()
           .reset_index(drop=True)
    )

    # convert % -> decimal
    for c in series_map.values():
        out[c] = out[c] / 100.0

    return out

def ComputeCredit(x):
    """
    Interpolates the Treasury key-rate curve to the bond's time-to-maturity
    (in **years**).

    Parameters
    ----------
    x : pandas.Series
        Must contain:
            - tmt          : time-to-maturity in months
            - oneyr -- thirtyr : key-rate yields
            - cusip_id, trd_exctn_dt

    Returns
    -------
    tuple
        (cusip_id, trd_exctn_dt, interpolated_yield)
    """
    x_years = x.tmt
    cusip   = x.cusip_id
    date    = x.trd_exctn_dt

    # 1, 2, 5, 7, 10, 20, 30-year nodes
    if x_years < 1:
        yld_interp = x.oneyr

    elif 1 <= x_years <= 2:
        yld_interp = np.interp(x_years, [1, 2], [x.oneyr, x.twoyr])

    elif 2 < x_years <= 5:
        yld_interp = np.interp(x_years, [2, 5], [x.twoyr, x.fiveyr])

    elif 5 < x_years <= 7:
        yld_interp = np.interp(x_years, [5, 7], [x.fiveyr, x.sevyr])

    elif 7 < x_years <= 10:
        yld_interp = np.interp(x_years, [7, 10], [x.sevyr, x.tenyr])

    elif 10 < x_years <= 20:
        yld_interp = np.interp(x_years, [10, 20], [x.tenyr, x.twentyr])

    elif 20 < x_years <= 30:
        yld_interp = np.interp(x_years, [20, 30], [x.twentyr, x.thirtyr])

    else:                                   # > 30 years
        yld_interp = x.thirtyr

    return (cusip, date, yld_interp)


def calculate_credit_spreads(traced_out, ylds, n_jobs=10):
    """
    Calculate credit spreads for bond data by interpolating Treasury yields.
    
    Parameters:
    -----------
    traced_out : DataFrame
        Bond data containing cusip_id, trd_exctn_dt, ytm, bond_maturity
    ylds : DataFrame
        Treasury yield data with various tenors
    n_jobs : int, default=10
        Number of parallel jobs to run
    
    Returns:
    --------
    DataFrame with credit spreads
    """
    # Create a copy to avoid modifying original data
    traced_cs = traced_out[['cusip_id', 'trd_exctn_dt', 'ytm', 'bond_maturity']].copy()
    traced_cs.rename(columns={'bond_maturity': 'tmt'}, inplace=True)
    
    # Merge with yields
    traced_cs = traced_cs.merge(ylds, on='trd_exctn_dt', how="left")
    
    # Check for missing yield data before processing
    missing_yields = traced_cs[traced_cs.isnull().any(axis=1)]
    if len(missing_yields) > 0:
        print(f"Warning: {len(missing_yields)} rows have missing yield data")
    
    # Process in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(ComputeCredit)(x) for x in tqdm(traced_cs.itertuples(index=False))
    )
    
    # Convert results to DataFrame
    spread_df = pd.DataFrame(
        results, columns=['cusip_id', 'trd_exctn_dt', 'yld_interp']
    )
    
    # Merge back with original data to calculate credit spread
    final_df = spread_df.merge(
        traced_cs[['cusip_id', 'trd_exctn_dt', 'ytm']], 
        on=['cusip_id', 'trd_exctn_dt'],
        how='left'
    )
    
    # Calculate credit spread (bond yield - treasury yield)
    final_df['credit_spread'] = final_df['ytm'] - final_df['yld_interp']
    
    return final_df

# ---------- Rating --> numeric helpers ----------
def convert_sp_to_numeric(r):
    mapping = {'AAA':1,'AA+':2,'AA':3,'AA-':4,'A+':5,'A':6,'A-':7,
               'BBB+':8,'BBB':9,'BBB-':10,'BB+':11,'BB':12,'BB-':13,
               'B+':14,'B':15,'B-':16,'CCC+':17,'CCC':18,'CCC-':19,
               'CC':20,'C':21,'D':22}
    return mapping.get(r, np.nan)

def convert_moodys_to_numeric(r):
    mapping = {'Aaa':1,'Aa1':2,'Aa2':3,'Aa3':4,'A1':5,'A2':6,'A3':7,
               'Baa1':8,'Baa2':9,'Baa3':10,'Ba1':11,'Ba2':12,'Ba3':13,
               'B1':14,'B2':15,'B3':16,'Caa1':17,'Caa2':18,'Caa3':19,
               'Ca':20,'C':21}
    return mapping.get(r, np.nan)

def numeric_to_naic(n):
    if pd.isna(n): return np.nan
    if   1 <= n <= 7:   return 1
    elif 8 <= n <=10:  return 2
    elif 11<= n <=13:  return 3
    elif 14<= n <=16:  return 4
    elif 17<= n <=19:  return 5
    elif 20<= n <=22:  return 6
    return np.nan

def fast_join_vectorized(df_src, ffi_tbl, ind_label):
    """
    Vectorised SIC-to-industry assignment.
    """
    df = df_src.copy()
    ffi = ffi_tbl.copy()

    # Ensure numeric dtype
    df["sic_code"]     = pd.to_numeric(df["sic_code"],    errors="coerce")
    ffi["sic_low"]     = pd.to_numeric(ffi["sic_low"],    errors="coerce")
    ffi["sic_high"]    = pd.to_numeric(ffi["sic_high"],   errors="coerce")

    out = df[["cusip_id", "sic_code"]].copy()
    out[ind_label] = np.nan

    # Vectorised mask per row in ffi
    for _, seg in ffi.iterrows():
        mask = (out["sic_code"] >= seg["sic_low"]) & (out["sic_code"] <= seg["sic_high"])
        out.loc[mask, ind_label] = seg["ind_num"]

    return out.drop(columns="sic_code")

def load_parquet_from_zip_url(url: str, member_name: str) -> pd.DataFrame:
    """
    Download a ZIP from `url`, read the `member_name` parquet inside, and return as DataFrame.
    Raises FileNotFoundError if the member is not found.
    """
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        if member_name not in names:
            raise FileNotFoundError(
                f"Member '{member_name}' not found in ZIP. Available: {names}"
            )
        with zf.open(member_name) as f:
            buf = io.BytesIO(f.read())
    return pd.read_parquet(buf)


def load_parquets_from_zip_url(
    url: str,
    members: list[str] | None = None,
    *,
    save_dir: str | Path | None = None,
    timeout: int = 120
) -> dict[str, pd.DataFrame]:
    """
    Download a ZIP from `url`, load specified parquet members, and return them as DataFrames.
    If `save_dir` is provided, also save each parquet to that folder (same filenames).

    Parameters
    ----------
    url : str
        ZIP download URL.
    members : list[str] | None
        Exact parquet names to extract from the ZIP. If None, loads all *.parquet in the ZIP.
    save_dir : str | Path | None
        If provided, saves each parquet to this folder.
    timeout : int
        HTTP timeout in seconds.

    Returns
    -------
    dict[str, DataFrame]
        Mapping {member_name: DataFrame}.

    Raises
    ------
    FileNotFoundError
        If any requested member is not found inside the ZIP.
    """
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()

    with ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        parquet_names = [n for n in names if n.lower().endswith(".parquet")]

        if members is None:
            targets = parquet_names
        else:
            missing = [m for m in members if m not in names]
            if missing:
                raise FileNotFoundError(
                    f"Missing in ZIP: {missing}. Available members: {names}"
                )
            targets = members

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        out: dict[str, pd.DataFrame] = {}
        for m in targets:
            with zf.open(m) as f:
                buf = io.BytesIO(f.read())

            # Optional: save parquet file to disk for reuse
            if save_dir is not None:
                out_path = save_dir / Path(m).name
                with open(out_path, "wb") as fout:
                    fout.write(buf.getbuffer())

                # Load from buffer (already in memory) to avoid re-read
                df = pd.read_parquet(io.BytesIO(buf.getvalue()))
            else:
                df = pd.read_parquet(buf)

            out[Path(m).name] = df

    return out


def extend_and_ffill_linker(
    dfl: pd.DataFrame,
    ffill_date: pd.Timestamp,
    *,
    group_col: str = "issuer_cusip",
    date_col: str = "date",
    id_cols: tuple[str, ...] = ("gvkey", "permno", "permco"),
) -> pd.DataFrame:
    """
    Forward-extend the linker from its max date to ffill_date *only* for issuers
    that have all id_cols non-missing at the snapshot max_link_date.
    No logging inside.
    """
    if dfl.empty:
        return dfl.copy()

    out = dfl.copy()

    # Work with month-end dates
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    me_ffill = pd.to_datetime(ffill_date) + MonthEnd(0)
    me_link_max = out[date_col].max()
    if pd.isna(me_link_max):
        return out
    me_link_max = me_link_max + MonthEnd(0)

    # Nothing to do if target is not beyond current max
    if me_ffill <= me_link_max:
        return out

    # Ensure all id columns exist
    missing = [c for c in id_cols if c not in out.columns]
    if missing:
        raise KeyError(f"Missing ID columns in dataframe: {missing}")

    # Snapshot at max_link_date (month-end aligned)
    snap = out.loc[(out[date_col] + MonthEnd(0)) == me_link_max, [group_col, *id_cols]].copy()

    # Eligible issuers: all IDs present (non-missing)
    elig_mask = snap[list(id_cols)].notna().all(axis=1)  # 
    snap = snap.loc[elig_mask].drop_duplicates(subset=[group_col])

    if snap.empty:
        return out

    # Build month-end range to append (strictly after max_link_date up to ffill_date inclusive)
    new_months = pd.date_range(me_link_max + MonthEnd(1), me_ffill, freq="ME")
    if len(new_months) == 0:
        return out

    # Cartesian product: eligible issuers new months
    ext_dates = pd.DataFrame({date_col: new_months})
    ext = snap.merge(ext_dates, how="cross")

    # Populate helper columns if present in input
    if "yyyymm" in out.columns:
        ext["yyyymm"] = (ext[date_col].dt.year * 100 + ext[date_col].dt.month).astype("Int64")
    if "year_month" in out.columns:
        ext["year_month"] = ext[date_col].dt.strftime("%Y-%m")

    # Ensure all original columns exist in ext (fill with NA if not derivable)
    for c in out.columns:
        if c not in ext.columns:
            ext[c] = pd.NA

    # Reorder to match original columns
    ext = ext[out.columns]

    # Concatenate and return
    result = pd.concat([out, ext], ignore_index=True)
    result.sort_values([group_col, date_col], inplace=True, kind="mergesort")
    result.reset_index(drop=True, inplace=True)
    return result


def _check_internet_connectivity(host: str = "8.8.8.8", port: int = 53, timeout: int = 3) -> bool:
    """
    Check if internet connectivity is available.

    Args:
        host: Host to test connectivity (default: Google DNS)
        port: Port to test (default: 53/DNS)
        timeout: Timeout in seconds

    Returns:
        True if internet is reachable, False otherwise
    """
    import socket
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False


def get_liu_wu_yields(
    url: str = "https://docs.google.com/spreadsheets/d/11HsxLl_u2tBNt3FyN5iXGsIKLwxvVz7t/edit?usp=sharing",
    start_date: str | pd.Timestamp = "2000-01-31",
    local_file: str = "data/liu_wu_yields.xlsx"
) -> pd.DataFrame:
    """
    Download Liu-Wu zero-coupon Treasury yields from Google Sheets.
    If internet is not available (e.g., on WRDS compute nodes), falls back to local file.

    Returns columns matching FRED format:
      trd_exctn_dt, oneyr, twoyr, fiveyr, sevyr, tenyr, twentyr, thirtyr

    Yields are converted to decimals (e.g., 0.045 for 4.5%).

    Args:
        url: Google Sheets URL (sharing link)
        start_date: Start date for filtering data
        local_file: Path to local Excel file (used when internet unavailable)

    Returns:
        DataFrame with treasury yields at key maturities
    """

    # Check if internet is available
    has_internet = _check_internet_connectivity()

    # Try to load data - either from internet or local file
    if has_internet:
        try:
            logging.info("Internet available - downloading Liu-Wu yields from Google Sheets")
            # Convert sharing URL to export URL
            if "/edit?" in url or "/edit#" in url:
                sheet_id = url.split("/d/")[1].split("/")[0]
                export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx&id={sheet_id}"
            else:
                export_url = url

            # Download the Excel file
            response = requests.get(export_url, timeout=30)
            response.raise_for_status()

            # Load Excel from bytes (header row is at index 8, data starts at row 9)
            df = pd.read_excel(BytesIO(response.content), header=8)
            logging.info("Successfully downloaded Liu-Wu yields from Google Sheets")

        except Exception as e:
            logging.warning(f"Failed to download from internet: {e}")
            logging.info(f"Falling back to local file: {local_file}")
            has_internet = False  # Trigger local file fallback

    if not has_internet:
        # No internet - use local file
        from pathlib import Path
        local_path = Path(local_file)

        if not local_path.exists():
            raise FileNotFoundError(
                f"No internet connection and local file not found: {local_file}\n"
                f"Please download the file manually:\n"
                f"  wget -O {local_file} \"https://docs.google.com/spreadsheets/d/11HsxLl_u2tBNt3FyN5iXGsIKLwxvVz7t/export?format=xlsx&id=11HsxLl_u2tBNt3FyN5iXGsIKLwxvVz7t\"\n"
                f"Or run this from a machine with internet access."
            )

        logging.info(f"Internet not available - loading Liu-Wu yields from local file: {local_file}")
        df = pd.read_excel(local_path, header=8)
        logging.info(f"Successfully loaded Liu-Wu yields from {local_file}")

    # Strip whitespace from all column names
    df.columns = df.columns.str.strip()

    # Map Liu-Wu maturities to FRED-style column names
    # Liu-Wu uses "12 m", "24 m", etc. for months
    # We extract: 12m (1yr), 24m (2yr), 60m (5yr), 84m (7yr), 120m (10yr), 240m (20yr), 360m (30yr)
    maturity_map = {
        '12 m': 'oneyr',
        '24 m': 'twoyr',
        '60 m': 'fiveyr',
        '84 m': 'sevyr',
        '120 m': 'tenyr',
        '240 m': 'twentyr',
        '360 m': 'thirtyr'
    }

    # First column contains dates in YYYYMMDD format
    date_col = df.columns[0]

    # Create result dataframe
    result = pd.DataFrame()

    # Convert date from YYYYMMDD integer to datetime
    result['trd_exctn_dt'] = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d', errors='coerce')

    # Extract each maturity and convert from percentage to decimal
    for mat_label, out_col in maturity_map.items():
        if mat_label in df.columns:
            # Liu-Wu data is in percentage points (e.g., 3.5 for 3.5%)
            # Convert to decimal (e.g., 0.035)
            result[out_col] = pd.to_numeric(df[mat_label], errors='coerce') / 100.0
        else:
            raise ValueError(f"Expected maturity '{mat_label}' not found in Liu-Wu data")

    # Filter by start date
    start = pd.to_datetime(start_date)
    result = result[result['trd_exctn_dt'] >= start].copy()

    # Resample and ffill()
    result = result.set_index('trd_exctn_dt')
    result = result.resample("D").last()
    result = result.ffill()
    result = result.reset_index()

    # Remove rows with invalid dates
    result = result.dropna(subset=['trd_exctn_dt'])

    # Sort by date and forward fill missing values
    result = result.sort_values('trd_exctn_dt').reset_index(drop=True)

    return result


#### Distressed Filtering ####
@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def _detect_anomalies_ultra(
    prices,
    is_ultra_low,
    is_round,
    valid_mask,
    lookback,
    lookforward,
    min_normal_price_ratio,
):
    """
    Ultra-optimized anomaly detection with fastmath and nogil.
    """
    n = len(prices)
    flags = np.zeros(n, dtype=np.int8)
    anomaly_types = np.zeros(n, dtype=np.int8)
    
    for i in range(n):
        if not ((is_ultra_low[i] or is_round[i]) and valid_mask[i]):
            continue
        
        current_price = prices[i]
        start_back = max(0, i - lookback)
        end_fwd = min(n, i + lookforward + 1)
        
        # Count and collect surrounding prices in one pass
        n_surr = 0
        for j in range(start_back, i):
            if valid_mask[j] and prices[j] > current_price:
                n_surr += 1
        for j in range(i+1, end_fwd):
            if valid_mask[j] and prices[j] > current_price:
                n_surr += 1
        
        if n_surr == 0:
            continue
        
        # Pre-allocate exact size
        surrounding = np.empty(n_surr, dtype=np.float64)
        idx = 0
        for j in range(start_back, i):
            if valid_mask[j] and prices[j] > current_price:
                surrounding[idx] = prices[j]
                idx += 1
        for j in range(i+1, end_fwd):
            if valid_mask[j] and prices[j] > current_price:
                surrounding[idx] = prices[j]
                idx += 1
        
        # Compute median
        surrounding.sort()
        if n_surr % 2 == 0:
            median_surrounding = (surrounding[n_surr//2 - 1] + surrounding[n_surr//2]) * 0.5
        else:
            median_surrounding = surrounding[n_surr//2]
        
        price_ratio = median_surrounding / (current_price + 1e-10)
        
        if price_ratio >= min_normal_price_ratio:
            flags[i] = 1
            if is_ultra_low[i] and is_round[i]:
                anomaly_types[i] = 3
            elif is_ultra_low[i]:
                anomaly_types[i] = 1
            else:
                anomaly_types[i] = 2
    
    return flags, anomaly_types


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def _detect_spikes_ultra(
    prices,
    is_high,
    is_round,
    valid_mask,
    lookback,
    lookforward,
    min_spike_ratio,
    recovery_ratio,
):
    """
    Ultra-optimized spike detection with fastmath and nogil.
    """
    n = len(prices)
    flags = np.zeros(n, dtype=np.int8)
    spike_types = np.zeros(n, dtype=np.int8)
    
    for i in range(n):
        if not ((is_high[i] or is_round[i]) and valid_mask[i]):
            continue
        
        current_price = prices[i]
        start_back = max(0, i - lookback)
        
        # Count pre-prices
        n_pre = 0
        for j in range(start_back, i):
            if valid_mask[j] and prices[j] < current_price:
                n_pre += 1
        
        if n_pre == 0:
            continue
        
        # Collect pre-prices
        pre_prices = np.empty(n_pre, dtype=np.float64)
        idx = 0
        for j in range(start_back, i):
            if valid_mask[j] and prices[j] < current_price:
                pre_prices[idx] = prices[j]
                idx += 1
        
        # Compute median
        pre_prices.sort()
        if n_pre % 2 == 0:
            median_pre = (pre_prices[n_pre//2 - 1] + pre_prices[n_pre//2]) * 0.5
        else:
            median_pre = pre_prices[n_pre//2]
        
        spike_ratio = current_price / (median_pre + 1e-10)
        
        if spike_ratio < min_spike_ratio:
            continue
        
        # Check recovery
        recovery_threshold = median_pre * recovery_ratio
        end_fwd = min(n, i + lookforward + 1)
        has_recovery = False
        for j in range(i+1, end_fwd):
            if valid_mask[j] and prices[j] <= recovery_threshold:
                has_recovery = True
                break
        
        if not has_recovery:
            continue
        
        flags[i] = 1
        if is_high[i] and is_round[i]:
            spike_types[i] = 3
        elif is_high[i]:
            spike_types[i] = 1
        else:
            spike_types[i] = 2
    
    return flags, spike_types


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def _detect_plateaus_ultra(
    prices,
    is_ultra_low,
    is_round,
    valid_mask,
    min_plateau_days,
    pre_post_price_ratio,
):
    """
    Ultra-optimized plateau detection with fastmath and nogil.
    """
    n = len(prices)
    flags = np.zeros(n, dtype=np.int8)
    plateau_ids = np.full(n, -1, dtype=np.int32)
    
    plateau_count = 0
    i = 0
    
    while i < n:
        if not (valid_mask[i] and (is_ultra_low[i] or is_round[i])):
            i += 1
            continue
        
        current_price = prices[i]
        j = i + 1
        
        # Find extent of plateau with exact equality
        while j < n and prices[j] == current_price:
            j += 1
        
        plateau_length = j - i
        
        if plateau_length >= min_plateau_days:
            # Check immediate adjacent prices
            pre_price = -1.0
            if i > 0 and valid_mask[i - 1]:
                pre_price = prices[i - 1]
            
            post_price = -1.0
            if j < n and valid_mask[j]:
                post_price = prices[j]
            
            # Determine if suspicious
            is_suspicious = False
            
            if pre_price > 0:
                if pre_price / (current_price + 1e-10) >= pre_post_price_ratio:
                    is_suspicious = True
            
            if post_price > 0:
                if post_price / (current_price + 1e-10) >= pre_post_price_ratio:
                    is_suspicious = True
            
            # 
            if is_round[i]:
                is_suspicious = True
            
            if is_suspicious:
                for k in range(i, j):
                    flags[k] = 1
                    plateau_ids[k] = plateau_count
                plateau_count += 1
        
        i = j
    
    return flags, plateau_ids, plateau_count


@jit(nopython=True, cache=True, fastmath=True)
def _compute_round_mask(prices, round_numbers, round_tolerance, valid_mask):
    """
    Ultra-optimized round number detection.
    """
    n = len(prices)
    n_rounds = len(round_numbers)
    is_round = np.zeros(n, dtype=np.bool_)
    
    for i in range(n):
        if not valid_mask[i]:
            continue
        price = prices[i]
        for j in range(n_rounds):
            if abs(price - round_numbers[j]) < round_tolerance:
                is_round[i] = True
                break
    
    return is_round


def ultra_distressed_filter(
    df: pd.DataFrame,
    *,
    id_col: str = "cusip_id",
    date_col: str = "trd_exctn_dt",
    price_col: str = "pr",
    enable_anomaly_filter: bool = True,
    ultra_low_threshold: float = 0.10,
    min_normal_price_ratio: float = 3.0,
    enable_spike_filter: bool = True,
    high_spike_threshold: float = 5.0,
    min_spike_ratio: float = 3.0,
    recovery_ratio: float = 2.0,
    enable_plateau_filter: bool = True,
    plateau_ultra_low_threshold: float = 0.15,
    min_plateau_days: int = 2,
    suspicious_round_numbers: List[float] = [0.001, 0.01, 0.05, 0.10, 
                                             0.25,  0.50, 0.75, 1.00],
    round_tolerance: float = 0.0001,
    lookback: int = 5,
    lookforward: int = 5,
    pre_post_price_ratio: float = 3.0,
    enable_intraday_filter: bool = True,
    price_cols: list = ["prc_ew", "prc_vw", "prc_first", "prc_last"],
    intraday_range_threshold: float = 0.75,
    intraday_price_threshold: float = 20.0,
    n_jobs: int = -1,
    verbose: bool = False,
    keep_flag_columns: bool = False,
) -> pd.DataFrame:
    """
    Apply all refined filters with implementation. 
    Uses enhanced Numba compilation with fastmath and nogil for maximum speed. 
    """
    
    out = df.copy()
    out = out.sort_values([id_col, date_col]).reset_index(drop=True)
    
    # Pre-round all price columns at once
    for col in [price_col] + price_cols:
        if col in out.columns:
            out[col] = out[col].round(4)
    
    if verbose:
        print("="*80)
        print("APPLYING ULTRA-DISTRESSED FILTERS")
        print("="*80)
    
    n = len(out)
    
    # Initialize all output arrays
    flag_anomalous_price = np.zeros(n, dtype=np.int8)
    anomaly_type = np.full(n, '', dtype=object)
    flag_upward_spike = np.zeros(n, dtype=np.int8)
    spike_type = np.full(n, '', dtype=object)
    flag_plateau_sequence = np.zeros(n, dtype=np.int8)
    plateau_id = np.full(n, -1, dtype=np.int32)
    
    # Pre-convert to numpy array
    round_numbers = np.array(suspicious_round_numbers, dtype=np.float64)
    
    if verbose:
        print("\nProcessing filters ...")

    # Process each CUSIP - use groupby to pre-compute indices ONCE (major speedup)
    total_plateau_count = 0

    # Pre-compute all group indices using groupby (O(n) instead of O(n*k))
    grouped = out.groupby(id_col, sort=False)
    cusip_indices = {cusip: grp.index.to_numpy() for cusip, grp in grouped}
    unique_cusips = list(cusip_indices.keys())

    if verbose:
        try:
            from tqdm import tqdm
            cusip_iter = tqdm(unique_cusips, desc="  Processing CUSIPs")
        except ImportError:
            cusip_iter = unique_cusips
    else:
        cusip_iter = unique_cusips

    type_map = {0: '', 1: 'ultra_low', 2: 'round_number', 3: 'ultra_low_round'}
    spike_type_map = {0: '', 1: 'high_spike', 2: 'round_spike', 3: 'high_round_spike'}

    # Pre-extract price column as numpy array for faster indexing
    prices_all = out[price_col].to_numpy(dtype=np.float64)

    for cusip in cusip_iter:
        cusip_idx = cusip_indices[cusip]
        
        # Determine minimum observations required
        min_obs_required = []
        if enable_anomaly_filter or enable_spike_filter:
            min_obs_required.append(3)
        if enable_plateau_filter:
            min_obs_required.append(min_plateau_days)
        
        min_obs = min(min_obs_required) if min_obs_required else 3
        if len(cusip_idx) < min_obs:
            continue

        # Extract prices using pre-extracted array (faster than .loc)
        prices = prices_all[cusip_idx]
        
        # Compute common masks once
        valid_mask = ~np.isnan(prices)
        
        # Vectorized round number detection 
        is_round_all = _compute_round_mask(prices, round_numbers, round_tolerance, valid_mask)
        
        # For spike filter: only round numbers > 0.50
        is_round_spike = is_round_all & (prices > 0.50)
        
        # FILTER 1: ANOMALY DETECTION
        if enable_anomaly_filter and len(cusip_idx) >= 3:
            is_ultra_low_anomaly = prices < ultra_low_threshold
            
            flags_anomaly, anomaly_types_int = _detect_anomalies_ultra(
                prices,
                is_ultra_low_anomaly,
                is_round_all,
                valid_mask,
                lookback,
                lookforward,
                min_normal_price_ratio,
            )
            
            # Map results back
            flag_anomalous_price[cusip_idx] = flags_anomaly
            for i in range(len(cusip_idx)):
                if flags_anomaly[i] == 1:
                    anomaly_type[cusip_idx[i]] = type_map[anomaly_types_int[i]]
        
        # FILTER 2: SPIKE DETECTION
        if enable_spike_filter and len(cusip_idx) >= 3:
            is_high = prices > high_spike_threshold
            
            flags_spike, spike_types_int = _detect_spikes_ultra(
                prices,
                is_high,
                is_round_spike,
                valid_mask,
                lookback,
                lookforward,
                min_spike_ratio,
                recovery_ratio,
            )
            
            # Map results back
            flag_upward_spike[cusip_idx] = flags_spike
            for i in range(len(cusip_idx)):
                if flags_spike[i] == 1:
                    spike_type[cusip_idx[i]] = spike_type_map[spike_types_int[i]]
        
        # FILTER 3: PLATEAU DETECTION
        if enable_plateau_filter and len(cusip_idx) >= min_plateau_days:
            is_ultra_low_plateau = prices < plateau_ultra_low_threshold
            
            flags_plateau, plateau_ids_local, plateau_count = _detect_plateaus_ultra(
                prices,
                is_ultra_low_plateau,
                is_round_all,
                valid_mask,
                min_plateau_days,
                pre_post_price_ratio,
            )
            
            # Map results back with globally unique plateau IDs
            flag_plateau_sequence[cusip_idx] = flags_plateau
            for i in range(len(cusip_idx)):
                if plateau_ids_local[i] >= 0:
                    plateau_id[cusip_idx[i]] = plateau_ids_local[i] + total_plateau_count
            
            total_plateau_count += plateau_count
    
    # Assign results to dataframe
    out['flag_anomalous_price'] = flag_anomalous_price
    out['anomaly_type'] = anomaly_type
    out['flag_upward_spike'] = flag_upward_spike
    out['spike_type'] = spike_type
    out['flag_plateau_sequence'] = flag_plateau_sequence
    out['plateau_id'] = plateau_id
    
    # FILTER 4: INTRADAY INCONSISTENCY
    if enable_intraday_filter:
        if verbose:
            print("\nApplying intraday ultra-distrssed filter ...")
        flag_intraday = flag_intraday_inconsistency_vectorized(
            out,
            price_cols=price_cols,
            intraday_range_threshold=intraday_range_threshold,
            intraday_price_threshold=intraday_price_threshold,
            verbose=verbose,
        )
        out['flag_intraday_inconsistent'] = flag_intraday
    else:
        out['flag_intraday_inconsistent'] = 0
    
    # Compute combined flag
    out['flag_refined_any'] = (
        (out['flag_anomalous_price'] == 1) |
        (out['flag_upward_spike'] == 1) |
        (out['flag_plateau_sequence'] == 1) |
        (out['flag_intraday_inconsistent'] == 1)
    ).astype(np.int8)
    
    if verbose:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total observations: {len(out):,}")
        print(f"Anomalous prices (downward): {out['flag_anomalous_price'].sum():,}")
        print(f"Upward spikes: {out['flag_upward_spike'].sum():,}")
        print(f"Plateau sequences: {out['flag_plateau_sequence'].sum():,}")
        print(f"Intraday inconsistent: {out['flag_intraday_inconsistent'].sum():,}")
        print(f"ANY FLAG: {out['flag_refined_any'].sum():,} ({100*out['flag_refined_any'].sum()/len(out):.2f}%)")
    
    # Drop individual flag columns to save RAM (only keep flag_refined_any) unless requested
    if not keep_flag_columns:
        cols_to_drop = [
            'flag_anomalous_price', 'flag_upward_spike', 'flag_plateau_sequence',
            'flag_intraday_inconsistent', 'anomaly_type', 'spike_type', 'plateau_id'
        ]
        out = out.drop(columns=[c for c in cols_to_drop if c in out.columns])

    return out


def flag_intraday_inconsistency_vectorized(
    df: pd.DataFrame,
    *,
    price_cols: list = ["prc_ew", "prc_vw", "prc_first", "prc_last"],
    intraday_range_threshold: float = 0.75,
    intraday_price_threshold: float = 20.0,
    verbose: bool = False,
) -> np.ndarray:
    """
    Vectorized intraday inconsistency detection.
    """
    available_cols = [col for col in price_cols if col in df.columns]
    
    if len(available_cols) < 2:
        return np.zeros(len(df), dtype=np.int8)
    
    price_data = df[available_cols].values
    
    low_price_mask = np.any(price_data < intraday_price_threshold, axis=1)
    low_price_mask = low_price_mask & ~np.all(pd.isna(price_data), axis=1)
    
    flag_intraday = np.zeros(len(df), dtype=np.int8)
    low_indices = np.where(low_price_mask)[0]
    
    for idx in low_indices:
        prices = price_data[idx]
        prices = prices[~pd.isna(prices)]
        
        if len(prices) < 2:
            continue
        
        price_range = prices.max() - prices.min()
        price_mean = prices.mean()
        
        if price_mean > 0:
            range_pct = price_range / price_mean
            if range_pct > intraday_range_threshold:
                flag_intraday[idx] = 1
    
    if verbose:
        print(f"  Intraday inconsistency flags: {flag_intraday.sum()}")
    
    return flag_intraday

# ============================================================================
# PLOTTING AND REPORT HELPER FUNCTIONS FOR STEP 10
# ============================================================================
@dataclass
class PlotParams:
    """Plotting parameters matching GitHub _error_plot_helpers.py defaults."""
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


def create_time_series_plots(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "stage1_figures",
    params: PlotParams = None,
    rating_filter: str = None,
) -> Path:
    """
    Create 4x2 grid of time-series plots showing weekly averages.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'trd_exctn_dt' and the variables to plot
    output_dir : Path
        Directory to save the figure
    filename : str
        Base filename (extension will be added based on params.export_format)
    params : PlotParams
        Plotting parameters
    rating_filter : str, optional
        Filter bonds by rating category:
        - None or 'all': All bonds
        - 'investment_grade': spc_rating 1-10 inclusive
        - 'non_investment_grade': spc_rating >10 and <=21
        - 'defaulted': spc_rating == 22
    
    Returns
    -------
    Path to saved figure
    """
    import gc
    if params is None:
        params = PlotParams()
    
    apply_plot_params(params)
    
    # Apply rating filter if specified
    if rating_filter and rating_filter != 'all':
        if 'spc_rating' not in df.columns:
            raise ValueError("Cannot apply rating filter: 'spc_rating' column not found")
        
        if rating_filter == 'investment_grade':
            df = df[(df['spc_rating'] >= 1) & (df['spc_rating'] <= 10)].copy()
        elif rating_filter == 'non_investment_grade':
            df = df[(df['spc_rating'] > 10) & (df['spc_rating'] <= 21)].copy()
        elif rating_filter == 'defaulted':
            df = df[df['spc_rating'] == 22].copy()
        else:
            raise ValueError(f"Unknown rating_filter: {rating_filter}")
    
    # Special handling for defaulted bonds: compute market cap
    if rating_filter == 'defaulted':
        if 'bond_amt_outstanding' in df.columns and 'pr' in df.columns:
            # market_cap in billions: (bond_amt_outstanding * pr * 10) / 1e9
            df['market_cap'] = (df['bond_amt_outstanding'] * df['pr'] * 10) / 1e9
    
    # Define panels: (column_name, panel_label)
    # For defaulted bonds, use market_cap instead of sp_rating for Panel F
    if rating_filter == 'defaulted' and 'market_cap' in df.columns:
        panels = [
            ('pr', 'A: Clean Price (VW)'),
            ('prfull', 'B: Dirty Price (VW)'),
            ('ytm', 'C: Yield'),
            ('credit_spread', 'D: Spread'),
            ('mod_dur', 'E: Duration (Mod.)'),
            ('market_cap', 'F: Total Market Cap. (mln)'),  # Special for defaulted
            ('dvolume', 'G: Dollar Volume'),
            ('convexity', 'H: Convexity'),
        ]
    else:
        panels = [
            ('pr', 'A: Clean Price (VW)'),
            ('prfull', 'B: Dirty Price (VW)'),
            ('ytm', 'C: Yield'),
            ('credit_spread', 'D: Spread'),
            ('mod_dur', 'E: Duration (Mod.)'),
            ('sp_rating', 'F: Rating (SP)'),
            ('dvolume', 'G: Dollar Volume'),
            ('convexity', 'H: Convexity'),
        ]
    
    # Extract only the columns we need (avoid object columns and extra data)
    plot_cols = ['trd_exctn_dt'] + [col for col, _ in panels if col in df.columns]
    df_plot = df[plot_cols].copy()
    
    # Convert any Float64 (nullable) to float64 (standard) for consistent aggregation
    for col in df_plot.columns:
        if df_plot[col].dtype.name == 'Float64':
            df_plot[col] = df_plot[col].astype('float64')
    
    # Resample to weekly using Monday week-start
    # For market_cap (defaulted bonds), use SUM aggregation; for others use MEAN
    if rating_filter == 'defaulted' and 'market_cap' in df_plot.columns:
        # Separate market_cap for SUM aggregation
        df_weekly_mean = df_plot.drop(columns=['market_cap']).set_index('trd_exctn_dt').resample('W-MON').mean()
        df_weekly_sum = df_plot[['trd_exctn_dt', 'market_cap']].set_index('trd_exctn_dt').resample('W-MON').sum()
        df_weekly = pd.concat([df_weekly_mean, df_weekly_sum], axis=1).reset_index()
    else:
        df_weekly = df_plot.set_index('trd_exctn_dt').resample('W-MON').mean().reset_index()
    
    # A4 portrait dimensions
    fig_w, fig_h = 8.27, 11.69
    
    fig = plt.figure(figsize=(fig_w, fig_h))
    
    # Grid margins for 4x2
    gs = fig.add_gridspec(
        nrows=4, ncols=2,
        left=0.07, right=0.995, bottom=0.045, top=0.93,
        wspace=0.12, hspace=0.14
    )
    
    axes = gs.subplots().ravel()
    
    for ax, (col, label) in zip(axes, panels):
        if col not in df_weekly.columns:
            ax.text(0.5, 0.5, f"Missing: {col}", ha="center", va="center")
            ax.axis("off")
            continue
        
        # Plot time-series
        dates = df_weekly['trd_exctn_dt']
        values = df_weekly[col].copy()
        
        # Scale ytm and credit_spread by 100 for display (show as percent)
        if col in ['ytm', 'credit_spread']:
            values = values * 100
        
        ax.plot(
            dates, values,
            color=params.line_color,
            alpha=params.line_alpha,
            lw=params.line_lw,
        )
        
        ax.set_title(label, pad=2)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, alpha=params.grid_alpha, linewidth=params.grid_lw)
        
        # Format y-axis:
        # - For defaulted bonds: ytm and credit_spread use 0 decimal places
        # - For other rating filters: ytm and credit_spread use 1 decimal place
        # - For market_cap: use 1 decimal place
        from matplotlib.ticker import FormatStrFormatter
        if col in ['ytm', 'credit_spread']:
            if rating_filter == 'defaulted':
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            else:
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        elif col == 'market_cap':
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        # Format date axis
        locator = AutoDateLocator(minticks=3, maxticks=8, interval_multiples=True)
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
        bbox_inches=None,
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
    gc.collect()
    return out_path, df_weekly


# =============================================================================
# COMMENTED OUT: create_dynamics_of_default_plot
# This function causes memory issues in stage1 pipeline due to the large
# cross-join operations required to track defaulted bonds over time.
# The function is preserved here for use in stage2 where memory constraints
# are less severe.
# =============================================================================
# def create_dynamics_of_default_plot(
#     df: pd.DataFrame,
#     output_dir: Path,
#     filename: str = "stage1_dynamics_default",
#     params: PlotParams = None,
# ) -> Path:
#     """
#     Create 2x1 plot showing dynamics of defaulted bonds over time.
#
#     Panel A: Count of Defaulted Bonds (weekly)
#     Panel B: Defaulted Bonds (%) - percentage of total bonds
#
#     Parameters
#     ----------
#     df : pd.DataFrame
#         Must contain 'trd_exctn_dt' and 'spc_rating'
#     output_dir : Path
#         Directory to save the figure
#     filename : str
#         Base filename (extension will be added)
#     params : PlotParams
#         Plotting parameters
#
#     Returns
#     -------
#     Path to saved figure
#     """
#     if params is None:
#         params = PlotParams()
#
#     apply_plot_params(params)
#
#     # Check required columns
#     if 'spc_rating' not in df.columns or 'trd_exctn_dt' not in df.columns:
#         raise ValueError("DataFrame must contain 'spc_rating' and 'trd_exctn_dt' columns")
#
#     # Prepare data: need to count ALL alive bonds per day, not just those that traded
#     # MEMORY-OPTIMIZED: Only expand defaulted bonds (reduces memory by ~95%)
#     import gc
#
#     df_temp = df[['cusip_id', 'trd_exctn_dt', 'spc_rating']].copy()
#     df_temp = df_temp.sort_values(['cusip_id', 'trd_exctn_dt'])
#
#     # Step 1: Get bond lifespans for ALL bonds
#     bond_life = df_temp.groupby('cusip_id')['trd_exctn_dt'].agg(['min', 'max']).reset_index()
#     bond_life.columns = ['cusip_id', 'first_trade', 'last_trade']
#
#     # Step 2: For each bond, find first default date (first date where rating == 22)
#     defaulted = df_temp[df_temp['spc_rating'] == 22].copy()
#     first_default = defaulted.groupby('cusip_id')['trd_exctn_dt'].min().reset_index()
#     first_default.columns = ['cusip_id', 'first_default_date']
#     del defaulted
#     gc.collect()
#
#     # Step 3: For each defaulted bond, find first upgrade date (vectorized)
#     df_with_default = df_temp.merge(first_default, on='cusip_id', how='inner')
#     del df_temp
#     gc.collect()
#
#     after_default = df_with_default[df_with_default['trd_exctn_dt'] > df_with_default['first_default_date']]
#     upgrades = after_default[after_default['spc_rating'] < 22]
#     del df_with_default, after_default
#     gc.collect()
#
#     if len(upgrades) > 0:
#         first_upgrade = upgrades.groupby('cusip_id')['trd_exctn_dt'].min().reset_index()
#         first_upgrade.columns = ['cusip_id', 'first_upgrade_date']
#     else:
#         first_upgrade = pd.DataFrame(columns=['cusip_id', 'first_upgrade_date'])
#     del upgrades
#     gc.collect()
#
#     # Step 4: MEMORY OPTIMIZATION - Only keep bonds that were ever defaulted
#     defaulted_cusips = first_default['cusip_id'].unique()
#     bond_info_defaulted = bond_life[bond_life['cusip_id'].isin(defaulted_cusips)].copy()
#
#     # Merge default and upgrade dates only for defaulted bonds
#     bond_info_defaulted = bond_info_defaulted.merge(first_default, on='cusip_id', how='left')
#     if len(first_upgrade) > 0:
#         bond_info_defaulted = bond_info_defaulted.merge(first_upgrade, on='cusip_id', how='left')
#     else:
#         bond_info_defaulted['first_upgrade_date'] = pd.NaT
#     del first_default, first_upgrade
#     gc.collect()
#
#     # Step 5: Get all unique dates (use business days to reduce size)
#     all_dates = pd.date_range(
#         start=bond_life['first_trade'].min(),
#         end=bond_life['last_trade'].max(),
#         freq='B'  # Business days only
#     )
#
#     # Step 6: Cross-join ONLY for defaulted bonds (much smaller!)
#     df_dates = pd.DataFrame({'trd_exctn_dt': all_dates})
#     df_dates['_key'] = 1
#     bond_info_temp = bond_info_defaulted.copy()
#     bond_info_temp['_key'] = 1
#
#     # This cross-join is ~95% smaller (only defaulted bonds, not all bonds)
#     df_expanded_defaulted = df_dates.merge(bond_info_temp, on='_key').drop('_key', axis=1)
#     del bond_info_temp, bond_info_defaulted
#     gc.collect()
#
#     # Filter to alive defaulted bonds
#     alive_mask = (
#         (df_expanded_defaulted['first_trade'] <= df_expanded_defaulted['trd_exctn_dt']) &
#         (df_expanded_defaulted['last_trade'] >= df_expanded_defaulted['trd_exctn_dt'])
#     )
#     df_alive_defaulted = df_expanded_defaulted[alive_mask].copy()
#     del df_expanded_defaulted, alive_mask
#     gc.collect()
#
#     # Apply defaulted mask (currently defaulted, not yet upgraded)
#     defaulted_mask = (
#         df_alive_defaulted['first_default_date'].notna() &
#         (df_alive_defaulted['first_default_date'] <= df_alive_defaulted['trd_exctn_dt']) &
#         (df_alive_defaulted['first_upgrade_date'].isna() |
#          (df_alive_defaulted['trd_exctn_dt'] < df_alive_defaulted['first_upgrade_date']))
#     )
#     df_alive_defaulted = df_alive_defaulted[defaulted_mask].copy()
#     del defaulted_mask
#     gc.collect()
#
#     # Count defaulted bonds per day
#     df_defaulted_daily = df_alive_defaulted.groupby('trd_exctn_dt').size().reset_index(name='defaulted_bonds')
#     del df_alive_defaulted
#     gc.collect()
#
#     # Step 7: Count total alive bonds per day (all bonds)
#     # MEMORY OPTIMIZATION: Process in chunks to avoid huge cross-join
#     # Instead of expanding all bonds × all dates at once, process dates in chunks
#
#     chunk_size_days = 365  # Process 1 year of dates at a time
#     all_dates_list = list(all_dates)
#     total_results = []
#
#     bond_life_np = bond_life[['cusip_id', 'first_trade', 'last_trade']].copy()
#
#     for i in range(0, len(all_dates_list), chunk_size_days):
#         chunk_dates = all_dates_list[i:i + chunk_size_days]
#
#         # For each date in chunk, count bonds where first_trade <= date <= last_trade
#         chunk_results = []
#         for date in chunk_dates:
#             alive_count = (
#                 (bond_life_np['first_trade'] <= date) &
#                 (bond_life_np['last_trade'] >= date)
#             ).sum()
#             chunk_results.append({'trd_exctn_dt': date, 'total_bonds': alive_count})
#
#         total_results.extend(chunk_results)
#         gc.collect()
#
#     df_total_daily = pd.DataFrame(total_results)
#     del bond_life_np, total_results, all_dates_list, bond_life
#     gc.collect()
#
#     # Merge total and defaulted counts
#     df_daily = df_total_daily.merge(df_defaulted_daily, on='trd_exctn_dt', how='left')
#     df_daily['defaulted_bonds'] = df_daily['defaulted_bonds'].fillna(0).astype(int)
#     del df_total_daily, df_defaulted_daily
#     gc.collect()
#
#     # Resample to weekly (sum for counts)
#     df_weekly = df_daily.set_index('trd_exctn_dt').resample('W-MON').sum().reset_index()
#
#     # Compute percentage
#     df_weekly['defaulted_pct'] = (df_weekly['defaulted_bonds'] / df_weekly['total_bonds']) * 100
#
#     # A4 portrait dimensions - adjust for 2x1 layout
#     fig_w, fig_h = 8.27, 5.85  # Half height of standard 4x2 layout
#
#     fig = plt.figure(figsize=(fig_w, fig_h))
#
#     # Grid margins for 2x1
#     gs = fig.add_gridspec(
#         nrows=2, ncols=1,
#         left=0.10, right=0.95, bottom=0.10, top=0.93,
#         hspace=0.18
#     )
#
#     axes = gs.subplots().ravel()
#
#     # Panel A: Count of Defaulted Bonds
#     ax = axes[0]
#     dates = df_weekly['trd_exctn_dt']
#     values = df_weekly['defaulted_bonds']
#
#     ax.plot(
#         dates, values,
#         color=params.line_color,
#         alpha=params.line_alpha,
#         lw=params.line_lw,
#     )
#
#     ax.set_title('A: Count of Defaulted Bonds', pad=2)
#     ax.set_xlabel("")
#     ax.set_ylabel("")
#     ax.grid(True, alpha=params.grid_alpha, linewidth=params.grid_lw)
#
#     # Format date axis
#     locator = AutoDateLocator(minticks=3, maxticks=8, interval_multiples=True)
#     ax.xaxis.set_major_locator(locator)
#     ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
#     ax.tick_params(axis="x", labelsize=params.tick_size, pad=1)
#     ax.tick_params(axis="y", labelsize=params.tick_size)
#     ax.margins(x=0.01)
#
#     # Panel B: Defaulted Bonds (%)
#     ax = axes[1]
#     values = df_weekly['defaulted_pct']
#
#     ax.plot(
#         dates, values,
#         color=params.line_color,
#         alpha=params.line_alpha,
#         lw=params.line_lw,
#     )
#
#     ax.set_title('B: Defaulted Bonds (%)', pad=2)
#     ax.set_xlabel("")
#     ax.set_ylabel("")
#     ax.grid(True, alpha=params.grid_alpha, linewidth=params.grid_lw)
#
#     # Format y-axis with 1 decimal place
#     from matplotlib.ticker import FormatStrFormatter
#     ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#
#     # Format date axis
#     ax.xaxis.set_major_locator(locator)
#     ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
#     ax.tick_params(axis="x", labelsize=params.tick_size, pad=1)
#     ax.tick_params(axis="y", labelsize=params.tick_size)
#     ax.margins(x=0.01)
#
#     # Save figure
#     ext = params.export_format.lower()
#     if ext not in ["pdf", "png", "jpg", "jpeg"]:
#         ext = "pdf"
#
#     out_path = output_dir / f"{filename}.{ext}"
#
#     savefig_kwargs = dict(
#         format=ext,
#         bbox_inches=None,
#         facecolor="white",
#         edgecolor="none",
#         transparent=params.transparent,
#     )
#
#     if ext in ("png", "jpg", "jpeg"):
#         savefig_kwargs["dpi"] = params.figure_dpi
#
#     if ext == "png":
#         savefig_kwargs["pil_kwargs"] = {"optimize": True, "compress_level": 6}
#     elif ext in ("jpg", "jpeg"):
#         savefig_kwargs["pil_kwargs"] = {"quality": 85, "optimize": True, "progressive": True}
#
#     fig.savefig(out_path, **savefig_kwargs)
#     plt.close(fig)
#
#     # Prepare export dataframe with clean column names
#     df_export = df_weekly[['trd_exctn_dt', 'defaulted_bonds', 'defaulted_pct']].copy()
#     df_export.columns = ['date', 'count_defaulted', 'pct_defaulted']
#
#     # Clean up memory before return
#     del df_weekly, df_daily
#     gc.collect()
#
#     return out_path, df_export
# =============================================================================


def compute_business_days_per_month(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Compute number of U.S. business days per month.
    
    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: year_month (YYYY-MM), business_days (int)
    """
    from pandas.tseries.offsets import BDay
    
    # Generate all business days in range
    bdays = pd.bdate_range(start=start_date, end=end_date, freq='B')
    
    # Convert to DataFrame and extract year-month
    df_bdays = pd.DataFrame({'date': bdays})
    df_bdays['year_month'] = df_bdays['date'].dt.to_period('M').astype(str)
    
    # Count business days per month
    bdays_per_month = df_bdays.groupby('year_month').size().reset_index(name='business_days')
    
    return bdays_per_month


def compute_trade_counts_by_month(df: pd.DataFrame, rating_filter: str = None) -> pd.DataFrame:
    """
    Compute average trade counts per bond per month, accounting for zero-trade months.

    For each bond and month within its first-last trade window:
    - Count days with valid price (pr not null)
    - Count days with valid bid price (prc_bid not null)
    - Count days with valid ask price (prc_ask not null)

    Then average these counts (including zeros) across bonds for each month.
    MEMORY OPTIMIZED: Select columns first, filter, then copy (reduces memory by ~90%).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: trd_exctn_dt, cusip_id, pr, prc_bid, prc_ask, spc_rating
    rating_filter : str, optional
        'investment_grade', 'non_investment_grade', or 'defaulted'

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: year_month, avg_pr_days, avg_bid_days, avg_ask_days
    """
    import gc

    # MEMORY OPTIMIZATION: Select only needed columns FIRST (reduces by ~85%)
    cols_needed = ['trd_exctn_dt', 'cusip_id', 'pr', 'prc_bid', 'prc_ask', 'spc_rating']
    df = df[cols_needed]

    # Apply rating filter BEFORE copying (reduces by 40-95% depending on rating)
    if rating_filter == 'investment_grade':
        df = df[(df['spc_rating'] >= 1) & (df['spc_rating'] <= 10)]
    elif rating_filter == 'non_investment_grade':
        df = df[(df['spc_rating'] > 10) & (df['spc_rating'] <= 21)]
    elif rating_filter == 'defaulted':
        df = df[df['spc_rating'] == 22]

    # NOW copy (much smaller - only filtered rows and needed columns)
    df = df.copy()

    # Extract year-month
    df['year_month'] = df['trd_exctn_dt'].dt.to_period('M')

    # Step 1: Get bond lifespans (one row per bond - lightweight!)
    bond_life = df.groupby('cusip_id')['year_month'].agg(['min', 'max']).reset_index()
    bond_life.columns = ['cusip_id', 'first_month', 'last_month']

    # Step 2: Get actual trade counts (only where trades exist)
    actual_counts = df.groupby(['cusip_id', 'year_month']).agg({
        'pr': 'count',
        'prc_bid': 'count',
        'prc_ask': 'count',
    }).reset_index()
    actual_counts.columns = ['cusip_id', 'year_month', 'pr_days', 'bid_days', 'ask_days']

    # Free memory
    del df
    gc.collect()

    # Step 3: Get all unique months
    all_months = sorted(actual_counts['year_month'].unique())

    # Step 4: Loop over months and compute averages including zeros
    results = []
    for month in all_months:
        # Vectorized: which bonds are alive this month?
        alive_mask = (bond_life['first_month'] <= month) & (bond_life['last_month'] >= month)
        num_alive = alive_mask.sum()

        if num_alive == 0:
            continue

        # Get trades for this month
        month_trades = actual_counts[actual_counts['year_month'] == month]

        # Sum actual trade days
        total_pr = month_trades['pr_days'].sum()
        total_bid = month_trades['bid_days'].sum()
        total_ask = month_trades['ask_days'].sum()

        # Average including zeros (divide by ALL alive bonds, not just traded)
        avg_pr = total_pr / num_alive
        avg_bid = total_bid / num_alive
        avg_ask = total_ask / num_alive

        results.append({
            'year_month': str(month),
            'avg_pr_days': avg_pr,
            'avg_bid_days': avg_bid,
            'avg_ask_days': avg_ask,
        })
    gc.collect()
    return pd.DataFrame(results)


def create_trade_sparsity_count_plot(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "stage1_trade_sparsity_count",
    params: PlotParams = None,
) -> Path:
    """
    Create Figure 6: Average Number of Trade Days per Month.
    
    3x1 subplot showing:
    - Panel A: Investment Grade
    - Panel B: Non-Investment Grade
    - Panel C: Defaulted
    
    Each panel shows:
    - Dashed line: Business days in month
    - Black solid: Avg days with valid pr
    - Red solid: Avg days with valid prc_bid
    - Blue solid: Avg days with valid prc_ask
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain: trd_exctn_dt, cusip_id, pr, prc_bid, prc_ask, spc_rating
    output_dir : Path
        Directory to save the figure
    filename : str
        Base filename (extension will be added)
    params : PlotParams
        Plotting parameters
    
    Returns
    -------
    Path to saved figure
    
    Caption
    -------
    This figure shows the average number of days per month that bonds have valid 
    price observations, by rating category. For each month, we compute the average 
    number of days per bond with a valid price (black line), bid price (red line), 
    and ask price (blue line). The dashed horizontal line shows the number of 
    business days in each month. Lower values indicate less frequent trading or 
    price updates. The y-axis represents the count of days per month (0-30), 
    averaged across all bonds in each rating category. A bond is considered "alive" 
    from its first observed trade date to its last observed trade date in TRACE, 
    for the purposes of computing average trades within a month.
    """
    if params is None:
        params = PlotParams()
    
    apply_plot_params(params)
    
    # Get date range from data
    min_date = df['trd_exctn_dt'].min().strftime('%Y-%m-%d')
    max_date = df['trd_exctn_dt'].max().strftime('%Y-%m-%d')
    
    # Compute business days per month
    bdays_df = compute_business_days_per_month(min_date, max_date)
    
    # Compute trade counts for each rating category
    # MEMORY OPTIMIZED: Add garbage collection after each category
    rating_categories = [
        ('investment_grade', 'A: Investment Grade'),
        ('non_investment_grade', 'B: Non-Investment Grade'),
        ('defaulted', 'C: Defaulted'),
    ]

    import gc
    trade_data = {}
    for rating_filter, _ in rating_categories:
        trade_data[rating_filter] = compute_trade_counts_by_month(df, rating_filter)
        # Force garbage collection after each rating category (frees ~1-2GB)
        gc.collect()
    
    # A4 portrait dimensions - adjust for 3x1 layout
    fig_w, fig_h = 8.27, 8.77  # 3/4 height of standard 4x2 layout
    
    fig = plt.figure(figsize=(fig_w, fig_h))
    
    # Grid margins for 3x1
    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        left=0.10, right=0.95, bottom=0.07, top=0.93,
        hspace=0.18
    )
    
    axes = gs.subplots().ravel()
    
    for ax, (rating_filter, panel_label) in zip(axes, rating_categories):
        # Get data for this rating category
        df_trade = trade_data[rating_filter]
        
        # Merge with business days
        df_plot = df_trade.merge(bdays_df, on='year_month', how='left')
        
        # Convert year_month to datetime for plotting
        df_plot['date'] = pd.to_datetime(df_plot['year_month'] + '-01')
        
        # Plot business days (dashed line, gray)
        ax.plot(
            df_plot['date'], df_plot['business_days'],
            color='0.4', linestyle='--', linewidth=params.line_lw, alpha=0.7,
            label='Business Days'
        )
        
        # Plot trade counts
        ax.plot(
            df_plot['date'], df_plot['avg_pr_days'],
            color=params.line_color, linestyle='-', linewidth=params.line_lw,
            alpha=params.line_alpha, label='Price'
        )
        
        ax.plot(
            df_plot['date'], df_plot['avg_bid_days'],
            color='red', linestyle='-', linewidth=params.line_lw,
            alpha=params.line_alpha, label='Bid'
        )
        
        ax.plot(
            df_plot['date'], df_plot['avg_ask_days'],
            color='blue', linestyle='-', linewidth=params.line_lw,
            alpha=params.line_alpha, label='Ask'
        )
        
        ax.set_title(panel_label, pad=2)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, alpha=params.grid_alpha, linewidth=params.grid_lw)
        
        # Add legend (only in first panel)
        if ax == axes[0]:
            ax.legend(loc='upper left', fontsize=params.legend_size, frameon=False)
        
        # Format y-axis with 0 decimal places
        from matplotlib.ticker import FormatStrFormatter
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        
        # Format date axis
        from matplotlib.dates import AutoDateLocator, DateFormatter
        locator = AutoDateLocator(minticks=3, maxticks=8, interval_multiples=True)
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
        bbox_inches=None,
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

    # Clean up memory
    gc.collect()

    return out_path


def create_trade_sparsity_probability_plot(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "stage1_trade_sparsity_probability",
    params: PlotParams = None,
) -> Path:
    """
    Create Figure 7: Historical Probability of Trade.
    
    3x1 subplot showing:
    - Panel A: Investment Grade
    - Panel B: Non-Investment Grade
    - Panel C: Defaulted
    
    Each panel shows probability (0-100%) of trade on any given day:
    - Black line: (Avg days with pr) / (Business days) 
    - Red line: (Avg days with prc_bid) / (Business days) 
    - Blue line: (Avg days with prc_ask) / (Business days) 
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain: trd_exctn_dt, cusip_id, pr, prc_bid, prc_ask, spc_rating
    output_dir : Path
        Directory to save the figure
    filename : str
        Base filename (extension will be added)
    params : PlotParams
        Plotting parameters
    
    Returns
    -------
    Path to saved figure
    
    Caption
    -------
    This figure shows the historical probability of observing a valid price quote 
    on any given business day, by rating category. The probability is computed as 
    the ratio of average days traded per month (from Figure 6) to the number of 
    business days in that month, expressed as a percentage. For example, a value 
    of 50 percent means that on average, bonds in that category have a valid price 
    on half of all business days in the month. The black line shows prices, the 
    red line shows bid prices, and the blue line shows ask prices. Higher values 
    indicate more liquid bonds with more frequent price observations. A bond is 
    considered "alive" from its first observed trade date to its last observed 
    trade date in TRACE, for the purposes of computing average trades within a month.
    """
    if params is None:
        params = PlotParams()
    
    apply_plot_params(params)
    
    # Get date range from data
    min_date = df['trd_exctn_dt'].min().strftime('%Y-%m-%d')
    max_date = df['trd_exctn_dt'].max().strftime('%Y-%m-%d')
    
    # Compute business days per month
    bdays_df = compute_business_days_per_month(min_date, max_date)
    
    # Compute trade counts for each rating category
    # MEMORY OPTIMIZED: Add garbage collection after each category
    rating_categories = [
        ('investment_grade', 'A: Investment Grade'),
        ('non_investment_grade', 'B: Non-Investment Grade'),
        ('defaulted', 'C: Defaulted'),
    ]

    import gc
    trade_data = {}
    for rating_filter, _ in rating_categories:
        trade_data[rating_filter] = compute_trade_counts_by_month(df, rating_filter)
        # Force garbage collection after each rating category (frees ~1-2GB)
        gc.collect()
    
    # A4 portrait dimensions - adjust for 3x1 layout
    fig_w, fig_h = 8.27, 8.77  # 3/4 height of standard 4x2 layout
    
    fig = plt.figure(figsize=(fig_w, fig_h))
    
    # Grid margins for 3x1
    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        left=0.10, right=0.95, bottom=0.07, top=0.93,
        hspace=0.18
    )
    
    axes = gs.subplots().ravel()
    
    for ax, (rating_filter, panel_label) in zip(axes, rating_categories):
        # Get data for this rating category
        df_trade = trade_data[rating_filter]
        
        # Merge with business days
        df_plot = df_trade.merge(bdays_df, on='year_month', how='left')
        
        # Convert year_month to datetime for plotting
        df_plot['date'] = pd.to_datetime(df_plot['year_month'] + '-01')
        
        # Compute probabilities (in percentage)
        df_plot['prob_pr'] = (df_plot['avg_pr_days'] / df_plot['business_days']) * 100
        df_plot['prob_bid'] = (df_plot['avg_bid_days'] / df_plot['business_days']) * 100
        df_plot['prob_ask'] = (df_plot['avg_ask_days'] / df_plot['business_days']) * 100
        
        # Plot probabilities
        ax.plot(
            df_plot['date'], df_plot['prob_pr'],
            color=params.line_color, linestyle='-', linewidth=params.line_lw,
            alpha=params.line_alpha, label='Price'
        )
        
        ax.plot(
            df_plot['date'], df_plot['prob_bid'],
            color='red', linestyle='-', linewidth=params.line_lw,
            alpha=params.line_alpha, label='Bid'
        )
        
        ax.plot(
            df_plot['date'], df_plot['prob_ask'],
            color='blue', linestyle='-', linewidth=params.line_lw,
            alpha=params.line_alpha, label='Ask'
        )
        
        ax.set_title(panel_label, pad=2)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, alpha=params.grid_alpha, linewidth=params.grid_lw)
        
        # Set y-axis to 0-100% range
        ax.set_ylim(0, 100)
        
        # Format y-axis with 0 decimal places
        from matplotlib.ticker import FormatStrFormatter
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        
        # Add legend (only in first panel)
        if ax == axes[0]:
            ax.legend(loc='upper left', fontsize=params.legend_size, frameon=False)
        
        # Format date axis
        from matplotlib.dates import AutoDateLocator, DateFormatter
        locator = AutoDateLocator(minticks=3, maxticks=8, interval_multiples=True)
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
        bbox_inches=None,
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

    # Clean up memory
    gc.collect()

    return out_path


def compute_concentration_stats(df: pd.DataFrame, rating_filter: str = None) -> dict:
    """
    Compute trading concentration metrics for Table 8.

    MEMORY OPTIMIZED: Select columns first, filter, then copy (reduces memory by ~90%).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: cusip_id, dvolume, spc_rating
    rating_filter : str, optional
        'investment_grade', 'non_investment_grade', or 'defaulted'

    Returns
    -------
    dict
        Concentration statistics
    """
    import gc

    # MEMORY OPTIMIZATION: Select only needed columns FIRST (reduces by ~85%)
    cols_needed = ['cusip_id', 'dvolume', 'spc_rating']
    df = df[cols_needed]

    # Apply rating filter BEFORE copying (reduces by 40-95%)
    if rating_filter == 'investment_grade':
        df = df[(df['spc_rating'] >= 1) & (df['spc_rating'] <= 10)]
    elif rating_filter == 'non_investment_grade':
        df = df[(df['spc_rating'] > 10) & (df['spc_rating'] <= 21)]
    elif rating_filter == 'defaulted':
        df = df[df['spc_rating'] == 22]

    # NOW copy (much smaller - only filtered rows and needed columns)
    df = df.copy()
    
    # Total volume per bond (FAST: groupby sum)
    bond_volume = df.groupby('cusip_id')['dvolume'].sum().sort_values(ascending=False)
    total_volume = bond_volume.sum()
    
    # Cumulative volume share (FAST: cumsum)
    bond_volume_pct = (bond_volume / total_volume * 100).cumsum()
    
    # Find % of bonds for 50%, 75%, 90% of volume
    n_bonds = len(bond_volume)
    pct_for_50 = (bond_volume_pct <= 50).sum() / n_bonds * 100
    pct_for_75 = (bond_volume_pct <= 75).sum() / n_bonds * 100
    pct_for_90 = (bond_volume_pct <= 90).sum() / n_bonds * 100
    
    # Top 10% and 25% share
    n_top10 = max(1, int(n_bonds * 0.10))
    n_top25 = max(1, int(n_bonds * 0.25))
    top10_share = bond_volume.iloc[:n_top10].sum() / total_volume * 100
    top25_share = bond_volume.iloc[:n_top25].sum() / total_volume * 100
    
    # Herfindahl index (FAST: vectorized)
    market_shares = bond_volume / total_volume
    herfindahl = (market_shares ** 2).sum() * 10000  # Scale to 0-10000

    # Free memory
    gc.collect()

    return {
        'top10_share': top10_share,
        'top25_share': top25_share,
        'pct_for_50': pct_for_50,
        'pct_for_90': pct_for_90,
        'herfindahl': herfindahl,
    }


def create_trade_frequency_histogram(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "stage1_trade_frequency_hist",
    params: PlotParams = None,
) -> Path:
    """
    Create Figure 8: Cross-sectional distribution of monthly trading days.

    3x1 histogram showing distribution of days traded per month per bond.

    MEMORY OPTIMIZED: Select columns first, filter, then copy (reduces memory by ~90%).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: trd_exctn_dt, cusip_id, pr, spc_rating
    output_dir : Path
        Directory to save the figure
    filename : str
        Base filename
    params : PlotParams
        Plotting parameters

    Returns
    -------
    Path to saved figure

    Caption
    -------
    This figure shows the cross-sectional distribution of the number of days
    per month that each bond trades (has a valid price observation). Each panel
    represents a different rating category. The y-axis shows the percentage of
    bond-month observations falling into each bin. The red dashed line indicates
    the median number of trading days per month. A bond is considered "alive"
    from its first observed trade date to its last observed trade date in TRACE,
    for the purposes of computing average trades within a month.
    """
    import gc

    if params is None:
        params = PlotParams()

    apply_plot_params(params)

    # Define rating categories
    rating_categories = [
        ('investment_grade', 'A: Investment Grade'),
        ('non_investment_grade', 'B: Non-Investment Grade'),
        ('defaulted', 'C: Defaulted'),
    ]

    # Compute trading days per bond per month for each category (including zeros)
    hist_data = {}
    for rating_filter, _ in rating_categories:
        # MEMORY OPTIMIZATION: Select only needed columns FIRST (reduces by ~85%)
        cols_needed = ['trd_exctn_dt', 'cusip_id', 'pr', 'spc_rating']
        df_filt = df[cols_needed]

        # Apply rating filter BEFORE copying (reduces by 40-95%)
        if rating_filter == 'investment_grade':
            df_filt = df_filt[(df_filt['spc_rating'] >= 1) & (df_filt['spc_rating'] <= 10)]
        elif rating_filter == 'non_investment_grade':
            df_filt = df_filt[(df_filt['spc_rating'] > 10) & (df_filt['spc_rating'] <= 21)]
        elif rating_filter == 'defaulted':
            df_filt = df_filt[df_filt['spc_rating'] == 22]

        # NOW copy (much smaller - only filtered rows and needed columns)
        df_filt = df_filt.copy()
        
        # Extract year-month
        df_filt['year_month'] = df_filt['trd_exctn_dt'].dt.to_period('M')
        
        # Step 1: Get bond lifespans (lightweight)
        bond_life = df_filt.groupby('cusip_id')['year_month'].agg(['min', 'max']).reset_index()
        bond_life.columns = ['cusip_id', 'first_month', 'last_month']
        
        # Step 2: Get actual trade counts
        actual_counts = df_filt.groupby(['cusip_id', 'year_month'])['pr'].count().reset_index()
        actual_counts.columns = ['cusip_id', 'year_month', 'days']
        
        # Step 3: Calculate total bond-months (sum of months alive for each bond)
        # Vectorized: use ordinal values to compute month difference
        bond_life['n_months'] = bond_life['last_month'].view('int64') - bond_life['first_month'].view('int64') + 1
        total_bond_months = bond_life['n_months'].sum()
        
        # Step 4: Get distribution
        # - Actual trades contribute their counts
        # - Zero trades = total_bond_months - len(actual_counts)
        trade_counts = actual_counts['days'].values
        n_zeros = total_bond_months - len(trade_counts)
        
        # Combine: zeros + actual trade counts
        all_counts = np.concatenate([np.zeros(n_zeros, dtype=int), trade_counts])
        hist_data[rating_filter] = all_counts

        # Free memory after processing each category
        gc.collect()

    # Create figure (3x1 layout)
    fig_w, fig_h = 8.27, 8.77
    fig = plt.figure(figsize=(fig_w, fig_h))
    
    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        left=0.10, right=0.95, bottom=0.07, top=0.93,
        hspace=0.18
    )
    
    axes = gs.subplots().ravel()
    
    for idx, (ax, (rating_filter, panel_label)) in enumerate(zip(axes, rating_categories)):
        data = hist_data[rating_filter]
        
        # Create histogram with density=True to get percentages
        ax.hist(data, bins=range(0, 24), density=True, weights=np.ones(len(data)) * 100,
                color=params.line_color, alpha=0.7, edgecolor='white', linewidth=0.5)
        
        # Add median line
        median_val = np.median(data)
        ax.axvline(median_val, color='red', linestyle='--', 
                   linewidth=params.line_lw, label=f'Median: {median_val:.0f}')
        
        ax.set_title(panel_label, pad=2)
        
        # Only show x-axis label on final subplot (Panel C)
        if idx == 2:  # Last panel
            ax.set_xlabel("Days Traded per Month")
        else:
            ax.set_xlabel("")
        
        ax.set_ylabel("% of Bond-Month Obs.")
        ax.grid(True, alpha=params.grid_alpha, linewidth=params.grid_lw, axis='y')
        
        ax.legend(loc='upper right', fontsize=params.legend_size, frameon=False)
        ax.tick_params(axis="x", labelsize=params.tick_size)
        ax.tick_params(axis="y", labelsize=params.tick_size)
    
    # Save figure
    ext = params.export_format.lower()
    if ext not in ["pdf", "png", "jpg", "jpeg"]:
        ext = "pdf"
    
    out_path = output_dir / f"{filename}.{ext}"
    
    savefig_kwargs = dict(
        format=ext,
        bbox_inches=None,
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

    # Clean up memory
    gc.collect()

    return out_path


def create_zero_trade_bonds_plot(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "stage1_zero_trade_bonds",
    params: PlotParams = None,
) -> Path:
    """
    Create Figure 9: Zero-trade bonds over time.
    
    3x1 time series showing % of bonds with no valid price/bid/ask in each month.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain: trd_exctn_dt, cusip_id, pr, prc_bid, prc_ask, spc_rating
    output_dir : Path
        Directory to save the figure
    filename : str
        Base filename
    params : PlotParams
        Plotting parameters
    
    Returns
    -------
    Path to saved figure
    
    Caption
    -------
    This figure shows the percentage of bonds with zero trading activity in each month,
    by rating category. Trading activity is measured separately for price (blue line), 
    bid (red line), and ask (green line) observations. A bond is counted as having zero 
    trades in a given month if it has no valid observations of that type during that month. 
    The y-axis ranges from 0-100%, showing the proportion of all bonds in each rating 
    category that had no trading activity. Higher values indicate more illiquid bonds or 
    periods with less market activity.
    """
    if params is None:
        params = PlotParams()
    
    apply_plot_params(params)
    
    # Define rating categories
    rating_categories = [
        ('investment_grade', 'A: Investment Grade'),
        ('non_investment_grade', 'B: Non-Investment Grade'),
        ('defaulted', 'C: Defaulted'),
    ]
    
    # Compute zero-trade % for each category
    zero_data = {}
    for rating_filter, _ in rating_categories:
        df_filt = df.copy()
        
        # Apply rating filter
        if rating_filter == 'investment_grade':
            df_filt = df_filt[(df_filt['spc_rating'] >= 1) & (df_filt['spc_rating'] <= 10)]
        elif rating_filter == 'non_investment_grade':
            df_filt = df_filt[(df_filt['spc_rating'] > 10) & (df_filt['spc_rating'] <= 21)]
        elif rating_filter == 'defaulted':
            df_filt = df_filt[df_filt['spc_rating'] == 22]
        
        df_filt['year_month'] = df_filt['trd_exctn_dt'].dt.to_period('M')
        
        # Step 1: Get bond lifespans (lightweight)
        bond_life = df_filt.groupby('cusip_id')['year_month'].agg(['min', 'max']).reset_index()
        bond_life.columns = ['cusip_id', 'first_month', 'last_month']
        
        # Step 2: Count bonds with at least one valid observation per month
        active_pr = df_filt.groupby(['cusip_id', 'year_month'])['pr'].count().reset_index()
        active_pr.columns = ['cusip_id', 'year_month', 'pr_count']
        
        active_bid = df_filt.groupby(['cusip_id', 'year_month'])['prc_bid'].count().reset_index()
        active_bid.columns = ['cusip_id', 'year_month', 'bid_count']
        
        active_ask = df_filt.groupby(['cusip_id', 'year_month'])['prc_ask'].count().reset_index()
        active_ask.columns = ['cusip_id', 'year_month', 'ask_count']
        
        # Step 3: Get all unique months
        all_months = sorted(df_filt['year_month'].unique())
        
        # Step 4: Loop over months and compute zero-trade percentages
        results = []
        for month in all_months:
            # Vectorized: which bonds are alive this month?
            alive_mask = (bond_life['first_month'] <= month) & (bond_life['last_month'] >= month)
            num_alive = alive_mask.sum()
            
            if num_alive == 0:
                continue
            
            # Count bonds that traded this month (for each type)
            traded_pr = len(active_pr[active_pr['year_month'] == month])
            traded_bid = len(active_bid[active_bid['year_month'] == month])
            traded_ask = len(active_ask[active_ask['year_month'] == month])
            
            # Zero-trade % = (alive - traded) / alive * 100
            zero_pr_pct = ((num_alive - traded_pr) / num_alive) * 100
            zero_bid_pct = ((num_alive - traded_bid) / num_alive) * 100
            zero_ask_pct = ((num_alive - traded_ask) / num_alive) * 100
            
            results.append({
                'year_month': str(month),
                'pr': zero_pr_pct,
                'prc_bid': zero_bid_pct,
                'prc_ask': zero_ask_pct,
            })
        
        monthly_zeros = pd.DataFrame(results)
        monthly_zeros['date'] = pd.to_datetime(monthly_zeros['year_month'] + '-01')
        zero_data[rating_filter] = monthly_zeros
    
    # Create figure
    fig_w, fig_h = 8.27, 8.77
    fig = plt.figure(figsize=(fig_w, fig_h))
    
    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        left=0.10, right=0.95, bottom=0.07, top=0.93,
        hspace=0.18
    )
    
    axes = gs.subplots().ravel()
    
    for ax, (rating_filter, panel_label) in zip(axes, rating_categories):
        df_plot = zero_data[rating_filter]
        
        # Plot lines
        ax.plot(df_plot['date'], df_plot['pr'],
                color=params.line_color, linestyle='-', linewidth=params.line_lw,
                alpha=params.line_alpha, label='Price')
        
        ax.plot(df_plot['date'], df_plot['prc_bid'],
                color='red', linestyle='-', linewidth=params.line_lw,
                alpha=params.line_alpha, label='Bid')
        
        ax.plot(df_plot['date'], df_plot['prc_ask'],
                color='blue', linestyle='-', linewidth=params.line_lw,
                alpha=params.line_alpha, label='Ask')
        
        ax.set_title(panel_label, pad=2)
        ax.set_xlabel("")
        ax.set_ylabel("% of Bonds")
        ax.grid(True, alpha=params.grid_alpha, linewidth=params.grid_lw)
        
        # Set y-axis to 0-100%
        ax.set_ylim(0, 100)
        
        # Format y-axis with 0 decimal places
        from matplotlib.ticker import FormatStrFormatter
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        
        # Legend only in first panel
        if ax == axes[0]:
            ax.legend(loc='upper left', fontsize=params.legend_size, frameon=False)
        
        # Format date axis
        from matplotlib.dates import AutoDateLocator, DateFormatter
        locator = AutoDateLocator(minticks=3, maxticks=8, interval_multiples=True)
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
        bbox_inches=None,
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
    
    return out_path


def create_concentration_over_time_plot(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "stage1_concentration_time",
    params: PlotParams = None,
) -> Path:
    """
    Create Figure 10: Trading concentration over time.

    3x1 time series showing % of bonds accounting for X% of dollar volume.

    MEMORY OPTIMIZED: Select columns first, filter, then copy (reduces memory by ~90%).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: trd_exctn_dt, cusip_id, dvolume, spc_rating
    output_dir : Path
        Directory to save the figure
    filename : str
        Base filename
    params : PlotParams
        Plotting parameters

    Returns
    -------
    Path to saved figure

    Caption
    -------
    This figure shows the evolution of trading concentration over time by rating category.
    Each panel displays three lines representing the percentage of bonds needed to account
    for 50 percent, 75 percent, and 90 percent of total dollar volume in each month. Lower
    values indicate higher concentration. For example, if the "50 percent Volume" line shows
    a value of 5 percent, this means that just 5 percent of all bonds account for half of
    the total trading volume in that month, indicating very high concentration. Conversely,
    if the value is 40 percent, then 40 percent of bonds are needed to account for half the
    volume, indicating lower concentration and more dispersed trading activity. The measure
    is computed monthly using dollar volume per bond.
    """
    import gc

    if params is None:
        params = PlotParams()

    apply_plot_params(params)

    rating_categories = [
        ('investment_grade', 'A: Investment Grade'),
        ('non_investment_grade', 'B: Non-Investment Grade'),
        ('defaulted', 'C: Defaulted'),
    ]

    conc_data = {}
    for rating_filter, _ in rating_categories:
        # MEMORY OPTIMIZATION: Select only needed columns FIRST (reduces by ~85%)
        cols_needed = ['trd_exctn_dt', 'cusip_id', 'dvolume', 'spc_rating']
        df_filt = df[cols_needed]

        # Apply rating filter BEFORE copying (reduces by 40-95%)
        if rating_filter == 'investment_grade':
            df_filt = df_filt[(df_filt['spc_rating'] >= 1) & (df_filt['spc_rating'] <= 10)]
        elif rating_filter == 'non_investment_grade':
            df_filt = df_filt[(df_filt['spc_rating'] > 10) & (df_filt['spc_rating'] <= 21)]
        elif rating_filter == 'defaulted':
            df_filt = df_filt[df_filt['spc_rating'] == 22]

        # NOW copy (much smaller - only filtered rows and needed columns)
        # Then select final columns for processing
        df_filt = df_filt[['trd_exctn_dt', 'cusip_id', 'dvolume']].copy()
        df_filt['year_month'] = df_filt['trd_exctn_dt'].dt.to_period('M').astype(str)
        
        bond_volume_monthly = df_filt.groupby(['year_month', 'cusip_id'])['dvolume'].sum().reset_index()
        
        def compute_concentration(group):
            bond_volume = group['dvolume'].sort_values(ascending=False).values
            n_bonds = len(bond_volume)
            
            if n_bonds == 0:
                return pd.Series({
                    'pct_for_50': np.nan,
                    'pct_for_75': np.nan,
                    'pct_for_90': np.nan,
                })
            
            total_volume = bond_volume.sum()
            cumsum_pct = np.cumsum(bond_volume) / total_volume * 100
            
            pct_for_50 = (cumsum_pct <= 50).sum() / n_bonds * 100
            pct_for_75 = (cumsum_pct <= 75).sum() / n_bonds * 100
            pct_for_90 = (cumsum_pct <= 90).sum() / n_bonds * 100
            
            return pd.Series({
                'pct_for_50': pct_for_50,
                'pct_for_75': pct_for_75,
                'pct_for_90': pct_for_90,
            })
        
        df_conc = bond_volume_monthly.groupby('year_month').apply(compute_concentration).reset_index()
        df_conc = df_conc.dropna()
        df_conc['date'] = pd.to_datetime(df_conc['year_month'] + '-01')
        df_conc = df_conc.sort_values('date')
        conc_data[rating_filter] = df_conc

        # Free memory after processing each category
        gc.collect()

    fig_w, fig_h = 8.27, 8.77
    fig = plt.figure(figsize=(fig_w, fig_h))
    
    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        left=0.10, right=0.95, bottom=0.07, top=0.93,
        hspace=0.18
    )
    
    axes = gs.subplots().ravel()
    
    for ax, (rating_filter, panel_label) in zip(axes, rating_categories):
        df_plot = conc_data[rating_filter]
        
        ax.plot(df_plot['date'], df_plot['pct_for_50'],
                color=params.line_color, linestyle='-', linewidth=params.line_lw,
                alpha=params.line_alpha, label='50% Volume')
        
        ax.plot(df_plot['date'], df_plot['pct_for_75'],
                color='red', linestyle='-', linewidth=params.line_lw,
                alpha=params.line_alpha, label='75% Volume')
        
        ax.plot(df_plot['date'], df_plot['pct_for_90'],
                color='blue', linestyle='-', linewidth=params.line_lw,
                alpha=params.line_alpha, label='90% Volume')
        
        ax.set_title(panel_label, pad=2)
        ax.set_xlabel("")
        ax.set_ylabel("% of Bonds")
        ax.grid(True, alpha=params.grid_alpha, linewidth=params.grid_lw)

        # Different decimal places per panel: A=0, B=2, C=3
        from matplotlib.ticker import FormatStrFormatter
        if rating_filter == 'investment_grade':
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        elif rating_filter == 'non_investment_grade':
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        else:  # defaulted
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        if ax == axes[0]:
            ax.legend(loc='upper right', fontsize=params.legend_size, frameon=False)
        
        from matplotlib.dates import AutoDateLocator, DateFormatter
        locator = AutoDateLocator(minticks=3, maxticks=8, interval_multiples=True)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", labelsize=params.tick_size, pad=1)
        ax.tick_params(axis="y", labelsize=params.tick_size)
        ax.margins(x=0.01)
    
    ext = params.export_format.lower()
    if ext not in ["pdf", "png", "jpg", "jpeg"]:
        ext = "pdf"
    
    out_path = output_dir / f"{filename}.{ext}"
    
    savefig_kwargs = dict(
        format=ext,
        bbox_inches=None,
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

    # Clean up memory
    gc.collect()

    return out_path


def create_active_dormant_bonds_plot(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "stage1_active_dormant_bonds",
    params: PlotParams = None,
) -> Path:
    """
    Create Figure 11: Active vs Dormant bonds over time.
    
    3x1 line plot showing percentage of bonds by activity status.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain: trd_exctn_dt, cusip_id, pr, spc_rating
    output_dir : Path
        Directory to save the figure
    filename : str
        Base filename
    params : PlotParams
        Plotting parameters
    
    Returns
    -------
    Path to saved figure
    
    Caption
    -------
    This figure shows the distribution of bonds by trading activity status over time, 
    by rating category. For each month, we identify all unique bonds that have traded 
    at least once up to that month. Each bond is then classified based on days since 
    its last trade (measured from the end of that month): Active (traded in last 30 days), 
    Inactive (last trade 31-90 days ago), and Dormant (last trade more than 90 days ago). 
    The y-axis shows the percentage of bonds in each category, normalized so that the 
    three categories sum to 100 percent in each month. Higher percentages of active bonds 
    indicate more liquid market conditions, while higher dormant percentages suggest 
    many bonds have ceased trading activity.
    """
    if params is None:
        params = PlotParams()
    
    apply_plot_params(params)
    
    # Define rating categories
    rating_categories = [
        ('investment_grade', 'A: Investment Grade'),
        ('non_investment_grade', 'B: Non-Investment Grade'),
        ('defaulted', 'C: Defaulted'),
    ]
    
    # Compute activity status for each category
    activity_data = {}
    for rating_filter, _ in rating_categories:
        df_filt = df.copy()
        
        # Apply rating filter
        if rating_filter == 'investment_grade':
            df_filt = df_filt[(df_filt['spc_rating'] >= 1) & (df_filt['spc_rating'] <= 10)]
        elif rating_filter == 'non_investment_grade':
            df_filt = df_filt[(df_filt['spc_rating'] > 10) & (df_filt['spc_rating'] <= 21)]
        elif rating_filter == 'defaulted':
            df_filt = df_filt[df_filt['spc_rating'] == 22]
        
        # Keep only valid prices and create year_month
        df_filt = df_filt[df_filt['pr'].notna()].copy()
        df_filt['year_month'] = df_filt['trd_exctn_dt'].dt.to_period('M')
        
        # Pre-sort data by date for efficient filtering (once, outside loop)
        df_filt = df_filt.sort_values('trd_exctn_dt')
        
        # Get all unique months
        all_months = sorted(df_filt['year_month'].unique())
        
        # Get bond lifespans for alive check
        bond_life = df_filt.groupby('cusip_id')['year_month'].agg(['min', 'max']).reset_index()
        bond_life.columns = ['cusip_id', 'first_month', 'last_month']
        
        # For each month, classify all bonds alive in that month
        monthly_activity = []
        
        for ym in all_months:
            month_end = ym.to_timestamp('M')  # Last day of month
            
            # Vectorized: which bonds are alive this month?
            alive_mask = (bond_life['first_month'] <= ym) & (bond_life['last_month'] >= ym)
            alive_bonds = set(bond_life[alive_mask]['cusip_id'])
            
            if len(alive_bonds) == 0:
                continue
            
            # Get last trade date UP TO this month for alive bonds (not global!)
            # Efficient: pre-sorted data, filter once
            trades_up_to = df_filt[(df_filt['year_month'] <= ym) & 
                                   (df_filt['cusip_id'].isin(alive_bonds))]
            last_trade_up_to = trades_up_to.groupby('cusip_id')['trd_exctn_dt'].max()
            
            # Calculate days since last trade from end of month (vectorized)
            days_since = (month_end - last_trade_up_to).dt.days
            
            # Classify (vectorized counts)
            active = (days_since <= 30).sum()
            inactive = ((days_since > 30) & (days_since <= 90)).sum()
            dormant = (days_since > 90).sum()
            
            # Normalize to percentages (sum to 100% within month)
            total = active + inactive + dormant
            if total > 0:
                active_pct = (active / total) * 100
                inactive_pct = (inactive / total) * 100
                dormant_pct = (dormant / total) * 100
            else:
                active_pct = inactive_pct = dormant_pct = 0
            
            monthly_activity.append({
                'year_month': str(ym),
                'active': active_pct,
                'inactive': inactive_pct,
                'dormant': dormant_pct,
            })
        
        df_activity = pd.DataFrame(monthly_activity)
        df_activity['date'] = pd.to_datetime(df_activity['year_month'] + '-01')
        activity_data[rating_filter] = df_activity
    
    # Create figure
    fig_w, fig_h = 8.27, 8.77
    fig = plt.figure(figsize=(fig_w, fig_h))
    
    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        left=0.10, right=0.95, bottom=0.07, top=0.93,
        hspace=0.18
    )
    
    axes = gs.subplots().ravel()
    
    for ax, (rating_filter, panel_label) in zip(axes, rating_categories):
        df_plot = activity_data[rating_filter]
        
        # Plot lines
        ax.plot(df_plot['date'], df_plot['active'],
                color=params.line_color, linestyle='-', linewidth=params.line_lw,
                alpha=params.line_alpha, label='Active (<30d)')
        
        ax.plot(df_plot['date'], df_plot['inactive'],
                color='red', linestyle='-', linewidth=params.line_lw,
                alpha=params.line_alpha, label='Inactive (31-90d)')
        
        ax.plot(df_plot['date'], df_plot['dormant'],
                color='blue', linestyle='-', linewidth=params.line_lw,
                alpha=params.line_alpha, label='Dormant (>90d)')
        
        ax.set_title(panel_label, pad=2)
        ax.set_xlabel("")
        ax.set_ylabel("% of Bonds")
        ax.grid(True, alpha=params.grid_alpha, linewidth=params.grid_lw)
        
        # Set y-axis to 0-100%
        ax.set_ylim(0, 100)
        
        # Format y-axis with 0 decimal places
        from matplotlib.ticker import FormatStrFormatter
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        
        # Legend only in first panel
        if ax == axes[0]:
            ax.legend(loc='upper left', fontsize=params.legend_size, frameon=False)
        
        # Format date axis
        from matplotlib.dates import AutoDateLocator, DateFormatter
        locator = AutoDateLocator(minticks=3, maxticks=8, interval_multiples=True)
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
        bbox_inches=None,
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
    
    return out_path


def create_herfindahl_over_time_plot(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "stage1_herfindahl_time",
    params: PlotParams = None,
) -> Path:
    """
    Create Figure 12: Herfindahl-Hirschman Index over time.
    
    3x1 line plot showing HHI (0-10,000 scale) by rating category.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain: trd_exctn_dt, cusip_id, dvolume, spc_rating
    output_dir : Path
        Directory to save the figure
    filename : str
        Base filename
    params : PlotParams
        Plotting parameters
    
    Returns
    -------
    Path to saved figure
    
    Caption
    -------
    This figure shows the evolution of the Herfindahl-Hirschman Index (HHI) over time 
    by rating category. The HHI is computed monthly based on dollar volume per bond and 
    scaled to 0-10,000. HHI is calculated as $HHI = 10000 times sum_{i=1}^{N} s_i^2$ 
    where $s_i$ is bond $i$'s share of total dollar volume in that month. Higher values 
    indicate more concentrated trading, with a value of 10,000 representing a monopoly 
    (one bond accounts for all trading) and lower values indicating more dispersed 
    trading activity across bonds. Values below 1,500 are typically considered 
    unconcentrated, 1,500-2,500 moderately concentrated, and above 2,500 highly 
    concentrated markets.
    """
    if params is None:
        params = PlotParams()
    
    apply_plot_params(params)
    
    # Define rating categories
    rating_categories = [
        ('investment_grade', 'A: Investment Grade'),
        ('non_investment_grade', 'B: Non-Investment Grade'),
        ('defaulted', 'C: Defaulted'),
    ]
    
    # Compute HHI for each category
    hhi_data = {}
    for rating_filter, _ in rating_categories:
        df_filt = df.copy()
        
        # Apply rating filter
        if rating_filter == 'investment_grade':
            df_filt = df_filt[(df_filt['spc_rating'] >= 1) & (df_filt['spc_rating'] <= 10)]
        elif rating_filter == 'non_investment_grade':
            df_filt = df_filt[(df_filt['spc_rating'] > 10) & (df_filt['spc_rating'] <= 21)]
        elif rating_filter == 'defaulted':
            df_filt = df_filt[df_filt['spc_rating'] == 22]
        
        df_filt['year_month'] = df_filt['trd_exctn_dt'].dt.to_period('M').astype(str)
        
        # For each month, compute HHI
        monthly_hhi = []
        for ym in sorted(df_filt['year_month'].unique()):  # CRITICAL: sort months!
            df_month = df_filt[df_filt['year_month'] == ym]
            
            # Total volume per bond (FAST: groupby sum)
            bond_volume = df_month.groupby('cusip_id')['dvolume'].sum()
            if len(bond_volume) == 0:
                continue
            
            total_volume = bond_volume.sum()
            
            # Compute HHI (0-10000 scale)
            market_shares = bond_volume / total_volume
            hhi = (market_shares ** 2).sum() * 10000
            
            monthly_hhi.append({
                'year_month': ym,
                'hhi': hhi,
            })
        
        df_hhi = pd.DataFrame(monthly_hhi)
        df_hhi['date'] = pd.to_datetime(df_hhi['year_month'] + '-01')
        df_hhi = df_hhi.sort_values('date')  # CRITICAL: ensure sorted!
        hhi_data[rating_filter] = df_hhi
    
    # Create figure
    fig_w, fig_h = 8.27, 8.77
    fig = plt.figure(figsize=(fig_w, fig_h))
    
    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        left=0.10, right=0.95, bottom=0.07, top=0.93,
        hspace=0.18
    )
    
    axes = gs.subplots().ravel()
    
    for ax, (rating_filter, panel_label) in zip(axes, rating_categories):
        df_plot = hhi_data[rating_filter]
        
        # Plot HHI line
        ax.plot(df_plot['date'], df_plot['hhi'],
                color=params.line_color, linestyle='-', linewidth=params.line_lw,
                alpha=params.line_alpha)
        
        ax.set_title(panel_label, pad=2)
        ax.set_xlabel("")
        ax.set_ylabel("HHI (0-10,000)")
        ax.grid(True, alpha=params.grid_alpha, linewidth=params.grid_lw)
        
        # Set y-axis to 0-10000
        ax.set_ylim(0, 10000)
        
        # Format y-axis with 0 decimal places
        from matplotlib.ticker import FormatStrFormatter
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        
        # Format date axis
        from matplotlib.dates import AutoDateLocator, DateFormatter
        locator = AutoDateLocator(minticks=3, maxticks=8, interval_multiples=True)
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
        bbox_inches=None,
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
    
    return out_path


def create_trading_intensity_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "stage1_trading_intensity_heatmap",
    params: PlotParams = None,
) -> Path:
    """
    Create Figure 13: Heatmap of trading intensity (month -- rating group).
    
    Rows: 5 rating groups (1-5, 6-10, 11-15, 16-21, 22)
    Columns: Year-Month
    Color: Average probability of trade (%)
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain: trd_exctn_dt, cusip_id, pr, spc_rating
    output_dir : Path
        Directory to save the figure
    filename : str
        Base filename
    params : PlotParams
        Plotting parameters
    
    Returns
    -------
    Path to saved figure
    
    Caption
    -------
    This figure displays a heatmap showing the intensity of trading activity across 
    different rating categories and time periods. The y-axis shows five rating groups 
    (from highest rated AAA-A minus at the top to defaulted D at the bottom), while 
    the x-axis shows time in monthly intervals. The color intensity represents the 
    average probability of observing a valid price on any business day within that 
    month for bonds in that rating category, expressed as a percentage. Darker colors 
    indicate higher trading intensity (bonds trade more frequently), while lighter 
    colors indicate lower trading intensity or illiquid periods. This visualization 
    allows for quick identification of patterns in trading activity across rating 
    categories and time, such as periods of market stress or differential liquidity 
    between rating grades. A bond is considered "alive" from its first observed trade 
    date to its last observed trade date in TRACE, for the purposes of computing 
    average trades within a month.
    """
    if params is None:
        params = PlotParams()
    
    apply_plot_params(params)
    
    # Define rating groups
    rating_groups = [
        (1, 5, '1-5 (AAA-A-)'),
        (6, 10, '6-10 (BBB+-)'),
        (11, 15, '11-15 (BB+-)'),
        (16, 21, '16-21 (B+-)'),
        (22, 22, '22 (D)'),
    ]
    
    # Get business days per month
    min_date = df['trd_exctn_dt'].min().strftime('%Y-%m-%d')
    max_date = df['trd_exctn_dt'].max().strftime('%Y-%m-%d')
    bdays_df = compute_business_days_per_month(min_date, max_date)
    
    # Compute probability for each rating group and month (accounting for zero-trade months)
    heatmap_data = []
    
    for min_rating, max_rating, label in rating_groups:
        # Memory optimization: select columns first, filter, then copy
        df_temp = df[['trd_exctn_dt', 'cusip_id', 'pr', 'spc_rating']]
        df_temp = df_temp[(df_temp['spc_rating'] >= min_rating) & (df_temp['spc_rating'] <= max_rating)]
        df_group = df_temp.copy()
        df_group['year_month'] = df_group['trd_exctn_dt'].dt.to_period('M')
        
        # Step 1: Get bond lifespans (lightweight)
        bond_life = df_group.groupby('cusip_id')['year_month'].agg(['min', 'max']).reset_index()
        bond_life.columns = ['cusip_id', 'first_month', 'last_month']
        
        # Step 2: Get actual trade counts (only where trades exist)
        trade_counts = df_group.groupby(['cusip_id', 'year_month'])['pr'].count().reset_index()
        trade_counts.columns = ['cusip_id', 'year_month', 'days_traded']
        
        # Step 3: Get all unique months
        all_months = sorted(df_group['year_month'].unique())
        
        # Step 4: Loop over months and compute averages including zeros
        results = []
        for month in all_months:
            # Vectorized: which bonds are alive this month?
            alive_mask = (bond_life['first_month'] <= month) & (bond_life['last_month'] >= month)
            num_alive = alive_mask.sum()
            
            if num_alive == 0:
                continue
            
            # Get trades for this month
            month_trades = trade_counts[trade_counts['year_month'] == month]
            total_days = month_trades['days_traded'].sum()
            
            # Average including zeros (divide by ALL alive bonds, not just traded)
            avg_days = total_days / num_alive
            
            results.append({
                'year_month': str(month),
                'avg_days': avg_days,
            })
        
        avg_days_df = pd.DataFrame(results)
        
        # Merge with business days
        avg_days_df = avg_days_df.merge(bdays_df, on='year_month', how='left')
        
        # Compute probability
        avg_days_df['probability'] = (avg_days_df['avg_days'] / avg_days_df['business_days']) * 100
        avg_days_df['rating_group'] = label

        heatmap_data.append(avg_days_df[['year_month', 'rating_group', 'probability']])

        # Clean up memory after processing each rating group
        gc.collect()
    
    # Combine all groups
    df_heatmap = pd.concat(heatmap_data, ignore_index=True)
    
    # Pivot for heatmap
    heatmap_pivot = df_heatmap.pivot(index='rating_group', columns='year_month', values='probability')
    
    # Reorder rows (reverse so AAA at top)
    row_order = [label for _, _, label in rating_groups]
    heatmap_pivot = heatmap_pivot.reindex(row_order)
    
    # Create figure (wider for many months)
    fig_w, fig_h = 11.69, 5.85  # A4 landscape-ish
    fig = plt.figure(figsize=(fig_w, fig_h))
    
    ax = fig.add_subplot(111)
    
    # Create heatmap
    import matplotlib.colors as mcolors
    cmap = plt.cm.YlOrRd  # Yellow to Orange to Red
    
    im = ax.imshow(heatmap_pivot.values, cmap=cmap, aspect='auto', 
                   vmin=0, vmax=100, interpolation='nearest')
    
    # Set ticks
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=params.tick_size)
    
    # X-axis: show every Nth month
    n_months = len(heatmap_pivot.columns)
    tick_spacing = max(1, n_months // 20)  # ~20 ticks max
    x_ticks = range(0, n_months, tick_spacing)
    x_labels = [heatmap_pivot.columns[i] for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=params.tick_size - 1)
    
    ax.set_xlabel("Year-Month", fontsize=params.label_size)
    ax.set_ylabel("Rating Group", fontsize=params.label_size)
    ax.set_title("Trading Intensity: Probability of Trade (%)", fontsize=params.title_size, pad=10)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02, fraction=0.046)
    cbar.set_label('Probability (%)', fontsize=params.label_size)
    cbar.ax.tick_params(labelsize=params.tick_size)
    
    plt.tight_layout()
    
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

    # Clean up memory
    gc.collect()

    return out_path


def make_data_availability_table(df: pd.DataFrame, min_date: str, max_date: str) -> str:
    """
    Generate Table 3: Data Availability by Rating Category.
    
    Shows the number of observations and % missing for key variables across
    four rating categories: All Bonds, Investment Grade, Non-Investment Grade, 
    and Defaulted bonds.
    
    Parameters
    ----------
    df : pd.DataFrame
        The final cleaned dataset
    min_date : str
        Minimum date in data (YYYY-MM-DD)
    max_date : str
        Maximum date in data (YYYY-MM-DD)
    
    Returns
    -------
    str
        LaTeX table code
    """
    
    # Define the variables and their display names
    variables = [
        ('pr', 'Price (VW)'),
        ('prc_bid', 'Price (Bid)'),
        ('prc_ask', 'Price (Ask)'),
        ('credit_spread', 'Spread'),
        ('sp_rating', 'Rating (SP)'),
        ('mdy_rating', 'Rating (MD)'),
        ('permno', 'PERMNO'),
    ]
    
    # Define rating categories
    df_all = df.copy()
    df_ig = df[(df['spc_rating'] >= 1) & (df['spc_rating'] <= 10)].copy()
    df_nig = df[(df['spc_rating'] > 10) & (df['spc_rating'] <= 21)].copy()
    df_def = df[df['spc_rating'] == 22].copy()
    
    # Helper function to format numbers with commas
    def format_with_commas(n):
        return f"{n:,}"
    
    # Compute statistics for each panel
    def compute_stats(df_panel):
        total_rows = len(df_panel)
        stats = []
        for var_name, display_name in variables:
            if var_name in df_panel.columns:
                non_null_count = df_panel[var_name].notna().sum()
                null_count = df_panel[var_name].isna().sum()
                pct_missing = (null_count / total_rows * 100) if total_rows > 0 else 0.0
                stats.append({
                    'variable': display_name,
                    'observations': non_null_count,
                    'pct_missing': pct_missing
                })
            else:
                # Variable not in dataframe
                stats.append({
                    'variable': display_name,
                    'observations': 0,
                    'pct_missing': 100.0
                })
        return stats
    
    stats_all = compute_stats(df_all)
    stats_ig = compute_stats(df_ig)
    stats_nig = compute_stats(df_nig)
    stats_def = compute_stats(df_def)
    
    # Build LaTeX table rows
    rows = []
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
    
    # Note about excluded variables
    note_text = (
        r"This table reports data availability for key variables across rating categories. "
        r"For each panel, we report the number of non-missing observations and the "
        r"percentage of missing values. "
        r"Panel A includes all bonds in the sample. "
        r"Panel B includes investment grade bonds (S\&P ratings 1--10, AAA to BBB$-$). "
        r"Panel C includes non-investment grade bonds (S\&P ratings 11--21, BB+ to CCC$-$). "
        r"Panel D includes defaulted bonds (S\&P rating 22, D). "
        r"The sample spans the period " + min_date + r" to " + max_date + r". "
        r"All other variables in the dataset (not shown) have zero missing observations. "
        r"Accrued interest is computed assuming a 2-day settlement period using the modified following rule "
        r"using the \href{https://www.quantlib.org/}{\texttt{QuantLib}} Python package. "
        r"Spreads (credit spreads) are computed by interpolating over \texttt{bond\_maturity} using the "
        r"constant maturity zero-coupon yield curve data at key rates, made available from "
        r"\cite{liu2021reconstructing}'s \href{https://sites.google.com/view/jingcynthiawu/yield-data}{\texttt{website}}."
    )
    
    latex = r"""
\begin{table}[!ht]
\begin{center}
\footnotesize
\caption{Data Availability by Rating Category}
\label{tab:data_availability}\vspace{2mm}
\scalebox{0.9}{%
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


def make_concentration_table(stats_ig: dict, stats_nig: dict, stats_def: dict,
                             min_date: str, max_date: str) -> str:
    """
    Generate Table 8: Trading Concentration Metrics.
    
    Parameters
    ----------
    stats_ig : dict
        Statistics for investment grade
    stats_nig : dict
        Statistics for non-investment grade
    stats_def : dict
        Statistics for defaulted
    min_date : str
        Minimum date (YYYY-MM-DD)
    max_date : str
        Maximum date (YYYY-MM-DD)
    
    Returns
    -------
    str
        LaTeX table code
    """
    
    latex = r"""
\begin{table}[!ht]
\begin{center}
\footnotesize
\caption{Trading Concentration Metrics by Rating Category}
\label{tab:concentration}\vspace{2mm}
\begin{tabular}{lrrr}
\midrule
Metric & Investment & Non-Investment & Defaulted \\
       & Grade (1-10) & Grade (11-21) & (22) \\
\midrule
\multicolumn{4}{l}{\textbf{Panel A: Volume Share of Top Bonds (\%)}} \\
\midrule
Top 10\% of bonds & """ + f"{stats_ig['top10_share']:.3f}" + r""" & """ + f"{stats_nig['top10_share']:.3f}" + r""" & """ + f"{stats_def['top10_share']:.3f}" + r""" \\
Top 25\% of bonds & """ + f"{stats_ig['top25_share']:.3f}" + r""" & """ + f"{stats_nig['top25_share']:.3f}" + r""" & """ + f"{stats_def['top25_share']:.3f}" + r""" \\
\midrule
\multicolumn{4}{l}{\textbf{Panel B: Bonds Needed for Volume Share (\% of bonds)}} \\
\midrule
50\% of volume & """ + f"{stats_ig['pct_for_50']:.3f}" + r""" & """ + f"{stats_nig['pct_for_50']:.3f}" + r""" & """ + f"{stats_def['pct_for_50']:.3f}" + r""" \\
90\% of volume & """ + f"{stats_ig['pct_for_90']:.3f}" + r""" & """ + f"{stats_nig['pct_for_90']:.3f}" + r""" & """ + f"{stats_def['pct_for_90']:.3f}" + r""" \\
\bottomrule
\end{tabular}
\end{center}
\begin{spacing}{1}
\footnotesize{
This table presents trading concentration metrics by rating category for the period """ + min_date + r""" to """ + max_date + r""".
Panel A shows the percentage of total dollar volume (\texttt{dvolume}) captured by the 
top 10\% and 25\% most actively traded bonds (higher values indicate more concentration).
Panel B shows the percentage of bonds needed to account for 50\% and 90\% of total 
dollar volume (lower values indicate higher concentration, e.g., 5\% means that just 
5\% of bonds account for 50\% of all trading volume).
Concentration is computed over the entire sample period using dollar volume per bond.
}
\end{spacing}
\vspace{-2mm}
\end{table}
""".strip()
    
    return latex


def compute_pooled_stats_fixed(df: pd.DataFrame, stat_vars: list) -> pd.DataFrame:
    """
    Compute pooled descriptive statistics (Panel A).
    
    Changes from original:
    - Drops Skew and Kurt columns
    - Rounds all to 2 decimal places
    - Scales YTM and Spread by 100 (present as percentages)
    - Includes sp_rating and mdy_rating
    """
    stats_list = []
    
    for var_name, label in stat_vars:
        if var_name not in df.columns:
            continue
        
        series = df[var_name].dropna()
        if len(series) == 0:
            continue
        
        # Scale if needed
        if var_name in ['ytm', 'credit_spread']:
            series = series * 100
        
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


def compute_cross_sectional_stats_fixed(df: pd.DataFrame, stat_vars: list) -> pd.DataFrame:
    """
    Compute cross-sectional statistics (Panel B) - time-series averages of daily stats.
    
    Changes from original:
    - Drops Skew and Kurt columns
    - Rounds all to 2 decimal places
    - Scales YTM and Spread by 100 (present as percentages)
    - Includes sp_rating and mdy_rating
    """
    stats_list = []
    
    for var_name, label in stat_vars:
        if var_name not in df.columns:
            continue
        
        # Compute daily statistics
        daily_mean = df.groupby('trd_exctn_dt')[var_name].mean()
        daily_median = df.groupby('trd_exctn_dt')[var_name].median()
        daily_std = df.groupby('trd_exctn_dt')[var_name].std()
        daily_p1 = df.groupby('trd_exctn_dt')[var_name].quantile(0.01)
        daily_p5 = df.groupby('trd_exctn_dt')[var_name].quantile(0.05)
        daily_p95 = df.groupby('trd_exctn_dt')[var_name].quantile(0.95)
        daily_p99 = df.groupby('trd_exctn_dt')[var_name].quantile(0.99)
        
        # Time-series average of daily stats
        stats = {
            'Variable': label,
            'Mean': daily_mean.mean(),
            'Median': daily_median.mean(),
            'SD': daily_std.mean(),
            'P1': daily_p1.mean(),
            'P5': daily_p5.mean(),
            'P95': daily_p95.mean(),
            'P99': daily_p99.mean(),
        }
        
        # Scale YTM and Spread by 100 AFTER computation
        if var_name in ['ytm', 'credit_spread']:
            for key in ['Mean', 'Median', 'SD', 'P1', 'P5', 'P95', 'P99']:
                stats[key] = stats[key] * 100
        
        # Round to 2 decimals
        for key in ['Mean', 'Median', 'SD', 'P1', 'P5', 'P95', 'P99']:
            stats[key] = round(stats[key], 2)
        
        stats_list.append(stats)
    
    return pd.DataFrame(stats_list)


def make_descriptive_stats_table_fixed(panel_a: pd.DataFrame, panel_b: pd.DataFrame, 
                                       min_date: str, max_date: str) -> str:
    """
    Generate Table 3: TRACE Daily Descriptive Statistics.
    
    Parameters
    ----------
    panel_a : pd.DataFrame
        Pooled statistics
    panel_b : pd.DataFrame
        Cross-sectional statistics
    min_date : str
        Minimum date in data (YYYY-MM-DD)
    max_date : str
        Maximum date in data (YYYY-MM-DD)
    
    Changes from original:
    - No Skew and Kurt columns
    - All values already rounded to 2 decimals in compute functions
    - Variables ytm, credit_spread, dvolume, qvolume formatted in \texttt{} in caption
    """
    
    def format_stats_df(df: pd.DataFrame) -> str:
        rows = []
        for _, row in df.iterrows():
            var = escape_latex(row['Variable'])
            vals = [
                f"{row['Mean']:.2f}",
                f"{row['Median']:.2f}",
                f"{row['SD']:.2f}",
                f"{row['P1']:.2f}",
                f"{row['P5']:.2f}",
                f"{row['P95']:.2f}",
                f"{row['P99']:.2f}",
            ]
            rows.append(f"{var} & " + " & ".join(vals) + r" \\")
        return "\n".join(rows)
    
    panel_a_tex = format_stats_df(panel_a)
    panel_b_tex = format_stats_df(panel_b)
    
    latex = r"""
\begin{table}[!ht]
\begin{center}
\footnotesize
\caption{TRACE Daily Descriptive Statistics}
\label{tab:descriptive_stats}\vspace{2mm}
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
\end{center}
\begin{spacing}{1}
\footnotesize{
This table presents descriptive statistics for the cleaned TRACE daily dataset.
Panel A shows statistics pooled across all cusip-date observations.
Panel B shows time-series averages of daily cross-sectional statistics.
The sample spans the period """ + min_date + r""" to """ + max_date + r""".
All prices are in percentage of par, 100\% implies a dollar value of \$1000. All par values are exactly \$1000.
Yield to maturity (\texttt{ytm}) and Spread (\texttt{credit\_spread}) are in percentage points.
Duration, Bond Maturity and Age are in years.
Volumes (\texttt{dvolume} and \texttt{qvolume}) are in millions of U.S. dollars.
Ratings are in numeric format (AAA = 1, ..., D = 22).
Variables \texttt{ytm}, \texttt{credit\_spread}, \texttt{dvolume}, and \texttt{qvolume} are winsorized at 0.5th and 99.5th percentiles within each date.
Accrued interest is computed assuming a 2-day settlement period using the modified following rule using the \href{https://www.quantlib.org/}{\texttt{QuantLib}} Python package.
Spreads (credit spreads) are computed by interpolating over \texttt{bond\_maturity} using the constant maturity zero-coupon yield curve data at key rates, made available from 
\cite{liu2021reconstructing}'s \href{https://sites.google.com/view/jingcynthiawu/yield-data}{\texttt{website}}.
}
\end{spacing}
\vspace{-2mm}
\end{table}
""".strip()
    
    return latex

def make_descriptive_stats_table_by_rating(panel_a: pd.DataFrame, panel_b: pd.DataFrame, 
                                           min_date: str, max_date: str,
                                           table_number: int, title: str, 
                                           rating_range_text: str) -> str:
    """
    Generate descriptive statistics table for specific rating category.
    
    Parameters
    ----------
    panel_a : pd.DataFrame
        Pooled statistics
    panel_b : pd.DataFrame
        Cross-sectional statistics
    min_date : str
        Minimum date in data (YYYY-MM-DD)
    max_date : str
        Maximum date in data (YYYY-MM-DD)
    table_number : int
        Table number (4, 5, or 6)
    title : str
        Table title (e.g., "Investment Grade Corporate Bonds")
    rating_range_text : str
        Rating range description (e.g., "Ratings 1-10 (AAA to BBB-)")
    
    Returns
    -------
    str
        LaTeX table code
    """
    
    def format_stats_df(df: pd.DataFrame) -> str:
        rows = []
        for _, row in df.iterrows():
            var = escape_latex(row['Variable'])
            vals = [
                f"{row['Mean']:.2f}",
                f"{row['Median']:.2f}",
                f"{row['SD']:.2f}",
                f"{row['P1']:.2f}",
                f"{row['P5']:.2f}",
                f"{row['P95']:.2f}",
                f"{row['P99']:.2f}",
            ]
            rows.append(f"{var} & " + " & ".join(vals) + r" \\")
        return "\n".join(rows)
    
    panel_a_tex = format_stats_df(panel_a)
    panel_b_tex = format_stats_df(panel_b)
    
    latex = r"""
\begin{table}[!ht]
\begin{center}
\footnotesize
\caption{TRACE Daily Descriptive Statistics: """ + title + r"""}
\label{tab:descriptive_stats_""" + str(table_number) + r"""}\vspace{2mm}
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
\end{center}
\begin{spacing}{1}
\footnotesize{
This table presents descriptive statistics for the cleaned TRACE daily dataset, restricted to """ + rating_range_text + r""".
Panel A shows statistics pooled across all cusip-date observations.
Panel B shows time-series averages of daily cross-sectional statistics.
The sample spans the period """ + min_date + r""" to """ + max_date + r""".
All prices are in percentage of par, 100\% implies a dollar value of \$1000. All par values are exactly \$1000.
Yield to maturity (\texttt{ytm}) and Spread (\texttt{credit\_spread}) are in percentage points.
Duration, Bond Maturity and Age are in years.
Volumes (\texttt{dvolume} and \texttt{qvolume}) are in millions of U.S. dollars.
Ratings are in numeric format (AAA = 1, ..., D = 22).
Variables \texttt{ytm}, and \texttt{credit\_spread} are winsorized at 0.5th and 99.5th percentiles within each date.
Accrued interest is computed assuming a 2-day settlement period using the modified following rule using the \href{https://www.quantlib.org/}{\texttt{QuantLib}} Python package.
Spreads (credit spreads) are computed by interpolating over \texttt{bond\_maturity} using the constant maturity zero-coupon yield curve data at key rates, 
made available from \cite{liu2021reconstructing}'s \href{https://sites.google.com/view/jingcynthiawu/yield-data}{\texttt{website}}.
}
\end{spacing}
\vspace{-2mm}
\end{table}
""".strip()
    
    return latex



# ============================================================================
# LATEX HELPER FUNCTIONS FOR REPORTS
# ============================================================================

def escape_latex(s: str) -> str:
    """Escape special LaTeX characters."""
    s = str(s)
    s = s.replace('_', r'\_')
    s = s.replace('%', r'\%')
    s = s.replace('&', r'\&')
    s = s.replace('#', r'\#')
    return s


def format_value_latex(v):
    """Format a value for LaTeX table cell."""
    if isinstance(v, bool):
        return r"\texttt{On}" if v else r"\texttt{Off}"
    if isinstance(v, (int, np.integer)):
        return f"{int(v)}"
    if isinstance(v, (float, np.floating)):
        return f"{v:.8g}"
    if isinstance(v, (tuple, list)):
        inside = ", ".join(format_value_latex(x) for x in v)
        inside = inside.replace(r"\texttt{", "").replace("}", "")
        bracket = "(" if isinstance(v, tuple) else "["
        close = ")" if isinstance(v, tuple) else "]"
        return r"\texttt{" + bracket + inside + close + "}"
    s = escape_latex(str(v))
    return r"\texttt{" + s + "}"


def make_inputs_table(config_dict: dict, date_cut_off: str, final_filter_config: dict, 
                      min_date: str) -> str:
    """
    Generate Table 1: Daily Data Filter Configuration.
    
    Parameters
    ----------
    config_dict : dict
        Ultra distressed filter configuration
    date_cut_off : str
        Date cutoff string (YYYY-MM-DD)
    final_filter_config : dict
        Final filter configuration with price_threshold and dip_threshold
    min_date : str
        Minimum date in the data (YYYY-MM-DD)
    """
    rows = []
    
    # Add date_cut_off at the top
    key_tex = r"\texttt{date\_cut\_off}"
    val_tex = format_value_latex(date_cut_off)
    rows.append(f"{key_tex} & {val_tex} \\\\")
    
    # Add ultra distressed config parameters
    for key, val in config_dict.items():
        key_tex = r"\texttt{" + escape_latex(key) + "}"
        val_tex = format_value_latex(val)
        rows.append(f"{key_tex} & {val_tex} \\\\")
    
    # Add final filter parameters at the bottom
    for key in ['price_threshold', 'dip_threshold']:
        if key in final_filter_config:
            key_tex = r"\texttt{" + escape_latex(key) + "}"
            val_tex = format_value_latex(final_filter_config[key])
            rows.append(f"{key_tex} & {val_tex} \\\\")
    
    rows_tex = "\n".join(rows)
    
    # Build caption with date range
    date_range_str = f"The data spans the period {min_date} to {date_cut_off}."
    
    latex = r"""
\begin{table}[!ht]
\begin{center}
\footnotesize
\caption{Daily Data Filter Configuration}
\label{tab:daily_inputs}\vspace{2mm}
\begin{tabular}{lc}
\midrule
Parameter & Value \\
\midrule
""" + rows_tex + r"""
\bottomrule
\end{tabular}
\end{center}
\begin{spacing}{1}
\footnotesize{
This table documents the filter configuration parameters used in the data pipeline. 
The ultra distressed filter configuration parameters (lines 2--""" + str(len(config_dict)+1) + r""") 
identify and flag potentially erroneous trades. The \texttt{price\_threshold} removes bonds trading above 300\% of par.
The \texttt{dip\_threshold} removes observations only in 2002--07 (the very first years TRACE was introduced, hence more prone to issues) 
if there is an absolute price change of greater than 35\% between the first price record and the second price record for a bond; this filter 
catches corner cases the other filters did not catch. """ + date_range_str + r"""
}
\end{spacing}
\vspace{-2mm}
\end{table}
""".strip()
    
    return latex


def make_filter_records_table(records: list) -> str:
    """Generate Table 2: TRACE Daily Filter Records."""
    rows = []
    for stage, n_pre, n_post, removed, pct in records:
        stage_tex = r"\texttt{" + escape_latex(stage) + "}"
        rows.append(
            f"{stage_tex} & {n_pre:,} & {n_post:,} & {removed:,} & {pct:.3f} \\\\"
        )
    
    rows_tex = "\n".join(rows)
    
    latex = r"""
\begin{table}[!ht]
\begin{center}
\footnotesize
\caption{TRACE Daily Filter Records}
\label{tab:filter_records}\vspace{2mm}
\begin{tabular}{lrrrr}
\midrule
Filter & N$_{pre}$ & N$_{post}$ & Removed & \% Removed \\
\midrule
""" + rows_tex + r"""
\bottomrule
\end{tabular}
\end{center}
\begin{spacing}{1}
\footnotesize{
This table presents the sequential application of filters to the TRACE daily dataset.
N$_{pre}$ and N$_{post}$ indicate row counts before and after each filter.
\% Removed is calculated as the number of transactions removed divided by the total starting transactions.
The filters are defined as follows: 
\texttt{valid\_accrued\_vars} requires valid variables from FISD relating to a valid \texttt{dated\_date}, 
\texttt{interest\_frequency}, and \texttt{pr} (value-weighted bond price); it also requires \texttt{bond\_maturity} and \texttt{bond\_age} $>$ 0.
\texttt{valid\_rating} requires a bond to have either a valid S\&P or Moody's rating.
\texttt{valid\_maturity} requires a bond to have a maturity $>$ 1 year.
\texttt{distressed\_errors} implements the \texttt{ultra\_distressed\_filter()} function with parameters set as in Table 1.
\texttt{2002-07\_filter} removes observations only in 2002--07 (the very first years TRACE was introduced, hence more prone to issues) 
if there is an absolute price change of greater than 35\% between the first price record and the second price record for a bond; this filter catches 
corner cases the other filters did not catch.
\texttt{high\_prc} removes bonds trading above the \texttt{price\_threshold} specified in Table 1.
}
\end{spacing}
\vspace{-2mm}
\end{table}
""".strip()
    
    return latex


def build_latex_document(table1: str, table2: str, table3: str, table4: str,
                        table5: str = None, table6: str = None, table7: str = None,
                        table8: str = None,
                        fig_filenames: list = None, author: str = None) -> str:
    """
    Build complete LaTeX document with optional rating-specific tables and figures.

    Parameters
    ----------
    table1 : str
        Configuration parameters table
    table2 : str
        Filter records table
    table3 : str
        Data availability table
    table4 : str
        Descriptive statistics table (all bonds)
    table5 : str, optional
        Descriptive statistics table (investment grade bonds)
    table6 : str, optional
        Descriptive statistics table (non-investment grade bonds)
    table7 : str, optional
        Descriptive statistics table (defaulted bonds)
    table8 : str, optional
        Trading concentration metrics table
    fig_filenames : list of tuples, optional
        List of (filename, caption) tuples for figures
        E.g., [('fig1.pdf', 'All Bonds'), ('fig2.pdf', 'Investment Grade')]
    author : str, optional
        Author name for the document

    Returns
    -------
    str
        Complete LaTeX document
    """
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d')
    
    # Build author line if provided
    author_line = f"\\author{{{author}}}" if author else ""
    
    # Build rating-specific tables section if provided
    rating_tables_section = ""
    if table5:
        rating_tables_section += r"""

\clearpage
""" + table5
    if table6:
        rating_tables_section += r"""

\clearpage
""" + table6
    if table7:
        rating_tables_section += r"""

\clearpage
""" + table7
    if table8:
        rating_tables_section += r"""

\clearpage
""" + table8

    # Build figures section if provided
    figures_section = ""
    if fig_filenames:
        figures_section = r"""

\clearpage
\section{Time-Series Plots}
"""
        for i, (filename, caption) in enumerate(fig_filenames, start=1):
            if i > 1:
                figures_section += r"\clearpage" + "\n"

            # Add scalebox for Figures 1-4 (time-series plots), Figure 13 (trade size distribution), Figure 14-15 (bond characteristics, rating/maturity)
            if i <= 4 or i == 13 or i == 14 or i == 15:
                figures_section += r"""
\begin{figure}[h!]
\centering
\scalebox{0.8}{%
  \includegraphics[width=\textwidth,height=\textheight,keepaspectratio]{""" + filename + r"""}%
}
\caption{""" + caption + r"""}
\end{figure}
"""
            else:
                figures_section += r"""
\begin{figure}[h!]
\centering
\includegraphics[width=\textwidth,height=\textheight,keepaspectratio]{""" + filename + r"""}
\caption{""" + caption + r"""}
\end{figure}
"""
    
    doc = r"""\documentclass[11pt]{article}
\usepackage{graphicx,booktabs,geometry,ragged2e,setspace}
\usepackage{amsmath,amssymb}
\usepackage[round,authoryear]{natbib}
\usepackage{hyperref}
\geometry{margin=1in}
\title{Stage 1 TRACE Daily Data Report}
""" + author_line + r"""
\date{""" + timestamp + r"""}
\begin{document}
\maketitle

\begin{abstract}
This document presents a comprehensive analysis of corporate bond market data derived from the 
Trade Reporting and Compliance Engine (TRACE) database. We construct a daily panel of corporate 
bond metrics using the \href{https://www.quantlib.org/}{\texttt{QuantLib}} Python package, 
computing credit spreads, accrued interest, duration measures, and other key bond characteristics. 
The analysis includes detailed descriptive statistics and visualizations that examine trading 
patterns, liquidity dynamics, and rating-based market segmentation. This work is part of the 
\href{https://openbondassetpricing.com/}{Open Source Bond Asset Pricing} initiative 
\citep{DickersonRobottiRossetti_2024}, which aims to provide transparent and reproducible methods 
for corporate bond research.
\end{abstract}

\section{Configuration Parameters}
""" + table1 + r"""

\clearpage
\section{Filter Records}
""" + table2 + r"""

\clearpage
\section{Data Availability}
""" + table3 + r"""

\clearpage
\section{Descriptive Statistics}
""" + table4 + rating_tables_section + figures_section + r"""

\clearpage
\bibliographystyle{apalike}
\bibliography{references}
\end{document}
"""
    
    return doc



def get_references_bib() -> str:
    """Return bibliography content."""
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
  note = {Working Paper, Warwick Business School},
  title = {Common Pitfalls in the Evaluation of Corporate Bond Strategies},
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

@article{liu2021reconstructing,
  title={Reconstructing the yield curve},
  author={Liu, Yan and Wu, Jing Cynthia},
  journal={Journal of Financial Economics},
  volume={142},
  number={3},
  pages={1395--1425},
  year={2021},
  publisher={Elsevier}
}
""".strip()


def standardize_float_dtypes(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Convert Float64 (nullable) columns to float64 (standard numpy) for consistency.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to standardize
    verbose : bool
        Print conversion info
    
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized dtypes
    """
    df = df.copy()
    float64_cols = []
    
    for col in df.columns:
        if df[col].dtype.name == 'Float64':
            df[col] = df[col].astype('float64')
            float64_cols.append(col)
    
    if verbose and float64_cols:
        print(f"Converted {len(float64_cols)} columns from Float64 to float64:")
        for col in float64_cols:
            print(f"  - {col}")
    
    return df

def add_ff_industries(fisd_df: pd.DataFrame, verbose: bool = True):
    """
    Add Fama-French 17 and 30 industry classifications to FISD data based on SIC codes.

    Downloads and parses both FF17 and FF30 industry classification files, then matches
    SIC codes to industries using fast vectorized operations. Returns only numeric codes
    (ff17num, ff30num) to save RAM; mappings returned separately for plotting.

    If internet is not available (e.g., on WRDS compute nodes), falls back to local files.

    Parameters
    ----------
    fisd_df : pd.DataFrame
        FISD dataframe with 'sic_code' column
    verbose : bool
        Print progress information

    Returns
    -------
    tuple
        (fisd_df, ff17_mapping, ff30_mapping)
        - fisd_df: DataFrame with added 'ff17num' (1-17) and 'ff30num' (1-30) columns
        - ff17_mapping: dict mapping industry number to name for FF17
        - ff30_mapping: dict mapping industry number to name for FF30
        Missing/unmatched SIC codes are assigned to industry 17 (Other) for FF17
        and industry 30 (Other) for FF30

    Notes
    -----
    - Uses pandas IntervalIndex for fast O(log n) lookups
    - Vectorized for performance with large datasets (100k+ rows)
    - Only numeric codes stored in dataframe to minimize RAM usage
    - Automatically handles offline mode for WRDS compute nodes
    """
    if verbose:
        print("Adding Fama-French 17 and 30 industry classifications...")

    fisd_df = fisd_df.copy()

    # Check internet connectivity once at the start
    has_internet = _check_internet_connectivity()

    # ========================================================================
    # Process FF17
    # ========================================================================
    ff17_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes17.zip"
    ff17_local_file = "data/Siccodes17.txt"
    ff17_mapping = {}

    try:
        # Try to get FF17 content - either from internet or local file
        content = None

        if has_internet:
            try:
                if verbose:
                    print(f"  Internet available - downloading FF17 file...")
                response = requests.get(ff17_url, timeout=30)
                response.raise_for_status()

                # Extract text file from zip
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    with z.open('Siccodes17.txt') as f:
                        content = f.read().decode('utf-8', errors='ignore')

                if verbose:
                    print("  Successfully downloaded FF17 from internet")

            except Exception as e:
                if verbose:
                    print(f"  Failed to download FF17 from internet: {e}")
                    print(f"  Falling back to local file: {ff17_local_file}")
                has_internet = False  # Trigger local file fallback

        if not has_internet or content is None:
            # No internet or download failed - use local file
            from pathlib import Path
            local_path = Path(ff17_local_file)

            if not local_path.exists():
                raise FileNotFoundError(
                    f"No internet connection and local file not found: {ff17_local_file}\n"
                    f"Please download the file manually:\n"
                    f"  wget -O data/Siccodes17.zip \"{ff17_url}\"\n"
                    f"  unzip data/Siccodes17.zip -d data/\n"
                    f"Or run from the WRDS login node (which has internet access)."
                )

            if verbose:
                print(f"  Internet not available - loading FF17 from local file: {ff17_local_file}")
            with open(local_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            if verbose:
                print(f"  Successfully loaded FF17 from {ff17_local_file}")
        
        # Parse FF17 industry definitions
        industries = []
        current_ind_num = None
        current_ind_name = None

        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if parts and parts[0].isdigit() and len(parts[0]) <= 2:
                current_ind_num = int(parts[0])
                if len(parts) >= 2:
                    current_ind_name = parts[1]
                continue

            if current_ind_num is not None and '-' in line:
                range_str = line.split()[0] if line.split() else ''
                if '-' in range_str:
                    try:
                        start_str, end_str = range_str.split('-')
                        start_sic = int(start_str)
                        end_sic = int(end_str)
                        industries.append({
                            'ind_num': current_ind_num,
                            'ind_name': current_ind_name,
                            'sic_start': start_sic,
                            'sic_end': end_sic
                        })
                    except (ValueError, IndexError):
                        continue
        
        if verbose:
            print(f"  Parsed {len(industries)} SIC code ranges across 17 industries")
        
        # Build FF17 lookup
        ind_df = pd.DataFrame(industries)
        ff17_mapping = ind_df.groupby('ind_num')['ind_name'].first().to_dict()
        ff17_mapping[17] = "Other"
        
        intervals = pd.IntervalIndex.from_arrays(
            ind_df['sic_start'], 
            ind_df['sic_end'], 
            closed='both'
        )
        interval_to_ind_ff17 = dict(zip(intervals, ind_df['ind_num']))
        
        def match_sic_to_ff17(sic_code):
            if pd.isna(sic_code):
                return 17
            try:
                sic_int = int(sic_code)
                for interval, ind_num in interval_to_ind_ff17.items():
                    if sic_int in interval:
                        return ind_num
                return 17
            except (ValueError, TypeError):
                return 17
        
        if verbose:
            print(f"  Matching {len(fisd_df):,} SIC codes to FF17 industries...")
        
        fisd_df['ff17num'] = fisd_df['sic_code'].apply(match_sic_to_ff17)
        
        if verbose:
            ind_counts = fisd_df['ff17num'].value_counts().sort_index()
            print(f"  FF17 distribution:")
            for ind_num, count in ind_counts.items():
                ind_name = ff17_mapping.get(ind_num, "Unknown")
                pct = 100 * count / len(fisd_df)
                print(f"    Industry {ind_num:2d} ({ind_name:12s}): {count:6,} bonds ({pct:5.2f}%)")
        
        if verbose:
            print(" FF17 industries added successfully")
        
    except Exception as e:
        if verbose:
            print(f" Error adding FF17 industries: {e}")
            print("  Defaulting all bonds to industry 17 (Other)")
        fisd_df['ff17num'] = 17
        ff17_mapping = {17: "Other"}
    
    # ========================================================================
    # Process FF30
    # ========================================================================
    ff30_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes30.zip"
    ff30_local_file = "data/Siccodes30.txt"
    ff30_mapping = {}

    try:
        # Try to get FF30 content - either from internet or local file
        content = None

        if has_internet:
            try:
                if verbose:
                    print(f"  Internet available - downloading FF30 file...")
                response = requests.get(ff30_url, timeout=30)
                response.raise_for_status()

                # Extract text file from zip
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    with z.open('Siccodes30.txt') as f:
                        content = f.read().decode('utf-8', errors='ignore')

                if verbose:
                    print("  Successfully downloaded FF30 from internet")

            except Exception as e:
                if verbose:
                    print(f"  Failed to download FF30 from internet: {e}")
                    print(f"  Falling back to local file: {ff30_local_file}")
                has_internet = False  # Trigger local file fallback

        if not has_internet or content is None:
            # No internet or download failed - use local file
            from pathlib import Path
            local_path = Path(ff30_local_file)

            if not local_path.exists():
                raise FileNotFoundError(
                    f"No internet connection and local file not found: {ff30_local_file}\n"
                    f"Please download the file manually:\n"
                    f"  wget -O data/Siccodes30.zip \"{ff30_url}\"\n"
                    f"  unzip data/Siccodes30.zip -d data/\n"
                    f"Or run from the WRDS login node (which has internet access)."
                )

            if verbose:
                print(f"  Internet not available - loading FF30 from local file: {ff30_local_file}")
            with open(local_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            if verbose:
                print(f"  Successfully loaded FF30 from {ff30_local_file}")

        # Parse FF30 industry definitions
        industries = []
        current_ind_num = None
        current_ind_name = None

        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if parts and parts[0].isdigit() and len(parts[0]) <= 2:
                current_ind_num = int(parts[0])
                if len(parts) >= 2:
                    current_ind_name = parts[1]
                continue

            if current_ind_num is not None and '-' in line:
                range_str = line.split()[0] if line.split() else ''
                if '-' in range_str:
                    try:
                        start_str, end_str = range_str.split('-')
                        start_sic = int(start_str)
                        end_sic = int(end_str)
                        industries.append({
                            'ind_num': current_ind_num,
                            'ind_name': current_ind_name,
                            'sic_start': start_sic,
                            'sic_end': end_sic
                        })
                    except (ValueError, IndexError):
                        continue
        
        if verbose:
            print(f"  Parsed {len(industries)} SIC code ranges across 30 industries")
        
        # Build FF30 lookup
        ind_df = pd.DataFrame(industries)
        ff30_mapping = ind_df.groupby('ind_num')['ind_name'].first().to_dict()
        ff30_mapping[30] = "Other"
        
        intervals = pd.IntervalIndex.from_arrays(
            ind_df['sic_start'], 
            ind_df['sic_end'], 
            closed='both'
        )
        interval_to_ind_ff30 = dict(zip(intervals, ind_df['ind_num']))
        
        def match_sic_to_ff30(sic_code):
            if pd.isna(sic_code):
                return 30
            try:
                sic_int = int(sic_code)
                for interval, ind_num in interval_to_ind_ff30.items():
                    if sic_int in interval:
                        return ind_num
                return 30
            except (ValueError, TypeError):
                return 30
        
        if verbose:
            print(f"  Matching {len(fisd_df):,} SIC codes to FF30 industries...")
        
        fisd_df['ff30num'] = fisd_df['sic_code'].apply(match_sic_to_ff30)
        
        if verbose:
            ind_counts = fisd_df['ff30num'].value_counts().sort_index()
            print(f"  FF30 distribution:")
            for ind_num, count in ind_counts.items():
                ind_name = ff30_mapping.get(ind_num, "Unknown")
                pct = 100 * count / len(fisd_df)
                print(f"    Industry {ind_num:2d} ({ind_name:12s}): {count:6,} bonds ({pct:5.2f}%)")
        
        if verbose:
            print(" FF30 industries added successfully")
        
    except Exception as e:
        if verbose:
            print(f" Error adding FF30 industries: {e}")
            print("  Defaulting all bonds to industry 30 (Other)")
        fisd_df['ff30num'] = 30
        ff30_mapping = {30: "Other"}
    
    return fisd_df, ff17_mapping, ff30_mapping


def create_industry_marketcap_evolution_plot(
    df: pd.DataFrame,
    output_dir: Path,
    ff_column: str,
    industry_mapping: dict,
    filename: str = "stage1_industry_marketcap_evolution",
    params: PlotParams = None,
) -> Path:
    """
    Create stacked area plot showing industry composition of corporate bond market cap over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'trd_exctn_dt', 'pr', 'bond_amt_outstanding', and ff_column
    output_dir : Path
        Directory to save the figure
    ff_column : str
        Name of industry column to use (e.g., 'ff17num' or 'ff30num')
    industry_mapping : dict
        Mapping from industry number to industry name
    filename : str
        Base filename (extension will be added based on params.export_format)
    params : PlotParams
        Plotting parameters
    
    Returns
    -------
    Path to saved figure
    """
    if params is None:
        params = PlotParams()
    
    apply_plot_params(params)

    # Memory optimization: select columns first, copy, then compute market_cap
    df_plot = df[['trd_exctn_dt', 'pr', 'bond_amt_outstanding', ff_column]].copy()
    df_plot['market_cap'] = (df_plot['pr'] * 10 * df_plot['bond_amt_outstanding']) / 1e12

    # Keep only necessary columns for plotting
    df_plot = df_plot[['trd_exctn_dt', 'market_cap', ff_column]]
    
    # Remove rows with missing values
    df_plot = df_plot.dropna(subset=['market_cap', ff_column])
    
    # Aggregate to weekly (W-MON) using SUM
    df_plot['week'] = pd.to_datetime(df_plot['trd_exctn_dt']).dt.to_period('W-MON').dt.to_timestamp()
    df_weekly = df_plot.groupby(['week', ff_column])['market_cap'].sum().reset_index()
    
    # Calculate total market cap by week
    total_by_week = df_weekly.groupby('week')['market_cap'].sum().reset_index()
    total_by_week.rename(columns={'market_cap': 'total_market_cap'}, inplace=True)
    
    # Merge to get percentages
    df_weekly = df_weekly.merge(total_by_week, on='week')
    df_weekly['pct'] = 100 * df_weekly['market_cap'] / df_weekly['total_market_cap']
    
    # Pivot to get each industry as a column
    df_pivot = df_weekly.pivot(index='week', columns=ff_column, values='pct').fillna(0)
    
    # Compute mean % across time for each industry and sort ascending (for stacking)
    industry_means = df_pivot.mean().sort_values(ascending=True)
    ind_nums = industry_means.index.tolist()
    df_pivot = df_pivot[ind_nums]
    
    # Determine number of industries for legend layout
    n_industries = len(ind_nums)
    
    # Create figure - single plot with legend below
    fig_w, fig_h = 8.27, 6.5  # A bit taller to accommodate legend below
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    # Define colors for industries (last industry "Other" is light blue)
    # Extend to support up to 30 industries
    colors_17 = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#add8e6'  # Light blue for "Other"
    ]
    colors_30 = colors_17 + [
        '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5', '#ff9896',
        '#c5b0d5', '#c49c94', '#f7b6d2', '#7f7f7f', '#bcbd22',
        '#17becf', '#aec7e8', '#add8e6'  # Last is light blue for "Other"
    ]
    
    colors = colors_30 if n_industries > 17 else colors_17
    
    # Create stacked area plot
    ax.stackplot(
        df_pivot.index,
        *[df_pivot[ind].values for ind in ind_nums],
        colors=colors[:len(ind_nums)],
        alpha=0.85,
        labels=[industry_mapping.get(ind, f"Industry {ind}") for ind in ind_nums]
    )
    
    # Formatting
    ax.set_xlabel("")
    ax.set_ylabel("Market Capitalization (%)", fontsize=params.label_size)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=params.grid_alpha, linewidth=params.grid_lw, axis='y')
    ax.margins(x=0.01)
    
    # Format date axis
    from matplotlib.dates import AutoDateLocator, DateFormatter
    locator = AutoDateLocator(minticks=5, maxticks=10, interval_multiples=True)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=params.tick_size)
    ax.tick_params(axis="y", labelsize=params.tick_size)
    
    # Create legend below plot
    # FF17: 4 columns (5 rows), FF30: 5 columns (6 rows)
    ncol = 5 if n_industries > 17 else 4
    handles, labels = ax.get_legend_handles_labels()
    
    # Reverse to match stacking order (top to bottom)
    handles = handles[::-1]
    labels = labels[::-1]
    
    # Create legend with small round markers
    legend = ax.legend(
        handles, labels,
        title='Industries ordered by average market cap % (highest to lowest)',
        title_fontsize=8,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=ncol,
        fontsize=8,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
        markerscale=0.7,
    )
    
    # Adjust layout to make room for legend
    plt.subplots_adjust(bottom=0.22, top=0.97, left=0.08, right=0.98)
    
    # Save figure
    ext = params.export_format.lower()
    if ext not in ["pdf", "png", "jpg", "jpeg"]:
        ext = "pdf"
    
    out_path = output_dir / f"{filename}.{ext}"
    
    savefig_kwargs = dict(
        format=ext,
        bbox_inches='tight',  # Use tight to include legend
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

    # Clean up memory
    gc.collect()

    return out_path


def create_industry_dvolume_evolution_plot(
    df: pd.DataFrame,
    output_dir: Path,
    ff_column: str,
    industry_mapping: dict,
    filename: str = "stage1_industry_dvolume_evolution",
    params: PlotParams = None,
) -> Path:
    """
    Create stacked area plot showing industry composition of corporate bond dollar volume over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'trd_exctn_dt', 'dvolume', and ff_column
    output_dir : Path
        Directory to save the figure
    ff_column : str
        Name of industry column to use (e.g., 'ff17num' or 'ff30num')
    industry_mapping : dict
        Mapping from industry number to industry name
    filename : str
        Base filename (extension will be added based on params.export_format)
    params : PlotParams
        Plotting parameters
    
    Returns
    -------
    Path to saved figure
    """
    if params is None:
        params = PlotParams()
    
    apply_plot_params(params)
    
    # Keep only necessary columns
    df_plot = df[['trd_exctn_dt', 'dvolume', ff_column]].copy()
    
    # Remove rows with missing values
    df_plot = df_plot.dropna(subset=['dvolume', ff_column])
    
    # Aggregate to weekly (W-MON) using SUM
    df_plot['week'] = pd.to_datetime(df_plot['trd_exctn_dt']).dt.to_period('W-MON').dt.to_timestamp()
    df_weekly = df_plot.groupby(['week', ff_column])['dvolume'].sum().reset_index()
    
    # Calculate total dvolume by week
    total_by_week = df_weekly.groupby('week')['dvolume'].sum().reset_index()
    total_by_week.rename(columns={'dvolume': 'total_dvolume'}, inplace=True)
    
    # Merge to get percentages
    df_weekly = df_weekly.merge(total_by_week, on='week')
    df_weekly['pct'] = 100 * df_weekly['dvolume'] / df_weekly['total_dvolume']
    
    # Pivot to get each industry as a column
    df_pivot = df_weekly.pivot(index='week', columns=ff_column, values='pct').fillna(0)
    
    # Compute mean % across time for each industry and sort ascending (for stacking)
    industry_means = df_pivot.mean().sort_values(ascending=True)
    ind_nums = industry_means.index.tolist()
    df_pivot = df_pivot[ind_nums]
    
    # Determine number of industries for legend layout
    n_industries = len(ind_nums)
    
    # Create figure - single plot with legend below
    fig_w, fig_h = 8.27, 6.5  # A bit taller to accommodate legend below
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    # Define colors for industries (last industry "Other" is light blue)
    # Extend to support up to 30 industries
    colors_17 = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#add8e6'  # Light blue for "Other"
    ]
    colors_30 = colors_17 + [
        '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5', '#ff9896',
        '#c5b0d5', '#c49c94', '#f7b6d2', '#7f7f7f', '#bcbd22',
        '#17becf', '#aec7e8', '#add8e6'  # Last is light blue for "Other"
    ]
    
    colors = colors_30 if n_industries > 17 else colors_17
    
    # Create stacked area plot
    ax.stackplot(
        df_pivot.index,
        *[df_pivot[ind].values for ind in ind_nums],
        colors=colors[:len(ind_nums)],
        alpha=0.85,
        labels=[industry_mapping.get(ind, f"Industry {ind}") for ind in ind_nums]
    )
    
    # Formatting
    ax.set_xlabel("")
    ax.set_ylabel("Dollar Volume (%)", fontsize=params.label_size)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=params.grid_alpha, linewidth=params.grid_lw, axis='y')
    ax.margins(x=0.01)
    
    # Format date axis
    from matplotlib.dates import AutoDateLocator, DateFormatter
    locator = AutoDateLocator(minticks=5, maxticks=10, interval_multiples=True)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=params.tick_size)
    ax.tick_params(axis="y", labelsize=params.tick_size)
    
    # Create legend below plot
    # FF17: 4 columns (5 rows), FF30: 5 columns (6 rows)
    ncol = 5 if n_industries > 17 else 4
    handles, labels = ax.get_legend_handles_labels()
    
    # Reverse to match stacking order (top to bottom)
    handles = handles[::-1]
    labels = labels[::-1]
    
    # Create legend with small round markers
    legend = ax.legend(
        handles, labels,
        title='Industries ordered by average volume % (highest to lowest)',
        title_fontsize=8,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=ncol,
        fontsize=8,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
        markerscale=0.7,
    )
    
    # Adjust layout to make room for legend
    plt.subplots_adjust(bottom=0.22, top=0.97, left=0.08, right=0.98)
    
    # Save figure
    ext = params.export_format.lower()
    if ext not in ["pdf", "png", "jpg", "jpeg"]:
        ext = "pdf"
    
    out_path = output_dir / f"{filename}.{ext}"
    
    savefig_kwargs = dict(
        format=ext,
        bbox_inches='tight',  # Use tight to include legend
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

    # Clean up memory
    gc.collect()

    return out_path

def create_trade_size_distribution_plot(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "stage1_trade_size_distribution",
    params: PlotParams = None,
) -> Path:
    """
    Create cumulative line plot showing percentage of trades below dollar volume thresholds over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'trd_exctn_dt', 'dvolume', and 'spc_rating'
    output_dir : Path
        Directory to save the figure
    filename : str
        Base filename (extension will be added based on params.export_format)
    params : PlotParams
        Plotting parameters
    
    Returns
    -------
    Path to saved figure
    """
    if params is None:
        params = PlotParams()
    
    apply_plot_params(params)
    
    bucket_edges = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, np.inf]
    bucket_labels = [
        '0k-5k', '5k-10k', '10k-20k', '20k-50k', '50k-100k',
        '100k-200k', '200k-500k', '500k-1M', '1M-2M', '2M-5M',
        '5M-10M', '10M-20M', '20M+'
    ]
    
    threshold_labels = [
        '≤5k', '≤10k', '≤20k', '≤50k', '≤100k',
        '≤200k', '≤500k', '≤1M', '≤2M', '≤5M',
        '≤10M', '≤20M', '≤20M+'
    ]
    
    rating_categories = [
        ('all_bonds', 'A: All Bonds'),
        ('investment_grade', 'B: Investment Grade'),
        ('non_investment_grade', 'C: Non-Investment Grade and Defaulted'),
    ]
    
    panel_data = {}
    
    for rating_filter, _ in rating_categories:
        # Memory optimization: select columns first, filter, then copy
        df_temp = df[['trd_exctn_dt', 'dvolume', 'spc_rating']]

        if rating_filter == 'investment_grade':
            df_temp = df_temp[(df_temp['spc_rating'] >= 1) & (df_temp['spc_rating'] <= 10)]
        elif rating_filter == 'non_investment_grade':
            df_temp = df_temp[(df_temp['spc_rating'] >= 11) & (df_temp['spc_rating'] <= 22)]

        df_filt = df_temp[['trd_exctn_dt', 'dvolume']].copy()
        df_filt['year_month'] = df_filt['trd_exctn_dt'].dt.to_period('M').dt.to_timestamp()
        df_filt['bucket'] = pd.cut(df_filt['dvolume'], bins=bucket_edges, labels=bucket_labels, right=False)
        
        trade_counts = df_filt.groupby(['year_month', 'bucket']).size().reset_index(name='count')
        total_by_month = df_filt.groupby('year_month').size().reset_index(name='total')
        
        trade_counts = trade_counts.merge(total_by_month, on='year_month')
        trade_counts['pct'] = 100 * trade_counts['count'] / trade_counts['total']
        
        df_pivot = trade_counts.pivot(index='year_month', columns='bucket', values='pct').fillna(0)
        df_pivot = df_pivot[bucket_labels]
        
        df_cumulative = df_pivot.cumsum(axis=1)
        df_cumulative.columns = threshold_labels

        panel_data[rating_filter] = df_cumulative

        # Clean up memory after processing each rating category
        gc.collect()

    fig_w, fig_h = 8.27, 11.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    
    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        left=0.10, right=0.95, bottom=0.12, top=0.97,
        hspace=0.18
    )
    
    axes = gs.subplots().ravel()
    
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a'
    ]
    
    for ax, (rating_filter, panel_label) in zip(axes, rating_categories):
        df_plot = panel_data[rating_filter]
        
        for i, threshold in enumerate(threshold_labels):
            ax.plot(
                df_plot.index,
                df_plot[threshold],
                color=colors[i],
                linewidth=params.line_lw,
                alpha=params.line_alpha,
                label=threshold
            )
        
        ax.set_title(panel_label, pad=2)
        ax.set_xlabel("")
        ax.set_ylabel("Cumulative Day Frequency (%)", fontsize=params.label_size)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=params.grid_alpha, linewidth=params.grid_lw)
        ax.margins(x=0.01)
        
        from matplotlib.dates import AutoDateLocator, DateFormatter
        locator = AutoDateLocator(minticks=5, maxticks=10, interval_multiples=True)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
        ax.tick_params(axis="x", labelsize=params.tick_size)
        ax.tick_params(axis="y", labelsize=params.tick_size)
    
    handles, labels = axes[0].get_legend_handles_labels()
    
    fig.legend(
        handles, labels,
        title='Cumulative % of days with total daily volume below threshold',
        title_fontsize=8,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.06),
        ncol=7,
        fontsize=8,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
        markerscale=0.7,
    )
    
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

    # Clean up memory
    gc.collect()

    return out_path

def create_bond_characteristics_evolution_plot(
    df: pd.DataFrame,
    fisd: pd.DataFrame,
    output_dir: Path,
    filename: str = "stage1_bond_characteristics_evolution",
    params: PlotParams = None,
) -> tuple:
    """
    Create 3-panel stacked area plot showing evolution of bond characteristics over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        Trade-level data with 'cusip_id', 'trd_exctn_dt', 'pr'
    fisd : pd.DataFrame
        FISD data with 'cusip_id', 'bond_type', 'country_domicile', 'rule_144a'
    output_dir : Path
        Directory to save the figure
    filename : str
        Base filename (extension will be added based on params.export_format)
    params : PlotParams
        Plotting parameters
    
    Returns
    -------
    tuple: (Path to saved figure, DataFrame with weekly data for export)
    """
    if params is None:
        params = PlotParams()
    
    apply_plot_params(params)
    
    # ========================================================================
    # Step 1: Create lookup dict from fisd (RAM efficient)
    # ========================================================================
    needed_cols = ['cusip_id', 'bond_type', 'country_domicile', 'rule_144a']
    fisd_subset = fisd[needed_cols].copy()
    
    # Drop NaN values to exclude from analysis
    fisd_subset = fisd_subset.dropna(subset=['bond_type', 'country_domicile', 'rule_144a'])
    
    # Create lookup dict: cusip_id -> (bond_type, country_domicile, rule_144a)
    lookup_dict = {
        row['cusip_id']: (row['bond_type'], row['country_domicile'], row['rule_144a'])
        for _, row in fisd_subset.iterrows()
    }
    
    # ========================================================================
    # Step 2: Select only necessary columns from final_df and map characteristics
    # ========================================================================
    df_trades = df[['cusip_id', 'trd_exctn_dt', 'pr']].copy()
    
    # Map characteristics using lookup dict
    df_trades['bond_type'] = df_trades['cusip_id'].map(lambda x: lookup_dict.get(x, (None, None, None))[0])
    df_trades['country_domicile'] = df_trades['cusip_id'].map(lambda x: lookup_dict.get(x, (None, None, None))[1])
    df_trades['rule_144a'] = df_trades['cusip_id'].map(lambda x: lookup_dict.get(x, (None, None, None))[2])
    
    # Drop trades with missing characteristics
    df_trades = df_trades.dropna(subset=['bond_type', 'country_domicile', 'rule_144a'])
    
    # Add week column
    df_trades['week'] = pd.to_datetime(df_trades['trd_exctn_dt']).dt.to_period('W-MON').dt.to_timestamp()
    
    # ========================================================================
    # Step 3: Process Panel A - Bond Type (Top 5 + Other)
    # ========================================================================
    # Count trades per week per bond_type
    df_bondtype = df_trades.groupby(['week', 'bond_type']).size().reset_index(name='count')
    
    # Calculate total trades by week
    total_by_week = df_bondtype.groupby('week')['count'].sum().reset_index()
    total_by_week.rename(columns={'count': 'total'}, inplace=True)
    
    # Merge to get percentages
    df_bondtype = df_bondtype.merge(total_by_week, on='week')
    df_bondtype['pct'] = 100 * df_bondtype['count'] / df_bondtype['total']
    
    # Pivot to get each bond_type as column
    df_bondtype_pivot = df_bondtype.pivot(index='week', columns='bond_type', values='pct').fillna(0)
    
    # Compute mean % across time for each bond_type
    bondtype_means = df_bondtype_pivot.mean().sort_values(ascending=False)
    
    # Select top 5, aggregate rest into "Other"
    top5_bondtypes = bondtype_means.head(5).index.tolist()
    other_bondtypes = bondtype_means.iloc[5:].index.tolist()
    
    if other_bondtypes:
        df_bondtype_pivot['Other'] = df_bondtype_pivot[other_bondtypes].sum(axis=1)
        df_bondtype_pivot = df_bondtype_pivot[top5_bondtypes + ['Other']]
    else:
        df_bondtype_pivot = df_bondtype_pivot[top5_bondtypes]
    
    # Re-sort by mean (descending for stacking - highest % at bottom)
    bondtype_means_final = df_bondtype_pivot.mean().sort_values(ascending=False)
    df_bondtype_pivot = df_bondtype_pivot[bondtype_means_final.index.tolist()]
    
    # ========================================================================
    # Step 4: Process Panel B - Country Domicile (Top 5 + Other)
    # ========================================================================
    # Count trades per week per country
    df_country = df_trades.groupby(['week', 'country_domicile']).size().reset_index(name='count')
    
    # Calculate total trades by week
    total_by_week_country = df_country.groupby('week')['count'].sum().reset_index()
    total_by_week_country.rename(columns={'count': 'total'}, inplace=True)
    
    # Merge to get percentages
    df_country = df_country.merge(total_by_week_country, on='week')
    df_country['pct'] = 100 * df_country['count'] / df_country['total']
    
    # Pivot to get each country as column
    df_country_pivot = df_country.pivot(index='week', columns='country_domicile', values='pct').fillna(0)
    
    # Compute mean % across time for each country
    country_means = df_country_pivot.mean().sort_values(ascending=False)
    
    # Select top 5, aggregate rest into "Other"
    top5_countries = country_means.head(5).index.tolist()
    other_countries = country_means.iloc[5:].index.tolist()
    
    if other_countries:
        df_country_pivot['Other'] = df_country_pivot[other_countries].sum(axis=1)
        df_country_pivot = df_country_pivot[top5_countries + ['Other']]
    else:
        df_country_pivot = df_country_pivot[top5_countries]
    
    # Re-sort by mean (descending for stacking - highest % at bottom)
    country_means_final = df_country_pivot.mean().sort_values(ascending=False)
    df_country_pivot = df_country_pivot[country_means_final.index.tolist()]
    
    # ========================================================================
    # Step 5: Process Panel C - Rule 144A (Binary Y/N)
    # ========================================================================
    # Count trades per week per 144a status
    df_144a = df_trades.groupby(['week', 'rule_144a']).size().reset_index(name='count')
    
    # Calculate total trades by week
    total_by_week_144a = df_144a.groupby('week')['count'].sum().reset_index()
    total_by_week_144a.rename(columns={'count': 'total'}, inplace=True)
    
    # Merge to get percentages
    df_144a = df_144a.merge(total_by_week_144a, on='week')
    df_144a['pct'] = 100 * df_144a['count'] / df_144a['total']
    
    # Pivot to get Y and N as columns
    df_144a_pivot = df_144a.pivot(index='week', columns='rule_144a', values='pct').fillna(0)
    
    # Ensure both Y and N exist
    if 'Y' not in df_144a_pivot.columns:
        df_144a_pivot['Y'] = 0
    if 'N' not in df_144a_pivot.columns:
        df_144a_pivot['N'] = 0
    
    # Sort by mean (descending) - highest at bottom
    rule144a_means = df_144a_pivot.mean().sort_values(ascending=False)
    df_144a_pivot = df_144a_pivot[rule144a_means.index.tolist()]
    
    # ========================================================================
    # Step 6: Create 3x1 stacked area plot
    # ========================================================================
    fig_w, fig_h = 8.27, 11.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    
    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        left=0.10, right=0.95, bottom=0.10, top=0.97,
        hspace=0.30
    )
    
    axes = gs.subplots().ravel()
    
    # Define colors (6 categories max per panel: top 5 + Other)
    colors_6 = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#add8e6'  # Light blue for "Other"
    ]
    
    colors_2 = ['#ff7f0e', '#1f77b4']  # Orange for N, Blue for Y
    
    # ========================================================================
    # Panel A: Bond Type
    # ========================================================================
    bondtype_cols = df_bondtype_pivot.columns.tolist()
    
    axes[0].stackplot(
        df_bondtype_pivot.index,
        *[df_bondtype_pivot[col].values for col in bondtype_cols],
        colors=colors_6[:len(bondtype_cols)],
        alpha=0.85,
        labels=bondtype_cols
    )
    
    axes[0].set_ylabel("Trade Share (%)", fontsize=params.label_size)
    axes[0].set_ylim(0, 100)
    axes[0].set_title("A: Bond Type", pad=2)
    axes[0].grid(True, alpha=params.grid_alpha, linewidth=params.grid_lw, axis='y')
    axes[0].margins(x=0.01)
    axes[0].tick_params(axis="both", labelsize=params.tick_size)
    
    # Legend for Panel A (reversed to show highest at top)
    handles_a, labels_a = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles_a, labels_a,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=7,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    
    # ========================================================================
    # Panel B: Country Domicile
    # ========================================================================
    country_cols = df_country_pivot.columns.tolist()
    
    axes[1].stackplot(
        df_country_pivot.index,
        *[df_country_pivot[col].values for col in country_cols],
        colors=colors_6[:len(country_cols)],
        alpha=0.85,
        labels=country_cols
    )
    
    axes[1].set_ylabel("Trade Share (%)", fontsize=params.label_size)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("B: Country Domicile", pad=2)
    axes[1].grid(True, alpha=params.grid_alpha, linewidth=params.grid_lw, axis='y')
    axes[1].margins(x=0.01)
    axes[1].tick_params(axis="both", labelsize=params.tick_size)
    
    # Legend for Panel B (reversed to show highest at top)
    handles_b, labels_b = axes[1].get_legend_handles_labels()
    axes[1].legend(
        handles_b, labels_b,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=7,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    
    # ========================================================================
    # Panel C: Rule 144A
    # ========================================================================
    rule144a_cols = df_144a_pivot.columns.tolist()
    
    axes[2].stackplot(
        df_144a_pivot.index,
        *[df_144a_pivot[col].values for col in rule144a_cols],
        colors=colors_2[:len(rule144a_cols)],
        alpha=0.85,
        labels=['Non-144A', '144A']  # Cleaner labels
    )
    
    axes[2].set_ylabel("Trade Share (%)", fontsize=params.label_size)
    axes[2].set_xlabel("", fontsize=params.label_size)
    axes[2].set_ylim(0, 100)
    axes[2].set_title("C: Rule 144A", pad=2)
    axes[2].grid(True, alpha=params.grid_alpha, linewidth=params.grid_lw, axis='y')
    axes[2].margins(x=0.01)
    axes[2].tick_params(axis="both", labelsize=params.tick_size)
    
    # Legend for Panel C (reversed)
    handles_c, labels_c = axes[2].get_legend_handles_labels()
    axes[2].legend(
        handles_c, labels_c,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=7,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    
    # Format date axis for all panels
    from matplotlib.dates import AutoDateLocator, DateFormatter
    locator = AutoDateLocator(minticks=5, maxticks=10, interval_multiples=True)
    formatter = DateFormatter("%Y")
    
    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    
    # ========================================================================
    # Save figure
    # ========================================================================
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
    
    # ========================================================================
    # Prepare data for CSV export
    # ========================================================================
    # Combine all three panels into one DataFrame for export
    df_export = pd.DataFrame()
    df_export['week'] = df_bondtype_pivot.index
    
    # Add bond type columns with prefix
    for col in df_bondtype_pivot.columns:
        df_export[f'bondtype_{col}'] = df_bondtype_pivot[col].values
    
    # Add country columns with prefix
    for col in df_country_pivot.columns:
        df_export[f'country_{col}'] = df_country_pivot[col].values
    
    # Add 144A columns with prefix
    for col in df_144a_pivot.columns:
        df_export[f'rule144a_{col}'] = df_144a_pivot[col].values

    # Clean up memory
    gc.collect()

    return out_path, df_export


def create_rating_maturity_evolution_plot(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "stage1_rating_maturity_evolution",
    params: PlotParams = None,
) -> tuple:
    """
    Create 2-panel stacked area plot showing evolution of rating and maturity categories over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        Trade-level data with 'trd_exctn_dt', 'spc_rating', 'bond_maturity'
    output_dir : Path
        Directory to save the figure
    filename : str
        Base filename (extension will be added based on params.export_format)
    params : PlotParams
        Plotting parameters
    
    Returns
    -------
    tuple: (Path to saved figure, DataFrame with weekly data for export)
    """
    if params is None:
        params = PlotParams()
    
    apply_plot_params(params)
    
    # ========================================================================
    # Step 1: Select necessary columns and create categories
    # ========================================================================
    # Memory optimization: only select columns that are actually used
    # (pr is not needed since we only count rows with .size())
    df_trades = df[['trd_exctn_dt', 'spc_rating', 'bond_maturity']].copy()
    
    # Drop trades with missing spc_rating or bond_maturity
    df_trades = df_trades.dropna(subset=['spc_rating', 'bond_maturity'])
    
    # Add week column
    df_trades['week'] = pd.to_datetime(df_trades['trd_exctn_dt']).dt.to_period('W-MON').dt.to_timestamp()
    
    # ========================================================================
    # Step 2: Create Rating Category (Panel A)
    # ========================================================================
    def rating_to_category(rating):
        """Map numeric S&P rating to NAIC category."""
        if 1 <= rating <= 7:
            return 'AAA+ to A-'
        elif 8 <= rating <= 10:
            return 'BBB+ to BBB-'
        elif 11 <= rating <= 13:
            return 'BB+ to BB-'
        elif 14 <= rating <= 16:
            return 'B+ to B-'
        elif 17 <= rating <= 21:
            return 'CCC+ to C'
        elif rating == 22:
            return 'D'
        else:
            return None
    
    df_trades['rating_category'] = df_trades['spc_rating'].apply(rating_to_category)
    df_trades = df_trades.dropna(subset=['rating_category'])
    
    # Count trades per week per rating_category
    df_rating = df_trades.groupby(['week', 'rating_category']).size().reset_index(name='count')
    
    # Calculate total trades by week
    total_by_week_rating = df_rating.groupby('week')['count'].sum().reset_index()
    total_by_week_rating.rename(columns={'count': 'total'}, inplace=True)
    
    # Merge to get percentages
    df_rating = df_rating.merge(total_by_week_rating, on='week')
    df_rating['pct'] = 100 * df_rating['count'] / df_rating['total']
    
    # Pivot to get each rating_category as column
    df_rating_pivot = df_rating.pivot(index='week', columns='rating_category', values='pct').fillna(0)
    
    # Compute mean % across time and sort descending
    rating_means = df_rating_pivot.mean().sort_values(ascending=False)
    df_rating_pivot = df_rating_pivot[rating_means.index.tolist()]
    
    # ========================================================================
    # Step 3: Create Maturity Category (Panel B)
    # ========================================================================
    def maturity_to_category(maturity):
        """Map bond maturity to category."""
        if maturity < 3:
            return '1 to 3 Year'
        elif 3 <= maturity < 5:
            return '3 to 5 Year'
        elif 5 <= maturity < 10:
            return '5 to 10 Year'
        elif maturity >= 10:
            return '10 Year Plus'
        else:
            return None
    
    df_trades['maturity_category'] = df_trades['bond_maturity'].apply(maturity_to_category)
    df_trades = df_trades.dropna(subset=['maturity_category'])
    
    # Count trades per week per maturity_category
    df_maturity = df_trades.groupby(['week', 'maturity_category']).size().reset_index(name='count')
    
    # Calculate total trades by week
    total_by_week_maturity = df_maturity.groupby('week')['count'].sum().reset_index()
    total_by_week_maturity.rename(columns={'count': 'total'}, inplace=True)
    
    # Merge to get percentages
    df_maturity = df_maturity.merge(total_by_week_maturity, on='week')
    df_maturity['pct'] = 100 * df_maturity['count'] / df_maturity['total']
    
    # Pivot to get each maturity_category as column
    df_maturity_pivot = df_maturity.pivot(index='week', columns='maturity_category', values='pct').fillna(0)
    
    # Compute mean % across time and sort descending
    maturity_means = df_maturity_pivot.mean().sort_values(ascending=False)
    df_maturity_pivot = df_maturity_pivot[maturity_means.index.tolist()]
    
    # ========================================================================
    # Step 4: Create 2x1 stacked area plot
    # ========================================================================
    fig_w, fig_h = 8.27, 8.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    
    gs = fig.add_gridspec(
        nrows=2, ncols=1,
        left=0.10, right=0.95, bottom=0.10, top=0.97,
        hspace=0.30
    )
    
    axes = gs.subplots().ravel()
    
    # Define colors (6 categories max for rating, 4 for maturity)
    colors_6 = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b'
    ]
    
    colors_4 = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'
    ]
    
    # ========================================================================
    # Panel A: Rating Category
    # ========================================================================
    rating_cols = df_rating_pivot.columns.tolist()
    
    axes[0].stackplot(
        df_rating_pivot.index,
        *[df_rating_pivot[col].values for col in rating_cols],
        colors=colors_6[:len(rating_cols)],
        alpha=0.85,
        labels=rating_cols
    )
    
    axes[0].set_ylabel("Trade Share (%)", fontsize=params.label_size)
    axes[0].set_ylim(0, 100)
    axes[0].set_title("A: Rating Category", pad=2)
    axes[0].grid(True, alpha=params.grid_alpha, linewidth=params.grid_lw, axis='y')
    axes[0].margins(x=0.01)
    axes[0].tick_params(axis="both", labelsize=params.tick_size)
    
    # Legend for Panel A (no reversal - highest first)
    handles_a, labels_a = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles_a, labels_a,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=7,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    
    # ========================================================================
    # Panel B: Maturity Category
    # ========================================================================
    maturity_cols = df_maturity_pivot.columns.tolist()
    
    axes[1].stackplot(
        df_maturity_pivot.index,
        *[df_maturity_pivot[col].values for col in maturity_cols],
        colors=colors_4[:len(maturity_cols)],
        alpha=0.85,
        labels=maturity_cols
    )
    
    axes[1].set_ylabel("Trade Share (%)", fontsize=params.label_size)
    axes[1].set_xlabel("", fontsize=params.label_size)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("B: Maturity Category", pad=2)
    axes[1].grid(True, alpha=params.grid_alpha, linewidth=params.grid_lw, axis='y')
    axes[1].margins(x=0.01)
    axes[1].tick_params(axis="both", labelsize=params.tick_size)
    
    # Legend for Panel B (no reversal - highest first)
    handles_b, labels_b = axes[1].get_legend_handles_labels()
    axes[1].legend(
        handles_b, labels_b,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=7,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    
    # Format date axis for all panels
    from matplotlib.dates import AutoDateLocator, DateFormatter
    locator = AutoDateLocator(minticks=5, maxticks=10, interval_multiples=True)
    formatter = DateFormatter("%Y")
    
    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    
    # ========================================================================
    # Save figure
    # ========================================================================
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
    
    # ========================================================================
    # Prepare data for CSV export
    # ========================================================================
    # Combine both panels into one DataFrame for export
    df_export = pd.DataFrame()
    df_export['week'] = df_rating_pivot.index
    
    # Add rating columns with prefix
    for col in df_rating_pivot.columns:
        df_export[f'rating_{col}'] = df_rating_pivot[col].values
    
    # Add maturity columns with prefix
    for col in df_maturity_pivot.columns:
        df_export[f'maturity_{col}'] = df_maturity_pivot[col].values

    # Clean up memory
    gc.collect()

    return out_path, df_export

# ============================================================================
# Memory Tracking Utilities
# ============================================================================

def log_memory_usage(location_name):
    """
    Get current memory usage in GB (cross-platform).
    
    Parameters
    ----------
    location_name : str
        Descriptive name for this measurement point
        
    Returns
    -------
    dict or None
        Dictionary with 'gb' and 'location' keys if successful, None otherwise
    """
    try:
        import psutil
        process = psutil.Process()
        mem_gb = process.memory_info().rss / (1024**3)
        return {'gb': mem_gb, 'location': location_name}
    except ImportError:
        return None  # psutil not available
    except Exception:
        return None  # Other error


def log_memory_delta(start_mem, end_mem, func_name):
    """
    Log memory change between two measurement points (print + log).
    
    Parameters
    ----------
    start_mem : dict or None
        Memory measurement from log_memory_usage() at function start
    end_mem : dict or None
        Memory measurement from log_memory_usage() at function end
    func_name : str
        Name of the function being tracked
    """
    if start_mem is None or end_mem is None:
        msg = f"[MEMORY] {func_name}: Tracking unavailable (install psutil)"
        print(msg)
        logging.warning(msg)
        return
    
    start_gb = start_mem['gb']
    end_gb = end_mem['gb']
    delta_gb = end_gb - start_gb
    
    msg = f"[MEMORY] {func_name}: START {start_gb:.2f}GB | END {end_gb:.2f}GB | DELTA {delta_gb:+.2f}GB"
    print(msg)
    logging.info(msg)


def optimize_dtypes(df, categorical_cols=None, float32=True, downcast_ints=True):
    """
    Optimize DataFrame memory usage by converting to smaller dtypes.

    This function can significantly reduce memory footprint by:
    - Converting float64 to float32 (50% reduction for numeric columns)
    - Downcasting integers to smallest sufficient type
    - Converting low-cardinality string columns to category type

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to optimize
    categorical_cols : list of str, optional
        Column names to convert to category dtype. If None, will auto-detect
        columns with < 50% unique values.
    float32 : bool, default=True
        Convert float64 columns to float32
    downcast_ints : bool, default=True
        Downcast integer columns to smallest sufficient type

    Returns
    -------
    pd.DataFrame
        Optimized DataFrame (modified in place, but also returned)

    Examples
    --------
    >>> df = optimize_dtypes(df, categorical_cols=['cusip_id', 'rating_type'])
    >>> df = optimize_dtypes(df, float32=True, downcast_ints=True)
    """
    mem_before = df.memory_usage(deep=True).sum() / 1024**3

    # Convert float64 to float32
    if float32:
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            df[col] = df[col].astype('float32')

    # Downcast integers
    if downcast_ints:
        int_cols = df.select_dtypes(include=['int64', 'int32']).columns
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], downcast='integer')

    # Convert to categorical
    if categorical_cols is None:
        # Auto-detect: columns with < 50% unique values
        categorical_cols = []
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:
                categorical_cols.append(col)

    for col in categorical_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    mem_after = df.memory_usage(deep=True).sum() / 1024**3
    reduction_pct = (1 - mem_after / mem_before) * 100

    logging.info(f"[DTYPE OPTIMIZATION] {mem_before:.2f}GB -> {mem_after:.2f}GB "
                 f"({reduction_pct:.1f}% reduction)")

    return df

