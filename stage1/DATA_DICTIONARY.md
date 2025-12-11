# Stage 1 Data Dictionary

Comprehensive documentation for the Stage 1 output dataset. Available in zipped parquet format on [Open Bond Asset Pricing](https://openbondassetpricing.com/data). All proprietary data (GVKEY, ratings etc.) set to NaN.

---

## Overview

Stage 1 produces a single enriched daily bond dataset that combines:
- Clean TRACE prices from Stage 0
- Bond analytics computed via QuantLib (duration, convexity, YTM, credit spreads)
- Bond characteristics from FISD (coupon, maturity, issuer, amount outstanding)
- Credit ratings from S&P and Moody's
- Equity identifiers 
- Fama-French industry classifications

---

## File Information

| Property | Value |
|----------|-------|
| **Location** | `stage1/data/stage1_YYYYMMDD.parquet` |
| **Format** | Apache Parquet (columnar, compressed) |
| **Structure** | Panel data: one row per (cusip_id, trd_exctn_dt) |
| **Size** | ~500MB - 2GB (depending on time period) |
| **Rows** | ~30 million (full sample 2002-present) |
| **Columns** | 44 |
| **Download** | Available in zipped parquet format on [Open Bond Asset Pricing](https://openbondassetpricing.com/data) |

---

## Price Convention

**All prices are in percentage of par.**

| Price Value | Meaning | Dollar Value (for $1,000 par) |
|-------------|---------|-------------------------------|
| 100 | Par | $1,000 |
| 99 | 99% of par | $990 |
| 105.5 | 105.5% of par | $1,055 |
| 85.25 | 85.25% of par | $852.50 |

All bonds in the dataset have a principal amount of $1,000.

---

## Notes

- **\* Columns marked with asterisk** are not included in the output file but can be obtained by merging with FISD data in `stage0/enhanced/trace_enhanced_fisd_YYYYMMDD.parquet`
- **† Columns marked with dagger** are excluded from the public download due to proprietary data restrictions
- **‡ Columns marked with double dagger** are excluded from the public download to reduce file size

---

## Variable Reference

### Identifiers

| Column | Type | Description |
|--------|------|-------------|
| `cusip_id` | category | 9-character CUSIP identifier (unique bond ID) |
| `issuer_cusip`* | category | 6-character issuer CUSIP (identifies the issuing company) |
| `permno` | Int32 | CRSP PERMNO equity identifier (links to stock data) |
| `permco` | Int32 | CRSP PERMCO company identifier |
| `gvkey`† | Int32 | Compustat GVKEY identifier (links to accounting data) |
| `trd_exctn_dt` | datetime | Trade execution date |

---

### Computed Bond Analytics (QuantLib)

| Column | Type | Description |
|--------|------|-------------|
| `pr` | float32 | Volume-weighted clean price (% of par) |
| `prfull` | float32 | Dirty price = pr + acclast (% of par) |
| `acclast` | float32 | Accrued interest — pure time-accrued interest component |
| `accpmt` | float32 | Accumulated coupon payments since issue |
| `accall` | float32 | Accumulated payments — includes cash flows + accrued interest; used for return calculations |
| `ytm` | float64 | Yield to maturity (annualized, decimal) |
| `mod_dur` | float32 | Modified duration (years) |
| `mac_dur` | float32 | Macaulay duration (years) |
| `convexity` | float32 | Bond convexity |
| `bond_maturity` | float32 | Time to maturity (years) |
| `credit_spread` | float64 | Credit spread over duration-matched Treasury yield |

#### Price Definitions

| Term | Formula | Use Case |
|------|---------|----------|
| **Clean Price** | Quoted price without accrued interest | Trading, quoting |
| **Dirty Price** | `pr + acclast` | Actual settlement price, market cap |

#### Accrued Interest Variables

| Variable | Description | Use Case |
|----------|-------------|----------|
| `acclast` | Interest accrued since last coupon payment | Dirty price, market cap |
| `accpmt` | Cumulative coupon payments since bond issuance | Tracking total cash flows |
| `accall` | Accumulated payments including cash flows and accrued interest | **Return calculations** |

---

### TRACE Pricing (from Stage 0)

| Column | Type | Description |
|--------|------|-------------|
| `prc_ew` | float32 | Equal-weighted average price |
| `prc_vw_par` | float32 | Par volume-weighted average price |
| `prc_first` | float32 | First trade price of the day |
| `prc_last` | float32 | Last trade price of the day |
| `prc_hi` | float32 | Highest price of the day |
| `prc_lo` | float32 | Lowest price of the day |
| `trade_count` | Int16 | Number of trades |
| `time_ew`‡ | float32 | Average trade time (seconds after midnight) |
| `time_last`‡ | Int32 | Last trade time (seconds after midnight) |
| `qvolume` | float32 | Par volume (millions USD) |
| `dvolume` | float32 | Dollar volume (millions USD) |

---

### Dealer Bid/Ask Metrics

| Column | Type | Description |
|--------|------|-------------|
| `prc_bid` | float32 | Dealer bid price, value-weighted (% of par) |
| `bid_last` | float32 | Last dealer bid price of day (% of par) |
| `bid_time_ew`‡ | float32 | Average dealer bid time (seconds after midnight) |
| `bid_time_last`‡ | Int32 | Last dealer bid time (seconds after midnight) |
| `prc_ask` | float32 | Dealer ask price, value-weighted (% of par) |
| `bid_count`‡ | Int16 | Number of dealer buys |
| `ask_count`‡ | Int16 | Number of dealer sells |

---

### Database Source

| Column | Type | Description |
|--------|------|-------------|
| `db_type` | Int8 | Source TRACE database: 1=Enhanced, 2=Standard, 3=144A |

---

### Bond Characteristics (from FISD)

| Column | Type | Description |
|--------|------|-------------|
| `coupon`* | float32 | Annual coupon rate (%) |
| `principal_amt`* | Int16 | Principal amount per bond (typically $1,000) |
| `bond_age` | float32 | Bond age since issuance (years) |
| `bond_amt_outstanding` | Int64 | Number of bond units outstanding |
| `callable`* | Int8 | Callable flag: 1=callable, 0=not callable |

---

### Industry Classifications

| Column | Type | Description |
|--------|------|-------------|
| `ff17num` | int8 | Fama-French 17 industry classification |
| `ff30num` | int8 | Fama-French 30 industry classification |

---

### Credit Ratings

| Column | Type | Description |
|--------|------|-------------|
| `sp_rating`† | Int8 | S&P credit rating (1-22, where 22=default) |
| `sp_naic`* | Int8 | S&P NAIC category (1-6) |
| `mdy_rating`† | Int8 | Moody's credit rating (1-21, where 21=default) |
| `spc_rating`† | Int8 | S&P composite rating (1-22) |
| `mdc_rating`† | Int8 | Moody's composite rating (1-22) |
| `comp_rating`* | float64 | Average of spc_rating and mdc_rating |

#### S&P Rating Scale (sp_rating, spc_rating)

| Code | Rating | Category |
|------|--------|----------|
| 1 | AAA | Investment Grade |
| 2 | AA+ | Investment Grade |
| 3 | AA | Investment Grade |
| 4 | AA- | Investment Grade |
| 5 | A+ | Investment Grade |
| 6 | A | Investment Grade |
| 7 | A- | Investment Grade |
| 8 | BBB+ | Investment Grade |
| 9 | BBB | Investment Grade |
| 10 | BBB- | Investment Grade |
| 11 | BB+ | High Yield |
| 12 | BB | High Yield |
| 13 | BB- | High Yield |
| 14 | B+ | High Yield |
| 15 | B | High Yield |
| 16 | B- | High Yield |
| 17 | CCC+ | High Yield |
| 18 | CCC | High Yield |
| 19 | CCC- | High Yield |
| 20 | CC | High Yield |
| 21 | C | High Yield |
| 22 | D | Default |

#### Moody's Rating Scale (mdy_rating)

| Code | Rating | Category |
|------|--------|----------|
| 1 | Aaa | Investment Grade |
| 2 | Aa1 | Investment Grade |
| 3 | Aa2 | Investment Grade |
| 4 | Aa3 | Investment Grade |
| 5 | A1 | Investment Grade |
| 6 | A2 | Investment Grade |
| 7 | A3 | Investment Grade |
| 8 | Baa1 | Investment Grade |
| 9 | Baa2 | Investment Grade |
| 10 | Baa3 | Investment Grade |
| 11 | Ba1 | High Yield |
| 12 | Ba2 | High Yield |
| 13 | Ba3 | High Yield |
| 14 | B1 | High Yield |
| 15 | B2 | High Yield |
| 16 | B3 | High Yield |
| 17 | Caa1 | High Yield |
| 18 | Caa2 | High Yield |
| 19 | Caa3 | High Yield |
| 20 | Ca | High Yield |
| 21 | C/D | Default |

#### NAIC Categories (sp_naic)

| Code | Category | S&P Ratings |
|------|----------|-------------|
| 1 | Highest Quality | AAA, AA+, AA, AA- |
| 2 | High Quality | A+, A, A- |
| 3 | Medium Quality | BBB+, BBB, BBB- |
| 4 | Low Quality | BB+, BB, BB- |
| 5 | Lowest Quality | B+, B, B-, CCC+, CCC, CCC- |
| 6 | In or Near Default | CC, C, D |

#### Composite Ratings

| Variable | Description |
|----------|-------------|
| `spc_rating` | S&P rating; if missing, filled with `mdy_rating` (Moody's 21 → 22 for default alignment) |
| `mdc_rating` | Moody's rating; if missing, filled with `sp_rating` (S&P 22 → 21 for default alignment) |
| `comp_rating` | Average of `spc_rating` and `mdc_rating` |

---

## Related Documentation

- [Stage 1 README](README_stage1.md) — Full Stage 1 documentation
- [Stage 1 QUICKSTART](QUICKSTART_stage1.md) — Quick start guide
- [Main README](../README.md) — Project overview
- [Ultra-distressed Filter](README_distressed_filter.md) — Filter documentation

---

**Last Updated:** December 2025
