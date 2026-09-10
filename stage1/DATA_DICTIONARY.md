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

❗**Before using this data, read
[Sample-defining operations](#sample-defining-operations).** Six row filters and a
per-date winsorization are applied before the file is written. None of them is visible
in the schema, and the largest drops ~9% of bond-days.

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
| `permno` | Int32 | CRSP PERMNO equity identifier (links to stock data) |
| `permco` | Int32 | CRSP PERMCO company identifier |
| `gvkey` | Int32 | Compustat GVKEY identifier (links to accounting data) |
| `trd_exctn_dt` | datetime | Trade execution date |

#### How bonds are linked to firms

The equity identifiers come from the **bond-firm linker** published at
[openbondassetpricing.com](https://openbondassetpricing.com/), downloaded by
`run_pipeline.sh`. The mapping is **bond-level and dated**: one row per
(9-character CUSIP, ownership window `[w0, w1]`), so a bond that changes hands --
through an acquisition, a spin-off or a rename -- points at the right firm in each
period rather than at whichever firm owned it last.

A consequence worth stating plainly: **`permno` is NULL for about 12% of bond-days**,
and that is deliberate. It is NULL when the bond is outside every window we can
support -- most often when the bond still trades after the firm's equity stopped
being listed. A missing identifier is an answer; a stale one is a silent error. If
you need to know *why* a particular bond has no link, the published release ships
`fl_verdicts.parquet`, which records every refusal and its reason, and
`firm_names.parquet`, which maps `permno` to a dated firm name.

**`gvkey` is stored numerically here** for continuity with earlier releases, but
Compustat's GVKEY is a 6-character zero-padded string. Re-pad it (`f"{gvkey:06d}"`)
before joining to Compustat.

`issuer_cusip` (the 6-character issuer CUSIP) is **not** in the output. It was the
join key for the previous issuer-month linker and is no longer used; derive it as
`cusip_id.str[:6]` if you need it.

---

### Computed Bond Analytics (QuantLib)

| Column | Type | Description |
|--------|------|-------------|
| `pr` | float32 | **Dollar**-volume-weighted clean price, % of par (`Σ price x dollar_vol / Σ dollar_vol`). Contrast `prc_vw_par`, which weights by par quantity |
| `prfull` | float32 | Dirty price = pr + acclast (% of par) |
| `acclast` | float32 | Accrued interest — pure time-accrued interest component |
| `accpmt` | float32 | Accumulated coupon payments since issue |
| `accall` | float32 | Accumulated payments — includes cash flows + accrued interest; used for return calculations |
| `ytm` | float64 | Yield to maturity (annualized, decimal) |
| `mod_dur` | float32 | Modified duration (years) |
| `mac_dur` | float32 | Macaulay duration (years) |
| `convexity` | float32 | Bond convexity |
| `bond_maturity` | float32 | Time to maturity (years) |
| `credit_spread` | float64 | Credit spread over the **maturity**-matched Treasury yield (the Liu-Wu curve interpolated at `bond_maturity`, not at duration) |

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
| `db_type` | Int8 | Source TRACE database: 1=Enhanced, 2=Standard, 3=144A. ❗**2 never appears.** Standard is opt-in and, when run, survives only for dates after the last Enhanced date — which the trailing `DATE_CUT_OFF` always precedes. Filtering `db_type == 2` returns zero rows by construction. |

---

### Bond Characteristics (from FISD)

| Column | Type | Description |
|--------|------|-------------|
| `bond_age` | float32 | Bond age since issuance (years) |
| `bond_amt_outstanding` | Int64 | Amount outstanding, in **thousands of dollars** |

`bond_amt_outstanding` is FISD's `amount_outstanding` as-of the trade date (the most
recent record at or before it), falling back to `offering_amt` where no history
exists. It is in **$ thousands**, exactly as FISD reports it -- no rescaling is
applied. So a bond with $250,000,000 outstanding carries `250000`. Market value is
`bond_amt_outstanding * (pr + acclast) * 10`, in dollars: x1000 to convert thousands
to dollars, /100 because the price is a percent of par.

`coupon`, `principal_amt` and `callable` are used during processing but are **not**
in the output file; take them from FISD directly if you need them.

---

### Industry Classifications

| Column | Type | Description |
|--------|------|-------------|
| `ff12num` | int8 | Fama-French 12 industry classification (1-12) |
| `ff17num` | int8 | Fama-French 17 industry classification (1-17) |
| `ff30num` | int8 | Fama-French 30 industry classification (1-30) |

Assigned from the **issuer's SIC code in FISD** (not from CRSP via PERMNO). A bond
whose SIC matches no range falls into that scheme's "Other" bucket -- 12, 17 or 30
respectively -- so these columns are never null.

---

### Credit Ratings

| Column | Type | Description |
|--------|------|-------------|
| `sp_rating`† | Int8 | S&P credit rating (1-22, where 22=default) |
| `mdy_rating`† | Int8 | Moody's credit rating (1-21, where 21=default) |
| `spc_rating`† | Int8 | S&P composite rating (1-22) |
| `mdc_rating`† | Int8 | Moody's composite rating (1-22) |

`sp_naic` and `comp_rating` are computed during processing but are **not** in the
output file.

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

---

## Sample-defining operations

**The panel is not the raw join.** Six row filters and one transformation are applied
before the file is written, in this order. None of them is visible in the schema, so
this section is the only place they are recorded. All are in
`stage1_pipeline.py::step10a_build_filter_tables`.

### Row filters (applied in order)

| # | Filter | Rule | Removes |
|---|---|---|---|
| 1 | `valid_accrued_vars` | accrued-interest inputs must be present | ~0.0% |
| 2 | `valid_rating` | `spc_rating` **or** `mdc_rating` must be non-null | ~1.5% |
| 3 | `valid_maturity` | `bond_maturity >= 1.0` — **every bond-day inside one year of maturity is dropped** | ~9.4% |
| 4 | `distressed_errors` | `flag_refined_any != 1` (the ultra-distressed filter) | ~0.0% |
| 5 | `2002_07_filter` | `prc_dip != 1` — first price change in **July 2002** exceeding **35** points of par | ~0.0% |
| 6 | `high_prc` | `prc_high != 1`, i.e. `pr <= 300` (% of par) | ~0.0% |

The percentages are from a representative run and are of the pre-filter row count; the
exact figures for **your** run are logged line by line, and land in Table 2 of the
Stage 1 data report. Filter 3 is much the largest, and is a deliberate sample choice —
bonds within a year of maturity behave differently and are conventionally excluded.

Thresholds live in `FINAL_FILTER_CONFIG` (`_stage1_settings.py`):
`price_threshold = 300`, `dip_threshold = 35`.

### Winsorization

**`ytm` and `credit_spread` are winsorised at the 0.5% and 99.5% quantiles WITHIN each
`trd_exctn_dt`** — per-date, not pooled, and applied after the six filters:

```python
final_df[var] = final_df.groupby('trd_exctn_dt')[var].transform(winsorize_group)
```

❗Two consequences worth knowing before you use either column:

- **Tail statistics are computed on clipped data.** Extreme yields and spreads have been
  pulled to their per-date 0.5/99.5 bounds, not removed.
- **The bounds depend on the cross-section on that date**, so the same bond-day can take
  a different value in a run over a different universe. This is why a small test sample
  will not reproduce a full run's `ytm` exactly at the tails.

#### NAIC Categories (sp_naic)

| Code | Category | S&P Ratings |
|------|----------|-------------|
| 1 | Highest Quality | AAA through A- (numeric 1-7) |
| 2 | High Quality | BBB+, BBB, BBB- (8-10) |
| 3 | Medium Quality | BB+, BB, BB- (11-13) |
| 4 | Low Quality | B+, B, B- (14-16) |
| 5 | Lowest Quality | CCC+, CCC, CCC- (17-19) |
| 6 | In or Near Default | CC, C, D (20-22) |

#### Composite Ratings

| Variable | Description |
|----------|-------------|
| `spc_rating` | S&P rating; if missing, filled with `mdy_rating` (Moody's 21 → 22 for default alignment) |
| `mdc_rating` | Moody's rating, first rescaled 21 → 22 so its default bucket lines up with S&P's; if still missing, filled with `sp_rating` on the raw 1-22 scale. There is no S&P 22 → 21 mapping. |

`comp_rating` (their average) is computed during processing but is not in the output.

---

## Related Documentation

- [Stage 1 README](README_stage1.md) — Full Stage 1 documentation
- [Stage 1 QUICKSTART](QUICKSTART_stage1.md) — Quick start guide
- [Main README](../README.md) — Project overview
- [Ultra-distressed Filter](README_distressed_filter.md) — Filter documentation

---

**Last Updated:** September 2026
