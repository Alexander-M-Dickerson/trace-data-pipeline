# Stage 1 Data Dictionary

This file describes the Stage 1 output dataset. The dataset is available in zipped parquet
format on [Open Source Bond Asset Pricing](https://openbondassetpricing.com/data).

The file you build differs from the file you download. A panel you build yourself has all 44
columns with every value populated. The published download has 32 columns. The four agency
ratings and the `permco` and `gvkey` identifiers (†) are removed because they are licensed, and
six trade-timing and count columns (‡) are removed to keep the file small. `permno` stays. The
restriction applies to redistribution and does not limit the pipeline. The public file is written
by `stage2/make_release.py --what daily`, which refuses to write a file that still has any of
those columns.

## Overview

Stage 1 produces one enriched daily bond dataset from the sources below.

- Clean TRACE prices from Stage 0
- Bond analytics computed with QuantLib (duration, convexity, YTM, credit spreads)
- Bond characteristics from FISD (coupon, maturity, issuer, amount outstanding)
- Credit ratings from S&P and Moody's
- Equity identifiers (CRSP PERMNO/PERMCO and Compustat GVKEY)
- Fama-French industry classifications

> Read [Sample-defining operations](#sample-defining-operations) before using this data. Six
> row filters and a per-date winsorization are applied before the file is written. None of
> them is visible in the schema, and the largest drops 7% of bond-days.

## File Information

| Property | Value |
|----------|-------|
| Location | `stage1/data/stage1_YYYYMMDD.parquet` |
| Format | Apache Parquet (columnar, compressed) |
| Structure | Panel data with one row per (cusip_id, trd_exctn_dt) |
| Size | about 2.7 GB for the full sample as built (2026-09-21 run) |
| Rows | ~31 million (full sample 2002-present, 31,344,732 in the 2026 vintage) |
| Columns | 44 as built, 32 in the public download (see the note above) |
| Download | Available in zipped parquet format on [Open Source Bond Asset Pricing](https://openbondassetpricing.com/data) |

## Price Convention

All prices are in percentage of par.

| Price Value | Meaning | Dollar Value (for $1,000 par) |
|-------------|---------|-------------------------------|
| 100 | Par | $1,000 |
| 99 | 99% of par | $990 |
| 105.5 | 105.5% of par | $1,055 |
| 85.25 | 85.25% of par | $852.50 |

All bonds in the dataset have a principal amount of $1,000.

## Notes

Two groups of columns are dropped from the public download.

- Columns marked with a dagger (†) are excluded because of proprietary data restrictions.
- Columns marked with a double dagger (‡) are excluded to reduce the file size.

## Variable Reference

### Identifiers

| Column | Type | Description |
|--------|------|-------------|
| `cusip_id` | category | 9-character CUSIP identifier (unique bond ID) |
| `permno` | Int32 | CRSP PERMNO equity identifier (links to stock data) |
| `permco`† | Int32 | CRSP PERMCO company identifier. Not in the public download. It is populated in a panel you build. |
| `gvkey`† | Int32 | Compustat GVKEY identifier (links to accounting data). Not in the public download. It is populated in a panel you build. |
| `trd_exctn_dt` | datetime | Trade execution date |

#### How bonds are linked to firms

The equity identifiers come from the bond-firm linker published at
[openbondassetpricing.com](https://openbondassetpricing.com/), downloaded by
`run_pipeline.sh`. The mapping is bond-level and dated, with one row per
(9-character CUSIP, ownership window). A bond that changes hands through an acquisition, a
spin-off or a rename therefore points at the right firm in each period, and not at whichever
firm owned it last.

The linker ships two dated windows. `w0`/`w1` is the evidence window. It answers *when is this
mapping provable*, and it is the bond's life intersected with the firm's CRSP listing. `i0`/`i1`
is the identity window. It answers *whose bond is this*, and it stays open where no successor or
acquisition contradicts it. Use evidence when you are joining equity-side data (CRSP returns,
market cap), because outside it there is no listed firm to join to. Use identity when the firm id
is a label you group by. `bond_firm_linker_2026/SCHEMA.md`, shipped inside the bundle, is the
authority.

This daily panel joins the identity window, and so does the Stage 2 monthly panel
(`_stage1_settings.LINKER_WINDOW`, `stage2/_stage2_settings.LINKER_WINDOW`, and a test that
fails if the two ever differ). A bond does not stop being a firm's bond because the firm's
CRSP listing ended. On every row of the linker the evidence window lies inside the identity
window with the same firm, so the identity join never changes an id the evidence join would
give. It adds labels outside the evidence window.

> Before 2026-09-21 this stage joined the evidence window while Stage 2 joined identity, and
> the two panels named a different firm on about 2% of bond-months. When the join moved, measured
> on the 2026-09-10 run, 644,754 bond-days gained an id, none lost one and none changed, and
> `permno` coverage went from 87.8% to 89.8% of bond-days. If you need ids only where equity
> data exists, set `LINKER_WINDOW = ("w0", "w1")`, or re-join the linker yourself.

The daily and monthly ids can still differ on a handful of bond-months (about 140 of 1.95
million), because the monthly panel takes the id at the calendar month-end and a window can
start or end between a bond's last trade and that date.

`permno` is NULL for about 10% of bond-days, and that is deliberate. About three quarters of
those are bonds the linker does not hold at all, because no listed firm can be evidenced behind
them. The rest are dates outside every window of a bond the linker does hold. The identifier is
left missing in both cases because a stale one would be wrong and nothing would show it. If
you need to know why a particular bond has no link, the published release ships
`fl_verdicts.parquet`, which records every refusal and its reason, and
`firm_names.parquet`, which maps `permno` to a dated firm name.

`gvkey` is stored numerically here for continuity with earlier releases, but
Compustat's GVKEY is a 6-character zero-padded string. Re-pad it (`f"{gvkey:06d}"`)
before joining to Compustat.

`issuer_cusip` (the 6-character issuer CUSIP) is not in the output. It was the
join key for the previous issuer-month linker and is no longer used. Derive it as
`cusip_id.str[:6]` if you need it.

### Computed Bond Analytics (QuantLib)

| Column | Type | Description |
|--------|------|-------------|
| `pr` | float32 | Dollar-volume-weighted clean price, % of par (`Σ price x dollar_vol / Σ dollar_vol`). Contrast `prc_vw_par`, which weights by par quantity |
| `prfull` | float32 | Dirty price = pr + acclast (% of par) |
| `acclast` | float32 | Accrued interest, the pure time-accrued interest component |
| `accpmt` | float32 | Accumulated coupon payments since issue |
| `accall` | float32 | Accumulated payments (cash flows + accrued interest), used for return calculations |
| `ytm` | float64 | Yield to maturity (annualized, decimal) |
| `mod_dur` | float32 | Modified duration (years) |
| `mac_dur` | float32 | Macaulay duration (years) |
| `convexity` | float32 | Bond convexity |
| `bond_maturity` | float32 | Time to maturity (years) |
| `credit_spread` | float64 | Credit spread over the maturity-matched Treasury yield (the Liu-Wu curve interpolated at `bond_maturity`, not at duration) |

#### Price Definitions

| Term | Formula | Use Case |
|------|---------|----------|
| Clean price | Quoted price without accrued interest | Trading, quoting |
| Dirty price | `pr + acclast` | Actual settlement price, market cap |

#### Accrued Interest Variables

| Variable | Description | Use Case |
|----------|-------------|----------|
| `acclast` | Interest accrued since last coupon payment | Dirty price, market cap |
| `accpmt` | Cumulative coupon payments since bond issuance | Tracking total cash flows |
| `accall` | Accumulated payments including cash flows and accrued interest | Return calculations |

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

### Database Source

| Column | Type | Description |
|--------|------|-------------|
| `db_type` | Int8 | Source TRACE database (1=Enhanced, 2=Standard, 3=144A). 2 does not appear in the default configuration. Standard is opt-in (`TRACE_MEMBERS` is `enhanced 144a`), and when it is run it survives only for dates after the last Enhanced date. Every automatic `DATE_CUT_OFF`, `auto:complete` included, ends the sample on or before the last Enhanced date, so that window is always cut away and `db_type == 2` has zero rows by construction (`tests/smoke_assertions.py` checks this). Only a literal `DATE_CUT_OFF` later than the last Enhanced date lets it through. |

### Bond Characteristics (from FISD)

| Column | Type | Description |
|--------|------|-------------|
| `bond_age` | float32 | Bond age since issuance (years) |
| `bond_amt_outstanding` | Int64 | Amount outstanding, in **thousands of dollars** |

`bond_amt_outstanding` is FISD's `amount_outstanding` as of the trade date (the most
recent record at or before it), falling back to `offering_amt` where no history
exists. It is in $ thousands, exactly as FISD reports it, and no rescaling is
applied. A bond with $250,000,000 outstanding therefore has the value `250000`. Market
value is `bond_amt_outstanding * (pr + acclast) * 10`, in dollars. That is x1000 to convert
thousands to dollars and /100 because the price is a percent of par.

`coupon`, `principal_amt` and `callable` are used during processing but are not
in the output file. Take them from FISD directly if you need them.

### Industry Classifications

| Column | Type | Description |
|--------|------|-------------|
| `ff12num` | int8 | Fama-French 12 industry classification (1-12) |
| `ff17num` | int8 | Fama-French 17 industry classification (1-17) |
| `ff30num` | int8 | Fama-French 30 industry classification (1-30) |

These are assigned from the issuer's SIC code in FISD (not from CRSP via PERMNO). A bond
whose SIC matches no range falls into that scheme's "Other" bucket (12, 17 or 30
respectively), so these columns are never null.

### Credit Ratings

| Column | Type | Description |
|--------|------|-------------|
| `sp_rating`† | Int8 | S&P credit rating (1-22, where 22=default) |
| `mdy_rating`† | Int8 | Moody's credit rating (1-21, where 21=default) |
| `spc_rating`† | Int8 | S&P composite rating (1-22) |
| `mdc_rating`† | Int8 | Moody's composite rating (1-22) |

`sp_naic` and `comp_rating` are computed during processing but are not in the
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
| 21 | C | Default (Moody's has no D) |

#### NAIC Categories (sp_naic)

`sp_naic` is computed during processing but is not in the main file. It is saved as
`sp_naic_numeric` in `stage1/data/sp_ratings_<stamp>.parquet`.

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
| `spc_rating` | S&P rating. If missing, filled with `mdy_rating` (Moody's 21 → 22 for default alignment) |
| `mdc_rating` | Moody's rating, first rescaled 21 → 22 so its default bucket lines up with S&P's. If still missing, filled with `sp_rating` on the raw 1-22 scale. There is no S&P 22 → 21 mapping. |

`comp_rating` (their average) is computed during processing but is not in the output.

## Sample-defining operations

The panel is not the raw join. Six row filters and one transformation are applied
before the file is written, in this order. None of them is visible in the schema, so
this section is the only place they are recorded. Filter 1 is applied in
`stage1_pipeline.py::step4_merge_fisd`. The others, and the winsorization, are applied in
`step10a_build_filter_tables`, which also records all six.

### Row filters (applied in order)

| # | Filter | Rule | Removes |
|---|---|---|---|
| 1 | `valid_accrued_vars` | the inputs QuantLib needs, which are `bond_maturity > 0`, `bond_age > 0`, `dated_date` present, and `interest_frequency` not in {-1, 13, 16} | 0.17% |
| 2 | `valid_rating` | `spc_rating` or `mdc_rating` must be non-null | 1.12% |
| 3 | `valid_maturity` | `bond_maturity >= 1.0`, so **every bond-day inside one year of maturity** is dropped | 7.02% |
| 4 | `distressed_errors` | `flag_refined_any != 1` (the ultra-distressed filter) | 0.01% |
| 5 | `2002_07_filter` | `prc_dip != 1` (the flag for a first price change in July 2002 exceeding 35 points of par) | 6 rows |
| 6 | `high_prc` | `prc_high != 1`, i.e. `pr <= 300` (% of par) | 20 rows |

The figures are from the 2026-09-21 run, which the 2026 vintage was built from, as a share of
the 34,189,014 rows before filter 1. Together the six removed 8.32%, leaving 31,344,732. The
exact figures for your run are logged line by line and appear in Table 2 of the
Stage 1 data report. Filter 3 is much the largest and is a deliberate sample choice.
Bonds within a year of maturity behave differently and are conventionally excluded.

The thresholds are in `FINAL_FILTER_CONFIG` (`_stage1_settings.py`), as
`price_threshold = 300` and `dip_threshold = 35`.

### Winsorization

`ytm` and `credit_spread` are winsorised at the 0.5% and 99.5% quantiles within each
`trd_exctn_dt`. The winsorization is per-date, not pooled, and is applied after the six
filters.

```python
final_df[var] = final_df.groupby('trd_exctn_dt')[var].transform(winsorize_group)
```

This has two consequences when you use either column.

- Tail statistics are computed on clipped data. Extreme yields and spreads have been
  pulled to their per-date 0.5/99.5 bounds, not removed.
- The bounds depend on the cross-section on that date, so the same bond-day can take
  a different value in a run over a different universe. This is why a small test sample
  will not reproduce a full run's `ytm` exactly at the tails.

## Related Documentation

- [Stage 1 README](README_stage1.md), the full Stage 1 documentation
- [Stage 1 QUICKSTART](QUICKSTART_stage1.md), the quick start guide
- [Main README](../README.md), the project overview
- [Ultra-distressed filter](README_distressed_filter.md), the filter documentation

Last updated in September 2026.
