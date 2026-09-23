# Stage 0 — Data Dictionary

Stage 0 turns the raw TRACE message tape into a **cleaned bond-day panel**: one row per
`(cusip_id, trd_exctn_dt)`, after cancellations, corrections, reversals, agency double-counts
(Enhanced only), decimal-shift errors, bounce-backs and initial-price errors have been removed.

It writes one panel per TRACE member, and they share the same 21 column names.

## Contents

1. [The daily bond-day panel](#the-daily-bond-day-panel)
2. [The FISD attribute file](#the-fisd-attribute-file)
3. [Audit and diagnostic files](#audit-and-diagnostic-files)

---

## The daily bond-day panel

`enhanced/trace_enhanced_YYYYMMDD.parquet`, `144a/trace_144a_YYYYMMDD.parquet`,
`standard/trace_standard_YYYYMMDD.parquet` — **21 columns**, one row per bond-day.

### Identifiers

| Column | Type | Definition |
|---|---|---|
| `cusip_id` | string | 9-character bond CUSIP. Panel key with `trd_exctn_dt`. |
| `trd_exctn_dt` | timestamp | Trade execution date (the time is always midnight). Panel key with `cusip_id`. |

### Prices

All prices are **clean** (per 100 of par, accrued interest excluded), aggregated over all the
day's surviving trades, customer and inter-dealer alike. Only the bid and ask columns below are
restricted to customer trades.

| Column | Type | Definition |
|---|---|---|
| `prc_ew` | float | Equal-weighted mean price across the day's trades. |
| `prc_vw` | float | Volume-weighted mean price, weighted by **dollar** volume. |
| `prc_vw_par` | float | Volume-weighted mean price, weighted by **par** volume. |
| `prc_first` | float | Price of the first trade of the day. |
| `prc_last` | float | Price of the last trade of the day. |
| `prc_hi` | float | Highest trade price of the day. |
| `prc_lo` | float | Lowest trade price of the day. |

### Trade timing

| Column | Type | Definition |
|---|---|---|
| `time_ew` | float | Equal-weighted mean execution time, **seconds since midnight**. |
| `time_last` | int | Execution time of the last trade, seconds since midnight. |

### Volume

| Column | Type | Definition |
|---|---|---|
| `qvolume` | float | Par volume traded, in **millions**. |
| `dvolume` | float | Dollar volume traded, in **millions**. |

### Dealer bid and ask

> ❗**TRACE reports the DEALER's side.** The split is `rpt_side_cd == 'B'` (the dealer is
> buying) and `'S'` (the dealer is selling), both restricted to trades whose counterparty is a
> customer (`cntra_mp_id == 'C'` in Enhanced, `contra_party_type == 'C'` in Standard and 144A). A customer **BUY** therefore appears as `rpt_side_cd == 'S'`.
> Getting this backwards is the single commonest TRACE bug.

| Column | Type | Definition |
|---|---|---|
| `prc_bid` | float | Dealer **bid**, value-weighted — the dealer buys, so the customer sells. |
| `prc_ask` | float | Dealer **ask**, value-weighted — the dealer sells, so the customer buys. |
| `bid_last` | float | Price of the last dealer-bid trade of the day. |
| `bid_time_ew` | float | Equal-weighted mean time of dealer-bid trades, seconds since midnight. |
| `bid_time_last` | int | Time of the last dealer-bid trade, seconds since midnight. |

### Counts

| Column | Type | Definition |
|---|---|---|
| `trade_count` | int | Number of surviving trades on the day, customer and inter-dealer. |
| `bid_count` | float | Number of dealer buys from customers (= customer sells). NULL, not 0, on a day with none. |
| `ask_count` | float | Number of dealer sells to customers (= customer buys). 0 on a day with dealer buys but no sells; NULL on a day with no customer trade at all. |

> **Column ORDER is not part of the Stage 0 contract.** The names and their meanings are;
> members may order them differently (`bid_time_ew`/`bid_time_last` sit at positions 17-18 in
> Enhanced and 20-21 in 144A). `tests/smoke_assertions.py` compares the schema as a set.
> Stage 2's panel is the opposite — there the order *is* frozen, in `stage2/lib/contract.py`.

---

## The FISD attribute file

`enhanced/trace_enhanced_fisd_YYYYMMDD.parquet` — **23 columns**, one row per bond. 144A writes
a file with the same 23 columns, `144a/trace_fisd_144a_YYYYMMDD.parquet`, restricted by one more
screen that is not a setting: `rule_144a == 'Y'` or `private_placement == 'Y'` (19,333 bonds in
the 2026-09-21 run, against 116,201 in the Enhanced file). Stage 1 and Stage 2 read the Enhanced
one. Not a panel: it carries the static issue attributes used to build the traded universe, and
it is an **input to both Stage 1 and Stage 2** (Stage 2 reads `issue_id`, `rule_144a`,
`country_domicile` and `sic_code` from it).

| Column | Type | Definition |
|---|---|---|
| `complete_cusip` | string | 9-character bond CUSIP. Key. |
| `issue_id` | float | FISD issue identifier (whole numbers, stored as float). |
| `issue_name` | string | Issue description. |
| `issuer_id` | float | FISD issuer identifier (whole numbers, stored as float). |
| `foreign_currency` | string | `Y` if the bond is denominated in a foreign currency. |
| `coupon_type` | string | FISD coupon type (`F` fixed, `V` variable, `Z` zero). |
| `coupon` | float | Annual coupon rate, percent. |
| `convertible` | string | `Y` if convertible. |
| `asset_backed` | string | `Y` if asset-backed. |
| `rule_144a` | string | `Y` if a Rule 144A private placement. Source of the Stage 2 panel's `144a` column. |
| `bond_type` | string | FISD bond type (`CDEB`, `CMTN`, ...). |
| `private_placement` | string | `Y` if privately placed. |
| `interest_frequency` | int | FISD's payment-frequency code: 1, 2, 4 and 12 are payments per year; the file also holds FISD's codes 0 and 99, which are not counts. |
| `dated_date` | timestamp | Date interest begins accruing. |
| `day_count_basis` | string | Accrual convention. |
| `offering_date` | timestamp | Issue date. |
| `maturity` | timestamp | Maturity date. |
| `principal_amt` | float | Principal amount per bond. |
| `offering_amt` | float | Amount offered, **$1,000 units** — multiply by 1,000 for dollars. |
| `country_domicile` | string | Issuer country of domicile. Source of the Stage 2 panel's `country`. |
| `sic_code` | string | SIC industry code. Feeds the FF12/FF17/FF30 industry assignments. |
| `tenor` | float | Years from offering to maturity. |
| `index` | int | Row number (from 1) of the screened universe. |

---

## Audit and diagnostic files

These record what each filter removed. They are diagnostics, not research data.

| File | What it holds |
|---|---|
| `drr_filters_audit_<member>_YYYYMMDD.parquet` | Per-filter before/after row counts for the DRR filter chain. Check `volume_filter` here to confirm whether a dollar floor was applied. |
| `dick_nielsen_filters_audit_<member>_YYYYMMDD.parquet` | Per-filter counts for the Dick-Nielsen cancel/correct/reversal steps (and, in Enhanced, the agency step). |
| `fisd_filters_<member>_YYYYMMDD.parquet` | How many bonds each FISD screen removed (`stage`, `rows_before`, `rows_after`, `removed`). |
| `decimal_shift_cusips_<member>_YYYYMMDD.parquet` | Bonds with a corrected decimal-shift error. |
| `bounce_back_cusips_<member>_YYYYMMDD.parquet` | Bonds with a removed bounce-back. |
| `init_price_cusips_<member>_YYYYMMDD.parquet` | Bonds with a removed initial-price error. |
| `cusip_row_counts_YYYYMMDD.parquet` | Rows per CUSIP, used to plan the chunked pull. |

See [README_stage0.md](README_stage0.md) for how each filter works, and
[README_bounce_back_filter.md](README_bounce_back_filter.md) and
[README_decimal_shift_corrector.md](README_decimal_shift_corrector.md) for the two with the most
moving parts.
