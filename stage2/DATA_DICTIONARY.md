# Stage 2 Data Dictionary

This document describes the variables computed in the Stage 2 bond data pipeline.

---

## Contents

1. [How to Use the Monthly Data](#how-to-use-the-monthly-data)
   - [Market Microstructure Adjusted Signals and Returns](#market-microstructure-adjusted-signals-and-returns)
   - [Excess and Duration-Adjusted Return-Based Signals](#excess-and-duration-adjusted-return-based-signals)
   - [Alternative Month-End Returns](#alternative-month-end-returns)

2. [Signal Definitions](#signal-definitions)
   - [Bond Identifiers and Return Metrics](#bond-identifiers-and-return-metrics)
   - [Bond Characteristics](#bond-characteristics)
   - [Cluster I: Spreads, Yields, Size](#cluster-i-spreads-yields-size)
   - [Cluster II: Value](#cluster-ii-value)
   - [Cluster III: Momentum & Reversal](#cluster-iii-momentum--reversal)
   - [Cluster IV: Illiquidity](#cluster-iv-illiquidity)
   - [Cluster V: Volatility & Risk](#cluster-v-volatility--risk)
   - [Cluster VI: Market Risk](#cluster-vi-market-risk)
   - [Cluster VII: Credit & Default Betas](#cluster-vii-credit--default-betas)
   - [Cluster VIII: Volatility & Liquidity Betas](#cluster-viii-volatility--liquidity-betas)
   - [Cluster IX: Macro & Other Betas](#cluster-ix-macro--other-betas)

3. [Redaction: the file you download is not the file the pipeline builds](#redaction-the-file-you-download-is-not-the-file-the-pipeline-builds)
4. [Bond Returns](#bond-returns)
   - [Month-End Return (`ret_vw`)](#month-end-return-ret_vw)
   - [Month-Begin Return (`ret_vw_bgn`)](#month-begin-return-ret_vw_bgn)
   - [Implementation Gap (`igap_bgn`)](#implementation-gap-igap_bgn)
   - [Latent Implementation Bias (`lib`)](#latent-implementation-bias-lib)
   - [Signal Gap for Price-Based Signals](#signal-gap-for-price-based-signals)
   - [The Three Approaches](#the-three-approaches)
   - [Feasibility Bias](#feasibility-bias)
   - [Excess and Duration-Adjusted Returns](#excess-and-duration-adjusted-returns)

5. [Defaulted Bond Returns](#defaulted-bond-returns)
   - [Default Identification](#default-identification)
   - [Standard Return (Non-Default)](#standard-return-non-default)
   - [Default Event Return](#default-event-return)
   - [Trading Under Default (Flat Return)](#trading-under-default-flat-return)

6. [Technical Appendix](#technical-appendix)
   - [Factor Models](#factor-models)
   - [Duration-Adjusted Factor Substitution](#duration-adjusted-factor-substitution)
   - [Estimation Methodology](#estimation-methodology)
   - [Model Specifications](#model-specifications)
   - [Factor Definitions](#factor-definitions)
   - [Momentum and Reversal Signals](#momentum-and-reversal-signals)
   - [Value Signals](#value-signals)
   - [Illiquidity Measures](#illiquidity-measures)
   - [Within-Month Risk Statistics](#within-month-risk-statistics)
   - [Output Files](#output-files)

7. [References](#references)

---

## How to Use the Monthly Data

All data in the panel is sampled at the end of month $t$, with one exception. `lib` and `libd` sit on the row for month $t$ and measure the price move from that month-end to the first trade of month $t+1$, so they look one month ahead. They describe a cost, and are not signals to sort on. See [Latent Implementation Bias](#latent-implementation-bias-lib).

**Sample start dates:** Most signals have a sample start date of 2002-08, including those that require a rolling estimation period. Signals requiring rolling data use ICE/BofA data to generate observations prior to 2002-08, enabling a consistent start date of 2002-08 for the main panel. Rolling signals that require TRACE-based illiquidity data (e.g., Amihud and Pastor-Stambaugh liquidity betas) start 2003-07.

Several datasets are available on the [Open Bond Asset Pricing](https://openbondassetpricing.com/) website.

### Market Microstructure Adjusted Signals and Returns

The main panel includes several identifiers, additional variables, and signals. As standard, we provide market microstructure-adjusted (MMN) price-based signals, which includes all variables related to bond price, yield, spread, value, illiquidity, within-month risk, and daily betas. The signals employ a minimum 1-business day gap before the month-end price used for returns. In the 2026 vintage the average (median) gap is 1.66 (1) business days. The maximum allowed gap is capped at 10 business days, and long gaps are rare (the 95th percentile is 5 days and the 99th is 7). The methodology is outlined in the [Signal Gap](#signal-gap-for-price-based-signals) section below.

**Main panel usage:** Use month-end returns (`ret_vw`) with the MMN-adjusted signals provided in the main panel.

As an alternative, researchers may access and download the unadjusted (noisy) price-based signals from [openbondassetpricing.com](https://openbondassetpricing.com/). The file is named `mmn_price_based_signals_YYYYmmdd.parquet`; all variables in this file have the suffix `_mmn` (e.g., `ytm_mmn`, `cs_mmn`, `val_hz_mmn`).

**Noisy signals usage:** If using signals from `mmn_price_based_signals_YYYYmmdd.parquet`, you **must** use month-begin returns (`ret_vw_bgn`) to avoid microstructure bias.

Our recommendation is to use the main panel data as provided with `ret_vw`. However, results are extremely similar using either method.

### Excess and Duration-Adjusted Return-Based Signals

The main panel assumes researchers will form factors with excess returns, $r - r^{f}$, computed from the panel as `ret_vw - rfret`. If research designs instead rely on duration-adjusted returns, $r^{x} = r - r^{Tsy}$ — which is what **`ret_vwx` means throughout this document, namely `ret_vw - tret`** — we provide signals specifically computed with duration-adjusted returns. These include all factor betas in `betas_x_<YYYY>.parquet` and momentum/long-term reversal variables in `mom_retx_<YYYY>.parquet`, where `<YYYY>` is the vintage year of the release (see [Output Files](#output-files)).

> `ret_vwx` is **only** the duration-adjusted return. There is no column for the excess return; subtract `rfret` from `ret_vw` yourself if that is what you want.

The Treasury return `tret` is the U.S. Treasury bond return that is duration-matched to each corporate bond's modified duration (`md_dur`). For each bond $i$ in each month $t$, we linearly interpolate using key rate U.S. Treasury bond returns from WRDS and the bond's modified duration, following the method of Andreani, Palhares, and Richardson (2024).

The short-term reversal signal can be computed as the current `str` variable minus `tret`. This signal is already MMN-adjusted in the main panel.

### Alternative Month-End Returns

For researchers requiring alternative return measures, we publish
`returns_alt_<YYYY>.parquet`. It is keyed on `cusip` + `date` like the main panel, and each
column differs from `ret_vw` only in which price is used at each month end:

| Column | Type | Definition |
|---|---|---|
| `cusip` | string | 9-character bond CUSIP. Panel key with `date`. |
| `date` | datetime | Calendar month end. Panel key with `cusip`. |
| `ret_vwp` | float32 | Return computed from **par-weighted** prices on day $d$. |
| `ret_ew` | float32 | Return computed from **equal-weighted** prices on day $d$. |
| `ret_1st` | float32 | Return computed from the **first** available trade price on day $d$. |
| `ret_lst` | float32 | Return computed from the **last** available trade price on day $d$. |
| `ret_bid` | float32 | Return computed from the volume-weighted average **bid** price on day $d$. |
| `tret` | float32 | Duration-matched Treasury return, as in the main panel. |

where day $d$ is in the last 5 business days of months $t$ and $t+1$. The five alternative
Treasury benchmarks (`tret_bns` and the rest) are in the main panel only, not in this file.

All five return measures use the same total-return construction as `ret_vw` -- the coupon is
carried through accrued interest, not inferred from a dirty-price ratio -- so they differ from
each other and from `ret_vw` only in the price input.

---

## Signal Definitions

The table below provides definitions for all signals in the database.

**Return-based signals** (betas, momentum, reversals, VaR, ES): We compute both standard and duration-adjusted versions. The duration-adjusted variant uses $r^x = r - r^{Tsy}$ and is stored in `betas_x_<YYYY>.parquet` (for all factor betas) and `mom_retx_<YYYY>.parquet` (for momentum and long-term reversal signals).

**Price-based signals** (yields, spreads, value, book-to-market, prior 1-month return): All price-based signals in the main panel are market microstructure adjusted (MMN) by default, observed with a minimum 1-business-day gap before the month-end price used for returns. Researchers preferring unadjusted signals can download `mmn_price_based_signals_<YYYY>.parquet` from [openbondassetpricing.com](https://openbondassetpricing.com/); all variables in this file have the suffix `_mmn`.

For all signals requiring a rolling window (betas, VaR, ES), we use a 36-month rolling window with a minimum of 12 observations, denoted 36(12). For long-term reversal signals requiring >12 months of history, we use an expanding window that starts at 12-3 and ramps up to the target horizon (e.g., 48-12 or 30-6) to preserve sample coverage.

---

### Bond Identifiers and Return Metrics

| Mnemonic | Name | Description |
|----------|------|-------------|
| `cusip` | Bond Identifier | 9-digit CUSIP bond identifier. |
| `date` | Date | True month-end date (YYYY-MM-DD). |
| `issuer_cusip` | Issuer CUSIP | 6-digit firm identifier (first 6 digits of CUSIP). |
| `permno` | CRSP PERMNO | CRSP permanent security identifier. |
| `permco` | CRSP PERMCO | CRSP permanent company identifier. **Null in the published file** -- see *Redaction* below. |
| `gvkey` | Compustat GVKEY | Compustat global company key. **Null in the published file** -- see *Redaction* below. |
| `hprd` | Holding Period | NYSE trading sessions from the start trade (dt_s) to the end trade (dt_e) of the month-end return. Runs 15 to 27, because each trade may fall anywhere in the last 5 sessions of its month. |
| `lib` | Latent Implementation Bias | Clean price return from month-end to the next month-begin. On the row for month $t$: $\text{LIB}_t = P_{t+1}^{\text{bgn}} / P_t^{\text{end}} - 1$, the gap that FOLLOWS that month's end price. |
| `libd` | LIB (Dirty) | LIB computed using dirty prices (includes accrued interest and coupon). |
| `ret_type` | Return Type | Return classification: `standard`, `trad_in_def`, or `default_evnt`. |
| `ff17num` | FF17 Industry | Fama-French 17-industry classification. |
| `ff30num` | FF30 Industry | Fama-French 30-industry classification. |
| `mcap_s` | Market Cap (Start) | Bond market capitalization at end of month $t-1$. |
| `mcap_e` | Market Cap (End) | Bond market capitalization at end of month $t$. |
| `tret` | Treasury Return | Duration-matched U.S. Treasury portfolio return. Used to compute duration-adjusted returns: $r^x = r - \texttt{tret}$. |
| `tret_bns` | Treasury Return, Duration-Weighted Cash Flows | Treasury benchmark built from the bond's own cash flows, weighted by their present-value shares (Macaulay duration weights) and applied to zero-coupon Treasury returns from the Gurkaynak-Sack-Wright curve. Subtract from `ret_vw` for a duration-adjusted return. |
| `tret_cfm` | Treasury Return, Cash-Flow-Weighted | As `tret_bns`, but weighting the same cash flows by their future value rather than their present value. Places more weight on long-dated cash flows and so overstates a bond's interest rate risk; reported as the authors' own robustness alternative. |
| `tret_gprs` | Treasury Return, Exact Duration Match | As `tret_bns`, refined so the Treasury portfolio's duration equals the bond's exactly. The present-value weights are duration-matched only when the term structure is flat; this solves for the Treasury yield that restores the match and reweights accordingly. |
| `tret_cls` | Treasury Return, Cash-Flow-Matched | As `tret_bns`, but discounting the same cash flows on the Treasury zero curve instead of at the bond's own yield to maturity. This replicates the bond's cash flows rather than its duration, so it removes slope and curvature effects as well as parallel shifts. The difference from `tret_bns` is the authors' higher-order component. |
| `tret_mat` | Treasury Return, Maturity-Matched | The twin of `tret`: the same key-rate Treasury index returns, the same interpolation nodes and rounding, but interpolated at the bond's remaining MATURITY rather than its modified duration. Included as the maturity-matched comparator; duration matching is the more precise of the two. |
| `ret_vw` | Total Return (End) | Month-end to month-end total return. |
| `ret_vw_bgn` | Total Return (Begin) | Month-begin to month-end total return within the same month. |
| `dt_s` | Date Start | Trade date for month-end price in month $t-1$ (last 5 BD). |
| `dt_e` | Date End | Trade date for month-end price in month $t$ (last 5 BD). |
| `dt_s_bgn` | Date Start (Begin) | Trade date for month-begin price in month $t$ (first 5 BD). |
| `dt_e_bgn` | Date End (Begin) | Trade date for month-end price in month $t$ (last 5 BD) for begin returns. |
| `hprd_bgn` | Holding Period (Begin) | NYSE trading sessions from the month-begin trade (dt_s_bgn) to the month-end trade (dt_e_bgn) of the same month. |
| `igap_bgn` | Implementation Gap | NYSE trading sessions from the last session of month $t-1$ to the month-begin trade of month $t$ (dt_s_bgn); 1 to 5. |
| `sig_dt` | Signal Date | Date when price-based signal was observed (minimum 1 BD before month-end price). |
| `sig_gap` | Signal Gap | Business days between signal observation and month-end price; ranges 1–10 BD. |
| `rfret` | Risk-Free Rate | Monthly risk-free rate from Fama-French. Used for excess returns: $r^x = r - r^f$. |
**The five `tret_*` benchmarks before 1986.** The Treasury curve's long end is held flat beyond the longest tenor the Gurkaynak-Sack-Wright curve actually fitted, because no Treasury that long existed to price. This is the convention of Gurkaynak, Sack and Wright (2007) and of Ghaderi, Plante, Roussanov and Seo (2026), whose Appendix A states it in as many words -- "we conservatively apply flat extrapolation when necessary". It is what makes a 1973 start possible at all: without it no bond outliving the fitted curve could be priced, and the early sample would simply be absent. Measured cost: RMSE 15.8 bps, bias -0.3 bps, and exactly zero for any bond whose cash flows end inside the fitted range. The benchmarks diverge from one another most in this early period, for the same reason -- the curve is least anchored there.

---

### Bond Characteristics

| Mnemonic | Name | Description |
|----------|------|-------------|
| `spc_rat` | S&P Composite Rating | Composite credit rating: S&P rating if available, otherwise Moody's rating. Scale: 1 (AAA) to 21 (C), 22 = Default. **Collapsed to {1, 11} in the published file** -- see *Redaction* below. |
| `mdc_rat` | Moody's Composite Rating | Composite credit rating: Moody's rating if available, otherwise S&P rating. Scale: 1 (Aaa) to 21 (C), 22 = Default. **Collapsed to {1, 11} in the published file** -- see *Redaction* below. |
| `call` | Callable Indicator | Indicator for embedded call option (1 = callable, 0 = non-callable). |
| `fce_val` | Face Value | Bond amount outstanding (face value); units of the bond outstanding. |
| `144a` | Rule 144A Indicator | Dummy variable: 1 if bond is Rule 144A, 0 otherwise. |
| `country` | Country | Country of issuance (e.g., `USA` for U.S. bonds). |

---

### Cluster I: Spreads, Yields, Size

| Mnemonic | Name | Description |
|----------|------|-------------|
| `tmat` | Time to Maturity | Years remaining until bond maturity. |
| `age` | Bond Age | Years since bond issuance. |
| `ytm` | Yield to Maturity | Annualized yield to maturity. Computed with QuantLib. |
| `cs` | Credit Spread | Annualized credit spread: yield minus maturity-matched U.S. Treasury yield. |
| `md_dur` | Modified Duration | Modified duration measuring price sensitivity to yield changes. Computed with QuantLib. |
| `convx` | Convexity | Second-order price sensitivity to yield changes. Computed with QuantLib. |
| `sze` | Bond Size | Bond market capitalization: dirty price times amount outstanding ($ millions). |
| `dcs6` | 6-Month Spread Narrowing (log) | Log change in credit spread over prior 6 months: $\log(cs_{t-6}) - \log(cs_t)$. If the spread is missing exactly 6 months ago, the nearest adjacent month is used within a ±1 month band, the EARLIER month first. |
| `cs_mu12_1` | 12-Month Average Spread | Rolling 12-month average credit spread, skipping the prior month. Requires minimum 6 observations. |

---

### Cluster II: Value

| Mnemonic | Name | Description |
|----------|------|-------------|
| `bbtm` | Bond Book-to-Market | Par price divided by market price of the bond. All bonds have a par value of \$1,000 (100%). |
| `val_hz` | Value (HZ) | Percentage deviation of observed credit spread from fitted "fair" spread: $(cs - \widehat{cs}) / \widehat{cs}$. Controls: rating, industry, 3-month spread change, callable. Fitted spread is estimated each month via cross-sectional regression; because cs is used, this signal is also exposed to microstructure noise. |
| `val_hz_dts` | Value (HZ, DtS-adjusted) | HZ value signal demeaned within duration-times-spread quintiles to control for systematic spread-duration risk. |
| `val_ipr` | Value (IPR) | Log-spread residual from fair-value regression: $\log(cs) - X'\hat{\beta}$. Controls: rating, industry, log duration, volatility, callable. Fitted spread is estimated each month via cross-sectional regression; because cs is used, this signal is also exposed to microstructure noise. |
| `val_ipr_dts` | Value (IPR, DtS-adjusted) | IPR value signal demeaned within duration-times-spread quintiles. |

---

### Cluster III: Momentum & Reversal

| Mnemonic | Name | Description |
|----------|------|-------------|
| `mom3_1` | 3-Month Momentum | Cumulative return months $t-2$ to $t-1$, skipping prior month. |
| `mom6_1` | 6-Month Momentum | Cumulative return months $t-5$ to $t-1$. |
| `mom9_1` | 9-Month Momentum | Cumulative return months $t-8$ to $t-1$. |
| `mom12_1` | 12-Month Momentum | Cumulative return months $t-11$ to $t-1$. |
| `mom12_7` | Intermediate Momentum | Cumulative return months $t-11$ to $t-7$, excluding recent returns. |
| `sysmom3_1` | Systematic Momentum (3-1) | Sum of fitted values over months $t-2$ to $t-1$ from a 36(12) rolling CAPMB regression. |
| `sysmom6_1` | Systematic Momentum (6-1) | Sum of fitted values over months $t-5$ to $t-1$ from a 36(12) rolling CAPMB regression (bond market factor). |
| `sysmom12_1` | Systematic Momentum (12-1) | Sum of fitted values over months $t-11$ to $t-1$ from a 36(12) rolling CAPMB regression. |
| `idimom3_1` | Idiosyncratic Momentum (3-1) | Sum of residuals over months $t-2$ to $t-1$ from a 36(12) rolling CAPMB regression. |
| `idimom6_1` | Idiosyncratic Momentum (6-1) | Sum of residuals over months $t-5$ to $t-1$ from a 36(12) rolling CAPMB regression (bond market factor). |
| `idimom12_1` | Idiosyncratic Momentum (12-1) | Sum of residuals over months $t-11$ to $t-1$ from a 36(12) rolling CAPMB regression. |
| `imom1` | Industry Momentum (1) | Equal-weighted average prior-month return of other bonds in the same FF17 industry. |
| `imom3_1` | Industry Momentum (3-1) | Equal-weighted average 3-1 momentum of other bonds in the same FF17 industry. |
| `imom12_1` | Industry Momentum (12-1) | Equal-weighted average 12-1 momentum of other bonds in the same FF17 industry. |
| `ltr24_3` | Long-Term Reversal (24-3) | Cumulative return months $t-23$ to $t-3$. Uses expanding window starting at 12-3. |
| `ltr30_6` | Long-Term Reversal (30-6) | Cumulative return months $t-29$ to $t-6$. Uses expanding window starting at 12-3, ramping to 30-6. |
| `ltr48_12` | Long-Term Reversal (48-12) | Cumulative return months $t-47$ to $t-12$. Uses expanding window starting at 12-3, ramping to 48-12. |
| `iltr24_3` | Industry LTR (24-3) | Equal-weighted average 24-3 LTR of other bonds in the same FF17 industry. |
| `iltr30_6` | Industry LTR (30-6) | Equal-weighted average 30-6 LTR of other bonds in the same FF17 industry. |
| `iltr48_12` | Industry LTR (48-12) | Equal-weighted average 48-12 LTR of other bonds in the same FF17 industry. |
| `str` | Short-Term Reversal | Prior month return $r_{t-1}$. |

---

### Cluster IV: Illiquidity

| Mnemonic | Name | Description |
|----------|------|-------------|
| `pi` | Price Impact | Pastor-Stambaugh liquidity: negated coefficient from return reversal regression on signed volume. Requires a minimum of 5 daily returns. |
| `ami` | Amihud Illiquidity | Within-month mean of daily $\|r_t\|/\text{dvol}_t$. Requires a minimum of 5 daily returns. |
| `ami_v` | Amihud Volatility | Monthly standard deviation of daily Amihud ratios. Requires a minimum of 5 daily returns. |
| `roll` | Roll Spread | Implicit bid-ask spread: $2\sqrt{\max(-\text{Cov}(r_t, r_{t-1}), 0)}$. Requires a minimum of 5 daily returns. |
| `ilq` | Roll Autocovariance | Negative autocovariance of log returns × 100. Requires a minimum of 5 daily returns. |
| `spd_rel` | Relative Bid-Ask Spread | Volume-weighted $(P^{ask} - P^{bid})/\text{mid}$. Requires a minimum of 5 prices. |
| `spd_abs` | Absolute Bid-Ask Spread | Volume-weighted $(P^{ask} - P^{bid})$ in dollars. Requires a minimum of 5 prices. |
| `cs_sprd` | Corwin-Schultz Spread | High-low spread estimator using two-day price ranges. Requires a minimum of 5 prices. |
| `ar_sprd` | Abdi-Ranaldo Spread | Closing price spread estimator. Requires a minimum of 5 prices. |
| `p_zro` | No-Trade Day Proportion | Fraction of business days with no valid price. Returns are never inspected, so the old name "Zero-Return Proportion" contradicted this description. |
| `p_fht` | FHT Spread | Spread derived from zero-return proportion: $2\sigma\Phi^{-1}((1+p_{zro})/2)$. Requires a minimum of 5 daily returns. |
| `vov` | Volatility-over-Volume | Liquidity proxy: $2.5 \times \sigma^{0.6} / \bar{V}^{0.25}$, where $\sigma$ is the volatility of daily RETURNS and $\bar{V}$ mean volume. Requires a minimum of 5 daily returns. ❗Not the volatility OF volume. |
| `lix` | Negative LIX (Illiquidity) | $-\log_{10}[(V \times P_{close}) / (P_{high} - P_{low})]$. Requires a minimum of 5 prices. ❗Stored NEGATED (`lib/illiq_pandas.py`), so it rises with ILLIQUIDITY -- 99.29% of non-null values are negative, median -2.181. |

---

### Cluster V: Volatility & Risk

| Mnemonic | Name | Description |
|----------|------|-------------|
| `dvol` | Daily Volatility | Standard deviation of daily returns within the month. |
| `dskew` | Daily Skewness | Central skewness of daily returns within the month. |
| `dkurt` | Daily Kurtosis | Excess kurtosis of daily returns within the month. |
| `rvol` | Realized Volatility | Within-month square root of sum of squared daily returns: $\sqrt{\sum r_t^2}$. Requires a minimum of 5 daily returns. |
| `rsj` | Realized Signed Jump | Within-month asymmetry in positive vs. negative squared returns: $(RV^+ - RV^-)/RV$. Requires a minimum of 5 daily returns. |
| `rsk` | Realized Skewness | Within-month third moment of daily returns scaled by realized volatility. Requires a minimum of 5 daily returns. |
| `rkt` | Realized Kurtosis | Within-month fourth moment of daily returns scaled by realized volatility. Requires a minimum of 5 daily returns. |
| `var_90` | 90% Value-at-Risk | 10th percentile loss from empirical daily return distribution over a 36(12) rolling window. |
| `var_95` | 95% Value-at-Risk | 5th percentile loss from empirical daily return distribution over a 36(12) rolling window. |
| `es_90` | 90% Expected Shortfall | Mean of worst 10% of daily returns over a 36(12) rolling window. |
| `dvol_sys` | Systematic Volatility | Standard deviation of systematic returns (CAPMB fitted values) within the month. Requires a minimum of 5 daily returns. |
| `dvol_idio` | Idiosyncratic Volatility | Standard deviation of idiosyncratic returns (CAPMB residuals) within the month. Requires a minimum of 5 daily returns. |
| `ivol_mkt` | Idiosyncratic Volatility (MKT) | Residual volatility from joint MKTRF+MKTB regression. |
| `ivol_bbw` | Idiosyncratic Volatility (BBW) | Residual volatility from BBW 4-factor regression. |
| `ivol_vp` | Idiosyncratic Volatility (VP) | Residual volatility from VOLPSB regression. |
| `iskew` | Idiosyncratic Skewness | Skewness of residuals from coskewness regression. |

---

### Cluster VI: Market Risk

| Mnemonic | Name | Description |
|----------|------|-------------|
| `b_mktrf_mkt` | Equity Market Beta | Beta on MKTRF from joint regression with MKTB (36-month rolling). |
| `b_mktb_mkt` | Bond Market Beta | Beta on MKTB from joint regression with MKTRF (36-month rolling). |
| `b_mktb` | Bond Market Beta (Univariate) | Beta from univariate regression on MKTB. |
| `b_mktbx_dcapm` | Duration-Adj Market Beta | Beta on MKTBX from duration-adjusted CAPM. |
| `b_term_dcapm` | Term Premium Beta | Beta on TERM = MKTB − MKTBX from duration-adjusted CAPM. |
| `b_mktb_dn` | Downside Market Beta | Beta on $\min(\text{MKTB}, 0)$ from asymmetric market model. |
| `b_mktb_up` | Upside Market Beta | Beta on $\max(\text{MKTB}, 0)$ from asymmetric market model. |
| `b_termb` | Term Beta | Beta on TERMB from market regression. |
| `db_mkt` | Daily Market Beta | Beta from within-month daily regression on cross-sectional mean return. |

---

### Cluster VII: Credit & Default Betas

| Mnemonic | Name | Description |
|----------|------|-------------|
| `b_drf` | Downside Risk Beta (Univariate) | Beta from univariate regression on DRF. |
| `b_crf` | Credit Risk Beta (Univariate) | Beta from univariate regression on CRF. |
| `b_lrf` | Liquidity Risk Beta (Univariate) | Beta from univariate regression on LRF. |
| `b_defb` | Default Beta | Beta on DEFB from duration-adjusted market regression. |

---

### Cluster VIII: Volatility & Liquidity Betas

| Mnemonic | Name | Description |
|----------|------|-------------|
| `b_dvix` | VIX Innovation Beta | Sum of contemporaneous and lagged ΔVIX betas from MKTB+MKTRF regression. |
| `b_dvix_va` | VIX Beta (Amihud) | ΔVIX beta from FF3+VIX+Amihud specification. |
| `b_dvix_vp` | VIX Beta (PSB) | ΔVIX beta from FF3+VIX+PSB specification. |
| `b_dvix_dn` | Downside VIX Beta | Beta on $\min(\Delta\text{VIX}, 0)$ from asymmetric VIX model. |
| `b_dvix_up` | Upside VIX Beta | Beta on $\max(\Delta\text{VIX}, 0)$ from asymmetric VIX model. |
| `b_psb` | Pastor-Stambaugh Beta | Beta on bond market liquidity factor PSB. |
| `b_psb_m` | PSB Beta (Multi-Factor) | PSB beta controlling for FF3+MKTBX+TERM. |
| `b_amd_m` | Amihud Beta (Multi-Factor) | Amihud beta controlling for FF3+MKTBX+TERM. |
| `b_amd` | Amihud Beta | Beta on aggregate Amihud illiquidity factor. |
| `b_coskew` | Coskewness Beta | Beta on $\text{MKTB}^2$ from coskewness regression. |
| `b_vix` | VIX Level Beta | Beta on VIX level from monthly regression. |
| `b_dvixd` | Daily VIX Innovation Beta | Beta on daily ΔVIX within the month. Requires a minimum of 5 daily returns. |
| `b_illiq` | Illiquidity Beta | Beta on aggregate bond market illiquidity factor. |

---

### Cluster IX: Macro & Other Betas

| Mnemonic | Name | Description |
|----------|------|-------------|
| `b_dunc` | Macro Uncertainty Beta | Beta on changes in Jurado, Ludvigson, and Ng (2015) macro uncertainty index. |
| `b_duncr` | Real Uncertainty Beta | Beta on changes in Jurado, Ludvigson, and Ng (2015) real uncertainty component. |
| `b_duncf` | Financial Uncertainty Beta | Beta on changes in Jurado, Ludvigson, and Ng (2015) financial uncertainty component. |
| `b_unc` | Uncertainty Level Beta | Beta on Jurado, Ludvigson, and Ng (2015) macro uncertainty level. |
| `b_dunc3` | 3-Month Uncertainty Change Beta | Beta on 3-month change in Jurado, Ludvigson, and Ng (2015) macro uncertainty. |
| `b_dunc6` | 6-Month Uncertainty Change Beta | Beta on 6-month change in Jurado, Ludvigson, and Ng (2015) macro uncertainty. |
| `b_dcpi` | Inflation Beta | Beta on monthly CPI changes. |
| `b_cpi_vol6` | Inflation Volatility Beta | Beta on 6-month rolling CPI volatility. |
| `b_dcredit` | Credit Spread Change Beta | Beta on monthly changes in BAA-AAA spread. |
| `b_credit` | Credit Spread Level Beta | Beta on BAA-AAA credit spread level. |
| `b_cptlt` | Intermediary Capital Beta | Beta on traded intermediary capital ratio. |
| `b_rvol` | Realized Volatility Beta | Beta on aggregate realized volatility factor. |
| `b_rsj` | Realized Jump Beta | Beta on aggregate realized signed jump factor. |
| `b_lvl` | Level Factor Beta | Beta on the average over key-rate U.S. Treasury yields. |
| `b_ysp` | Yield Spread Beta | Beta on yield spread factor. |
| `b_epu` | Economic Policy Uncertainty Beta | Beta on EPU index level. |
| `b_epum` | Monetary Policy Uncertainty Beta | Beta on monetary policy uncertainty index level. |
| `b_eput` | Tax Policy Uncertainty Beta | Beta on the tax policy uncertainty index level (Baker-Bloom-Davis categorical EPU, column "3. Taxes"). NOT trade policy -- a separate category in the same workbook, never read. |
---


## Redaction: the file you download is not the file the pipeline builds

Four columns differ between a panel **you build** from your own WRDS subscription and the panel
**published** at openbondassetpricing.com. The identifiers are proprietary and the agency ratings
are licensed, so redistribution is restricted -- building them yourself is not.

| column | in your build | in the published file |
|---|---|---|
| `permno` | populated | **populated** (kept) |
| `permco` | populated | null |
| `gvkey` | populated | null |
| `spc_rat` | 1..22 | 1 = investment grade, 11 = non-investment grade and default |
| `mdc_rat` | 1..22 | 1 = investment grade, 11 = non-investment grade and default |

A bond with no rating stays missing in both -- it is never silently labelled investment grade.

This is applied by `make_release.redact_for_publication`, and `make_release.assert_publishable`
refuses to package a frame that still carries full identifiers or raw ratings. Measured against
the released vintage: `permco` and `gvkey` are 100% null there, ratings take only {1, 11}, while
the raw build has them 86.6% / 86.4% populated on the 1..22 scale.

## Bond Returns

Monthly bond returns are computed from volume-weighted clean prices, accrued interest, and coupon payments. We compute two return types based on the measurement window.

---

### Month-End Return (`ret_vw`)

The month-end return measures performance from the end of month $t$ to the end of month $t+1$:

$$r_{i,t+1}^{End} = \frac{P_{i,t+1} + AI_{i,t+1} + C_{i,t+1}}{P_{i,t} + AI_{i,t}} - 1$$

where:
- $P_{i,t+1}$ = volume-weighted clean price at month-end $t+1$
- $AI_{i,t+1}$ = accrued interest at month-end $t+1$
- $C_{i,t+1}$ = coupon payment during month $t+1$ (if any)

**Validity Criterion:** A month-end return is valid only if the bond trades within the last 5 business days of both month $t$ and month $t+1$ (NYSE calendar).

```
                      MONTH-END RETURN TIMELINE
   ═══════════════════════════════════════════════════════════════════

          Month t                                Month t+1
   ┌──────────────────────┐               ┌──────────────────────┐
   │                      │               │                      │
   │              ┌─────┐ │               │              ┌─────┐ │
   │              │5 BD │ │               │              │5 BD │ │
   │              └──┬──┘ │               │              └──┬──┘ │
   │                 │    │               │                 │    │
   └─────────────────┼────┘               └─────────────────┼────┘
                     ▼                                      ▼
   ──────────────────●──────────────────────────────────────●──────▶ time
                   dt_s                                   dt_e
                     │                                      │
                     │◄──────────── hprd ──────────────────►│
                     │         (holding period)             │

   ● dt_s : Trade date in month t   (last 5 BD)
   ● dt_e : Trade date in month t+1 (last 5 BD)
```

| Variable | Description |
|----------|-------------|
| `ret_vw` | Month-end total return $r_{i,t+1}^{End}$ |
| `dt_s` | Trade date used for month $t$ price |
| `dt_e` | Trade date used for month $t+1$ price |
| `hprd` | NYSE trading sessions from `dt_s` to `dt_e`. Runs 15 to 27, because each trade may fall anywhere in the last 5 sessions of its month |

---

### Month-Begin Return (`ret_vw_bgn`)

The month-begin return measures performance within a single month, from the beginning to the end:

$$r_{i,t+1}^{Bgn} = \frac{P_{i,t+1}^{end} + AI_{i,t+1}^{end} + C_{i,t+1}}{P_{i,t+1}^{bgn} + AI_{i,t+1}^{bgn}} - 1$$

where:
- $P_{i,t+1}^{end}, AI_{i,t+1}^{end}$ = price and accrued interest from the last 5 business days of month $t+1$
- $P_{i,t+1}^{bgn}, AI_{i,t+1}^{bgn}$ = price and accrued interest from the first 5 business days of month $t+1$

**Validity Criterion:** A month-begin return is valid only if the bond trades within both the first 5 and last 5 business days of month $t+1$ (NYSE calendar).

```
                     MONTH-BEGIN RETURN TIMELINE
   ═══════════════════════════════════════════════════════════════════

                              Month t+1
   ┌──────────────────────────────────────────────────────────────┐
   │                                                              │
   │  ┌─────┐                                          ┌─────┐   │
   │  │5 BD │                                          │5 BD │   │
   │  └──┬──┘                                          └──┬──┘   │
   │     │                                                │      │
   └─────┼────────────────────────────────────────────────┼──────┘
         ▼                                                ▼
   ──────●────────────────────────────────────────────────●──────▶ time
      dt_s_bgn                                         dt_e_bgn
         │                                                │
         │◄─────────────── hprd_bgn ─────────────────────►│
         │              (holding period)                  │

   ● dt_s_bgn : Trade date at month-begin (first 5 BD of month t+1)
   ● dt_e_bgn : Trade date at month-end   (last 5 BD of month t+1)

   NOTE: Both prices measured within the SAME month t+1
```

| Variable | Description |
|----------|-------------|
| `ret_vw_bgn` | Month-begin total return $r_{i,t+1}^{Bgn}$ |
| `dt_s_bgn` | Trade date used for month-begin price |
| `dt_e_bgn` | Trade date used for month-end price |
| `hprd_bgn` | NYSE trading sessions from `dt_s_bgn` to `dt_e_bgn` |

---

### Implementation Gap (`igap_bgn`)

The implementation gap measures how long a trader waits, once the month has turned, for the first trade. On the row for month $t$, `igap_bgn` counts the NYSE sessions from the last session of month $t-1$ to the month-begin trade of month $t$ (`dt_s_bgn`). It runs 1 to 5, because the month-begin trade must fall in the first 5 sessions of the month.

It is measured from the last SESSION of month $t-1$, whether or not the bond traded that day. It is not measured from the bond's own month-end trade, which can sit up to 4 sessions earlier.

```
                        IMPLEMENTATION GAP TIMELINE
   ═══════════════════════════════════════════════════════════════════

        Month t-1                              Month t
   ┌───────────────────┐               ┌──────────────────────────────┐
   │                   │               │ ┌─────┐                      │
   │                   │               │ │5 BD │                      │
   │                   │               │ └──┬──┘                      │
   └───────────────────┤               └────┼─────────────────────────┘
                       ▼                    ▼
   ────────────────────●────────────────────●─────────────────────────▶
               last NYSE session         dt_s_bgn
                of month t-1                │
                       │◄─── igap_bgn ─────►│
                       │      (1 to 5)      │
                       │                    │
                 Month turns           Trade executed
                                      (month-begin t)
```

| Variable | Description |
|----------|-------------|
| `igap_bgn` | NYSE sessions from the last session of month $t-1$ to the month-begin trade of month $t$ (`dt_s_bgn`), 1 to 5 |

---

### Latent Implementation Bias (`lib`)

The **Latent Implementation Bias (LIB)** captures the cost of not being able to trade at the signal observation price.

**Motivation:** A representative trader who observes a price-based signal at the end of month $t$ cannot trade at that exact price. Instead, they must wait until the next trading opportunity—the first available price in month $t+1$. The month-begin return framework accounts for this by using the month-begin price as the entry point.

**LIB** is the clean price return between the signal observation price and the actual execution price:

$$\text{LIB}_{i,t+1} = \frac{P_{i,t+1}^{bgn}}{P_{i,t}} - 1$$

where:
- $P_{i,t}$ = month-end clean price at $t$ (signal observation price, denominator of $r^{End}$)
- $P_{i,t+1}^{bgn}$ = month-begin clean price at $t+1$ (execution price, denominator of $r^{Bgn}$)

**Which row it sits on.** In the panel the value is stored on the row for month $t$, the month of the end price it follows. So `lib` on a row dated month $t$ is the gap that comes AFTER that row's `ret_vw`, and it lines up with the NEXT row's `ret_vw_bgn`. `igap_bgn` runs the other way: on the row for month $t$ it looks back to the last session of month $t-1$.

```
             LATENT IMPLEMENTATION BIAS: SIGNAL TO EXECUTION
   ═══════════════════════════════════════════════════════════════════

        Month t                                Month t+1
   ┌───────────────────┐               ┌──────────────────────────────┐
   │           ┌─────┐ │               │ ┌─────┐              ┌─────┐ │
   │           │5 BD │ │               │ │5 BD │              │5 BD │ │
   │           └──┬──┘ │               │ └──┬──┘              └──┬──┘ │
   └──────────────┼────┘               └────┼───────────────────┼─────┘
                  ▼                         ▼                   ▼
   ───────────────●─────────────────────────●───────────────────●────▶
               P_{t}                   P_{t+1}^{bgn}       P_{t+1}^{end}
                  │                         │                   │
                  │◄──────── LIB ──────────►│                   │
                  │    (latent cost)        │                   │
                  │                         │                   │
                  │                         │◄─── r_{t+1}^{Bgn} ───►│
                  │                         │  (realized return)│
                  │                                             │
                  │◄─────────────── r_{t+1}^{End} ─────────────────►│
                  │           (includes LIB + r^{Bgn})          │

   ┌─────────────────────────────────────────────────────────────────┐
   │  DECOMPOSITION:  r_{t+1}^{End} ≈ LIB + r_{t+1}^{Bgn}            │
   │                                                                 │
   │  • LIB captures microstructure noise / bid-ask bounce          │
   │  • r_{t+1}^{Bgn} captures the "true" implementable return      │
   └─────────────────────────────────────────────────────────────────┘
```

**Interpretation:**
- **LIB > 0**: Price rose between signal observation and execution (adverse move for buyers)
- **LIB < 0**: Price fell between signal observation and execution (favorable move for buyers)
- Month-begin returns remove LIB, providing a cleaner measure of implementable returns

| Variable | Description |
|----------|-------------|
| `lib` | Latent Implementation Bias: $(P_{i,t+1}^{bgn} / P_{i,t}) - 1$ (clean price return), stored on the row for month $t$ |
| `libd` | Latent Implementation Bias Dirty: uses dirty prices (includes accrued interest), stored on the row for month $t$ |

---

### Signal Gap for Price-Based Signals

The **Signal Gap** addresses market microstructure noise in price-based signals. When forming portfolios based on signals derived from bond prices (yields, spreads, value measures), using the same price for both the signal and the return denominator introduces a mechanical relationship that can bias results.

To address this, we compute price-based signals using prices observed **at least 1 business day before** the month-end price used for returns. The variable `sig_gap` records the number of business days between signal observation (`sig_dt`) and the month-end price date (`dt_e`).

```
                   SIGNAL GAP FOR PRICE-BASED SIGNALS
   ═══════════════════════════════════════════════════════════════════

                              Month t
   ┌──────────────────────────────────────────────────────────────────┐
   │                                                                  │
   │                              ┌─────────────────────────┐         │
   │                              │  Last 10 BD of month    │         │
   │                              └─────────────────────────┘         │
   │                                                                  │
   │                       sig_dt              dt_e                   │
   │                         │                  │                     │
   │                         ▼                  ▼                     │
   └─────────────────────────●──────────────────●─────────────────────┘
                             │                  │
                             │◄─── sig_gap ────►│
                             │   (1-10 BD)      │
                             │                  │
                      Signal observed     Price for return
                     (price-based signal)  (denominator)

   ┌─────────────────────────────────────────────────────────────────┐
   │  REQUIREMENT: sig_gap ≥ 1 business day                         │
   │                                                                 │
   │  • Avoids using same price for signal and return calculation   │
   │  • Mitigates bid-ask bounce contamination                      │
   │  • sig_gap capped at 10 BD to ensure signal relevance          │
   └─────────────────────────────────────────────────────────────────┘
```

**MMN-adjusted signals in main panel:** The following variables are computed with the 1–10 BD signal gap and are provided as standard in the main TRACE panel downloads:

| Category | Variables |
|----------|-----------|
| Yields & Spreads | `ytm`, `md_dur`, `convx`, `cs`, `dcs6` |
| Value | `bbtm`, `val_hz`, `val_hz_dts`, `val_ipr`, `val_ipr_dts` |
| Size | `sze` |
| Short-term Reversal | `str` |
| Illiquidity | `pi`, `ami`, `ami_v`, `lix`, `ilq`, `roll`, `spd_abs`, `spd_rel`, `cs_sprd`, `ar_sprd`, `p_zro`, `p_fht`, `vov` |
| Within-Month Risk | `dvol`, `dskew`, `dkurt`, `dvol_sys`, `dvol_idio`, `rvol`, `rsj`, `rsk`, `rkt` |
| Within-Month Betas | `db_mkt`, `b_vix`, `b_dvixd` |

**Usage with main panel:** Use month-end returns (`ret_vw`) with the MMN-adjusted signals above.

**Noisy versions available separately:** The unadjusted (noisy) versions of all signals above are provided in `mmn_price_based_signals_YYYYmmdd.parquet` with the `_mmn` suffix (e.g., `ytm_mmn`, `cs_mmn`, `val_hz_mmn`, `ami_mmn`, etc.).

**Usage with noisy signals:** If using signals from `mmn_price_based_signals_YYYYmmdd.parquet`, you **must** use month-begin returns (`ret_vw_bgn`) to avoid microstructure bias. Both methods produce similar results.

| Variable | Description |
|----------|-------------|
| `sig_dt` | Date when price-based signal was observed (minimum 1 BD before `dt_e`) |
| `sig_gap` | Business days between signal observation and month-end price; ranges 1–10 BD |

---

### The Three Approaches

We compare portfolio performance using three approaches that differ in signal timing and return measurement:

**Approach 1: With Noise.**
Uses the month-end price $P_t^{\text{end}}$ in *both* the signal and return computation. Portfolio weights $\omega_t$ are computed from signals observed at $P_t^{\text{end}}$, and returns are the standard month-end returns $r_{t+1}^{\text{End}}$. This creates mechanical correlation between signal and return, potentially biasing factor performance upward.

**Approach 2: Adjusted Signal (Default in Main Panel).**
Uses signals computed with at least a 1-business day gap before $P_t^{\text{end}}$, breaking the mechanical correlation. Portfolio weights $\omega_t^{a}$ are computed from gapped signals at $P_{t-\Delta}^{s}$ where $\Delta \geq 1$ BD (max 10 BD). Returns remain the standard month-end returns $r_{t+1}^{\text{End}}$.

**Approach 3: Adjusted Return.**
Uses the same signal timing as Approach 1, but returns are computed from month-begin to month-end within month $t+1$. Portfolio weights $\omega_t$ are computed from $P_t^{\text{end}}$, but returns are the month-begin returns $r_{t+1}^{\text{Bgn}}$. This captures the *implementable* return for a trader observing a signal at month-end.

```
                    THE THREE APPROACHES
   ═══════════════════════════════════════════════════════════════════

   Approach 1: With Noise (Baseline)
   ─────────────────────────────────
   Signal:  P_t^{end}     ────►  ω_t
   Return:  r_{t+1}^{End}
   Issue:   Same price in signal and return denominator

   Approach 2: Adjusted Signal (Main Panel Default)
   ─────────────────────────────────────────────────
   Signal:  P_{t-Δ}^{s}   ────►  ω_t^a    (Δ ≥ 1 BD, max 10 BD)
   Return:  r_{t+1}^{End}
   Fix:     Signal observed before return price

   Approach 3: Adjusted Return
   ───────────────────────────
   Signal:  P_t^{end}     ────►  ω_t
   Return:  r_{t+1}^{Bgn}
   Fix:     Implementable return (trade at month-begin)
```

**Note:** Approaches 2 and 3 are both "cleaned" of microstructure noise, while Approach 1 is noisy. The main panel uses Approach 2 by default.

---

### Feasibility Bias

The **Feasibility Bias** quantifies the inflation in factor performance due to market microstructure noise. We compute two bias measures:

**Bias (1)−(2): Signal Adjustment.**

$$\text{Bias}_{1-2} = (\omega_t - \omega_t^{a}) \times r_{t+1}^{\text{End}}$$

This measures the bias from using different weights (noisy vs. gapped signal) while holding the return constant.

**Bias (1)−(3): Return Adjustment.**

$$\text{Bias}_{1-3} = \omega_t \times (r_{t+1}^{\text{End}} - r_{t+1}^{\text{Bgn}})$$

This measures the bias from using the same weights but different returns (month-end vs. month-begin).

**Interpretation:**
- Positive bias indicates the noisy approach (Approach 1) overstates factor performance
- The magnitude of bias varies by signal type—price-based signals (value, spreads) typically show larger bias than return-based signals (momentum, betas)
- Both bias measures should be small for well-constructed factors

---

### Excess and Duration-Adjusted Returns

Users can compute adjusted returns from the variables in the main panel:

**Excess returns** remove the risk-free rate:

$$r_{i,t+1}^{x} = r_{i,t+1} - r_{t+1}^{f}$$

Compute as: `ret_vw - rfret` (month-end) or `ret_vw_bgn - rfret` (month-begin)

**Duration-adjusted returns** remove systematic interest rate exposure by subtracting the return on a duration-matched U.S. Treasury bond:

$$r_{i,t+1}^{x} = r_{i,t+1} - r_{i,t+1}^{Tsy}$$

Compute as: `ret_vw - tret` (month-end) or `ret_vw_bgn - tret` (month-begin)

| Variable | Description |
|----------|-------------|
| `tret` | Duration-matched U.S. Treasury return; linearly interpolated from WRDS key rate Treasury returns using bond modified duration (`md_dur`) |
| `tret_bns` | Treasury benchmark from the bond's OWN cash flows, present-value (duration) weighted, on the Gurkaynak-Sack-Wright zero curve (van Binsbergen, Nozawa & Schwert) |
| `tret_cfm` | The same ladder weighted by future value instead of present value; overstates interest rate risk (their Internet Appendix A1) |
| `tret_gprs` | The exact duration match: present-value weights are duration-matched only on a flat curve, so the Treasury yield is solved for and the weights restated (Ghaderi, Plante, Roussanov & Seo, App. B.2) |
| `tret_cls` | The CASH-FLOW match: the same ladder discounted on the Treasury curve rather than the bond's own yield, so it also removes slope and curvature, not just parallel shifts (Cui, Lu & Song 2026) |
| `tret_mat` | `tret`'s twin -- same series, nodes and rounding, interpolated at MATURITY rather than modified duration (Bessembinder, Kahle, Maxwell & Xu 2009) |
| `rfret` | Monthly risk-free rate from Fama-French |

> **The four `tret_*` columns are benchmarks only — nothing else in the panel uses them.** Every
> duration-adjusted quantity we ship (`ret_vwx`, the 68 rolling beta and momentum columns, `str`,
> `var_95`, `es_95`) is built from `ret_vw - tret`, the incumbent duration-interpolated benchmark.
> Subtracting a different `tret_*` from `ret_vw` is left to you; no beta or momentum column has
> been re-estimated against one.

---

## Defaulted Bond Returns

When bonds enter or trade under default, coupon payments cease and the return formulas are adjusted accordingly. Default status is determined by credit ratings.

### Default Identification

We use the raw agency rating variables to identify default status:

| Rating Agency | Variable | Default Level |
|--------------|----------|---------------|
| S&P | `sp_rat` | 22 |
| Moody's | `mdy_rat` | 21 |

**A bond is considered in default if:**
- `sp_rat == 22` **OR** `mdy_rat == 21`

---

### Standard Return (Non-Default)

For bonds not in default, we use the standard total return formula with dirty prices:

$$r_{i,t+1}^{std} = \frac{P_{i,t+1}^{dirty} + C_{i,t+1} - P_{i,t}^{dirty}}{P_{i,t}^{dirty}}$$

where $P^{dirty} = P^{clean} + AI$ (clean price plus accrued interest).

```
                        STANDARD RETURN (NON-DEFAULT)
   ═══════════════════════════════════════════════════════════════════

        Month t                                Month t+1
   ┌───────────────────┐               ┌──────────────────────────────┐
   │    NOT DEFAULT    │               │         NOT DEFAULT          │
   │   (rating < 22)   │               │        (rating < 22)         │
   │                   │               │                              │
   │   P_{t}^dirty     │               │   P_{t+1}^dirty + Coupon     │
   └───────────────────┘               └──────────────────────────────┘
            │                                      │
            ▼                                      ▼
   ─────────●──────────────────────────────────────●──────────────────▶
         prfull_{t}                      prfull_{t+1} + coupon
            │                                      │
            └──────────── r_{t+1}^{std} ──────────►│

   ret_type = 'standard'
```

---

### Default Event Return

When a bond **transitions INTO default**, coupon payments cease immediately. The return compares the clean price at default to the prior dirty price:

$$r_{i,t+1}^{def} = \frac{P_{i,t+1}^{clean} - P_{i,t}^{dirty}}{P_{i,t}^{dirty}}$$

```
                         DEFAULT EVENT RETURN
   ═══════════════════════════════════════════════════════════════════

        Month t                                Month t+1
   ┌───────────────────┐               ┌──────────────────────────────┐
   │    NOT DEFAULT    │     ──▶      │          DEFAULT             │
   │   (rating < 22)   │   ENTERS     │        (rating = 22)         │
   │                   │   DEFAULT    │                              │
   │   P_{t}^dirty     │               │        P_{t+1}^clean         │
   │  (with accrued)   │               │   (NO accrued interest)      │
   └───────────────────┘               └──────────────────────────────┘
            │                                      │
            ▼                                      ▼
   ─────────●──────────────────────────────────────●──────────────────▶
         prfull_{t}                             pr_{t+1}
            │                                      │
            └──────────── r_{t+1}^{def} ──────────►│

   ┌─────────────────────────────────────────────────────────────────┐
   │  Coupon payments CEASE at default                               │
   │  Accrued interest drops to zero                                 │
   │  Return reflects full loss including forfeited accrued         │
   └─────────────────────────────────────────────────────────────────┘

   ret_type = 'default_evnt'
```

---

### Trading Under Default (Flat Return)

When a bond **remains in default**, there are no coupon accruals. Returns are based solely on clean price changes:

$$r_{i,t+1}^{flat} = \frac{P_{i,t+1}^{clean}}{P_{i,t}^{clean}} - 1$$

```
                       TRADING UNDER DEFAULT (FLAT)
   ═══════════════════════════════════════════════════════════════════

        Month t                                Month t+1
   ┌───────────────────┐               ┌──────────────────────────────┐
   │      DEFAULT      │               │          DEFAULT             │
   │   (rating = 22)   │    STAYS     │        (rating = 22)         │
   │                   │   DEFAULT    │                              │
   │    P_{t}^clean    │               │        P_{t+1}^clean         │
   │   (no accrued)    │               │       (no accrued)           │
   └───────────────────┘               └──────────────────────────────┘
            │                                      │
            ▼                                      ▼
   ─────────●──────────────────────────────────────●──────────────────▶
          pr_{t}                                pr_{t+1}
            │                                      │
            └─────────── r_{t+1}^{flat} ──────────►│

   ┌─────────────────────────────────────────────────────────────────┐
   │  No coupon payments while in default                            │
   │  P^dirty = P^clean (accrued interest = 0)                       │
   │  "Flat" trading: price-only returns                             │
   └─────────────────────────────────────────────────────────────────┘

   ret_type = 'trad_in_def'
```

---

### Default Return Summary

| Return Type | Condition | Formula | `ret_type` |
|-------------|-----------|---------|------------|
| Standard | Not in default | $(P_{t+1}^{dirty} + C_{t+1} - P_t^{dirty}) / P_t^{dirty}$ | `standard` |
| Default Event | Enters default | $(P_{t+1}^{clean} - P_t^{dirty}) / P_t^{dirty}$ | `default_evnt` |
| Trading Under Default | Remains in default | $P_{t+1}^{clean} / P_t^{clean} - 1$ | `trad_in_def` |

**Note:** The same logic applies to both month-end (`ret_vw`) and month-begin (`ret_vw_bgn`) returns.

---

## Technical Appendix

This appendix provides detailed methodology for factor models, beta estimation, and signal computation.

---

### Factor Models

Rolling betas are estimated using a 36-month rolling window with a minimum of 12 observations. For each factor model, betas are computed twice:

1. **Standard returns** (`ret_vw`): Value-weighted bond returns
2. **Duration-adjusted returns** (`ret_vwx`): `ret_vwx = ret_vw - tret`

where `tret` is the matched Treasury return based on bond duration.

| Model | Factors | Outputs Kept | SUM Operation | ivol |
|-------|---------|--------------|---------------|------|
| MKT | mktrf, mktb | all | - | Yes |
| BBW | mktb, drf, crf, lrf | - | - | Yes |
| DCAPM | mktbx, term | all | - | No |
| VOLAM | mktrf, smb, hml, vix, dvix, dvixlag, amd | dvix | dvix | No |
| VOLPSB | mktrf, smb, hml, vix, dvix, dvixlag, psb | dvix | dvix | Yes |
| PSBm | mktrf, smb, hml, mktbx, term, psb | psb | - | No |
| AMDM | mktrf, smb, hml, mktbx, term, amd | amd | - | No |
| VIX | mktb, mktrf, dvix, dvixlag | dvix | dvix | No |
| INFLV | mktb, cpi_vol6 | cpi_vol6 | - | No |
| UNC | mktb, dunc | dunc | - | No |
| UNCL | mktb, unc | unc | - | No |
| UNC3 | mktb, dunc3 | dunc3 | - | No |
| UNC6 | mktb, dunc6 | dunc6 | - | No |
| UNCr | mktb, duncr | duncr | - | No |
| UNCf | mktb, duncf | duncf | - | No |
| CREDd | mktb, dcredit | dcredit | - | No |
| CREDl | mktb, credit | credit | - | No |
| INFL | mktb, dcpi | dcpi | - | No |
| HKM | mktrf, cptlt | cptlt | - | No |
| RVOL | mktb, rvol | rvol | - | No |
| RSJ | mktb, rsj | rsj | - | No |
| PSB | mktb, psb | psb | - | No |
| AMD | mktb, amd | amd | - | No |
| ILLIQ | mktb, illiq | illiq | - | No |
| DVIX_ASYM | mktrf, dvix_down, dvix_up | dvix_down, dvix_up | - | No |
| COSKEW | mktb, mktb_sq | mktb_sq | - | No |
| DEFTERM | termb, defb | defb, termb | - | No |
| DRF | drf | all | - | No |
| CRF | crf | all | - | No |
| LRF | lrf | all | - | No |
| MKTB | mktb | all | - | No |
| LVL | lvl | all | - | No |
| YSP | ysp | all | - | No |
| MKTB_ASYM | mktb_down, mktb_up | all | - | No |
| EPU | mktb, epu | epu | - | No |
| EPUm | mktb, epum | epum | - | No |
| EPUt | mktb, eput | eput | - | No |

---

### Duration-Adjusted Factor Substitution

When computing betas with duration-adjusted returns (`ret_vwx`), bond market factors are replaced with their duration-adjusted equivalents:

| Standard Factor | Duration-Adjusted Factor |
|-----------------|--------------------------|
| `mktb` | `mktbx` |
| `drf` | `drfx` |
| `crf` | `crfx` |
| `lrf` | `lrfx` |

---

### Estimation Methodology

#### Rolling OLS Regression

For each bond $i$ at time $t$, betas are estimated using a rolling window:

$$r_{i,s} = \alpha_i + \sum_{k=1}^{K} \beta_{i,k} F_{k,s} + \varepsilon_{i,s}, \quad s \in [t-W+1, t]$$

where:
- $r_{i,s}$ = bond return (ret_vw or ret_vwx)
- $F_{k,s}$ = factor $k$ at time $s$
- $W$ = window size (36 months)
- Minimum observations = 12

#### Idiosyncratic Volatility

$$\text{ivol} = \sqrt{\frac{\sum_{s} \hat{\varepsilon}_{i,s}^2}{n - K - 1}}$$

where $n$ is the number of observations and $K$ is the number of factors.

---

### Model Specifications

#### MKT (Market Model)

$$r_{i,t} = \alpha + \beta^{rf} \cdot MKTRF_t + \beta^{b} \cdot MKTB_t + \varepsilon_{i,t}$$

**Outputs:** `b_mktrf_mkt`, `b_mktb_mkt`, `ivol_mkt`

#### BBW (Bai-Bali-Wen 4-Factor Model)

$$r_{i,t} = \alpha + \beta^{m} \cdot MKTB_t + \beta^{d} \cdot DRF_t + \beta^{c} \cdot CRF_t + \beta^{l} \cdot LRF_t + \varepsilon_{i,t}$$

**Output:** `ivol_bbw`

#### DCAPM (Duration-Adjusted CAPM)

Duration-based bond pricing model (van Binsbergen, Nozawa & Schwert, 2025):

$$r_{i,t}^x = \alpha + \beta^{mx} \cdot MKTBX_t + \beta^{term} \cdot TERM_t + \varepsilon_{i,t}$$

where $TERM_t = MKTB_t - MKTBX_t$ (duration premium)

**Outputs:** `b_mktbx_dcapm`, `b_term_dcapm`

#### VOLAM (Volatility with Amihud)

Volatility factor model (Chung, Wang & Wu, 2019):

$$r_{i,t} = \alpha + \beta^{rf} \cdot MKTRF_t + \beta^{smb} \cdot SMB_t + \beta^{hml} \cdot HML_t + \beta^{vix} \cdot VIX_t + \beta^{dvix} \cdot \Delta VIX_t + \beta^{dvixlag} \cdot \Delta VIX_{t-1} + \beta^{amd} \cdot AMD_t + \varepsilon_{i,t}$$

**Output:** `b_dvix_va` = $\beta^{dvix} + \beta^{dvixlag}$

#### VOLPSB (Volatility with Pastor-Stambaugh Bond)

Volatility factor model with bond liquidity (Chung, Wang & Wu, 2019):

$$r_{i,t} = \alpha + \beta^{rf} \cdot MKTRF_t + \beta^{smb} \cdot SMB_t + \beta^{hml} \cdot HML_t + \beta^{vix} \cdot VIX_t + \beta^{dvix} \cdot \Delta VIX_t + \beta^{dvixlag} \cdot \Delta VIX_{t-1} + \beta^{psb} \cdot PSB_t + \varepsilon_{i,t}$$

**Outputs:** `b_dvix_vp` = $\beta^{dvix} + \beta^{dvixlag}$, `ivol_vp`

#### PSBm (Pastor-Stambaugh Bond Multi-Factor Model)

Bond liquidity risk model (Lin, Wang & Wu, 2011):

$$r_{i,t} = \alpha + \beta^{rf} \cdot MKTRF_t + \beta^{smb} \cdot SMB_t + \beta^{hml} \cdot HML_t + \beta^{mx} \cdot MKTBX_t + \beta^{term} \cdot TERM_t + \beta^{psb} \cdot PSB_t + \varepsilon_{i,t}$$

**Output:** `b_psb_m`

#### AMDM (Amihud Multi-Factor Model)

Bond illiquidity risk model with Amihud factor:

$$r_{i,t} = \alpha + \beta^{rf} \cdot MKTRF_t + \beta^{smb} \cdot SMB_t + \beta^{hml} \cdot HML_t + \beta^{mx} \cdot MKTBX_t + \beta^{term} \cdot TERM_t + \beta^{amd} \cdot AMD_t + \varepsilon_{i,t}$$

**Output:** `b_amd_m`

#### VIX (VIX Innovation Model)

$$r_{i,t} = \alpha + \beta^{b} \cdot MKTB_t + \beta^{rf} \cdot MKTRF_t + \beta^{dvix} \cdot \Delta VIX_t + \beta^{dvixlag} \cdot \Delta VIX_{t-1} + \varepsilon_{i,t}$$

**Output:** `b_dvix` = $\beta^{dvix} + \beta^{dvixlag}$

#### Two-Factor Models

For models with a single factor of interest plus market control:

$$r_{i,t} = \alpha + \beta^{b} \cdot MKTB_t + \beta^{x} \cdot X_t + \varepsilon_{i,t}$$

| Model | $X_t$ | Output |
|-------|-------|--------|
| INFLV | CPI_VOL6 | `b_cpi_vol6` |
| UNC | dUNC | `b_dunc` |
| UNCr | dUNCr | `b_duncr` |
| UNCf | dUNCf | `b_duncf` |
| UNCl | UNC | `b_unc` |
| UNC3 | dUNC3 | `b_dunc3` |
| UNC6 | dUNC6 | `b_dunc6` |
| CREDd | dCREDIT | `b_dcredit` |
| CREDl | CREDIT | `b_credit` |
| INFL | dCPI | `b_dcpi` |
| RVOL | RVOL | `b_rvol` |
| RSJ | RSJ | `b_rsj` |
| PSB | PSB | `b_psb` |
| AMD | AMD | `b_amd` |
| ILLIQ | ILLIQ | `b_illiq` |

#### HKM (He-Kelly-Manela Intermediary Capital)

$$r_{i,t} = \alpha + \beta^{rf} \cdot MKTRF_t + \beta^{cptl} \cdot CPTLT_t + \varepsilon_{i,t}$$

**Output:** `b_cptlt`

#### ILLIQ (Aggregate Illiquidity)

$$r_{i,t} = \alpha + \beta^{b} \cdot MKTB_t + \beta^{illiq} \cdot ILLIQ_t + \varepsilon_{i,t}$$

**Output:** `b_illiq`

#### DVIX_ASYM (Asymmetric VIX Innovation)

Separates VIX innovations into positive and negative components:

$$r_{i,t} = \alpha + \beta^{rf} \cdot MKTRF_t + \beta^{dn} \cdot \Delta VIX^-_t + \beta^{up} \cdot \Delta VIX^+_t + \varepsilon_{i,t}$$

where:
- $\Delta VIX^-_t = \min(\Delta VIX_t, 0)$ (volatility decreases)
- $\Delta VIX^+_t = \max(\Delta VIX_t, 0)$ (volatility increases)

**Outputs:** `b_dvix_dn`, `b_dvix_up`

#### COSKEW (Coskewness Model)

Measures sensitivity to squared market returns (Harvey & Siddique, 2000):

$$r_{i,t} = \alpha + \beta^{b} \cdot MKTB_t + \beta^{sq} \cdot MKTB^2_t + \varepsilon_{i,t}$$

**Output:** `b_coskew` $= \beta^{sq}$

**Idiosyncratic skewness** (`iskew`) is computed as the rolling skewness of residuals $\hat{\varepsilon}_{i,t}$ from this regression, using the same 36-month window:

$$\text{iskew}_{i,t} = \frac{\sqrt{n(n-1)}}{n-2} \cdot \frac{\frac{1}{n}\sum_{s}(\hat{\varepsilon}_{i,s} - \bar{\varepsilon})^3}{\left[\frac{1}{n}\sum_{s}(\hat{\varepsilon}_{i,s} - \bar{\varepsilon})^2\right]^{3/2}}$$

#### Daily Within-Month VIX Betas

Computed from daily returns within each bond-month using simple OLS:

$$r_{i,t} = \alpha + \beta^{vix} \cdot VIX_t + \varepsilon_{i,t}$$
$$r_{i,t} = \alpha + \beta^{dvix} \cdot \Delta VIX_t + \varepsilon_{i,t}$$

**Outputs:** `b_vix`, `b_dvixd`

#### DEFTERM (Gebhardt, Hvidkjaer & Swaminathan, 2005; Fama & French, 1993)

The default and term premia enter **one** two-factor regression, which yields both loadings:

$$r_{i,t} = \alpha + \beta^{term} \cdot TERMB_t + \beta^{def} \cdot DEFB_t + \varepsilon_{i,t}$$

**Outputs:** `b_defb`, `b_termb`

The two premia are constructed as in Fama and French (1993):

$$DEFB_t = r^{\text{LT corp}}_t - r^{\text{LT govt}}_t \qquad TERMB_t = r^{\text{LT govt}}_t - r^{f}_t$$

where the long-term corporate leg is the value-weighted return on panel bonds with at least
10 years to maturity, and the long-term government leg is the CRSP 20-year key-rate Treasury
total return. Before 2002-08 both series are spliced from the published extended (pre-TRACE)
factor file, exactly as the other BBW factors are.

The two premia are negatively correlated (−0.45 over the full extended history, −0.50 in the
TRACE era), which is what makes them jointly estimable rather than two names for one series.

#### Univariate Factor Betas (Dickerson, Mueller & Robotti, 2023)

Univariate regressions isolating individual factor exposures:

$$r_{i,t} = \alpha + \beta \cdot F_t + \varepsilon_{i,t}$$

| Factor | Output |
|--------|--------|
| DRF | `b_drf` |
| CRF | `b_crf` |
| LRF | `b_lrf` |
| MKTB | `b_mktb` |

#### LVL / YSP (Koijen, Lustig & Van Nieuwerburgh, 2017)

Bond return decomposition into level and yield spread components. Both are estimated
**univariate**, with no market control:

**LVL Model:**
$$r_{i,t} = \alpha + \beta^{lvl} \cdot LVL_t + \varepsilon_{i,t}$$

**Output:** `b_lvl`

**YSP Model:**
$$r_{i,t} = \alpha + \beta^{ysp} \cdot YSP_t + \varepsilon_{i,t}$$

**Output:** `b_ysp`

#### MKTB_ASYM (Ang, Chen & Xing, 2006)

Asymmetric market beta model separating upside and downside market movements:

$$r_{i,t} = \alpha + \beta^{dn} \cdot MKTB^-_t + \beta^{up} \cdot MKTB^+_t + \varepsilon_{i,t}$$

where:
- $MKTB^-_t = \min(MKTB_t, 0)$ (market declines)
- $MKTB^+_t = \max(MKTB_t, 0)$ (market advances)

**Outputs:** `b_mktb_dn`, `b_mktb_up`

#### EPU (Baker, Bloom & Davis, 2016)

Economic Policy Uncertainty betas:

$$r_{i,t} = \alpha + \beta^{b} \cdot MKTB_t + \beta^{epu} \cdot EPU_t + \varepsilon_{i,t}$$

| Uncertainty Type | Factor | Output |
|------------------|--------|--------|
| Overall EPU | epu | `b_epu` |
| Monetary Policy | epum | `b_epum` |
| Taxes ("3. Taxes") | eput | `b_eput` |

---

### Factor Definitions

❗**A characteristic and a factor are different objects, and five of them share a name.**
`lix`, `roll`, `rsj`, `rvol` and `vov` each appear **twice** in this document: once as a
bond-level characteristic in the main panel, and once as a monthly factor series in the
table below. They are not the same number.

| | characteristic (main panel) | factor (this table) |
|---|---|---|
| grain | one value per bond-month | one value per month |
| construction | measured on the bond's own trades | equal-weighted mean across **USA-domiciled** bonds |
| units | a **level** — e.g. `roll` averages 0.79, `vov` 0.13, `spd_rel` 0.0068 | for ten of the twelve illiquidity factors, the monthly **CHANGE** in that mean — `ROLL` averages −0.009 and is positive only 45% of the time |

The differenced factors are `AMD`, `LIX`, `ILLIQ`, `ROLL`, `SPRD`, `CSS`, `ARS`, `FHTS`,
`VOV` and `RVOL`. The two that are **not** differenced, and are levels of the mean, are
`PSB` and `RSJ`. (Verified against the build: `max|factor − diff(level)| = 0` for the ten,
and `max|factor − level| = 0` for the two.)

Differencing is deliberate. An illiquidity level is highly persistent, so a beta estimated
on the level would load mostly on the trend; the innovation is the priced quantity.

| Factor | Description | Source |
|--------|-------------|--------|
| `mktrf` | Equity market excess return | FF |
| `smb` | Small minus Big | FF |
| `hml` | High minus Low | FF |
| `mktb` | Bond market excess return (VW) | BBW |
| `mktbx` | Duration-adjusted bond market return | BBW |
| `mktb_sq` | Squared bond market return | Computed |
| `drf` | Downside risk factor | BBW |
| `crf` | Credit risk factor | BBW |
| `lrf` | Liquidity risk factor | BBW |
| `drfx` | Downside risk factor, duration-adjusted | BBW |
| `crfx` | Credit risk factor, duration-adjusted | BBW |
| `lrfx` | Liquidity risk factor, duration-adjusted | BBW |
| `term` | Term premium (MKTB - MKTBX) | Computed |
| `mktb_bns` | Bond market excess return, on the `tret_bns` benchmark | BBW twin |
| `drf_bns` | Downside risk factor, on the `tret_bns` benchmark | BBW twin |
| `crf_bns` | Credit risk factor, on the `tret_bns` benchmark | BBW twin |
| `lrf_bns` | Liquidity risk factor, on the `tret_bns` benchmark | BBW twin |
| `term_bns` | Term premium against the `tret_bns` benchmark | Computed |
| `mktb_cls` | Bond market excess return, on the `tret_cls` benchmark | BBW twin |
| `drf_cls` | Downside risk factor, on the `tret_cls` benchmark | BBW twin |
| `crf_cls` | Credit risk factor, on the `tret_cls` benchmark | BBW twin |
| `lrf_cls` | Liquidity risk factor, on the `tret_cls` benchmark | BBW twin |
| `term_cls` | Term premium against the `tret_cls` benchmark | Computed |
| `vix` | VIX level (scaled: /100/sqrt(12)) | CBOE |
| `dvix` | VIX first difference | Computed |
| `dvixlag` | Lagged VIX first difference | Computed |
| `dvix_down` | Negative VIX innovations: min(dvix, 0) | Computed |
| `dvix_up` | Positive VIX innovations: max(dvix, 0) | Computed |
| `cpi_vol6` | 6-month rolling CPI volatility | Computed |
| `dcpi` | CPI change | BLS |
| `credit` | Credit spread (BAA - AAA) / 12 | FRED |
| `dcredit` | Credit spread change | Computed |
| `unc` | Macro uncertainty index | JLN |
| `dunc` | Macro uncertainty change | JLN |
| `duncr` | Real uncertainty change | JLN |
| `duncf` | Financial uncertainty change | JLN |
| `uncr` | Real uncertainty index (level) | JLN |
| `uncf` | Financial uncertainty index (level) | JLN |
| `dunc3` | Macro uncertainty, 3-month change: `unc.diff(3)` | Computed |
| `dunc6` | Macro uncertainty, 6-month change: `unc.diff(6)` | Computed |
| `cptlt` | Intermediary capital: the value-weighted investment return, in excess of `rf` | HKM |
| `cptl` | Intermediary capital risk factor (the non-traded HKM factor) | HKM |
| `rvol` | Realized volatility | Computed |
| `rsj` | Realized skewness (scaled: /100) | Computed |
| `psb` | Pastor-Stambaugh bond liquidity | Computed |
| `amd` | Amihud illiquidity | Computed |
| `illiq` | Aggregate illiquidity factor | Computed |
| `ars` | Abdi-Ranaldo spread factor: monthly **change** in the equal-weighted mean of `ar_sprd` across USA-domiciled bonds | Computed |
| `css` | Corwin-Schultz spread factor: monthly **change** in the equal-weighted mean of `cs_sprd` across USA-domiciled bonds | Computed |
| `fhts` | FHT spread factor: monthly **change** in the equal-weighted mean of `p_fht` across USA-domiciled bonds | Computed |
| `sprd` | Bid-ask spread factor: monthly **change** in the equal-weighted mean of `spd_rel` across USA-domiciled bonds | Computed |
| `defb` | Default premium factor | GHS |
| `termb` | Term premium factor | GHS |
| `lvl` | Level factor (yield curve level) | KLN |
| `ysp` | Yield spread factor | KLN |
| `rf` | Risk-free rate, from the Ken French monthly factor file | Ken French |
| `mktb_down` | Negative bond market return: min(mktb, 0) | Computed |
| `mktb_up` | Positive bond market return: max(mktb, 0) | Computed |
| `epu` | Economic policy uncertainty index level | BBD |
| `epum` | Monetary policy uncertainty index level | BBD |
| `eput` | Trade policy uncertainty index level | BBD |

❗**The ten `_bns` / `_cls` series are OPTIONAL and only exist if you ran
`stage2/make_excess_blocks.py --benchmark {bns,cls,all}`.** They are the BBW bond-market factors
re-estimated on the alternative Treasury benchmark's excess return, and they exist because a
benchmark's betas must regress on *that* benchmark's own factors. Pairing a `tret_bns` excess
return with the `tret`-based factors produces `b_*` columns that look perfectly normal and mean
nothing -- which is why `compute_all_betas` raises on a missing twin rather than falling back.
`term` is swapped alongside the four bond factors for the same reason. A default build does not
write them and nothing downstream requires them.

---

### Momentum and Reversal Signals

All momentum and reversal signals are computed from monthly bond returns. Signals follow the L-S convention: sum returns from month $t-L+1$ to $t-S$ (skipping the most recent $S$ months).

#### Standard Momentum

Cumulative past returns over various horizons (Gebhardt, Hvidkjaer & Swaminathan, 2005):

$$\text{mom}_{L,S} = \sum_{s=S}^{L-1} r_{i,t-s}$$

| Variable | Formula | Description |
|----------|---------|-------------|
| `mom3_1` | $\sum_{s=1}^{2} r_{t-s}$ | 3-month momentum, skip 1 |
| `mom6_1` | $\sum_{s=1}^{5} r_{t-s}$ | 6-month momentum, skip 1 |
| `mom9_1` | $\sum_{s=1}^{8} r_{t-s}$ | 9-month momentum, skip 1 |
| `mom12_1` | $\sum_{s=1}^{11} r_{t-s}$ | 12-month momentum, skip 1 |

#### Intermediate Momentum

Momentum computed from months 7-12, excluding recent returns (Novy-Marx, 2012):

$$\text{mom12-7}_{i,t} = \sum_{s=7}^{11} r_{i,t-s}$$

| Variable | Description |
|----------|-------------|
| `mom12_7` | Intermediate momentum (months 7-12) |

#### Systematic and Idiosyncratic Momentum

Decomposes total momentum into market-driven and bond-specific components using residuals from a rolling factor model (Blitz, Huij & Martens, 2011):

**Methodology:** At each month $t$ with valid rolling OLS estimates ($\alpha_t$, $\beta_t$):

1. **Fitted values**: $\hat{r}_s = \alpha_t + \beta_t \cdot F_s$ for each $s$ in the rolling window
2. **Systematic momentum**: $\sum_{s=S}^{L-1} \hat{r}_{t-s}$ (sum of fitted values)
3. **Idiosyncratic momentum**: $\sum_{s=S}^{L-1} (r_{t-s} - \hat{r}_{t-s})$ (sum of residuals)

Key: Uses **current** $\alpha_t$, $\beta_t$ to compute fitted values for all positions.

| Variable | Description |
|----------|-------------|
| `sysmom3_1` | Systematic 3-1 momentum |
| `sysmom6_1` | Systematic 6-1 momentum |
| `sysmom12_1` | Systematic 12-1 momentum |
| `idimom3_1` | Idiosyncratic 3-1 momentum |
| `idimom6_1` | Idiosyncratic 6-1 momentum |
| `idimom12_1` | Idiosyncratic 12-1 momentum |

Default factor: `mktb` (bond market excess return). For duration-adjusted returns, uses `mktbx`.

#### Long-Term Reversal

Cumulative past returns over longer horizons, excluding recent months:

**Primary measure** (Bali, Subrahmanyam & Wen, 2021):

$$\text{ltr48-12}_{i,t} = \sum_{s=12}^{47} r_{i,t-s}$$

**Alternative windows** (Subrahmanyam, 2023):

| Variable | Formula | Description |
|----------|---------|-------------|
| `ltr48_12` | $\sum_{s=12}^{47} r_{t-s}$ | 48-month LTR, skip 12 |
| `ltr30_6` | $\sum_{s=6}^{29} r_{t-s}$ | 30-month LTR, skip 6 |
| `ltr24_3` | $\sum_{s=3}^{23} r_{t-s}$ | 24-month LTR, skip 3 |

#### Industry Momentum and Reversal

Cross-bond momentum spillovers within the same industry (Wang, Wu & Yang, 2024). For each bond, computes the equal-weighted average momentum/reversal of other bonds in the same industry:

$$\text{imom}_{i,t} = \frac{1}{N_{ind}-1} \sum_{j \in \text{ind}(i), j \neq i} \text{mom}_{j,t}$$

| Variable | Description |
|----------|-------------|
| `imom1` | Industry momentum (1-month) |
| `imom3_1` | Industry momentum (3-1) |
| `imom12_1` | Industry momentum (12-1) |
| `iltr48_12` | Industry long-term reversal (48-12) |
| `iltr30_6` | Industry long-term reversal (30-6) |
| `iltr24_3` | Industry long-term reversal (24-3) |

#### Short-Term Reversal and Risk Measures

Downside risk measures computed from daily returns within the month (Bai, Bali & Wen, 2019):

**Short-Term Reversal:**

| Variable | Description |
|----------|-------------|
| `str` | Short-term reversal (past month return) |

**Value-at-Risk and Expected Shortfall:**

$$\text{VaR}_\alpha = -F^{-1}(\alpha)$$

$$\text{ES}_\alpha = -\mathbb{E}[r \mid r < -\text{VaR}_\alpha]$$

where $F^{-1}$ is the empirical quantile function of daily returns within the month.

| Variable | Description |
|----------|-------------|
| `var_90` | 90% Value-at-Risk (10th percentile loss) |
| `var_95` | 95% Value-at-Risk (5th percentile loss) |
| `es_90` | 90% Expected Shortfall (mean of worst 10% returns) |

---

### Value Signals

Value signals measure the deviation of observed credit spreads from "fair" spreads implied by bond characteristics. All regressions use log spreads following Gilchrist & Zakrajšek (2012).

#### Monthly Cross-Sectional Spread Regression

For each month $t$, estimate:

$$\log(cs_{i,t}) = x_{i,t}^\top \beta_t + \varepsilon_{i,t}$$

where $x_{i,t}$ includes rating controls, industry dummies and the model's own characteristics, listed under each model below. The fitted "fair" spread uses a lognormal retransformation:

$$\widehat{cs}_{i,t} = \exp\left(x_{i,t}^\top \widehat{\beta}_t + \tfrac{1}{2}\widehat{\sigma}_t^2\right)$$

#### Houweling & Van Zundert (2017) Value

Percentage deviation from fitted spread:

$$\text{val-hz}_{i,t} = \frac{cs_{i,t} - \widehat{cs}_{i,t}}{\widehat{cs}_{i,t}}$$

**Controls:** Rating dummies (`spc_rat`), FF17 industry dummies, 3-month spread change, call indicator. The regression carries no maturity control (`lib/value.py`, `x_cols=['dcs3', 'call']`).

| Variable | Description |
|----------|-------------|
| `val_hz` | Value signal (% deviation from fair spread) |
| `val_hz_dts` | Value signal with DtS adjustment |

#### Israel, Palhares & Richardson (2018) Value

Log-spread residual from fair-value regression:

$$\text{val-ipr}_{i,t} = \log(cs_{i,t}) - x_{i,t}^\top \widehat{\beta}_t$$

**Controls:** Numeric rating, FF17 industry dummies, log duration, excess return volatility, call indicator.

| Variable | Description |
|----------|-------------|
| `val_ipr` | Value signal (log-spread residual) |
| `val_ipr_dts` | Value signal with DtS adjustment |

#### DtS Risk Adjustment

The Duration-times-Spread (DtS) adjustment removes common variation among bonds with similar spread-duration exposure (Israel, Palhares & Richardson, 2018).

**Step 1:** Compute duration-times-spread: $DtS_{i,t} = dur_{i,t} \times cs_{i,t}$

**Step 2:** Assign to quintiles $Q_{i,t} \in \{1,\ldots,5\}$ based on cross-sectional DtS distribution.

**Step 3:** Demean within (month, quintile):

$$val^{dts}_{i,t} = val_{i,t} - \bar{val}_{t,Q_{i,t}}$$

This isolates bond-specific value after controlling for systematic variation in spread-duration risk.

---

### Illiquidity Measures

All illiquidity measures are computed at the bond-month level from daily TRACE data. Each measure has two versions:
- **Full-sample**: Uses all trades in the month
- **Adjusted**: Excludes the last trade of each month to avoid end-of-month effects

Minimum observation requirement: 5 valid daily observations per bond-month (configurable).

#### Price Impact (PI)

Pastor & Stambaugh (2003) liquidity measure based on return reversals. The coefficient $\gamma$ captures the price impact of signed order flow:

**Regression:**
$$r^e_{i,t+1} = \theta + \psi \cdot r_{i,t} + \gamma \cdot \mathrm{sign}(r^e_{i,t}) \cdot Q_{i,t} + \varepsilon_{i,t}$$

where:
- $r^e_{i,t} = r_{i,t} - \bar{r}_t$ is the bond's excess return over the market index
- $Q_{i,t}$ is trading volume at day $t$

**Output:** `pi` $= -\gamma$ (negated so higher values indicate more illiquidity)

#### Amihud Illiquidity (AMI)

Price impact per unit of trading volume (Amihud, 2002).

$$\text{ami}_{i,m} = \frac{1}{N_m} \sum_{t=1}^{N_m} \frac{|r_{i,t}|}{\text{dvol}_{i,t}}$$

**Outputs:**
- `ami`: Monthly mean of daily Amihud ratios
- `ami_v`: Monthly standard deviation of daily Amihud ratios

#### LIX (Liquidity Index)

Combines trading volume and price range into a single liquidity measure (Danyliv, Bland & Nicholass, 2014).

$$\text{lix}_{i,t} = \log_{10}\left(\frac{V_{i,t} \cdot P^{close}_{i,t}}{P^{high}_{i,t} - P^{low}_{i,t}}\right)$$

Monthly `lix` is the mean of daily values.

#### Roll Spread (ILQ, ROLL)

Implicit bid-ask spread from return autocovariance (Roll, 1984).

$$\text{ilq}_{i,m} = -\text{Cov}(\ln(1+r_{i,t}), \ln(1+r_{i,t-1}))$$

$$\text{roll}_{i,m} = \begin{cases} 2\sqrt{\text{ilq}_{i,m}} & \text{if } \text{ilq}_{i,m} > 0 \\ 0 & \text{otherwise} \end{cases}$$

**Outputs:**
- `ilq`: Negative autocovariance of log returns (scaled by 100)
- `roll`: Roll effective spread estimate

#### Hong-Warga Spreads (SPD_ABS, SPD_REL)

Volume-weighted bid-ask spreads using quoted prices.

$$P^{bid}_m = \frac{\sum_t w_t \cdot P^{bid}_t}{\sum_t w_t}, \quad P^{ask}_m = \frac{\sum_t w_t \cdot P^{ask}_t}{\sum_t w_t}$$

where $w_t = \text{dvol}_t$ (dollar volume weights).

**Outputs:**
- `spd_abs`: Absolute spread $= P^{ask}_m - P^{bid}_m$
- `spd_rel`: Relative spread $= (P^{ask}_m - P^{bid}_m) / \text{mid}_m$

#### Corwin-Schultz Spread (CS_SPRD)

High-low spread estimator using two-day price ranges (Corwin & Schultz, 2012).

$$\beta = [\ln(H_t/L_t)]^2 + [\ln(H_{t-1}/L_{t-1})]^2$$

$$\gamma = \left[\ln\left(\frac{\max(H_t, H_{t-1})}{\min(L_t, L_{t-1})}\right)\right]^2$$

$$\alpha = \frac{\sqrt{2\beta} - \sqrt{\beta}}{3 - 2\sqrt{2}} - \sqrt{\gamma}$$

$$\text{cs}_{i,t} = \frac{2(e^\alpha - 1)}{1 + e^\alpha}$$

Monthly `cs_sprd` is the mean of daily estimates. Negative alphas are set to zero.

#### Abdi-Ranaldo Spread (AR_SPRD)

Closing price spread estimator (Abdi & Ranaldo, 2017).

$$\eta_t = \frac{1}{2}(\ln H_t + \ln L_t)$$

$$s^2 = 4\left[\ln C_t - \frac{1}{2}(\eta_t + \eta_{t-1})\right]^2 - (\eta_t - \eta_{t-1})^2$$

$$\text{ar}_{i,t} = \sqrt{\max(s^2, 0)}$$

Monthly `ar_sprd` is the mean of daily estimates.

#### No-Trade Day Proportion (P_ZRO)

Fraction of potential trading days with no valid price (Fong, Holden & Trzcinka, 2017).

$$\text{p-zro}_{i,m} = \frac{B_m - N^{price}_{i,m}}{B_m}$$

where $B_m$ is the number of NYSE business days in month $m$ and $N^{price}_{i,m}$ is the count of days with valid prices.

#### FHT Spread (P_FHT)

Spread estimate derived from the proportion of zero-return days (Fong, Holden & Trzcinka, 2017).

$$\text{p-fht}_{i,m} = 2 \sigma_{i,m} \cdot \Phi^{-1}\left(\frac{1 + \text{p-zro}_{i,m}}{2}\right)$$

where $\sigma_{i,m}$ is the within-month return volatility and $\Phi^{-1}$ is the inverse standard normal CDF.

#### Volatility-over-Volume (VOV)

Liquidity proxy combining return volatility and trading volume (Tobek, 2016).

$$\text{vov}_{i,m} = 2.5 \cdot \frac{\sigma_{i,m}^{0.60}}{\bar{V}_{i,m}^{0.25}}$$

where $\sigma_{i,m}$ is the standard deviation of daily returns and $\bar{V}_{i,m}$ is the mean daily dollar volume.

---

### Within-Month Risk Statistics

Computed from daily returns within each bond-month $m$. Uses the cross-sectional mean of daily bond returns as the market factor.

**Data Requirements:** All within-month statistics require:

1. **Minimum observations:** $N_{i,m} \geq 5$ valid daily returns
2. **Valid return criterion:** A daily return $r_{i,t}$ is valid only if the business-day gap satisfies:
   $$\Delta_{i,t} = \text{BusDays}(t-1, t) \leq 5$$
3. **Calendar:** NYSE trading calendar

These requirements apply to all signals computed from daily data within the month (illiquidity measures, realized moments, within-month betas, etc.).

#### Daily Volatility and Higher Moments

| Variable | Description |
|----------|-------------|
| `dvol` | Standard deviation of daily returns within the month |
| `dskew` | Central skewness: $\frac{1}{N}\sum(r_t - \bar{r})^3 / \sigma^3$ |
| `dkurt` | Excess kurtosis: $\frac{1}{N}\sum(r_t - \bar{r})^4 / \sigma^4 - 3$ |

#### Daily Market Beta and Volatility Decomposition

From within-month regression: $r_{i,d} = \alpha + \beta \cdot \bar{r}_{d} + \varepsilon_{i,d}$

| Variable | Description |
|----------|-------------|
| `db_mkt` | Market beta: $\text{Cov}(r_i, \bar{r}) / \text{Var}(\bar{r})$ |
| `dvol_sys` | Systematic volatility: $|\beta| \cdot \sigma_{\bar{r}}$ |
| `dvol_idio` | Idiosyncratic volatility: $\sqrt{\sigma_r^2 - \beta^2 \sigma_{\bar{r}}^2}$ |

#### Realized Moments

Based on squared returns (no mean adjustment).

| Variable | Formula | Description |
|----------|---------|-------------|
| `rvol` | $\sqrt{\sum r_t^2}$ | Realized volatility |
| `rsj` | $(RV^+ - RV^-) / RV$ where $RV^+ = \sqrt{\sum_{r>0} r_t^2}$ | Realized signed jump variation |
| `rsk` | $\sqrt{N} \sum r_t^3 / (\sum r_t^2)^{3/2}$ | Realized skewness |
| `rkt` | $N \sum r_t^4 / (\sum r_t^2)^2$ | Realized kurtosis |

---

### Collinearity Notes

None of the 145 columns duplicates another: across all 9,045 numeric pairs, **no two are
bit-identical**, and the test suite asserts that. But several are near-substitutes, and
anyone forming clusters, running a horse race or building a factor zoo should know which.
Measured on the 2026 panel (1.95 M bond-months):

| pair | \|corr\| | why they are close |
|---|---|---|
| `mcap_e` / `sze` | 0.9998 | both are market value; `sze` is the panel's size variable and `mcap_e` the end-of-month market cap |
| `mcap_s` / `sze` | 0.9981 | as above, `mcap_s` measured at the signal date |
| `mcap_s` / `mcap_e` | 0.9979 | the same quantity a few business days apart |
| `fce_val` / `mcap_e` | 0.9817 | face value against market value; they differ only through the price |
| `dvol` / `dvol_idio` | 0.9817 | total daily volatility is mostly idiosyncratic for corporate bonds |
| `b_mktbx_dcapm` / `b_defb` | **0.9750** | see below |
| `ivol_mkt` / `ivol_bbw` | 0.9708 | residual volatility from two nested market models |

❗**`b_defb` and `b_mktbx_dcapm`.** These correlate 0.97, which is high enough to matter if you
put both in the same regression or treat them as separate cluster members. This is economically
sensible — bonds with high market betas are bonds with high default betas. For reference,
`b_defb` correlates 0.87 with `b_mktb`.

Within the Credit and Default Betas cluster the loadings are related but distinct:

| | `b_drf` | `b_crf` | `b_lrf` | `b_defb` | `b_termb` |
|---|---|---|---|---|---|
| `b_drf` | 1.00 | 0.37 | 0.79 | 0.88 | 0.67 |
| `b_crf` | | 1.00 | 0.39 | 0.53 | −0.04 |
| `b_lrf` | | | 1.00 | 0.73 | 0.54 |
| `b_defb` | | | | 1.00 | 0.64 |
| `b_termb` | | | | | 1.00 |

---

### The MMN Sidecar

`mmn_price_based_signals_<YYYY>.parquet` is published alongside the main panel and holds the
**unadjusted twin** of every price-based signal. Two keys (`cusip`, `date`) and 38 signals.

The distinction matters and is easy to get wrong:

| | main panel | `_mmn` sidecar |
|---|---|---|
| the signal | market-microstructure-noise **adjusted**: observed with at least a one-business-day gap before the month-end price used for the return | **unadjusted**: signal and return share the same closing print |
| the return to use with it | `ret_vw` | `ret_vw_bgn` |
| in the three-approaches language | Approach 2, *Adjusted Signal* | Approach 3, *Adjusted Return* |

❗**Do not mix them.** Pairing an unadjusted signal with `ret_vw` puts the same bid-ask bounce in
both the signal and the return, and manufactures predictability that is not there. The effect is
large, not marginal: for a short-term reversal signal built both ways, the bond-level AR(1) is
−0.219 unadjusted against −0.046 adjusted — the gap removes about 79% of the apparent reversal.

`str` is the one column where the naming inverts: `main_panel.str` is the MMN-**adjusted**
reversal signal, while `str_mmn` in the sidecar equals `ret_vw`.

To use the sidecar, merge it on `cusip` and `date` and switch the return column:

```python
panel = pd.read_parquet("main_panel_<YYYY>.parquet")
mmn   = pd.read_parquet("mmn_price_based_signals_<YYYY>.parquet")
df    = panel.merge(mmn, on=["cusip", "date"], how="left")
# now sort on e.g. df["cs_mmn"] and hold df["ret_vw_bgn"], never df["ret_vw"]
```

Every mnemonic below is the panel column of the same name without the suffix, computed without
the signal gap. The definitions are therefore the base signal's; see the tables above.

| Mnemonic | Base signal | Name |
|----------|-------------|------|
| `ami_mmn` | `ami` | Amihud Illiquidity |
| `ami_v_mmn` | `ami_v` | Amihud Volatility |
| `ar_sprd_mmn` | `ar_sprd` | Abdi-Ranaldo Spread |
| `b_dvixd_mmn` | `b_dvixd` | Daily VIX Innovation Beta |
| `b_vix_mmn` | `b_vix` | VIX Level Beta |
| `bbtm_mmn` | `bbtm` | Bond Book-to-Market |
| `convx_mmn` | `convx` | Convexity |
| `cs_mmn` | `cs` | Credit Spread |
| `cs_mu12_1_mmn` | `cs_mu12_1` | 12-Month Average Spread |
| `cs_sprd_mmn` | `cs_sprd` | Corwin-Schultz Spread |
| `db_mkt_mmn` | `db_mkt` | Daily Market Beta |
| `dcs6_mmn` | `dcs6` | 6-Month Spread Narrowing (log) |
| `dkurt_mmn` | `dkurt` | Daily Kurtosis |
| `dskew_mmn` | `dskew` | Daily Skewness |
| `dvol_idio_mmn` | `dvol_idio` | Idiosyncratic Volatility |
| `dvol_mmn` | `dvol` | Daily Volatility |
| `dvol_sys_mmn` | `dvol_sys` | Systematic Volatility |
| `ilq_mmn` | `ilq` | Roll Autocovariance |
| `lix_mmn` | `lix` | Negative LIX (Illiquidity) |
| `md_dur_mmn` | `md_dur` | Modified Duration |
| `p_fht_mmn` | `p_fht` | FHT Spread |
| `p_zro_mmn` | `p_zro` | No-Trade Day Proportion |
| `pi_mmn` | `pi` | Price Impact |
| `rkt_mmn` | `rkt` | Realized Kurtosis |
| `roll_mmn` | `roll` | Roll Spread |
| `rsj_mmn` | `rsj` | Realized Signed Jump |
| `rsk_mmn` | `rsk` | Realized Skewness |
| `rvol_mmn` | `rvol` | Realized Volatility |
| `spd_abs_mmn` | `spd_abs` | Absolute Bid-Ask Spread |
| `spd_rel_mmn` | `spd_rel` | Relative Bid-Ask Spread |
| `str_mmn` | `str` | Short-Term Reversal |
| `sze_mmn` | `sze` | Bond Size |
| `val_hz_dts_mmn` | `val_hz_dts` | Value (HZ, DtS-adjusted) |
| `val_hz_mmn` | `val_hz` | Value (HZ) |
| `val_ipr_dts_mmn` | `val_ipr_dts` | Value (IPR, DtS-adjusted) |
| `val_ipr_mmn` | `val_ipr` | Value (IPR) |
| `vov_mmn` | `vov` | Volatility-over-Volume |
| `ytm_mmn` | `ytm` | Yield to Maturity |

---

### Output Files

Released artifacts carry the **vintage year** `<YYYY>` — the year the release is published
under, not the last month of data. A build in progress writes files named for its Stage 1 date
stamp instead; the release step renames them.

| File | Description |
|------|-------------|
| `main_panel_<YYYY>.parquet` | Main panel with MMN-adjusted price-based signals — the 145 columns defined in this document |
| `mmn_price_based_signals_<YYYY>.parquet` | Unadjusted twins of the price-based signals, suffix `_mmn` (see below) |
| `betas_x_<YYYY>.parquet` | Betas from duration-adjusted returns ($r^x = r - r^{Tsy}$) |
| `mom_retx_<YYYY>.parquet` | Momentum/LTR signals from duration-adjusted returns |
| `returns_alt_<YYYY>.parquet` | Alternative return measures |
| `factors_<YYYY>.parquet` | The exogenous monthly factor panel every rolling beta was estimated on |

**Why the factor panel is published.** Stage 2 assembles its factors from public sources at
build time, and those sources revise: Ken French restates SMB/HML, FRED re-seasonally-adjusts
CPI, and the Ludvigson uncertainty series is re-estimated each release. A panel built next month
will not reproduce a panel published today. Shipping the exact factor file a release consumed is
what lets a published number be checked — point Stage 2 at it with `FACTOR_SOURCE = "pinned"`.

❗In `factors_<YYYY>.parquet` the columns `mktb`, `mktbx`, `term`, `drf`, `crf`, `drfx`, `crfx`,
`defb` and `termb` are the **published extended (pre-TRACE) series**, which end in 2023-01. They
are not the bond-market factors the panel uses: Stage 2 drops them on read and rebuilds them from
its own TRACE data, keeping the extended values only before 2002-08.

The panel's column names **and their order** are frozen in `lib/contract.py` and asserted at the
end of every build. Adding, removing or moving a column is a public API change and a CHANGELOG
entry, never a silent edit.

---

## References

**Factor Models:**
- Bai, J., Bali, T. G., & Wen, Q. (2019). Common risk factors in the cross-section of corporate bond returns. *Journal of Financial Economics*.
- Bai, J., Bali, T. G., & Wen, Q. (2021). Is there a risk-return tradeoff in the corporate bond market? Time-series and cross-sectional evidence. *Journal of Financial Economics*, 142(3), 1017-1037.
- van Binsbergen, J. H., Nozawa, Y., & Schwert, M. (2025). Duration-based valuation of corporate bonds. *Review of Financial Studies*, 38(1), 158-191.
- Chung, K. H., Wang, J., & Wu, C. (2019). Volatility and the cross-section of corporate bond returns. *Journal of Financial Economics*, 133(2), 397-417.
- Lin, H., Wang, J., & Wu, C. (2011). Liquidity risk and expected corporate bond returns. *Journal of Financial Economics*, 99(3), 628-650.
- He, Z., Kelly, B., & Manela, A. (2017). Intermediary asset pricing: New evidence from many asset classes. *Journal of Financial Economics*.
- Harvey, C. R., & Siddique, A. (2000). Conditional skewness in asset pricing tests. *Journal of Finance*, 55(3), 1263-1295.

**Risk Premia:**
- Bali, T. G., Subrahmanyam, A., & Wen, Q. (2021). The macroeconomic uncertainty premium in the corporate bond market. *Journal of Financial and Quantitative Analysis*, 56(5), 1653-1678.
- Ceballos, L. (2021). Inflation volatility risk and the cross-section of corporate bond returns. Working paper.
- Dickerson, A., Mueller, P., & Robotti, C. (2023). Priced risk in corporate bonds. *Journal of Financial Economics*, 150(2), 103707.
- Gebhardt, W. R., Hvidkjaer, S., & Swaminathan, B. (2005). The cross-section of expected corporate bond returns: Betas or characteristics? *Journal of Financial Economics*, 75(1), 85-114.
- Koijen, R. S., Lustig, H., & Van Nieuwerburgh, S. (2017). The cross-section and time series of stock and bond returns. *Journal of Monetary Economics*, 88, 50-69.
- Ang, A., Chen, J., & Xing, Y. (2006). Downside risk. *Review of Financial Studies*, 19(4), 1191-1239.
- Baker, S. R., Bloom, N., & Davis, S. J. (2016). Measuring economic policy uncertainty. *Quarterly Journal of Economics*, 131(4), 1593-1636.

**Illiquidity Measures:**
- Pastor, L., & Stambaugh, R. F. (2003). Liquidity risk and expected stock returns. *Journal of Political Economy*, 111(3), 642-685.
- Amihud, Y. (2002). Illiquidity and stock returns: Cross-section and time-series effects. *Journal of Financial Markets*.
- Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient market. *Journal of Finance*.
- Hong, G., & Warga, A. (2000). An empirical study of bond market transactions. *Financial Analysts Journal*.
- Corwin, S. A., & Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. *Journal of Finance*.
- Abdi, F., & Ranaldo, A. (2017). A simple estimation of bid-ask spreads from daily close, high, and low prices. *Review of Financial Studies*.
- Fong, K. Y. L., Holden, C. W., & Trzcinka, C. A. (2017). What are the best liquidity proxies for global research? *Review of Finance*, 21, 1355-1401.
- Danyliv, O., Bland, B., & Nicholass, D. (2014). Convenient liquidity measure for financial markets. Working paper.
- Tobek, O. (2016). Liquidity proxies using daily trading volume. Working paper.

**Momentum and Reversal:**
- Gebhardt, W. R., Hvidkjaer, S., & Swaminathan, B. (2005). Stock and bond market interaction: Does momentum spill over? *Journal of Financial Economics*, 75(3), 651-690.
- Novy-Marx, R. (2012). Is momentum really momentum? *Journal of Financial Economics*, 103(3), 429-453.
- Blitz, D., Huij, J., & Martens, M. (2011). Residual momentum. *Journal of Empirical Finance*, 18(3), 506-521.
- Bali, T. G., Subrahmanyam, A., & Wen, Q. (2021). Long-term reversals in the corporate bond market. *Journal of Financial Economics*, 139(2), 656-677.
- Subrahmanyam, A. (2023). Corporate bond data projects: Some clarifications. Working paper.
- Wang, J., Wu, D., & Yang, L. (2024). Cross-bond momentum spillovers. Working paper.

**Value Signals:**
- Houweling, P., & Van Zundert, J. (2017). Factor investing in the corporate bond market. *Financial Analysts Journal*, 73(2), 100-115.
- Israel, R., Palhares, D., & Richardson, S. (2018). Common factors in corporate bond returns. *Journal of Investment Management*, 16(2), 17-46.
- Gilchrist, S., & Zakrajšek, E. (2012). Credit spreads and business cycle fluctuations. *American Economic Review*, 102(4), 1692-1720.

**Duration-Adjusted Bond Returns and Treasury Benchmarks:**
- Andreani, M., Palhares, D., & Richardson, S. (2024). Computing corporate bond returns: A word (or two) of caution. *Review of Accounting Studies*, 29(4), 3887-3906.
- Bessembinder, H., Kahle, K. M., Maxwell, W. F., & Xu, D. (2009). Measuring abnormal bond performance. *Review of Financial Studies*, 22(10), 4219-4258.
- Cui, L., Lu, X., & Song, Z. (2026). Understanding corporate bond excess returns. Working paper.
- Ghaderi, M., Plante, S., Roussanov, N., & Seo, S. B. (2026). Reconstructing a century of U.S. corporate bonds: Credit risk in historical perspective. Working paper.
- Gürkaynak, R. S., Sack, B., & Wright, J. H. (2007). The U.S. Treasury yield curve: 1961 to the present. *Journal of Monetary Economics*, 54(8), 2291-2304.

**Equity Factors:**
- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.
