# Table IA.VIII reconciled — the paper, Stage 2's dictionary, and the code

Table IA.VIII (*Signal Definitions and Citations*) defines every column of the
145-column Stage-2 panel. Three documents describe those columns: the paper's printed
table, `stage2/DATA_DICTIONARY.md`, and the code that computes each one. This is what
happens when you check all three against each other.

**Where they disagree, the code wins.** That is not a preference — a definition that
does not describe what the code does is wrong however carefully it was written, and the
only way to find out is to open the function.

Stage 3 now generates the table from `spec/signal_definitions.json`
(`s4_zoo/t_ia08.py`). Fifty-one rows there differ from the printed table; each carries
`why_corrected` and what was printed (`paper_prints`, or `paper_prints_citation` where only
the citation changed), is marked `†` in the rendered table, and is listed below. `python s4_zoo/t_ia08.py --diffs` prints them.

---

## The headline: coverage is exact

| check | result |
|---|---|
| mnemonics defined in the spec | **145** |
| rows Table IA.VIII prints | **140**: all but the five Treasury benchmarks added in 2026 |
| columns in `stage2/lib/contract.PANEL_COLUMNS` | **145** |
| in IA.VIII but not in the contract | **none** |
| in the contract but not in IA.VIII | **none** |
| Stage 3's 108 sorted signals missing a definition | **none** |
| of the 145, undocumented in `stage2/DATA_DICTIONARY.md` | **none** |
| citation keys in IA.VIII without an entry in the reference list (`stage2/_report_helpers.get_references_bib`) | **none** |
| cluster memberships, paper vs `zoo_engine.CLUSTERS` vs `s3_nse/clusters.py` | **identical** |

145 = 108 sorted signals + 37 identifiers, returns and characteristics. Every signal
Stage 3 sorts has a definition, and a test asserts it. The five alternative Treasury
benchmarks (`tret_bns`, `tret_cfm`, `tret_gprs`, `tret_cls`, `tret_mat`) carry no citation.

**The discrepancies are all in prose, definitions and citations.** None is a missing or
extra variable, and none changes which signals are sorted.

---

## 1. The paper's prose contradicts its own table — four of nine clusters ❗

Section IA.1 introduces each cluster with a count. Those counts do not match the table
that follows.

| cluster | prose says | the table has | |
|---|---|---|---|
| I. Spreads, Yields, Size | 9 | 9 | |
| II. Value | 5 | 5 | |
| III. Momentum & Reversal | **22** | **21** | ✗ |
| IV. Illiquidity | **14** | **13** | ✗ |
| V. Volatility & Risk | 16 | 16 | |
| VI. Market Risk | 9 | 9 | |
| VII. Credit & Default Betas | 4 | 4 | |
| VIII. Volatility & Liquidity Betas | **14** | **13** | ✗ |
| IX. Macro & Other Betas | **15** | **18** | ✗ |
| **total** | **108** | **108** | |

Both add to 108, which is why nothing caught it: −1 −1 −1 +3 = 0. Any check on the
total would have passed. The *table* is the one Stage 3 and Stage 2 agree with, and it is
the one the sorts actually use.

Only the counts are wrong. The prose descriptions are **compressed throughout**, not
enumerations — cluster II describes 5 signals in 2 clauses, V covers 16 in 11, IX covers
18 in 7 — so although cluster IV's sentence names twelve things for thirteen members
(`ilq`, Roll Autocovariance, goes unnamed), that is the same house style as every other
cluster and not a second defect.

---

## 2. `dcs6`: the lookup band is wrong, and the tie-break is backwards ❗

This is the one that matters. `dcs6` is an FDR survivor.

| source | says |
|---|---|
| the paper | "±2 month band, with **+** favored over **−**" |
| `stage2/DATA_DICTIONARY.md`, Cluster I table | "searches with **±1** month bandwidth", with no tie-break stated |
| **the code** | `DSPREAD_BANDWIDTH = 1`; `offsets = [0, -1, +1]` |

`DSPREAD_BANDWIDTH` in `_stage2_settings.py` sets the bandwidth to 1. `lib/value.py` builds
the search order as `[0]` then `[-j, +j]` for `j` in `1..bandwidth`, and takes the first
hit. So the search is **one month either side**, and when both are available it takes the
**earlier** month — the longer lag — not the later one.

The paper is wrong in the width *and* backwards in the direction. Stage 2's dictionary
had the width right and was silent on the tie-break.

**Fixed here**: the spec states ±1, earlier month first. Stage 2's dictionary now states
the tie-break too ("taking the EARLIER month first"), since a reader who needs the band
needs the order.

---

## 3. `mdc_rat`: a rating scale that contradicts the row above it

| source | says |
|---|---|
| the paper's `mdc_rat` row | 1 (AAA) to **20** (CCC−), **21** = Default |
| the paper's `spc_rat` row, one line above | 1 (AAA) to **21** (CCC−), **22** = Default |
| `stage2/DATA_DICTIONARY.md`, Bond Characteristics table | 1 to 21, 22 = Default, for **both** |

`spc_rat` and `mdc_rat` are the same scale reached by two different priority orders
(S&P first, or Moody's first). They cannot have different lengths. The paper
contradicts itself within two lines; 21/22 is the scale the data uses.

**Fixed here.**

❗**Both rows also mislabel rating 21 as CCC−.** On the numeric scale Stage 1 builds
(`stage1/helper_functions.convert_sp_to_numeric`), CCC− is 19, CC is 20, C is 21 and
D is 22; on Moody's side Caa3 is 19, Ca is 20 and C is 21. The range was right and
only the label on its last non-default grade was wrong. Stage 2's dictionary and report
carried the same label and are corrected with it.

---

## 4. Four momentum rows cite the wrong paper by the same authors

`mom3_1`, `mom6_1`, `mom9_1` and `mom12_1` cite `gebhardt2005cross` — Gebhardt,
Hvidkjaer and Swaminathan (2005), *The cross-section of expected corporate bond returns:
betas or characteristics?*

Bond momentum is the **other** GHS 2005 paper: *Stock and bond market interaction: does
momentum spill over?*, same three authors, same year, same journal (`gebhardt2005stock`).

`ytm`, `b_termb` and `b_defb` cite `gebhardt2005cross` correctly; that paper is about
betas and characteristics.

**Fixed here.**

---

## 5. `lib` looks forward and `igap_bgn` looks back, on the same row

Checked against the data, the two rows now say exactly what each column holds.

| | the paper prints | what the panel holds on the row for month $t$ | rows that match |
|---|---|---|---|
| `lib` | $P_t^{\mathrm{bgn}} / P_{t-1}^{\mathrm{end}} - 1$ | $P_{t+1}^{\mathrm{bgn}} / P_t^{\mathrm{end}} - 1$, the gap AFTER that month's end price | forward form 1,650,561 of 1,650,561. Printed form 2,192 |
| `igap_bgn` | business days between the month-end PRICE ($t{-}1$) and the month-begin price ($t$) | NYSE sessions from the last SESSION of $t{-}1$ to the month-begin trade of $t$, 1 to 5 | last-session form 1,801,790 of 1,801,790. From the month-end trade 1,215,816 of 1,584,054 |

`lib` is re-dated to $t-1$ in `stage2/steps/step1_returns.py` before it is merged onto the
month-end frame, so it sits beside the end price it follows. `igap_bgn` is not re-dated. It
is measured from `date_end_bus_lag`, the last NYSE session of the previous month, whether
or not the bond traded that day.

So the two columns point in opposite directions on the same row. That is the intended
behaviour and the data is unchanged. What changed is the text. The printed `lib` row
stamps the gap with the other month, and the printed `igap_bgn` row names a starting
point the code does not use. Stage 2's dictionary had `igap_bgn` wrong in a third way,
between month-end $t$ and month-begin $t+1$, and also said no column carries a lead or a
lag. Both are corrected.

**Fixed here**: both rows state the direction and the row they sit on.

---

## 6. Stage 2 documents all 145 — including `144a`

Every one of the 145 mnemonics has an entry in `stage2/DATA_DICTIONARY.md`, and every
column of `contract.PANEL_COLUMNS` does too. No gaps either way.

---

## 7. Stage 2's dictionary has no citation column

78 of the 145 rows in IA.VIII carry a citation, drawing on 37 distinct works.
`stage2/DATA_DICTIONARY.md` carries none, so a reader there cannot find where a signal
comes from without opening the paper.

Not fixed — adding the references to a data dictionary is a judgement call about what
that document is for. Noted so the choice is deliberate.

---

## 8. Four vocabularies for the same clusters

The cluster *memberships* are identical everywhere. The *names* are not:

| the paper (IA.VIII) | the paper (prose) | `s4_zoo/zoo_engine.py` | `s3_nse/clusters.py` |
|---|---|---|---|
| Cluster VII: Credit & Default Betas | VII. Credit & Default Betas | Credit & Default Betas | Credit & Default **Risk** |
| Cluster VIII: Volatility & Liquidity Betas | VIII. Volatility & Liquidity Betas | **Vol. & Liq.** Betas | Volatility & Liquidity **Risk** |
| Cluster IX: Macro & Other Betas | IX. Macro & Other Betas | Macro & Other Betas | Macro & Other **Risk** |

Section 5 reports risk groups and the zoo reports beta groups, which is why the two
differ. It is a real trap: **any join on a cluster NAME fails**. Join on the signal.
`test_the_two_cluster_maps_are_the_same_partition` pins the memberships so a signal
cannot drift between them unnoticed.

---

## 9. `hprd` and `hprd_bgn` were never calendar days ❗

| source | says |
|---|---|
| the paper, both rows | holding period "in calendar days" |
| `stage2/DATA_DICTIONARY.md` before 3.3.0, the table rows | the same |
| `stage2/DATA_DICTIONARY.md`, the Bond Returns section | business days between `dt_s` and `dt_e` |
| **the code before 3.3.0** | NYSE sessions from `dt_s` to the CALENDAR month-end |
| **the code** | NYSE sessions from `dt_s` to `dt_e` |

Two defects in one row. The unit was wrong everywhere it was written down: the column is
a count of NYSE sessions, taken from the calendar lookup. And the end point was wrong in
the code: `hprd` counted to the last calendar day of the month, a date the return does not
use, so it overstated the window of every return whose end trade fell before the last
session. The data report printed holding periods of 25 and 26 for a monthly return, which
is how it was found.

`stage2/steps/step1_returns.py` now counts to `dt_e`, the end trade. The column is the
window of the return printed beside it, and
`contract.assert_holding_period_is_the_window` checks that on every row of every build.
It runs 15 to 27. A month has at most 23 sessions, and the start trade may sit up to 4
sessions before the last session of the month before, so 27 is the ceiling and not an
error. No return changes. The filter `hprd > 0` keeps the same rows, because the end
trade follows the start trade on every one.

`hprd_bgn` always counted sessions between its two trades (`dt_s_bgn` to `dt_e_bgn`),
observed 10 to 22. Only its text was wrong.

**Fixed here**, in the code, the dictionary and the spec. The paper's Table IA.II reports
`hprd`, and Stage 3 computes that row from the corrected column.

---

## 10. `val_hz` has no maturity control

The paper lists the controls as rating, industry, **maturity**, 3-month spread change and
callable. `stage2/lib/value.py` calls `compute_value(model_type='hz', x_cols=['dcs3',
'call'])` with rating and FF17 industry dummies. Nothing else enters the regression, so
maturity is not a control.

**Fixed here**: the spec and the dictionary list the four controls the code uses. Whether
the regression SHOULD carry maturity is a separate question for the authors and is not
decided here.

---

## 11. `b_credit` cites the wrong paper by the same first author

The paper's row cites Dickerson, Mueller and Robotti (2023), *Priced risk in corporate
bonds*. The credit spread level beta comes from Dickerson, Julliard and Mueller (2026),
*The co-pricing factor zoo*, Journal of Financial Economics 182, 104295.

The Stage 2 data report cited the right paper all along. Of 78 cited rows, this is the
only one where the report and the printed table named different work, apart from the four
momentum rows of section 4.

**Fixed here.**

---

## 12. VaR and expected shortfall are monthly, not daily

`var_90`, `var_95` and `es_90` were printed as measures of "daily" returns. The code computes
them from monthly ones: `stage2/steps/step6_momentum.py` passes `lib/var_es.py` the bond-month
return panel, and the window is 36 months with at least 12 -- the "36(12)" the same sentence
already printed.

**Fixed here**, and in `stage2/DATA_DICTIONARY.md`.

---

## 13. `str` is the month the momentum signals skip, not the one before it

| source | says |
|---|---|
| the paper, and `stage2/DATA_DICTIONARY.md` before 4.0.0 | prior month return, $r_{t-1}$ |
| **the panel**, the row for month $t$ | the return over month $t$, measured with the signal gap |

On the 2026 panel `str` correlates 0.90 with `ret_vw` on the same row and −0.04 with
`ret_vw` on the row before. `mom3_1` on the same row compounds months $t-2$ and $t-1$, so
`str` is the month the momentum signals skip, which is what a short-term reversal signal
is. It is not identical to `ret_vw` because it is measured with the signal gap, as every
price-based signal in the main panel is. Its unadjusted twin, `str_mmn` in the sidecar, is
`ret_vw` itself.

**Fixed here**, and in `stage2/DATA_DICTIONARY.md`, whose momentum formulas now compound
returns as the code does. `mom3_1`'s printed "skipping prior month" now says the month it
skips, $t$.

---

## 14. The industry signals average every bond in the SIC code, the bond included

| source | says |
|---|---|
| the paper, six rows (`imom1`, `imom3_1`, `imom12_1`, `iltr24_3`, `iltr30_6`, `iltr48_12`) | the equal-weighted average signal of OTHER bonds in the same FF17 industry |
| **the code** | the equal-weighted return of every bond sharing the FISD `sic_code` that month, the bond itself included, compounded over the signal's window |

`stage2/lib/momentum.py` groups returns by (`sic_code`, month), averages them, and compounds
that industry return over the same windows as `mom` and `ltr`. It never groups by FF17, and it
never leaves the bond out. So every bond in a (`sic_code`, month) group carries the same value.

**Fixed here**, and in `stage2/DATA_DICTIONARY.md`.

---

## 15. Ten more rows the code settles

| column | the paper printed | the code |
|---|---|---|
| `cs_mu12_1` | skips the prior month | averages the 12 observations before month $t$, $t-1$ included; the published column matches that on 100% of 1,665,078 rows |
| `spd_abs`, `spd_rel` | a minimum of 5 prices | no minimum; 78,937 published values come from months with fewer than 5 priced days |
| `p_fht` | $p_{zro}$ in the formula | the full-month share, though the published $\sigma$ and `p_zro` column leave out the last day of trading |
| `dvol_sys`, `dvol_idio` | a CAPMB regression | a regression on the day's equal-weighted average bond return |
| `b_vix` | a monthly regression | a regression of daily returns within the month |
| `b_defb`, `b_termb` | market regressions | one two-factor regression on TERMB and DEFB |
| `fce_val` | units of the bond outstanding | thousands of dollars (median 500,000, a $500 million issue) |

**Fixed here**, and in `stage2/DATA_DICTIONARY.md`.

---

## 16. Seventeen rows checked against the data

Each was recomputed from the code, and where a number is quoted, measured on the 2026-09-21
panel.

| column | the paper printed | the code |
|---|---|---|
| `rsj` | $(RV^+ - RV^-)/RV$, a ratio within $\pm 1$ | up-minus-down volatility over the realized variance; runs from -2,142 to 2,254 |
| `ilq`, `roll` | autocovariance $\times 100$; units unstated | log returns in percent, so `ilq` is 10,000 times the decimal autocovariance and `roll` is in percent of price |
| `p_zro` | the share of days with no price | the gap-adjusted share, which counts the last trading day as unpriced: never 0 |
| `lix` | a minimum of 5 prices | a minimum of 5 daily Amihud ratios; days with high = low left out |
| `tret` | duration unstated | the duration on the month-end trade, not the gap-adjusted `md_dur` |
| `country` | country of issuance | the issuer's domicile |
| `rfret` | $r^x = r - r^f$ | $r - r^f$; $r^x$ is the duration-adjusted return |
| `iskew` | residuals of the coskewness regression | the same, with each month's rolling betas, and raw returns before the betas exist |
| `b_mktbx_dcapm`, `b_term_dcapm` | the duration-adjusted return on the left; TERM = MKTB - MKTBX | `ret_vw` on the left; TERM = MKTB + $r^f$ - MKTBX |
| `b_dvix_va`, `b_dvix_vp` | a $\Delta$VIX beta | the sum of the betas on this month's and last month's change |
| `b_dvixd` | the daily $\Delta$VIX | the change between the bond's own trade dates, at most 5 days apart |
| `b_dcpi`, `b_cpi_vol6` | CPI changes | lagged changes in the CPI index, in index points |
| `b_cptlt` | the capital ratio | the traded intermediary factor, a return |

**Fixed here**, and in `stage2/DATA_DICTIONARY.md`. Two of these describe what the code does
where it may not do what was meant (`rsj`, and TERM keeping $r^f$); they are recorded, not
changed, because changing them changes published numbers.

---

## 17. The wording is the paper's, and `vov` says what $\bar{V}$ is (2026-09-28)

The revised paper renders Table IA.VIII from this spec. Until 4.2.1 it could not do so
verbatim: nine descriptions carried notes meant for readers of the code (capitals for
emphasis, "as the code computes it", a file name, a summary statistic, what a column is NOT),
and the paper printed its own cleaner wording beside them. The descriptions now say what
the paper prints; each note that carried information moved to the row's `why_corrected`
(`lix`'s statistic) or is already there (`p_zro`'s old name, `b_mktbx_dcapm`'s `betas_x`).

| column | now says |
|---|---|
| `lib`, `dcs6`, `tret_mat` | the same, without the capitals |
| `rsj`, `p_zro`, `b_mktbx_dcapm`, `b_eput` | the same, without the note for code readers |
| `tret` | the same, with the column names in code type |
| `lix` | the corrected definition, without the file name and statistic; the paper had kept "a minimum of 5 prices", which the code does not apply |
| `vov` | ❗ what $\sigma$ and $\bar{V}$ are: the volatility of daily returns and the mean daily dollar volume (`avg(dvol)`); the paper as submitted defined neither, under the name "Volatility of Volume" |

The five Treasury benchmarks added to the panel in 2026 (`tret_bns`, `tret_cfm`, `tret_cls`,
`tret_gprs`, `tret_mat`) are defined here and in the Stage 2 data report, but the paper's
table does not list them, so Table IA.VIII leaves them out (`in_table_ia08: false`).

---

## How to re-run this

```bash
cd stage3
python s4_zoo/t_ia08.py --diffs     # the fifty corrected rows and what settled each
python s4_zoo/t_ia08.py             # -> reports/tables/table_ia08.tex
python -m pytest tests/ -k definition -q  # every sorted signal has a definition
```

The spec is `spec/signal_definitions.json`. It is the paper's text with fifty rows
corrected, each recording what was printed and why it changed — so the printed table can
always be reconstructed from it, and no correction is silent.
