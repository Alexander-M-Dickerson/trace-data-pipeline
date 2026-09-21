# Table IA.VIII reconciled — the paper, Stage 2's dictionary, and the code

Table IA.VIII (*Signal Definitions and Citations*) defines every column of the
145-column Stage-2 panel. Three documents describe those columns: the paper's printed
table, `stage2/DATA_DICTIONARY.md`, and the code that computes each one. This is what
happens when you check all three against each other.

**Where they disagree, the code wins.** That is not a preference — a definition that
does not describe what the code does is wrong however carefully it was written, and the
only way to find out is to open the function.

Stage 3 now generates the table from `spec/signal_definitions.json`
(`s4_zoo/t_ia08.py`). Twelve rows there differ from the printed table; each carries
`paper_prints` and `why_corrected`, is marked `†` in the rendered table, and is listed
below. `python s4_zoo/t_ia08.py --diffs` prints them.

---

## The headline: coverage is exact

| check | result |
|---|---|
| mnemonics in IA.VIII | **145** |
| columns in `stage2/lib/contract.PANEL_COLUMNS` | **145** |
| in IA.VIII but not in the contract | **none** |
| in the contract but not in IA.VIII | **none** |
| Stage 3's 108 sorted signals missing a definition | **none** |
| of the 145, undocumented in `stage2/DATA_DICTIONARY.md` | **none** |
| citation keys in IA.VIII not resolvable in the paper's `references.bib` | **none** |
| cluster memberships, paper vs `zoo_engine.CLUSTERS` vs `s3_nse/clusters.py` | **identical** |

145 = 108 sorted signals + 37 identifiers, returns and characteristics. Every signal
Stage 3 sorts has a definition, and a test asserts it. The count was 140 when this
reconciliation was first run (2026-09-12); the five alternative Treasury benchmarks
(`tret_bns`, `tret_cfm`, `tret_gprs`, `tret_cls`, `tret_mat`) joined the panel on 2026-09-14
and carry no citation.

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

**Fixed in the paper on 2026-09-12** — four digits in each of `main.tex` (lines 2062,
2065, 2077, 2080) and `internet-appendix.tex` (547, 550, 562, 565), which carry the same
nine sentences. Nothing else changed, and all three PDFs rebuild at their expected page
counts (main-ms 53, internet-appendix 38, main 94).

Only the digits needed changing. The prose descriptions are **compressed throughout**,
not enumerations — cluster II describes 5 signals in 2 clauses, V covers 16 in 11, IX
covers 18 in 7 — so although cluster IV's sentence names twelve things for thirteen
members (`ilq`, Roll Autocovariance, goes unnamed), that is the same house style as
every other cluster and not a second defect.

---

## 2. `dcs6`: the lookup band is wrong, and the tie-break is backwards ❗

This is the one that matters. `dcs6` is an FDR survivor.

| source | says |
|---|---|
| the paper | "±2 month band, with **+** favored over **−**" |
| `stage2/DATA_DICTIONARY.md`, Cluster I table | "searches with **±1** month bandwidth" (no tie-break stated, as first read) |
| **the code** | `DSPREAD_BANDWIDTH = 1`; `offsets = [0, -1, +1]` |

`_stage2_settings.py:132` sets the bandwidth to 1. `lib/value.py:779-781` builds the
search order as `[0]` then `[-j, +j]` for `j` in `1..bandwidth`, and `lib/value.py:816`
takes the first hit. So the search is **one month either side**, and when both are
available it takes the **earlier** month — the longer lag — not the later one.

The paper is wrong in the width *and* backwards in the direction. Stage 2's dictionary
has the width right and is silent on the tie-break.

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

❗**Both rows also mislabel rating 21 as CCC−** (corrected 2026-09-21, which makes
`spc_rat` the seventh corrected row; sections 5, 9 and 10 add five more). On the numeric scale Stage 1 builds
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
momentum spill over?*, same three authors, same year, same journal. It is already in
`references.bib` as `gebhardt2005stock` — **and is cited nowhere in the paper**, which is
the tell.

`ytm`, `b_termb` and `b_defb` cite `gebhardt2005cross` correctly; that paper is about
betas and characteristics.

**Fixed here.**

---

## 5. `lib` looks forward and `igap_bgn` looks back, on the same row

Read on 2026-09-12 as one object under two indexing conventions, and left alone. Checked
against the data on 2026-09-21, that reading was wrong about `igap_bgn`, and the two rows
now say exactly what each column holds.

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

Worth recording how nearly this was reported as a gap: the first pass matched dictionary
rows with `` `([A-Za-z_][A-Za-z_0-9]*)` ``, which cannot match `` `144a` `` because the
mnemonic starts with a digit. The checker reported one missing row and the row was
there all along, in the Bond Characteristics table. A coverage check that is wrong about its
own alphabet reports a clean bill of health for 139 names and invents a defect in the
140th.

---

## 7. Stage 2's dictionary has no citation column

78 of the 145 rows in IA.VIII carry a citation, drawing on 36 distinct works (35 when
first counted; the momentum correction in section 4 brought in `gebhardt2005stock`).
`stage2/DATA_DICTIONARY.md` carries none, so a reader there cannot find where a signal
comes from without opening the paper.

Not fixed — adding 36 references to a data dictionary is a judgement call about what
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
| `stage2/DATA_DICTIONARY.md`, the table rows | the same |
| `stage2/DATA_DICTIONARY.md`, the Bond Returns section | business days between `dt_s` and `dt_e` |
| **the code, until 2026-09-21** | NYSE sessions from `dt_s` to the CALENDAR month-end |
| **the code, now** | NYSE sessions from `dt_s` to `dt_e` |

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
`hprd`, so that row is recomputed with the manuscript.

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

## How to re-run this

```bash
cd stage3
python s4_zoo/t_ia08.py --diffs     # the twelve corrected rows and what settled each
python s4_zoo/t_ia08.py             # -> reports/tables/table_ia08.tex
python -m pytest tests/ -k ia08 -q  # every sorted signal has a definition
```

The spec is `spec/signal_definitions.json`. It is the paper's text with twelve rows
corrected, each recording what was printed and why it changed — so the printed table can
always be reconstructed from it, and no correction is silent.
