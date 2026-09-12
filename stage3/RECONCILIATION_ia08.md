# Table IA.VIII reconciled — the paper, Stage 2's dictionary, and the code

Table IA.VIII (*Signal Definitions and Citations*) defines every column of the
140-column Stage-2 panel. Three documents describe those columns: the paper's printed
table, `stage2/DATA_DICTIONARY.md`, and the code that computes each one. This is what
happens when you check all three against each other.

**Where they disagree, the code wins.** That is not a preference — a definition that
does not describe what the code does is wrong however carefully it was written, and the
only way to find out is to open the function.

Stage 3 now generates the table from `spec/signal_definitions.json`
(`s4_zoo/t_ia08.py`). Six rows there differ from the printed table; each carries
`paper_prints` and `why_corrected`, is marked `†` in the rendered table, and is listed
below. `python s4_zoo/t_ia08.py --diffs` prints them.

---

## The headline: coverage is exact

| check | result |
|---|---|
| mnemonics in IA.VIII | **140** |
| columns in `stage2/lib/contract.PANEL_COLUMNS` | **140** |
| in IA.VIII but not in the contract | **none** |
| in the contract but not in IA.VIII | **none** |
| Stage 3's 108 sorted signals missing a definition | **none** |
| of the 140, undocumented in `stage2/DATA_DICTIONARY.md` | **none** |
| citation keys in IA.VIII not resolvable in `references.bib` | **none** |
| cluster memberships, paper vs `zoo_engine.CLUSTERS` vs `s3_nse/clusters.py` | **identical** |

140 = 108 sorted signals + 32 identifiers, returns and characteristics. Every signal
Stage 3 sorts has a definition, and a test asserts it.

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
| `stage2/DATA_DICTIONARY.md:184` | "searches with **±1** month bandwidth" (no tie-break stated) |
| **the code** | `DSPREAD_BANDWIDTH = 1`; `offsets = [0, -1, +1]` |

`_stage2_settings.py:132` sets the bandwidth to 1. `lib/value.py:779-781` builds the
search order as `[0]` then `[-j, +j]` for `j` in `1..bandwidth`, and `lib/value.py:816`
takes the first hit. So the search is **one month either side**, and when both are
available it takes the **earlier** month — the longer lag — not the later one.

The paper is wrong in the width *and* backwards in the direction. Stage 2's dictionary
has the width right and is silent on the tie-break.

**Fixed here**: the spec states ±1, earlier month first. **Also fix**: Stage 2's
dictionary should state the tie-break, since a reader who needs the band needs the order.

---

## 3. `mdc_rat`: a rating scale that contradicts the row above it

| source | says |
|---|---|
| the paper's `mdc_rat` row | 1 (AAA) to **20** (CCC−), **21** = Default |
| the paper's `spc_rat` row, one line above | 1 (AAA) to **21** (CCC−), **22** = Default |
| `stage2/DATA_DICTIONARY.md:164-165` | 1 to 21, 22 = Default, for **both** |

`spc_rat` and `mdc_rat` are the same scale reached by two different priority orders
(S&P first, or Moody's first). They cannot have different lengths. The paper
contradicts itself within two lines; 21/22 is the scale the data uses.

**Fixed here.**

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

## 5. `lib` and `igap_bgn` are defined one month-index apart

| | the paper | Stage 2 |
|---|---|---|
| `lib` | $P_t^{\mathrm{bgn}} / P_{t-1}^{\mathrm{end}} - 1$ | $P_{t+1}^{\mathrm{bgn}} / P_t^{\mathrm{end}} - 1$ |
| `igap_bgn` | business days between month-end ($t{-}1$) and month-begin ($t$) | between month-end ($t$) and month-begin ($t{+}1$) |

The same object under two indexing conventions — the paper stamps the observation with
the month-begin price's month, Stage 2 with the month-end price's. Both are internally
consistent and neither is wrong.

**Not changed**, because changing it would make the spec disagree with the printed table
for no gain. Recorded here because a reader comparing the two documents will otherwise
suspect an off-by-one, and because anyone joining on the date needs to know which
convention a given file uses.

---

## 6. Stage 2 documents all 140 — including `144a`

Every one of the 140 mnemonics has an entry in `stage2/DATA_DICTIONARY.md`, and every
column of `contract.PANEL_COLUMNS` does too. No gaps either way.

Worth recording how nearly this was reported as a gap: the first pass matched dictionary
rows with `` `([A-Za-z_][A-Za-z_0-9]*)` ``, which cannot match `` `144a` `` because the
mnemonic starts with a digit. The checker reported one missing row and the row was
there all along, at `DATA_DICTIONARY.md:168`. A coverage check that is wrong about its
own alphabet reports a clean bill of health for 139 names and invents a defect in the
140th.

---

## 7. Stage 2's dictionary has no citation column

78 of the 140 rows in IA.VIII carry a citation, drawing on 35 distinct works.
`stage2/DATA_DICTIONARY.md` carries none, so a reader there cannot find where a signal
comes from without opening the paper.

Not fixed — adding 35 references to a data dictionary is a judgement call about what
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

## How to re-run this

```bash
cd stage3
python s4_zoo/t_ia08.py --diffs     # the six corrected rows and what settled each
python s4_zoo/t_ia08.py             # -> reports/tables/table_ia08.tex
python -m pytest tests/ -k ia08 -q  # every sorted signal has a definition
```

The spec is `spec/signal_definitions.json`. It is the paper's text with six rows
corrected, each recording what was printed and why it changed — so the printed table can
always be reconstructed from it, and no correction is silent.
