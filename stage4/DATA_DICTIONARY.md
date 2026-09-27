# Stage 4 data dictionary

The files `build_factors.py` writes under `stage4/output/`. They have the layout and columns of
the TRACE-only factor files on openbondassetpricing.com.

## The long panel: `<sort>_sort_trace_<vintage>.parquet`

One row per month, factor, leg, weighting, rating band and return type. Every factor appears
in every month: a month in which a factor cannot be formed has an empty `return`.

| column | type | meaning |
|---|---|---|
| `date` | datetime | month end of the return |
| `factor` | string | the signal's name with the sort and band in it: `age` (single, all bonds), `age_ig`, `age_hy`; within-firm `age_wf`, `age_wf_ig`, `age_wf_hy`. The non-investment-grade band is written `hy` |
| `freq` | int | always 1: a one-month holding period, rebalanced monthly |
| `leg` | string | `l` the long leg (highest signal), `s` the short leg (lowest), `ls` long minus short |
| `weighting` | string | `ew` equal weights, `vw` value weights (market value at the end of the previous month) |
| `return` | float | the leg's return in the month, as a decimal (0.01 is 1%). For single sorts `ls` is `l` minus `s`. For within-firm sorts see below |
| `turnover` | float | the leg's turnover at rebalancing, as PyBondLab computes it; for `ls`, the average of the two legs |
| `count` | float | bonds in the leg; for `ls`, the two legs added |
| `rating_type` | string | `all`, `ig` or `nig`: the band the portfolios were formed in |
| `return_type` | string | `exc`, `dur`, `dbns` or `dcls` (see [README_stage4.md](README_stage4.md)) |
| `num_portfolios` | int | single sorts only: 10 for `all`, 5 within a band |

**Within-firm sorts.** The firm is the bond's `permno`, so a bond with no `permno` (11.2% of the
2026-09-21 panel's rows) never enters them. The factor is the `ls` leg. Each month, inside each firm and each of
three rating groups (AAA to A-, BBB, below investment grade), it is the return of the firm's high-signal bonds minus its low-signal
bonds, each weighted equally (`ew`) or by market value (`vw`); those differences are averaged across firms (equally for `ew`, by the firm's bond market
value for `vw`), then across the rating groups that have a firm that month. A firm counts in a
rating group when it has at least two bonds there with a signal and a market value, at least
two distinct signal values, and a bond in each leg. Its bonds strictly above the two-thirds
point of its distinct signal values form its long leg, those strictly below the one-third point
its short leg; firms are weighted by the market value of the bonds in their two legs for `vw`.
The `l` and `s` legs pool every firm's high and
low bonds into one portfolio each, so they include the differences between firms that the
factor removes, and `ls` is not `l` minus `s`. This is PyBondLab's definition, and the
published files have always been built this way.

The long and short legs of `exc` are excess returns; the duration-adjusted return types are
already differences against a Treasury return. The series are unflipped.

## The CSVs: `<sort>_sort_<return>_<band>_<weighting>.csv`

The `ls` leg of one return type, band and weighting, pivoted wide: a `date` column, then one
column per factor, named as in the long panel. 4 return types x 3 bands x 2 weightings = 24
files per sort, written with 10 significant digits.

## `flip_set.json`

| key | meaning |
|---|---|
| `keys` | `factor|weighting|return_type`: how each entry in `flips` is named |
| `flips` | for each of those, `true` if the full-sample mean of the `ls` return is negative |
| `n_keys`, `n_would_flip` | how many entries, and how many are `true` |

To reproduce a sign-corrected series, negate `ls` (and swap `l` and `s`) wherever `flips` is
`true`.

## `MANIFEST.json`

What produced the folder: the vintage and build time, rows, months and factors, the grid, the
Stage 2 inputs with their sha256, the PyBondLab version and content hash, the git commit of this
repository, the run time, and every file in the folder with its size and sha256.
