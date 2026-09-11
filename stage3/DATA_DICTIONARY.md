# Stage 3 — Data Dictionary

Every artifact Stage 3 writes, and what its columns mean.

## Contents

0. [The compiled report](#the-compiled-report) — `reports/exhibits.pdf`
1. [Long-format sort panels](#long-format-sort-panels) — `data/sorts/`
2. [The uncertainty grids](#the-uncertainty-grids) — `data/grids/`
3. [Statistics frames](#statistics-frames) — `data/<section>/`
4. [Result manifests](#result-manifests) — `data/<section>/*.json`
5. [Sample windows](#sample-windows)
6. [Conventions that decide what a number means](#conventions-that-decide-what-a-number-means)

---

## The compiled report

`reports/exhibits.pdf` — every exhibit in one document, built by `make_report.py`.

The exhibit numbers in it are **the paper's** (Table IA.XII, Figure IA.3), announced by
each heading. LaTeX's own sequential numbering is suppressed, so a reference to "Table 4"
means the same table it does in the paper.

Its title page records what produced it: the sample windows, the PyBondLab build with its
content hash and whether the fast kernels were present, and every input with its size and
sha256 prefix. `reports/exhibits.tex` is the assembled source, and
`reports/exhibits.build.log` the pdflatex transcript.

---

## Long-format sort panels

`data/sorts/*.csv` — one row per (date, factor, leg, weighting).

| column | type | meaning |
|---|---|---|
| `date` | date | **return realisation date (t+1)**. Formation was one month earlier, so the first row of every series is NaN. |
| `factor` | text | signal mnemonic. ❗A trailing `*` means the series has been **sign-corrected** — see the conventions below. |
| `freq` | int | holding period in months |
| `leg` | text | `l` long, `s` short, `ls` long minus short |
| `weighting` | text | `ew` or `vw` |
| `return` | float | portfolio return as a **decimal**, in excess of the one-month T-bill |
| `turnover` | float | two-way turnover for that leg, decimal; null where turnover was not requested |
| `count` | int | bonds in the leg that month |

The month-begin sets also carry `lib` and `ilq` — the portfolio-level characteristic
spreads, which Table 2's decomposition tests against the return gap.

### File naming

**The three approaches** (`data/sorts/`):

```
{ret_type}_{sort}_{rating}_{signal_type}_{timing}_p{N}.csv
```

`ret_type` ∈ {`exc`, `dur`} · `sort` ∈ {`single`, `wf`} · `rating` ∈ {`all`, `ig`, `nig`} ·
`signal_type` ∈ {`baseline`, `mmn`} · `timing` ∈ {`end`, `bgn`} · `N` = number of
portfolios (10 all-bond single, 5 rating splits, 2 within-firm).

**The 108-signal census** (`data/sorts/lib/`):

```
bond_{single_sort|within_firm}_lib_all_{end|bgn}_p{10|2}_h1.csv
```

**The factor zoo** (`data/sorts/zoo/`):

```
bond_{single_sort|within_firm}_all_p{10|2}_h1.csv
```

> ❗One file per (sort, set, rating) carries **every signal**, and each exhibit selects
> the ones it prints. The filename does not encode the signal list, so running a
> producer twice with different `--signals` leaves whichever ran last.

---

## The uncertainty grids

### Method uncertainty — `data/grids/mua/<signal>.parquet`

One row per (date, spec, leg). 216 specs per signal: 2 weightings × 3 portfolio counts ×
3 breakpoint universes × 3 rating filters × 4 maturity buckets.

| column | meaning |
|---|---|
| `date` | return realisation date |
| `signal` | the signal this grid is for |
| `spec_id` | `{EW\|VW}_{Tp\|Qp\|Dp}_Q_{bp}_{rating}_{maturity}` |
| `leg` | `L`, `S` or `LS` |
| `return` | decimal |
| `nbonds` | **realised** bonds held — how many were in the portfolio that earned the return, not how many were selected at formation |

24 of the 216 are infeasible (an investment-grade breakpoint universe crossed with a
high-yield rating filter) and are all-NaN rather than absent, so the frame is a rectangle.

### Data uncertainty — `data/grids/dua/series/<rating>/<signal>.parquet`

One row per (date, weighting, filter_config): the ex-ante monthly return under each of
120 cleaning filters — 48 return-trim thresholds, 30 price screens, 30 bounce-back
screens, 12 winsorization levels — plus the baseline.

`data/grids/dua/dua_{premia,alpha,baselines}_{paper,full}.parquet` hold the per-path
statistics computed from those series; `dua_config_locations.parquet` records each
filter's tail location as the fit itself reported it.

> ❗Series are saved **untruncated**. The sample window is applied at the statistics
> layer, by truncating the series and recomputing — a stored statistic is never
> truncated. The one exception is the Section-4 winsorization sweep, where the threshold
> is a full-sample quantile by construction, so the window is a producer argument there.

---

## Statistics frames

`data/<section>/*.csv`. These are what the exhibits format; nothing downstream
recomputes a regression.

**Section 3** (`data/s1_lib/*_stats.csv`) — one row per (design cell, factor, quantity):

| column | meaning |
|---|---|
| `ret_type`, `sort`, `rating`, `weighting` | the design cell |
| `factor` | base mnemonic, decorations and sign flag removed |
| `quantity` | `<approach>.mu`, `<approach>.alpha`, `<pair>.d_mu`, `<pair>.d_alpha` |
| `value` | in **percent** per month |
| `tstat` | Newey-West |
| `T`, `nw_lags` | sample length and `floor(T**0.25)` |

Approaches are `unadjusted`, `adj_signal`, `adj_return`; pairs are `bias_1_2` and
`bias_1_3`.

**Section 4** (`data/s2_lab/*_stats.csv`) — one row per (factor, leg, variant, stat):
`leg` ∈ {long, short, ls}, `variant` ∈ {wins, base, bias}, `stat` ∈ {mu, alpha}.
❗`T` is **each series' own** length here, not one shared number.

**Section 5** (`data/s3_nse/*.csv`) — one row per cluster: `mu_mean`, `mu_median`,
`nse_mu`, `ratio_mu`, the same four for alpha, and `n_paths`. NSE is the interquartile
range of the estimate across paths; Ratio divides it by the average conventional
standard error.

**The zoo** (`data/s4_zoo/*_cells.csv`) — one row per printed factor: `T`, `start`,
`end`, `mu`, `sd`, `t_mu`, `sr`, `alpha`, `t_alpha`, `ir`, and `shaded` for
Benjamini-Hochberg survival. Means, SDs and alphas are annualized and in percent.

---

## Result manifests

Every result carries one, written by `drrlib.write_result`:

| field | what |
|---|---|
| `written_utc`, `wall_s` | when, and how long |
| `git_commit`, `git_branch`, `code_dirty` | the code that produced it |
| `pybondlab` | version, path, git state, content hash, whether the fast kernels were present |
| `inputs` | each input's path, size and sha256 prefix |
| `python` | interpreter version |

A number whose inputs, code version and engine are not recorded cannot be defended
later, which is why this block is written by one function rather than by each caller.

`reports/timings.jsonl` carries one line per run: the phase split, the wall clock, and
the run's own check — with `ok` written **before** the timings, because a fast run that
produced an incomplete artifact is not a result.

---

## Sample windows

| window | span | T | used by |
|---|---|---|---|
| `lib` | 2002-09-30 → 2024-12-31 | 268 | Section 3 and the Section-5 `paper` window |
| `lab` | 2002-08-31 → 2024-12-31 | 269 | Section 4 |
| `full` | to the panel's own frontier | derived | the `full` window variants, and the zoo |

Both fixed windows give `floor(T**0.25) = 4` Newey-West lags. **Assert the length before
trusting a t-statistic**: the lag count is derived from it, so a window one month off
moves every number in the table. `drrlib.assert_sample` is the check, and the exhibits
call it.

Sorts are formed from **2002-07-31** so the first printed return, 2002-09-30, has a
formed portfolio behind it.

---

## Conventions that decide what a number means

**Sign correction.** A trailing `*` means PyBondLab negated that series, deciding from
the sign of its full-sample mean at extract time. ❗The flip set therefore depends on
**the sample that was sorted**: two runs over different windows can legitimately
disagree about which factors are starred. Compare flip *sets*, never assume they match,
and never difference a starred series against an unstarred one without re-orienting
both. Table B.1 ships both variants precisely because of this.

**Units.** Sort panels and grids store **decimals**. Tables print percent. The
conversion happens once, at the boundary — except the DUA statistics, which are already
in percent at source, and the MUA summary, which is decimal and scaled once at load.

**Two sign-correction baselines in Section 5.** The tables sign off
`VW_Qp_Q_all_all_all`; the figures sign off `VW_Dp_Q_all_all_all`. Deliberate. Using one
where the other belongs silently flips a subset of the factors.

**Two Ratio rules in Section 5.** Table 5 computes std/mean(SE) on the **pairwise-matched
sample**; Table 6 uses **independent skipna**. Each table wants its own; they are not
interchangeable.

**Four selection rules across four figures.** The data-uncertainty figures pick their
top four per cluster by the **mean of the level** frame; the method-uncertainty figures
by the **median of the t-statistic** frame. So IA.5 and IA.6 select different signal
sets from each other. Never reuse one figure's selection for another.

**Degenerate strategies** — a leg empty in some month inside its active window — are
excluded from every printed MUA statistic, and derived from the realised counts rather
than declared. Months **before** a late-starting signal exists do not count as empty.

**Bias is tested on the difference series**, not by differencing two separately
estimated means. Same point estimate, very different standard error.

**The alpha bias is asymmetric on purpose**: the point estimate is the difference of two
separately estimated alphas, while the t-statistic comes from regressing the paired bias
series on MKTB. The point estimate is a difference of levels; the test is on the
difference series.
