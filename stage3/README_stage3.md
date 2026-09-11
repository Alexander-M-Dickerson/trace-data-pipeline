# Stage 3 — portfolio sorts, uncertainty grids and the paper's exhibits

Stage 3 sits on top of the Stage-2 monthly bond panel and produces every table and
figure of *The Corporate Bond Factor Replication Crisis* — main text, appendix and
Internet Appendix: **32 tables and 11 figures**.

It runs on your own computer, like Stage 2. It needs no WRDS connection.

```bash
cd stage3
python tools/check_inputs.py     # are the five inputs there and the right shape?
python _run_stage3.py --list     # the 40 steps, and what is already built
bash run_stage3.sh               # everything, ending in reports/exhibits.pdf
```

The last step compiles every exhibit into **`reports/exhibits.pdf`** — 49 pages. That
is deliberately part of the run and not an afterthought: the exhibits are LaTeX
fragments, and a fragment that will not compile looks perfectly fine sitting on disk.
Compiling is what catches it.

---

## What Stage 3 produces

| output | what |
|---|---|
| `data/sorts/` | long-format long-short panels: `date, factor, freq, leg, weighting, return, turnover` |
| `data/grids/` | the two uncertainty grids, one parquet per signal |
| `data/<section>/` | the tidy statistics frames the exhibits format, plus a manifest per result |
| `reports/tables/` | the paper's tables, as LaTeX fragments |
| `reports/figures/` | the paper's figures, as PDF |
| **`reports/exhibits.pdf`** | **all 43 of them compiled into one document**, in the paper's order and under the paper's exhibit numbers, with a provenance page |
| `reports/timings.jsonl` | one line per run: phases, wall clock, and the run's own check |

Nothing downstream of a statistics frame recomputes a regression. Each section computes
its numbers **once**; every table and figure is a formatter over that frame. That is why
a section's figures cannot quietly disagree with its tables — and where they could, the
figure drivers check themselves against the statistics engine at 1e-9 and say so.

---

## Inputs

Five files, all from Stage 1 and Stage 2, all resolved automatically from the pipeline
tree beside this folder and all overridable from the environment:

| variable | what it is | used by |
|---|---|---|
| `STAGE2_PANEL` | the monthly bond panel | every section |
| `STAGE2_MMN` | the unadjusted `*_mmn` signal twins | the three approaches (Sections 3, 4) |
| `STAGE2_BBW` | `bbw_factors.parquet` — **MKTB** | every CAPM_B alpha |
| `STAGE2_FACTORS` | `factors.parquet` — the risk-free rate, VIX, the macro set | excess returns, Figure 7 |
| `STAGE1_DAILY` | the daily bond-day panel | the data appendix only |

> ❗**`STAGE2_BBW` and `STAGE2_FACTORS` are different files.** MKTB lives in the first,
> the risk-free rate in the second. They are not interchangeable.

`python tools/check_inputs.py` checks all five against `spec/inputs.json`: rows, the 198
columns Stage 3 reads, and the date span. It reports every problem at once, so a missing
column surfaces in a second rather than forty minutes into a grid.

---

## PyBondLab

Stage 3 runs every sort through PyBondLab, so **which copy is on `sys.path` is part of
the result**. `pblenv.py` makes that explicit: it puts the chosen build first, asserts
the import resolved there (a second install in the environment can otherwise shadow it
silently), and records the build's version, git state and a content hash in every
manifest Stage 3 writes.

```bash
export PYBONDLAB_DIR=/path/to/PyBondLab     # a checkout; unset = whatever is installed
```

> ❗**The uncertainty grids need a build carrying the fast kernels** —
> `PyBondLab.fast_sorts` and `PyBondLab.anomaly_assay_fast`. These are not in the 0.2.0
> release that Stage 2 pins. `pblenv.require_fast()` checks before the fan-out starts,
> rather than letting 108 workers each fail on an import.
>
> Everything except `--section nse` runs on the released PyBondLab. The `--fast` flag
> elsewhere is an optional speed-up, not a requirement.

---

## The three approaches

Every price-based signal is built twice — once from the month-end price, once from a
price observed at least one business day earlier — and every return is available on two
windows, month-end and month-begin. That gives the three columns the bias tables report:

1. **unadjusted** — noisy month-end signal, month-end return
2. **adjusted signal** — gapped signal, month-end return
3. **adjusted return** — noisy month-end signal, month-begin return

Bias (1)−(2) varies the portfolio **weights** holding the return fixed. Bias (1)−(3)
varies the **return window** holding the weights fixed — that second one is the latent
implementation bias the paper is about.

---

## The steps

`python _run_stage3.py --list` prints them. Two kinds, costing very different amounts:

**Producers** run sorts through PyBondLab and save return series. A producer is
**skipped when its output already exists**, so re-running after a crash resumes rather
than restarting; `--force` recomputes.

| section | producer | rough cost |
|---|---|---|
| `lib` | the three approaches, all bonds and both rating splits | ~30 s each with the kernels |
| `lib` | the 108-signal month-end/month-begin sorts, x4 | ~35 s each |
| `lab` | the winsorization sweep, 2 tails x 3 ratings | ~15 s |
| `nse` | the **MUA grid** — 108 signals x 216 method choices | ~3 min |
| `nse` | the **DUA grid** — 108 signals x 120 filters x 3 ratings | ~6 min, then ~1 min for its statistics |
| `zoo` | all 108 signals, single and within-firm | ~2 min |

**Exhibits** read those series and render. Seconds each, always re-rendered. The final
step compiles them all into one PDF.

❗`reports/exhibits.pdf` carries **your** numbers, from whatever panel Stage 2 built.
Nothing in it compares them to the paper's printed ones, and its title page says so -- a
PDF of tables under familiar captions is exactly the kind of artifact that gets mistaken
for the original.

Figures 1, 2 and 5 of the paper are schematics drawn in LaTeX -- a research framework, a
return timeline, a look-ahead illustration. They have no data behind them, so Stage 3
does not produce them, and the document says so rather than leaving a gap.

Those timings are with the fast kernels, on 24 cores. Without them the grids are hours
rather than minutes — that is what the kernels are for.

---

## Conventions that will bite

- Portfolio outputs are indexed by the **return realisation date (t+1)**, not the
  formation date. The first row is always NaN.
- A factor mnemonic ending in `*` has been **sign-corrected**: PyBondLab negated that
  series, deciding from the sign of its full-sample mean at extract time. ❗The flip set
  therefore depends on **the sample that was sorted** — two runs over different windows
  can legitimately disagree about which factors are starred. Compare flip *sets*, never
  assume they match, and never difference a starred series against an unstarred one
  without re-orienting both.
- Returns are stored as **decimals**. Tables print percent. Convert once, at the
  boundary.
- Newey-West lags are `floor(T**0.25)` everywhere. **Assert T before trusting a
  t-statistic** — the lag count is derived from it, so a window one month off moves
  every number in the table.
- Each bias is tested on the **difference series**, not by differencing two separately
  estimated means. Same point estimate, very different standard error.
- `spc_rat` must be cast to float64 before it reaches PyBondLab, or numba breaks.
- A duplicate `(cusip, date)` silently corrupts the sort rather than raising. Every
  producer checks.
- Do not filter to `ret_type == 'standard'`: the published factors keep defaulted bonds.
- Three sample windows coexist — Section 3 and Section 5 use T = 268, Section 4 T = 269,
  and the zoo runs to the panel's own frontier. All give 4 Newey-West lags.

### Where the paper disagrees with itself

Two exhibits ship in **two variants**, because the published table and the paper's own
definitions differ. Both are produced; neither is silently corrected.

| exhibit | as published | the other variant |
|---|---|---|
| Table B.1 | `tableB1.tex` differences the stored series without undoing the extract-time sign flips | `tableB1_corrected.tex` undoes them first. The run reports how many end/begin pairs are flip-mismatched; for those, the as-published Δ is −(r_End + r_Bgn) rather than a bias estimate |
| Table IA.IX | `table_ia09.tex` places `b_rvol` in "Vol. & Liq. Betas", where the printed table puts it | `table_ia09_dictionary.tex` places it in "Macro & Other Betas", where the paper's own signal dictionary puts it |

Table 3 also prints one factor (`b_dunc6`) that Table 4 does not quantify — the paper's
own 16-versus-15 gap. It is emitted and marked, with `--quantified-only` to drop it.

---

## Layout

```
stage3/
  _stage3_settings.py     every path and constant; nothing hard-coded
  _run_stage3.py          the entry point: 39 steps, producers then exhibits
  run_stage3.sh           contract check, then the above
  paths.py                paths derived from the settings
  pblenv.py               which PyBondLab, asserted and fingerprinted
  drrlib.py               loading, Newey-West, CAPM_B alpha, paired difference, manifests
  helper_functions.py     the four small conventions, in one place
  captions.py             every table caption, keyed by LaTeX label
  bench.py                phase timings + each run's own completeness check
  fastrun.py              process-parallel fan-out for the grids
  latex_format.py         house number formatting ($-$ minus, {,} thousands)
  make_report.py          assembles every exhibit into one compiled PDF
  s0_data/                the data appendix
  s1_lib/                 Section 3 — latent implementation bias
  s2_lab/                 Section 4 — look-ahead bias
  s3_nse/                 Section 5 — non-standard errors
  s4_zoo/                 the factor zoo
  spec/inputs.json        the input contract
  tools/check_inputs.py   enforces it
  tests/                  the package's own tests
```

> ⚠ **Naming trap.** `s1_lib/` implements the paper's **Section 3**; `s3_nse/` implements
> its **Section 5**. The folder numbers are the order the sections were built, not the
> section numbers.

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for the column contract and
[QUICKSTART_stage3.md](QUICKSTART_stage3.md) for a minimal run.
