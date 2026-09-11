# Stage 3 — portfolio sorts, uncertainty grids and the paper's exhibits

Stage 3 sits on top of the Stage-2 monthly bond panel and reproduces the exhibits of
*The Corporate Bond Factor Replication Crisis* (Dickerson, Robotti and Rossetti) — main
text, appendix and Internet Appendix.

It writes **32 table files and 11 figures**. Twenty-eight of the tables are the paper's;
the other four are Stage 3's own — two alternate variants where the paper disagrees with
itself (see below) and the two uncaptioned inline count blocks from Section IA.3.

It runs on your own computer, like Stage 2. It needs no WRDS connection.

**Beyond Stage 2's requirements it needs two things:** `PyBondLab`, installed by the
repository's `requirements.txt`, and **pdflatex** (TeX Live or MiKTeX) for the last step.
`tools/check_inputs.py` warns if pdflatex is missing rather than failing -- without it
every table and figure is still written, you just do not get the assembled PDF.

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
| `data/sorts/` | long-format long-short panels: `date, factor, freq, leg, weighting, return, turnover, count` |
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
| `STAGE2_MMN` | the unadjusted `*_mmn` signal twins | the three approaches (Section 3) |
| `STAGE2_BBW` | `bbw_factors.parquet` — **MKTB** | every CAPM_B alpha |
| `STAGE2_FACTORS` | `factors.parquet` — the risk-free rate, VIX, the macro set | excess returns, Figure 7 |
| `STAGE1_DAILY` | the daily bond-day panel | the data appendix only |

> ❗**`STAGE2_BBW` and `STAGE2_FACTORS` are different files.** MKTB lives in the first,
> the risk-free rate in the second. They are not interchangeable.

`python tools/check_inputs.py` checks all five against `spec/inputs.json`: rows, every
column Stage 3 reads (198 entries across the five files, 193 distinct names), and the
date span. It reports every problem at once, so a missing column surfaces in a second
rather than forty minutes into a grid.

❗`spec/inputs.json` is **maintained by hand**. It is the contract Stage 3 asserts
against, not a description generated from the code, so a producer that starts reading a
new column will not appear there until someone adds it.

---

## PyBondLab

[PyBondLab](https://github.com/GiulioRossetti94/PyBondLab) is the portfolio-sorting
library the paper is built on. `requirements.txt` installs it. Stage 3 runs every sort
through it, so **which copy is on `sys.path` is part of the result**: `pblenv.py` puts
the chosen build first, asserts the import resolved there (a second install in the
environment can otherwise shadow it silently), and records the build's version, git state
and a content hash in every manifest Stage 3 writes.

```bash
export PYBONDLAB_DIR=/path/to/PyBondLab     # a checkout; unset = whatever is installed
```

> ❗**Only the uncertainty grids (`--section nse`) need the fast kernels** —
> `PyBondLab.fast_sorts` and `PyBondLab.anomaly_assay_fast`. The 0.2.0 release that
> Stage 2 pins does not carry them, and `pblenv.require_fast()` says so before the
> fan-out starts rather than letting 108 workers each fail on an import.
>
> **Everything else genuinely runs without them.** `_run_stage3.py` asks the installed
> engine once and takes the slow path automatically — same numbers, longer. Measured on
> one sort: 43.8 s without the kernels against 3.2 s with them. `--no-fast` forces the
> slow path even when they are available, which is how you check the two agree.
>
> At the time of writing, the build carrying the kernels and the released 0.2.0 report
> the **same version string**, so there is no version test to give you — `pblenv` looks
> for the modules themselves. Check what you have with `python pblenv.py`.

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

Measured on the cold run of 2026-09-11, 24 cores with the kernels:

| section | producer | cost |
|---|---|---|
| `lib` | the three approaches, all bonds and both rating splits | 25–28 s each |
| `lib` | the 108-signal month-end/month-begin sorts, x4 | 27–28 s each |
| `lab` | the winsorization sweep, 2 tails x 3 ratings | 19 s |
| `nse` | the **MUA grid** — 108 signals x 216 method choices | 142 s |
| `nse` | the **DUA grid** — 108 signals x 108 filters x 3 ratings | 241 s, then 32 s for its statistics |
| `zoo` | all 108 signals, single and within-firm | 49 s |

**Exhibits** read those series and render. Seconds each, always re-rendered. The final
step compiles them all into one PDF.

❗`reports/exhibits.pdf` carries **your** numbers, from whatever panel Stage 2 built.
Nothing in it compares them to the paper's printed ones, and its title page says so -- a
PDF of tables under familiar captions is exactly the kind of artifact that gets mistaken
for the original.

Figures 1, 2 and 5 of the paper are schematics drawn in LaTeX -- a research framework, a
return timeline, a look-ahead illustration. They have no data behind them, so Stage 3
does not produce them, and the document says so rather than leaving a gap.

Table IA.VIII is the same case: it is the paper's **signal dictionary**, a hand-written
list of the 140 panel fields and what each one means. There is no computation behind it,
so Stage 3 does not emit it either. The Stage-2 `DATA_DICTIONARY.md` is the live version
of that table.

Those timings are with the fast kernels, on 24 cores. Without them a sort takes roughly
fourteen times as long (43.8 s against 3.2 s, measured on one), which is what turns the
grids from minutes into hours — and is what the kernels are for.

---

## What it costs

Measured on a **cold run** — `data/` and `reports/` wiped first — on 24 cores / 128 GB, Windows, with a PyBondLab build carrying the fast kernels, 2026-09-11:

**833 s = 13.9 minutes** end to end, `tools/check_inputs.py` through `reports/exhibits.pdf`.

| section | `--section` | wall | share |
|---|---|---|---|
| Section 5 -- the two uncertainty grids | `nse` | 438 s | 56% |
| Section 3 -- latent implementation bias | `lib` | 215 s | 27% |
| data appendix | `data` | 57 s | 7% |
| the factor zoo | `zoo` | 55 s | 7% |
| Section 4 -- look-ahead bias | `lab` | 23 s | 3% |

Those are the benched steps (787 s of the 833 s); the rest is process start-up, the 40 subprocess launches and the LaTeX compile. `reports/timings.jsonl` carries one line per step, and the run prints its own five slowest at the end.

### Disk and memory

- **Inputs**: 3.9 GB, of which the Stage-1 daily panel is most. They are read, never copied.
- **Outputs**: about 348 MB under `data/` and `reports/` together. Both are gitignored.
- **Peak per grid worker**: 0.7 GB, recorded by the grid itself as `max_worker_rss_gb` (needs `psutil`, which `requirements.txt` installs). The grids are bounded by cores, not by memory.

### If your machine is smaller

**Recommended minimum: 8 cores, 32 GB RAM, about 5 GB free disk.**

❗That figure is **derived from the measurements above, not tested** — we have not run Stage 3 on an 8-core machine. It comes from the two numbers that actually bind: a producer holds the panel plus its own columns (3–4 GB), and a grid worker peaks well under 1 GB, so eight workers and the parent fit inside 32 GB with room to spare. Treat it as a starting point and watch the first grid.

Everything scales down through flags rather than edits. Lower them in this order:

1. **`--workers`** on `s3_nse/run_dua_grid.py` and `s3_nse/run_mua_grid.py`. Both now size themselves from `os.cpu_count()` and fit whichever of workers/threads you did not pin around the one you did, so on a smaller machine the defaults are already smaller. Pin it lower if memory, not cores, is your limit.
2. **`--threads`** — numba threads per worker. Fewer, larger workers beat more, thinner ones once you are oversubscribed.
3. **`--chunk`** on the DUA grid — signals per fit. Smaller chunks hold less at once and checkpoint more often.

Two more knobs worth knowing:

- `STAGE3_MEMORY_LIMIT` caps DuckDB in the data appendix, which is the step that scans the 31-million-row daily panel. Left unset it takes a share of free RAM; set it (`STAGE3_MEMORY_LIMIT=8GB`) if something else on the machine needs the memory.
- `STAGE3_WORKERS` sets the default worker count for every fan-out at once, without touching a flag.

**On 16 GB**, run section by section (`--section lib`, then `lab`, and so on) rather than the whole chain, and pin `--workers 4 --threads 2` on both grids. Section 5 is the one that will hurt; it is also the only section that needs the fast kernels.

Without the fast kernels every sort takes roughly fourteen times as long (43.8 s against 3.2 s, measured on one), so the two grids become hours rather than minutes — which is why Stage 3 refuses to start them rather than letting you find out.

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
- ❗**One denominator, derived in one place.** Section 5 counts "construction paths",
  and the paper counts its grid down in a single ladder whose bottom line is every
  denominator it prints: 216 candidates per signal, less 24 inadmissible (an IG
  breakpoint universe cannot sort high-yield bonds), less 24 redundant (forming within
  IG, IG breakpoints *are* the full-universe ones), = 168 x 108 = 18,144, less those
  that "produce months with empty long or short legs", = well-defined factor return
  series.

  Stage 3 reproduces that ladder explicitly in a **status ledger**
  (`data/s3_nse/mua_summary/mua_status_{window}.parquet`, one row for every one of the
  23,328 cells) and `nse_engine.usable()` is the only thing that reads it. Table 6's
  `N`, Table IA.XVIII's `n_spec` and Table IA.XIX's pool are therefore the same number
  by construction rather than by coincidence. See
  [DATA_DICTIONARY.md](DATA_DICTIONARY.md#the-degeneracy-ledger).

- ❗**A signal's own start or end date is NOT degeneracy.** Signals do not all span the
  panel -- twelve of the 108 start late, and three end early because the underlying data
  does. Every month after a signal's data ends has no bonds, so every leg minimum is
  zero, so a one-sided window reads the whole signal as degenerate. The active window is
  two-sided, and its upper bound comes from the **signal**, never from the cell being
  judged: a cell that sets its own bound can delete the months that prove it empty.

- ❗**Section 5's PATH COUNTS are not reproducible run to run, and the reason is in
  PyBondLab.** Running the identical MUA grid twice over identical data flips a small
  number of restricted-breakpoint-universe cells (`ig_bp`, `lg_bp`) between a full
  279-month series and nothing at all, always in EW/VW pairs -- measured at roughly
  0.1-0.5% of the 23,328 cells. **Every cell that is populated is bit-identical
  between runs**: the arithmetic is stable, and no printed VALUE moves. What moves is
  how many paths there are, and therefore every count Section 5 reports.

  Two contributing causes are fixed here, and both were real: the panel was handed to
  the engine in whatever order DuckDB's parallel scan produced (three consecutive
  loads, three different orders -- now pinned with `ORDER BY date, cusip`), and
  `skip_invalid` let the validator return a different COLUMN SET each run (now off, so
  the grid is a fixed 216 columns). Together they halve it. The remainder is the
  engine's own: it does not reproduce when `assay_anomaly_fast` is called repeatedly
  inside one process, only across the spawned workers, and it is not the numba cache
  or the thread count.

  What Stage 3 does about it: `s3_nse/run_mua_grid.py` counts the affected cells every run,
  prints a warning and records `n_unstable_empty_cells` in the manifest, so two runs
  can be compared; `mua_summarize` reindexes onto the full 216 so the statistics frame
  is a rectangle either way; and `s3_nse/t06_mua_nse.py` **fails** its twin-invariance check
  when it bites, rather than printing a number as though nothing happened.
- Three sample windows coexist. Section 3 and Section 5 run on a fixed T = 268 and
  assert it. Section 4's window **spans** 269 months, but its series are not all that
  long — each is its own length, T 257 to 268 in this build — so there is no single T to
  assert there; every row carries its own. The zoo SORTS to the panel's own frontier and
  every zoo exhibit then truncates to 2024-12-31. Every window in the build gives 4
  Newey-West lags.

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
  _run_stage3.py          the entry point: 40 steps, producers then exhibits then the report
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
  README_stage3.md        this file
  QUICKSTART_stage3.md    a minimal run
  DATA_DICTIONARY.md      every artifact and column
  data/                   everything Stage 3 computes        (gitignored)
  reports/                the exhibits, and exhibits.pdf     (gitignored)
```

> ⚠ **Naming trap.** `s1_lib/` implements the paper's **Section 3**; `s3_nse/` implements
> its **Section 5**. The folder numbers are the order the sections were built, not the
> section numbers.

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for the column contract and
[QUICKSTART_stage3.md](QUICKSTART_stage3.md) for a minimal run.
