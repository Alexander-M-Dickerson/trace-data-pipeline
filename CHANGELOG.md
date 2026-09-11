# Changelog

All notable changes to the TRACE Data Pipeline project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing pending.

---

## [3.2.0] - 2026-09-11

**Stage 3.** The pipeline now goes all the way from the raw TRACE tape to the exhibits the
data was built for: **32 table files and 11 figures** reproducing *The Corporate Bond Factor
Replication Crisis* -- main text, appendix and Internet Appendix -- from the Stage-2 monthly
panel. Twenty-eight of the tables are the paper's; the other four are Stage 3's own -- two
alternate variants where the paper disagrees with itself (below) and the two uncaptioned
inline count blocks from Section IA.3. Stage 3 runs on your own machine, like Stage 2, and
needs no WRDS connection.

### Added

- **`stage3/`** -- 9,341 Python lines: five section packages plus the shared layer they all
  use (paths, the engine selector, the statistics library, the fan-out, the report
  assembler). Two kinds of step, and the difference
  matters: **producers** run sorts through PyBondLab and save return series (minutes to tens
  of minutes; the two uncertainty grids are the long ones), and are SKIPPED when their output
  exists, so a run that stops can simply be run again. **Exhibits** read those series and
  render; seconds each.

  Each section computes its statistics **once** into a tidy frame, and every table and figure
  is a formatter over that frame. Nothing downstream of a statistics frame refits a
  regression -- which is why a section's figures cannot quietly disagree with its tables, and
  where they could, the figure drivers check themselves against the statistics engine at 1e-9.

  | section | what it produces |
  |---|---|
  | data appendix | Tables A.1-A.3 and IA.I-IA.VII |
  | Section 3 | Tables 1, 2, IA.XII-XIV, B.1; Figures 3, 4, IA.1 |
  | Section 4 | Tables 3, 4, IA.XV, IA.XVI; Figures 6, 7, 8, IA.2 |
  | Section 5 | Tables 5, 6, IA.XVII-XIX; Figures IA.3-IA.6 |
  | factor zoo | Tables IA.IX-IA.XI and the two inline count tables |

- **`stage3/tools/check_inputs.py` + `spec/inputs.json`** -- the input contract: five files,
  198 required-column entries across them (193 distinct names), checked for rows, columns and
  date span before anything long starts. It reports every problem at once, so a missing column
  surfaces in a second rather than forty minutes into a grid. It also warns -- not fails --
  when `pdflatex` is absent, since `--no-compile` is a legitimate way to run.

  ❗`spec/inputs.json` is **maintained by hand**. It is the contract Stage 3 asserts against,
  not a description generated from the code, so a producer that starts reading a new column
  will not appear there until someone adds it. `tests/test_stage3_contract.py` checks the
  contract against the settings, which catches the common drift but not that one.

- **`stage3/pblenv.py`** -- which PyBondLab produced a number is part of that number. It puts
  the chosen build first on `sys.path`, **asserts the import resolved inside it** (a second
  install in the environment can otherwise shadow it silently), and records the build's
  version, git state and a content hash in every manifest Stage 3 writes.

- **`stage3/make_report.py`** -- assembles all 43 exhibits into **`reports/exhibits.pdf`**
  (49 pages), in the paper's order and under the paper's own exhibit numbers. It is the
  last step of every run rather than an afterthought, and that is the point: a LaTeX
  fragment that will not compile looks perfectly fine sitting on disk. Wiring the compile
  into the run is what catches it.

  Its title page is a provenance page. It states that these are **not** the paper's
  printed numbers and that nothing in the document compares them -- a PDF of tables under
  familiar captions is exactly the kind of artifact that gets mistaken for the original.
  It also lists **every distinct PyBondLab build that contributed**, with what each
  produced and whether it carried the fast kernels: a producer is skipped when its output
  exists, so re-rendering exhibits later under a different build is normal, and naming
  only one engine would misdescribe most of the document.

- **`stage3/tests/`** -- 49 tests that pass with no configuration at all, which is the
  fresh-clone case: no absolute user path anywhere; every runner step names a script that
  exists AND every driver is in the runner; every rendered exhibit has a caption; the input
  contract and the settings agree; `check_inputs` fails rather than skips on a missing input;
  every window gives 4 Newey-West lags; no shared module sits in the package unimported.

  Four of them scan **what a run writes**, not the code: no absolute path and no private
  reference in any `data/**/*.json` or in `reports/exhibits.tex`. Those files are gitignored,
  so they never reach a reviewer through `git diff` -- but they are exactly what travels in a
  zip or on OSF, which is how a replication package normally travels, and a leak was found in
  one before the gate existed.

### Notes for anyone reading the output

- ❗**Two exhibits ship in two variants**, because the published table and the paper's own
  definitions differ. Both are produced; neither is silently corrected. `tableB1.tex`
  differences the stored series as published, without undoing the extract-time sign flips;
  `tableB1_corrected.tex` undoes them first, and the run reports how many end/begin pairs
  were flip-mismatched. `table_ia09.tex` places `b_rvol` where the printed table does;
  `table_ia09_dictionary.tex` where the paper's own signal dictionary does.
- ❗**A trailing `*` on a factor means the series was sign-corrected**, and the decision is
  made from the full-sample mean of **the sample that was sorted**. Two runs over different
  windows can legitimately disagree about which factors are starred. Compare flip *sets*, and
  never difference a starred series against an unstarred one without re-orienting both.
- ❗**Only Section 5 needs a PyBondLab build carrying the fast kernels** (`fast_sorts`,
  `anomaly_assay_fast`), which the pinned 0.2.0 release does not have. Point `PYBONDLAB_DIR`
  at one; `pblenv.require_fast()` checks before the fan-out starts rather than letting 108
  workers each fail on an import.

  Everything else genuinely runs on the pinned release: `_run_stage3.py` asks the installed
  engine once at startup and takes the slow path automatically, with the same numbers --
  roughly fourteen times the sort time (43.8 s against 3.2 s, measured on one sort).
  `--no-fast` forces the slow path even when the kernels are there, which is how you check
  the two agree. The pin is NOT floated to fix any of this -- Stage 2's factor series depend
  on it exactly. There is also no minimum version to quote: at the time of writing the build
  with the kernels and the release without them both report `0.2.0`, so `pblenv` looks for
  the modules rather than comparing version strings.

### Changed

- **`README.md`** carries Stage 3 in the which-machine table and in the step-by-step
  walkthrough (steps 11-12); **`QUICKSTART.md`** carries it in the which-machine table and
  points at `stage3/QUICKSTART_stage3.md` for the run itself. Both state plainly that Stage 3
  is optional: stages 0-2 build the data, and the panel is useful on its own.
- **`requirements.txt`** gains a Stage 3 section. It adds no new packages -- everything Stage 3
  imports was already required -- but it explains the PyBondLab situation above, where someone
  would otherwise be tempted to float the pin.

---

## [3.1.0] - 2026-09-11

**Public-readiness.** Stages 0-2 were complete but the repository was not something to hand a
stranger: it named a private repository, carried a personal WRDS account, cited three dozen
documents that do not exist here, and shipped two scripts that fail on a fresh clone. This
release is that clean-up, plus the sample-end fix the 2026 vintage needed.

### Added

- **`stage0/DATA_DICTIONARY.md`** -- Stage 0 had no data dictionary. The only complete column
  list lived in the FAQ; `README.md` documented 16 of the 21 panel columns and
  `stage0/README_stage0.md` 14. The new dictionary covers the 21-column bond-day panel and the
  **23-column FISD attribute file**, which was undocumented anywhere despite being an input to
  both Stage 1 and Stage 2.
- **Column-input classification** in `stage2/lib/contract.py` -- every panel column is now
  classified by what it is computed FROM: signed trade prints, within-month daily returns,
  trade dates, or factors that are themselves built from the trade tape. 51 of the 140 need
  something beyond a month-end price. `column_input()` answers it for one column.
- **`contract.assert_mmn_twins`** -- every price-based signal must have its unadjusted
  `<col>_mmn` twin in the sidecar. Both forms are published, and pairing them the wrong way
  round is the mistake it exists to stop: on short-term reversal the unadjusted form has
  AR(1) -0.22 against the adjusted form's -0.05. Called by step 7 against the sidecar it just
  wrote.
- **`stage2/tests/test_propagation_drill.py`** -- skips each step of adding a column and proves
  the matching gate fires.

### Changed

- **`DATE_CUT_OFF` now defaults to `"auto:complete"`** -- the last month for which every source
  has data through its final trading session. `"auto:-3mo"` measures three whole months back
  from a date that is itself mid-month, so it discards complete months: on the 2026-09-10 run
  Enhanced ends 2025-12-04, so it cut at 2025-09-30 and threw away October and November. Both
  are ordinary months -- 11,848 and 11,612 Enhanced bonds at 10.9 and 11.1 trades per bond-day,
  against a 2025 range of 11.3-11.9k bonds and 10-12 trades. Only December is unusable:
  Enhanced stops on the 4th and no bond trades in its last seven days. `auto:complete` returns
  2025-11-30 there. The month's final session is taken from the data, not an exchange calendar,
  so a month whose last session is the 28th is not mistaken for a short one, and a month needs
  at least 80% of the trailing median number of trading days to qualify. `auto:-Nmo` still
  works, for reproducing an older vintage.
- **Provenance of the pre-2002 factor backfill is now stated precisely.** It is estimated on
  the **Lehman Brothers Fixed Income Data** (also known as the Warga Fixed Income data) and the
  investment-grade and high-yield **Bank of America (BAML) constituent bonds** distributed by
  the **Intercontinental Exchange (ICE)**. Those bond data are licensed and cannot be
  redistributed; the finished factor series can be, and is published.
- **`stage2/DATA_DICTIONARY.md` documents the redaction.** It described full identifiers and a
  1..22 rating scale while the release README told downloaders those columns were nulled and
  collapsed -- the two disagreed about the file in the user's hands.
- **`returns_alt`'s five columns** moved from prose bullets to dictionary rows, and
  `test_column_contract.py` now checks that file. Adding a column to it previously failed
  nothing, anywhere.
- **`QUICKSTART.md` covers Stage 2**, which it had never mentioned, and its file tree carries
  `stage2/`. **`CONTRIBUTING.md`** names pytest and the one command that runs ~110 tests, and
  states that five of six `stage2/tests/test_column_contract.py` tests skip without a built
  panel -- so that gate passes vacuously on a fresh clone.

### Fixed

- ❗**`stage2/check_external_data.py` raised `KeyError` on a fresh clone.** It indexed
  `AUX["moody"]` and `AUX["sp"]`, which this configuration does not define, while building its
  source list at import time -- so it failed before `main()` ran, for exactly the people it is
  for. `AUX` and `GOLDEN_OUTPUTS` are now read as the optional things they are.
- **`stage2/tests/test_pin_golden.py` used `cfg.GOLDEN_DAILY_INPUT`**, which is defined
  nowhere. It is `cfg.daily_input()`.
- **The `auto:-Nmo` basis was the POOLED last trade date**, which is 144A's. From the
  2026-09-10 log: pooled last trade `2026-06-05`, Enhanced max `2025-12-04`, `auto:-3mo`
  resolved to `2026-03-31` -- every month from 2025-12 on is one in which no Enhanced bond can
  have a month-end price. The spec is now measured from the least current source; Standard
  (db_type 2) is not a separate population, since Stage 1 clips it to start where Enhanced
  ends. New: `_stage1_settings.cut_off_basis` / `last_complete_month` /
  `resolve_cut_off_from_data` / `source_frontiers`, and `tests/test_cut_off_basis.py`.
- **`stage2/validate_stage2.py --help` raised `KeyError: 'returns'`** -- the CLI built
  argparse's `choices` from a dict that is empty in a public clone, so the exception fired
  before argparse could print anything. Covered by `stage2/tests/test_validate_cli.py`.
- **Documentation that told a reader something false**: the version footer said 2.2.3 against
  a 3.0.0 CHANGELOG; Stage 2 was badged `IN DEVELOPMENT` nine lines above "Code complete";
  `stage2/README_stage2.md` said "the build engine lands next" after the engine shipped and was
  released; the FAQ invited people to email for beta access to code already in the repository.
  The row counts in `README.md` and `FAQ.md` quoted the run the cut-off bug produced
  (28,662,808 rows to 2025-12-31) rather than the shipped 31,344,732 to 2025-11-28; "50+
  signals" against the actual 108; and the repository tree pointed at a Stage 1 reports
  directory that does not exist.

### Removed

- **The private-panel registry left `stage2/lib/contract.py`.** It described two panels this
  repository cannot build -- their external filenames, a frontier pin and a pre-2002 coverage
  gate with no code path here. What replaced it is more useful to a public reader: the
  column-input classification above.
- **Pointers a reader cannot follow** -- a private repository and two files inside it, an
  internal handover document, an assumptions ledger, a debug log, line-number citations into a
  reference implementation that is not distributed, and the internal gate IDs. The substance
  stays; the coordinates do not. Also a personal WRDS account and login hostname.

---

## [3.0.0] - 2026-09-10

**Stage 2 ships**: the monthly asset-pricing panel, 140 columns per bond-month, built from
your own Stage 1 output on your own machine. A major version because Stage 2 is a new
public artifact with a frozen column contract, and because two of its numbers change what
earlier work would have produced.

### Added

- **`stage2/`** — the monthly panel. Seven steps, each in a fresh process (step 2 is ~25 s
  that way and ~91 s in a reused one; DuckDB's parallelism collapses in a long-lived
  process). Measured end to end on 24 cores: 125 s, 111 s, 31 s, 68 s, 53 s, 13 s, 61 s.
  - `_run_stage2.py` with `--dry-run`, `--from-step/--to-step`, `--limit-cusips`,
    `--factor-source`, `--validate`
  - `lib/` and `steps/`, ported from a validated reference implementation and proven
    identical to it key-by-key across 12 outputs before any change was made
  - `validate_coverage.py`, which separates a real coverage failure from an upstream one
- **`lib/contract.py`** — the 140 column names **and their order**, frozen and asserted at
  the end of every build. Changing it is a public API change.
- **The data report** — `_build_data_report.py` + `run_build_data_reports.sh`: 14 tables and
  11 figures, and a PDF where `pdflatex` exists. Comparison suites against the DFPS and
  WRDS bond databases are on by default; `--no-external` drops them and needs no network.
- ❗**`make_release.py` REDACTS the panel before publishing it, and refuses to package one
  that is not redacted.** This is the most consequential fact about the published file and it
  was missing from these notes. `permco` and `gvkey` are set to null and `spc_rat`/`mdc_rat`
  are collapsed to investment grade (1) against non-investment-grade and default (11);
  `permno` is kept. The identifiers are proprietary and the agency ratings are licensed, so
  **REDISTRIBUTION** is restricted -- a panel you build from your own WRDS subscription keeps
  every identifier and the full 1..22 scale. `assert_publishable` refuses a frame that still
  carries them, and `stage2/tests/test_release_redaction.py` (10 tests) pins both halves.
  The packager also derives the vintage year from the data rather than hard-coding it.
- **`lib/frontier.py`** -- refuses to publish a final month that is not a real cross-section.
  Stage 1 pools TRACE Enhanced with 144A, which report on different lags; when the cut-off
  lands where Enhanced has a few days and 144A has the full month, the panel still emits that
  month but only 144A bonds have a month-end price. Measured on the 2026-09-09 run: 1,914
  bonds, 100% 144A, against a trailing median of ~10,700 at ~22%. `make_release` stops unless
  `--truncate-frontier` is given. `stage2/tests/test_frontier.py`, 9 tests.
- **Published factor panel** — `factors_<YYYY>.parquet` ships with each vintage, so
  `--factor-source pinned` reproduces a released number exactly. Public factor sources
  revise; without the pin, next month's build would not match this month's release.
- **Documentation** — `README_stage2.md`, `QUICKSTART_stage2.md`, `README_Value.md`, and a
  `DATA_DICTIONARY.md` that now covers the `_mmn` sidecar's 38 columns, every factor series,
  and the near-duplicate pairs above 0.97.
- **Gates** — the panel's columns must equal the report's definitions and the dictionary's
  mnemonics; the documented factor-model table must equal `BETA_MODELS`; no two panel
  columns may be bit-identical.

### Changed

- ❗❗**`volume_filter_toggle` now defaults to `False`.** It shipped as `True` with a
  `("dollar", 10000)` threshold, which does not match the data OSBAP publishes: the
  published panels keep trades of **all** sizes. A user who cloned this repository and
  ran it could not reproduce the published series, and would have had no way to tell --
  nothing failed, the numbers were simply different.

  The floor is not small. Measured against the previously published vintage on 27,136,631
  overlapping bond-days:

  | | no floor (published) | with the $10,000 floor |
  |---|---|---|
  | trades per Enhanced bond-day | 8.34 | 6.15 (−26%) |
  | bond-days retained | — | ~93% in every year 2010-2024 |
  | daily `pr` identical | — | 67.4% |
  | daily `qvolume` identical | — | 65.8% |
  | monthly `ret_vw` identical | — | 54.0% |

  144A is barely touched (3.92 to 3.85 trades per bond-day, 95.4% identical) because its
  denominations are institutional; the floor removes retail-sized Enhanced trades. The
  bond universe is unchanged -- 538 of 68,140 CUSIPs -- so it removes trades, not
  securities.

  Confirmed against the raw tape rather than inferred: on the bond-days where a
  floored build and the published data disagree most, the published trade count tracks
  the RAW total and the floored one tracks the `>= $10,000` subset, ratios matching to a
  few percent. `30219GAN8` on 2018-09-19 has 6,488 raw trades of which 418 clear the
  floor; the published data reports 6,506 and a floored build reports 432.

  `stage0/README_stage0.md` documented the default as `True`, matching the code and not
  the data. Both are corrected, and the threshold is now documented as applying only when
  the toggle is explicitly turned on.

- ❗**`b_defb` is now a real default beta.** It was not one. The DEF model regressed on
  `mktbx` alone and `defb` was only an output rename of that loading — no `defb` series
  existed anywhere. In the duration-adjusted panel `FACTOR_SWAP` rewrites `mktb` to `mktbx`,
  which made the DEF and MKTB models *the same regression*: `b_defb` came out
  **bit-identical to `b_mktb`** over 1.8 M rows. Both premia are now built for real —
  `DEFB` = long-corporate minus long-government return, `TERMB` = long-government minus the
  risk-free rate — and estimated in one two-factor regression, which is also better
  conditioned than the specification the old documentation claimed (VIF 1.33 against 6.67).
  `corr(b_defb, b_mktb)` is now 0.94, not 1.00.
- ❗**`b_defb` and `b_termb` start 1997-12**, matching every other beta. They began in
  2003-07 until the published pre-TRACE factor series gained the two columns.
- ❗**Firm identifiers come from Stage 1**, not from a separate issuer-level linker. The
  link is now bond-level and dated rather than issuer-cusip6 and forward-filled. `gvkey`
  coverage falls from 95.0% to 86.5%: the old linker carried 111,430 rows (6.2%) with a
  gvkey but no permno, and the new one is permno-anchored by construction. Firm-level
  results move; this is the better link.
- **Factors are rebuilt from public sources by default** (`--factor-source public`) rather
  than read from a private pin. Correlation with the pinned vintage exceeds 0.997 on every
  column, and the public build runs fresher.
- `b_cptlt`, `b_dcpi` and `b_cpi_vol6` end before the panel does — He-Kelly-Manela have not
  published past 2025-05, and FRED's CPIAUCSL is missing an observation. Upstream limits,
  now reported as such instead of failing the coverage gate.

### Fixed

- **30 rows were missing from every descriptive table in the data report.** The statistics
  helper skips a variable it cannot find, and the report asked for `mod_dur`, `conv` and
  `pr` while the panel carries `md_dur`, `convx` and no price column. Three variables ×
  ten panels, no error anywhere. The variable list is now checked against the panel up
  front and fails loudly.
- **The report's price was inverted.** `bbtm` is book-to-market — built as `100/pr` — so
  the price is `100/bbtm`, not `bbtm*100`. Defaulted bonds now show a mean price of 57.6
  where the old arithmetic printed 174.
- **The panel's columns were silently reordered** by the DEF/TERM fix: `b_defb` and
  `b_termb` swapped positions. Caught by the new contract, and corrected.
- **The factor-model table documented 32 of 37 models**, two of which no longer existed,
  and named regressors the code does not use — `LVL` and `YSP` are documented as
  two-factor models on `mktb` but run univariate.
- **`ret_vwx` was defined two ways in one sentence**, as `ret_vw - rfret` and as
  `ret_vw - tret`. It is always `ret_vw - tret`.
- **`requirements.txt` was missing DuckDB**, the entire Stage 2 engine, along with
  PyBondLab, scipy and statsmodels — and closed with a claim that everything else was
  standard library, which an AST sweep showed was false. `PyBondLab` is pinned to `==0.2.0`,
  verified to reproduce this repository's bond factors bit for bit.
- **`validate_coverage.py` defaulted to a panel name from the reference engine**, so a
  public run found no file.
- **`_run_stage2.py` refused inputs that `build_panel.py` accepted** — the validator and
  the build resolved the auxiliary files by two different paths, only one of which honoured
  the environment overrides.

---

## [2.2.3] - 2026-09-10

A documentation audit run against the code rather than by reading. The gate added in
2.2.0 (`tests/test_docs.py`) checks that files named in the docs exist and that quoted
numbers match the code; it cannot tell whether a sentence is TRUE. This round is that
gap. Two of the nine worked examples in the filter READMEs taught the opposite of what
the code does, and re-deriving them turned up a real bug.

### Fixed
- **Cross-bond leak in `flag_price_change_errors`.** In the
  `use_unique_trailing_median=False` branch the `.shift(1)` sat OUTSIDE the groupby, so
  the first print of each bond took the previous bond's rolling median as its anchor
  (measured: 201.5 instead of NaN, a 150-point phantom displacement that would flag the
  row). Fixed in both engines. **Production output is unchanged** -- all three call
  sites pass `True`, and an A/B against the previous commit over 519,590 real trades
  showed 0 diffs in every filter column, on both engines.
- **`OUTPUT_FORMAT = "csv"` now fails at import instead of hours later.** Stage 0
  honours the setting and writes `.csv.gzip`, but Stage 1 and `_build_error_files.py`
  both call `pd.read_parquet` on a hard-coded `*.parquet` name, so the run died late
  with a misleading *"Expected: stage0/<member>/trace_<member>_<stamp>.parquet"*.
  `stage0/_trace_settings.py` now raises on any value but `"parquet"`, and the four
  docs that advertised the CSV option say so.
- **Worked examples re-derived by executing the filters.** Every number in
  `README_bounce_back_filter.md` and `README_decimal_shift_corrector.md` now comes from
  a run, not from reasoning. Bounce-back Example 4 described a lookahead that
  `par_only=True` skips entirely; Example 5 was titled "Blame Reassignment" and narrated
  a branch that never executes (the candidate opens at the row the doc reassigns FROM).
  All four decimal-shift examples were 4-5 prints long, below the centered anchor's
  `min_periods = w+1 = 6`, so each silently used the forward fallback -- **Example 3's
  verdict flipped from ACCEPT to REJECT**, and Examples 2 and 4 rejected at Gate 1 with
  a raw error of exactly 0.0000, never reaching the gate they claimed to illustrate.
- **Decimal-shift anchor documented correctly.** It is not a median of unique values in
  the window; it is a plain median over a frame de-duplicated on `(id, date, price)`.
  Both fallbacks INCLUDE the point itself, so a bad print contaminates its own anchor.
  Condition 5 misstated the search: only the plausibility condition filters candidates,
  and the remaining gates apply to the argmin winner alone.
- **`anchor` is not "rolling only"** -- the `else` branch is live and gives a
  per-`(id, date)` median.
- **Neither filter sorts.** Both READMEs and both engines' docstrings claimed a sort by
  `[id_col, date_col, time_col]`. Verified by AST: no `sort_values`, no `reset_index`,
  and `time_col` never reaches executable code. They use the row order they are given.
- **Individual `qsub` submission was unrunnable as documented** -- the wrappers live in
  `stage0/`, `cd stage0` themselves and log to `stage0/logs/`, so they must be submitted
  from the repository root; and they carry no `-pe`/`-l` directives, so a bare `qsub`
  gives one slot while the code still opens `CONCURRENCY[member]` connections.
- **"Getting the code onto WRDS" built a tree that cannot run.** All three options
  extracted `stage0/` alone, without the root `config.py` it imports or
  `run_pipeline.sh`. They now keep the whole repository.
- **Output trees corrected against a real run.** QUICKSTART showed a
  `stage0/<member>/reports/` folder that does not exist (reports go to
  `stage0/data_reports/<member>/`) and listed 2 of the 9 files each member writes; the
  stage-0 README used `*_audit_*` globs that missed three more.
- Two more instances of the pre-2.2.2 "Stage 1 is held on the report job" claim.
- Ten `## ` sections missing from three tables of contents; a dead `ForkLift` URL; the
  filter READMEs' citation year (2024 -> 2025); a factor set missing `0.01`.
- **Two broken code fences.** A stray closing ``` in `stage0/quickstart.md` swallowed
  three paragraphs into a code block, and `stage1/README_stage1.md` opened an empty
  ```` ```python ```` immediately before another one. Neither is visible in the source.
  The stage-0 one also still said "Submits three SGE jobs: Enhanced, Standard, 144A" --
  Standard has been opt-in since 2.2.0.

### Added
- **`tests/test_docs.py` gains six structural checks**, for the classes that came back
  after being fixed: output trees vs the filename table the stage-0 writer actually
  loops over (per file -- pooling let one doc cover another's omission), `qsub <script>`
  paths resolving from the repo root, TOC completeness, in-page anchors, `OUTPUT_FORMAT`,
  and code-fence balance. Each was mutation-tested: break the guarded thing and the
  suite goes red with the right message. The qsub check found a second bad path on its
  first run that hand-reading had missed.
  The module docstring now states the limit: these checks cannot tell whether a sentence
  is TRUE. Every one of them passed while nine worked examples carried invented
  arithmetic and one carried the wrong verdict.

### Notes
- No numeric path changed. Gates green: `test_docs`, `test_chunk_plan`,
  `test_chunk_scheduler`, 0 broken in-page anchors across 13 docs.

---

## [2.2.2] - 2026-09-10

The data-report job took 50.9 minutes of the 2026-09-09 run. Investigating whether its
WRDS pulls could be sped up found that the pulls were 17% of it, and that the job did
not need to be on the critical path at all.

### Changed
- **Stage 1 no longer waits for the data reports.** It holds on the stage-0 data jobs,
  so the reports run ALONGSIDE it instead of before it. Stage 1 never read anything the
  report job produces -- it reads exactly two paths, the member panels and the FISD file
  (`stage1_pipeline.py:267` and `:424`); the `data_reports` it refers to is its own. That
  chain put ~51 minutes on the wall clock for nothing. The two overlap safely: the
  stage-0 jobs have released their connections by then, so the reports (5) plus stage 1
  (1) sit at 6 against the measured ceiling of 7.
- **The Enhanced report re-clean now pulls 5 chunks at once**, on the same scheduler and
  connection pool stage 0 uses (`REPORTS_CONCURRENCY`, `STAGE0_REPORT_WORKERS` to
  override). Measured: that loop was 45.5 of the job's 50.9 minutes, 83% of it clean
  rather than fetch, at 0.092 ms/row with a 0.991 correlation to row count -- the same
  work stage 0 does at the same per-worker rate, done one chunk at a time. A packing
  simulation over the 46 real chunk times gives 9.2 min at 5 workers (4.9x), matching
  what stage 0 measured. **144A's report clean is left serial on purpose: it takes 19
  seconds.**
- `stage0/run_build_data_reports.sh` asks for `-pe onenode 5 -l m_mem_free=8G` (40 GB)
  rather than 32 GB on one slot. Measured peak RSS on the serial run was 4.78 GB.

### Fixed
- ❗**An `IndexError` that would have killed the report run at the very end.** In
  `create_daily_standard_trace.error_checks` the CUSIP check did
  `chunk_index = 0 if len(cusip_chunks) == 1 else i` against a loop that is
  `enumerate(cusip_chunks, start=1)`. With more than one chunk it validated every
  iteration against the NEXT chunk's CUSIP set -- a meaningless `missing` count -- and
  the last iteration indexed one past the end. It had never fired only because 144A's
  flagged universe has fit in a single chunk. Both engines now validate against the
  chunk actually being processed.
- The decimal-shift re-application in `error_checks` sat outside its
  `if f["decimal_shift_corrector"]:` block, so turning that filter off raised
  `KeyError: 'dec_shift_flag'`; and the CUSIP check read `bb_cusips`/`ds_cusips`, which
  exist only when their filters ran.
- **The report loop logged two different chunk numbers for one iteration** --
  "Processing chunk {i+1}" and "Chunk {i}: took". Pairing a log by the printed number
  therefore mismatches rows against times, which sent this very investigation down a
  false trail (an apparent 325k-row chunk taking 198 s next to a 1.7M-row chunk taking
  64 s, suggesting a pathological per-CUSIP cost that does not exist). Now consistent.

### Added
- **`./run_smoke_test.sh --with-figures`**. `--with-reports` alone could not cover any
  of the above: `error_checks` sits inside `if STAGE0_OUTPUT_FIGURES:` and the harness
  sets that to 0, so it went green having executed none of it. Even with figures on, the
  smoke's flagged universe is ~14 CUSIPs -- one chunk -- so the scheduler would take its
  serial path. `--with-figures` turns figures on AND sets `STAGE0_REPORT_CHUNK_SIZE` so
  the universe splits across several chunks and the concurrent path actually runs.

### Verification
`error_checks` run serially and across 4 workers over the same 6 chunks of real WRDS
data returns identical `dfds`, `dfbb` and `dfie` (`DataFrame.equals`) and identical
CUSIP lists -- those three frames are what draw every figure. The 276-line chunk body
was moved verbatim, diffed mechanically. Smoke 28/28 with the pool exercised (3 workers,
3 connections, 4 chunks), and all 18 stage-0 files still byte-identical to the reference
banked before the concurrency work began.

---

## [2.2.1] - 2026-09-10

A stage-1 bug fix. The 2026-09-09 full run cleared stage 0 in 2 hours and then lost
~1.5 hours of stage-1 work to a duplicate-key abort that named the wrong culprit.

### Fixed
- **Stage 1 aborted with "N duplicate (cusip_id, trd_exctn_dt) rows after the linker
  merge."** The linker did not cause it. `fisd.fisd_mergedissue` is one row per
  **issue_id**, not per CUSIP, and step 6 left-joined it on `cusip_id` to fill missing
  offering amounts without collapsing it first. A bond holding two issue records --
  `29357JAC0` carries an ABS row and a CDEB row with the same issuer, maturity and
  coupon -- therefore had EVERY one of its bond-days duplicated. 268 of them survived
  the filters on that run.

  The lookup is now collapsed to one row per CUSIP (`_offering_amt_by_cusip`, keeping
  the largest offering amount -- the same rule the `mergent_amounts` dedup above uses,
  and the one that selects the same issue record stage 0's FISD screen keeps).

  Reproduced against live WRDS data before and after: 304 bond-days x 2 issue records
  -> 608 rows before, 304 -> 304 after, offering amounts still fully populated.

  Note the cross-database dedup was NOT at fault and is unchanged: step 2 already sorts
  on `(cusip_id, trd_exctn_dt, db_type)` and keeps first, so Enhanced beats 144A and the
  panel leaves step 2 unique. The Enhanced/144A tape overlap is real -- 365 bond-days
  over 252 CUSIPs measured locally -- and was already handled.

- **Every many-to-one key join now fails AT the join, naming the key.** The panel's
  uniqueness was not re-checked anywhere between step 2 and step 7, so a defect created
  in step 6 surfaced five steps and an hour and a half later. `_merge_1to1` checks the
  right frame's key before merging and reports the offending values; it now guards the
  FISD characteristic merge (step 4), the cusip->issue_id map and the call-dummy join
  (step 6), as well as the offering-amount lookup. All four are safe against today's
  data -- the point is that a future FISD vintage gaining multiplicity fails in seconds
  with the key named.

- **The step-7 duplicate check no longer blames the linker.** It is a whole-frame
  duplicate count, not a merge-integrity check, and `merge_asof` on a by-key cannot fan
  out. It now says so, reports the row count it did not change, and names example keys.

- **Late aborts that would kill a multi-hour run with an unhelpful message**:
  `interest_frequency` and `issue_id` used a bare `.astype(int)` that raises on any NaN
  a FISD left-join leaves; both now report and handle the gap.

- **A guard that could never fire.** `if 'table1_tex' not in globals()` in step 10 was
  always False -- both names are bound to `None` at module scope. It now tests the
  value, which is what was meant.

### Changed
- **Stage 1 requests `-pe onenode 4 -l m_mem_free=10G` (40 GB)** instead of 24 GB on a
  single slot. It was already running 4 joblib workers inside a 1-slot allocation --
  `_stage1_settings.py` resolves `N_CORES` from `$NSLOTS` -- and the panel reached
  16.35 GB by step 6, with steps 5, 8 and 10 each peaking near 2x the panel while they
  concatenate chunk frames. `m_mem_free` is charged per slot, so this is 4 x 10G against
  the WRDS caps of 8 cores and 48 GB.

### Added
- **`tests/test_merge_keys.py`** -- asserts, in ~30 seconds and without a pipeline run,
  the key uniqueness each merge depends on, and reproduces the fan-out above before and
  after the fix. The smoke test could not catch this class: it asserts panel uniqueness
  and passed 28/28, because a four-chunk sample contains no multi-issue CUSIP. The
  property that matters is not "is the output unique" but "is every lookup keyed the way
  its join assumes".

---

## [2.2.0] - 2026-09-09

Stage 0 was the pipeline's long pole: ~4 hours for Enhanced, spent in a chunk loop
that ran strictly one CUSIP chunk at a time on a single WRDS connection while the job
held a whole compute node. This release runs those chunks concurrently.

### Added
- **Concurrent chunk fetching.** Enhanced pulls 5 CUSIP chunks at once, each worker
  process holding its own WRDS connection; 144A runs alongside on one; Standard, when
  requested, runs afterwards and may use the whole budget. Set by `CONCURRENCY` in
  `stage0/_trace_settings.py`, overridable per run with `STAGE0_WORKERS`.
- **`stage0/_wrds_pool.py`** -- one connection per worker, opened inside the child so
  no socket is inherited across a fork, with a serialised staggered handshake.
- **`stage0/_chunk_runner.py`** -- row-balanced chunk planning plus the scheduler,
  which returns results in chunk order however they completed and aborts the run if
  any chunk is missing.
- **Row-balanced chunks.** Chunks are packed to ~750,000 trade rows instead of a fixed
  250 CUSIPs. Measured over the Enhanced universe (111,727 CUSIPs / 345,874,974
  trades), the worst chunk falls from 3,392,802 rows to 749,992 -- a 4.5x cut in the
  memory a job must reserve, since that is set by the worst chunk and not the average.
  `chunk_size` keeps its old meaning for the report job's own chunking.
- **Grid resource requests** derived from each member's worker count and validated
  against the WRDS caps (8 cores, 48 GB per job) before submission. `m_mem_free` is
  charged PER SLOT, so an over-request does not error -- it pends forever, silently.
- **A test suite**, the repo's first: `run_smoke_test.sh` runs stage0 -> reports ->
  stage1 on a few chunks in minutes and asserts 28 cross-stage invariants;
  `tests/test_chunk_plan.py` and `tests/test_chunk_scheduler.py` cover the partition
  and scheduling properties without needing WRDS; `tests/probe_wrds_connections.py`
  measures your own account's connection ceiling.
- **`download_inputs.sh`** -- stage 1's external inputs (Liu-Wu yields, the bond-firm
  linker, the FF industry files) split out of `run_pipeline.sh` into their own script.
  They must be fetched on the LOGIN NODE, because compute nodes have no internet, so a
  submitted smoke test could never fetch them itself. It now exits non-zero when a file
  is missing instead of warning and continuing: every one is required for stage 1's 44
  columns, so a miss is a run that dies hours later having burned the grid time.
- **Canonical output ordering.** Stage 0 sorts by `(cusip_id, trd_exctn_dt)` before
  export. Row order no longer depends on the work plan, which is what makes a
  before/after comparison meaningful at all. The row SET is unchanged.

### Changed
- **`TRACE_MEMBERS` now drives submission**, not just what later stages read.
  `run_pipeline.sh` used to submit all three members regardless.
- **The default is `["enhanced", "144a"]`. Standard is opt-in.** Stage 1 keeps
  Standard rows only after the last Enhanced date, so nearly all of a Standard run was
  being discarded. Ask for it with
  `TRACE_MEMBERS="enhanced standard 144a" ./run_pipeline.sh`.
- Per-filter log lines are collected per chunk and emitted by the parent in chunk
  order, so the `.out` reads the same at any worker count.
- The NYSE trading calendar is built once per run rather than rebuilt inside every
  chunk (185 ms -> 12 ms per chunk).

### Fixed
- **`db_type` was assigned by POSITION in `TRACE_MEMBERS`** (`db_type = i` over
  `enumerate`), so dropping a member silently renumbered the rest. With the new
  default, 144A would have become `db_type = 2`, and the overlap clip immediately
  below keeps `db_type == 2` rows only after the last Enhanced date -- deleting
  almost every 144A row. The job would have exited 0 with a normal-looking file.
  Reproduced deliberately before fixing: 1,362 of 1,527 144A rows destroyed, 11%
  retention, exit code 0. Now an explicit `DB_TYPE_BY_MEMBER` map, with a startup
  assertion that every configured member has a code.
- **`-hold_jid ${J1},${J2},${J3}` broke when a member was not submitted**, leaving an
  unset variable and a malformed `-hold_jid 123,,125`. The hold list is now built from
  the jobs actually submitted, verified across four member sets.
- **`DATE_CUT_OFF` could auto-roll past the treasury curve** and abort any run
  including 144A -- a regression from 2.1.0's own auto-roll, caught by 2.1.0's own
  guard. Now clamped to the last date the curve covers.
- **The report job could not find stage-0 files if the run crossed a date boundary.**
  It resolves the stamp from its own start date, probing +/-1 day; a longer gap failed
  outright. It now falls back to the newest complete set on disk, as stage 1 already
  did.
- **`create_daily_stage1.py` located `stage1_pipeline.py` via `ROOT_PATH`** instead of
  its own directory, so it broke under a redirected root.
- **A refused WRDS connection was fatal where a dropped one was retried.** Both
  engines' `_raw_sql_with_retry` now recognise it. Note that the wrds package reports
  EVERY connect failure as `EOFError: EOF when reading a line`, because its failure
  path calls `input()` -- that covers both a missing username and the connection
  limit, and the pool now distinguishes them.
- **`_run_*_trace.py` had no `__main__` guard**, which on a spawn platform would have
  had every pool worker restart the whole pipeline recursively.
- The `ChainedAssignmentError` FutureWarning is suppressed by message, so stage-0
  `.err` logs stay readable without hiding every other warning.

Three more surfaced the first time the smoke test was SUBMITTED to a real grid rather
than run locally, which is the only way it is meant to be used on WRDS:

- **`run_smoke_test.sh` resolved the repo to SGE's spool directory.** `qsub` does not run
  the script where it sits -- it copies it into the spool tree and runs the copy -- so
  `dirname "$BASH_SOURCE"` gave `/gridware/sge/default/spool/<node>/job_scripts` and
  every path built from it pointed nowhere. The visible symptom was the run reporting
  five external inputs as missing while they sat in the repo. It now tries
  `SGE_O_WORKDIR`, then `PWD`, then the script's own directory, taking the first that
  actually contains `stage0/` and `stage1/`.
- **`#$ -o smoke/logs/smoke.out` on a fresh clone.** SGE opens the output file before
  running a line, and that directory does not exist until the script creates it, so a
  fresh clone went straight to `Eqw` having executed nothing. Output now goes to
  `smoke_test.out` in the repo root.
- **Every shell script was committed as `100644`**, so a fresh clone on Linux gets
  "Permission denied" from `./download_inputs.sh` or `./run_pipeline.sh`. Long-standing,
  and caused by `core.filemode=false` on the Windows clone these are authored from, where
  `chmod +x` never reaches a commit. Set explicitly with `git update-index --chmod=+x`;
  `run_pipeline.sh` also now invokes its sibling through `bash` so it does not care.

### Documentation
- Repository structure listings in `README.md` and `QUICKSTART.md` rebuilt against the
  actual file list. They had drifted: both showed `stage0/QUICKSTART_stage0.md` and
  `stage1/requirements.txt`, neither of which exists, and neither listed `config.py`,
  `FAQ.md`, `stage1/stage1_pipeline.py` or `stage1/DATA_DICTIONARY.md`.
- **`run_all_trace.sh` no longer exists but was still referenced 23 times** across the
  FAQ and both stage-0 guides. Replaced with `run_pipeline.sh` throughout.
- **The stage-0 quick start told you to set `WRDS_USERNAME` in `_trace_settings.py`.**
  That has not worked since the shared `config.py` was introduced -- `_trace_settings.py`
  imports the value from there, so editing it does nothing. Corrected, with the
  environment-variable route given first.
- `OSBAP_Linker_*.parquet` renamed to `bond_firm_linker_2026/` wherever it appeared;
  the linker changed in 2.1.0 and the docs had not followed.
- `ff12num` shipped in 2.1.0 but the headline feature lists still said "17 and 30".
- `CONTRIBUTING.md` documents the test suite and states the bar for any stage-0
  scheduling change: byte-identical parquet output against a banked reference.

### Verification
Concurrent output is byte-identical to serial, on both engines, with chunks completing
out of order and with several chunks per worker: all stage-0 parquet files match,
audit tables included, and the replayed filter logs match character for character.
The measured WRDS connection ceiling on the development account is 7 held
simultaneously (the 8th fails); the budget leaves one spare for a mid-run reconnect.

Confirmed live on the WRDS grid: five workers opened five connections inside one second
and ran five chunks in 18 s of wall clock against 81.3 s of serial work (4.5x), with
completions arriving out of order and all 28 smoke assertions still passing. The
`-pe onenode 5 -l m_mem_free=8G` request placed immediately.

---

## [2.1.0] - 2026-09-09

### Added
- **`ff12num`**: the Fama-French 12 industry classification, alongside the existing
  FF17 and FF30. Verified against an independent build across all 69,091 traded
  CUSIPs with zero differences -- as were `ff17num` and `ff30num`, proving they were
  not disturbed.
- **`PRICE_NORM`** (Stage 0, on by default): rescales unit-quoted bonds to percent of
  par. Small-denomination issues ($10, $25, $100 notes) are quoted in unit dollars,
  so a $10 note at par prints `10.00` rather than `100` -- and every filter
  downstream assumes percent of par, so those bonds read as deeply distressed with
  tenfold-understated volume. **This is a no-op under the default settings**, because
  `principal_amt_eq_1000_only` keeps only $1,000-principal bonds; it matters when you
  turn that screen off.
- **Auto-rolling `DATE_CUT_OFF`**: accepts `"auto:-Nmo"` (default `"auto:-3mo"`) and
  resolves against the data's last trade date, so the sample end tracks the data
  instead of needing an edit each vintage. A fixed `"YYYY-MM-DD"` still works.
- **`limit_chunks`** (Stage 0): process only the first N CUSIP chunks, so a config
  change can be checked in minutes rather than a ~4-hour run. Default `None`.
- **Fail-fast input validation**: `validate_config()` now checks for the files
  `run_pipeline.sh` downloads on the login node, and Stage 1 asserts that the
  treasury curve covers the panel before spending an hour on analytics.
- **`.gitattributes`**: pins the shell runners to LF endings. On a Windows clone they
  previously came down as CRLF, which `bash` on the WRDS cloud rejects.

### Changed
- **BREAKING for `permno` / `permco` / `gvkey` consumers.** The bond-firm linker is
  now bond-level and dated: one row per (9-character CUSIP, ownership window), rather
  than issuer-CUSIP-6 matched to a calendar month and forward-filled. Bonds are
  attributed to the firm that owned them *at the time* rather than to whichever firm
  owned them last. Measured against the previous linker on a 30.4M-row panel:
  3,424 bonds gain a link, 4,808 are relabelled, 1,424 are absent from the new
  linker, and 4,177 lose their identifiers outside the ownership window -- mostly
  bonds still trading after the firm's equity stopped being listed. Row-level
  coverage falls from 89.94% to 87.90%. That is the intended direction: a missing
  link is an answer, a stale one is a silent error.
  The release also ships `fl_verdicts.parquet` (every refusal and its reason) and
  `firm_names.parquet` (permno to a dated firm name).
- `gvkey` remains `Int32`; the source ships it zero-padded, so re-pad to 6 characters
  before joining to Compustat.
- `issuer_cusip` is no longer a join key. It was already absent from the output.

### Fixed
- **Step 5 I/O amplification.** Each chunk read the whole accumulated parquet back,
  concatenated and rewrote it -- roughly 25 GB of I/O to produce a 2.5 GB file, while
  holding three copies of the data inside a 24 GB job. Chunks now write part files
  that are concatenated once. Output is unchanged.
- **Worker over-subscription.** `N_CORES` defaulted to the host's core count while
  Grid Engine grants this job a single slot, and joblib copies data per worker. It
  now defaults to 4, honours `STAGE1_N_CORES` or `NSLOTS`, and is capped by the real
  core count. `calculate_credit_spreads` no longer falls back to a hard-coded 10.
  *No scheduler directive changed*: `m_mem_free` is a per-slot request here, so
  adding `-pe onenode N` would multiply the memory request N-fold.
- **Artifact stamp mismatch.** One run could emit `stage1_20251206.parquet` beside
  `sp_ratings_20251118.parquet`, and a run crossing midnight could stamp its own
  outputs with two different dates. Every artifact of a run now shares one stamp.
- Removed dtype casts for columns already dropped; aligned the ultra-distressed
  filter's defaults with the config the pipeline actually passes (documentation only
  -- the filter's behaviour is unchanged); escaped the last invalid escape sequence,
  so the repo compiles clean under `-W error::SyntaxWarning`.
- Documentation corrections: the data dictionary listed 7 columns that are not in the
  output and is now checked against the real schema; `bond_amt_outstanding` is in
  **thousands of dollars** (the README said millions, the dictionary said "bond
  units"); Stage 0's output files are `trace_<member>_YYYYMMDD.parquet`, not
  `<member>_YYYYMMDD.parquet`; and three claims in the 2.0.0 notes above did not
  match the code (no OAS is computed; the ratings come from Mergent FISD rather than
  an unnamed WRDS source; SIC codes come from FISD's issuer table, not from CRSP via
  PERMNO).

### Investigated -- no change
- **`dated_date` filtering** is retained. Of 17,845 FISD CUSIPs with no `dated_date`,
  only **5** actually trade in TRACE, so relaxing the screen would gain nothing.
- **`bond_amt_outstanding` scaling** is correct as-is: raw FISD `amount_outstanding`
  in $ thousands, with no rescaling anywhere in the pipeline.
- **Standard TRACE (`db_type=2`) never survives Stage 1.** Standard rows are kept only
  for dates after the last Enhanced date, and any trailing cutoff falls before that,
  so the two conditions cannot both hold. This was already true under the previous
  fixed cutoff -- which is why shipped Stage 1 files contain only db_type 1 and 3.

### Considered, not included
Order-flow measures (`qbuy`, `qsell`, `order_imbalance` and their 28-day trailing
sums), O'Hara-Zhou realized half-spreads, a quoted `bid_ask_bps`, ask-side symmetry
(`ask_last`, `ask_time_ew`, `ask_time_last`), and any change to Stage 0's serial
chunk loop.

---

## [2.0.0] - 2025-12-11

### Added - Stage 1 Release (Bond Analytics)

#### Core Features
- **Complete bond analytics pipeline** enriching Stage 0 daily data with comprehensive metrics
- **Automated orchestration** via `run_pipeline.sh` (handles both Stage 0 and Stage 1)
- **Research-ready output** with ~50+ variables per bond-day observation

#### Bond Analytics via QuantLib
- **Yield-to-maturity (YTM)** calculations using QuantLib bond pricing engine
- **Macaulay duration** and **modified duration** (interest rate sensitivity)
- **Convexity** (second-order price sensitivity)
- **Credit spreads** computed against Liu-Wu zero-coupon treasury yields
- **Robust error handling** for bonds with missing or invalid parameters
- **Efficient multi-core processing** with joblib parallelization

#### Credit Ratings Integration
- **S&P ratings** from Mergent FISD (`fisd.fisd_ratings`, `rating_type='SPR'`)
  - Numeric ratings (1-22 scale)
  - NAIC designations
- **Moody's ratings** from Mergent FISD (`fisd.fisd_ratings`, `rating_type='MR'`)
  - Numeric ratings (1-21 scale)
- **Automatic rating alignment** with bond-month observations

#### Equity Identifiers 
- **CRSP identifiers**: PERMNO and PERMCO
- **Compustat identifier**: GVKEY

#### Industry Classifications
- **Fama-French 17 industry classification**
- **Fama-French 30 industry classification**
- **SIC code mapping** from the issuer's SIC code in Mergent FISD

#### Ultra-Distressed Bond Filters
- **Price anomaly detection** to identify suspicious observations
- **Five-stage filtering methodology**:
  1. **Anomalous price detection**: Ultra-low prices with normal price context
  2. **Upward spike detection**: High prices inconsistent with recent trading
  3. **Plateau sequence detection**: Sustained ultra-low price sequences
  4. **Intraday inconsistency**: Wide intraday ranges at distressed prices
  5. **Round number detection**: Suspicious exact prices (0.01, 0.10, etc.)
- **Refined composite flag** (`flag_refined_any`) combining all detection methods
- **CUSIP-level export** tracking all flagged bonds with detailed statistics
  - Export file: `stage1/data/ultra_distressed_cusips_{date}.csv`
  - Includes flag counts, percentages, and date ranges

#### Treasury Yield Integration
- **Liu-Wu zero-coupon treasury yields** (1961-present)
  - Downloaded automatically from public source
  - Monthly interpolated yields (1-30 years maturity)
  - Used for credit spread calculations
- **FRED treasury yields** (alternative source, configurable)

#### Configuration & Settings
- **Harmonized configuration system** with single source of truth
  - `config.py`: Shared settings across all stages
  - `TRACE_MEMBERS`: Dataset selection (enhanced, standard, 144a)
  - `STAGE0_OUTPUT_FIGURES`: Control Stage 0 error plots (slow)
  - Stage 1 always generates comprehensive reports (no toggle)
- **Auto-detection features**:
  - Stage 0 date stamp from parquet files
  - CPU core count optimization
  - Root path detection
- **Minimal user configuration** required (just WRDS username)

#### Performance Optimizations
- **Memory efficiency**:
  - CUSIP columns use category dtype (~75% memory savings)
  - Optimized groupby operations for 30M+ row datasets
  - Efficient parquet compression
  - Strategic garbage collection
- **Processing speed**:
  - Multi-core parallelization for bond analytics
  - Chunked processing for large datasets
  - Vectorized operations throughout pipeline
- **WRDS quota monitoring**:
  - Pre-flight disk space check before pipeline execution
  - Parses WRDS quota (not filesystem) for accurate warnings
  - Warns if < 4 GB available (prevents job failures)
  - `FORCE_RUN=1` override for advanced users

#### Output Files & Reports
- **Comprehensive daily bond dataset** (`stage1_YYYYMMDD.parquet`)
  - All Stage 0 price/volume metrics
  - FISD bond characteristics
  - QuantLib analytics (duration, convexity, YTM, OAS, spreads)
  - Credit ratings (S&P and Moody's)
  - Equity identifiers (PERMNO, PERMCO, GVKEY)
  - Industry classifications (FF17, FF30)
  - Ultra-distressed filter flags
- **LaTeX data quality reports** 
  - 8 comprehensive tables analyzing data quality
  - Time-series visualization plots
  - Filter effect summaries
  - Organized output structure
- **Flagged CUSIP export** for quality control
  - CSV file with all ultra-distressed flagged bonds
  - Statistics per CUSIP (total obs, flagged obs, percentages)
  - Breakdown by flag type
  - Date range for each flagged bond

#### Documentation
- **Comprehensive Stage 1 documentation**:
  - `stage1/README_stage1.md`: Full technical documentation
  - `stage1/QUICKSTART_stage1.md`: Quick start guide
  - `stage1/README_distressed_filter.md`: Ultra-distressed filter methodology
- **Updated FAQ** with Stage 1-specific sections
  - Configuration guidance
  - Output file descriptions
  - Troubleshooting disk space warnings
  - Performance optimization tips
- **Updated main README** reflecting Stage 1 availability

#### Infrastructure & Automation
- **Unified pipeline orchestrator** (`run_pipeline.sh`)
  - Pre-stage: Download required data files (Liu-Wu yields, OSBAP linker, FF classifications)
  - Stage 0: Parallel TRACE extraction (Enhanced, Standard, 144A)
  - Stage 0: Report generation after extraction
  - Stage 1: Bond analytics after Stage 0 completion
  - Automatic job dependency management with SGE `-hold_jid`
- **Disk space validation**:
  - Checks WRDS user quota before execution
  - Prevents pipeline failures from insufficient space
  - Clear warnings with remediation steps
- **Automatic data downloads** on login node (WRDS compute nodes have no internet)
  - Liu-Wu treasury yields
  - OSBAP linker file (ISIN/FIGI/Bloomberg identifiers)
  - Fama-French industry classifications (FF17, FF30)

#### Runtime Performance
- **Stage 1 processing time**: ~3 hours (WRDS Cloud, 2-4 cores)
- **Complete pipeline (Stage 0 + Stage 1)**: ~7-10 hours total
  - Stage 0 (Enhanced): ~4 hours
  - Stage 0 (Standard): ~30-60 minutes
  - Stage 0 (144A): ~30-60 minutes
  - Stage 0 (Reports): ~30-60 minutes
  - Stage 1: ~2 hours

### Changed
- **Configuration structure** now harmonized across all stages
  - `config.py` is single source of truth for shared settings
  - Removed redundant `TRACE_MEMBERS` from `stage1/_stage1_settings.py`
  - Removed unused `GENERATE_REPORTS` and `OUTPUT_FIGURES` from Stage 1
  - Stage 0 figure generation controlled via `STAGE0_OUTPUT_FIGURES` in `config.py`

### Fixed
- **Disk space check** now uses WRDS quota instead of filesystem space
  - Previous version showed 8TB+ available (filesystem) when user had <2GB (quota)
  - Now correctly parses `quota` command output
  - Accurate warnings prevent job failures from disk space exhaustion
- **CUSIP export performance** optimized for 30M+ row datasets
  - Replaced O(n*m) loop-based approach with O(n) vectorized groupby
  - ~1000-5000x faster for typical datasets
  - Completes in seconds instead of hours

---

## [1.0.0] - 2025-11-01

### Added - Initial Public Beta Release

#### Core Processing Pipeline (Stage 0)
- **Enhanced TRACE processing** (2002-07-01 to present)
  - Full intraday to daily conversion pipeline
  - Configurable parameters via `_trace_settings.py`
  - Automated parallel job submission with `run_all_trace.sh`
  - Output to dedicated `enhanced/` subfolder
  
- **Standard TRACE processing** (configurable start date, default 2024-10-01)
  - Pre-2012 and post-2012 cleaning rules
  - Reversal trade handling specific to Standard TRACE
  - Output to dedicated `standard/` subfolder
  
- **Rule 144A TRACE processing** (2002-07-01 to present)
  - Same cleaning pipeline as Standard TRACE
  - Dedicated processing for private placement bonds
  - Output to dedicated `144a/` subfolder

#### Data Cleaning & Error Correction
- **Decimal-shift correction algorithm**
  - Automatic detection and correction of multiplicative price errors
  - Handles 10x, 0.1x, 100x, and 0.01x errors
  - Novel algorithms by Dickerson, Robotti & Rossetti (2025)
  
- **Bounce-back filter**
  - Identifies and removes erroneous price spikes
  - Detects prices that revert quickly to previous levels
  - Configurable threshold parameters
  
- **Dick-Nielsen filters** (2009, 2014)
  - Cancellation filtering
  - Correction filtering
  - Agency trade de-duplication
  - Reversal handling
  
- **van Binsbergen, Nozawa & Schwert filters** (2025)
  - Advanced trade filtering
  - Duration-based validation

#### Data Quality & Validation
- **Price range filters**
  - Minimum price validation (> 0)
  - Maximum price validation (<= 1000)
  
- **Volume filters**
  - Dollar volume thresholds
  - Par volume thresholds
  - Configurable limits per dataset
  
- **Additional filters**
  - Trading calendar validation (NYSE calendar)
  - Time-of-day filtering (configurable windows)
  - Yield != price trade filtering
  - Volume > 50% offering amount filtering
  - Execution date > maturity date filtering

#### Output & Reporting
- **Daily aggregated metrics**
  - Equal-weighted price (`prc_ew`)
  - Volume-weighted price - dollar (`prc_vw`)
  - Volume-weighted price - par (`prc_vw_par`)
  - First trade price (`prc_first`)
  - Last trade price (`prc_last`)
  - Trade count (`trade_count`)
  - Par volume in millions (`qvolume`)
  - Dollar volume in millions (`dvolume`)
  - Customer-side bid price - value-weighted (`prc_bid`)
  - Customer-side ask price - value-weighted (`prc_ask`)
  - Daily high price (`prc_hi`)
  - Daily low price (`prc_lo`)
  - Bid trade count (`bid_count`)
  - Ask trade count (`ask_count`)
  
- **Audit logging system**
  - Transaction-level audit trails
  - Row count reconciliation at each filter stage
  - CUSIP-level correction lists
  - Comprehensive filter effect documentation
  
- **LaTeX report generation**
  - Automated quality reports for each dataset
  - Detailed filtering statistics
  - Optional time-series visualization plots
  - Organized in `data_reports/` with dataset subfolders
  - Bibliography and citation support

#### Automation & Infrastructure
- **Parallel job execution**
  - `run_all_trace.sh` master script
  - SGE job dependency management with `-hold_jid`
  - Automatic report generation after data processing
  - Individual dataset runners: `run_enhanced_trace.sh`, `run_standard_trace.sh`, `run_144a_trace.sh`
  
- **Output organization**
  - Dataset-specific subfolders (`enhanced/`, `standard/`, `144a/`)
  - Centralized reports folder (`data_reports/`)
  - Parquet format for efficient storage
  - Comprehensive log files in `logs/` directory
  
- **WRDS Cloud integration**
  - Password-less authentication via `.pgpass`
  - Efficient chunked processing (default 250 CUSIPs per chunk)
  - Memory-optimized design (~4-8GB per job)
  - Fast execution (~5 hours complete pipeline)

#### Documentation
- **Comprehensive README files**
  - Main project README with overview
  - Stage 0 detailed documentation (`stage0/README_stage0.md`)
  - Contributing guidelines (`CONTRIBUTING.md`)
  - Clear installation and setup instructions
  
- **Configuration documentation**
  - All parameters explained in `_trace_settings.py`
  - Per-dataset override examples
  - Filter parameter descriptions
  - Aggregation metric specifications
  
- **Troubleshooting guide**
  - Common issues and solutions
  - WRDS setup guidance
  - Performance optimization tips

#### Project Infrastructure
- **MIT License**
  - Open source availability
  - Permissive licensing for research use
  
- **Version control**
  - GitHub repository structure
  - Issue tracking setup
  - Pull request templates
  
- **Dependencies**
  - Python 3.10+ requirement
  - Clear `requirements.txt` for Stage 0
  - WRDS subscription requirements documented

#### Academic Integration
- **Citations and references**
  - Primary citation: Dickerson, Robotti & Rossetti (2025)
  - Secondary citation: Dickerson & Rossetti (2025)
  - Acknowledgment of foundational methods
  
- **Open Bond Asset Pricing integration**
  - Part of broader [Open Bond Asset Pricing project](https://openbondassetpricing.com/)
  - Companion [PyBondLab repository](https://github.com/GiulioRossetti94/PyBondLab) for factor construction
  - Reproducible research framework

### Performance Characteristics
- **Runtime (WRDS Cloud)**
  - Enhanced TRACE: ~4 hours
  - Standard TRACE: ~30-60 minutes
  - Rule 144A: ~30-60 minutes
  - Report generation: ~30-60 minutes
  - Total pipeline: ~5 hours
  
- **Resource usage**
  - Memory: ~4-8GB per job
  - Disk: ~1-2GB per dataset (Parquet format)
  - Parallel execution supported
  
- **Data scale**
  - Enhanced TRACE: ~30 million rows (2002-present)
  - Standard TRACE: ~2-3 million rows (2024-present)
  - Rule 144A: ~5-8 million rows (2002-present)

---

## Project Roadmap

### Version 1.x - Stage 0 Enhancements (Ongoing)
- Bug fixes and performance improvements
- Additional filter options
- Enhanced documentation
- Community contributions integration

### Version 2.0 - Stage 1 Release (November 2025)
- Daily bond metrics calculation module
- Duration and convexity measures
- Credit spread computation
- Yield calculations

### Version 3.0 - Stage 2 Release (November 2025)
- Monthly panel construction
- 50+ bond characteristic signals
- Factor construction tools
- Portfolio-ready outputs
- Integration with PyBondLab

---

## Support & Contribution

For questions, issues, or to contribute:
- **Email**: alexander.dickerson1@unsw.edu.au
- **GitHub Issues**: [trace-data-pipeline/issues](https://github.com/Alexander-M-Dickerson/trace-data-pipeline/issues)
- **Collaboration**: Beta testers and collaborators welcome!

---

## Acknowledgments

This pipeline implements and extends methods from:
- Dick-Nielsen, J. (2009). Liquidity biases in TRACE. *The Journal of Fixed Income*, 19(2), 43-55.
- Dick-Nielsen, J. (2014). How to clean enhanced TRACE data. Working Paper.
- van Binsbergen, J. H., Nozawa, Y., & Schwert, M. (2025). Duration-based valuation of corporate bonds. *The Review of Financial Studies*, 38(1), 158-191.
