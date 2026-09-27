# Stage 0 with an AI assistant

Stage 0 runs on the WRDS grid. It pulls TRACE trades from WRDS one chunk of CUSIPs at a time,
cleans them, and writes one daily bond panel per TRACE database: Enhanced, Rule 144A, and
Standard when the user asks for it. Read the repository's [AGENTS.md](../AGENTS.md) first; the
full guide is [README_stage0.md](README_stage0.md).

## How it runs

- `./run_pipeline.sh`, from the repository root, submits one SGE job per database, then the
  report job and stage 1, each waiting on the jobs it needs [ref:entry.wrds]. The job scripts
  `run_enhanced_trace.sh`, `run_144a_trace.sh` and `run_standard_trace.sh` start
  `_run_enhanced_trace.py`, `_run_144a_trace.py` and `_run_standard_trace.py`, which run the
  cleaner.
- **There are two cleaners.** `create_daily_enhanced_trace.py` cleans Enhanced TRACE;
  `create_daily_standard_trace.py` cleans Standard TRACE and Rule 144A. They apply the same
  filters in the same order, each in its own copy of the code, so a fix to one usually belongs
  in the other. Their shared filters are tagged as groups: `grep -rn "filter.decimal_shift" .`
  finds both copies. Six of the filters are also written out a second time in each cleaner, in
  the code that rebuilds the data reports; those copies are members of the same groups, so a
  group can have four members, and all of them change together.
- Several chunks are fetched at once, one WRDS connection per worker (`CONCURRENCY`). The output
  is sorted by `(cusip_id, trd_exctn_dt)` before it is written, so it comes out byte-identical
  whether one worker ran the chunks or six.

## The filters, in the order they run

The FISD screens choose the bonds first [ref:filter.fisd_universe]. Then, for each chunk:

| # | filter | tag |
|---|---|---|
| 0 | price-scale normalization | [ref:filter.price_scale] |
| 1 | cancellations, corrections and reversals (Dick-Nielsen) | [ref:filter.dick_nielsen] |
|   | Enhanced before 2012-02-06, after it, and agency de-duplication | [ref:filter.dn_pre_2012] [ref:filter.dn_post_2012] [ref:filter.agency] |
|   | Standard and 144A reversals | [ref:filter.reversals_standard] |
| 2 | decimal-shift corrector | [ref:filter.decimal_shift] |
| 3 | trading time (off by default) | [ref:filter.trade_time] |
| 4 | trading calendar | [ref:filter.calendar] |
| 5 | price range | [ref:filter.price_range] |
| 6 | trade size (off by default) | [ref:filter.volume] |
| 7 | bounce-back | [ref:filter.bounce_back] |
| 8 | yield reported as the price | [ref:filter.yield_as_price] |
| 9 | trade size against the offering amount | [ref:filter.volume_vs_amount] |
| 10 | trades after maturity | [ref:filter.after_maturity] |
| 11 | initial price errors | [ref:filter.initial_price] |

The numbers match the `# Filter N:` comments in the cleaners. The surviving trades are then
summed to one row per bond and day, whose columns carry `daily.` tags (see [TAGS.md](../TAGS.md)).
[README_decimal_shift_corrector.md](README_decimal_shift_corrector.md) and
[README_bounce_back_filter.md](README_bounce_back_filter.md) explain the two subtlest filters.

## Settings

All in `_trace_settings.py`: `FILTER_SWITCHES` turns each filter on or off, `FISD_PARAMS` sets
the bond universe, `DS_PARAMS`, `BB_PARAMS` and `INIT_ERROR` tune three of the filters, and
`CONCURRENCY` and `MEM_PER_SLOT_GB` size the jobs. ❗Write all eleven keys in any
`FILTER_SWITCHES` you write: a key left out falls back to the engine's own default, and for
`volume_filter_toggle` that turns on a $10,000 floor the published data does not have.

## Traps

- **Connections.** WRDS allows 7 held at once; Enhanced and 144A run together and must leave
  one free [ref:rule.connection_cap]. A refused connection, a missing username and a missing
  `~/.pgpass` entry all surface as `EOFError: EOF when reading a line` [ref:trap.eof_error].
- **Memory is charged per slot.** A job asking for more than 8 slots or 48 GB waits in the queue
  forever without an error, so `qsub_resources()` refuses to ask [ref:rule.memory_per_slot]. A
  job stuck at `qw`: lower that database's `CONCURRENCY` and resubmit.
- **Parquet only.** Every later stage reads parquet, so another `OUTPUT_FORMAT` stops the run
  at start-up [ref:rule.parquet_only].

## What it writes

`stage0/<database>/trace_<database>_<YYYYMMDD>.parquet` (the daily panel), the database's FISD
file, the filter audit files and the CUSIP lists of the three flagging filters, plus
`stage0/data_reports/`. The full list is under "Outputs" in [README_stage0.md](README_stage0.md).

## When something fails

Read the job's log in `stage0/logs/`, then "Troubleshooting" in
[README_stage0.md](README_stage0.md#troubleshooting) and the [FAQ](../FAQ.md#troubleshooting).
