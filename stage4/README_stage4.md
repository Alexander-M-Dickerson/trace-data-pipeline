# Stage 4: the TRACE-only bond factors

Stage 4 turns the monthly panel Stage 2 built into the TRACE-only bond factors published on
[openbondassetpricing.com](https://openbondassetpricing.com), then checks yours against the
published files, cell by cell. It runs on your own computer, and nothing in it connects to WRDS.

## What it builds

108 signals, each sorted two ways:

- **single sorts**: deciles over all bonds, quintiles within the investment-grade band and
  within the non-investment-grade band;
- **within-firm sorts**: high minus low within each firm (`permno`), for firms with at least
  two bonds.

Each for four return types and three rating bands (all, IG, non-IG):

| return type | the return sorted |
|---|---|
| `exc` | `ret_vw` minus the one-month T-bill |
| `dur` | `ret_vw - tret`, duration-adjusted against the duration-matched Treasury return |
| `dbns` | the same against `tret_bns` |
| `dcls` | the same against `tret_cls` |

For the three duration-adjusted types, the 68 beta and momentum signals are the ones Stage 2
estimated on that same adjusted return (its blocks `betas_x` and `mom_retx`, or their `_bns`
and `_cls` versions), and `str` is adjusted too. Holding period one month, value weights from
the month before, rebalanced monthly. Every choice is in [`spec/factors.json`](spec/factors.json).

The series are **unflipped**: a factor whose long-short mean is negative stays negative.
`flip_set.json` records which factors the full-sample sign rule would flip, per return type.

## Before you start

1. Stage 2 has finished: Stage 4 reads `stage2/output/panel/main_panel_stage1.parquet`, and
   `betas_x`, `mom_retx` and `factors.parquet` in `stage2/output/blocks/stage1/`.
2. For `dbns` and `dcls`, Stage 2's optional benchmark blocks exist:

   ```bash
   cd ../stage2 && python make_excess_blocks.py --mode stage1 --benchmark all
   ```

3. The requirements for stages 2-4 are installed, in Python 3.11 or newer:

   ```bash
   python -m pip install -r requirements-local.txt
   python -m pip install --no-deps pybondlab==0.3.0
   ```

   PyBondLab 0.3.0 declares `numpy<2` and this repository installs numpy 2, so it goes in
   without its dependency list; [`requirements-local.txt`](../requirements-local.txt) says why
   that is safe.

`python build_factors.py --dry-run` checks all of this and names the command that makes
anything missing.

## Run it

From `stage4/`:

```bash
bash run_stage4.sh                  # build both sorts, then compare with the published files
```

or step by step:

```bash
python build_factors.py --dry-run   # the inputs, and the PyBondLab release in use
python build_factors.py             # both sorts, all four return types
python compare_published.py         # downloads the published files once, then compares
```

About 8 minutes on 24 cores, measured on the 2026-09-21 run's panel. `--sort single` or
`--sort within_firm` builds one sort; `--signals cs mom6_1` is a quick run on a few signals and
writes to `output/_subset/`, so it never replaces a full build. `run_stage4.sh` calls the
`python` on PATH; set `PY=/path/to/python` to choose another.

## What you get

```
stage4/output/
├── single_sort_panel_trace_<vintage>/
│   ├── single_sort_trace_<vintage>.parquet    the long panel: every leg, weighting, band, return type
│   ├── flip_set.json                          what the sign rule would flip
│   └── MANIFEST.json                          inputs by sha256, PyBondLab, row counts
├── single_sort_trace_<vintage>_csv/
│   ├── single_sort_<return>_<band>_<weighting>.csv   24 files: the long-short leg, one column per factor
│   ├── flip_set.json
│   └── MANIFEST.json
├── within_firm_sort_panel_trace_<vintage>/    the same, for the within-firm sorts
└── within_firm_sort_trace_<vintage>_csv/
```

Each folder holds what the matching archive on openbondassetpricing.com holds, apart from its
README. `<vintage>` is the release year, taken from the Stage 1 file's date stamp; set
`STAGE4_VINTAGE` to choose another. The columns are in [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

## Checking against the published files

`compare_published.py` downloads the published TRACE-only archive for your vintage (about
37 MB per sort, kept in `output/_published/`) and reports, per return type and band, the
largest difference in `return`, `turnover` and `count`, the cells missing on one side only, and
whether the flip sets agree. It exits 0 when everything is identical, or within `--tol`.

On the 2026-09-21 run, Stage 4's output is identical, all 4,354,560 rows, to the factors
OSBAP built from that run. A panel built from a newer WRDS run will
differ, as it should; the comparison then tells you by how much.

## Settings

| environment variable | default | what it does |
|---|---|---|
| `STAGE2_DIR` | `../stage2` | where Stage 2's output is |
| `STAGE1_DIR` | `../stage1` | where the Stage 1 file (for the vintage year) is |
| `STAGE4_MODE` | `stage1` | Stage 2's input mode, in the panel's file name |
| `STAGE4_OUTPUT` | `stage4/output` | where the factors are written |
| `STAGE4_VINTAGE` | the Stage 1 file's year | the year in the file names |

## If something goes wrong

| message | what to do |
|---|---|
| `PyBondLab ... is installed, and this repository runs on 0.3.0` | run the two install lines above |
| `missing Stage 2 inputs` | run the command printed under each missing file |
| `... signals are not in the ... panel` | the panel is not a full Stage 2 build; rebuild it |
| `DIFFERS` from `compare_published.py` | expected if your panel comes from a different WRDS run; otherwise check `MANIFEST.json` against the Stage 2 files you meant to use |

## Tests

```bash
python -m pytest tests -q
```

They use small synthetic panels and need no Stage 2 output.
