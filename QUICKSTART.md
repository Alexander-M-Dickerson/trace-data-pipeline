# TRACE Data Pipeline — Quick Start Guide

Processes TRACE data from intraday trades to a monthly bond panel. One script runs Stages 0
and 1 on WRDS, and Stage 2 then runs on your own computer.

---

## What This Pipeline Does

Transforms raw TRACE data into a research-ready bond dataset with:

- ✨ **Clean TRACE prices** (Enhanced, Standard, and 144A)
- 📊 **Bond characteristics** from FISD
- 📈 **Analytics** (YTM, duration, convexity, credit spreads via QuantLib)
- ⭐ **Credit ratings** (S&P and Moody's)
- 🔗 **Equity identifiers** (PERMNO, PERMCO, GVKEY)
- 🚨 **Quality filters** (ultra-distressed bond detection)
- 🏭 **Industry classifications** (Fama-French 12, 17 and 30)

**Output:** A parquet file with 44 columns per bond-day (see [stage1/DATA_DICTIONARY.md](stage1/DATA_DICTIONARY.md)).

---

## Prerequisites

- ✅ **WRDS account** with TRACE, FISD, and ratings access
- ✅ **WRDS Cloud access** (or local Python environment)
- ✅ **No password file on WRDS**: jobs on the WRDS Cloud connect without one
- ✅ **Python ≥ 3.10** (the 2026-09-21 run used 3.14.5 on the WRDS Cloud)

---

## Two machines, one pipeline

Before you start, know where you will be:

| | Where | What |
|---|---|---|
| Stages 0 and 1 | **WRDS Cloud** | build the daily bond panel from the raw TRACE tape (~5 h) |
| The hand-off | you | zip it on WRDS, `scp` it to your own computer (~5 GB) |
| Stage 2 | **your own computer** | build the monthly panel from that file (~8-18 min) |
| Stage 3 | **your own computer** | sorts, uncertainty grids and the paper's exhibits (~15-20 min, optional) |
| Stage 4 | **your own computer** | the TRACE-only bond factors openbondassetpricing.com publishes, checked against the published files (~8 min, optional) |

Steps 1-3 below are all **on WRDS**. The switch to your own computer happens at
[Download Results](#download-results-to-your-local-machine), and Stage 2 follows it.

---

## Quick Start (3 Steps) — on WRDS

### Step 1: Clone and Configure

```bash
# SSH to WRDS Cloud
ssh <your_wrds_id>@wrds-cloud.wharton.upenn.edu

# Clone the repository
cd ~
git clone https://github.com/Alexander-M-Dickerson/trace-data-pipeline.git
cd trace-data-pipeline
```

**Configure your WRDS username and author:**

The pipeline reads `WRDS_USERNAME` and `AUTHOR` from the environment or `config.py`.

**Option A — Set environment variable (recommended):**
```bash
export WRDS_USERNAME="your_wrds_id"
```

Make it persistent for future sessions:
```bash
echo 'export WRDS_USERNAME="your_wrds_id"' >> ~/.bashrc
source ~/.bashrc
```

**Option B — Edit `config.py` directly:**
```bash
nano config.py
```

Change the default values:
```python
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_username")  # Change to your WRDS ID
AUTHOR = "Your Name"  # Change from default "Open Source Bond Asset Pricing"
```

Save and exit (Ctrl+O, Enter, Ctrl+X).

**Note:** Your password goes nowhere: jobs on the WRDS Cloud connect without it.

---

### Step 2: Install Dependencies

**For Stages 0 and 1** (both need it; stage 0 uses `pandas_market_calendars` and `SQLAlchemy`,
which the system Python may not have):
```bash
python -m pip install --user -r requirements.txt
```

**Alternative - Virtual environment (optional):**
```bash
# Create virtual environment in project root. --system-site-packages keeps the WRDS
# Cloud's own pandas 2.2 visible: pip has no build of it for the Cloud's Python 3.14.
python3 -m venv --system-site-packages venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

**Required packages for Stages 0 and 1:** the list, with minimum versions, is
`requirements.txt`; install it whole: `download_inputs.sh` already needs `duckdb`, to check the
bond-firm linker.

❗Do **not** pin these to exact versions on the WRDS Cloud. `--user` installs shadow the
system packages, and WRDS already ships newer ones than any pin here (numpy 2.4.6,
pyarrow 24.0.0 on Python 3.14) — pinning downgrades a working environment. Use
`requirements.txt`, which states minimums.

---

### Step 3: Run the Pipeline

The scripts ship executable, so a fresh clone needs no `chmod`.

```bash
# OPTIONAL BUT RECOMMENDED: prove the chain works first, in ~10 minutes
bash download_inputs.sh     # LOGIN NODE only -- compute nodes have no internet
qsub run_smoke_test.sh      # result in smoke_test.out
# wait until qstat no longer lists it: it and the full run both hold WRDS connections

# Run the complete pipeline
./run_pipeline.sh
```

**What happens:**

1. **Pre-Stage** (`download_inputs.sh`, on the login node):
   - Liu-Wu treasury zero-coupon yields
   - bond-firm linker (equity identifiers)
   - Fama-French industry classifications

2. **Stage 0** (TRACE data extraction) -- submits exactly the members in
   `TRACE_MEMBERS`, which defaults to Enhanced + 144A:
   - Enhanced TRACE (2002-07 onward), pulling 5 CUSIP chunks at a time
   - 144A TRACE (first trade 2003-10), running alongside it
   - Standard TRACE (2024-present) -- OPT-IN; when requested it runs after the other
     two so it can use the whole WRDS connection budget
   - Data quality reports

3. **Stage 1** (bond analytics):
   - Merge FISD characteristics
   - Compute bond analytics with QuantLib
   - Merge credit ratings
   - Merge equity identifiers
   - Apply quality filters
   - Generate final dataset

**Runtime:** about 5 hours total on the WRDS Cloud with default settings (5.3 h on 2026-09-21, the run the 2026 vintage was built from)

---


## Monitor Progress

```bash
# Check job status
qstat

# Monitor Stage 0 logs
tail -f stage0/logs/01_enhanced.out
tail -f stage0/logs/02_standard.out
tail -f stage0/logs/03_144a.out

# Monitor Stage 1 logs
tail -f stage1/logs/stage1.out

# Check for errors
tail -f stage0/logs/*.err
tail -f stage1/logs/stage1.err
```

Press Ctrl+C to stop tailing logs.

---

## Check Output

```bash
# Stage 0 outputs
ls -lh stage0/enhanced/trace_enhanced_*.parquet
ls -lh stage0/standard/trace_standard_*.parquet
ls -lh stage0/144a/trace_144a_*.parquet

# Stage 1 output (final dataset)
ls -lh stage1/data/stage1_*.parquet

# Data quality reports
ls stage0/data_reports/enhanced/
ls stage1/data_reports/
```

**Expected output structure:**
```
trace-data-pipeline/
├── stage0/
│   ├── enhanced/                                    # nine files per member
│   │   ├── trace_enhanced_YYYYMMDD.parquet          # the daily panel, ~2.2 GB
│   │   ├── trace_enhanced_fisd_YYYYMMDD.parquet     # the FISD universe used
│   │   ├── fisd_filters_enhanced_YYYYMMDD.parquet
│   │   ├── dick_nielsen_filters_audit_enhanced_YYYYMMDD.parquet
│   │   ├── drr_filters_audit_enhanced_YYYYMMDD.parquet
│   │   ├── bounce_back_cusips_enhanced_YYYYMMDD.parquet
│   │   ├── decimal_shift_cusips_enhanced_YYYYMMDD.parquet
│   │   ├── init_price_cusips_enhanced_YYYYMMDD.parquet
│   │   └── cusip_row_counts_YYYYMMDD.parquet        # the chunk plan's row census
│   ├── 144a/                                        # the same nine files, named for the member
│   │   ├── trace_144a_YYYYMMDD.parquet              # (cusip_row_counts has no member name)
│   │   └── trace_fisd_144a_YYYYMMDD.parquet         # note the word order: fisd before 144a
│   ├── standard/                                    # only if you opt Standard in
│   │   └── trace_standard_YYYYMMDD.parquet
│   ├── data_reports/                                # NOT under the member folders
│   │   ├── enhanced/
│   │   │   ├── enhanced_data_report.tex
│   │   │   ├── references.bib
│   │   │   ├── enhanced_fig_page_NNN_{ds,bb}.pdf
│   │   │   └── enhanced_ie_fig_page_NNN_ie.pdf
│   │   └── 144a/
│   └── logs/                                        # NN_member.out / .err, _data_reports.out / .err
└── stage1/
    ├── data/
    │   ├── stage1_YYYYMMDD.parquet                  # THE final dataset
    │   ├── sp_ratings_YYYYMMDD.parquet
    │   ├── moodys_ratings_YYYYMMDD.parquet
    │   ├── call_dummy_YYYYMMDD.parquet
    │   ├── ultra_distressed_cusips_YYYYMMDD.csv
    │   ├── bond_firm_linker_2026/                   # from download_inputs.sh
    │   ├── Siccodes{12,17,30}.txt                   # from download_inputs.sh
    │   ├── liu_wu_yields.xlsx                       # from download_inputs.sh
    │   └── data_reports/                            # the distressed-filter report
    ├── data_reports/                                # the stage-1 data report
    │   ├── stage1_data_report_YYYYMMDD.tex
    │   ├── stage1_figures_YYYYMMDD*.pdf
    │   └── time_series_data/
    └── logs/
```

---

## Download Results to Your Local Machine

The pipeline generates a large folder (~5 GB) with hundreds of files. **The recommended approach is to zip the folder first**, then download a single file.

### Step 1: Zip the Folder on WRDS (via SSH)

Your WRDS home directory has limited space (~10 GB). Use the scratch space (~500 GB) to create the zip file.

**Connect to WRDS Cloud:**
```bash
# Windows (PowerShell, Windows Terminal, or PuTTY)
ssh {wrds_username}@wrds-cloud.wharton.upenn.edu

# Mac/Linux (Terminal)
ssh {wrds_username}@wrds-cloud.wharton.upenn.edu
```

**Create the zip file in scratch space:**
```bash
cd ~
rm -f "/scratch/$(basename "$(dirname "$HOME")")/trace-data-pipeline.zip"   # zip would otherwise ADD to an old archive
zip -r "/scratch/$(basename "$(dirname "$HOME")")/trace-data-pipeline.zip" trace-data-pipeline/
```

This compresses the folder and stores it in scratch space, avoiding home directory quota issues.
`$(basename "$(dirname "$HOME")")` is your institution code, the folder between `/home` and your username. Type the line as written; nothing in it is a placeholder. For the `scp` line below, `{institution}` IS a placeholder: replace it, braces included, with that code (`echo $HOME` on WRDS shows it).

> ❗`zip` stores whatever path you hand it. Given the absolute `~/trace-data-pipeline/`,
> it strips the leading `/` and stores `home/{institution}/{wrds_username}/trace-data-pipeline/...`,
> so the archive extracts four directories deep. Run it from `~` on a **relative** path,
> as above, and the archive extracts to a single clean `trace-data-pipeline/` folder.


### Step 2: Download the Zip File to Your Local Machine

**From your LOCAL machine** (not WRDS), run:

**Windows (PowerShell or Windows Terminal):**
```powershell
scp {wrds_username}@wrds-cloud.wharton.upenn.edu:/scratch/{institution}/trace-data-pipeline.zip "{local_destination}"
```

**Mac/Linux (Terminal):**
```bash
scp {wrds_username}@wrds-cloud.wharton.upenn.edu:/scratch/{institution}/trace-data-pipeline.zip "{local_destination}"
```

**Windows (WinSCP - GUI alternative):**
1. Connect to `wrds-cloud.wharton.upenn.edu` with your WRDS credentials
2. Navigate to `/scratch/{institution}/`
3. Download `trace-data-pipeline.zip`

### Step 3: Extract the Zip File Locally

**Windows:**
- Right-click `trace-data-pipeline.zip` → **Extract All...**, and remove the trailing
  `\trace-data-pipeline` from the folder it proposes. The zip already holds that folder, so
  keeping it gives `trace-data-pipeline\trace-data-pipeline\`.

**Mac:**
- Double-click `trace-data-pipeline.zip` (extracts automatically)

**Linux:**
```bash
unzip trace-data-pipeline.zip -d "{local_destination}"
```

### Step 4: Clean Up (Optional)

After confirming the download, remove the zip from scratch space:
```bash
# On WRDS Cloud
rm /scratch/{institution}/trace-data-pipeline.zip
```

### Placeholder Reference

| Placeholder | Description | Example |
|-------------|-------------|---------|
| `{wrds_username}` | Your WRDS username | `jsmith` |
| `{institution}` | Your institution's WRDS scratch folder | `wharton`, `chicago`, `nyu` |
| `{local_destination}` | Path on your local machine | `~/Downloads` (Mac/Linux) or `C:\Users\YourName\Downloads` (Windows) |

---

## Verify Your Data

This reads only the file's metadata, so it is safe on the WRDS login node, where heavy work
is not allowed:

```python
import pyarrow.parquet as pq

f = pq.ParquetFile('stage1/data/stage1_YYYYMMDD.parquet')   # your date
print(f"{f.metadata.num_rows:,} rows, {len(f.schema_arrow.names)} columns")   # 44 columns
print(f.schema_arrow.names)
```

---

## Now switch to your own computer: Stage 2

Everything above happened **on WRDS**. Stage 2 happens **on your own computer**, on the
files you just downloaded. Its first run connects to WRDS once, to fetch and cache a few
series (CRSP Treasury returns, Fama-French factors, VIX and FISD coupon terms), so it needs
your WRDS username, in `config.py` or `export WRDS_USERNAME=...`; later runs read the cache.

First install the requirements for stages 2-4 there, in Python 3.11 to 3.13, in an environment of
their own (the install replaces packages other projects may rely on):

```bash
cd trace-data-pipeline          # the folder you just unzipped
python -m venv .venv
source .venv/bin/activate       # macOS/Linux; Windows: .venv\Scripts\activate
python -m pip install -r requirements-local.txt
python -m pip install --no-deps pybondlab==0.3.0
```

Activate it again in every new terminal before running stages 2-4. (If Windows PowerShell refuses
to run `activate`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.)

PyBondLab 0.3.0 declares `numpy<2` and this repository installs numpy 2, so it goes in without
its dependency list; `requirements-local.txt` says why that is safe. Stages 2-4 stop at start-up
and print these two lines if PyBondLab is missing or a different version.

❗**On Windows, before you `git pull` a newer release into this folder**, run
`git config core.fileMode false` in it once. Unzipping on Windows drops the executable bit from
every `.sh` file, git then counts each one as changed, and the pull stops with "Please commit your
changes or stash them before you merge". Nothing in the files changed; the setting tells git to
ignore the bit.

Then build the panel:

```bash
cd stage2
python _run_stage2.py --dry-run     # resolve and validate the config, build nothing
python _run_stage2.py               # the full build, ~8-18 minutes on 24 cores
```

`--factor-source pinned` uses the factor file published with your vintage instead of
rebuilding it from public sources that revise their history. It reproduces a published panel only when stages 0 and 1 are the same WRDS run; a new run ends later and carries WRDS's revisions, so its panel differs whichever factors it uses. See [stage2/QUICKSTART_stage2.md](stage2/QUICKSTART_stage2.md). To match it exactly in 139 of 145 columns, the other six within 1e-13, also install the exact
package versions it was built with:
`python -m pip install -r requirements-local.txt -c constraints-2026.txt`, which the file
explains.

It writes `output/panel/main_panel_<mode>.parquet` -- 145 columns per bond-month -- plus the
unadjusted `_mmn` twins, the factor series, and the beta and momentum blocks. Every column is
defined in [stage2/DATA_DICTIONARY.md](stage2/DATA_DICTIONARY.md).

The build asserts its own column contract at the end: the 145 names **and their order** are
frozen in `stage2/lib/contract.py`, so a change to the model list cannot silently permute the
published file.

To package a vintage for distribution:

```bash
python make_excess_blocks.py --mode stage1 --verify           # first: the default rebuilt, checked
python make_excess_blocks.py --mode stage1 --benchmark all   # the blocks the release packs
python make_release.py              # the stage1 build; --mode <mode> for another
```

which redacts the proprietary identifiers and licensed ratings, and refuses to write a bundle
that still carries them. See [stage2/QUICKSTART_stage2.md](stage2/QUICKSTART_stage2.md).

## Optional: Stages 3 and 4

Both run on your own computer and read what Stage 2 wrote, plus Stage 1's daily file (Stage 3's
data appendix, Stage 4's vintage year). Neither connects to WRDS; Stage 4 downloads the
published factor files to compare with.

```bash
cd ../stage3 && bash run_stage3.sh   # 33 tables and 11 figures, ~15-20 min
cd ../stage2 && python make_excess_blocks.py --mode stage1 --verify && \
  python make_excess_blocks.py --mode stage1 --benchmark all   # Stage 4 reads these
cd ../stage4 && bash run_stage4.sh   # the TRACE-only bond factors, then the check against
                                     # the published files, ~8 min
```

See [stage3/QUICKSTART_stage3.md](stage3/QUICKSTART_stage3.md) and
[stage4/README_stage4.md](stage4/README_stage4.md).


## Troubleshooting

### Pipeline fails immediately

**Check:** Are you in the root directory?
```bash
pwd  # Should show: .../trace-data-pipeline
ls   # Should show: stage0/ stage1/ config.py run_pipeline.sh
```

**Fix:** Navigate to root directory
```bash
cd ~/trace-data-pipeline
```

---

### "WRDS connection failed"

**On the WRDS Cloud** no password file is needed: jobs there connect without one. A failed
connection is almost always the username: `WRDS_USERNAME` unset, or still `your_wrds_username`
in `config.py` (`python3 doctor.py --wrds` checks it). If the username is right, it is WRDS's
limit of 7 connections held at once (`stage0/README_stage0.md`).

**On your own computer** (stage 2's first run), the password must be saved where the `wrds`
package looks. Connect once by hand; it asks for the password and offers to save it
(`~/.pgpass`, or `%APPDATA%\postgresql\pgpass.conf` on Windows):
```bash
python -c "import wrds; wrds.Connection()"
```

---

### "ModuleNotFoundError: No module named 'QuantLib'"

**Fix:** Install QuantLib (`requirements.txt` asks for `>=1.36`)
```bash
pip install --user QuantLib
```

---

### Stage 1 fails with "Stage0 output files not found"

**Check:** Did Stage 0 complete successfully?
```bash
ls stage0/enhanced/trace_enhanced_*.parquet
```

**Fix:** `run_pipeline.sh` starts Stage 1 only after Stage 0 has ended, so this means a
Stage 0 job failed. Read its log for the first error, fix it, and run `./run_pipeline.sh` again:
```bash
grep -n -m5 -i "error" stage0/logs/*.err
```

---

### Jobs are running very slowly

**Check:** WRDS system status
```bash
qstat -f  # Check cluster load
```

**Speed up:**
1. Process fewer datasets (edit `TRACE_MEMBERS` in `config.py`)
2. Not the reports: `STAGE0_OUTPUT_FIGURES = False` in `config.py` skips their figures, but the
   report job runs beside Stage 1, so the run does not finish any sooner

---

### Memory errors ("Killed")

**Fix:** Use more, smaller chunks. A WRDS job may hold at most 48 GB, and Stage 1 asks for
40 GB (4 slots of 10 GB):
```python
# In stage1/_stage1_settings.py
N_CHUNKS = 20    # default 10; more chunks means a smaller peak
N_CORES = None   # leave it: it follows the 4 slots the job was granted
```

---

## What's Next?

After the pipeline completes:

1. **Explore your data**: Load `stage1_YYYYMMDD.parquet` into pandas/R
2. **Read detailed docs**: See `README.md` for variable definitions
3. **Check data quality**: Review reports in `stage1/data_reports/`
4. **Customize filters**: Edit `stage1/_stage1_settings.py` for custom filters
5. **Run incrementally**: Re-run Stage 1 with different settings without re-running Stage 0

---

## File Structure Overview

What each stage reads and writes, and what every file in the repository does:
[CODE_MAP.md](CODE_MAP.md). For which doc answers which question: [INDEX.md](INDEX.md).

---

## Getting Help

- 📖 **Detailed docs**: See `README.md` for comprehensive documentation
- 🚀 **Stage-specific guide**: See `stage1/QUICKSTART_stage1.md` for Stage 1 details
- 📧 **Email**: alexander.dickerson1@unsw.edu.au
- 🐛 **Issues**: [GitHub Issues](https://github.com/Alexander-M-Dickerson/trace-data-pipeline/issues)

---

## Command Reference

```bash
# Initial setup
git clone https://github.com/Alexander-M-Dickerson/trace-data-pipeline.git
cd trace-data-pipeline
nano config.py  # Set WRDS_USERNAME

# Install dependencies (Stages 0 and 1)
python -m pip install --user -r requirements.txt

# Fetch stage 1's external inputs (login node) and check the chain end to end
bash download_inputs.sh
qsub run_smoke_test.sh             # ~10 min -> smoke_test.out
# wait until qstat no longer lists it: it and the full run both hold WRDS connections

# Run complete pipeline
./run_pipeline.sh

# Monitor
qstat                              # Job status ('hqw' = held, waiting: normal)
tail -f stage0/logs/01_enhanced.out # Stage 0 progress
tail -f stage1/logs/stage1.out     # Stage 1 progress

# Check output
ls -lh stage0/enhanced/trace_enhanced_*.parquet
ls -lh stage1/data/stage1_*.parquet

# Download (from local machine) - see "Download Results" section above
# Step 1: SSH to WRDS and zip to scratch
ssh {wrds_username}@wrds-cloud.wharton.upenn.edu
cd ~ && zip -r "/scratch/$(basename "$(dirname "$HOME")")/trace-data-pipeline.zip" trace-data-pipeline/

# Step 2: Download zip (from LOCAL machine)
scp {wrds_username}@wrds-cloud.wharton.upenn.edu:/scratch/{institution}/trace-data-pipeline.zip ./
```

---

