# TRACE Data Pipeline — Quick Start Guide

Processes TRACE data from intraday to monthly with one execution script.

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

**Output:** A comprehensive parquet file with 50+ variables per bond-day.

---

## Prerequisites

- ✅ **WRDS account** with TRACE, FISD, and ratings access
- ✅ **WRDS Cloud access** (or local Python environment)
- ✅ **`.pgpass`** configured for passwordless WRDS authentication
- ✅ **Python ≥ 3.10** (Python 3.12+ recommended)

---

## Quick Start (3 Steps)

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
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_username")  # Change to your WRDS ID
AUTHOR = "Your Name"  # Change from default "Open Source Bond Asset Pricing"
```

Save and exit (Ctrl+O, Enter, Ctrl+X).

**Note:** You do **not** need to put your password in code; `.pgpass` supplies it automatically.

---

### Step 2: Install Dependencies

**For Stage 0 (TRACE extraction):**
```bash
# No installation needed - uses system Python with pandas, numpy, wrds, pyarrow
```

**For Stage 1 (bond analytics):**
```bash
# Stage 1 requires additional packages
python -m pip install --user -r requirements.txt
```

**Alternative - Virtual environment (optional):**
```bash
# Create virtual environment in project root
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

**Required packages for Stage 1** (minimums, from `requirements.txt`):
- pandas>=2.2.3, numpy>=2.0, pyarrow>=20.0.0, wrds>=3.3.0, tqdm
- QuantLib>=1.36, joblib>=1.4
- openpyxl, requests, matplotlib>=3.8.0

❗Do **not** pin these to exact versions on the WRDS Cloud. `--user` installs shadow the
system packages, and WRDS already ships newer ones than any pin here (numpy 2.4.6,
pyarrow 24.0.0 on Python 3.14) — pinning downgrades a working environment. Use
`requirements.txt`, which states minimums.

---

### Step 3: Run the Pipeline

```bash
# Make the scripts executable (once per clone)
chmod +x *.sh stage0/*.sh stage1/*.sh

# OPTIONAL BUT RECOMMENDED: prove the chain works first, in ~10 minutes
bash download_inputs.sh     # LOGIN NODE only -- compute nodes have no internet
qsub run_smoke_test.sh      # result in smoke_test.out

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
   - Enhanced TRACE (2002-present), pulling 5 CUSIP chunks at a time
   - 144A TRACE (2002-present), running alongside it
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

**Runtime:** ~4-6 hours total (WRDS Cloud with default settings)

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
ls stage0/enhanced/reports/
ls stage1/data/reports/
```

**Expected output structure:**
```
trace-data-pipeline/
├── stage0/
│   ├── enhanced/
│   │   ├── trace_enhanced_YYYYMMDD.parquet          # ~500MB-2GB
│   │   ├── trace_enhanced_fisd_YYYYMMDD.parquet
│   │   └── reports/
│   ├── standard/
│   │   └── trace_standard_YYYYMMDD.parquet
│   └── 144a/
│       └── trace_144a_YYYYMMDD.parquet
└── stage1/
    └── data/
        ├── stage1_YYYYMMDD.parquet                   # Final dataset
        └── reports/
```

---

## Download Results to Your Local Machine

The pipeline generates a large folder (~6 GB) with hundreds of files. **The recommended approach is to zip the folder first**, then download a single file.

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
cd /scratch/{institution}/
zip -r trace-data-pipeline.zip ~/trace-data-pipeline/
```

This compresses the folder and stores it in scratch space, avoiding home directory quota issues.

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
- Right-click `trace-data-pipeline.zip` → **Extract All...**

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

```python
import pandas as pd

# Load the final dataset
df = pd.read_parquet('stage1/data/stage1_20251119.parquet')  # Use your date

print(f"Dataset shape: {df.shape}")
print(f"\nColumns ({len(df.columns)}):")
print(df.columns.tolist())

print(f"\nSample data:")
print(df.head())
```

---

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

**Check:** Is `.pgpass` configured?
```bash
cat ~/.pgpass
```

Should contain:
```
wrds-pgdata.wharton.upenn.edu:9737:wrds:your_username:your_password
```

**Fix:** Set permissions
```bash
chmod 600 ~/.pgpass
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

**Fix:** Wait for Stage 0 to complete, or check logs for errors:
```bash
tail -f stage0/logs/*.err
```

---

### Jobs are running very slowly

**Check:** WRDS system status
```bash
qstat -f  # Check cluster load
```

**Speed up:**
1. Process fewer datasets (edit `TRACE_MEMBERS` in `config.py`)
2. Disable reports (`STAGE0_OUTPUT_FIGURES = False` in `config.py`)

---

### Memory errors ("Killed")

**Fix:** Reduce parallel processing (24GB is the hard cap on WRDS):
```python
# In stage1/_stage1_settings.py
N_CORES = 1  # Use fewer cores on WRDS
```

---

## What's Next?

After the pipeline completes:

1. **Explore your data**: Load `stage1_YYYYMMDD.parquet` into pandas/R
2. **Read detailed docs**: See `README.md` for variable definitions
3. **Check data quality**: Review reports in `stage1/data/reports/`
4. **Customize filters**: Edit `stage1/_stage1_settings.py` for custom filters
5. **Run incrementally**: Re-run Stage 1 with different settings without re-running Stage 0

---

## File Structure Overview

```
trace-data-pipeline/
├── config.py                        # Shared configuration (WRDS_USERNAME, TRACE_MEMBERS)
├── run_pipeline.sh                  # Main pipeline orchestrator
├── download_inputs.sh               # Stage 1's external inputs (LOGIN NODE)
├── run_smoke_test.sh                # Whole-chain check in minutes
├── README.md                        # Detailed documentation
├── QUICKSTART.md                    # This file
│
├── FAQ.md                           # Common questions
│
├── tests/                           # Run before you commit
│   ├── smoke_assertions.py          # The 28 cross-stage invariants
│   ├── test_chunk_plan.py           # No WRDS needed
│   ├── test_chunk_scheduler.py      # No WRDS needed
│   ├── test_merge_keys.py           # No WRDS needed by default
│   ├── test_docs.py                 # No WRDS needed
│   └── probe_wrds_connections.py    # Measures your connection ceiling
│
├── stage0/                          # TRACE data extraction
│   ├── run_enhanced_trace.sh        # SGE job script
│   ├── run_standard_trace.sh
│   ├── run_144a_trace.sh
│   ├── run_build_data_reports.sh
│   ├── create_daily_enhanced_trace.py
│   ├── create_daily_standard_trace.py
│   ├── _chunk_runner.py             # Chunk planning + concurrent scheduler
│   ├── _wrds_pool.py                # One WRDS connection per worker
│   ├── _trace_settings.py           # Stage 0 configuration (incl. CONCURRENCY)
│   ├── logs/                        # Job logs
│   ├── enhanced/                    # Enhanced TRACE outputs
│   ├── standard/                    # Standard TRACE outputs
│   └── 144a/                        # 144A TRACE outputs
│
└── stage1/                          # Bond analytics and enrichment
    ├── run_stage1.sh                # SGE job script
    ├── _run_stage1.py               # Main entry point
    ├── stage1_pipeline.py           # Pipeline logic
    ├── _stage1_settings.py          # Stage 1 configuration
    ├── QUICKSTART_stage1.md         # Stage 1 specific guide
    ├── logs/                        # Job logs
    └── data/                        # Stage 1 outputs
        ├── stage1_YYYYMMDD.parquet  # Final dataset
        ├── liu_wu_yields.xlsx       # Treasury yields (auto-downloaded)
        ├── bond_firm_linker_2026/   # Equity linker (auto-downloaded)
        ├── Siccodes12.txt           # FF12 industries (auto-downloaded)
        ├── Siccodes17.txt           # FF17 industries (auto-downloaded)
        └── Siccodes30.txt           # FF30 industries (auto-downloaded)
```

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

# Install dependencies (Stage 1 only)
python -m pip install --user -r requirements.txt

# Fetch stage 1's external inputs (login node) and check the chain end to end
chmod +x *.sh stage0/*.sh stage1/*.sh
bash download_inputs.sh
qsub run_smoke_test.sh             # ~10 min -> smoke_test.out

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
cd /scratch/{institution}/ && zip -r trace-data-pipeline.zip ~/trace-data-pipeline/

# Step 2: Download zip (from LOCAL machine)
scp {wrds_username}@wrds-cloud.wharton.upenn.edu:/scratch/{institution}/trace-data-pipeline.zip ./
```

---

