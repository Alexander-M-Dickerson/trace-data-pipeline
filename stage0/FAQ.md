# Frequently Asked Questions (FAQ)

## Table of Contents
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [Contributing](#contributing)
- [Academic Use](#academic-use)

---

## Getting Started

### Do I need a WRDS subscription?
Yes, Stage 0 requires WRDS access with TRACE Enhanced, Standard, or 144A entitlements. You can check your entitlements by logging into WRDS and viewing your subscriptions.

### Can I run this on my local machine?
Stage 0 is specifically designed for the WRDS Cloud environment due to database access requirements. Stages 1 and 2 (coming November 2025) will run on your local machine with WRDS connection.

### How long does processing take?
Using `./run_all_trace.sh` (complete automated pipeline):
- **Enhanced TRACE**: ~4 hours
- **Standard TRACE**: ~30-60 minutes
- **Rule 144A**: ~30-60 minutes
- **Report generation**: ~30-60 minutes
- **Total**: ~5 hours for everything

### What if I only want Enhanced TRACE?
You can run individual datasets:
```bash
./run_enhanced_trace.sh    # Enhanced only
./run_standard_trace.sh    # Standard only
./run_144a_trace.sh        # 144A only
```

### What Python version do I need?
Python 3.10 or higher is required. Check your version:
```bash
python --version
```

### Do I need to install anything besides Python packages?
For Stage 0, you need:
- WRDS subscription with TRACE access
- SSH access to WRDS Cloud
- `.pgpass` file configured for password-less authentication
- Required Python packages (installed via `requirements.txt`)

---

## Configuration

### How do I change the date range?
Edit `_trace_settings.py` and modify the `PER_DATASET` dictionary:

```python
PER_DATASET = {
    "enhanced": dict(),  # Uses default full sample
    "standard": dict(start_date="2023-01-01", data_type="standard"),
    "144a": dict(start_date="2023-01-01", data_type="144a"),
}
```

**Note**: Enhanced TRACE does not currently support custom start dates and processes the full sample (2002-07-01 to present).

### How do I adjust memory usage?
If you encounter memory errors, reduce the `chunk_size` in `_trace_settings.py`:

```python
COMMON_KWARGS = {
  chunk_size    = 150,
}
```

Reducing chunk size means you pull *fewer* bonds each iteration. Reducing RAM requirements.

### Can I disable certain filters?
Yes! Edit the `FILTER_SWITCHES`, `FISD_PARAMS`, `COMMON_KWARGS`, `DS_PARAMS` or `BB_PARAMS` dictionaries in `_trace_settings.py`. 

### How do I change which bonds are included?
Modify the `FISD_PARAMS` dictionary in `_trace_settings.py`:

```python
FISD_PARAMS = {
   "currency_usd_only": True,                   # foreign_currency == 'N'
   "fixed_rate_only"  : True,                   # coupon_type != 'V'
    # ... etc
}
```

### Can I change the output format from Parquet to CSV?
The code uses Parquet by default for efficiency. To change to CSV, you'll need to modify `output_format` in `COMMON_KWARGS` in `_trace_settings.py`.

---

## Output Files

### What format are the output files?
Output files are in **Parquet format** by default (compressed, efficient). Parquet is a columnar storage format that:
- Compresses well (~5-10x smaller than CSV)
- Loads faster
- Preserves data types
- Works with pandas, R, and many other tools

### How do I read the output files?
Using Python/pandas:
```python
import pandas as pd

# Read a single dataset
df = pd.read_parquet('enhanced/enhanced_20250120.parquet')

# Read multiple files
import glob
files = glob.glob('enhanced/*.parquet')
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
```

Using R:
```r
library(arrow)
df <- read_parquet('enhanced/enhanced_20250120.parquet')
```

### What columns are in the output files?
All datasets (Enhanced, Standard, 144A) have the same column structure:

**Identifiers**:
- `cusip_id`: 9-character CUSIP identifier
- `trd_exctn_dt`: Trade execution date

**Price metrics**:
- `prc_ew`: Equal-weighted price
- `prc_vw`: Volume-weighted price (dollar)
- `prc_vw_par`: Volume-weighted price (par)
- `prc_first`: First trade price of day
- `prc_last`: Last trade price of day
- `prc_lo`: Low price of the day
- `prc_hi`: High price of the day

**Volume metrics** (in millions):
- `qvolume`: Par volume
- `dvolume`: Dollar volume

**Bid/Ask metrics**:
- `prc_bid`: Customer-side bid (value-weighted)
- `prc_ask`: Customer-side ask (value-weighted)

**Count metrics**:
- `trade_count`: Number of trades
- `bid_count`: Number of customer buys
- `ask_count`: Number of customer sells

### What do the audit files contain?
Audit files track row counts at each filter stage:
- `dick_nielsen_filters_audit_*.parquet`: Dick-Nielsen filter effects
- `drr_filters_audit_*.parquet`: Decimal-shift and bounce-back filter effects
- `fisd_filters_*.parquet`: FISD universe construction audit

These help you understand how many transactions were removed at each cleaning step.

### What are the CUSIP list files?
These files identify bonds that were corrected:
- `decimal_shift_cusips_*.parquet`: CUSIPs with decimal-shift corrections
- `bounce_back_cusips_*.parquet`: CUSIPs with bounce-back flags

Useful for understanding which bonds had price errors.

### Where are the reports saved?
Reports are saved in `data_reports/[enhanced|standard|144a]/` with:
- LaTeX source file (`.tex`)
- Bibliography file (`references.bib`)
- Figures (`.pdf` files)

You can compile the LaTeX to PDF or view the figures directly.

### How do I download files from WRDS Cloud?

**Windows (using WinSCP)**:
1. Install WinSCP
2. Connect to `wrds-cloud.wharton.upenn.edu`
3. Navigate to your project directory
4. Download the folders

**Mac/Linux (using scp)**:
```bash
scp -r username@wrds-cloud.wharton.upenn.edu:~/proj/stage0/enhanced ./local_destination/
scp -r username@wrds-cloud.wharton.upenn.edu:~/proj/stage0/data_reports ./local_destination/
```

---

## Troubleshooting

### My job keeps failing with memory errors
**Solutions**:
1. Reduce `chunk_size` in `_trace_settings.py` (try 100 or 150)
2. Request more memory in shell scripts by adding:
   ```bash
   #$ -l m_mem_free=16G
   ```

### I'm getting "Permission denied" errors
**Solution**: Make scripts executable:
```bash
chmod +x run_all_trace.sh run_enhanced_trace.sh run_standard_trace.sh run_144a_trace.sh
```

### I'm getting "bad interpreter" or `^M` errors
**Solution**: Convert Windows line endings to Unix:
```bash
sed -i 's/\r$//' run_all_trace.sh
# Or fix all shell scripts:
find . -name "*.sh" -exec sed -i 's/\r$//' {} \;
```

### My job is stuck in queue (status 'qw')
**Possible causes**:
- WRDS resource constraints during peak hours
- Queue is full
- Requested resources not available

**Check status**:
```bash
qstat          # View all your jobs
qstat -j <job_id>  # View specific job details
```

### No data is being returned
**Check**:
1. Verify date ranges in `_trace_settings.py`
2. Confirm WRDS entitlements (Enhanced/Standard/144A)
3. Review logs for SQL errors: `cat logs/01_enhanced.err`
4. Ensure FISD filters aren't too restrictive

### Empty output files
**Check**:
1. Date ranges are correct
2. FISD filters aren't excluding all bonds
3. Review audit files to see where rows were dropped

### ImportError: No module named 'wrds' (or other package)
**Solution**: Install missing packages:
```bash
pip install --user wrds pandas numpy pandas-market-calendars
# Or install all requirements:
pip install --user -r requirements.txt
```

### The report generation job (build_reports) never starts
**Explanation**: This is normal! When using `./run_all_trace.sh`, the report job shows status `hqw` (holding) until all three data jobs complete. It will automatically start when ready.

**Check progress**:
```bash
qstat  # Look for 'hqw' status - this means it's waiting for dependencies
```

### LaTeX compilation errors in reports
**Possible issues**:
- Missing figure files
- Missing `references.bib`
- Incomplete LaTeX installation

**Solution**: Ensure all figure files generated before compiling, or set `output_figures = False` in `_build_error_files.py`.

---

## Performance

### How can I make processing faster?
**Options**:
1. **Increase chunk size** (if you have enough memory):
   ```python
    COMMON_KWARGS = {
    chunk_size    = 400,
    ...
                    }
```
2. **Disable figure generation** (saves 30-60 minutes):
   Set `output_figures = False` in `_build_error_files.py`

### Can I run multiple datasets simultaneously?
Yes! That's exactly what `./run_all_trace.sh` does. It submits Enhanced, Standard, and 144A as parallel jobs.

---

## Contributing

### How can I contribute?
See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines. Key ways to contribute:
1. **Report bugs** using GitHub issues
2. **Suggest features** via feature requests
3. **Submit pull requests** for bug fixes or enhancements
4. **Improve documentation**
5. **Test on different WRDS environments**

### I found a bug. What should I do?
Create a GitHub issue with:
- Clear bug description
- Steps to reproduce
- Your environment (Python version, WRDS setup)
- Relevant log files
- Expected vs actual behavior

Use the bug report template when creating the issue.

### I have an idea for a new feature
Great! Create a feature request issue explaining:
- What problem it solves
- How it would work
- Who would benefit
- Whether you can help implement it

### How do I test my changes?
1. Test locally if possible
2. Test on WRDS Cloud with sample data
3. Check log files for errors
4. Verify output format matches expectations
5. Run on small date range before full dataset

---

## Academic Use

### How do I cite this pipeline?
See the Citation section in [README.md](README.md). Use both references:

**Primary**:
```
Dickerson, A., Robotti, C., & Rossetti, G. (2025). 
Common pitfalls in the evaluation of corporate bond strategies.
Working Paper.
```

**Secondary**:
```
Dickerson, A., & Rossetti, G. (2025). 
Constructing TRACE Corporate Bond Datasets.
Working Paper.
```

### Can I use this for my research?
Absolutely! This project is MIT licensed and designed for academic research. Please cite appropriately.

### Will I be acknowledged for contributions?
Yes! Contributors are acknowledged in:
- The project README
- Release notes for significant contributions
- The academic paper underlying this work (for major contributors)

### What's the connection to PyBondLab?
This pipeline produces clean TRACE data. [PyBondLab](https://github.com/GiulioRossetti94/PyBondLab) is the companion repository for constructing corporate bond asset pricing factors from this data.

Workflow:
1. **trace-data-pipeline** (this repo): Clean raw TRACE → daily panels
2. **PyBondLab**: Daily panels → bond characteristics → factor portfolios

### Is this part of a larger project?
Yes! This is part of the [Open Bond Asset Pricing](https://openbondassetpricing.com/) project, which aims to provide open-source tools for corporate bond research.

---

## Support

### Where can I get help?
- **GitHub Issues**: [trace-data-pipeline/issues](https://github.com/Alexander-M-Dickerson/trace-data-pipeline/issues)
- **Email**: alexander.dickerson1@unsw.edu.au
- **Documentation**: 
  - [Main README](README.md)
  - [Stage 0 README](stage0/README_stage0.md)
  - [Contributing Guide](CONTRIBUTING.md)

### How quickly will I get a response?
We aim to respond to issues within 1-3 business days. For urgent matters, email directly.

### Can I schedule a call to discuss my use case?
Yes! For collaboration or complex use cases, email alexander.dickerson1@unsw.edu.au to arrange a discussion.

---

## Stages 1 & 2

### When will Stages 1 and 2 be released?
Both stages are scheduled for **November 2025**.

### What will Stage 1 include?
Stage 1 calculates daily bond metrics:
- Accrued interest
- Credit spreads
- Duration and convexity
- Yield-to-maturity
- Additional bond characteristics

### What will Stage 2 include?
Stage 2 produces monthly panels with:
- 50+ bond characteristic signals
- Credit risk factors
- Liquidity measures
- Momentum and reversal signals
- Portfolio-ready outputs

### Can I beta test Stages 1 and 2?
Yes! Email alexander.dickerson1@unsw.edu.au to express interest in beta testing.

---

**Last updated**: January 2025
