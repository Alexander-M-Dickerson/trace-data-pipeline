```markdown
# Quick Start (Stage 0) — WRDS Cloud

## Prerequisites
- SSH access to WRDS Cloud
- WRDS account with TRACE access
- Python ≥ 3.10
- `.pgpass` for passwordless DB auth:
````


---

## 1) Clone on WRDS

```bash
ssh <your_wrds_id>@wrds-cloud.wharton.upenn.edu
cd ~
git clone https://github.com/Alexander-M-Dickerson/trace-data-pipeline.git
cd trace-data-pipeline/stage0
```

---

## 2. Set up environment and install dependencies

### Option A — `venv` (default)

```bash
python3 -m venv ~/wrds_env
source ~/wrds_env/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt   # do NOT add --user
```

### Option B — conda (if installed)

```bash
conda create -n wrds_env python=3.13 -y
conda activate wrds_env
python -m pip install -r requirements.txt   # do NOT add --user
```

> Use `--user` only if you are **not** in any virtual/conda environment (system Python):
>
> ```bash
> python -m pip install --user -r requirements.txt
> ```


---

## 3) Configure WRDS username

The code reads `WRDS_USERNAME` from the environment.

**Option A — set an environment variable:**

```bash
export WRDS_USERNAME="<your_wrds_id>"
```

Make it persistent for future logins:

```bash
echo 'export WRDS_USERNAME="<your_wrds_id>"' >> ~/.bashrc
source ~/.bashrc
```

**Option B — default fallback in `_trace_settings.py`:**

```python
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "<your_wrds_id>")
```

You do **not** need to put the password in code; `.pgpass` supplies it.

---

## 4) Run the pipeline

You have two equivalent ways to start the master script:

**Method 1 — run via bash (no permission change needed):**

```bash
bash run_all_trace.sh
```

**Method 2 — make it executable once, then run directly:**

```bash
chmod +x run_all_trace.sh      # one-time setup
./run_all_trace.sh
```

What this does:

* Submits three SGE jobs: `Enhanced`, `Standard`, `144A`
* Submits a report job with `-hold_jid` that waits for the above to finish
* Jobs run on the WRDS cluster via `qsub`, so disconnecting SSH is safe

Outputs:

```
enhanced/
standard/
144a/
data_reports/
logs/
```

---

## 5) Monitor jobs and view logs

List your jobs:

```bash
qstat
```

State codes:

* `r`   = running
* `qw`  = queued, waiting
* `hqw` = on hold (waiting for dependencies)
* `Eqw` = error

Job details:

```bash
qstat -j <jobID>
```

Live logs:

```bash
tail -f logs/01_enhanced.out
tail -f logs/01_enhanced.err
```

Stop tail: `Ctrl + C`

Cancel a job:

```bash
qdel <jobID>
```

---

## 6) FAQ / common pitfalls

**Q: I used `pip install --user` inside venv/conda. Is that a problem?**
A: Yes. That installs to `~/.local/...` and bypasses your env. Reinstall **without** `--user` after activating venv/conda.

**Q: Do jobs stop if I log out or lose SSH?**
A: No. Anything submitted via `qsub` keeps running on the cluster.

**Q: The script asks for a password.**
A: Fix `.pgpass` and its permissions: `chmod 600 ~/.pgpass`.




