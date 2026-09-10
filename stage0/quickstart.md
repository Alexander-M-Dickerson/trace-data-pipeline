
# Quick Start (Stage 0) — WRDS Cloud

## Prerequisites
- SSH access to WRDS Cloud
- WRDS account with TRACE access
- Python ≥ 3.10
- `.pgpass` for passwordless DB auth:

---

## 1) Clone on WRDS

```bash
ssh <your_wrds_id>@wrds-cloud.wharton.upenn.edu
cd ~
git clone https://github.com/Alexander-M-Dickerson/trace-data-pipeline.git
cd trace-data-pipeline
```

❗Stay at the repo ROOT. `run_pipeline.sh` lives there, and it aborts unless `stage0/`
and `stage1/` are both directly beneath the working directory.

---

## 2. Set up environment and install dependencies

### Option A — `venv` (default)

```bash
python3 -m venv ~/wrds_env
source ~/wrds_env/bin/activate
python -m pip install -U pip
python -m pip install -r ../requirements.txt   # do NOT add --user
```

### Option B — conda (if installed)

```bash
conda create -n wrds_env python=3.13 -y
conda activate wrds_env
python -m pip install -r ../requirements.txt   # do NOT add --user
```

> Use `--user` only if you are **not** in any virtual/conda environment (system Python):
>
> ```bash
> python -m pip install --user -r ../requirements.txt
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

**Option B — the fallback in `config.py`** (the repo ROOT, not `_trace_settings.py`,
which merely imports it):

```python
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_username")
```

❗`run_smoke_test.sh` reads the ENVIRONMENT, not `config.py`, and refuses to start
without it -- so if you take Option B, still `export WRDS_USERNAME` before using it.

You do **not** need to put the password in code; `.pgpass` supplies it.

---

## 4) Run the pipeline

From the repo ROOT (not `stage0/`):

```bash
./run_pipeline.sh
```

The scripts ship executable, so no `chmod` is needed on a fresh clone. If an older
clone gives "Permission denied", either `bash run_pipeline.sh` or
`chmod +x *.sh stage0/*.sh stage1/*.sh` once.

`run_pipeline.sh` fetches Stage 1's external inputs itself (on the login node, which is
the only place with internet), then submits Stage 0 for each member in `TRACE_MEMBERS`,
the data-report job, and Stage 1.

What this does:

* Submits one SGE job per member in `TRACE_MEMBERS` -- by default `enhanced` and
  `144a`, which go in together. `standard` is opt-in and is held behind them so it
  gets the whole WRDS connection budget.
* Submits the data-report job with `-hold_jid` on those stage-0 jobs.
* Submits Stage 1 with `-hold_jid` on the same stage-0 jobs, so it runs **alongside**
  the reports rather than after them (v2.2.2).
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




