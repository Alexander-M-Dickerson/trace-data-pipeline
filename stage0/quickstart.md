
# Quick Start (Stage 0) — WRDS Cloud

## Prerequisites
- SSH access to WRDS Cloud
- WRDS account with TRACE access
- Python ≥ 3.10
- No password file: jobs on the WRDS Cloud connect without one

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

### Option A — `venv`

```bash
# --system-site-packages keeps the WRDS Cloud's own pandas 2.2 visible: pip has no build of it
# for the Cloud's Python 3.14.
python3 -m venv --system-site-packages ~/wrds_env
source ~/wrds_env/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt      # do NOT add --user
```

❗In every later session, `source ~/wrds_env/bin/activate` again before `./run_pipeline.sh` or
`qsub`. The jobs take your shell's environment when you submit them (`#$ -V`), so they run
whichever Python is active then.

### Option B — conda (if installed)

```bash
conda create -n wrds_env python=3.13 -y
conda activate wrds_env
python -m pip install -r requirements.txt      # do NOT add --user
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

**Option B — the fallback in `config.py`** (the repo ROOT, not `_trace_settings.py`,
which merely imports it):

```python
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_username")
```

`run_smoke_test.sh` reads the username the way the stages do, from the environment or
`config.py`, so either option works for it.

Your password goes nowhere: jobs on the WRDS Cloud connect without it.

---

## 4) Run the pipeline

From the repo ROOT (not `stage0/`):

```bash
./run_pipeline.sh
```

The scripts ship executable, so no `chmod` is needed on a fresh clone. If an older
clone gives "Permission denied", either `bash run_pipeline.sh` or
`chmod +x *.sh stage0/*.sh stage1/*.sh` once.

`run_pipeline.sh` first checks your home quota (`check_disk_space.sh`: it stops if less than
6 GB is free, since a full run writes about 5 GB; `FORCE_RUN=1 ./run_pipeline.sh` overrides). It then fetches Stage 1's external inputs (on the login node,
which is the only place with internet), and submits Stage 0 for each member in
`TRACE_MEMBERS`, the data-report job, and Stage 1.

What this does:

* Submits one SGE job per member in `TRACE_MEMBERS` -- by default `enhanced` and
  `144a`, which go in together. `standard` is opt-in and is held behind them so it
  gets the whole WRDS connection budget.
* Submits the data-report job with `-hold_jid` on those stage-0 jobs.
* Submits Stage 1 with `-hold_jid` on the same stage-0 jobs, so it runs **alongside**
  the reports rather than after them (v2.2.2).
* Jobs run on the WRDS cluster via `qsub`, so disconnecting SSH is safe

Outputs, under `stage0/`:

```
stage0/enhanced/
stage0/144a/
stage0/standard/        # only if you opt Standard in
stage0/data_reports/
stage0/logs/
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
tail -f stage0/logs/01_enhanced.out
tail -f stage0/logs/01_enhanced.err
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
A: On the WRDS Cloud that means the username is wrong: check `WRDS_USERNAME`
(`python3 doctor.py --wrds`). No password file is needed there.




