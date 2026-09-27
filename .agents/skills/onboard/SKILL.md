---
name: onboard
description: Take a fresh clone of trace-data-pipeline to a machine that is ready to run it - the user's own computer for stages 2 to 4, or the WRDS login node for stages 0 and 1. Use when the Python environment, the packages, the WRDS username or the input files are in doubt, before a first run, or when a stage stops before doing any work.
---

# Onboard

Read first: `AGENTS.md` at the repository root. Then `QUICKSTART.md`, which this follows.

## Use when

- a fresh clone, or a machine that has never run the pipeline
- a stage stops at start-up (a missing package, PyBondLab, the WRDS username, an input file)
- the user asks "what do I do next?"

## Do not use when

- the machine is ready and the user wants a stage run: use `run-wrds`, `build-panel`,
  `reproduce-exhibits` or `build-factors`
- the question is what the data or code means: use `explain`

## First, find out where you are

Ask, or tell from the prompt and the shell: the WRDS login node (stages 0 and 1), or the
user's own computer (stages 2 to 4)? Then run the matching check from the repository root and
read every line it prints:

```bash
python doctor.py            # the user's own computer
python3 doctor.py --wrds    # the WRDS login node
```

`doctor.py` downloads nothing, opens no WRDS connection and writes nothing of its own, so it is always
safe to run. Its last line is the next step.

## On the user's own computer

1. **A virtual environment for this repository, always**, even when `doctor.py` reports a
   supported Python: the install replaces packages that other projects may depend on,
   PyBondLab above all. Make it with Python 3.11, 3.12 or 3.13: `py -3.13 -m venv .venv` on
   Windows, `python3.13 -m venv .venv` elsewhere. Ask the user before installing into any
   environment that already exists, the base Anaconda one included.
2. **Call that environment's Python by its path** from here on: `.venv/Scripts/python` on
   Windows, `.venv/bin/python` elsewhere (written `VENVPY` below). Your shell does not keep an
   activated environment from one command to the next, so a bare `python` would be the
   system's. The `.sh` runners take it as `PY=VENVPY bash run_stage3.sh`.
3. **Install into it, from the repository root, exactly these two lines, in this order:**
   ```bash
   VENVPY -m pip install -r requirements-local.txt
   VENVPY -m pip install --no-deps pybondlab==0.3.0
   ```
   Only when the user wants to reproduce the published 2026 files as closely as possible, add
   `-c constraints-2026.txt` to the first line. Never install PyBondLab any other way: stages 2
   to 4 check its version at start-up.
4. **The WRDS login**, for stage 2's first run only, which fetches and caches a few series.
   The username: `export WRDS_USERNAME="your_wrds_id"` (PowerShell:
   `$env:WRDS_USERNAME="your_wrds_id"`), or edit `config.py`. The password: have the user run
   `VENVPY -c "import wrds; wrds.Connection()"` once, themselves, in a terminal. It asks for
   the username and then the password, and offers to save them in the file the `wrds` package reads (`~/.pgpass`, or
   `%APPDATA%\postgresql\pgpass.conf` on Windows), because a build running in the background
   cannot answer a password prompt. Ask the user for the username, never
   guess one. The password the user writes into that file: never ask for it in the chat, and
   never write it into the repository.
5. **The data.** Stage 2 reads the `stage0/` and `stage1/` folders a WRDS run produced. If they
   are missing, point the user to "Download Results to Your Local Machine" in `QUICKSTART.md`.
   The daily file published on openbondassetpricing.com cannot stand in for them: stage 2
   refuses it, because its ratings are removed.
6. Run `VENVPY doctor.py` again, then the tests, and report both:
   ```bash
   VENVPY -m pytest stage2/tests tests stage3/tests stage4/tests -q
   ```

## On the WRDS login node

1. Install per "Step 2: Install Dependencies" in `QUICKSTART.md` (`requirements.txt`; on the
   Cloud's Python, a venv needs `--system-site-packages`).
2. Set the username: `export WRDS_USERNAME="your_wrds_id"`, and add the same line to `~/.bashrc`.
3. Fetch stage 1's inputs, here, because compute nodes have no internet:
   `bash download_inputs.sh`. `bash download_inputs.sh --check` says what is already there.
4. Run `python3 doctor.py --wrds` again. When it is clean, hand over to `run-wrds`.

## Stop and report when

- a check fails for a reason these steps do not cover: show its message, do not work around it
- on Windows, the install fails with "No such file or directory" under `site-packages` and a
  hint about long paths: the clone sits too deep. Suggest moving it to a short folder, or
  enabling Windows long paths, which needs an administrator
- the user has no WRDS account, or no TRACE, FISD or ratings access: stages 0 and 1 cannot run

## Never

- open a WRDS connection from the login node to "test" it: every connection belongs in a job
- edit code to get past a start-up check
- install packages outside the two lines above on the user's computer
- install into the base Anaconda environment, or any other shared one, without the user's yes
