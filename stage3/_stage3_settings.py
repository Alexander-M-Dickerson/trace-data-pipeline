"""_stage3_settings.py -- every path and constant Stage 3 uses.

Mirrors `stage0/_trace_settings.py` and `stage1/_stage1_settings.py`. Nothing here may be
an absolute path baked into a script: a home directory belonging to one machine is exactly
what makes a replication package unrunnable elsewhere. Every input is resolved from the
Stage 1 / Stage 2 output tree that sits beside this folder, and every one of them can be
overridden from the environment.

    python _run_stage3.py --dry-run        # print what resolved, compute nothing

The five inputs, and what each is for:

    STAGE2_PANEL     the monthly bond panel                  every section
    STAGE2_MMN       the unadjusted `*_mmn` signal twins     the three approaches (S1, S2)
    STAGE2_BBW       bbw_factors.parquet -- MKTB             every CAPM_B alpha
    STAGE2_FACTORS   factors.parquet -- rf and the macro set excess returns
    STAGE1_DAILY     the daily bond-day panel                the data appendix only

`STAGE2_BBW` and `STAGE2_FACTORS` are two different files and are not interchangeable:
MKTB lives in the first, the risk-free rate in the second.

A duration-adjusted run (STAGE3_RETURNS, below) also reads its return type's two stage 2
blocks, `STAGE2_<BLOCK>`: the beta and momentum signals estimated on that return.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent                       # the trace-data-pipeline root
# The root's shared modules (numeric_setup, pybondlab_pin). Appended, so nothing there can
# shadow a Stage 3 module.
if str(PIPELINE) not in sys.path:
    sys.path.append(str(PIPELINE))
import numeric_setup  # noqa: E402,F401  (pandas computes the same whatever else is installed)
import return_types as _RT  # noqa: E402  (the one definition of a return type, shared with stage 4)
STAGE0_DIR = Path(os.environ.get("STAGE0_DIR", PIPELINE / "stage0"))
STAGE1_DIR = Path(os.environ.get("STAGE1_DIR", PIPELINE / "stage1"))
STAGE2_DIR = Path(os.environ.get("STAGE2_DIR", PIPELINE / "stage2"))

# Stage 2 writes its artifacts under an input-mode name; the public pipeline has exactly
# one input, so the mode is "stage1" unless you built under another name.
MODE = os.environ.get("STAGE3_MODE", "stage1")

# --- The return every section sorts --------------------------------------------
# [ref:rule.return_types] `exc` (the default) is the return the paper uses. `dur`, `dbns` and
# `dcls` are duration-adjusted against tret, tret_bns and tret_cls, with the return-based
# signals replaced by stage 2's blocks estimated on that same return (returns.py). Set it with
# STAGE3_RETURNS or `run_stage3.sh --returns <type>`.
#
# ❗A duration-adjusted run writes its OWN tree, variants/<type>/, and refuses the standard
# one, so it can never overwrite the exhibits of the standard run.
RETURNS = _RT.check(os.environ.get("STAGE3_RETURNS", _RT.DEFAULT))
_STANDARD_DATA, _STANDARD_REPORTS = HERE / "data", HERE / "reports"
if RETURNS == _RT.DEFAULT:
    _DATA_DEFAULT, _REPORTS_DEFAULT = _STANDARD_DATA, _STANDARD_REPORTS
else:
    _DATA_DEFAULT = HERE / "variants" / RETURNS / "data"
    _REPORTS_DEFAULT = HERE / "variants" / RETURNS / "reports"

DATA = Path(os.environ.get("STAGE3_DATA", _DATA_DEFAULT))
REPORTS = Path(os.environ.get("STAGE3_REPORTS", _REPORTS_DEFAULT))
if RETURNS != _RT.DEFAULT and ({DATA.resolve(), REPORTS.resolve()}
                               & {_STANDARD_DATA.resolve(), _STANDARD_REPORTS.resolve()}):
    raise SystemExit(f"STAGE3_RETURNS={RETURNS} would write into the standard run's "
                     f"{_STANDARD_DATA.name}/ or {_STANDARD_REPORTS.name}/. Unset STAGE3_DATA "
                     f"and STAGE3_REPORTS (the run then writes to variants/{RETURNS}/), or point "
                     "them somewhere else.")


def _latest(d: Path, prefix: str) -> Path | None:
    """The newest `prefix<stamp>.parquet` in `d` -- how Stage 2 finds its Stage 1 input."""
    if not d.is_dir():
        return None
    hits = sorted(d.glob(f"{prefix}*.parquet"))
    return hits[-1] if hits else None


def _resolve(env: str, default: Path | None) -> Path | None:
    v = os.environ.get(env)
    return Path(v) if v else default


_PANEL_DIR = STAGE2_DIR / "output" / "panel"
_BLOCKS = STAGE2_DIR / "output" / "blocks" / MODE

# --- Stage 2 inputs ---------------------------------------------------------
STAGE2_PANEL = _resolve("STAGE2_PANEL", _PANEL_DIR / f"main_panel_{MODE}.parquet")
STAGE2_MMN = _resolve("STAGE2_MMN", _latest(_BLOCKS, "mmn_price_based_signals_"))
STAGE2_BBW = _resolve("STAGE2_BBW", _BLOCKS / "bbw_factors.parquet")
STAGE2_FACTORS = _resolve("STAGE2_FACTORS", _BLOCKS / "factors.parquet")

# --- Stage 1 input (the data appendix only) ---------------------------------
STAGE1_DAILY = _resolve("STAGE1_DAILY", _latest(STAGE1_DIR / "data", "stage1_"))

INPUTS = {
    "STAGE2_PANEL": STAGE2_PANEL,
    "STAGE2_MMN": STAGE2_MMN,
    "STAGE2_BBW": STAGE2_BBW,
    "STAGE2_FACTORS": STAGE2_FACTORS,
    "STAGE1_DAILY": STAGE1_DAILY,
}

# --- The return types' blocks (a duration-adjusted run only) --------------------
# Every block any return type reads is declared, so spec/inputs.json and this map list the
# same inputs; only the running type's two are required.
_BLOCK_FILES = {b: _resolve(f"STAGE2_{b.upper()}", _BLOCKS / f"{b}.parquet")
                for b in _RT.all_blocks()}
_BLOCK_KEYS = {f"STAGE2_{b.upper()}": b for b in _BLOCK_FILES}
INPUTS.update({k: _BLOCK_FILES[b] for k, b in _BLOCK_KEYS.items()})
RETURN_INPUTS = [f"STAGE2_{b.upper()}" for b in _RT.blocks(RETURNS)]


def block_path(name: str) -> Path:
    """A return type's stage 2 block file, by its name (`betas_bns`)."""
    return _BLOCK_FILES[name]


def needed(key: str) -> bool:
    """Is this input read in this run? A block only by its own return type."""
    return key not in _BLOCK_KEYS or key in RETURN_INPUTS

# --- PyBondLab --------------------------------------------------------------
# Stage 3 runs through the PyBondLab release `pybondlab_pin.py` (at the repository root)
# names, installed from requirements-local.txt. `pblenv.py` checks it before any sort.

# --- Sample conventions ------------------------------------------------------
# Both windows give floor(T**0.25) = 4. Assert the length before trusting a t-statistic:
# every standard error in the paper is derived from it.
SAMPLE = {
    "lib": {"start": "2002-09-30", "end": "2024-12-31"},   # T = 268
    "lab": {"start": "2002-08-31", "end": "2024-12-31"},   # T = 269
}
# The panel starts before the sample does: sorts are formed from 2002-07-31 so the first
# printed return, 2002-09-30, has a formed portfolio behind it.
FORMATION_START = "2002-07-31"

# Producers save FULL-LENGTH series; the window is applied at the statistics layer by
# truncating the series and recomputing. A stored statistic is never truncated.
WINDOW_POLICY = "produce-untruncated-truncate-at-stats"

# --- Portfolio construction ---------------------------------------------------
N_PORTFOLIOS = {"single_all": 10, "single_ig": 5, "single_nig": 5, "within_firm": 2}
HOLDING_PERIOD = 1
REBALANCE = "monthly"
MIN_BONDS_PER_FIRM = 2          # within-firm sorts

# --- Inference ----------------------------------------------------------------
NW_LAG_RULE = "floor(T**0.25)"
ALPHA_FACTORS = ("MKTB",)       # CAPM_B

# --- Column mapping into PyBondLab --------------------------------------------
COLUMNS = {"ID": "cusip", "ret": "ret_vw", "VW": "mcap_e", "RATING_NUM": "spc_rat"}
FIRM_ID_COL = "permno"

# --- Parallelism ---------------------------------------------------------------
# The uncertainty grids fan out over fresh processes. On Windows the panel is re-pickled
# into every worker, so returns flatten well before the core count does; the grids read
# their own column slice per worker instead of inheriting the panel.
#
# None means "decide from this machine": `fastrun.sized()` resolves it against
# os.cpu_count() at the point of use, keeping workers x threads inside the core count.
# Pin it with STAGE3_WORKERS, or per-run with each grid's --workers.
N_WORKERS = int(os.environ.get("STAGE3_WORKERS", "0")) or None

# DuckDB's ceiling for the daily-panel scans in the data appendix. None means "derive it
# from free RAM at connection time" -- see s0_data/data_engine._con.
MEMORY_LIMIT = os.environ.get("STAGE3_MEMORY_LIMIT") or None


def missing_inputs() -> list[str]:
    """Which inputs this run reads are absent. `tools/check_inputs.py` reports them properly."""
    return [k for k, v in INPUTS.items()
            if needed(k) and (v is None or not Path(v).exists())]
