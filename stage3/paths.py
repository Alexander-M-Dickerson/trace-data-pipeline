"""paths.py -- every path Stage 3 uses, derived from `_stage3_settings.py`.

Import this; never hard-code a path anywhere else. `tools/check_inputs.py` audits what is
present before a run starts, and `tests/test_no_absolute_paths.py` fails on any absolute
user path that creeps into the tree -- a home directory baked into a script is the defect
that makes a replication package unrunnable on the next machine.

Run it to see what resolved:

    python paths.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # stage3/
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _stage3_settings as S    # noqa: E402

STAGE3 = HERE
PIPELINE = S.PIPELINE

# --- inputs (Stage 1 and Stage 2) -------------------------------------------
PANEL = S.STAGE2_PANEL          # the monthly bond panel
MMN = S.STAGE2_MMN              # the unadjusted *_mmn signal twins
BBW = S.STAGE2_BBW              # MKTB and the other bond factors
FACTORS = S.STAGE2_FACTORS      # rf and the macro/uncertainty set
DAILY = S.STAGE1_DAILY          # the daily bond-day panel (data appendix only)

# --- outputs ----------------------------------------------------------------
DATA = S.DATA                   # computed artifacts: sorts, stats, factor series, grids
REPORTS = S.REPORTS             # rendered exhibits

SORTS = DATA / "sorts"          # long-format long-short panels (the file grammar)
STATS = DATA / "stats"          # one tidy statistics frame per section
FACTOR_SERIES = DATA / "factors"    # wide factor return series, for distribution
GRIDS = DATA / "grids"          # the uncertainty grids (large, one parquet per signal)
CACHE = DATA / "_cache"         # signature-keyed intermediates

TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"
TIMINGS = REPORTS / "timings.jsonl"

SECTIONS = ("s0_data", "s1_lib", "s2_lab", "s3_nse", "s4_zoo")

for _d in (DATA, REPORTS, SORTS, STATS, FACTOR_SERIES, GRIDS, CACHE, TABLES, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)


def section_results(section: str) -> Path:
    """data/<section>/, created on demand. Raises on an unknown section name."""
    if section not in SECTIONS:
        raise ValueError(f"unknown section {section!r}; known: {SECTIONS}")
    d = DATA / section
    d.mkdir(parents=True, exist_ok=True)
    return d


def cube(kind: str, sig: str) -> Path:
    """Signature-keyed cache path: cube('lib_stats', sig) -> data/_cache/lib_stats_<sig>.parquet."""
    return CACHE / f"{kind}_{sig}.parquet"


_AUDIT = [("PANEL", PANEL), ("MMN", MMN), ("BBW", BBW), ("FACTORS", FACTORS),
          ("DAILY", DAILY)]

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    missing = 0
    for name, p in _AUDIT:
        ok = p is not None and Path(p).exists()
        missing += (not ok)
        print(f"{'OK  ' if ok else 'MISS'} {name:9s} {p if p else '(unresolved)'}")
    print(f"\n{len(_AUDIT) - missing}/{len(_AUDIT)} inputs present")
    print(f"outputs -> {DATA}  and  {REPORTS}")
    raise SystemExit(1 if missing else 0)
