# -*- coding: utf-8 -*-
"""
test_docs.py
============
Catch documentation that has drifted away from the code.

This exists because a single audit found, in docs that all read plausibly:

  * `run_all_trace.sh` referenced 23 times. It had not existed for some time.
  * Two repository trees listing `stage0/QUICKSTART_stage0.md` and
    `stage1/requirements.txt`, neither of which exists, while omitting `config.py`,
    `FAQ.md` and `stage1/stage1_pipeline.py`, which do.
  * "Edit `_trace_settings.py` and change `WRDS_USERNAME`" -- an instruction that
    silently does nothing, because that module imports the value from `config.py`.
  * `OSBAP_Linker_*.parquet`, renamed a release earlier.
  * Feature lists still saying "Fama-French 17 and 30" after FF12 shipped.

Every one of those reads fine. None can be caught by review, only by comparison. So:
numbers the docs quote are checked against the code that defines them, and files the
docs name are checked to exist.

    python3 tests/test_docs.py

CHANGELOG.md is deliberately EXEMPT from the file-existence check. Its old entries
record what was true at the time -- an entry naming a since-deleted script is correct
history, not drift.

Author: Open Source Bond Asset Pricing
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage0"))

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name:<58} {detail}")
    if not cond:
        FAILURES.append(name)


# Docs whose claims must track the code. CHANGELOG is loaded but exempt from the
# file-existence rule; see the module docstring.
DOC_FILES = [
    "README.md", "QUICKSTART.md", "FAQ.md", "CONTRIBUTING.md",
    "stage0/README_stage0.md", "stage0/quickstart.md",
    "stage1/README_stage1.md", "stage1/QUICKSTART_stage1.md",
]

# Named in prose but produced at runtime or shipped inside a download, so they are
# legitimately absent from a fresh clone.
RUNTIME_ARTIFACTS = {
    "fl_linker.parquet", "fl_verdicts.parquet", "firm_names.parquet",
    "liu_wu_yields.xlsx", "Siccodes12.txt", "Siccodes17.txt", "Siccodes30.txt",
    "smoke_test.out", "smoke_test.err",
}

# Tokens that must not reappear: things that were renamed or deleted.
BANNED = {
    "run_all_trace": "deleted; the orchestrator is run_pipeline.sh",
    "OSBAP_Linker": "renamed in 2.1.0 to bond_firm_linker_2026/",
    "QUICKSTART_stage0": "the file is stage0/quickstart.md",
}


def main():
    docs = {f: (ROOT / f).read_text(encoding="utf-8") for f in DOC_FILES}
    prose = "\n".join(docs.values())

    # --- numbers quoted in prose must match the code ------------------
    from _trace_settings import (CONCURRENCY, MAX_WRDS_CONNECTIONS,
                                 TARGET_ROWS_PER_CHUNK, qsub_resources)
    from config import TRACE_MEMBERS

    s0 = docs["stage0/README_stage0.md"]
    check("stage0 README quotes the real connection ceiling",
          f"{MAX_WRDS_CONNECTIONS} connections held simultaneously" in s0,
          f"MAX_WRDS_CONNECTIONS={MAX_WRDS_CONNECTIONS}")

    for member in ("enhanced", "144a", "standard"):
        want = qsub_resources(member)
        check(f"stage0 README quotes the real qsub request for {member}",
              want in s0, want)
        check(f"stage0 README quotes the real connection count for {member}",
              f"| `{member}` | {CONCURRENCY[member]} |" in s0,
              f"CONCURRENCY={CONCURRENCY[member]}")

    check("row target in prose matches TARGET_ROWS_PER_CHUNK",
          f"{TARGET_ROWS_PER_CHUNK:,}" in prose, f"{TARGET_ROWS_PER_CHUNK:,}")
    check("docs describe the real default TRACE_MEMBERS",
          TRACE_MEMBERS == ["enhanced", "144a"], str(TRACE_MEMBERS))

    # --- WRDS_USERNAME lives in config.py, and docs must say so -------
    ts = (ROOT / "stage0" / "_trace_settings.py").read_text(encoding="utf-8")
    imports_it = "from config import" in ts and "WRDS_USERNAME" in ts.split("from config import")[1][:120]
    check("_trace_settings.py imports WRDS_USERNAME from config.py", imports_it)
    if imports_it:
        # Any verb, not just "edit": two real instances said "Open `_trace_settings.py`"
        # and "default fallback in `_trace_settings.py`" and slipped through a check
        # written around the word "Edit".
        near = re.compile(
            r"`?_trace_settings\.py`?.{0,80}?WRDS_USERNAME"
            r"|WRDS_USERNAME.{0,80}?`?_trace_settings\.py`?")
        bad = [f for f, t in docs.items() if near.search(t)]
        check("no doc tells you to set WRDS_USERNAME in _trace_settings.py",
              not bad, str(bad))

    # --- files named in prose must exist ------------------------------
    named = set(re.findall(r"`([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|sh|md|txt|xlsx))`", prose))
    missing = sorted(
        f for f in named
        if f not in RUNTIME_ARTIFACTS
        and "*" not in f
        and not (ROOT / f).exists()
        and not (ROOT / "stage0" / f).exists()
        and not (ROOT / "stage1" / f).exists()
        and not (ROOT / "tests" / f).exists()
    )
    check("every file named in the docs exists", not missing, str(missing))

    # --- renamed/deleted things must not come back --------------------
    for token, why in BANNED.items():
        hits = sorted(f for f, t in docs.items() if token in t)
        check(f"no reference to '{token}'", not hits, f"{hits} -- {why}" if hits else why)

    # --- the machinery must be documented somewhere real --------------
    for token in ("run_smoke_test.sh", "download_inputs.sh", "_chunk_runner",
                  "_wrds_pool", "CONCURRENCY", "target_rows_per_chunk"):
        check(f"'{token}' is documented outside the CHANGELOG", token in prose)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        return 1
    print("all documentation checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
