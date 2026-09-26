"""_stage4_settings.py -- what Stage 4 reads, where it writes, and the grid it builds.

Stage 4 turns Stage 2's monthly panel into the TRACE-only bond factors published on
openbondassetpricing.com. It reads only Stage 2's build output, on your own computer, and
opens no WRDS connection.

Every path can be overridden from the environment; the grid is `spec/factors.json`.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent                        # the trace-data-pipeline root
# pybondlab_pin.py sits at the root. Appended, so nothing there can shadow a Stage 4 module.
if str(PIPELINE) not in sys.path:
    sys.path.append(str(PIPELINE))
import numeric_setup  # noqa: E402,F401  (pandas computes the same whatever else is installed)

STAGE1_DIR = Path(os.environ.get("STAGE1_DIR", PIPELINE / "stage1"))
STAGE2_DIR = Path(os.environ.get("STAGE2_DIR", PIPELINE / "stage2"))
MODE = os.environ.get("STAGE4_MODE", "stage1")          # Stage 2's input mode

# ❗Stage 2's BUILD output, never `stage2/release/`. The released copy nulls permco and gvkey
# and collapses the ratings, which changes every within-firm sort.
PANEL = STAGE2_DIR / "output" / "panel" / f"main_panel_{MODE}.parquet"
BLOCKS = STAGE2_DIR / "output" / "blocks" / MODE
RISK_FREE = BLOCKS / "factors.parquet"        # the `rf` column: the one-month T-bill
RISK_FREE_COLUMN = "rf"

OUTPUT = Path(os.environ.get("STAGE4_OUTPUT", HERE / "output"))

SPEC = json.loads((HERE / "spec" / "factors.json").read_text(encoding="utf-8"))
SIGNALS: list[str] = SPEC["signals"]["names"]
RETURN_TYPES: dict = {k: v for k, v in SPEC["return_types"].items() if not k.startswith("_")}
SORTS = ("single", "within_firm")


def block_path(name: str) -> Path:
    return BLOCKS / f"{name}.parquet"


def vintage() -> str:
    """The four-digit year the factors are published under, e.g. "2026".

    The same rule as Stage 2's `release_vintage`: the year of the Stage 1 file the panel was
    built from, so next year's run names itself. STAGE4_VINTAGE overrides it.
    """
    override = os.environ.get("STAGE4_VINTAGE")
    if override:
        return override
    stamps = sorted(m.group(1) for p in (STAGE1_DIR / "data").glob("stage1_*.parquet")
                    if (m := re.fullmatch(r"stage1_(\d{8})\.parquet", p.name)))
    return stamps[-1][:4] if stamps else str(datetime.now(timezone.utc).year)
