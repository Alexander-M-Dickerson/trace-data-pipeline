# -*- coding: utf-8 -*-
"""
test_parity_vs_reference.py
===========================
Prove this port reproduces the reference implementation exactly.

The engine under lib/ and steps/ was ported from a validated implementation with only an
import rename. This test is what turns that claim into evidence: build both on the same
inputs and diff the outputs key-by-key.

It is a DEVELOPMENT test. It needs a reference build on disk, so it skips unless
STAGE2_REFERENCE_OUTPUT names one:

    STAGE2_REFERENCE_OUTPUT=/path/to/monthly_data/output \\
        python -m pytest tests/test_parity_vs_reference.py -v

Target is exact equality. Anything else is investigated before it is accepted -- the
point of building on unchanged inputs is that there is no legitimate reason to differ.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import _stage2_settings as cfg
from lib import validate_core as vc

_REF = os.environ.get("STAGE2_REFERENCE_OUTPUT")
REF_ROOT = Path(_REF) if _REF else None
MODE = os.environ.get("STAGE2_PARITY_MODE", "ours")

pytestmark = pytest.mark.skipif(
    REF_ROOT is None,
    reason="set STAGE2_REFERENCE_OUTPUT to a reference build's output/ directory",
)

# (label, our path under stage2/output, reference path under its output, join key)
CASES = [
    ("main_panel",     f"panel/main_panel_{MODE}.parquet",
                       "panel/main_panel_ours.parquet",          ("cusip", "date")),
    ("returns_alt",    f"blocks/{MODE}/returns_alt.parquet",
                       "blocks/ours/returns_alt.parquet",        ("cusip", "date")),
    ("illiq_factors",  f"blocks/{MODE}/illiq_factors.parquet",
                       "blocks/ours/illiq_factors.parquet",      ("date",)),
    ("bbw_factors",    f"blocks/{MODE}/bbw_factors.parquet",
                       "blocks/ours/bbw_factors.parquet",        ("date",)),
    ("factors_merged", f"blocks/{MODE}/factors_merged.parquet",
                       "blocks/ours/factors_merged.parquet",     ("date",)),
    ("betas_std",      f"blocks/{MODE}/betas_std.parquet",
                       "blocks/ours/betas_std.parquet",          ("cusip", "date")),
    ("betas_x",        f"blocks/{MODE}/betas_x.parquet",
                       "blocks/ours/betas_x.parquet",            ("cusip", "date")),
    ("mom_retx",       f"blocks/{MODE}/mom_retx.parquet",
                       "blocks/ours/mom_retx.parquet",           ("cusip", "date")),
    ("value_signals",  f"blocks/{MODE}/value_signals_adj.parquet",
                       "blocks/ours/value_signals_adj.parquet",  ("cusip", "date")),
    ("end_returns",    f"blocks/{MODE}/end_returns.parquet",
                       "blocks/ours/end_returns.parquet",        ("cusip", "date")),
    # The _mmn sidecar carries a date stamp in its filename, and the two builds derive
    # that stamp differently (we take it from the input file; the reference hard-coded
    # one). Matched by glob so a naming difference cannot masquerade as a data difference.
    ("mmn_signals",    f"blocks/{MODE}/mmn_price_based_signals_*.parquet",
                       "blocks/ours/mmn_price_based_signals_*.parquet", ("cusip", "date")),
]

# Rate-like columns get the looser tolerance the reference uses for them; see
# _stage2_settings.RATE_TOL. Everything else must meet FLOAT_TOL.
_RATE_PREFIXES = ("b_", "ivol_", "iskew", "sysmom", "idimom")
_RATE_EXTRA = ("ytm", "cs")


def _rate_cols(df) -> list[str]:
    return [c for c in df.columns
            if c.startswith(_RATE_PREFIXES) or c in _RATE_EXTRA
            or (c.endswith("_mmn") and c[:-4] in _RATE_EXTRA)]


@pytest.mark.parametrize("label,ours_rel,ref_rel,key",
                         CASES, ids=[c[0] for c in CASES])
def test_block_matches_reference(label, ours_rel, ref_rel, key):
    import pandas as pd

    def _resolve(root: Path, rel: str) -> Path | None:
        if "*" in rel:
            hits = sorted(root.glob(rel))
            return hits[-1] if hits else None
        p = root / rel
        return p if p.exists() else None

    ours_path = _resolve(cfg.OUTPUT_DIR, ours_rel)
    ref_path = _resolve(REF_ROOT, ref_rel)
    if ref_path is None:
        pytest.skip(f"reference block absent: {REF_ROOT / ref_rel}")
    assert ours_path is not None, (
        f"{label}: our build produced no {ours_rel}. Run the full build first."
    )

    ours = pd.read_parquet(ours_path)
    ref = pd.read_parquet(ref_path)

    passed, report = vc.diff_frames(
        ours, ref, key=list(key),
        float_tol=cfg.FLOAT_TOL, rate_tol=cfg.RATE_TOL,
        rate_cols=_rate_cols(ours),
    )

    # Report shape problems separately -- a key gap is a different failure from a value
    # drift, and saying which one it is saves the next person an hour.
    assert not report["key_gaps_ours"], (
        f"{label}: our build has {report['key_gaps_ours']} key(s) the reference does not "
        f"(ours {report['rows_ours']:,} rows vs reference {report['rows_golden']:,})"
    )
    assert not report["key_gaps_golden"], (
        f"{label}: {report['key_gaps_golden']} reference key(s) are missing from our "
        f"build (ours {report['rows_ours']:,} rows vs reference {report['rows_golden']:,})"
    )
    assert not report["cols_only_ours"], \
        f"{label}: columns only in our build: {report['cols_only_ours']}"
    assert not report["cols_only_golden"], \
        f"{label}: columns only in the reference: {report['cols_only_golden']}"

    failed = [r for r in report["columns"] if not r["pass"]]
    assert not failed, (
        f"{label}: {len(failed)} of {len(report['columns'])} column(s) differ:\n" +
        "\n".join(
            f"    {r['col']}: " +
            (f"max|d|={r.get('max_abs_d')} nan_mismatch={r.get('nan_mismatch')}"
             if r.get("kind") != "exact" else f"n_mismatch={r.get('n_mismatch')}")
            for r in failed[:12])
    )


def test_reference_build_is_present():
    """Fail loudly if the reference directory exists but holds no panel at all."""
    panel = REF_ROOT / "panel" / "main_panel_ours.parquet"
    assert panel.exists(), (
        f"STAGE2_REFERENCE_OUTPUT points at {REF_ROOT}, which has no "
        f"panel/main_panel_ours.parquet. Point it at a reference build's output/ folder."
    )
