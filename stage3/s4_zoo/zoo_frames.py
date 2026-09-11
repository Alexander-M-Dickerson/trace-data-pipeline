r"""zoo_frames.py -- the four factor-zoo statistics frames, computed once.

Tables IA.IX, IA.X, IA.XI and the two inline specification counts all read the same
four frames -- (value-weighted, equal-weighted) x (single, within-firm) -- each holding
one row per factor with its mean, alpha, t-statistics, Sharpe and information ratios,
and whether it survives Benjamini-Hochberg.

❗Every frame must carry all 108 factors, and that is asserted rather than hoped for.
The false-discovery threshold is computed over m = the number of factors in the frame,
so a frame short by one silently loosens the threshold for all the others.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # stage3/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import drrlib as D          # noqa: E402
import paths                # noqa: E402
import zoo_engine as Z      # noqa: E402

PANELS = {"Panel A": "single", "Panel B": "wf"}
SPEC_COLS = [("vw", "single"), ("vw", "wf"), ("ew", "single"), ("ew", "wf")]


def load_frames(weighting: str, *, end: str = Z.DATE_END) -> dict:
    """{(weighting, sort): statistics frame} for one weighting's two panels."""
    for sort in ("single", "wf"):
        f = Z.zoo_csv(sort)
        if not f.exists():
            raise SystemExit(
                f"the zoo sort CSVs are not under {f.parent}.\n"
                "  Produce them with `python s4_zoo/run_zoo_sorts.py`.")
    mktb = D.load_mktb(paths.BBW, start=Z.DATE_START, end=end)
    out = {}
    for sort in ("single", "wf"):
        wide = Z.load_wide(sort, weighting, end=end)
        df = Z.zoo_stats(wide, mktb)
        if len(df) != Z.N_FACTORS:
            raise AssertionError(
                f"{weighting}/{sort}: {len(df)} factors after the gates, expected "
                f"{Z.N_FACTORS}. The false-discovery threshold is computed over this "
                "count, so a short frame would loosen it for every factor.")
        out[(weighting, sort)] = df
    return out


def load_all(*, end: str = Z.DATE_END) -> dict:
    """All four frames: {(weighting, sort): frame}."""
    out = {}
    for w in ("vw", "ew"):
        out.update(load_frames(w, end=end))
    return out
