"""helper_functions.py -- the small conventions, in one importable place.

Mirrors `stage1/helper_functions.py`. These four are used everywhere in Stage 3 and
each of them is a decision someone could plausibly make differently; having them here
means the decision is made once.

The heavier shared machinery -- loading, Newey-West inference, CAPM_B alphas, the
paired-difference test, provenance -- lives in `drrlib.py`.
"""
from __future__ import annotations

import numpy as np


def nw_lags(T: int) -> int:
    """The one lag convention in Stage 3: floor(T**0.25). T=268 -> 4.

    Every standard error in the paper depends on it, so T must be right before a
    t-statistic means anything -- `drrlib.assert_sample` is the check.
    """
    return int(np.floor(T ** 0.25))


def strip_sign_flag(factor: str) -> tuple[str, float]:
    """('dcs6*') -> ('dcs6', -1.0). A trailing '*' means PyBondLab NEGATED the series.

    It decides the flip from the sign of the full-sample mean at extract time, so the
    flip set depends on the sample that was sorted. Two runs over different windows can
    disagree about which factors are starred, and that is correct rather than a bug --
    compare flip SETS, never assume they match.
    """
    return (factor[:-1], -1.0) if factor.endswith("*") else (factor, 1.0)


def base_mnemonic(factor: str) -> str:
    """'dcs6_mmn_wf_hy*' -> 'dcs6'. Strips the flip star and the decorations.

    The decorations stack in any order (`_mmn` noisy twin, `_wf` within-firm, `_ig` /
    `_hy` / `_nig` rating splits), so strip until stable.
    """
    f, _ = strip_sign_flag(factor)
    changed = True
    while changed:
        changed = False
        for suf in ("_wf", "_mmn", "_ig", "_hy", "_nig"):
            if f.endswith(suf):
                f = f[: -len(suf)]
                changed = True
    return f


def to_percent(x, dec: int = 2):
    """Sort panels store DECIMALS; every table prints percent. Convert at the boundary."""
    return round(float(x) * 100, dec)
