"""signal_set.py -- which panel columns are signals, read from the one list that defines them.

A column is a signal when its row in `spec/signal_definitions.json` (Table IA.VIII) sits in one
of the nine Cluster groups: 108 of the panel's 145 columns. Every other row -- identifiers,
returns, dates, ratings, amounts, industries, and the Treasury benchmark returns -- is not a
signal, and no section of Stage 3 may sort on it.

❗Until 2026-09-27 the Section 3 census chose its signals by EXCLUSION: every panel column not
on a hand-written list of identifier and return columns. The 2026 panel added five Treasury
benchmark returns (`tret_bns`, `tret_cfm`, `tret_gprs`, `tret_cls`, `tret_mat`) that the list did
not name, so the census sorted 113 "signals", five of them Treasury returns, and Table B.1
counted them. Selection is now by INCLUSION from the spec, and a panel column the spec does not
classify stops the run instead of being guessed at.

Adding a signal is a change to the spec (a Cluster row) and to the cluster maps of Sections 4
and 5; `tests/test_stage3_contract.py` holds all of them to this list.

Author: Open Source Bond Asset Pricing
"""
from __future__ import annotations

import json
from pathlib import Path

SPEC = Path(__file__).resolve().parent / "spec" / "signal_definitions.json"
_ROWS = json.loads(SPEC.read_text(encoding="utf-8"))["rows"]

# [tag:rule.signal_set] the signals are the spec's Cluster rows; everything else is never sorted
SIGNALS: tuple[str, ...] = tuple(r["mnemonic"] for r in _ROWS if r["group"].startswith("Cluster"))
NOT_SIGNALS: frozenset[str] = frozenset(
    r["mnemonic"] for r in _ROWS if not r["group"].startswith("Cluster"))
# Columns Stage 3 derives from the panel itself, never signals.
DERIVED: frozenset[str] = frozenset({"ret_vwx", "ret_vwx_bgn"})

_SIGNAL_SET = frozenset(SIGNALS)
assert len(SIGNALS) == len(_SIGNAL_SET), "a signal is listed twice in the spec"
assert not (_SIGNAL_SET & NOT_SIGNALS), "a column is both a signal and not one in the spec"


def select(columns, *, what: str = "the panel") -> list[str]:
    """The signals among `columns`, in their order.

    Unadjusted `_mmn` twins are left to the caller. Raises when a column is neither a signal
    nor a declared non-signal (so a new column cannot slip into a sort unclassified), and when
    a signal is missing (so a census cannot silently shrink).
    """
    cols = [str(c) for c in columns if not str(c).endswith("_mmn")]
    unknown = sorted(c for c in cols
                     if c not in _SIGNAL_SET and c not in NOT_SIGNALS and c not in DERIVED)
    if unknown:
        raise ValueError(
            f"{what} has {len(unknown)} column(s) that spec/signal_definitions.json does not "
            f"classify: {unknown}.\n  Give each a row in the spec: in a Cluster group if it is a "
            "signal, in another group if it is not. Nothing is sorted until then.")
    missing = [s for s in SIGNALS if s not in cols]
    if missing:
        raise ValueError(f"{what} lacks {len(missing)} of the {len(SIGNALS)} signals: {missing}")
    return [c for c in cols if c in _SIGNAL_SET]


def signal_of(factor: str) -> str:
    """The signal a PyBondLab factor name was sorted on.

    The census names its series `<signal>[_mmn][_wf][*]`: `_mmn` where the unadjusted twin was
    swapped in, `_wf` for a within-firm sort, `*` where PyBondLab flipped the sign.
    """
    name = str(factor).rstrip("*")
    for suffix in ("_wf", "_mmn"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


assert not [s for s in SIGNALS if s != signal_of(s)], "a signal's own name ends in a suffix"


def check_census(factors, *, what: str) -> None:
    """A finished sort must cover exactly the signals: none missing, nothing else.

    `factors` are PyBondLab factor names, read through `signal_of`.
    """
    got = {signal_of(f) for f in factors}
    extra, missing = sorted(got - _SIGNAL_SET), sorted(_SIGNAL_SET - got)
    if extra or missing:
        raise ValueError(
            f"{what} is not the census of the {len(SIGNALS)} signals: "
            f"{len(extra)} column(s) that are not signals {extra}, "
            f"{len(missing)} signal(s) missing {missing}.\n"
            "  Rebuild it, from stage3/: `python _run_stage3.py --section lib --force`. A census "
            "written before 4.1.1 holds five Treasury benchmark returns.")
