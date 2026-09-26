"""pblenv.py -- check the PyBondLab Stage 3 runs on, and record it with every number.

Stage 3 runs every sort through PyBondLab, so the release that produced a number is part of
the number. `pybondlab_pin.py`, at the repository root, names that release. This module
checks that it is the one installed, and records it:

    from pblenv import use
    prov = use()                 # stops, with the install lines, if it is not the pinned one
    import PyBondLab as pbl

`prov` -- the version and a content hash of the package, never a path -- goes into every
manifest Stage 3 writes, so any number can be traced back to the engine that made it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import _stage3_settings as S

import pybondlab_pin  # noqa: E402  (the repository root; _stage3_settings puts it on the path)

_ACTIVE: dict | None = None
_LOCATION: str | None = None


def problem() -> str | None:
    """What is wrong with the installed PyBondLab, with the fix; None when it is the pinned one."""
    return pybondlab_pin.check()


def use(*, quiet: bool = False) -> dict:
    """Check the installed PyBondLab against the pin and return its provenance record.

    Idempotent within a process. Stops with the install lines if PyBondLab is missing or a
    different release, so no number is ever attributed to an engine nobody chose.
    """
    global _ACTIVE, _LOCATION
    if _ACTIVE is not None:
        return dict(_ACTIVE)
    why = problem()
    if why:
        raise SystemExit("Stage 3 cannot start: " + why)

    import PyBondLab as pbl

    _ACTIVE = pybondlab_pin.fingerprint()
    # A manifest travels; a local path in it names someone's home directory and tells a
    # reader nothing they can act on. `_LOCATION` is for the console line and error messages.
    _LOCATION = str(Path(pbl.__file__).resolve().parent.parent)
    if not quiet:
        print(f"[pblenv] PyBondLab v{_ACTIVE['version']} tree={_ACTIVE['tree_sha256']}"
              f"  <- {_LOCATION}", flush=True)
    return dict(_ACTIVE)


def location() -> str:
    """Where the active PyBondLab was loaded from. For humans, never for a manifest."""
    return _LOCATION or "(not checked yet)"


def active() -> dict:
    """The provenance record of the PyBondLab in use. Raises if `use()` has not been called."""
    if _ACTIVE is None:
        raise RuntimeError("PyBondLab has not been checked -- call pblenv.use() first")
    return dict(_ACTIVE)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(use(), indent=2))
