"""pybondlab_pin.py -- the PyBondLab release Stages 2, 3 and 4 run on, and the check for it.

Stage 2's BBW factors and every portfolio sort in Stages 3 and 4 come out of PyBondLab, so
its version is part of every number those stages write. It is pinned exactly, here and
nowhere else: `requirements-local.txt` and the guides quote it, and `tests/test_docs.py`
checks that they agree with `VERSION`.

PyBondLab is installed in a second step, without its dependency list:

    python -m pip install -r requirements-local.txt
    python -m pip install --no-deps pybondlab==0.3.0

PyBondLab 0.3.0 declares `numpy<2`. This repository installs numpy 2, so pip would refuse
to install the two together. PyBondLab's code runs on numpy 2 -- its test suite passes on
numpy 2.5.3 -- and everything else it needs is already in `requirements-local.txt`.

    check()        None when the environment can run Stages 2-4 (this PyBondLab, and numba and
                   numexpr installed), else what is wrong and the fix
    fingerprint()  the version and a content hash of the installed package, for manifests
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

VERSION = "0.3.0"

INSTALL = ("python -m pip install -r requirements-local.txt\n"
           f"python -m pip install --no-deps pybondlab=={VERSION}")


def installed_version() -> str | None:
    """The `__version__` of the PyBondLab that `import PyBondLab` loads, or None."""
    if importlib.util.find_spec("PyBondLab") is None:
        return None
    import PyBondLab
    return getattr(PyBondLab, "__version__", "unknown")


def check() -> str | None:
    """None when PyBondLab is the pinned release and numba and numexpr are present; else the fix.

    numexpr changes numbers, not just speed: see numeric_setup.py."""
    problems = []
    got = installed_version()
    if got is None:
        problems.append("PyBondLab is not installed")
    elif got != VERSION:
        problems.append(f"PyBondLab {got} is installed, and this repository runs on {VERSION}")
    for pkg in ("numba", "numexpr"):
        if importlib.util.find_spec(pkg) is None:
            problems.append(f"{pkg} is not installed")
    if not problems:
        return None
    return ("; ".join(problems) + ". From the repository root:\n\n    "
            + INSTALL.replace("\n", "\n    ") + "\n")


def tree_sha256(pkg: Path) -> str:
    """Content hash of the package source -- the same PyBondLab gives the same hash."""
    h = hashlib.sha256()
    for p in sorted(pkg.rglob("*")):
        if p.is_file() and p.suffix in (".py", ".csv", ".json") and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(pkg)).replace("\\", "/").encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def fingerprint() -> dict:
    """What a manifest records about PyBondLab: the version and a content hash, never a path."""
    import PyBondLab
    pkg = Path(PyBondLab.__file__).resolve().parent
    return {"version": getattr(PyBondLab, "__version__", "unknown"),
            "tree_sha256": tree_sha256(pkg)[:16]}


if __name__ == "__main__":
    problem = check()
    print(problem or f"PyBondLab {VERSION}: ok  {fingerprint()}")
    raise SystemExit(1 if problem else 0)
