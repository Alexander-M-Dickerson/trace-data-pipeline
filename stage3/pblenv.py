"""pblenv.py -- select the PyBondLab build, and prove which one produced the numbers.

Stage 3 runs every sort through PyBondLab, so which copy of it is on `sys.path` is part of
the result. This module makes that explicit rather than incidental:

    from pblenv import use
    prov = use()                 # BEFORE importing PyBondLab
    import PyBondLab as pbl      # now guaranteed to be the build `prov` describes

`prov` goes into every manifest Stage 3 writes, so any number can be traced back to the
engine that made it.

Two ways to point it at a build:

  * `PYBONDLAB_DIR` (or `_stage3_settings.PYBONDLAB_DIR`) names a checkout. It is put at
    the FRONT of `sys.path` and the import is then asserted to have resolved inside it --
    a second PyBondLab installed in the environment can otherwise shadow it silently.
  * unset: whatever `import PyBondLab` finds. Fine when that install already carries the
    fast kernels.

❗The uncertainty grids (Section 5) need `PyBondLab.fast_sorts` and `anomaly_assay_fast`.
Those are not in the 0.2.0 release that Stage 2 pins. `require_fast()` checks for them and
says what to do, rather than letting a 40-hour run start and fail on an import.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import _stage3_settings as S

_ACTIVE: dict | None = None


def _git(root: Path, *args: str) -> str | None:
    """A git field for `root`, but only if `root` is itself a repository top-level.

    A checkout nested inside another repository would otherwise report the ENCLOSING
    repository's HEAD -- provenance for the wrong tree.
    """
    try:
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=15)
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != root.resolve():
            return None
        r = subprocess.run(["git", "-C", str(root), *args],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None


def tree_sha256(pkg: Path) -> str:
    """Content hash of the package source -- catches an edited build that git alone misses."""
    h = hashlib.sha256()
    for p in sorted(pkg.rglob("*")):
        if p.is_file() and p.suffix in (".py", ".csv", ".json") and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(pkg)).replace("\\", "/").encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def use(build_dir: str | Path | None = None, *, quiet: bool = False) -> dict:
    """Import PyBondLab from `build_dir` (or the configured one) and return its provenance.

    Idempotent within a process. Raises if PyBondLab was already imported by something
    else, because Python cannot swap a loaded package and a silent no-op here would
    attribute numbers to the wrong engine.
    """
    global _ACTIVE
    if _ACTIVE is not None:
        return dict(_ACTIVE)

    root = build_dir or os.environ.get("PYBONDLAB_DIR") or S.PYBONDLAB_DIR
    root = Path(root).resolve() if root else None

    if root is not None:
        if not (root / "PyBondLab" / "__init__.py").exists():
            raise FileNotFoundError(
                f"PYBONDLAB_DIR={root} does not contain a PyBondLab package.\n"
                "  Point it at the checkout root -- the directory that HOLDS PyBondLab/,\n"
                "  not at PyBondLab/ itself.")
        if "PyBondLab" in sys.modules:
            raise RuntimeError(
                "PyBondLab was imported before pblenv.use(), so the build cannot be\n"
                "  guaranteed. Import pblenv and call use() first.")
        sys.path.insert(0, str(root))

    import PyBondLab as pbl  # noqa: E402  (deliberate: after the path insert)

    resolved = Path(pbl.__file__).resolve()
    if root is not None and root not in resolved.parents:
        raise RuntimeError(
            f"PyBondLab resolved to {resolved}, which is OUTSIDE {root}.\n"
            "  Something else on sys.path is shadowing the requested build.")

    pkg = resolved.parent
    _ACTIVE = {
        "version": getattr(pbl, "__version__", "?"),
        "path": str(root) if root else str(pkg.parent),
        "module_file": str(resolved),
        "pinned": root is not None,
        "git_branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD") if root else None,
        "git_sha": _git(root, "rev-parse", "HEAD") if root else None,
        "git_dirty": bool(_git(root, "status", "--porcelain")) if root else None,
        "tree_sha256": tree_sha256(pkg)[:16],
        "has_fast_kernels": has_fast_kernels(),
    }
    if not quiet:
        a = _ACTIVE
        git = (f" {a['git_branch']}@{(a['git_sha'] or '')[:7]}"
               f"{'+dirty' if a['git_dirty'] else ''}") if a["git_sha"] else ""
        fast = "fast kernels: yes" if a["has_fast_kernels"] else "fast kernels: NO"
        print(f"[pblenv] PyBondLab v{a['version']} tree={a['tree_sha256']}{git}"
              f"  {fast}  <- {a['path']}", flush=True)
    return dict(_ACTIVE)


def active() -> dict:
    """The provenance record of the build in use. Raises if `use()` has not been called."""
    if _ACTIVE is None:
        raise RuntimeError("no PyBondLab build selected -- call pblenv.use() first")
    return dict(_ACTIVE)


def has_fast_kernels() -> bool:
    """Whether this build carries the kernels the uncertainty grids need."""
    import importlib.util
    try:
        return all(importlib.util.find_spec(m) is not None
                   for m in ("PyBondLab.fast_sorts", "PyBondLab.anomaly_assay_fast"))
    except (ImportError, ValueError):
        return False


def require_fast(what: str) -> None:
    """Refuse to start a run that needs the fast kernels on a build without them."""
    if has_fast_kernels():
        return
    raise SystemExit(
        f"{what} needs PyBondLab's fast kernels (PyBondLab.fast_sorts and\n"
        "  PyBondLab.anomaly_assay_fast), and the build on sys.path does not have them.\n"
        f"  Active build: {active().get('path') if _ACTIVE else '(not selected)'}\n"
        "  Set PYBONDLAB_DIR to a checkout that carries them -- see README_stage3.md,\n"
        "  section 'PyBondLab'. The sort and bias sections (--section lib/lab/zoo) run\n"
        "  on the released PyBondLab without them.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(use(), indent=2))
