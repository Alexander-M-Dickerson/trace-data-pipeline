r"""doctor.py -- is this machine ready to run the pipeline, and what is the next step?

    python doctor.py            your own computer: stages 2, 3 and 4
    python doctor.py --wrds     the WRDS login node: stages 0 and 1

It only looks: it downloads nothing, opens no WRDS connection and creates no files. Each stage
is asked through its own check -- stage 2's `_run_stage2.py --dry-run`, stage 3's
`tools/check_inputs.py`, stage 4's `build_factors.py --dry-run`, and on WRDS
`download_inputs.sh --check` and `check_disk_space.sh` -- so this file keeps no second copy
of what a stage needs. It exits 1 when the Python environment itself is not ready, else 0.
"""
# [tag:entry.doctor] python doctor.py: what is ready, what is missing, and the next step
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# The Python versions stages 2-4 install on; requirements-local.txt says why, and
# tests/test_doctor.py fails if the two disagree.
PYTHON_RANGE = ((3, 11), (3, 13))
PLACEHOLDER_USERNAME = "your_wrds_username"
PANEL = ROOT / "stage2" / "output" / "panel" / "main_panel_stage1.parquet"
EXHIBITS = ROOT / "stage3" / "reports" / "exhibits.pdf"
FACTORS = ROOT / "stage4" / "output"
TIMEOUT = 600


@dataclass
class Check:
    name: str
    ok: bool | None          # None: not checked yet, it waits on an earlier step
    detail: str = ""


def run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    """Run one of the stages' own checks and return (exit code, what it printed)."""
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, str(e)
    return r.returncode, r.stdout + r.stderr


def marker_holds(marker: str) -> bool:
    """Does an environment marker (`python_version<"3.14"`) hold for this Python? pip always
    ships `packaging`; without it, assume it holds."""
    try:
        from packaging.markers import Marker
    except ImportError:
        try:
            from pip._vendor.packaging.markers import Marker
        except ImportError:
            return True
    return Marker(marker).evaluate()


def requirement_names(path: Path) -> list[str]:
    """The package names a requirements file installs on this Python, following `-r` lines and
    skipping a line whose marker does not hold (numba is not installed on Python 3.14)."""
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("-r "):
            names += requirement_names(path.parent / line[3:].strip())
        elif line and not line.startswith("-"):
            spec, _, marker = line.partition(";")
            if marker.strip() and not marker_holds(marker.strip()):
                continue
            names.append(re.match(r"[A-Za-z0-9_.\-]+", spec.strip()).group(0))
    return list(dict.fromkeys(names))


def check_python(enforce_range: bool) -> Check:
    v = sys.version_info[:2]
    lo, hi = PYTHON_RANGE
    ok = lo <= v <= hi or not enforce_range
    detail = f"{sys.version.split()[0]} at {sys.executable}"
    if not ok:
        detail += (f" -- stages 2-4 need {lo[0]}.{lo[1]} to {hi[0]}.{hi[1]} "
                   f"(requirements-local.txt says why)")
    return Check("Python", ok, detail)


def check_packages(req: Path) -> Check:
    missing = []
    for name in requirement_names(req):
        try:
            md.version(name)
        except md.PackageNotFoundError:
            missing.append(name)
    n = len(requirement_names(req))
    if missing:
        return Check(f"packages in {req.name}", False, "not installed: " + ", ".join(missing))
    return Check(f"packages in {req.name}", True, f"all {n} installed")


def check_pybondlab() -> Check:
    import pybondlab_pin
    try:
        problem = pybondlab_pin.check()
    except Exception as e:  # noqa: BLE001 - report it, whatever it is, rather than stop
        return Check("PyBondLab, numba, numexpr", False, f"PyBondLab could not be checked: {e}")
    if problem:
        return Check("PyBondLab, numba, numexpr", False, problem.strip().splitlines()[0])
    return Check("PyBondLab, numba, numexpr", True, f"PyBondLab {pybondlab_pin.VERSION}")


def _box_body(out: str) -> str:
    """The message inside a stage's CONFIGURATION ERROR box, or its last lines."""
    m = re.search(r"CONFIGURATION ERROR\n=+\n(.*?)\n=+", out, re.S)
    text = m.group(1) if m else "\n".join(out.strip().splitlines()[-8:])
    return text.strip()


def check_stage2() -> Check:
    rc, out = run([sys.executable, "_run_stage2.py", "--dry-run"], ROOT / "stage2")
    if rc == 0:
        built = "the panel is built" if PANEL.exists() else "the panel is not built yet"
        return Check("stage 2 inputs", True, f"ready; {built}")
    return Check("stage 2 inputs", False, _box_body(out))


def check_stage3() -> Check:
    if not PANEL.exists():
        return Check("stage 3 inputs", None, "waits for stage 2's panel")
    rc, out = run([sys.executable, "tools/check_inputs.py", "--json"], ROOT / "stage3")
    try:
        rows = json.loads(out[out.index("{"):])["inputs"]
    except (ValueError, KeyError):
        rows = []
    bad = [f"{r['input']}: {'; '.join(r['problems'])}" for r in rows if r.get("problems")]
    if rc == 0 and not bad:
        done = "the exhibits are built" if EXHIBITS.exists() else "not run yet"
        return Check("stage 3 inputs", True, f"ready; {done}")
    return Check("stage 3 inputs", False, "\n".join(bad) or _box_body(out))


def _factors_built() -> bool:
    return any(FACTORS.glob("*_sort_panel_trace_*"))


def check_stage4() -> Check:
    if not PANEL.exists():
        return Check("stage 4 inputs", None, "waits for stage 2's panel")
    rc, out = run([sys.executable, "build_factors.py", "--dry-run"], ROOT / "stage4")
    if rc == 0:
        done = "the factors are built" if _factors_built() else "not run yet"
        return Check("stage 4 inputs", True, f"ready; {done}")
    tail = out.split("Stage 4 cannot start:", 1)[-1].strip()
    return Check("stage 4 inputs", False, "\n".join(tail.splitlines()[:6]))


def next_step_local(checks: dict[str, Check]) -> str:
    import pybondlab_pin
    if not checks["python"].ok:
        return ("install Python 3.11, 3.12 or 3.13, make a virtual environment with it, and "
                "install from the repository root:\n" + pybondlab_pin.INSTALL)
    if not checks["packages"].ok or not checks["pybondlab"].ok:
        return "install the requirements, from the repository root:\n" + pybondlab_pin.INSTALL
    s2 = checks["stage2"]
    if not s2.ok:
        if "directory not found" in s2.detail or "daily panel not found" in s2.detail:
            return ("copy the stage0/ and stage1/ folders your WRDS run produced into this "
                    "folder (QUICKSTART.md, \"Download Results to Your Local Machine\"), then "
                    "run python doctor.py again")
        if "WRDS_USERNAME" in s2.detail:
            return ("set your WRDS username (export WRDS_USERNAME=your_wrds_id, or edit config.py): "
                    "stage 2's first run fetches Treasury, Fama-French, VIX and FISD data")
        return "fix the stage 2 problem above (stage2/QUICKSTART_stage2.md), then run this again"
    if not PANEL.exists():
        return ("build the monthly panel: cd stage2 && python _run_stage2.py "
                "(about 8-18 minutes; stage2/QUICKSTART_stage2.md)")
    todo = []
    s3, s4 = checks["stage3"], checks["stage4"]
    if s3.ok is False:
        todo.append("stage 3's inputs fail their check (above; stage3/QUICKSTART_stage3.md)")
    elif not EXHIBITS.exists():
        todo.append("stage 3: bash stage3/run_stage3.sh (about 15-20 minutes)")
    if s4.ok is False and "_bns" in s4.detail:
        todo.append("stage 4 first needs the benchmark blocks: cd stage2 && python "
                    "make_excess_blocks.py --mode stage1 --benchmark all")
    elif s4.ok is False:
        todo.append("stage 4's inputs fail their check (above; stage4/README_stage4.md)")
    elif not _factors_built():
        todo.append("stage 4: bash stage4/run_stage4.sh (about 8 minutes)")
    return "; or ".join(todo) if todo else "everything is built; see INDEX.md for what to read"


def check_username() -> Check:
    import config
    user = config.WRDS_USERNAME
    if not user or user == PLACEHOLDER_USERNAME:
        return Check("WRDS_USERNAME", False,
                     "not set: export WRDS_USERNAME=your_wrds_id, or edit config.py")
    return Check("WRDS_USERNAME", True, user)


def local() -> tuple[list[Check], str, bool]:
    checks = {"python": check_python(True),
              "packages": check_packages(ROOT / "requirements-local.txt"),
              "pybondlab": check_pybondlab()}
    env_ok = all(c.ok for c in checks.values())
    if env_ok:
        checks["stage2"] = check_stage2()
        checks["stage3"] = check_stage3()
        checks["stage4"] = check_stage4()
    return list(checks.values()), next_step_local(checks), env_ok


def wrds() -> tuple[list[Check], str, bool]:
    import config
    checks = [check_python(False), check_packages(ROOT / "requirements.txt"),
              check_username(),
              Check("TRACE_MEMBERS", True, " ".join(config.TRACE_MEMBERS))]
    rc, out = run(["bash", "download_inputs.sh", "--check"], ROOT)
    missing = [l.split(":", 1)[1].strip() for l in out.splitlines() if "Missing or empty" in l]
    checks.append(Check("stage 1's downloaded inputs", rc == 0,
                        "all present" if rc == 0 else "missing: " + ", ".join(missing)))
    rc, out = run(["bash", "check_disk_space.sh"], ROOT)
    verdict = [l.strip() for l in out.splitlines() if l.strip().startswith("[")]
    checks.append(Check("room in your home quota", rc == 0,
                        verdict[-1] if verdict else out.strip()[-300:]))
    env_ok = checks[1].ok
    if not checks[1].ok:
        step = "install the requirements: python -m pip install -r requirements.txt"
    elif not checks[2].ok:
        step = "set your WRDS username: export WRDS_USERNAME=your_wrds_id"
    elif not checks[4].ok:
        step = "fetch stage 1's inputs, here on the login node: bash download_inputs.sh"
    elif not checks[5].ok:
        step = "free some space in your home directory (the message above says how much)"
    else:
        step = ("a 10-minute test on a few chunks: qsub run_smoke_test.sh; then the full run: "
                "./run_pipeline.sh (QUICKSTART.md)")
    return checks, step, env_ok


def report(title: str, checks: list[Check], step: str) -> str:
    mark = {True: "[ok]", False: "[no]", None: "[..]"}
    lines = [title, ""]
    for c in checks:
        first, *rest = [l for l in (c.detail or "").splitlines() if l.strip()] or [""]
        lines.append(f"  {mark[c.ok]}  {c.name:<34} {first}")
        lines += [f"  {'':<4}  {'':<34} {r}" for r in rest]
    lines += ["", "Next: " + step.replace("\n", "\n      ")]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wrds", action="store_true",
                    help="check the WRDS login node for stages 0 and 1 instead")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.wrds:
        checks, step, ok = wrds()
        title = "doctor: the WRDS login node, stages 0 and 1"
    else:
        checks, step, ok = local()
        title = "doctor: your own computer, stages 2, 3 and 4"
    print(report(title, checks, step))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
