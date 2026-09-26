# -*- coding: utf-8 -*-
"""
test_download_inputs.py
=======================
`download_inputs.sh` runs at the start of every `run_pipeline.sh`.

It used to download with `wget -O FILE`, which empties FILE before the download starts, so
one failed re-run replaced a good file with an empty one, and the final check asked only
whether each file EXISTED. Stage 1 then failed hours later on the grid. It now downloads
beside the file and moves it into place only when it arrived whole, and the final check
refuses an empty file.

The tests run the real script, copied into a scratch tree, under bash with a fake `wget`
and a fake `python3` first on PATH.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "download_inputs.sh"
BASH = shutil.which("bash")
# On Windows, `bash` may be WSL's launcher, which runs Linux programs and not this PATH's fakes.
_WSL = BASH is not None and ("windowsapps" in BASH.lower()
                             or BASH.lower().endswith(r"system32\bash.exe"))
pytestmark = pytest.mark.skipif(BASH is None or _WSL or shutil.which("unzip") is None,
                                reason="bash (not WSL's launcher) and unzip are needed")

# What an earlier run left behind, and must still be there after a failed re-run.
FILES = {"liu_wu_yields.xlsx": b"the yields",
         "bond_firm_linker_2026/fl_linker.parquet": b"the linker",
         "Siccodes12.txt": b"ff12", "Siccodes17.txt": b"ff17", "Siccodes30.txt": b"ff30"}

# `wget -O FILE URL` empties FILE, then fails -- what GNU wget does when the network drops.
WGET_EMPTIES_THEN_FAILS = ('while [ $# -gt 0 ]; do [ "$1" = -O ] && : > "$2"; shift; done; '
                           'exit 4')
WGET_WRITES_NOTHING = ('while [ $# -gt 0 ]; do [ "$1" = -O ] && : > "$2"; shift; done; exit 0')


def _fake(bindir: Path, name: str, body: str) -> None:
    p = bindir / name
    p.write_text("#!/bin/bash\n" + body + "\n", encoding="utf-8", newline="\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


def _tree(tmp_path: Path, *, earlier_run: bool) -> Path:
    work = tmp_path / "repo"
    (work / "stage1" / "data").mkdir(parents=True)
    shutil.copy(SCRIPT, work / SCRIPT.name)
    if earlier_run:
        for rel, body in FILES.items():
            p = work / "stage1" / "data" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(body)
    return work


def _run(tmp_path: Path, work: Path, wget: str, python3_rc: int = 0):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _fake(bindir, "wget", wget)
    _fake(bindir, "python3", f"echo 'the linker check says: count drifted'; exit {python3_rc}")
    env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ["PATH"])
    r = subprocess.run([BASH, SCRIPT.name], cwd=work, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.returncode, r.stdout + r.stderr


def _unchanged(work: Path) -> None:
    for rel, body in FILES.items():
        assert (work / "stage1" / "data" / rel).read_bytes() == body, f"{rel} was overwritten"
    assert not list((work / "stage1" / "data").rglob("*.part")), "a partial download was left"


def test_a_failed_download_keeps_what_an_earlier_run_left(tmp_path):
    """The case that happened: wget empties its target, then fails."""
    work = _tree(tmp_path, earlier_run=True)
    rc, out = _run(tmp_path, work, WGET_EMPTIES_THEN_FAILS)
    assert rc == 0, out
    _unchanged(work)
    assert "any copy from an earlier run is kept" in out


def test_an_empty_download_is_not_a_download(tmp_path):
    work = _tree(tmp_path, earlier_run=True)
    rc, out = _run(tmp_path, work, WGET_WRITES_NOTHING)
    assert rc == 0, out
    _unchanged(work)


def test_a_good_download_replaces_the_old_file_and_a_page_that_is_not_one_does_not(tmp_path):
    """An .xlsx is a zip, so it gets the zip check: a login page answered with 200 is refused."""
    import zipfile
    good = tmp_path / "fresh.zip"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("fresh.txt", "fresh")
    work = _tree(tmp_path, earlier_run=True)
    copy = f'while [ $# -gt 0 ]; do [ "$1" = -O ] && cp "{good.as_posix()}" "$2"; shift; done'
    rc, out = _run(tmp_path, work, copy)
    assert rc == 0, out
    assert (work / "stage1" / "data" / "liu_wu_yields.xlsx").read_bytes() == good.read_bytes()
    (tmp_path / "bin").rename(tmp_path / "bin_first")
    work2 = _tree(tmp_path / "second", earlier_run=True)
    rc, out = _run(tmp_path, work2, 'while [ $# -gt 0 ]; do [ "$1" = -O ] && '
                                    'echo "<html>sign in</html>" > "$2"; shift; done; exit 0')
    assert rc == 0, out
    _unchanged(work2)


def test_nothing_downloaded_and_nothing_there_stops_the_run(tmp_path):
    work = _tree(tmp_path, earlier_run=False)
    rc, out = _run(tmp_path, work, WGET_EMPTIES_THEN_FAILS)
    assert rc == 1, out
    assert "5 required file(s) missing" in out


def test_a_linker_that_fails_its_own_check_stops_the_run_and_says_why(tmp_path):
    work = _tree(tmp_path, earlier_run=True)
    (work / "stage1" / "data" / "bond_firm_linker_2026" / "verify_release.py").write_text("")
    rc, out = _run(tmp_path, work, WGET_EMPTIES_THEN_FAILS, python3_rc=1)
    assert rc == 1, out
    assert "count drifted" in out and "failed its own check" in out
