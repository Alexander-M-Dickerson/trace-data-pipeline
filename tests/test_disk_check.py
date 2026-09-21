# -*- coding: utf-8 -*-
"""
test_disk_check.py
==================
`check_disk_space.sh` decides whether `run_pipeline.sh` starts at all.

It used to read the space USED from the WRDS `quota` command. WRDS refreshes that figure every
30 minutes, and the usual way to start a run is to delete the last one first, so for half an
hour `quota` still counted the deleted folder. On 2026-09-21 it refused a run with 7.7 GB free
by reporting 2.6. The check now takes the LIMIT from `quota` and MEASURES the use with `du`.

The tests run the real script under bash with a fake `quota` and a fake `du` first on PATH.

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
SCRIPT = ROOT / "check_disk_space.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash is not available")

GB = 1024 * 1024          # KB in a GB, the unit `du -sk` prints


def _fake(bindir: Path, name: str, body: str) -> None:
    p = bindir / name
    p.write_text("#!/bin/bash\n" + body + "\n", encoding="utf-8", newline="\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


def _run(tmp_path: Path, *, quota_used: str | None, du_kb: int | None, env=None, in_home=True):
    home = tmp_path / "home"
    work = (home / "trace-data-pipeline") if in_home else (tmp_path / "elsewhere")
    work.mkdir(parents=True)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    if quota_used is not None:
        _fake(bindir, "quota",
              'echo "DIRECTORY  USED / LIMIT"\n'
              f'echo "    Home:  {quota_used}GB / 10GB"\n'
              'echo " Scratch:  6.77GB / 500GB"')
    if du_kb is None:
        _fake(bindir, "du", "exit 1")
    else:
        _fake(bindir, "du", f'echo "{du_kb}\t$2"')
    _fake(bindir, "timeout", 'shift; "$@"')
    e = dict(os.environ, HOME=str(home), PATH=str(bindir) + os.pathsep + os.environ["PATH"])
    e.pop("FORCE_RUN", None)
    e.update(env or {})
    r = subprocess.run([BASH, str(SCRIPT)], cwd=work, env=e, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.returncode, r.stdout + r.stderr


def test_the_case_that_happened(tmp_path):
    """quota says 7.36 used (stale), the disk holds 2.3. The run must START."""
    rc, out = _run(tmp_path, quota_used="7.36", du_kb=int(2.3 * GB))
    assert rc == 0, out
    assert "Available: 7.70 GB" in out
    assert "refreshes that figure every 30 minutes" in out          # and it says why they differ


def test_a_full_home_directory_is_still_refused(tmp_path):
    """The other direction: quota looks fine, the disk is nearly full. The run must NOT start."""
    rc, out = _run(tmp_path, quota_used="1.00", du_kb=int(8.5 * GB))
    assert rc == 1, out
    assert "NOT ENOUGH DISK SPACE" in out and "Available: 1.50 GB" in out
    assert "FORCE_RUN=1 ./run_pipeline.sh" in out                   # it says what to do


def test_force_run_reports_and_never_refuses(tmp_path):
    rc, out = _run(tmp_path, quota_used="9.00", du_kb=int(9.0 * GB), env={"FORCE_RUN": "1"})
    assert rc == 0 and "NOT ENOUGH DISK SPACE" in out and "FORCE_RUN=1 is set" in out


def test_when_du_cannot_measure_it_falls_back_to_quota_and_says_so(tmp_path):
    rc, out = _run(tmp_path, quota_used="3.00", du_kb=None)
    assert rc == 0, out
    assert "Could not measure" in out and "can be 30 minutes old" in out
    assert "Available: 7.00 GB" in out


def test_outside_the_home_directory_the_home_quota_is_not_the_limit(tmp_path):
    """A run kept on scratch is not limited by the 10 GB home quota."""
    rc, out = _run(tmp_path, quota_used="9.90", du_kb=int(9.9 * GB), in_home=False,
                   env={"DISK_CHECK_NEED_GB": "0"})
    assert rc == 0, out
    assert "free space on this filesystem" in out and "Home quota" not in out


def test_without_a_quota_command_it_uses_the_filesystem(tmp_path):
    rc, out = _run(tmp_path, quota_used=None, du_kb=int(1 * GB), env={"DISK_CHECK_NEED_GB": "0"})
    assert rc == 0 and "free space on this filesystem" in out


def test_run_pipeline_calls_it_and_has_no_quota_parsing_of_its_own():
    src = (ROOT / "run_pipeline.sh").read_text(encoding="utf-8")
    assert "bash check_disk_space.sh || exit 1" in src
    assert "QUOTA_LINE" not in src, "run_pipeline.sh parses `quota` itself again"
