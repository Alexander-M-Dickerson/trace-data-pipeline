"""doctor.py: what it reads, and the next step it names in each state.

    python -m pytest tests/test_doctor.py -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import doctor  # noqa: E402
from doctor import Check  # noqa: E402


def test_the_python_range_is_the_one_requirements_local_states():
    text = (ROOT / "requirements-local.txt").read_text(encoding="utf-8")
    m = re.search(r"Python (\d+)\.(\d+) to (\d+)\.(\d+)", text)
    assert m, "requirements-local.txt no longer states the Python range"
    lo, hi = (int(m[1]), int(m[2])), (int(m[3]), int(m[4]))
    assert doctor.PYTHON_RANGE == (lo, hi)


def test_it_reads_the_requirements_through_their_include():
    names = doctor.requirement_names(ROOT / "requirements-local.txt")
    assert {"pandas", "duckdb", "wrds", "numexpr", "pytest"} <= set(names)
    assert len(names) == len(set(names))


def test_a_requirement_whose_marker_does_not_hold_is_not_demanded(tmp_path):
    """requirements.txt installs numba only below Python 3.14, the WRDS Cloud's version."""
    req = tmp_path / "requirements.txt"
    req.write_text('pandas>=2.2\nnumba>=0.60; python_version<"3.0"\n', encoding="utf-8")
    assert doctor.requirement_names(req) == ["pandas"]
    assert doctor.marker_holds('python_version>="3.0"')


def test_it_never_connects_to_wrds_or_downloads():
    """The login-node rule: nothing here may open a WRDS connection."""
    src = (ROOT / "doctor.py").read_text(encoding="utf-8")
    for forbidden in ("import wrds", "wrds.Connection", "_wrds_pool", "urlopen", "requests"):
        assert forbidden not in src, forbidden


def _checks(**over):
    base = {"python": Check("Python", True), "packages": Check("packages", True),
            "pybondlab": Check("PyBondLab", True), "stage2": Check("stage 2", True),
            "stage3": Check("stage 3", True), "stage4": Check("stage 4", True)}
    base.update(over)
    return base


@pytest.mark.parametrize("over, panel, expect", [
    ({"python": Check("Python", False)}, False, "install Python 3.11"),
    ({"pybondlab": Check("PyBondLab", False)}, False, "--no-deps pybondlab=="),
    ({"stage2": Check("s2", False, "Stage 1 directory not found: x")}, False, "copy the stage0/"),
    ({"stage2": Check("s2", False, "WRDS_USERNAME is not set")}, False, "export WRDS_USERNAME"),
    ({}, False, "python _run_stage2.py"),
    ({}, True, "bash stage3/run_stage3.sh"),
    ({"stage4": Check("s4", False, "missing: betas_bns.parquet")}, True, "make_excess_blocks.py"),
    ({"stage3": Check("s3", False, "PANEL: not found")}, True, "stage 3's inputs fail"),
    ({"stage4": Check("s4", False, "PyBondLab 0.2.0")}, True, "stage 4's inputs fail"),
])
def test_the_next_step(monkeypatch, tmp_path, over, panel, expect):
    monkeypatch.setattr(doctor, "PANEL", tmp_path / ("panel.parquet" if panel else "absent"))
    monkeypatch.setattr(doctor, "EXHIBITS", tmp_path / "absent.pdf")
    monkeypatch.setattr(doctor, "FACTORS", tmp_path / "no_factors")
    if panel:
        doctor.PANEL.write_bytes(b"")
    assert expect in doctor.next_step_local(_checks(**over))


def test_a_failed_check_is_never_reported_as_everything_built(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor, "PANEL", tmp_path / "panel.parquet")
    monkeypatch.setattr(doctor, "EXHIBITS", tmp_path / "exhibits.pdf")
    monkeypatch.setattr(doctor, "FACTORS", tmp_path)
    for p in (doctor.PANEL, doctor.EXHIBITS, tmp_path / "single_sort_panel_trace_2026"):
        p.write_bytes(b"")
    assert "everything is built" in doctor.next_step_local(_checks())
    for bad in ("stage3", "stage4"):
        step = doctor.next_step_local(_checks(**{bad: Check(bad, False, "x")}))
        assert "everything is built" not in step, bad


def test_a_run_in_this_environment_exits_0_and_names_a_next_step(capsys):
    """In the environment the two install lines make, whatever data is present."""
    assert doctor.main([]) == 0
    assert "\nNext: " in capsys.readouterr().out
