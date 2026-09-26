"""The two root modules that make a run independent of the machine it is on.

    python -m pytest tests/test_environment.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numeric_setup  # noqa: E402
import pybondlab_pin  # noqa: E402


def test_a_wrong_or_missing_pybondlab_gets_the_two_install_lines(monkeypatch):
    for got in (None, "0.2.0"):
        monkeypatch.setattr(pybondlab_pin, "installed_version", lambda got=got: got)
        msg = pybondlab_pin.check()
        assert msg and "-r requirements-local.txt" in msg
        assert f"--no-deps pybondlab=={pybondlab_pin.VERSION}" in msg


def test_the_pinned_pybondlab_passes(monkeypatch):
    monkeypatch.setattr(pybondlab_pin, "installed_version", lambda: pybondlab_pin.VERSION)
    import importlib.util
    if importlib.util.find_spec("numba") is None:
        return                     # the numba branch is the message's job, tested above
    assert pybondlab_pin.check() is None


def test_pandas_does_not_hand_work_to_optional_packages():
    """numexpr compares float32 in float64 and numpy in float32; a value stored as
    float32(-0.2) is below -0.2 one way and not the other. With the option off the answer is
    numpy's, whatever is installed."""
    numeric_setup.apply()
    assert pd.get_option("compute.use_numexpr") is False
    assert pd.get_option("compute.use_bottleneck") is False
    # pandas hands arrays over 1,000,000 elements to numexpr; smaller ones never reach it
    s = pd.Series(np.full(2_000_000, -0.2, dtype="float32"))
    assert int((s < -0.2).sum()) == int((s.to_numpy() < -0.2).sum())
