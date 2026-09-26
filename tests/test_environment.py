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
    """In an environment installed as documented. A missing numba or numexpr FAILS here, with
    the message a user would see: an environment that lacks them is not the documented one."""
    monkeypatch.setattr(pybondlab_pin, "installed_version", lambda: pybondlab_pin.VERSION)
    msg = pybondlab_pin.check()
    assert msg is None, msg


def test_pandas_computes_as_the_published_build_did():
    """With numexpr, pandas compares a float32 column with a Python number in float64; numpy
    alone compares in float32, so a value stored as float32(-0.2) is below -0.2 one way and not
    the other. The published panels were built with numexpr, so it is required and used."""
    import importlib.util
    assert importlib.util.find_spec("numexpr") is not None,         "numexpr is required: python -m pip install -r requirements-local.txt"
    numeric_setup.apply()
    assert pd.get_option("compute.use_numexpr") is True
    assert pd.get_option("compute.use_bottleneck") is False
    # pandas hands arrays over 1,000,000 elements to numexpr; smaller ones never reach it
    s = pd.Series(np.full(2_000_000, -0.2, dtype="float32"))
    assert int((s < -0.2).sum()) == 2_000_000
