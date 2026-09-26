"""numeric_setup.py -- make pandas compute the same way whatever else is installed.

pandas hands large comparisons and arithmetic to `numexpr`, and some sums to `bottleneck`,
when those OPTIONAL packages happen to be installed, and they do not always compute as numpy
does. numexpr compares a float32 column with a number like 0.2 in float64, where numpy compares
in float32: a return stored as float32(-0.2) is "below -20%" one way and not the other.
Measured on the 2026-09-21 panel: Table IA.V counted 6,186 returns below -20% with numexpr
installed and 6,181 without, from the same code and the same file.

`requirements-local.txt` installs neither package, but many Python distributions ship both.
Stages 2, 3 and 4 call `apply()` when their settings load, so the numbers do not depend on it.
"""
from __future__ import annotations


def apply() -> None:
    import pandas as pd
    pd.set_option("compute.use_numexpr", False)
    pd.set_option("compute.use_bottleneck", False)


apply()
