"""numeric_setup.py -- make pandas compute the same way on every machine.

pandas hands large arithmetic and comparisons (over a million elements) to `numexpr` when it is
installed, and some sums to `bottleneck`, and neither computes exactly as numpy does. numexpr
evaluates float32 columns against a Python number in float64 where numpy stays in float32, and
Stage 2's illiquidity measures do their arithmetic in pandas on float32 daily data. Measured on
the 2026-09-21 run, the same code and inputs gave, with numexpr and without:

    ilq  up to 5.1e-4 apart     pi  4.1e-5     roll  1.5e-5
    Table IA.V: 6,186 returns below -20% against 6,181

The published panels were built with numexpr, so it is REQUIRED (`requirements-local.txt`
installs it, and the start-up check in `pybondlab_pin.py` refuses to run without it), and pandas
is told to use it. bottleneck was not installed in the build that produced the published files,
so pandas is told not to use it. Stages 2, 3 and 4 call `apply()` when their settings load.
"""
from __future__ import annotations


def apply() -> None:
    import pandas as pd
    pd.set_option("compute.use_numexpr", True)
    pd.set_option("compute.use_bottleneck", False)


apply()
