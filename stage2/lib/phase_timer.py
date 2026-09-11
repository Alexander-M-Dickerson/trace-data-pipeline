"""phase_timer.py -- lightweight per-phase wall-clock accounting for the step modules.

Usage:
    pt = PhaseTimer()
    with pt("pandas_ports"):
        ...
    meta["phases"] = pt.phases          # {"pandas_ports": 12.34, ...} in call order

Phases land in each step's meta JSON so every build leaves a profile behind -- the speed_up/ pass
(and any future regression hunt) reads them instead of re-instrumenting.
"""
from __future__ import annotations

import time
from contextlib import contextmanager


class PhaseTimer:
    """Accumulates named wall-clock phases in call order; re-entering a name accumulates."""

    def __init__(self) -> None:
        self.phases: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] = round(self.phases.get(name, 0.0) + time.perf_counter() - t0, 3)
