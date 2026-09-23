# -*- coding: utf-8 -*-
"""
test_worker_override.py
=======================
STAGE0_WORKERS sets how many chunks every stage0 member fetches at once. Until 2026-09-24
only the runners read it: the connection check and the qsub request still used
CONCURRENCY, so STAGE0_WORKERS=4 put 8 connections against a ceiling of 7 and passed.

No WRDS needed.

Author: Open Source Bond Asset Pricing
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage0"))


def _settings(monkeypatch, override):
    if override is None:
        monkeypatch.delenv("STAGE0_WORKERS", raising=False)
    else:
        monkeypatch.setenv("STAGE0_WORKERS", str(override))
    import _trace_settings
    return importlib.reload(_trace_settings)


def test_without_the_override_nothing_changes(monkeypatch):
    s = _settings(monkeypatch, None)
    assert {m: s.workers(m) for m in s.CONCURRENCY} == s.CONCURRENCY
    s.validate_connection_budget(["enhanced", "144a"])
    assert s.qsub_resources("enhanced").startswith(f"-pe onenode {s.CONCURRENCY['enhanced']} ")


def test_the_override_reaches_the_runners_the_check_and_the_request(monkeypatch):
    s = _settings(monkeypatch, 3)
    assert s.PER_DATASET["enhanced"]["n_workers"] == 3 == s.PER_DATASET["144a"]["n_workers"]
    s.validate_connection_budget(["enhanced", "144a"])          # 3 + 3 = 6: within the cap
    assert s.qsub_resources("144a").startswith("-pe onenode 3 ")


def test_an_override_over_the_connection_ceiling_is_refused(monkeypatch):
    s = _settings(monkeypatch, 4)                               # 4 + 4 = 8 > 7 - 1
    with pytest.raises(ValueError, match="connection budget"):
        s.validate_connection_budget(["enhanced", "144a"])
    s = _settings(monkeypatch, None)                            # leave the module as found
