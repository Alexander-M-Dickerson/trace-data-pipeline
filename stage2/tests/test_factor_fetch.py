"""test_factor_fetch.py -- the fetch-once framework behind the public factor build (W3, offline).

Covers lib/factor_fetch._cached: cache hit/miss/force semantics, lazy URL resolution (rotating
vintages), the sidecar meta fingerprint, and the date-seam checks (null dates fail loudly; duplicate
months pass verbatim -- the panel-level dedup resolves them, matching upstream)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

import _stage2_settings as cfg
from lib import factor_fetch as ffetch


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "FACTOR_CACHE_DIR", tmp_path)
    return tmp_path


def _frame(dates=("2001-01-31", "2001-02-28")) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(list(dates)), "x": range(len(dates))})


def test_cached_miss_builds_writes_cache_and_meta(cache_dir):
    calls = []

    def build(url):
        calls.append(url)
        return _frame()

    out = ffetch._cached("t1", "http://example/x.zip", build, force=False)
    assert calls == ["http://example/x.zip"]
    assert (cache_dir / "t1.parquet").exists()
    meta = json.loads((cache_dir / "t1.meta.json").read_text())
    assert meta["url"] == "http://example/x.zip" and meta["rows"] == 2
    assert meta["span"] == ["2001-01-31", "2001-02-28"] and meta["sha256"]
    pd.testing.assert_frame_equal(out, _frame())


def test_cached_hit_skips_builder_and_force_rebuilds(cache_dir):
    calls = []

    def build(url):
        calls.append(url)
        return _frame()

    ffetch._cached("t2", "u", build, force=False)
    hit = ffetch._cached("t2", "u", build, force=False)
    assert len(calls) == 1                       # cache hit: builder not called
    pd.testing.assert_frame_equal(hit, _frame())
    ffetch._cached("t2", "u", build, force=True)
    assert len(calls) == 2                       # force: re-fetched


def test_cached_resolves_callable_url_only_on_miss(cache_dir):
    resolved = []

    def lazy_url():
        resolved.append(1)
        return "http://discovered/current.zip"

    ffetch._cached("t3", lazy_url, lambda url: _frame(), force=False)
    ffetch._cached("t3", lazy_url, lambda url: _frame(), force=False)
    assert len(resolved) == 1                    # discovery ran once, not on the cache hit
    meta = json.loads((cache_dir / "t3.meta.json").read_text())
    assert meta["url"] == "http://discovered/current.zip"


def test_cached_rejects_null_dates_but_keeps_duplicates(cache_dir, capsys):
    bad = _frame()
    bad.loc[1, "date"] = pd.NaT
    with pytest.raises(AssertionError, match="non-null"):
        ffetch._cached("t4", "u", lambda url: bad, force=False)

    dup = _frame(("2025-01-31", "2025-01-31"))   # the 2026 HKM vintage ships 2025-01 twice
    out = ffetch._cached("t5", "u", lambda url: dup, force=False)
    assert len(out) == 2                         # kept verbatim; panel dedup keep-first resolves
    assert "duplicate month" in capsys.readouterr().out
