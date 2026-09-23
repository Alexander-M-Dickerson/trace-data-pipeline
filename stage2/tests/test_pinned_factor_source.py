# -*- coding: utf-8 -*-
"""
test_pinned_factor_source.py
============================
`--factor-source pinned` is the documented way to reproduce a published panel exactly
(QUICKSTART_stage2.md). Until 2026-09-23 the configuration check demanded FACTORS_PINNED_FILE,
so the command refused to start and the download of the published factor file in
steps/compute_factors.py could never be reached. These tests hold the check to the same
precedence the download uses: an explicit file, then a copy already downloaded, then the file
published for the vintage.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import _stage2_settings as cfg


def _config(**over):
    c = {"factors_pinned_file": None}
    c.update(over)
    return c


def test_published_vintage_needs_no_local_file(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "STAGE2_DATA", tmp_path)          # nothing downloaded yet
    monkeypatch.setattr(cfg, "release_vintage", lambda: "2026")
    monkeypatch.setattr(cfg, "FACTORS_PINNED_URL", {"2026": "https://example.org/f.zip"})
    origin, ok = cfg.pinned_factor_origin(_config())
    assert ok and "https://example.org/f.zip" in origin


def test_unpublished_vintage_is_refused_with_a_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "STAGE2_DATA", tmp_path)
    monkeypatch.setattr(cfg, "release_vintage", lambda: "2031")
    monkeypatch.setattr(cfg, "FACTORS_PINNED_URL", {"2026": "https://example.org/f.zip"})
    origin, ok = cfg.pinned_factor_origin(_config())
    assert not ok and "2031" in origin and "FACTORS_PINNED_FILE" in origin


def test_a_downloaded_copy_is_used(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "STAGE2_DATA", tmp_path)
    monkeypatch.setattr(cfg, "release_vintage", lambda: "2031")
    monkeypatch.setattr(cfg, "FACTORS_PINNED_URL", {})
    (tmp_path / cfg.FACTORS_PINNED_ZIPKEY.format(vintage="2031")).write_bytes(b"x")
    origin, ok = cfg.pinned_factor_origin(_config())
    assert ok and "downloaded earlier" in origin


def test_an_explicit_file_must_exist(tmp_path):
    missing = tmp_path / "nope.parquet"
    origin, ok = cfg.pinned_factor_origin(_config(factors_pinned_file=str(missing)))
    assert not ok and "not found" in origin
    present = tmp_path / "f.parquet"
    present.write_bytes(b"x")
    origin, ok = cfg.pinned_factor_origin(_config(factors_pinned_file=str(present)))
    assert ok


def test_a_copy_from_a_stale_url_is_fetched_again(monkeypatch, tmp_path):
    """A download is reused only if it came from the URL registered now. A copy from an older
    URL (on 2026-09-23, a WordPress upload that predated the _bns/_cls twins) is not."""
    monkeypatch.setattr(cfg, "STAGE2_DATA", tmp_path)
    monkeypatch.setattr(cfg, "release_vintage", lambda: "2026")
    monkeypatch.setattr(cfg, "FACTORS_PINNED_URL", {"2026": "https://example.org/new.zip"})
    cache = tmp_path / cfg.FACTORS_PINNED_ZIPKEY.format(vintage="2026")
    cache.write_bytes(b"x")
    (tmp_path / (cache.name + ".source")).write_text("https://example.org/old.zip\n", encoding="utf-8")
    origin, ok = cfg.pinned_factor_origin(_config())
    assert ok and "new.zip" in origin and "downloaded on first run" in origin
    (tmp_path / (cache.name + ".source")).write_text("https://example.org/new.zip\n", encoding="utf-8")
    origin, ok = cfg.pinned_factor_origin(_config())
    assert ok and "downloaded earlier" in origin
