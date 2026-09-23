# -*- coding: utf-8 -*-
"""
test_wrds_credentials.py
========================
Where Stage 2 gets its WRDS username, and when it asks for one.

Until 2026-09-23 the five WRDS fetchers read the name from the environment only, while their
own error said "Set it in config.py"; and the start-up check looked for caches in
data/factor_cache/, where none of the WRDS caches live, and accepted config.py's placeholder
name, so it never fired. A user who set the name in config.py passed the check and then
failed at the first fetch.

Author: Open Source Bond Asset Pricing
"""
from __future__ import annotations

import sys
from pathlib import Path

STAGE2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STAGE2))

import _stage2_settings as cfg  # noqa: E402

FETCHERS = ["lib/treasury.py", "lib/ff5.py", "lib/vix.py", "lib/factor_fetch.py",
            "lib/duration_adjusted.py"]


def test_the_environment_wins_then_config_py(monkeypatch):
    monkeypatch.setattr(cfg, "WRDS_USERNAME", "from_config")
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    assert cfg.wrds_username() == "from_config"
    monkeypatch.setenv("WRDS_USERNAME", "from_env")
    assert cfg.wrds_username() == "from_env"


def test_the_config_py_placeholder_counts_as_unset(monkeypatch):
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    for placeholder in ("your_wrds_username", "your_wrds_id", "", "  "):
        monkeypatch.setattr(cfg, "WRDS_USERNAME", placeholder)
        assert cfg.wrds_username() == "", placeholder


def test_every_fetcher_asks_the_settings_not_the_environment():
    for f in FETCHERS:
        src = (STAGE2 / f).read_text(encoding="utf-8")
        assert "cfg.wrds_username()" in src, f
        assert 'os.environ.get("WRDS_USERNAME"' not in src, f


def _config(source="pinned", user=""):
    return {"factor_source": source, "wrds_username": user}


def test_missing_caches_without_a_username_is_a_problem(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "STAGE2_DATA", tmp_path)
    monkeypatch.setattr(cfg, "FACTOR_CACHE_DIR", tmp_path / "factor_cache")
    msg = cfg.wrds_credentials_problem(_config())
    assert msg and all(f in msg for f in cfg.WRDS_CACHES)
    assert cfg.wrds_credentials_problem(_config(user="someone")) is None


def test_a_full_cache_needs_no_username(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "STAGE2_DATA", tmp_path)
    monkeypatch.setattr(cfg, "FACTOR_CACHE_DIR", tmp_path / "factor_cache")
    for f in cfg.WRDS_CACHES:
        (tmp_path / f).write_bytes(b"")
    assert cfg.wrds_credentials_problem(_config("pinned")) is None
    # the public factor source also fetches monthly VIX
    assert "vix_monthly.parquet" in cfg.wrds_credentials_problem(_config("public"))
    (tmp_path / "factor_cache").mkdir()
    (tmp_path / "factor_cache" / "vix_monthly.parquet").write_bytes(b"")
    assert cfg.wrds_credentials_problem(_config("public")) is None
