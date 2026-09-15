"""The 1997-2002 quote panel: its column contract, and the stale-cache guard.

`lib/quote.py` caches its download, so repointing QUOTE_URL at a new file changes nothing on a
machine that already holds the old one -- the build keeps running on stale data and every number
it produces looks reasonable. That is the failure these tests exist to prevent.

The 2026 file widened the panel from 9 columns to 14 by adding the five Treasury benchmarks.
Measured at the time: steps 3, 4 and 5 all consume byte-identical frames either way, because each
one selects its columns by name and then sorts by (cusip, date). `test_widening_cannot_reach_the_
steps` pins the property that makes that true, so a future widening cannot quietly stop being safe.
"""
import pandas as pd
import pytest

import _stage2_settings as cfg
from lib import quote


def _frame(cols):
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})


# --------------------------------------------------------------- the declared contract
def test_settings_declare_a_column_contract():
    """QUOTE_URL alone is not a contract -- the columns it must deliver are."""
    assert cfg.QUOTE_REQUIRED_COLS, "QUOTE_REQUIRED_COLS must not be empty"
    for c in ("cusip_id", "date", "ret_vw", "tret", "cs", "bbtm"):
        assert c in cfg.QUOTE_REQUIRED_COLS, f"{c} is read by a step and must be required"
    if getattr(cfg, "QUOTE_HAS_BENCHMARKS", False):
        assert set(cfg.QUOTE_BENCHMARK_COLS) == {
            "tret_bns", "tret_cfm", "tret_gprs", "tret_cls", "tret_mat"}


def test_the_url_and_the_benchmark_flag_agree():
    """A URL naming the extended file while the flag says otherwise is a configuration error."""
    extended = "_tret" in cfg.QUOTE_URL
    assert extended == bool(getattr(cfg, "QUOTE_HAS_BENCHMARKS", False)), (
        f"QUOTE_URL={cfg.QUOTE_URL} and QUOTE_HAS_BENCHMARKS="
        f"{getattr(cfg, 'QUOTE_HAS_BENCHMARKS', False)} disagree")


# --------------------------------------------------------------- the guard itself
def test_missing_flags_a_nine_column_cache():
    """The exact stale-cache shape: the old published file, under the new settings."""
    old = _frame(["cusip_id", "date", "permno", "ret_vw", "return_type", "cs", "tret",
                  "bbtm", "sze"])
    gone = quote._missing(old)
    if getattr(cfg, "QUOTE_HAS_BENCHMARKS", False):
        assert set(gone) == set(cfg.QUOTE_BENCHMARK_COLS), (
            "a nine-column cache must be rejected, not silently used")
    else:
        assert gone == []


def test_missing_accepts_the_extended_file():
    new = _frame(list(cfg.QUOTE_REQUIRED_COLS) + list(cfg.QUOTE_BENCHMARK_COLS)
                 + ["permno", "return_type"])
    assert quote._missing(new) == []


def test_missing_flags_a_dropped_required_column():
    """Widening is safe; NARROWING is what breaks the rolling windows."""
    short = _frame([c for c in cfg.QUOTE_REQUIRED_COLS if c != "tret"]
                   + list(cfg.QUOTE_BENCHMARK_COLS))
    assert "tret" in quote._missing(short)


def test_a_stale_cache_is_not_returned(tmp_path, monkeypatch):
    """End to end, offline: a cache missing the benchmarks must trigger a refetch, not be used."""
    stale = tmp_path / cfg.QUOTE_ZIPKEY
    pd.DataFrame({c: [0.0] for c in
                  ["cusip_id", "date", "permno", "ret_vw", "return_type", "cs", "tret",
                   "bbtm", "sze"]}).to_parquet(stale, index=False)
    monkeypatch.setattr(quote, "QUOTE_CACHE", stale)
    if not getattr(cfg, "QUOTE_HAS_BENCHMARKS", False):
        pytest.skip("nine-column panel configured; nothing to be stale against")

    called = {"n": 0}

    def _fake_fetch():
        called["n"] += 1
        return pd.DataFrame({c: [0.0] for c in
                             list(cfg.QUOTE_REQUIRED_COLS) + list(cfg.QUOTE_BENCHMARK_COLS)})

    monkeypatch.setattr(quote, "_fetch", _fake_fetch)
    monkeypatch.setattr(cfg, "ensure_dirs", lambda: None)
    out = quote.load_quote()
    assert called["n"] == 1, "the stale cache was used instead of refetching"
    assert quote._missing(out) == []


def test_a_download_that_disagrees_with_the_settings_raises(tmp_path, monkeypatch):
    """Silently building on the wrong file is worse than stopping."""
    if not getattr(cfg, "QUOTE_HAS_BENCHMARKS", False):
        pytest.skip("nine-column panel configured")
    monkeypatch.setattr(quote, "QUOTE_CACHE", tmp_path / "absent.parquet")
    monkeypatch.setattr(quote, "_fetch",
                        lambda: pd.DataFrame({c: [0.0] for c in cfg.QUOTE_REQUIRED_COLS}))
    with pytest.raises(RuntimeError, match="missing"):
        quote.load_quote()


# --------------------------------------------------------------- why widening is safe
def test_widening_cannot_reach_the_steps():
    """Steps 3/4/5 select quote columns BY NAME and then sort by (cusip, date).

    Those two properties together are what make a wider file a no-op. If a step ever starts
    consuming the frame wholesale, or stops sorting, this test is the thing that should have
    caught it -- so it reads the step sources rather than trusting a comment.
    """
    from pathlib import Path
    steps = Path(__file__).resolve().parents[1] / "steps"
    for name in ("step3_bbw.py", "step4_betas.py", "step5_value.py"):
        src = (steps / name).read_text(encoding="utf-8")
        assert "load_quote()" in src, f"{name} no longer loads the quote panel -- retire this test"
        assert 'sort_values(["cusip", "date"])' in src, (
            f"{name} stopped sorting by (cusip, date); a wider or reordered quote panel is no "
            f"longer provably a no-op there")


@pytest.mark.skipif(not (cfg.DATA_DIR / cfg.QUOTE_ZIPKEY).exists(),
                    reason="quote panel not cached in this clone")
def test_the_cached_file_satisfies_the_contract():
    df = pd.read_parquet(cfg.DATA_DIR / cfg.QUOTE_ZIPKEY)
    assert quote._missing(df) == []
    assert not df.duplicated(["cusip_id", "date"]).any(), (
        "duplicate keys would make drop_duplicates(keep='last') order-dependent")
