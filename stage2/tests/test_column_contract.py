# -*- coding: utf-8 -*-
"""
test_column_contract.py
=======================
The panel's column list is documented in two other places, and all three must agree.

    the panel          main_panel_<mode>.parquet -- what a user actually downloads
    the report         _report_helpers.get_signal_definitions() -- the appendix table
    the dictionary     DATA_DICTIONARY.md -- the reference document

Two sources of truth in agreement is a drift waiting to happen: nothing makes them
move together, so the first time a column is added, renamed or dropped, two of the
three silently disagree and the published documentation starts lying. This test is
what makes them move together.

It is not hypothetical. The report requested `mod_dur`, `conv` and `pr` while the
panel carried `md_dur`, `convx` and no price at all; the statistics helper skipped a
name it could not find, so thirty rows vanished from the shipped report with no error
anywhere. That class of defect is what this file exists to catch.

The dictionary is deliberately BROADER than the panel: it also documents the exogenous
factor series (`mktb`, `defb`, `epu`, ...), which live in factors_merged.parquet rather
than the panel. So the contract is:

    report definitions  ==  panel columns          (exactly)
    dictionary          >=  panel columns          (everything shipped is documented)
    dictionary - panel  ⊆   factors_merged columns (nothing documented is invented)

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

STAGE2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STAGE2))

import _stage2_settings as cfg  # noqa: E402

# Factor series carried by factors_merged that DATA_DICTIONARY.md does not yet define.
# Documenting them is an open task; the test below asserts the list has not grown, so a
# NEW undocumented factor fails immediately while the existing backlog stays visible.
UNDOCUMENTED_FACTORS = {
    "ars", "cptl", "crfx", "css", "drfx", "dunc3", "dunc6", "duncf", "duncr",
    "fhts", "lrfx", "rf", "sprd", "uncf", "uncr",
}


def _panel_path() -> Path | None:
    """The newest built panel, whichever mode it was built under."""
    hits = sorted(cfg.PANEL_DIR.glob("main_panel_*.parquet"),
                  key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def _blocks_for(panel: Path) -> Path:
    return cfg.BLOCKS_DIR / panel.stem.replace("main_panel_", "")


def _columns(path: Path) -> set[str]:
    import pyarrow.parquet as pq
    return set(pq.ParquetFile(path).schema.names)


def _dictionary_mnemonics() -> set[str]:
    """Every `mnemonic` in the leading column of a DATA_DICTIONARY.md table row."""
    txt = (STAGE2 / "DATA_DICTIONARY.md").read_text(encoding="utf-8")
    return set(re.findall(r"^\|\s*`([A-Za-z0-9_]+)`\s*\|", txt, flags=re.M))


def _report_mnemonics() -> list[str]:
    import _report_helpers as rpt
    return [s["mnemonic"] for p in rpt.get_signal_definitions() for s in p["signals"]]


def test_report_definitions_have_no_duplicates():
    """A mnemonic defined in two panels would print twice and inflate the count."""
    seen = _report_mnemonics()
    dupes = sorted({m for m in seen if seen.count(m) > 1})
    assert not dupes, f"mnemonic(s) defined more than once in the report: {dupes}"


def test_report_definitions_match_the_panel_exactly():
    """The appendix table must describe the shipped panel -- every column, only columns."""
    panel = _panel_path()
    if panel is None:
        pytest.skip("no built panel under output/panel/; run a build first")
    cols = _columns(panel)
    report = set(_report_mnemonics())
    undocumented = sorted(cols - report)
    invented = sorted(report - cols)
    assert not undocumented, (
        f"{len(undocumented)} panel column(s) have no definition in the report "
        f"appendix: {undocumented}\n"
        f"  Add them to _report_helpers.get_signal_definitions().")
    assert not invented, (
        f"{len(invented)} mnemonic(s) are defined in the report but are not columns "
        f"of {panel.name}: {invented}\n"
        f"  A renamed column is the usual cause -- the statistics helper skips a name "
        f"it cannot find, so the row disappears from every table silently.")


def test_every_panel_column_is_in_the_data_dictionary():
    panel = _panel_path()
    if panel is None:
        pytest.skip("no built panel under output/panel/; run a build first")
    missing = sorted(_columns(panel) - _dictionary_mnemonics())
    assert not missing, (
        f"{len(missing)} panel column(s) are absent from DATA_DICTIONARY.md: {missing}")


def test_the_dictionary_documents_nothing_that_does_not_exist():
    """Dictionary entries beyond the panel must be real factor series, not inventions."""
    panel = _panel_path()
    if panel is None:
        pytest.skip("no built panel under output/panel/; run a build first")
    fm = _blocks_for(panel) / "factors_merged.parquet"
    if not fm.exists():
        pytest.skip(f"no {fm.name} for this build")
    extra = _dictionary_mnemonics() - _columns(panel)
    orphans = sorted(extra - _columns(fm))
    assert not orphans, (
        f"{len(orphans)} DATA_DICTIONARY.md entr(ies) match neither a panel column nor "
        f"a factor series: {orphans}")


def test_the_undocumented_factor_backlog_has_not_grown():
    """A NEW undocumented factor series fails here; the known backlog stays visible."""
    panel = _panel_path()
    if panel is None:
        pytest.skip("no built panel under output/panel/; run a build first")
    fm = _blocks_for(panel) / "factors_merged.parquet"
    if not fm.exists():
        pytest.skip(f"no {fm.name} for this build")
    undocumented = _columns(fm) - _dictionary_mnemonics() - {"date"}
    new = sorted(undocumented - UNDOCUMENTED_FACTORS)
    assert not new, (
        f"{len(new)} factor series added without a DATA_DICTIONARY.md entry: {new}\n"
        f"  Document them, or add them to UNDOCUMENTED_FACTORS with a reason.")
    fixed = sorted(UNDOCUMENTED_FACTORS - undocumented)
    assert not fixed, (
        f"{len(fixed)} factor series are now documented: {fixed}\n"
        f"  Remove them from UNDOCUMENTED_FACTORS so the list keeps shrinking.")
