"""The code tags: the repository passes, and every rule fires when it is broken.

    python -m pytest tests/test_tags.py -q

The planted lines below are spelled in two pieces ("[" + "tag:") so that tagref, which reads
this file too, does not take them for real tags.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import tags  # noqa: E402

TAG, GROUP, REF, FILE = ("[" + k + ":" for k in ("tag", "group", "ref", "file"))


def test_the_repository_passes_and_its_index_is_current():
    ds = tags.read_directives()
    assert tags.problems(ds) == []
    assert (ROOT / tags.INDEX).read_text(encoding="utf-8") == tags.render(ds), \
        "TAGS.md is out of date: run `python tools/tags.py`"


def _problems(lines, path="tools/tags.py", panel=("a",), daily=()):
    return tags.problems(tags.parse_text(lines, path), ROOT, panel=panel, daily=daily)


GOOD = [f"x = 1  # {TAG}col.a] the column a", f"# relies on {REF}col.a]"]


def test_a_well_formed_set_passes():
    assert _problems(GOOD) == []


@pytest.mark.parametrize("extra, panel, expect", [
    ([f"# {REF}col.b]"], ("a",), "names no tag"),
    ([f"y = 2  # {TAG}col.a] again"], ("a",), "declared 2 times"),
    ([f"# {GROUP}rule.one] alone"], ("a",), "has one member"),
    ([f"# {GROUP}col.a] a"], ("a",), "both a tag and a group"),
    ([], ("a", "b"), "have no tag: b"),
    ([f"# {TAG}col.z] a column nobody has"], ("a",), "does not have: z"),
    ([f"# {TAG}misc.thing] no such namespace"], ("a",), "no known namespace"),
    ([f"# {TAG}rule.bare]"], ("a",), "no description"),
    ([f"# see {FILE}no/such/file.txt]"], ("a",), "does not exist"),
    ([f's = "{TAG}rule.in_string] in a string"'], ("a",), "outside a `#` comment"),
])
def test_each_rule_fires(extra, panel, expect):
    got = _problems(GOOD + extra, panel=panel)
    assert any(expect in p for p in got), got


def test_a_tag_in_a_docstring_is_refused_but_not_in_a_doc():
    lines = ['"""A module.', f"{TAG}rule.x] described here", '"""']
    assert any("outside a `#` comment" in p for p in _problems(lines + GOOD))
    assert _problems(lines + GOOD, path="README.md") == []


def test_the_pattern_is_tagrefs():
    """Case-insensitive, with spaces allowed inside the brackets and trimmed from the label."""
    (d,) = tags.parse_text(["# [ " + "TAG : col.a ] the column a"], "tools/tags.py")
    assert (d.kind, d.label) == ("tag", "col.a")


def test_file_pointers_resolve_as_tagref_does():
    """From the repository root, unless the pointer starts ./ or ../ (then from the file)."""
    assert _problems(GOOD + [f"# {FILE}README.md]"], path="stage2/_run_stage2.py") == []
    assert _problems(GOOD + [f"# {FILE}../README.md]"], path="stage2/_run_stage2.py") == []
    assert _problems(GOOD + [f"# {FILE}./README.md]"], path="stage2/_run_stage2.py")


def test_a_stale_index_fails_the_check(monkeypatch):
    monkeypatch.setattr(tags, "render", lambda ds: "not the index\n")
    assert tags.main(["--check"]) == 1


def test_the_column_lists_are_the_panels():
    """The lists the column rule checks against, read from the files that define them."""
    assert len(tags.panel_columns()) == 145
    assert len(tags.daily_columns()) == 44
