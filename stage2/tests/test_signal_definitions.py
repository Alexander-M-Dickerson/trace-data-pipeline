# -*- coding: utf-8 -*-
"""
test_signal_definitions.py
==========================
The panel's 145 columns are defined in three places a reader can land on:

    stage3/spec/signal_definitions.json   THE source. Table IA.VIII is rendered from it.
    the Stage 2 data report, Table 7      now READS the spec (_report_helpers.get_signal_definitions)
    stage2/DATA_DICTIONARY.md             hand-written markdown, held to the spec by this file

Why this exists. test_column_contract.py already proved the three carried the same MNEMONICS.
It never compared a name or a description, and they drifted: on 2026-09-16 five names were
corrected in the spec and the dictionary (b_eput is TAX policy uncertainty, lix is stored
negated, ...) and the report's own copy was missed, so the published data report went on
printing "Trade Policy Uncertainty Beta". Separately, all three called `hprd` "calendar days"
for years, which it never was.

So: the report has no copy any more, and the dictionary must say what the spec says.

What is compared: every column's NAME and DESCRIPTION, after removing what legitimately
differs -- LaTeX markup against markdown, a cross-reference to a figure only one document
prints, and the dictionary's notes about which columns are redacted on publication.

What this does NOT check: that a definition is TRUE. That is settled against the code, one
row at a time, and recorded in stage3/RECONCILIATION_ia08.md.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

STAGE2 = Path(__file__).resolve().parent.parent
REPO = STAGE2.parent
SPEC = REPO / "stage3" / "spec" / "signal_definitions.json"
DICT = STAGE2 / "DATA_DICTIONARY.md"
sys.path.insert(0, str(STAGE2))


def _spec() -> dict[str, dict]:
    return {r["mnemonic"]: r for r in json.loads(SPEC.read_text(encoding="utf-8"))["rows"]}


def _dictionary() -> dict[str, dict]:
    """The first table row for each mnemonic inside the Signal Definitions section."""
    txt = DICT.read_text(encoding="utf-8")
    sec = txt[txt.index("## Signal Definitions"):txt.index("## Redaction")]
    out: dict[str, dict] = {}
    for m in re.finditer(r"^\|\s*`([A-Za-z0-9_]+)`\s*\|\s*([^|]*?)\s*\|\s*(.*?)\s*\|\s*$", sec, re.M):
        out.setdefault(m.group(1), {"name": m.group(2), "description": m.group(3)})
    return out


_FIGURE_REF = [
    r"\(see Fig[^)]*\)",
    r"[;.]?\s*[Rr]efer to Panel [A-D] of Fig[^.]*\.?(\s*\\ref\{[^}]*\}\.?)?",
]
_REDACTION_NOTE = r"\*\*(?:Null|Collapsed)[^*]*\*\*\s*--\s*see \*Redaction\* below\."
_SYMBOLS = {r"\Delta": "Δ", r"\times": "×", r"\pm": "±", "\u2212": "-", "\u2013": "-",
            "\u2014": "-", "``": '"', "''": '"', "\u201c": '"', "\u201d": '"', "{-}": "-"}


def norm(text: str) -> str:
    """Reduce a definition to the words it says, so LaTeX and markdown can be compared."""
    t = text or ""
    t = re.sub(_REDACTION_NOTE, "", t)
    for pat in _FIGURE_REF:
        t = re.sub(pat, "", t)
    t = re.sub(r"\\(?:texttt|text|mathrm)\{([^{}]*)\}", r"\1", t)
    for a, b in _SYMBOLS.items():
        t = t.replace(a, b)
    t = t.replace("--", "-")
    t = re.sub(r"\\([&%$_#])", r"\1", t)          # \& \% \$ \_
    t = t.replace("\u2757", "")                   # the dictionary's emphasis mark
    t = re.sub(r"[`*${}\\~]", "", t)
    t = re.sub(r"\s+([.,;])", r"\1", t)           # the space a removed figure reference leaves
    t = re.sub(r"\s+", " ", t).strip().rstrip(".").strip()
    return t.lower()


def test_norm_removes_only_what_may_differ():
    assert norm(r"Computed with \texttt{QuantLib}.") == norm("Computed with QuantLib.")
    assert norm(r"S\&P rating, 100\% (see Fig. \ref{fig:x}).") == norm("S&P rating, 100%")
    assert norm("Identifier. **Null in the published file** -- see *Redaction* below.") == "identifier"
    assert norm(r"band of $\pm$1 month") == norm("band of ±1 month")
    assert norm("uses the EARLIER month") != norm("uses the LATER month")     # words still count


def test_the_three_sources_carry_the_same_columns():
    from lib import contract
    spec, dic = _spec(), _dictionary()
    assert list(spec) == list(contract.PANEL_COLUMNS) or set(spec) == set(contract.PANEL_COLUMNS)
    assert set(spec) == set(contract.PANEL_COLUMNS), sorted(set(spec) ^ set(contract.PANEL_COLUMNS))
    assert not (set(contract.PANEL_COLUMNS) - set(dic)), sorted(set(contract.PANEL_COLUMNS) - set(dic))


def test_the_report_has_no_copy_of_its_own():
    """Table 7 of the data report is read from the spec, in the spec's order."""
    import _report_helpers as rpt
    got = [(s["mnemonic"], s["name"]) for p in rpt.get_signal_definitions() for s in p["signals"]]
    want = [(r["mnemonic"], rpt._tex_escape(r["name"])) for r in _spec().values()]
    assert got == want
    src = (STAGE2 / "_report_helpers.py").read_text(encoding="utf-8")
    assert "'mnemonic': 'cusip'" not in src and '"mnemonic": "cusip"' not in src, (
        "a literal definition list is back in _report_helpers.py. The report reads "
        "stage3/spec/signal_definitions.json. A second copy is how it came to print stale names.")


def test_a_cited_work_reaches_the_reference_list():
    """The table cites with the spec's keys, so the report's bibliography must define them.

    Printing author-year WORDS instead was tried and dropped 39 works from the reference
    list, because nothing cited them any more. LaTeX prints '?' for an unknown key and
    carries on, so the helper refuses one.
    """
    import _report_helpers as rpt
    known = set(re.findall(r"@\w+\{([^,\s]+),", rpt.get_references_bib()))
    missing = {m: k for m, r in _spec().items() for k in (r.get("citation_keys") or []) if k not in known}
    assert not missing, f"cited in the spec, absent from get_references_bib(): {missing}"

    assert rpt._cite({"mnemonic": "x", "citation_keys": ["koijen2017"]}, known) == r"\citet{koijen2017}"
    assert rpt._cite({"mnemonic": "x", "citation_keys": [], "citation_text": "Cui, Lu and Song (2026)"},
                     known) == "Cui, Lu and Song (2026)"
    assert rpt._cite({"mnemonic": "x", "citation_keys": []}, known) == "--"
    with pytest.raises(KeyError, match="no_such_key"):
        rpt._cite({"mnemonic": "x", "citation_keys": ["no_such_key"]}, known)

    cells = {s["mnemonic"]: s["citation"] for p in rpt.get_signal_definitions() for s in p["signals"]}
    assert cells["mom3_1"] == r"\citet{gebhardt2005stock}"        # momentum, not betas-or-characteristics
    assert cells["ytm"] == r"\citet{gebhardt2005cross}"
    assert cells["tret_cls"] != "--"                              # words, no key, still printed


def test_the_report_escapes_text_and_leaves_latex_alone():
    import _report_helpers as rpt
    e = rpt._tex_escape
    assert e("S&P 100% dt_s") == r"S\&P 100\% dt\_s"
    assert e(r"$cs_{t-6}$ and cs_t") == r"$cs_{t-6}$ and cs\_t"
    assert e(r"see Fig. \ref{fig:return_timeline}") == r"see Fig. \ref{fig:return_timeline}"
    assert e(r"already \& escaped") == r"already \& escaped"


@pytest.mark.parametrize("field", ["name", "description"])
def test_the_dictionary_says_what_the_spec_says(field):
    spec, dic = _spec(), _dictionary()
    bad = []
    for m, r in spec.items():
        if m not in dic:
            continue
        a, b = norm(r[field]), norm(dic[m][field])
        if a != b:
            bad.append(f"  {m}\n      spec      : {r[field][:200]}\n      dictionary: {dic[m][field][:200]}")
    assert not bad, (
        f"{len(bad)} column(s) have a different {field.upper()} in stage2/DATA_DICTIONARY.md and "
        f"stage3/spec/signal_definitions.json:\n" + "\n".join(bad) +
        "\n\n  Decide which is right AGAINST THE CODE, fix both, and if the spec departs from the "
        "printed paper give the row `paper_prints` and `why_corrected`.")


def test_the_dictionary_groups_columns_as_the_spec_does():
    """Same group names, same order, and each column under the same group."""
    txt = DICT.read_text(encoding="utf-8")
    sec = txt[txt.index("## Signal Definitions"):txt.index("## Redaction")]
    group, got = None, {}
    for line in sec.splitlines():
        if line.startswith("### "):
            group = line[4:].strip()
        m = re.match(r"^\|\s*`([A-Za-z0-9_]+)`\s*\|", line)
        if m and group:
            got.setdefault(m.group(1), group)
    want = {m: r["group"] for m, r in _spec().items()}
    assert list(dict.fromkeys(got.values())) == list(dict.fromkeys(want.values()))
    wrong = {m: (got.get(m), g) for m, g in want.items() if got.get(m) != g}
    assert not wrong, f"(dictionary group, spec group) differ for: {wrong}"
    toc = txt[:txt.index("## How to Use the Monthly Data")]
    for g in dict.fromkeys(want.values()):
        assert f"[{g}](#" in toc, f"the Contents list has no entry for '{g}'"


def test_every_corrected_row_is_in_the_reconciliation():
    """A row that departs from the printed paper must be written up where a reader will look."""
    rec = (REPO / "stage3" / "RECONCILIATION_ia08.md").read_text(encoding="utf-8")
    fixed = [m for m, r in _spec().items() if "why_corrected" in r]
    missing = [m for m in fixed if f"`{m}`" not in rec]
    assert not missing, f"corrected in the spec but not mentioned in RECONCILIATION_ia08.md: {missing}"

    # The note states how many rows were corrected. A typed count went stale twice, so it is held.
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
             "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
             "eighteen", "nineteen", "twenty"]
    said = re.search(r"\. (\w+) rows there differ from the printed table", rec)
    assert said, "RECONCILIATION_ia08.md no longer says how many rows differ from the printed table"
    assert said.group(1).lower() == words[len(fixed)], (
        f"RECONCILIATION_ia08.md says {said.group(1)} rows differ, the spec corrects {len(fixed)}")
