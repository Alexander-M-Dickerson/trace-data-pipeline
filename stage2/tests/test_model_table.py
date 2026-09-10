# -*- coding: utf-8 -*-
"""
test_model_table.py
===================
DATA_DICTIONARY.md prints a table of every rolling-beta factor model. That table is
maintained by hand and the models live in code, so the two drift -- and they had:

    7 models were missing from the table entirely (coskew, defterm, dvix_asym, illiq,
      unc3, unc6, uncl)
    2 rows were stale (DEF and TERM, which no longer exist as separate models)
    2 rows named the wrong regressors: LVL and YSP were documented as two-factor models
      on `mktb` when the code runs them univariate

None of that raises anything. A reader takes the table at face value and estimates the
wrong specification, or believes a beta loads on a factor it never saw.

This test derives the table from `lib.betas.BETA_MODELS` and requires the document to
agree, model for model and regressor for regressor.

What it does NOT check: the LaTeX equations further down the appendix, the "Outputs Kept"
and "ivol" columns, or whether the prose around the table is true. Those are still read by
eye.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

STAGE2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STAGE2))

DICT = STAGE2 / "DATA_DICTIONARY.md"
HEADER = "| Model | Factors | Outputs Kept"


def _models_from_code() -> dict[str, str]:
    """{MODEL NAME: 'factor, factor, ...'} straight out of the BETA_MODELS literal.

    Read through the AST rather than by importing and running anything, so the test
    reports what the source says and cannot be fooled by a value assembled at run time.
    """
    tree = ast.parse((STAGE2 / "lib" / "betas.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "BETA_MODELS" for t in node.targets):
            literal = ast.literal_eval(node.value)
            return {m["name"].upper(): ", ".join(m["factors"]) for m in literal}
    raise AssertionError("no BETA_MODELS assignment found in lib/betas.py")


def _models_from_doc() -> dict[str, str]:
    txt = DICT.read_text(encoding="utf-8")
    block = txt[txt.index(HEADER):]
    block = block[:block.index("\n\n")]
    out = {}
    for line in block.splitlines()[2:]:            # skip header + separator
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and cells[0]:
            out[cells[0].upper()] = re.sub(r"\s+", " ", cells[1])
    return out


def test_the_model_table_is_present():
    assert DICT.exists(), f"no {DICT}"
    assert HEADER in DICT.read_text(encoding="utf-8"), (
        f"DATA_DICTIONARY.md has no factor-model table (looked for {HEADER!r})")


def test_every_model_in_the_code_is_documented():
    code, doc = _models_from_code(), _models_from_doc()
    missing = sorted(set(code) - set(doc))
    assert not missing, (
        f"{len(missing)} beta model(s) run in the code but are absent from the "
        f"DATA_DICTIONARY.md model table: {missing}\n"
        f"  Add a row for each: | NAME | {', '.join('factors')} | ... |")


def test_the_table_documents_no_model_that_does_not_exist():
    code, doc = _models_from_code(), _models_from_doc()
    stale = sorted(set(doc) - set(code))
    assert not stale, (
        f"{len(stale)} row(s) in the DATA_DICTIONARY.md model table name a model that "
        f"lib/betas.py does not run: {stale}\n"
        f"  Remove them, or restore the model.")


def test_documented_regressors_match_the_code():
    code, doc = _models_from_code(), _models_from_doc()
    wrong = [(k, doc[k], code[k]) for k in sorted(set(code) & set(doc)) if doc[k] != code[k]]
    assert not wrong, (
        f"{len(wrong)} model(s) are documented with the wrong regressors:\n"
        + "\n".join(f"    {k}: document says {d!r}, code runs {c!r}" for k, d, c in wrong))
