"""test_stage3_contract.py -- the gates that hold without any data on disk.

These run on a fresh clone, before Stage 2 has produced anything. They check the things
that would otherwise only surface in front of a user: an absolute path baked into a
script, a caption missing for an exhibit that renders one, a step in the runner whose
script does not exist, a private artifact still referenced.

    python -m pytest tests/ -q
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path

import pytest

STAGE3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE3))

PY_FILES = sorted(p for p in STAGE3.rglob("*.py")
                  if "__pycache__" not in p.parts and "tests" not in p.parts)
SECTION_DIRS = ("s0_data", "s1_lib", "s2_lab", "s3_nse", "s4_zoo")


# ---------------------------------------------------------------- portability
def test_no_absolute_user_paths():
    """A home directory baked into a script is what makes a package unrunnable elsewhere."""
    bad = re.compile(r"""["'](?:[A-Za-z]:[\\/]|/home/|/Users/|~/)""")
    hits = []
    for p in PY_FILES:
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if bad.search(line):
                hits.append(f"{p.relative_to(STAGE3)}:{i}: {line.strip()[:90]}")
    assert not hits, "absolute user paths:\n" + "\n".join(hits)


def test_every_module_parses():
    for p in PY_FILES:
        ast.parse(p.read_text(encoding="utf-8"), filename=str(p))


# --------------------------------------------------------- the public boundary
PRIVATE_MARKERS = [
    "Dropbox", "DRR_MUA", "drr_replication", "trace_duckdb", "monthly_data",
    "lehman", "lhm_ice", "STAGE3_RESULTS", "GOLD_PANEL", "GOLD_MMN", "GOLD_BBW",
    "OURS_PANEL", "probelib", "verifylib", "texparse", "MAIN_TEX",
]
# A LaTeX label is not a path: `tab:monthly_data_availability` is the paper's own name
# for an exhibit and has nothing to do with any repository.
MARKER_EXEMPT = re.compile(r"(?:tab|fig):[a-z0-9_]*")


@pytest.mark.parametrize("marker", PRIVATE_MARKERS)
def test_no_private_artifact_references(marker):
    """Stage 3 must not name a file, repository or module a public user cannot have."""
    hits = []
    for p in list(PY_FILES) + sorted(STAGE3.glob("*.md")) + [STAGE3 / "run_stage3.sh"]:
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if marker.lower() in MARKER_EXEMPT.sub("", line).lower():
                hits.append(f"{p.relative_to(STAGE3)}:{i}: {line.strip()[:90]}")
    assert not hits, f"references to {marker!r}:\n" + "\n".join(hits)


def test_no_diff_against_the_paper():
    """The public package produces exhibits; it never compares them to the printed ones."""
    banned = re.compile(r"diff_against_paper|parse_printed|printed_fig\d|"
                        r"pdf_mnemonics|EXPECTED_PAPER|KNOWN_.*DEVIATION")
    hits = [f"{p.relative_to(STAGE3)}" for p in PY_FILES
            if banned.search(p.read_text(encoding="utf-8"))]
    assert not hits, "paper-diff machinery still present in: " + ", ".join(hits)


# ------------------------------------------------------------------- the steps
def test_every_runner_step_exists():
    import _run_stage3 as R

    missing = [f"{s[2]} {' '.join(s[3])}" for s in R.STEPS
               if not (STAGE3 / s[2]).exists()]
    assert not missing, "steps naming a script that does not exist:\n" + "\n".join(missing)


def test_every_driver_is_in_the_runner():
    """A driver the runner does not call is not part of the replication."""
    import _run_stage3 as R

    called = {s[2] for s in R.STEPS}
    drivers = {f"{d}/{p.name}" for d in SECTION_DIRS
               for p in (STAGE3 / d).glob("[tf]*.py")
               if p.name not in ("two_row.py",)}
    drivers |= {f"{d}/{p.name}" for d in SECTION_DIRS
                for p in (STAGE3 / d).glob("run_*.py")}
    orphans = sorted(drivers - called)
    assert not orphans, "drivers not in _run_stage3.STEPS: " + ", ".join(orphans)


def test_runner_sections_match_settings():
    import _run_stage3 as R

    assert {s[0] for s in R.STEPS} == set(R.SECTIONS)


# ---------------------------------------------------------------- the captions
def test_every_rendered_label_has_a_caption():
    """An exhibit without a caption renders a table with an empty title."""
    import captions

    labels = set()
    for d in SECTION_DIRS:
        for p in (STAGE3 / d).glob("*.py"):
            src = p.read_text(encoding="utf-8")
            if "captions.caption(" not in src:
                continue
            labels |= set(re.findall(r'LABEL\s*=\s*"([^"]+)"', src))
            labels |= set(re.findall(r'"(tab:[a-z0-9_]+)"', src))
    unknown = sorted(l for l in labels if l not in captions.CAPTIONS)
    assert not unknown, "labels with no caption registered: " + ", ".join(unknown)


def test_caption_lookup_fails_loudly():
    import captions

    with pytest.raises(KeyError, match="no caption registered"):
        captions.caption("tab:does_not_exist")


# ------------------------------------------------------------ the input contract
def test_input_contract_is_wellformed():
    spec = json.loads((STAGE3 / "spec" / "inputs.json").read_text(encoding="utf-8"))
    import _stage3_settings as S

    assert set(spec["inputs"]) == set(S.INPUTS), (
        "spec/inputs.json and _stage3_settings.INPUTS disagree about the input set")
    for name, decl in spec["inputs"].items():
        assert decl.get("what"), f"{name}: no description"
        assert decl.get("consumers"), f"{name}: no consumers listed"
        assert decl.get("required_columns"), f"{name}: no required columns"


def test_panel_contract_covers_the_zoo_signals():
    """The 108 signals the zoo sorts must all be declared as panel columns."""
    spec = json.loads((STAGE3 / "spec" / "inputs.json").read_text(encoding="utf-8"))
    declared = set(spec["inputs"]["STAGE2_PANEL"]["required_columns"])
    src = (STAGE3 / "s4_zoo" / "zoo_engine.py").read_text(encoding="utf-8")
    signals = set(re.findall(r'"([a-z][a-z0-9_]*)"',
                             src[src.index("CLUSTERS = {"):src.index("CLUSTER_OF")]))
    signals -= {"name", "signals"}
    missing = sorted(signals - declared)
    assert not missing, "zoo signals absent from the panel contract: " + ", ".join(missing)


def test_check_inputs_reports_a_missing_file(tmp_path, monkeypatch):
    """The contract check must FAIL on a missing input, not skip it."""
    monkeypatch.setenv("STAGE2_PANEL", str(tmp_path / "nope.parquet"))
    for m in ("_stage3_settings", "tools.check_inputs"):
        sys.modules.pop(m, None)
    sys.path.insert(0, str(STAGE3 / "tools"))
    sys.modules.pop("check_inputs", None)
    import check_inputs

    results, ok = check_inputs.check()
    assert not ok
    panel = next(r for r in results if r["input"] == "STAGE2_PANEL")
    assert "not found" in panel["problems"]


# --------------------------------------------------------------- the conventions
def test_sign_flag_round_trip():
    import helper_functions as H

    assert H.strip_sign_flag("dcs6*") == ("dcs6", -1.0)
    assert H.strip_sign_flag("dcs6") == ("dcs6", 1.0)
    assert H.base_mnemonic("b_amd_m_wf*") == "b_amd_m"
    assert H.base_mnemonic("dcs6_mmn_wf_hy*") == "dcs6"


def test_nw_lag_rule():
    import helper_functions as H

    assert H.nw_lags(268) == 4
    assert H.nw_lags(269) == 4
    assert H.nw_lags(15) == 1


def test_sample_windows_give_four_lags():
    import _stage3_settings as S
    import helper_functions as H
    import pandas as pd

    for name, w in S.SAMPLE.items():
        T = len(pd.date_range(w["start"], w["end"], freq="ME"))
        assert H.nw_lags(T) == 4, f"{name}: T={T} gives {H.nw_lags(T)} lags, not 4"


def test_the_two_sections_use_different_cluster_maps():
    """s3_nse and s4_zoo disagree on b_rvol deliberately; unifying them moves cells."""
    sys.path.insert(0, str(STAGE3 / "s3_nse"))
    sys.path.insert(0, str(STAGE3 / "s4_zoo"))
    import clusters as C
    import zoo_engine as Z

    assert C.get_group_name(C.get_signal_group("b_rvol")) != Z.CLUSTER_OF["b_rvol"], (
        "the two cluster maps now agree on b_rvol -- if that was deliberate, update "
        "this test and the comment in clusters.py; if not, a map has drifted")


def test_no_unimported_shared_module():
    """A module nothing imports is not part of the replication.

    The shared modules at the top level are library code: if none of the section
    folders reaches for one, it is dead weight in a package meant to be read. (Entry
    points and drivers are exempt -- they are run, not imported.)
    """
    entry = {"_run_stage3", "_stage3_settings", "paths", "conftest", "make_report"}
    shared = {p.stem for p in STAGE3.glob("*.py")} - entry
    src = "\n".join(p.read_text(encoding="utf-8") for p in PY_FILES)
    orphans = sorted(m for m in shared
                     if not re.search(rf"^\s*(?:import {m}\b|from {m} import)",
                                      src, re.M))
    assert not orphans, ("shared modules nothing imports: " + ", ".join(orphans)
                         + " -- use them or delete them")
