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


def _artifacts() -> list[Path]:
    """The files a RUN produces: manifests and the assembled report source.

    ❗These are gitignored, so they never reach a reviewer through `git diff` -- but they
    are exactly what travels in a zip, on OSF, or in a replication archive. The code
    gates below were blind to them until a leak was found in one.

    ❗EVERYTHING under `reports/`, not just the `.tex`. The narrower glob had a hole
    exactly where a leak was: pdflatex writes the absolute path of the TeX installation
    into its transcript, so `exhibits.build.log` carried a home directory while this
    test passed. A gate that names the right threat and looks in the wrong place is
    worse than no gate.
    """
    out = list((STAGE3 / "data").rglob("*.json"))
    out += [p for p in (STAGE3 / "reports").rglob("*")
            if p.is_file() and p.suffix.lower() not in {".pdf", ".png", ".parquet"}]
    return [p for p in out if p.is_file()]


@pytest.mark.skipif(not (STAGE3 / "data").exists(),
                    reason="nothing produced yet -- run the pipeline first")
def test_produced_artifacts_carry_no_absolute_paths():
    """A manifest records provenance. An absolute path records someone's home directory."""
    bad = re.compile(r"[A-Za-z]:\\\\Users|[A-Za-z]:/Users|/home/|/Users/")
    hits = []
    for p in _artifacts():
        for i, line in enumerate(p.read_text(encoding="utf-8",
                                             errors="replace").splitlines(), 1):
            if bad.search(line):
                hits.append(f"{p.relative_to(STAGE3)}:{i}: {line.strip()[:100]}")
    assert not hits, ("absolute user paths in produced artifacts:\n"
                      + "\n".join(hits[:20]))


@pytest.mark.skipif(not (STAGE3 / "data").exists(),
                    reason="nothing produced yet -- run the pipeline first")
@pytest.mark.parametrize("marker", PRIVATE_MARKERS)
def test_produced_artifacts_carry_no_private_references(marker):
    """The same markers, applied to what a run writes rather than to the code."""
    hits = []
    for p in _artifacts():
        for i, line in enumerate(p.read_text(encoding="utf-8",
                                             errors="replace").splitlines(), 1):
            if marker.lower() in MARKER_EXEMPT.sub("", line).lower():
                hits.append(f"{p.relative_to(STAGE3)}:{i}: {line.strip()[:100]}")
    assert not hits, (f"{marker!r} in produced artifacts:\n" + "\n".join(hits[:20]))


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


# ---------------------------------------------------------------------------
# The degeneracy ledger: one classification, one denominator.
#
# ❗These are the tests that would have caught the defects of 2026-09-11: a signal's
# own end date read as degeneracy (an entire factor vanishing from the full window),
# and cells with no series at all counted as construction paths in two of the three
# Section-5 denominators.
# ---------------------------------------------------------------------------
LEDGER_DIR = STAGE3 / "data" / "s3_nse" / "mua_summary"
STATUSES = ("inadmissible", "redundant", "no_series", "short_sample",
            "empty_leg", "ok")
WINDOWS = ("paper", "full")


def _ledger(window: str):
    pd = pytest.importorskip("pandas")
    f = LEDGER_DIR / f"mua_status_{window}.parquet"
    if not f.exists():
        pytest.skip(f"no ledger yet -- run s3_nse/mua_summarize.py ({f.name})")
    return pd.read_parquet(f)


@pytest.mark.parametrize("window", WINDOWS)
def test_ledger_is_a_rectangle(window):
    """Every cell of the 108 x 216 grid gets exactly one status, and they sum."""
    led = _ledger(window)
    assert len(led) == 23_328, f"{window}: {len(led):,} rows, expected 23,328"
    assert not led.duplicated(["signal", "spec_id"]).any(), \
        f"{window}: duplicate (signal, spec_id) in the ledger"
    unknown = sorted(set(led["status"]) - set(STATUSES))
    assert not unknown, f"{window}: statuses outside the vocabulary: {unknown}"
    counts = led["status"].value_counts()
    assert int(counts.sum()) == 23_328, f"{window}: statuses sum to {int(counts.sum()):,}"


@pytest.mark.parametrize("window", WINDOWS)
def test_no_strategy_is_degenerate_on_tail_months_alone(window):
    """A signal's coverage is not a defect.

    Three of the 108 signals stop before the panel does. Every month after that has no
    bonds, and judging those months made the whole signal degenerate -- 168 of b_cptlt's
    168 strategies on the full window, which removed the factor from every exhibit. A
    cell flagged `empty_leg` must have an empty month INSIDE its own active span.
    """
    led = _ledger(window)
    bad = led[(led["status"] == "empty_leg") & (led["n_months_active"] <= 0)]
    assert bad.empty, (
        f"{window}: {len(bad)} strategies flagged degenerate with no active months at "
        f"all -- e.g. {bad['signal'].iloc[0]}/{bad['spec_id'].iloc[0]}. That is a "
        "coverage window being read as degeneracy.")


def test_the_degenerate_set_does_not_depend_on_the_reporting_window():
    """Degeneracy is a property of a strategy, not of the window you print.

    Before the window rule was made two-sided this was 80 on the paper window and 246
    on the full one, the difference being one signal whose data ends early.
    """
    a, b = _ledger("paper"), _ledger("full")
    ca = a["status"].value_counts().reindex(STATUSES, fill_value=0)
    cb = b["status"].value_counts().reindex(STATUSES, fill_value=0)
    assert int(ca["empty_leg"]) == int(cb["empty_leg"]), (
        f"empty_leg differs by window: paper {int(ca['empty_leg'])}, "
        f"full {int(cb['empty_leg'])} -- a longer window should not create degeneracy")
    assert int(ca["ok"]) == int(cb["ok"]), (
        f"usable paths differ by window: paper {int(ca['ok']):,}, "
        f"full {int(cb['ok']):,}")


@pytest.mark.parametrize("window", WINDOWS)
def test_no_cell_without_a_series_is_counted_as_a_path(window):
    """`ok` means a well-defined factor return series. A cell with none is not one."""
    led = _ledger(window)
    ok = led[led["status"] == "ok"]
    assert int((ok["n_obs"] == 0).sum()) == 0, (
        f"{window}: {int((ok['n_obs'] == 0).sum())} cells marked `ok` have no "
        "observations")
    assert int((ok["n_months_grid"] == 0).sum()) == 0, (
        f"{window}: cells marked `ok` with no months in the grid at all")


@pytest.mark.parametrize("window", WINDOWS)
def test_empty_cells_never_reach_the_all_breakpoint_universe(window):
    """❗The line between "a count moves" and "a value moves".

    Both sign-correction baselines are `all`-universe specs, and
    `clusters.apply_sign_correction` silently declines to flip a signal whose baseline
    mean is NaN. While the engine's instability stays confined to the restricted
    universes it can only move counts. If it ever reaches `all`, a whole signal's row
    changes sign and the printed numbers are wrong.
    """
    led = _ledger(window)
    empty = led[led["status"] == "no_series"]["spec_id"]
    reached = [sp for sp in empty if "_ig_bp_" not in sp and "_lg_bp_" not in sp]
    assert not reached, (
        f"{window}: {len(reached)} cell(s) with no series in the `all` breakpoint "
        f"universe, e.g. {reached[:3]}. A baseline spec can now be NaN, which silently "
        "un-flips a signal -- this moves VALUES, not just counts.")


@pytest.mark.parametrize("window", WINDOWS)
def test_every_printed_denominator_comes_from_the_ledger(window):
    """Table 6, IA.XVIII and IA.XIX must agree, and agree with the ledger.

    They did not: 18,064 / 18,038 / a pool built on 18,064. Each had inferred the answer
    from whichever artifact it happened to read.
    """
    pd = pytest.importorskip("pandas")
    led = _ledger(window)
    n_ok = int((led["status"] == "ok").sum())

    j = STAGE3 / "data" / "s3_nse" / f"table06_{window}.json"
    c = STAGE3 / "data" / "s3_nse" / f"table_ia18_portfolio_size_{window}.csv"
    k = STAGE3 / "data" / "s3_nse" / f"table_ia19_{window}.json"
    if not (j.exists() and c.exists() and k.exists()):
        pytest.skip("Section-5 exhibits not built yet")

    t6 = json.loads(j.read_text(encoding="utf-8"))["summary"]["n_paths"]
    assert t6 == n_ok, f"{window}: Table 6 prints {t6:,}, ledger says {n_ok:,}"

    ia18 = pd.read_csv(c)
    tot = int(ia18.loc[ia18["row"] == "All specifications", "n_spec"].iloc[0])
    assert tot == n_ok, f"{window}: IA.XVIII prints {tot:,}, ledger says {n_ok:,}"

    pool = json.loads(k.read_text(encoding="utf-8"))["summary"]["n_denominator_pool"]
    assert pool == n_ok - 648, (
        f"{window}: IA.XIX pool {pool:,}, expected {n_ok - 648:,} "
        "(usable paths less the six *_all_all_all per signal)")


def test_the_low_bond_threshold_has_one_definition():
    """The printed footnote quotes it; the `pct_low` column is computed from it."""
    src = (STAGE3 / "s3_nse" / "t18_portfolio_size.py").read_text(encoding="utf-8")
    assert "from mua_summarize import LOW_BOND_THRESHOLD" in src, (
        "t18_portfolio_size.py restates the low-bond threshold instead of importing it "
        "-- two copies drift and the footnote stops describing the column")


def test_the_ledger_cannot_be_older_than_the_grid():
    """A statistics layer derived from a grid that is no longer on disk is not a result.

    Re-running the grid does not invalidate the summarizer's completion marker, so the
    orchestrator skips it and every Section-5 exhibit quietly describes a different grid.
    Found on 2026-09-11: six cells with a full 268-month series in the grid and n_obs = 0
    in the summary, written seven minutes apart.
    """
    src = (STAGE3 / "s3_nse" / "nse_engine.py").read_text(encoding="utf-8")
    assert "STAGE3_ALLOW_STALE_LEDGER" in src and "is OLDER than the grid" in src, (
        "nse_engine.load_ledger no longer refuses a ledger older than its grid")


def test_a_failed_check_reaches_the_reader():
    """`write_result` runs before `b.check`, so a red exhibit still renders.

    That is the right call -- a missing table is worse than a flagged one -- but the
    reader holding the PDF has to be told, or a failed assertion looks like a clean run.
    """
    src = (STAGE3 / "make_report.py").read_text(encoding="utf-8")
    assert "def failed_checks(" in src and "Checks that did not pass" in src, (
        "make_report no longer surfaces failed checks on the provenance page")


def test_the_table6_footnote_arithmetic_closes():
    """The one piece of arithmetic a reader can check by eye must be right.

    The footnote used to report the empty-leg exclusions alone, so subtracting it from
    the 18,144-strategy grid landed 32 short with nothing on the page to explain the
    gap. Both exclusions are now stated.
    """
    f = STAGE3 / "reports" / "tables" / "table06_paper.tex"
    if not f.exists():
        pytest.skip("Table 6 not built yet")
    m = re.search(r"([\d,]+) construction paths of ([\d,]+), after excluding "
                  r"(\d+) with an empty leg and (\d+) never formed",
                  f.read_text(encoding="utf-8"))
    assert m, "Table 6's footnote no longer states its exclusions in a checkable form"
    kept, grid, empty, never = (int(x.replace(",", "")) for x in m.groups())
    assert grid - empty - never == kept, (
        f"the footnote does not close: {grid:,} - {empty} - {never} = "
        f"{grid - empty - never:,}, but it prints {kept:,}")
