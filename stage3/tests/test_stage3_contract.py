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


def test_the_two_cluster_maps_are_the_same_partition():
    """Section 5 and the zoo must group the 108 signals identically.

    This replaces a test that asserted the two maps DIFFER on `b_rvol`. They do not,
    and never did: it compared cluster NAMES, three of which differ because one section
    reports risk groups and the other beta groups, so it passed while its own failure
    message described something that was not happening. The partition is the thing
    worth pinning -- move a signal in one map and not the other and Section 5's cluster
    rows change with nothing to say so.
    """
    sys.path.insert(0, str(STAGE3 / "s3_nse"))
    sys.path.insert(0, str(STAGE3 / "s4_zoo"))
    import clusters as C
    import zoo_engine as Z

    nse, zoo = {}, {}
    for s, name in Z.CLUSTER_OF.items():
        nse.setdefault(C.get_group_name(C.get_signal_group(s)), set()).add(s)
        zoo.setdefault(name, set()).add(s)
    blocks = lambda d: sorted(tuple(sorted(v)) for v in d.values())
    assert blocks(nse) == blocks(zoo), (
        "the two cluster maps no longer group the signals the same way; Section 5's "
        "cluster rows and the zoo's would stop describing the same sets")
    # the names are allowed to differ, and exactly these three do
    assert set(nse) - set(zoo) == {"Credit & Default Risk", "Macro & Other Risk",
                                   "Volatility & Liquidity Risk"}, sorted(set(nse))


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


# ---------------------------------------------------------------------------
# Sample provenance: every exhibit states the months it was computed over, and
# states them from the data it used rather than from a settings constant.
# ---------------------------------------------------------------------------
# Exhibits with no sample by nature: the three cleaning-parameter tables read
# constants out of the Stage-0/1 source, and Table 3 is a classification list.
# The three exhibits with no sample BY NATURE, each for a stated reason -- not a
# convenience list. Anything else must record the months that produced it.
#   tablesA_filter_params  Tables A.1-A.3: the cleaning filters' parameter values
#   table03                Table 3: the paper's own classification of which factors
#                          are filter-sensitive, a constant list, no data read
#   table_ia08             Table IA.VIII: what each of the 145 panel columns MEANS.
#                          A definition does not have a sample, and giving it one
#                          would be a sentence with nothing behind it.
NO_SAMPLE = {"tablesA_filter_params", "table03", "table_ia08"}


def _summaries() -> dict:
    out = {}
    for f in sorted((STAGE3 / "data").glob("*/*.json")):
        if f.name.startswith("_"):
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d.get("summary"), dict) and "exhibit" in d["summary"]:
            out[f.stem] = d["summary"]
    return out


def test_every_exhibit_records_the_sample_it_used():
    """A caption that cannot say which months produced it is not provenance.

    Before this, 11 of 26 manifests recorded a span and 9 a T, under eight different
    key names -- and the whole of Section 5 recorded neither. A document then claimed
    one sample for exhibits computed over two different ones.
    """
    summ = _summaries()
    if not summ:
        pytest.skip("no exhibits built yet")
    missing = sorted(k for k, v in summ.items()
                     if k not in NO_SAMPLE and not isinstance(v.get("sample"), dict))
    assert not missing, (
        "exhibits with no sample block: " + ", ".join(missing)
        + " -- add `\"sample\": D.sample_block(...)` to the write_result payload")


def test_sample_blocks_are_well_formed():
    """One shape, so a caption can be rendered from any of them without asking."""
    for stem, v in _summaries().items():
        blk = v.get("sample")
        if not isinstance(blk, dict):
            continue
        assert blk.get("first") and blk.get("last"), f"{stem}: sample has no span"
        assert blk["first"] <= blk["last"], f"{stem}: sample runs backwards"
        assert blk.get("basis"), (
            f"{stem}: sample has no `basis` -- a reader needs to know whether T is "
            "asserted, per-series, or absent because the exhibit counts paths")
        if "T" in blk:
            assert blk.get("nw_lags") == int(blk["T"] ** 0.25), (
                f"{stem}: nw_lags does not follow from T")


def test_the_rendered_captions_state_their_sample():
    """The sentence has to reach the page, not just the manifest."""
    tables = sorted((STAGE3 / "reports" / "tables").glob("*.tex"))
    if not tables:
        pytest.skip("no tables built yet")
    # the parameter tables, the classification table and the definition table carry
    # no sample (see NO_SAMPLE above); the two inline count blocks carry no caption
    exempt = {"tableA1", "tableA2", "tableA3", "table03", "table_ia08",
              "inline_counts_alpha", "inline_counts_premium"}
    silent = [t.stem for t in tables
              if t.stem not in exempt
              and "Sample:" not in t.read_text(encoding="utf-8")]
    assert not silent, (
        "rendered tables with no sample sentence: " + ", ".join(silent))


def test_the_lab_series_match_the_window_their_manifest_claims():
    """Section 4's series are built FOR a window; the manifest must not outrun them.

    The defect this catches, which shipped once: `run_lab.py` skips a cell whose nine
    parquets already exist, but wrote its manifest unconditionally from the requested
    dates. Asking for a later `--date-end` therefore skipped every cell and rewrote
    only the manifest -- and since each Section-4 caption takes its sample from the
    series it was handed, the tables went on printing the old window while the run had
    been asked for a new one. Nothing failed; the document was just wrong about itself.
    """
    import json
    pd = pytest.importorskip("pandas")
    root = STAGE3 / "data" / "s2_lab" / "series"
    man = root / "manifest.json"
    parts = sorted(root.glob("*.parquet"))
    if not man.exists() or not parts:
        pytest.skip("the LAB series have not been produced")
    m = json.loads(man.read_text(encoding="utf-8"))
    want_lo, want_hi = str(m["date_start"])[:10], str(m["date_end"])[:10]
    spans = []
    for p in parts:
        idx = pd.to_datetime(pd.read_parquet(p).index)
        spans.append((str(idx.min())[:10], str(idx.max())[:10]))
    lo, hi = min(x for x, _ in spans), max(y for _, y in spans)
    assert lo >= want_lo and hi <= want_hi, (
        f"the manifest claims {want_lo} .. {want_hi} but the series span {lo} .. {hi}")
    # `hi` is the widest series, so this only asks that SOMETHING reached the end --
    # a signal whose own data stops early is coverage, not a skipped rebuild.
    assert hi[:7] == want_hi[:7], (
        f"the manifest claims the window ends {want_hi} but no series reaches past "
        f"{hi} -- the cells were skipped rather than rebuilt")


def test_the_exhibit_index_is_current():
    """INDEX.md is generated; regenerating it must produce no diff.

    Hand-written it would be stale in a month and nothing would say so -- the run still
    works when a driver is renamed, you just find out from a "Not produced" stub at the
    end of a 14-minute run.

    ❗Skipped unless every exhibit has been produced. One column -- the sample each
    exhibit states -- is read from the manifests, so a clone that has run nothing, or a
    run stopped half way, legitimately regenerates a different file. Failing there would
    mean a stranger's first `pytest` is red for no reason, and a gate that cries wolf is
    a gate people stop reading. It is exact once a full run exists, which is when it
    can catch anything.
    """
    import subprocess
    sys.path.insert(0, str(STAGE3))
    sys.path.insert(0, str(STAGE3 / "tools"))
    import build_index as B

    rs, _ = B.rows()
    if not all(r["produced"] for r in rs):
        pytest.skip("not every exhibit is built, so the sample column is incomplete")
    out = subprocess.run(
        [sys.executable, str(STAGE3 / "tools" / "build_index.py"), "--check"],
        capture_output=True, text=True, cwd=str(STAGE3))
    assert out.returncode == 0, (out.stdout + out.stderr).strip()


def test_every_exhibit_the_report_expects_has_a_driver():
    """Nothing asserted this before: the suite stayed green through a rename.

    `make_report.EXHIBITS` is the list of what the PDF prints. If a stem there is
    produced by no file in the tree, the report emits a stub instead of the exhibit,
    and the only symptom is a gap in a document nobody diffs.
    """
    sys.path.insert(0, str(STAGE3))
    sys.path.insert(0, str(STAGE3 / "tools"))
    import build_index as B

    rs, orphans = B.rows()
    assert not orphans, "no driver produces: " + ", ".join(orphans)
    assert len(rs) == sum(len(items) for _, _, items in B.MR.EXHIBITS)


def test_every_sorted_signal_has_a_definition():
    """Table IA.VIII must define every signal Stage 3 actually sorts.

    If it does not, the table documents a different universe from the one the paper
    reports on -- and the gap is invisible, because a missing definition does not stop
    anything from running.
    """
    import json
    sys.path.insert(0, str(STAGE3 / "s4_zoo"))
    import zoo_engine as Z

    spec = STAGE3 / "spec" / "signal_definitions.json"
    rows = json.loads(spec.read_text(encoding="utf-8"))["rows"]
    defined = {r["mnemonic"] for r in rows}
    assert not (set(Z.CLUSTER_OF) - defined), (
        "sorted with no definition: " + ", ".join(sorted(set(Z.CLUSTER_OF) - defined)))
    assert len(rows) == len(defined) == 145, (len(rows), len(defined))


def test_the_signal_spec_matches_the_stage2_contract():
    """The 145 defined names must be exactly the 145 panel columns, in any order.

    This is the join that makes the whole thing checkable: the paper's definition
    table, Stage 2's panel and Stage 3's sorts all describing one set of names. A
    variable added to Stage 2 and not to the spec shows up here rather than as a
    column nobody ever documented.
    """
    import json
    contract = STAGE3.parent / "stage2" / "lib" / "contract.py"
    if not contract.exists():
        pytest.skip("stage 2 is not beside stage 3 in this checkout")
    sys.path.insert(0, str(STAGE3.parent / "stage2"))
    from lib.contract import PANEL_COLUMNS

    rows = json.loads(
        (STAGE3 / "spec" / "signal_definitions.json").read_text(encoding="utf-8"))["rows"]
    spec = {r["mnemonic"] for r in rows}
    assert spec == set(PANEL_COLUMNS), {
        "defined, not a panel column": sorted(spec - set(PANEL_COLUMNS)),
        "a panel column, undefined": sorted(set(PANEL_COLUMNS) - spec)}


def test_corrected_definition_rows_say_what_was_printed_and_why():
    """A correction that does not record what it replaced is not reproducible.

    Some rows differ from the printed Table IA.VIII. Each must carry the paper's own
    text and the reason, so the printed table can be reconstructed from the spec and
    no change is silent.
    """
    import json
    rows = json.loads(
        (STAGE3 / "spec" / "signal_definitions.json").read_text(encoding="utf-8"))["rows"]
    fixed = [r for r in rows if "why_corrected" in r]
    assert fixed, "no corrections recorded -- see RECONCILIATION_ia08.md"
    for r in fixed:
        assert r.get("paper_prints") or r.get("paper_prints_citation"), (
            f"{r['mnemonic']}: corrected without recording what the paper prints")
        assert len(r["why_corrected"]) > 80, (
            f"{r['mnemonic']}: the reason is too short to be a reason")


def _declared_flags(script: str) -> set:
    """Every long option a driver declares, read from its own `add_argument` calls."""
    import ast
    src = (STAGE3 / script).read_text(encoding="utf-8")
    out = set()
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for a in node.args:
                if isinstance(a, ast.Constant) and str(a.value).startswith("--"):
                    out.add(a.value)
    return out


def test_the_orchestrator_only_passes_flags_that_exist():
    """A flag the orchestrator adds must be one the driver accepts.

    The defect this catches, found by a cold run on 2026-09-12: `--window full` was
    being passed to `s3_nse/run_dua_grid.py`, which has no such flag -- it computes
    BOTH windows in one pass. Every run had passed anyway, because that producer was
    always SKIPPED: its `_complete.json` existed, so the argv was never built. The
    first run that actually had to produce the grid died with argparse exit 2, and
    the whole chain stopped behind it.

    Static, so it does not need the flag to be reached.
    """
    sys.path.insert(0, str(STAGE3))
    import _run_stage3 as R

    bad = []
    for script, flag in R.SAMPLE_FLAG.items():
        if flag and flag not in _declared_flags(script):
            bad.append(f"{script} has no {flag}")
    for script, flag in R.TAKES_WINDOW.items():
        if flag not in _declared_flags(script):
            bad.append(f"{script} has no {flag}")
    for script in R.ACCEPTS_FAST:
        if "--fast" not in _declared_flags(script):
            bad.append(f"{script} has no --fast")
    for script in R.ACCEPTS_FORCE:
        if "--force" not in _declared_flags(script):
            bad.append(f"{script} has no --force")
    assert not bad, "the orchestrator would pass a flag that does not exist: " + \
        "; ".join(bad)


def test_every_step_names_a_script_that_exists():
    """A renamed driver must fail here, not 20 minutes into a run."""
    sys.path.insert(0, str(STAGE3))
    import _run_stage3 as R

    missing = sorted({script for _s, _k, script, _a, _t in R.STEPS
                      if not (STAGE3 / script).exists()})
    assert not missing, "steps naming a file that is not there: " + ", ".join(missing)


def test_the_paper_groups_the_signals_the_way_stage3_does():
    """Table IA.VIII's own cluster blocks must match both of Stage 3's maps.

    Until the spec existed this could not be checked: the paper's grouping lived in
    LaTeX. Now it is data, so a signal that moves cluster in the code -- or a
    transcription slip in the spec -- shows up here. Names are allowed to differ
    (Section 5 reports risk groups, the zoo reports beta groups); the PARTITION is not.

    Measured when this was written: all three agree on all 108.
    """
    import json
    import re
    from collections import defaultdict

    sys.path.insert(0, str(STAGE3 / "s3_nse"))
    sys.path.insert(0, str(STAGE3 / "s4_zoo"))
    import clusters as C
    import zoo_engine as Z

    rows = json.loads(
        (STAGE3 / "spec" / "signal_definitions.json").read_text(encoding="utf-8"))["rows"]
    paper = defaultdict(set)
    for r in rows:
        m = re.match(r"Cluster ([IVX]+):", r["group"] or "")
        if m:
            paper[m.group(1)].add(r["mnemonic"])

    zoo, nse = defaultdict(set), defaultdict(set)
    for sig, name in Z.CLUSTER_OF.items():
        zoo[name].add(sig)
        nse[C.get_group_name(C.get_signal_group(sig))].add(sig)

    def blocks(d):
        return sorted(tuple(sorted(v)) for v in d.values())

    assert sum(len(v) for v in paper.values()) == 108, "the spec lost a sorted signal"
    assert blocks(paper) == blocks(zoo), "the paper and the zoo group differently"
    assert blocks(paper) == blocks(nse), "the paper and Section 5 group differently"


def test_the_prose_cluster_counts_are_recorded_as_a_paper_defect():
    """The paper's prose counts contradict its own table; that must stay written down.

    22/14/14/15 in the prose against 21/13/13/18 in the table, in four of nine
    clusters. Both total 108, which is why nothing caught it for so long. Stage 3
    follows the TABLE. This test does not check the paper -- it checks that the finding
    is still on file, so a future tidy-up cannot quietly delete the only record of it.
    """
    doc = (STAGE3 / "RECONCILIATION_ia08.md").read_text(encoding="utf-8")
    for n in ("22", "21", "14", "13", "15", "18"):
        assert n in doc, f"the prose-vs-table cluster counts no longer name {n}"
    assert "contradicts its own table" in doc
