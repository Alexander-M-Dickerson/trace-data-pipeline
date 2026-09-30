"""test_returns.py -- the return type a run sorts (returns.py, return_types.py at the root).

  * the standard run (`exc`) forms every return exactly as it did, at each section's own
    convention, and touches no signal;
  * a duration-adjusted run swaps the return-based signals for its blocks' (all 68, or the
    few a loader read), adjusts `str` and `str_mmn`, and forms the return less its Treasury
    column, never less rf;
  * a duration-adjusted run writes its own tree and refuses the standard one;
  * its captions and manifests say which return it used; the input check asks for its
    blocks, and only its own;
  * no section forms a return except through returns.py;
  * compare_runs.py counts what moved between two runs.

    python -m pytest tests/test_returns.py -q
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STAGE3 = Path(__file__).resolve().parents[1]
for p in (STAGE3, STAGE3 / "tools"):
    sys.path.insert(0, str(p))

import _stage3_settings as S  # noqa: E402
import return_types as RT     # noqa: E402
import returns as R           # noqa: E402

SWAP = RT.SWAP_COLUMNS


def _panel(n_bonds: int = 8, n_months: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2012-01-31", periods=n_months, freq="ME")
    df = pd.DataFrame([{"cusip": f"C{b:03d}", "date": d} for b in range(n_bonds) for d in dates])
    n = len(df)
    for c in ("ret_vw", "ret_vw_bgn", "rfret", "str", "str_mmn"):
        df[c] = rng.normal(0.004, 0.02, n)
    df["tret_bns"] = rng.normal(0.002, 0.004, n).astype("float32")
    for c in SWAP:
        df[c] = 1.0
    return df


@pytest.fixture
def dbns(tmp_path, monkeypatch):
    """A duration-adjusted run (dbns) whose two blocks are small files holding 2.0 and 3.0."""
    df = _panel()
    betas, moms = RT.blocks("dbns")
    for name, value in ((betas, 2.0), (moms, 3.0)):
        cols = [c for c in SWAP if (c in SWAP[:51]) == (name == betas)]
        b = df[["cusip", "date"]].copy()
        for c in cols:
            b[c] = value
        b.sample(frac=1.0, random_state=1).to_parquet(tmp_path / f"{name}.parquet", index=False)
    monkeypatch.setattr(S, "RETURNS", "dbns")
    monkeypatch.setattr(S, "block_path", lambda name: tmp_path / f"{name}.parquet")
    return df


# --------------------------------------------------------------- the standard run
def test_the_standard_run_forms_every_return_as_it_did(monkeypatch):
    monkeypatch.setattr(S, "RETURNS", "exc")
    df = _panel()
    before = df.copy()
    pd.testing.assert_series_equal(R.ret(df, "ret_vw", rf="rfret"), before["ret_vw"] - before["rfret"],
                                   check_exact=True)
    pd.testing.assert_series_equal(R.ret(df, "ret_vw_bgn"), before["ret_vw_bgn"], check_exact=True)
    assert R.signals(df) is df
    R.set_returns(df, ("ret_vw", "ret_vw_bgn"))                      # raw: left alone
    pd.testing.assert_frame_equal(df, before, check_exact=True)
    R.set_returns(df, ("ret_vw",), rf="rfret")
    np.testing.assert_array_equal(df["ret_vw"], before["ret_vw"] - before["rfret"])
    assert (R.load_columns(), R.manifest(), R.caption_note(), R.input_paths()) == ([], {}, "", [])


# --------------------------------------------------------------- a duration-adjusted run
def test_a_duration_adjusted_run_swaps_every_signal_and_adjusts_the_returns(dbns):
    df = dbns.copy()
    out = R.signals(df)
    assert (out[SWAP[:51]] == 2.0).all().all() and (out[SWAP[51:]] == 3.0).all().all()
    t = dbns["tret_bns"].astype("float64").to_numpy()
    for c in ("str", "str_mmn"):                                   # returns themselves
        np.testing.assert_array_equal(out.sort_values(["cusip", "date"])[c].to_numpy(),
                                      dbns.sort_values(["cusip", "date"])[c].to_numpy() - t)
    np.testing.assert_array_equal(R.ret(dbns, "ret_vw", rf="rfret"), dbns["ret_vw"] - t)
    np.testing.assert_array_equal(R.ret(dbns, "ret_vw_bgn"), dbns["ret_vw_bgn"] - t)
    assert R.load_columns() == ["tret_bns"] and R.manifest() == {"returns": "dbns"}
    note = R.caption_note()
    assert r"tret\_bns" in note
    assert not re.search(r"(?<!\\)[_%&#$]", note), "the note goes into LaTeX captions"


def test_a_loader_that_read_a_few_signals_swaps_only_those_in_place(dbns):
    few = dbns[["cusip", "date", "ret_vw", "tret_bns", "mom6_1", "b_mktb", "str"]].copy()
    out = R.signals(few, names=["mom6_1", "b_mktb", "str"])
    assert list(out.columns) == list(few.columns)                  # nothing added, order kept
    assert (out["mom6_1"] == 3.0).all() and (out["b_mktb"] == 2.0).all()
    np.testing.assert_array_equal(out["str"], dbns["str"] - dbns["tret_bns"].astype("float64"))


def test_the_subset_swap_gives_the_full_swap_s_values(dbns):
    full = RT.swap(dbns.copy(), RT.blocks("dbns"), S.block_path)
    part = RT.swap(dbns.copy(), RT.blocks("dbns"), S.block_path, names=["b_amd", "mom12_1"])
    key = ["cusip", "date"]
    m = part.merge(full, on=key, suffixes=("", "_full"))
    np.testing.assert_array_equal(m["b_amd"], m["b_amd_full"])
    np.testing.assert_array_equal(m["mom12_1"], m["mom12_1_full"])


def test_the_caption_says_which_return(dbns, monkeypatch):
    import drrlib as D
    block = {"first": "2002-09-30", "last": "2025-11-30", "T": 279}
    assert D.sample_sentence(block).startswith("Sample: 2002-09 to 2025-11, T=279.")
    assert r"tret\_bns" in D.sample_sentence(block)
    monkeypatch.setattr(S, "RETURNS", "exc")
    assert D.sample_sentence(block) == "Sample: 2002-09 to 2025-11, T=279."


# --------------------------------------------------------------- where a run writes
def _settings(**env) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if not k.startswith("STAGE3_")}
    e.update(env)
    return subprocess.run([sys.executable, "-c",
                           "import _stage3_settings as S; print(S.RETURNS); print(S.DATA); "
                           "print(S.REPORTS)"],
                          cwd=STAGE3, env=e, capture_output=True, text=True)


def test_a_duration_adjusted_run_writes_its_own_tree():
    r = _settings(STAGE3_RETURNS="dbns")
    assert r.returncode == 0, r.stderr
    rt, data, reports = r.stdout.split("\n")[:3]
    assert rt == "dbns"
    assert Path(data) == STAGE3 / "variants" / "dbns" / "data"
    assert Path(reports) == STAGE3 / "variants" / "dbns" / "reports"


def test_a_duration_adjusted_run_refuses_the_standard_tree():
    r = _settings(STAGE3_RETURNS="dbns", STAGE3_DATA=str(STAGE3 / "data"))
    assert r.returncode != 0 and "standard run" in (r.stderr + r.stdout)


def test_an_unknown_return_type_stops():
    r = _settings(STAGE3_RETURNS="dfoo")
    assert r.returncode != 0 and "unknown return type" in (r.stderr + r.stdout)


def test_the_runner_reads_returns_before_the_settings_load():
    e = {k: v for k, v in os.environ.items() if not k.startswith("STAGE3_")}
    r = subprocess.run([sys.executable, "-c",
                        "import sys; sys.argv = ['_run_stage3.py', '--returns', 'dbns', '--list']; "
                        "import _run_stage3 as R; print(R.S.RETURNS); "
                        "print([s[4] for s in R.STEPS if s[2] == 's1_lib/run_sorts.py'][0]); "
                        "print(R.SECTION_INPUTS['lib'])"],
                       cwd=STAGE3, env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rt, target, lib_inputs = r.stdout.split("\n")[:3]
    assert rt == "dbns" and target.startswith("data/sorts/dbns_")
    assert "STAGE2_BETAS_BNS" in lib_inputs and "STAGE2_MOM_RETX_BNS" in lib_inputs


def test_the_input_check_asks_for_the_blocks_of_its_own_type_only(tmp_path, monkeypatch):
    import check_inputs
    monkeypatch.setattr(check_inputs.S, "RETURNS", "exc")
    names = [r["input"] for r in check_inputs.check()[0]]
    assert not any("BETAS" in n or "MOM_RETX" in n for n in names)
    monkeypatch.setattr(check_inputs.S, "RETURNS", "dbns")
    for k in ("STAGE2_BETAS_BNS", "STAGE2_MOM_RETX_BNS"):
        monkeypatch.setitem(check_inputs.S.INPUTS, k, tmp_path / f"{k}.parquet")
    rows = {r["input"]: r for r in check_inputs.check()[0]}
    assert set(rows) >= {"STAGE2_BETAS_BNS", "STAGE2_MOM_RETX_BNS"}
    assert "STAGE2_BETAS_CLS" not in rows
    probs = rows["STAGE2_BETAS_BNS"]["problems"]
    assert "not found" in probs
    assert any("make_excess_blocks.py --benchmark bns" in p for p in probs)


# --------------------------------------------------------------- one place forms a return
# The data appendix describes the panel as built; its duration-adjusted rows are its own
# descriptive columns and do not depend on the run's return type.
ALLOWED = {"s0_data/data_engine.py"}


def test_no_section_forms_a_return_except_through_returns_py():
    pat = re.compile(r'\[\s*"ret_vw(?:_bgn)?"\s*\]\s*-|-\s*\w+\[\s*"(?:rfret|rf|tret\w*)"\s*\]')
    hits = []
    for f in sorted(STAGE3.rglob("*.py")):
        rel = f.relative_to(STAGE3).as_posix()
        if rel.startswith(("tests/", "variants/")) or rel in ALLOWED or rel == "returns.py":
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.lstrip().startswith("#") and pat.search(line):
                hits.append(f"{rel}:{i}: {line.strip()}")
    assert not hits, ("a return formed outside returns.py, so a duration-adjusted run would "
                      "not reach it:\n" + "\n".join(hits))


# --------------------------------------------------------------- compare_runs.py
def test_compare_runs_counts_what_moved(tmp_path):
    import compare_runs as C
    cells = pd.DataFrame({"panel": ["A"] * 4, "factor": ["ytm", "ytm", "cs", "cs"],
                          "column": ["mu"] * 4, "kind": ["coef", "t", "coef", "t"],
                          "value": [0.5, 2.5, 0.2, 1.0]})
    stats = pd.DataFrame({"ret_type": ["exc"], "factor": ["ytm"], "value": [1.0], "tstat": [3.0]})
    keyless = pd.DataFrame({"year": [2002, 2003], "n": [5, 6]})
    for run, (c, s, k) in {"a": (cells, stats, keyless),
                           "b": (cells.assign(value=[-0.1, 1.5, 0.3, 2.2]),
                                 stats.assign(ret_type="dbns", tstat=1.0),
                                 keyless.assign(n=[5, 7]))}.items():
        d = tmp_path / run / "data" / "s1_lib"
        d.mkdir(parents=True)
        c.to_csv(d / "table01_cells.csv", index=False)
        s.to_csv(d / "table01_stats.csv", index=False)
        k.to_csv(d / "table_x.csv", index=False)
    out, infos, only = C.compare(C.data_dir(tmp_path / "a"), C.data_dir(tmp_path / "b"))
    by = {i["table"]: i for i in infos}
    t01 = by["s1_lib/table01_cells.csv"]
    assert (t01["cells"], t01["sign_changes"], t01["t_crossings"]) == (4, 1, 2)
    assert by["s1_lib/table01_stats.csv"]["only_a"] == 0          # ret_type is not a key
    assert by["s1_lib/table01_stats.csv"]["t_crossings"] == 1
    assert by["s1_lib/table_x.csv"]["aligned"] == "by row" and by["s1_lib/table_x.csv"]["changed"] == 1
    assert only == {"A": [], "B": []}
