"""Stage 4's rules, on small synthetic panels: no Stage 2 output is needed.

    python -m pytest stage4/tests -q

The check against the real published factors is `compare_published.py`, which needs a full
Stage 2 build; these tests cover what that comparison cannot isolate: the duration swap,
the flip set, the CSV pivot, the grid in the spec, and one small run through PyBondLab.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import _stage4_settings as S
from factorlib import inputs, release, sorts

SWAP = S.SPEC["swap_columns"]["names"]


def _panel(n_bonds: int = 60, n_months: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    rows = []
    for b in range(n_bonds):
        for d in dates:
            rows.append({"cusip": f"C{b:04d}", "date": d, "permno": 1000 + b // 3})
    df = pd.DataFrame(rows)
    n = len(df)
    df["ret_vw"] = rng.normal(0.005, 0.02, n)
    df["tret"] = df["tret_bns"] = df["tret_cls"] = rng.normal(0.002, 0.005, n).astype("float32")
    df["str"] = rng.normal(0, 0.02, n)
    df["mcap_e"] = rng.uniform(50, 500, n)
    df["spc_rat"] = rng.integers(1, 22, n).astype("float64")
    df["sig_a"] = rng.normal(size=n)
    df["sig_b"] = rng.normal(size=n)
    for c in SWAP:
        df[c] = 1.0
    return df


def _blocks(df: pd.DataFrame, tmp_path, value: float = 2.0, drop: str | None = None):
    keys = df[["cusip", "date"]]
    betas = [c for c in SWAP[:51]]
    moms = [c for c in SWAP[51:]]
    for name, cols in (("bx", betas), ("mx", moms)):
        b = keys.copy()
        for c in cols:
            if c != drop:
                b[c] = value
        b.to_parquet(tmp_path / f"{name}.parquet", index=False)


def test_the_spec_declares_the_published_grid():
    assert len(S.SIGNALS) == len(set(S.SIGNALS)) == 108
    assert list(S.RETURN_TYPES) == ["exc", "dur", "dbns", "dcls"]
    assert [b["label"] for b in S.SPEC["rating_bands"]] == ["all", "ig", "nig"]
    assert [b["suffix"] for b in S.SPEC["rating_bands"]] == [None, "ig", "hy"]
    assert S.SPEC["sorts"]["single"]["num_portfolios"] == {"all": 10, "ig": 5, "nig": 5}
    assert len(SWAP) == len(set(SWAP)) == 68


def test_the_swap_replaces_the_68_and_adjusts_the_returns(tmp_path, monkeypatch):
    df = _panel(10, 6)
    _blocks(df, tmp_path)
    monkeypatch.setattr(S, "block_path", lambda name: tmp_path / f"{name}.parquet")
    out = inputs.swap(df.copy(), ["bx", "mx"], "tret_bns")
    assert len(out) == len(df)
    assert (out[SWAP] == 2.0).all().all()          # the blocks' values, under the same names
    t = df["tret_bns"].astype("float64")
    np.testing.assert_array_equal(out["ret_vwx"], df["ret_vw"] - t)
    np.testing.assert_array_equal(out["str"], df["str"] - t)


def _swap_before_it_was_shared(df, blocks, tret_col):
    """inputs.swap as it stood before the swap moved to return_types.py, verbatim."""
    names = S.SPEC["swap_columns"]["names"]
    absent = [c for c in names if c not in df.columns]
    if absent:
        raise AssertionError(f"the panel lacks {len(absent)} of the 68 columns to swap: {absent}")
    df = df.drop(columns=names)
    added: list[str] = []
    for name in blocks:
        blk = pd.read_parquet(S.block_path(name))
        blk["date"] = pd.to_datetime(blk["date"])
        cols = [c for c in blk.columns if c not in ("cusip", "date")]
        added += cols
        n0 = len(df)
        df = df.merge(blk[["cusip", "date"] + cols], on=["cusip", "date"], how="left")
        if len(df) != n0:
            raise AssertionError(f"{name} changed the row count {n0:,} -> {len(df):,}: it "
                                 "has duplicate (cusip, date) keys")
    if set(names) - set(added):
        raise AssertionError(f"dropped but not added back: {sorted(set(names) - set(added))}")
    if set(added) - set(names):
        raise AssertionError(f"added but never dropped: {sorted(set(added) - set(names))}")
    t = df[tret_col].astype(df["ret_vw"].dtype)
    df["ret_vwx"] = df["ret_vw"] - t
    df["str"] = df["str"] - t
    return df


@pytest.mark.parametrize("rt", ["dur", "dbns", "dcls"])
def test_the_shared_swap_builds_the_same_panel(tmp_path, monkeypatch, rt):
    """The swap moved to return_types.py: stage 4's panels must be exactly what they were,
    values, dtypes and column order."""
    df = _panel(12, 8, seed=3)
    rng = np.random.default_rng(4)
    for name, cols in zip(S.RETURN_TYPES[rt]["blocks"], (SWAP[:51], SWAP[51:])):
        b = df[["cusip", "date"]].copy()
        for c in cols:
            b[c] = rng.normal(size=len(b)).astype("float32")
        b.sample(frac=1.0, random_state=5).to_parquet(tmp_path / f"{name}.parquet", index=False)
    monkeypatch.setattr(S, "block_path", lambda name: tmp_path / f"{name}.parquet")
    spec = S.RETURN_TYPES[rt]
    new = inputs.swap(df.copy(), spec["blocks"], spec["tret"])
    old = _swap_before_it_was_shared(df.copy(), spec["blocks"], spec["tret"])
    pd.testing.assert_frame_equal(new, old, check_exact=True)


def test_the_swap_refuses_a_block_that_misses_a_column(tmp_path, monkeypatch):
    df = _panel(10, 6)
    _blocks(df, tmp_path, drop="b_amd")
    monkeypatch.setattr(S, "block_path", lambda name: tmp_path / f"{name}.parquet")
    with pytest.raises(AssertionError, match="dropped but not added back"):
        inputs.swap(df, ["bx", "mx"], "tret")


def test_a_missing_block_names_the_command_that_makes_it(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "PANEL", tmp_path / "panel.parquet")
    monkeypatch.setattr(S, "RISK_FREE", tmp_path / "factors.parquet")
    monkeypatch.setattr(S, "block_path", lambda name: tmp_path / f"{name}.parquet")
    msg = "\n".join(inputs.missing_inputs(["exc", "dbns"]))
    assert "panel.parquet" in msg and "betas_bns.parquet" in msg
    assert "make_excess_blocks.py --benchmark bns" in msg


def test_the_flip_set_is_keyed_by_return_type():
    d = pd.date_range("2010-01-31", periods=4, freq="ME")
    rows = [{"date": x, "factor": "f", "leg": "ls", "weighting": "ew", "return_type": rt,
             "return": sign * 0.01} for rt, sign in (("exc", 1), ("dur", -1)) for x in d]
    flips = sorts.flip_set(pd.DataFrame(rows))
    assert flips == {"f|ew|dur": True, "f|ew|exc": False}
    with pytest.raises(AssertionError, match="more than once"):
        sorts.flip_set(pd.DataFrame(rows + rows))


def test_the_csvs_hold_every_long_short_cell():
    d = pd.date_range("2010-01-31", periods=3, freq="ME")
    rows = [{"date": x, "factor": f, "leg": leg, "weighting": w, "rating_type": "all",
             "return_type": "exc", "return": 0.01}
            for x in d for f in ("a", "b") for leg in ("l", "s", "ls") for w in ("ew", "vw")]
    csvs = release.wide_csvs(pd.DataFrame(rows))
    assert sorted(csvs) == ["exc_all_ew.csv", "exc_all_vw.csv"]
    assert all(w.shape == (3, 2) for w in csvs.values())


def test_a_small_panel_runs_through_pybondlab_unflipped():
    import pybondlab_pin
    if pybondlab_pin.check():
        pytest.skip("needs the pinned PyBondLab (requirements-local.txt)")
    df = _panel()
    df["ret_vwx"] = df["ret_vw"] - df["tret"].astype("float64")
    n_months = df["date"].nunique()
    for sort in S.SORTS:
        out = pd.concat(sorts.run_bands(df, ["sig_a", "sig_b"], sort=sort, return_type="dur"),
                        ignore_index=True)
        # every (month, signal, leg, weighting) in every band: a complete rectangle
        assert len(out) == 3 * n_months * 2 * 3 * 2
        assert set(out["rating_type"]) == {"all", "ig", "nig"}
        assert set(out["return_type"]) == {"dur"}
        assert out["factor"].str.contains("[*]").sum() == 0      # never sign-corrected


def test_each_return_type_sorts_its_own_return_column():
    """`dur` sorts `ret_vwx`, `exc` sorts `ret_vw`. A build that sorted the same column for both
    passed every other test here: shape and labels are the same either way. With the Treasury
    return set to zero the two must agree exactly, and with a real one they must not."""
    import pybondlab_pin
    if pybondlab_pin.check():
        pytest.skip("needs the pinned PyBondLab (requirements-local.txt)")

    def ls(df, rt):
        out = pd.concat(sorts.run_bands(df, ["sig_a"], sort="single", return_type=rt),
                        ignore_index=True)
        keep = (out["leg"] == "ls") & (out["weighting"] == "vw") & (out["rating_type"] == "all")
        return out[keep].sort_values("date")["return"].to_numpy()

    df = _panel()
    df["ret_vwx"] = df["ret_vw"]                       # tret = 0: the two returns are the same
    np.testing.assert_array_equal(ls(df, "exc"), ls(df, "dur"))
    df["ret_vwx"] = df["ret_vw"] - df["tret"].astype("float64")
    exc, dur = ls(df, "exc"), ls(df, "dur")
    ok = ~(np.isnan(exc) | np.isnan(dur))
    assert ok.sum() > 10 and np.abs(exc[ok] - dur[ok]).max() > 1e-6, \
        "the duration-adjusted factor is the excess-return factor: it sorted the wrong column"
