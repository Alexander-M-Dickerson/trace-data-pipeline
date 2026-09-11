"""drrlib.py -- the shared load / statistics / provenance layer.

One code path for every number the repo prints:

  * loading      the monthly panel, the *_mmn twins, the factor block, and the
                 long-format sort panels Stage 3 produces
  * inference    Newey-West means, CAPM_B alphas, and paired-difference tests, all
                 with lags = floor(T**0.25) -- the paper's single convention
  * provenance   write_result() stamps every JSON with its inputs' sha256, the git
                 commit, the wall clock, and WHICH PyBondLab build produced it

House conventions encoded here, each one asserted against the data:

  * returns in the sort panels are DECIMALS; the paper prints percent. Convert once,
    at the boundary, via `PCT`.
  * a factor mnemonic ending in "*" is PyBondLab sign-corrected. The paper prints the
    UNFLIPPED sign, so `strip_sign_flag` returns the sign to undo it. Getting this
    wrong flips dcs6 and str and nothing else -- a silent, plausible-looking error.
  * the LIB sample is 2002-09-30..2024-12-31, T = 268. Asserted, never assumed.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

import paths
# The four naming/lag conventions live in ONE place. They are re-exported here
# so `drrlib.nw_lags` and `drrlib.base_mnemonic` keep working -- two definitions
# of a sign-flip rule is exactly the kind of thing that drifts apart.
from helper_functions import (  # noqa: F401
    base_mnemonic, nw_lags, strip_sign_flag, to_percent)

PCT = 100.0                       # sort-panel returns are decimals; the paper prints %
SAMPLE_START = "2002-09-30"       # LIB / NSE tables: T = 268
SAMPLE_END = "2024-12-31"
SAMPLE_START_LAB = "2002-08-31"   # LAB tables: T = 269

# The window policy. Producers save FULL-LENGTH untruncated series; the sample window
# is applied at the statistics layer, by truncating the SERIES and recomputing. A stored
# statistic is never truncated -- truncating a mean is not the mean of the truncation, and
# the Newey-West lag count is derived from T.
SAMPLE_END_PAPER = "2024-12-31"
WINDOW_POLICY = "produce-untruncated-truncate-at-stats"
LIB_FACTORS = ("ytm", "cs", "bbtm", "dcs6", "val_ipr", "val_hz", "str")
ILLIQ_FACTORS = ("ami", "ar_sprd", "cs_sprd", "ilq", "spd_rel")

# The 30 signals built from a price or a return, and so carrying an unadjusted `_mmn`
# twin that the bias census swaps in. Everything else is a term, a rating, an industry
# or an amount: nothing a noisy month-end price can contaminate.
PRICE_BASED = (
    "ytm", "cs", "bbtm", "dcs6", "val_ipr", "val_hz", "str",
    "sze", "val_hz_dts", "val_ipr_dts",
    "pi", "ami", "ami_v", "lix", "ilq", "roll",
    "spd_abs", "spd_rel", "cs_sprd", "ar_sprd",
    "p_zro", "p_fht", "vov", "dvol", "dskew", "dkurt",
    "rvol", "rsj", "rsk", "rkt",
)


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------


def _ols(y: np.ndarray, X: np.ndarray, lags: int):
    import statsmodels.api as sm
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def nw_mean(x: pd.Series, lags: int | None = None) -> tuple[float, float]:
    """Mean of x and its Newey-West t-statistic (regression on a constant)."""
    v = pd.Series(x).dropna().astype(float)
    if v.empty:
        return float("nan"), float("nan")
    lags = nw_lags(len(v)) if lags is None else lags
    res = _ols(v.to_numpy(), np.ones((len(v), 1)), lags)
    return float(res.params[0]), float(res.tvalues[0])


def capm_alpha(y: pd.Series, mktb: pd.Series, lags: int | None = None) -> tuple[float, float]:
    """CAPM_B alpha: intercept of y on MKTB alone, HAC t-statistic.

    y and mktb are aligned on their index first; only overlapping non-null dates count.
    """
    df = pd.concat([pd.Series(y).rename("y"), pd.Series(mktb).rename("m")], axis=1).dropna()
    if df.empty:
        return float("nan"), float("nan")
    lags = nw_lags(len(df)) if lags is None else lags
    X = np.column_stack([np.ones(len(df)), df["m"].to_numpy(float)])
    res = _ols(df["y"].to_numpy(float), X, lags)
    return float(res.params[0]), float(res.tvalues[0])


def paired_diff_mean(a: pd.Series, b: pd.Series, lags: int | None = None) -> tuple[float, float]:
    """Mean of (a - b) and its NW t-statistic, on the paired (overlapping) sample.

    The paper tests the bias on the DIFFERENCE SERIES, not by differencing two
    separately estimated means -- the two give the same point estimate but very
    different standard errors.
    """
    df = pd.concat([pd.Series(a).rename("a"), pd.Series(b).rename("b")], axis=1).dropna()
    return nw_mean(df["a"] - df["b"], lags)


def paired_diff_alpha(a: pd.Series, b: pd.Series, mktb: pd.Series,
                      lags: int | None = None) -> tuple[float, float]:
    """Alpha of (a - b) on MKTB, and its HAC t-statistic."""
    df = pd.concat([pd.Series(a).rename("a"), pd.Series(b).rename("b")], axis=1).dropna()
    return capm_alpha(df["a"] - df["b"], mktb, lags)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_sort_panel(csv_path: Path, *, leg: str = "ls", weighting: str = "vw",
                    start: str = SAMPLE_START, end: str = SAMPLE_END,
                    undo_sign_flip: bool = True, value_col: str = "return") -> pd.DataFrame:
    """One stage3 long-format sort CSV -> a wide date x base-mnemonic value frame.

    Columns are BASE mnemonics ('ytm', 'cs', ...) so the three approaches line up
    across files that decorate the same factor differently (`ytm`, `ytm_mmn`,
    `ytm_mmn_wf`). Returns stay in decimals -- multiply by PCT at the print boundary.

    `value_col` picks any numeric column of the CSV ('return', 'turnover', or a
    characteristic like 'lib'/'ilq'). The sign-flip undo applies to whatever is
    loaded: PyBondLab negates BOTH the L-S return and the L-S characteristic
    spread when it sign-corrects (extract.py flips the legs), so the same undo is
    correct for both. Turnover is flip-invariant (an average of the two legs), so
    undoing there is a no-op in effect only when legs are symmetric -- load
    turnover with undo_sign_flip=False.
    """
    import duckdb

    p = Path(csv_path).as_posix()
    df = duckdb.sql(
        f'SELECT date, factor, "{value_col}" AS raw FROM read_csv_auto(\'{p}\') '
        f"WHERE leg = '{leg}' AND weighting = '{weighting}' "
        f"AND date BETWEEN DATE '{start}' AND DATE '{end}'"
    ).df()
    if df.empty:
        raise ValueError(f"no rows for leg={leg} weighting={weighting} in {csv_path}")
    df["date"] = pd.to_datetime(df["date"])
    sign = df["factor"].map(lambda f: strip_sign_flag(f)[1]) if undo_sign_flip else 1.0
    df["value"] = df["raw"] * sign
    df["mnemonic"] = df["factor"].map(base_mnemonic)
    wide = df.pivot(index="date", columns="mnemonic", values="value").sort_index()
    wide.columns.name = None
    return wide


def load_mktb(path: Path | None = None, *, start: str = SAMPLE_START,
              end: str = SAMPLE_END, column: str = "MKTB") -> pd.Series:
    """The bond market factor used for every CAPM_B alpha in the paper (decimals)."""
    import duckdb

    p = Path(path or paths.BBW).as_posix()
    df = duckdb.sql(f"SELECT date, \"{column}\" AS mktb FROM read_parquet('{p}')").df()
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["mktb"].sort_index().astype(float)
    return s.loc[str(start):str(end)]


def assert_sample(idx: pd.Index, expected_T: int, what: str = "sample") -> None:
    """Fail loudly on a wrong sample length -- T drives the NW lag count."""
    if len(idx) != expected_T:
        raise AssertionError(
            f"{what}: expected T={expected_T}, got {len(idx)} "
            f"({idx.min()} .. {idx.max()}). The lag rule and every t-statistic depend on T.")


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
_SHA_CACHE = paths.CACHE / "_sha_cache.json"


def sha256_file(path: Path) -> str:
    """sha256 of a file, cached on (size, mtime_ns) so multi-GB inputs hash once."""
    path = Path(path)
    st = path.stat()
    key = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    cache = {}
    if _SHA_CACHE.exists():
        try:
            cache = json.loads(_SHA_CACHE.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    if key in cache:
        return cache[key]
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    cache[key] = h.hexdigest()
    _SHA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _SHA_CACHE.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    return cache[key]


def fingerprint(path: Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {"path": str(p), "exists": True, "bytes": st.st_size,
            "sha256_16": sha256_file(p)[:16]}


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(paths.STAGE3), *args],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None


def write_result(name: str, payload: dict, *, section: str, inputs: list[Path],
                 t0: float | None = None, extra: dict | None = None) -> Path:
    """data/<section>/<name>.json, with the manifest block every result carries.

    A number whose inputs, code version and engine are not recorded cannot be defended
    later, so the block is written by this one function rather than by each caller.
    """
    import sys

    pbl = None
    if "pblenv" in sys.modules:
        try:
            pbl = sys.modules["pblenv"].active()
        except Exception:
            pbl = None
    rec = dict(payload)
    rec["manifest"] = {
        "name": name,
        "section": section,
        "written_utc": pd.Timestamp.utcnow().isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "code_dirty": bool(_git("status", "--porcelain")),
        "wall_s": round(time.perf_counter() - t0, 3) if t0 is not None else None,
        "inputs": [fingerprint(p) for p in inputs],
        "pybondlab": pbl,
        "python": sys.version.split()[0],
        **(extra or {}),
    }
    out = paths.section_results(section) / f"{name}.json"
    out.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return out
