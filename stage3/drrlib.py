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
    wrong is silent and plausible-looking. ❗How many factors it touches depends on
    the file -- two of the seven in the three-approach CSVs, around forty-five of the
    108 in the wide ones -- because the flip set is decided on the sample that was
    sorted. It is a property of the run, not a constant.
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


def sample_block(obj=None, *, window: str | None = None, basis: str = "",
                 T: int | None = None, T_min: int | None = None,
                 T_max: int | None = None, first=None, last=None,
                 **extra) -> dict:
    """THE sample-provenance record every exhibit carries. One shape, one place.

    ❗Built from the data the exhibit ACTUALLY used, never from a settings constant.
    Before this existed, only 11 of 26 result manifests recorded a span and 9 a T, under
    eight different key names -- and where a "sample" was recorded it was usually the
    REQUESTED window rather than the realised one, so a caption built on it would have
    described the intention instead of the result.

    Pass an index or a frame and the span and length are read off it; or pass `first`,
    `last` and `T`/`T_min`/`T_max` directly when the exhibit pools series of different
    lengths and no single index exists.

    `basis` says in words what the numbers rest on -- "the LIB window, T asserted",
    "each series' own length", "construction paths, not a time series" -- because the
    three cases are genuinely different and a reader needs to know which one they have.
    """
    import pandas as _pd

    idx = None
    if obj is not None:
        idx = obj.index if hasattr(obj, "index") and not isinstance(obj, _pd.Index) else obj
    if idx is not None and len(idx):
        ts = _pd.to_datetime(_pd.Index(idx))
        first = first or ts.min().strftime("%Y-%m-%d")
        last = last or ts.max().strftime("%Y-%m-%d")
        T = T if T is not None else len(ts)

    block: dict = {"first": first, "last": last, "basis": basis}
    if T is not None:
        block["T"] = int(T)
        block["nw_lags"] = nw_lags(int(T))
    if T_min is not None:
        block["T_min"] = int(T_min)
    if T_max is not None:
        block["T_max"] = int(T_max)
    if window:
        block["window"] = window
    block.update(extra)
    return block


def sample_sentence(block: dict | None) -> str:
    """The sentence that closes a caption, in the form the paper uses.

        Sample: 2002-09 to 2024-12, T=268.
        Sample: 2002-08 to 2024-12, T 257-268 by series.
        Sample: 2002-09 to 2024-12; 18,032 construction paths.

    ❗Derived, never typed. The titles in `captions.py` are the authors' words and do
    not change; this is the part that has to track whatever data the run was given, so
    that a document whose data appendix reaches 2025-11 says 2025-11.
    """
    if not block:
        return ""
    def _m(v):
        return str(v)[:7] if v else None
    first, last = _m(block.get("first")), _m(block.get("last"))
    span = f"Sample: {first} to {last}" if first and last else "Sample"
    bits = []
    if block.get("T") is not None:
        bits.append(f"T={block['T']}")
    elif block.get("T_min") is not None and block.get("T_max") is not None:
        lo, hi = block["T_min"], block["T_max"]
        bits.append(f"T={lo}" if lo == hi else f"T {lo}-{hi} by series")
    if block.get("n_paths") is not None:
        # the noun comes from the block: Section 5's two halves vary different things
        # (data-cleaning filters vs portfolio construction) and must not both be called
        # "construction paths"
        bits.append(f"{block['n_paths']:,} {block.get('paths_label', 'paths')}")
    if not bits:
        return span + "."
    sep = "; " if block.get("n_paths") is not None and block.get("T") is None else ", "
    return span + sep + ", ".join(bits) + "."


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
    # ❗Key on the PORTABLE path. The cache is written under data/ and ships with
    # everything else there, and an absolute key would put one machine's home
    # directory in the archive. Size + mtime still do the invalidating.
    key = f"{portable(path)}|{st.st_size}|{st.st_mtime_ns}"
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


def portable(p: Path) -> str:
    """A path as it should be RECORDED: relative to the pipeline, if it sits inside it.

    ❗A manifest is meant to travel -- in a zip, on OSF, in a replication archive. An
    absolute path records one machine's home directory and tells a reader nothing they
    can act on. Anything genuinely outside the pipeline tree keeps its absolute form,
    because there a relative path would be a lie.
    """
    p = Path(p)
    for base in (paths.PIPELINE, paths.STAGE3):
        try:
            return p.resolve().relative_to(Path(base).resolve()).as_posix()
        except ValueError:
            continue
    return str(p)


def fingerprint(path: Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": portable(p), "exists": False}
    st = p.stat()
    return {"path": portable(p), "exists": True, "bytes": st.st_size,
            "sha256_16": sha256_file(p)[:16]}


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(paths.STAGE3), *args],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None



def write_atomic(df, path: Path, **kw) -> Path:
    """Write a frame to a temp file beside its destination, then rename it into place.

    ❗The orchestrator decides to skip a producer by asking whether its output EXISTS.
    A Ctrl-C or an out-of-disk halfway through `to_csv` leaves a truncated file that
    exists, so the producer is skipped for ever and every exhibit downstream formats a
    short series without complaint. `os.replace` is atomic on both Windows and POSIX,
    so what lands is either the whole file or nothing.
    """
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix == ".csv":
        df.to_csv(tmp, **kw)
    else:
        df.to_parquet(tmp, **kw)
    os.replace(tmp, path)
    return path


def mark_complete(where: Path, ok: bool, what: dict) -> Path:
    """Write (or remove) a completion marker.

    ❗The orchestrator decides whether to skip a producer by looking for ONE file. If
    that file is an early artifact, a grid that dies two thirds of the way through
    leaves it behind, gets skipped for ever, and no amount of re-running helps. So the
    marker is written LAST and only when the run actually passed its own check.
    """
    import json as _json
    marker = Path(where)
    marker.parent.mkdir(parents=True, exist_ok=True)
    if ok:
        marker.write_text(_json.dumps(what, indent=1, default=str), encoding="utf-8")
    elif marker.exists():
        marker.unlink()          # a previously-complete grid is no longer complete
    return marker

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
