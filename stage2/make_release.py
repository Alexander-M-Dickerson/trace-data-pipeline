# -*- coding: utf-8 -*-
"""
make_release.py
===============
Package a built Stage 2 vintage for publication.

Build artifacts carry the Stage 1 date stamp (``factors.parquet`` under
``output/blocks/<mode>/``) so a panel is always traceable to the file it came from.
Released artifacts carry the vintage YEAR -- ``factors_2026.parquet`` -- which is what
users cite. This script does that rename and assembles the provenance that makes a
published number checkable.

    python3 make_release.py                     # panel, factor and BBW bundles
    python3 make_release.py --what panel        # just the panel
    python3 make_release.py --out-dir dist      # somewhere else
    python3 make_release.py --what bbw          # the BBW four-factor bundle
    python3 make_release.py --what daily        # the Stage 1 daily panel, public layout
    python3 make_release.py --mode <mode>       # a build other than the default stage1

❗**This script is for REDISTRIBUTION, and only redistribution.**

The panel we BUILD is not the panel we PUBLISH: `permco` and `gvkey` are proprietary
identifiers and the agency ratings are licensed, so both are redacted here before anything
is written, and `assert_publishable` refuses to package a file where that did not take.

**Your own build is NOT redacted and does not need to be.** The redaction exists so that
the files put on openbondassetpricing.com can be downloaded by people who hold no licence.
If you ran Stage 0-2 yourself you have a WRDS subscription and the licences that come with
it, so the panel under `output/panel/` is complete -- full `permco`, `gvkey` and raw
1-22 agency ratings -- and nothing in the build touches them. `redact_for_publication`
returns a copy and is called from this file alone; running it does not alter your data.

You only need this script if you are publishing a vintage for others to download.

The vintage is derived from the data (see ``_stage2_settings.release_vintage``), so next
year's run publishes itself.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import _stage2_settings as cfg
from lib import frontier

# Sources that are not fetched over the web and so have no sidecar; recorded by hand.
NON_WEB_SOURCES = {
    "vix_monthly": {
        "source": "WRDS table cboe.cboe",
        "note": "daily VIX, aggregated to month-end and scaled /100/sqrt(12)",
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _public_path(path: Path) -> str:
    """A path as it may appear in a published file: relative to the repository, never absolute.

    PROVENANCE.json is zipped into public bundles. An absolute path puts the builder's user
    name and disk layout into a download, and the 2026-09 BBW bundle carried one.
    """
    path = Path(path)
    try:
        return path.resolve().relative_to(Path(cfg.ROOT_PATH).resolve()).as_posix()
    except ValueError:
        return path.name


def _source(path: Path) -> dict:
    """What a released file was cut from, so a stale bundle can be told from a fresh one.

    Every gate downstream compared the served bytes with the staged bytes. None asked whether
    the staged file came from the CURRENT build, and a bundle cut two days before a rebuild was
    published. `source_sha256` is what that question is asked of.
    """
    return {"source": _public_path(path), "source_sha256": _sha256(path)}


def collect_provenance(factors: pd.DataFrame) -> dict:
    """Per-source URL, fetch time, sha256, rows and span, from the fetcher's sidecars."""
    sources = {}
    for meta_path in sorted(cfg.FACTOR_CACHE_DIR.glob("*.meta.json")):
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        sources[m["name"]] = {k: m[k] for k in ("url", "fetched_utc", "rows", "span", "sha256")
                              if k in m}
    for name, info in NON_WEB_SOURCES.items():
        sources.setdefault(name, dict(info))

    # Per-column last non-null month: the honest way to ship columns with different frontiers.
    ends = {}
    for c in factors.columns:
        if c == "date":
            continue
        nn = factors.loc[factors[c].notna(), "date"]
        ends[c] = str(nn.max())[:7] if not nn.empty else None

    return {
        "vintage": cfg.release_vintage(),
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(factors)),
        "span": [str(factors["date"].min())[:10], str(factors["date"].max())[:10]],
        "sources": sources,
        "column_last_month": ends,
    }


README = """# OSBAP monthly factor panel -- {vintage} vintage

`factors_{vintage}.parquet` -- the exogenous monthly factor time series used to estimate
every rolling beta in the Stage 2 bond panel of the same vintage.

**Why this file is published.** Stage 2 assembles this panel from public sources at build
time, and those sources revise: Ken French restates SMB/HML, FRED re-seasonally-adjusts
CPI, and the Ludvigson uncertainty series is re-estimated on each release. A panel built
next month therefore will not reproduce a panel published today. Shipping the exact factor
panel a release consumed is what lets anyone reproduce -- or check -- a published number.

To use it, point Stage 2 at this file:

```
FACTOR_SOURCE = "pinned"
FACTORS_PINNED_FILE = ".../factors_{vintage}.parquet"
```

## Contents

{rows} monthly observations, {span_start} to {span_end}, one row per month-end.

{coltable}

## Coverage, and two caveats

Series have different publication frontiers, which is normal -- each ends where its source
ends. `PROVENANCE.json` records the last month of **every** column; the full table is
below.

❗**Two known upstream limits**, neither caused by this pipeline:

| series | ends | why |
|---|---|---|
| `cptl`, `cptlt` | 2025-05 | He-Kelly-Manela have not published beyond that date |
| `dcpi`, `cpi_vol6` | see table | FRED's CPIAUCSL has **no 2025-10 observation**, so the CPI change is unavailable for the months either side, and the 6-month rolling volatility cannot clear the gap |

❗**`mktb`, `mktbx`, `term`, `drf`, `crf`, `drfx`, `crfx`, `defb` and `termb` in THIS FILE
are the published extended (pre-TRACE) series, ending 2023-01 -- they are not the
bond-market factors the panel actually uses.** Stage 2 drops these columns on read and
rebuilds them from its own TRACE data, keeping the extended series only before 2002-08.
Pinning this file reproduces that exactly. Reading these columns out of this file directly
does not give you the panel's market or default factor.

### Last month by column

{shorttable}

## Provenance

`PROVENANCE.json` records, for each source, the URL it was fetched from, the UTC fetch
time, its sha256, row count and span. Verify the parquet with:

```
python3 -c "import hashlib;print(hashlib.sha256(open('factors_{vintage}.parquet','rb').read()).hexdigest())"
```

Expected: `{sha}`

## Citation

Dickerson, A., Robotti, C., & Rossetti, G. (2026). *The Corporate Bond Factor Replication
Crisis.* Working Paper. Earlier versions circulated as "Common pitfalls in the evaluation of
corporate bond strategies."
"""

COLUMN_GROUPS = [
    ("Equity factors (Fama-French)", ["mktrf", "smb", "hml", "rf"]),
    ("Bond market (BBW)", ["mktb", "mktbx", "term", "drf", "crf", "drfx", "crfx",
                           "defb", "termb"]),
    ("Volatility", ["vix", "dvix", "dvixlag"]),
    ("Inflation and credit", ["dcpi", "cpi_vol6", "credit", "dcredit", "lvl", "ysp"]),
    ("Intermediary capital (He-Kelly-Manela)", ["cptl", "cptlt"]),
    ("Macro uncertainty (Ludvigson)", ["unc", "uncf", "uncr", "dunc", "duncf", "duncr",
                                       "dunc3", "dunc6"]),
    ("Policy uncertainty (Baker-Bloom-Davis)", ["epu", "epum", "eput"]),
]


def build_readme(factors: pd.DataFrame, prov: dict, sha: str) -> str:
    lines = ["| group | columns |", "|---|---|"]
    seen = set()
    for name, cols in COLUMN_GROUPS:
        present = [c for c in cols if c in factors.columns]
        seen |= set(present)
        if present:
            lines.append(f"| {name} | {', '.join(f'`{c}`' for c in present)} |")
    other = [c for c in factors.columns if c not in seen and c != "date"]
    if other:
        lines.append(f"| Other | {', '.join(f'`{c}`' for c in other)} |")

    ends = {c: v for c, v in sorted(prov["column_last_month"].items()) if v}
    st = ["| column | last month |", "|---|---|"] + [f"| `{c}` | {v} |" for c, v in ends.items()]

    return README.format(
        vintage=prov["vintage"], rows=f"{prov['rows']:,}",
        span_start=prov["span"][0], span_end=prov["span"][1],
        coltable="\n".join(lines), shorttable="\n".join(st), sha=sha)


# ---------------------------------------------------------------------------
# Publication redaction
# ---------------------------------------------------------------------------
# The panel we BUILD is not the panel we PUBLISH. Two things in it are not ours to
# redistribute to an unlicensed audience, and the released 2025 vintage redacts both.
# This applies to the DOWNLOAD only -- a user who runs the pipeline has the licences and
# keeps the full panel:
#
#   permco, gvkey       proprietary identifiers -> nulled
#   spc_rat, mdc_rat    licensed agency ratings -> collapsed to investment grade (1)
#                       vs non-investment grade and default (11)
#
# `permno` is published. Everything else -- returns, signals, factors -- is computed
# here from scratch and carries no restriction.
#
# This is the one step in the pipeline where getting it wrong means distributing data
# we have no right to distribute, so it is a named function with a gate behind it
# rather than a few lines inside main().

REDACT_NULL = ("permco", "gvkey")
REDACT_RATINGS = ("spc_rat", "mdc_rat")
IG_MAX = 10          # ratings 1..10 are investment grade; 11+ is not


def redact_for_publication(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the panel that is safe to publish."""
    out = df.copy()
    for col in REDACT_NULL:
        if col in out.columns:
            out[col] = np.nan
    for col in REDACT_RATINGS:
        if col in out.columns:
            r = pd.to_numeric(out[col], errors="coerce")
            out[col] = np.where(r.isna(), np.nan, np.where(r <= IG_MAX, 1, 11))
    return out


def assert_publishable(df: pd.DataFrame, what: str) -> None:
    """Refuse to package anything still carrying restricted data."""
    problems = []
    for col in REDACT_NULL:
        if col in df.columns:
            n = int(df[col].notna().sum())
            if n:
                problems.append(f"  {col}: {n:,} non-null values -- must be entirely null")
    for col in REDACT_RATINGS:
        if col in df.columns:
            vals = set(pd.to_numeric(df[col], errors="coerce").dropna().unique().tolist())
            extra = sorted(vals - {1, 11})
            if extra:
                problems.append(f"  {col}: carries raw rating values {extra[:12]} -- "
                                f"must be collapsed to 1 (IG) / 11 (NIG)")
    if problems:
        raise AssertionError(
            f"{what} is NOT publishable -- restricted data survived redaction:\n"
            + "\n".join(problems)
            + "\n\n  Nothing was written. Fix redact_for_publication() before releasing."
        )


PANEL_README = """\
+=====================================================================+
|      ___  ____  ____    _    ____                                   |
|     / _ \\/ ___|| __ )  / \\  |  _ \\    Open Source                   |
|    | | | \\___ \\|  _ \\ / _ \\ | |_) |   Bond Asset Pricing            |
|    | |_| |___) | |_) / ___ \\|  __/    Corporate Bond Factor Data    |
|     \\___/|____/|____/_/   \\_\\_|       {banner}|
+=====================================================================+

                    Sample Period: {span_start} - {span_end}
                    {n_signals} Corporate Bond Factor Signals
                    {rows} bond-months, {bonds} bonds

================================================================================
                              LEGAL DISCLAIMER
================================================================================

We do NOT distribute proprietary data. The following adjustments apply:

  - gvkey and permco are set to NaN (proprietary identifiers)

  - spc_rat and mdc_rat are AUGMENTED ratings, NOT raw S&P/Moody's ratings:
      * Set to 1  if original rating is 1-10  (Investment Grade)
      * Set to 11 if original rating is 11+   (Non-Investment Grade / Default)

  - If you hold a valid license for credit ratings, merging is trivial via cusip

All returns, signals, and factors in this database are computed by the authors.
No proprietary data from any data provider is distributed.

================================================================================
                              QUICK START
================================================================================

The main panel is ready to use AS-IS. Price-based signals are already
market-microstructure (MMN) adjusted.

    import pandas as pd
    data = pd.read_parquet('main_panel_{vintage}.parquet')

The {n_signals} factor signals begin after the 'sig_gap' column.

Keys are 'cusip' (9-digit CUSIP) and 'date' (calendar month-end).

================================================================================
                              FILES INCLUDED
================================================================================

Data is distributed in two packages. Download both for full functionality.

  PACKAGE 1: osbap_main_data_{vintage}.zip
  -----------------------------------------------------------------------------
  main_panel_{vintage}.parquet             Main panel with MMN-adjusted price signals
                                         ({n_signals} factor signals, ready to use)

  PACKAGE 2: osbap_additional_data_{vintage}.zip
  -----------------------------------------------------------------------------
  betas_x_{vintage}.parquet                Factor betas from duration-adjusted returns
  mom_retx_{vintage}.parquet               Momentum/LTR from duration-adjusted returns
  mmn_price_based_signals_{vintage}.parquet   Unadjusted price signals (*_mmn suffix)
  returns_alt_{vintage}.parquet            Alternative return measures

  ALSO PUBLISHED SEPARATELY
  -----------------------------------------------------------------------------
  factors_{vintage}.parquet                The exogenous monthly factor panel every
                                         rolling beta in this release was estimated
                                         on. Public factor sources revise, so a build
                                         run later will NOT reproduce this release
                                         unless it is pinned to this file.

  NOTE: Package 1 is sufficient for most research designs using excess returns.
        Package 2 is needed for duration-adjusted returns or advanced use cases.

  DOWNLOAD: https://openbondassetpricing.com/

================================================================================
                         HOW TO USE THE DATA
================================================================================

  DECISION 1: What return measure?

    EXCESS RETURNS (r - rf)          DURATION-ADJUSTED RETURNS (r - tret)
    ------------------------          ------------------------------------
    Use main_panel_{vintage}.            Use main_panel_{vintage} PLUS
    Compute ret_vw - rfret.           betas_x_{vintage} and mom_retx_{vintage},
    All signals in the main           merged on (cusip, date).
    panel are ready to use.           Compute ret_vw - tret.

  DECISION 2: Month-end or month-begin returns?

    MONTH-END (ret_vw)                MONTH-BEGIN (ret_vw_bgn)
    ------------------                ------------------------
    The standard choice. Use with     Use when you take the UNADJUSTED price
    the MMN-adjusted signals in       signals from mmn_price_based_signals,
    the main panel.                   or when testing implementable returns.

  !! If you use mmn_price_based_signals_{vintage}.parquet you MUST pair it with
     ret_vw_bgn. Pairing an unadjusted signal with ret_vw puts the same bid-ask
     bounce in both the signal and the return and manufactures predictability:
     measured at AR(1) -0.219 unadjusted against -0.046 adjusted.

================================================================================
                         KEY VARIABLES
================================================================================

  Variable        Description
  -----------------------------------------------------------------------------
  cusip           Bond identifier (9-digit CUSIP)
  date            Calendar month-end
  ret_vw          Month-end total return (volume-weighted)
  ret_vw_bgn      Month-begin total return
  tret            Duration-matched Treasury return
  rfret           Risk-free rate (Fama-French)
  md_dur          Modified duration
  spc_rat         Augmented S&P rating (1=IG, 11=NIG)
  mdc_rat         Augmented Moody's rating (1=IG, 11=NIG)

Every column is defined in DATA_DICTIONARY.md in the trace-data-pipeline
repository, which also documents the factor models behind each beta.

================================================================================
                              CITATION
================================================================================

Dickerson, A., Robotti, C., & Rossetti, G. (2026). The Corporate Bond Factor
Replication Crisis. Working Paper. Earlier versions circulated as "Common
pitfalls in the evaluation of corporate bond strategies."

Built with the public TRACE Data Pipeline:
https://github.com/Alexander-M-Dickerson/trace-data-pipeline
"""


# (release name, block subpath or None for the panel itself)
PANEL_ARTIFACTS = [
    ("main_panel", None),
    ("betas_x", "betas_x.parquet"),
    ("mom_retx", "mom_retx.parquet"),
    ("mmn_price_based_signals", "mmn_price_based_signals_*.parquet"),
    ("returns_alt", "returns_alt_final.parquet"),
]
MAIN_PACKAGE = {"main_panel"}


def _resolve(blocks: Path, pattern: str) -> Path | None:
    if "*" in pattern:
        hits = sorted(blocks.glob(pattern))
        return hits[-1] if hits else None
    p = blocks / pattern
    return p if p.exists() else None


def release_panel(mode: str, out_dir: Path, vintage: str,
                  truncate_frontier: bool = False) -> int:
    """Package the panel artifacts: redact, rename to the vintage, describe, zip."""
    panel_src = cfg.PANEL_DIR / f"main_panel_{mode}.parquet"
    if not panel_src.exists():
        print(f"ERROR: no panel at {panel_src}. Build with --input-mode {mode} first.")
        return 1
    blocks = cfg.BLOCKS_DIR / mode
    stage = out_dir / f"osbap_panel_{vintage}"
    stage.mkdir(parents=True, exist_ok=True)

    print(f"reading  {panel_src.name} ...")
    panel = pd.read_parquet(panel_src)
    # A month where Stage 1's cut-off landed inside TRACE Enhanced's reporting gap is
    # not a cross-section -- what survives is the 144A universe alone. Do not publish it.
    bad = frontier.degenerate_tail_months(panel)
    dropped: list[str] = []
    frontier_cut = None
    if not bad.empty:
        print()
        print("FRONTIER: the last month(s) are not a usable cross-section:")
        print(frontier.describe(bad))
        if not truncate_frontier:
            print()
            print("  Nothing was written. Re-run with --truncate-frontier to publish")
            print("  the panel up to the last healthy month, or rebuild Stage 1 with")
            print("  a cut-off inside BOTH sources.")
            return 1
        cut = bad.index.min()
        frontier_cut = pd.Timestamp(cut)
        dropped = [str(d)[:10] for d in bad.index]
        panel = panel[panel["date"] < cut].copy()
        print(f"  --truncate-frontier: dropped {', '.join(dropped)}; "
              f"panel now ends {str(panel['date'].max())[:10]}")
        print()

    panel = redact_for_publication(panel)
    assert_publishable(panel, f"main_panel_{vintage}.parquet")
    print(f"redacted {', '.join(REDACT_NULL)} -> null; "
          f"{', '.join(REDACT_RATINGS)} -> 1 (IG) / 11 (NIG)")

    written: list[tuple[str, Path]] = []
    sources: dict[str, dict] = {"main_panel": _source(panel_src)}
    target = stage / f"main_panel_{vintage}.parquet"
    panel.to_parquet(target, index=False, compression="zstd")
    written.append(("main_panel", target))
    n_signals = len(panel.columns) - list(panel.columns).index("sig_gap") - 1
    span = (str(panel["date"].min())[:10], str(panel["date"].max())[:10])
    rows, bonds = len(panel), panel["cusip"].nunique()
    del panel

    for name, pattern in PANEL_ARTIFACTS:
        if pattern is None:
            continue
        src = _resolve(blocks, pattern)
        if src is None:
            print(f"  WARNING: {name} not found under {blocks} -- omitted from the bundle")
            continue
        dst = stage / f"{name}_{vintage}.parquet"
        sources[name] = _source(src)
        if frontier_cut is None:
            shutil.copy2(src, dst)
        else:
            # Every artifact in a release stops at the same month. A sidecar reaching
            # past the panel is a merge that silently drops rows, or a reader who
            # believes the extra month is usable when the panel says it is not.
            side = pd.read_parquet(src)
            n0 = len(side)
            side = side[pd.to_datetime(side["date"]) < frontier_cut]
            side.to_parquet(dst, index=False, compression="zstd")
            print(f"  {name}: truncated to the panel's frontier "
                  f"({n0:,} -> {len(side):,} rows)")
            del side
        written.append((name, dst))

    # The banner is a fixed-width box; pad rather than trusting the vintage to be
    # the same length as whatever was there last year.
    readme = PANEL_README.format(
        vintage=vintage, span_start=span[0], span_end=span[1],
        banner=f"{vintage} vintage".ljust(30),
        n_signals=n_signals, rows=f"{rows:,}", bonds=f"{bonds:,}")
    (stage / "README.txt").write_text(readme, encoding="utf-8")

    prov = {"vintage": vintage,
            "built_utc": datetime.now(timezone.utc).isoformat(),
            "source_build": mode,
            "redaction": {"nulled": list(REDACT_NULL),
                          "ratings_collapsed": list(REDACT_RATINGS),
                          "investment_grade_max": IG_MAX},
            "frontier_months_dropped": dropped,
            "files": {}}
    for name, path in written:
        md = pq.ParquetFile(path).metadata
        prov["files"][path.name] = {
            "artifact": name, "bytes": path.stat().st_size,
            "rows": md.num_rows, "cols": md.num_columns, "sha256": _sha256(path),
            **sources[name]}
    (stage / "PROVENANCE.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")

    bundles = {
        f"osbap_main_data_{vintage}.zip": [p for n, p in written if n in MAIN_PACKAGE],
        f"osbap_additional_data_{vintage}.zip": [p for n, p in written if n not in MAIN_PACKAGE],
    }
    print()
    for zip_name, members in bundles.items():
        if not members:
            continue
        zp = out_dir / zip_name
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(stage / "README.txt", "README.txt")
            # The provenance travels WITH the download, so the served bytes can be asked
            # which build they were cut from.
            z.write(stage / "PROVENANCE.json", "PROVENANCE.json")
            for m in members:
                print(f"  packing {m.name} ({m.stat().st_size/1e6:,.0f} MB) ...")
                z.write(m, m.name)
        print(f"BUNDLE  {zp}\n        {zp.stat().st_size/1e6:,.0f} MB, "
              f"sha256 {_sha256(zp)}\n")

    print(f"vintage   : {vintage}   ({rows:,} bond-months, {bonds:,} bonds, "
          f"{n_signals} signals, {span[0]} -> {span[1]})")
    print(f"staged in : {stage}")
    return 0


def release_factors(mode: str, out_dir: Path, vintage: str) -> int:
    src = cfg.BLOCKS_DIR / mode / "factors.parquet"
    if not src.exists():
        print(f"ERROR: no factor panel at {src}. Run a build with --input-mode {mode} first.")
        return 1

    stage = out_dir / f"osbap_stage2_factors_{vintage}"
    stage.mkdir(parents=True, exist_ok=True)

    factors = pd.read_parquet(src)
    factors["date"] = pd.to_datetime(factors["date"])

    target = stage / f"factors_{vintage}.parquet"
    shutil.copy2(src, target)
    sha = _sha256(target)

    prov = collect_provenance(factors)
    prov["file"] = target.name
    prov["sha256"] = sha
    prov.update(_source(src))
    (stage / "PROVENANCE.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    (stage / "README.md").write_text(build_readme(factors, prov, sha), encoding="utf-8")

    zip_path = out_dir / f"osbap_stage2_factors_{vintage}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(stage.iterdir()):
            z.write(f, f.name)

    print(f"vintage        : {vintage}")
    print(f"factors        : {target.name}  ({target.stat().st_size/1e3:.0f} KB, "
          f"{prov['rows']:,} months, {prov['span'][0]} -> {prov['span'][1]})")
    print(f"sources        : {len(prov['sources'])} recorded in PROVENANCE.json")
    print(f"parquet sha256 : {sha}")
    print(f"\nBUNDLE  {zip_path}")
    print(f"        {zip_path.stat().st_size/1e3:.0f} KB, sha256 {_sha256(zip_path)}")
    return 0



# =============================================================================================
# THE BBW FOUR-FACTOR BUNDLE
#
# The corrected Bai, Bali and Wen (2019) factors -- MKTB, DRF, CRF, LRF -- for every return
# definition the panel uses, on the TRACE sample and on the extended (Lehman-ICE + TRACE)
# sample, with the authors' original series beside them so the correction can be measured.
# Stage 2 builds these anyway (step 3, for the betas); this packages them for download.
# =============================================================================================

BBW_FACTORS = ("MKTB", "DRF", "CRF", "LRF")

# Block suffix -> the public return-definition key. The keys are the ones the panel, the sorted
# factors and FactorViz already use: exc is the return over the one-month T-bill, dur over the
# duration-matched Treasury (`tret`, the Andreani-Palhares-Richardson key-rate benchmark),
# dbns and dcls over the van Binsbergen-Nozawa-Schwert and Cui-Lu-Song benchmarks.
BBW_RETURN_TYPES = (("", "exc"), ("x", "dur"), ("_bns", "dbns"), ("_cls", "dcls"))

# The authors' original series, committed under reference/ and shipped unchanged. Pinned: a
# file that hashes differently is a different claim, and the bundle refuses it.
BBW_ORIGINAL = cfg.STAGE2_DIR / "reference" / "bbw_factors_original_2004_2021.csv"
BBW_ORIGINAL_SHA256 = "4588e8ee406eb14510c05b2dd47e332c98b630abacd101acea009907987dbbb4"

# Where the TRACE-built factors take over from the Lehman-ICE backfill -- step4_betas.ICE_CUTOFF.
BBW_TRACE_START = pd.Timestamp("2002-08-31")


def bbw_columns() -> list[tuple[str, str, str]]:
    """(published name, TRACE block column, extended column) for the 16 factor series, in
    return-definition order: mktb_exc drf_exc crf_exc lrf_exc, then _dur, _dbns, _dcls."""
    out = []
    for suffix, key in BBW_RETURN_TYPES:
        for f in BBW_FACTORS:
            out.append((f"{f.lower()}_{key}", f"{f}{suffix}", f"{f.lower()}{suffix}"))
    return out


def _bbw_trace(blocks: Path) -> pd.DataFrame:
    """The TRACE-built factors, 2002-08 on: date + 16 columns, from the three step-3 blocks."""
    parts = []
    for name in ("bbw_factors.parquet", "bbw_factors_bns.parquet", "bbw_factors_cls.parquet"):
        p = blocks / name
        if not p.exists():
            raise FileNotFoundError(f"no {p}; run step 3 (and make_excess_blocks for the "
                                    f"benchmark twins) with --input-mode {blocks.name} first")
        b = pd.read_parquet(p)
        if "date" not in b.columns:
            b = b.reset_index()
        b["date"] = pd.to_datetime(b["date"])
        parts.append(b.set_index("date").sort_index())
    axes = [tuple(p.index) for p in parts]
    if any(a != axes[0] for a in axes[1:]):
        raise SystemExit("ERROR: the three BBW blocks do not share one date axis -- they were "
                         "built from different runs. Rebuild step 3 for every benchmark.")
    wide = pd.concat(parts, axis=1)
    out = pd.DataFrame({"date": wide.index})
    for pub, trace_col, _ in bbw_columns():
        if trace_col not in wide.columns:
            raise SystemExit(f"ERROR: {trace_col} is not in the BBW blocks; got {list(wide.columns)}")
        out[pub] = wide[trace_col].to_numpy()
    missing = out.drop(columns="date").isna().sum()
    if int(missing.sum()):
        raise SystemExit("ERROR: the TRACE factors have missing months, which they never have "
                         f"had:\n{missing[missing > 0]}")
    return out.reset_index(drop=True)


def _bbw_extended(blocks: Path) -> pd.DataFrame:
    """The extended factors, Lehman-ICE before 2002-08 and TRACE from it, as step 4 splices them
    into factors_merged. Trimmed to the months where at least one of the 16 series exists, so
    the macro columns' longer frontier does not pad the file with empty rows."""
    p = blocks / "factors_merged.parquet"
    if not p.exists():
        raise FileNotFoundError(f"no {p}; run step 4 with --input-mode {blocks.name} first")
    fm = pd.read_parquet(p)
    fm["date"] = pd.to_datetime(fm["date"])
    fm = fm.sort_values("date").reset_index(drop=True)
    out = pd.DataFrame({"date": fm["date"]})
    for pub, _, ext_col in bbw_columns():
        if ext_col not in fm.columns:
            raise SystemExit(f"ERROR: {ext_col} is not in factors_merged.parquet")
        out[pub] = fm[ext_col].to_numpy()
    have = out.drop(columns="date").notna().any(axis=1)
    return out[have].reset_index(drop=True)


def _gate_bbw(trace: pd.DataFrame, ext: pd.DataFrame) -> None:
    """Refuse what would publish a wrong series.

    ❗Extended after the cutoff IS the TRACE series -- step 4 keeps the TRACE-native columns
    from 2002-08-31 and splices the backfill only before it. So the two tables must agree
    exactly there; a difference means a block from a different run or mode. And LRF has no
    pre-TRACE history at all (the extension is the modified, three-factor variant), so it must
    be empty before the cutoff and complete from it."""
    cols = [c for c in trace.columns if c != "date"]
    post = ext[ext["date"] >= BBW_TRACE_START].reset_index(drop=True)
    if list(post["date"]) != list(trace["date"]):
        raise SystemExit("ERROR: the extended series from 2002-08 does not carry the TRACE "
                         f"months ({len(post)} vs {len(trace)})")
    d = (post[cols].to_numpy(dtype=float) - trace[cols].to_numpy(dtype=float))
    worst = float(np.nanmax(np.abs(d))) if d.size else 0.0
    if worst != 0.0 or np.isnan(d).any():
        raise SystemExit("ERROR: the extended series differs from the TRACE series after "
                         f"2002-08 (max |d| = {worst:.3g}); the blocks come from different runs")
    pre = ext[ext["date"] < BBW_TRACE_START]
    lrf = [c for c in cols if c.startswith("lrf_")]
    if pre[lrf].notna().any().any():
        raise SystemExit("ERROR: LRF carries pre-2002-08 values; the extension has no LRF leg")
    if pre.empty:
        raise SystemExit("ERROR: the extended series has no pre-2002-08 history; the "
                         "Lehman-ICE backfill did not reach factors_merged")
    if not BBW_ORIGINAL.exists():
        raise SystemExit(f"ERROR: {BBW_ORIGINAL} is missing")
    sha = _sha256(BBW_ORIGINAL)
    if sha != BBW_ORIGINAL_SHA256:
        raise SystemExit("ERROR: the original BBW series does not hash as pinned "
                         f"({sha[:16]} vs {BBW_ORIGINAL_SHA256[:16]}); refusing to ship a "
                         "changed file as the authors' original")


BBW_README = """# The corrected Bai, Bali and Wen (2019) bond factors -- {vintage} vintage

MKTB, DRF, CRF and LRF, the four factors of Bai, Bali and Wen (2019), rebuilt from the OSBAP
bond panel with the corrections described in Dickerson, Mueller and Robotti (2023), for
every return definition the panel uses, on two samples. The authors' original series is
included beside them so the correction can be measured.

## Files

| file | months | what |
|---|---|---|
| `bbw_factors_trace_{vintage}.parquet` / `.csv` | {trace_span} ({trace_n}) | built from TRACE alone |
| `bbw_factors_extended_{vintage}.parquet` / `.csv` | {ext_span} ({ext_n}) | Lehman and ICE quote data before 2002-08, TRACE from 2002-08 |
| `bbw_factors_original_2004_2021.csv` | 2004-08 to 2021-12 (209) | the authors' original series, unchanged |
| `PROVENANCE.json` | | which build the series came from, with hashes |
| `MANIFEST.json` | | every file with its sha256, checked by `verify_release.py` |

Both of our tables have the same 16 columns: `date` (calendar month-end), then each factor
for each return definition, named `<factor>_<return definition>`. Values are decimal monthly
returns, 0.01 = 1%.

| column stem | factor |
|---|---|
| `mktb` | The bond market factor. The value-weighted return of every bond in the panel, weights the previous month-end market value, over the benchmark named by the suffix. |
| `drf` | Downside risk. Bonds are sorted 5 x 5 on credit rating and on 5% value-at-risk measured over the previous 36 months (at least 12). The factor is the high-VaR minus low-VaR return, averaged across the rating quintiles. |
| `crf` | Credit risk. The same 5 x 5 sorts read the other way, the lowest-rated minus the highest-rated return, averaged across the quintiles of the other sort variable and across the three sorts (value-at-risk, illiquidity and short-term reversal). |
| `lrf` | Liquidity risk. The 5 x 5 sort on rating and on the Bao, Pan and Wang (2011) illiquidity measure, the negative autocovariance of consecutive daily log price changes. The factor is the illiquid minus liquid return, averaged across the rating quintiles. |

| suffix | the return each leg is measured on |
|---|---|
| `_exc` | total return minus the one-month Treasury bill |
| `_dur` | total return minus a duration-matched Treasury return interpolated across CRSP fixed-term indices (Andreani, Palhares and Richardson, 2024) |
| `_dbns` | total return minus a synthetic Treasury carrying the bond's own cash flows, discounted at the bond's own yield (van Binsbergen, Nozawa and Schwert, 2025) |
| `_dcls` | the same synthetic Treasury with cash flows weighted on the Treasury zero curve (Cui, Lu and Song, 2026) |

The duration-adjusted columns are not the excess columns with a benchmark subtracted. Every
sort is re-run on the duration-adjusted return, so the portfolios themselves differ.

## The two samples

**TRACE** starts 2002-08, the first month-end return in the TRACE data. **Extended** carries
the same series from 1973-02, spliced from the pre-TRACE extension published on
openbondassetpricing.com (built from the Lehman Brothers Fixed Income Database and the ICE
index constituents, licensed data whose finished factor series may be distributed). From
2002-08 the extended file IS the TRACE file, value for value; this bundle refuses to build
if that ever stops being true.

**LRF has no history before 2002-08** in either file, because the pre-TRACE extension is the
three-factor variant: the illiquidity measure needs transaction prices, and there are none
before TRACE. The extended `lrf_*` columns are empty before 2002-08 and complete from it.

## The original series

`bbw_factors_original_2004_2021.csv` is the factor file Bai, Bali and Wen distributed with
the 2019 Journal of Financial Economics paper, later retracted, obtained from Turan Bali's
website, which has since been taken down. It is shipped byte for byte, with the authors' own
column names (`MKTbond, DRF, CRF, LRF`), on the authors' sample of 2004-08 to 2021-12.
Dickerson, Mueller and Robotti (2023) document why those series cannot be reproduced from
the underlying data and what the corrected factors look like; the two tables above are the
corrected factors. Do not use the original series in new work. It is here so that anyone can
measure the difference.

## Months with data, by column

{last_months}

## Checking a download

Every file is listed in `MANIFEST.json` with its sha256. `verify_release.py`, published on the
same release page, re-hashes what is in the zip and compares.

## Citation

Dickerson, A., Mueller, P., & Robotti, C. (2023). Priced risk in corporate bonds. *Journal
of Financial Economics*, 150(2), 103707.

Bai, J., Bali, T. G., & Wen, Q. (2019). Common risk factors in the cross-section of corporate
bond returns. *Journal of Financial Economics*, 131(3), 619-642. (Retracted.)
"""


def _last_months_table(trace: pd.DataFrame, ext: pd.DataFrame) -> str:
    rows = ["| column | TRACE | extended |", "|---|---|---|"]
    for c in trace.columns:
        if c == "date":
            continue
        t = trace.loc[trace[c].notna(), "date"]
        e = ext.loc[ext[c].notna(), "date"]
        rows.append(f"| `{c}` | {str(t.min())[:7]} to {str(t.max())[:7]} | "
                    f"{str(e.min())[:7]} to {str(e.max())[:7]} |")
    return "\n".join(rows)


def release_bbw(mode: str, out_dir: Path, vintage: str) -> int:
    blocks = cfg.BLOCKS_DIR / mode
    trace = _bbw_trace(blocks)
    ext = _bbw_extended(blocks)
    _gate_bbw(trace, ext)

    stage = out_dir / f"osbap_bbw_factors_{vintage}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    def span(df: pd.DataFrame) -> str:
        return f"{str(df['date'].min())[:7]} to {str(df['date'].max())[:7]}"

    written = {}
    for name, df in (("trace", trace), ("extended", ext)):
        base = stage / f"bbw_factors_{name}_{vintage}"
        df.to_parquet(base.with_suffix(".parquet"), index=False)
        csv = df.copy()
        csv["date"] = csv["date"].dt.strftime("%Y-%m-%d")
        csv.to_csv(base.with_suffix(".csv"), index=False, float_format="%.10g")
        written[name] = {"rows": int(len(df)), "span": [str(df["date"].min())[:10],
                                                          str(df["date"].max())[:10]]}
    shutil.copyfile(BBW_ORIGINAL, stage / BBW_ORIGINAL.name)

    def block_info(name: str) -> dict:
        p = blocks / name
        b = pd.read_parquet(p)
        if "date" not in b.columns:
            b = b.reset_index()
        return {"source": _public_path(p), "source_sha256": _sha256(p),
                "sha256": _sha256(p), "rows": int(len(b)),
                "span": [str(pd.to_datetime(b["date"]).min())[:10],
                         str(pd.to_datetime(b["date"]).max())[:10]]}

    last = {name: {c: str(df.loc[df[c].notna(), "date"].max())[:7]
                   for c in df.columns if c != "date"}
            for name, df in (("trace", trace), ("extended", ext))}
    prov = {
        "vintage": vintage,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "trace_start": str(BBW_TRACE_START)[:10],
        "blocks": {n: block_info(n) for n in ("bbw_factors.parquet", "bbw_factors_bns.parquet",
                                               "bbw_factors_cls.parquet",
                                               "factors_merged.parquet")},
        "extended_backfill": {"url": cfg.BBW_EXTENDED_URL,
                              "cache": _public_path(cfg.BBW_EXTENDED_CACHE),
                              "sha256": _sha256(cfg.BBW_EXTENDED_CACHE)
                              if cfg.BBW_EXTENDED_CACHE.exists() else None},
        "original": {"file": BBW_ORIGINAL.name, "sha256": BBW_ORIGINAL_SHA256,
                     "rows": 209, "span": ["2004-08-31", "2021-12-31"], "source": "see README"},
        "tables": written,
        "column_last_month": last,
    }
    (stage / "PROVENANCE.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    (stage / "README.md").write_text(
        BBW_README.format(vintage=vintage, trace_span=span(trace), trace_n=len(trace),
                          ext_span=span(ext), ext_n=len(ext),
                          last_months=_last_months_table(trace, ext)),
        encoding="utf-8", newline="\n")

    members = {f.name: {"bytes": f.stat().st_size, "sha256": _sha256(f)}
               for f in sorted(stage.iterdir()) if f.name != "MANIFEST.json"}
    manifest = {"archive": f"osbap_bbw_factors_{vintage}.zip", "vintage": vintage,
                "built_utc": prov["built_utc"],
                "data_span": {"trace": written["trace"]["span"],
                              "extended": written["extended"]["span"]},
                "members": members}
    (stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zip_path = out_dir / f"osbap_bbw_factors_{vintage}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(stage.iterdir()):
            z.write(f, f.name)

    print(f"vintage        : {vintage}")
    print(f"trace          : {written['trace']['rows']} months, {span(trace)}")
    print(f"extended       : {written['extended']['rows']} months, {span(ext)}  "
          f"(== TRACE from {str(BBW_TRACE_START)[:7]}, max |d| = 0)")
    print(f"original       : {BBW_ORIGINAL.name}, sha256 {BBW_ORIGINAL_SHA256[:16]} (pinned)")
    print(f"\nBUNDLE  {zip_path}")
    print(f"        {zip_path.stat().st_size/1e3:.0f} KB, sha256 {_sha256(zip_path)}")
    return 0


# =============================================================================================
# THE DAILY PANEL
#
# Stage 1's bond-day panel, in the layout we publish. The file Stage 1 BUILDS carries the
# agency ratings and two proprietary identifiers, and is not ours to redistribute. On
# 2026-09-18 it was uploaded as it stood, because the website copied it with SELECT * and no
# release step existed for it. This is that step.
#
# The public layout is a WHITELIST. A column Stage 1 gains later is withheld until someone
# decides it is public, and the release refuses to run until they have decided.
# =============================================================================================

DAILY_PUBLIC_COLUMNS = (
    "cusip_id", "permno", "trd_exctn_dt",
    "pr", "prfull", "acclast", "accpmt", "accall",
    "ytm", "mod_dur", "mac_dur", "convexity", "bond_maturity", "credit_spread",
    "prc_ew", "prc_vw_par", "prc_first", "prc_last", "prc_hi", "prc_lo",
    "trade_count", "qvolume", "dvolume",
    "prc_bid", "bid_last", "prc_ask",
    "db_type", "ff12num", "ff17num", "ff30num",
    "bond_age", "bond_amt_outstanding",
)

# Every other column Stage 1 writes, and why it stays behind.
DAILY_WITHHELD = {
    "permco": "proprietary identifier (CRSP)",
    "gvkey": "proprietary identifier (Compustat)",
    "sp_rating": "licensed agency rating",
    "mdy_rating": "licensed agency rating",
    "spc_rating": "licensed agency rating (composite)",
    "mdc_rating": "licensed agency rating (composite)",
    "time_ew": "not in the public layout",
    "time_last": "not in the public layout",
    "bid_time_ew": "not in the public layout",
    "bid_time_last": "not in the public layout",
    "bid_count": "not in the public layout",
    "ask_count": "not in the public layout",
}
DAILY_LICENSED = tuple(c for c, why in DAILY_WITHHELD.items() if "not in the public" not in why)
_LICENSED_NAME = ("rating", "permco", "gvkey", "_rat")


def assert_daily_publishable(path: Path, source_rows: int | None = None) -> None:
    """Refuse a daily file that is not exactly the public layout."""
    pf = pq.ParquetFile(path)
    names = list(pf.schema.names)
    problems = []
    licensed = [c for c in names
                if c in DAILY_LICENSED or any(k in c.lower() for k in _LICENSED_NAME)]
    if licensed:
        problems.append(f"  carries licensed column(s): {licensed}")
    extra = [c for c in names if c not in DAILY_PUBLIC_COLUMNS and c not in licensed]
    if extra:
        problems.append(f"  carries column(s) outside the public layout: {extra}")
    missing = [c for c in DAILY_PUBLIC_COLUMNS if c not in names]
    if missing:
        problems.append(f"  lacks public column(s): {missing}")
    if not problems and tuple(names) != DAILY_PUBLIC_COLUMNS:
        problems.append("  has the public columns in a different order")
    if source_rows is not None and pf.metadata.num_rows != source_rows:
        problems.append(f"  has {pf.metadata.num_rows:,} rows, the source has {source_rows:,}")
    if problems:
        raise AssertionError(
            f"{Path(path).name} is NOT publishable:\n" + "\n".join(problems)
            + "\n\n  The public daily layout is make_release.DAILY_PUBLIC_COLUMNS, "
              f"{len(DAILY_PUBLIC_COLUMNS)} columns, and nothing else.")


DAILY_README = """# OSBAP daily bond panel -- {vintage} vintage

`{name}` -- one row per bond per trading day, {rows} bond-days, {span_start} to {span_end}.
Enhanced TRACE and Rule 144A TRACE combined (`db_type` 1 and 3), with no filter on par or
dollar volume. Built by Stage 1 of https://github.com/Alexander-M-Dickerson/trace-data-pipeline

{ncols} columns. Definitions are in `stage1/DATA_DICTIONARY.md` in that repository.

## What is not in this file

The panel the pipeline builds has {nsrc} columns. {nheld} are withheld from the download.

| column | why |
|---|---|
{withheld}

`permno` is published. Agency ratings, `permco` and `gvkey` are licensed data. Run the
pipeline with your own WRDS account and you get all {nsrc} columns.

`PROVENANCE.json` records the Stage 1 file this was cut from, with hashes.
"""


def release_daily(out_dir: Path, vintage: str) -> int:
    """Write Stage 1's daily panel in the public layout. Nothing else may be uploaded."""
    import duckdb

    src = cfg.daily_input()
    if not src.exists():
        print(f"ERROR: no Stage 1 daily panel at {src}.")
        return 1
    src_meta = pq.ParquetFile(src)
    src_cols = list(src_meta.schema.names)
    src_rows = src_meta.metadata.num_rows

    missing = [c for c in DAILY_PUBLIC_COLUMNS if c not in src_cols]
    undecided = [c for c in src_cols if c not in DAILY_PUBLIC_COLUMNS and c not in DAILY_WITHHELD]
    if missing or undecided:
        print(f"ERROR: {src.name} does not match the layout this release knows.")
        if missing:
            print(f"  public columns it lacks      : {missing}")
        if undecided:
            print(f"  columns nobody has classified: {undecided}")
            print("  Add each to DAILY_PUBLIC_COLUMNS or to DAILY_WITHHELD, with the reason.")
            print("  A column is withheld until someone decides it is public.")
        return 1

    stage = out_dir / f"osbap_daily_data_{vintage}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    target = stage / f"stage1_daily_panel_{vintage}.parquet"

    print(f"reading  {src.name}  ({src_rows:,} bond-days, {len(src_cols)} columns)")
    select = ", ".join(f'"{c}"' for c in DAILY_PUBLIC_COLUMNS)
    con = duckdb.connect()
    try:
        con.execute("SET preserve_insertion_order = true")
        con.execute(
            f"COPY (SELECT {select} FROM read_parquet('{src.as_posix()}')) "
            f"TO '{target.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)")
        assert_daily_publishable(target, source_rows=src_rows)

        # Same rows, same values. An order-independent hash over EVERY public column of every
        # row, taken on the file read and on the file written. A float sum would not do: a
        # parallel sum is not the same to the last bit on two files laid out differently.
        probe = ("count(*), count(permno), count(DISTINCT cusip_id), min(trd_exctn_dt), "
                 f"max(trd_exctn_dt), bit_xor(hash({select}))")
        a = con.execute(f"SELECT {probe} FROM read_parquet('{src.as_posix()}')").fetchone()
        b = con.execute(f"SELECT {probe} FROM read_parquet('{target.as_posix()}')").fetchone()
    finally:
        con.close()
    if a != b:
        target.unlink()
        raise AssertionError(f"the public daily file does not reproduce its source:\n  {a}\n  {b}")

    span = (str(b[3])[:10], str(b[4])[:10])
    withheld = [c for c in src_cols if c in DAILY_WITHHELD]
    prov = {"vintage": vintage,
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "file": target.name, "sha256": _sha256(target),
            "bytes": target.stat().st_size,
            "rows": int(b[0]), "bonds": int(b[2]), "span": list(span),
            "rows_with_permno": int(b[1]),
            "columns": list(DAILY_PUBLIC_COLUMNS),
            "withheld": {c: DAILY_WITHHELD[c] for c in withheld},
            "row_hash": str(b[5]),
            **_source(src), "source_columns": len(src_cols)}
    (stage / "PROVENANCE.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    (stage / "README.md").write_text(
        DAILY_README.format(
            vintage=vintage, name=target.name, rows=f"{b[0]:,}", span_start=span[0],
            span_end=span[1], ncols=len(DAILY_PUBLIC_COLUMNS), nsrc=len(src_cols),
            nheld=len(withheld),
            withheld="\n".join(f"| `{c}` | {DAILY_WITHHELD[c]} |" for c in withheld)),
        encoding="utf-8", newline="\n")

    print(f"withheld : {', '.join(withheld)}")
    print(f"written  : {target.name}  ({len(DAILY_PUBLIC_COLUMNS)} columns, {b[0]:,} bond-days, "
          f"{b[2]:,} bonds, {span[0]} -> {span[1]}, {target.stat().st_size/1e9:.2f} GB)")
    print(f"permno   : {b[1]:,} of {b[0]:,} bond-days ({100*b[1]/b[0]:.2f}%)")
    print(f"sha256   : {prov['sha256']}")
    print(f"staged in: {stage}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="package a built Stage 2 vintage for publication")
    ap.add_argument("--mode", default=cfg.INPUT_MODE,
                    help="the build to publish (its blocks/<mode>/ and panel); default "
                         f"{cfg.INPUT_MODE!r}, the mode _run_stage2.py builds")
    ap.add_argument("--what", choices=("all", "panel", "factors", "bbw", "daily"), default="all",
                    help="which bundles to build (default all, which includes daily)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write the bundles (default stage2/release/)")
    ap.add_argument("--truncate-frontier", action="store_true",
                    help="publish up to the last month that is a real cross-section, "
                         "dropping any collapsed trailing months (recorded in PROVENANCE).")
    args = ap.parse_args()

    vintage = cfg.release_vintage()
    out_dir = args.out_dir or (cfg.STAGE2_DIR / "release")
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = 0
    if args.what in ("all", "panel"):
        print("=" * 78 + f"\nPANEL BUNDLES ({vintage})\n" + "=" * 78)
        rc |= release_panel(args.mode, out_dir, vintage, args.truncate_frontier)
    if args.what in ("all", "factors"):
        print("\n" + "=" * 78 + f"\nFACTOR BUNDLE ({vintage})\n" + "=" * 78)
        rc |= release_factors(args.mode, out_dir, vintage)
    if args.what in ("all", "bbw"):
        print("\n" + "=" * 78 + f"\nBBW FOUR-FACTOR BUNDLE ({vintage})\n" + "=" * 78)
        rc |= release_bbw(args.mode, out_dir, vintage)
    if args.what in ("all", "daily"):
        print("\n" + "=" * 78 + f"\nDAILY PANEL, PUBLIC LAYOUT ({vintage})\n" + "=" * 78)
        rc |= release_daily(out_dir, vintage)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
