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

    python3 make_release.py --mode prod_final                 # panel + factor bundles
    python3 make_release.py --mode prod_final --what panel    # just the panel
    python3 make_release.py --mode prod_final --out-dir dist  # somewhere else

❗The panel we BUILD is not the panel we PUBLISH. `permco` and `gvkey` are proprietary
identifiers and the agency ratings are licensed, so both are redacted here before anything
is written -- and `assert_publishable` refuses to package a file where that did not take.

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

Dickerson, A., Robotti, C., & Rossetti, G. (2025). *Common pitfalls in the evaluation of
corporate bond strategies.* Working Paper.
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
# redistribute, and the released 2025 vintage redacts both:
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

Dickerson, A., Robotti, C., & Rossetti, G. (2025). Common pitfalls in the
evaluation of corporate bond strategies. Working Paper.

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


def release_panel(mode: str, out_dir: Path, vintage: str) -> int:
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
    panel = redact_for_publication(panel)
    assert_publishable(panel, f"main_panel_{vintage}.parquet")
    print(f"redacted {', '.join(REDACT_NULL)} -> null; "
          f"{', '.join(REDACT_RATINGS)} -> 1 (IG) / 11 (NIG)")

    written: list[tuple[str, Path]] = []
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
        shutil.copy2(src, dst)
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
            "files": {}}
    for name, path in written:
        md = pq.ParquetFile(path).metadata
        prov["files"][path.name] = {
            "artifact": name, "bytes": path.stat().st_size,
            "rows": md.num_rows, "cols": md.num_columns, "sha256": _sha256(path)}
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


def main() -> int:
    ap = argparse.ArgumentParser(description="package a built Stage 2 vintage for publication")
    ap.add_argument("--mode", default="prod_final",
                    help="the build to publish (its blocks/<mode>/ and panel)")
    ap.add_argument("--what", choices=("all", "panel", "factors"), default="all",
                    help="which bundles to build (default all)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write the bundles (default stage2/release/)")
    args = ap.parse_args()

    vintage = cfg.release_vintage()
    out_dir = args.out_dir or (cfg.STAGE2_DIR / "release")
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = 0
    if args.what in ("all", "panel"):
        print("=" * 78 + f"\nPANEL BUNDLES ({vintage})\n" + "=" * 78)
        rc |= release_panel(args.mode, out_dir, vintage)
    if args.what in ("all", "factors"):
        print("\n" + "=" * 78 + f"\nFACTOR BUNDLE ({vintage})\n" + "=" * 78)
        rc |= release_factors(args.mode, out_dir, vintage)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
