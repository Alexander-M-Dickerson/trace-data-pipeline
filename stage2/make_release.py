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

    python3 make_release.py --mode prod_final                 # factors bundle
    python3 make_release.py --mode prod_final --out-dir dist  # somewhere else

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

import pandas as pd

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

❗**`mktb`, `mktbx`, `term`, `drf`, `crf`, `drfx`, `crfx` in THIS FILE are the published
extended (pre-TRACE) series, ending 2023-01 -- they are not the bond-market factors the
panel actually uses.** Stage 2 drops these columns on read and rebuilds them from its own
TRACE data, keeping the extended series only before 2002-08. Pinning this file reproduces
that exactly. Reading these columns out of this file directly does not give you the
panel's market factor.

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
    ("Bond market (BBW)", ["mktb", "mktbx", "term", "drf", "crf", "drfx", "crfx"]),
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


def main() -> int:
    ap = argparse.ArgumentParser(description="package a built Stage 2 vintage for publication")
    ap.add_argument("--mode", default="prod_final",
                    help="the build whose blocks/<mode>/factors.parquet to publish")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write the bundle (default stage2/release/)")
    args = ap.parse_args()

    src = cfg.BLOCKS_DIR / args.mode / "factors.parquet"
    if not src.exists():
        print(f"ERROR: no factor panel at {src}. Run a build with --input-mode {args.mode} first.")
        return 1

    vintage = cfg.release_vintage()
    out_dir = args.out_dir or (cfg.STAGE2_DIR / "release")
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


if __name__ == "__main__":
    raise SystemExit(main())
