"""compute_factors.py -- the `create_factors.py` port: materialize the factor time-series panel
(`blocks/<mode>/factors.parquet`, the seam steps 4 and 7 read).

Two sources (cfg.FACTOR_SOURCE / --factor-source):
  pinned -- copy the golden Dec-2025 vintage (Phase-A exact; the reproduction path). A10.
  public -- assemble fresh from the public fetchers (lib/factor_fetch) + the published extended BBW
            series (lib/extended_factors). Runs with NO private input, anywhere -- but does NOT
            bit-match the pinned vintage (FRED/EPU/HKM/Ludvigson back-revise history; A10). The
            per-column divergence is documented in the divergence report written by this step.

Ground truth: `stage2/create_factors.py` (fetch order, the cptlt-rf subtraction, start-date
truncation, date dedup, then the extended-BBW lowercase outer merge -- lines 677-973).

CLI (the divergence reporter; run from stage2/):
    python -m steps.compute_factors [--refresh] [--report]
builds the public panel and, with --report, diffs it per column against the pinned golden vintage,
printing the table and writing output/factors_public_divergence.json.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import _stage2_settings as cfg
from lib import extended_factors, factor_fetch


def build_public(force_fetch: bool = False) -> pd.DataFrame:
    """The 34-column factor panel from public sources only. Grain: one row per month-end."""
    out = factor_fetch.fetch_ff5(force_fetch)
    for fetch in (factor_fetch.fetch_vix_monthly, factor_fetch.fetch_cpi_credit,
                  factor_fetch.fetch_intermediary_capital, factor_fetch.fetch_uncertainty,
                  factor_fetch.fetch_epu):
        out = out.merge(fetch(force_fetch), on="date", how="outer")

    out["cptlt"] = out["cptlt"] - out["rf"]            # upstream step 7: excess investment return

    out = out.sort_values("date").reset_index(drop=True)
    out = out[out["date"] >= pd.Timestamp(cfg.FACTORS_START_DATE)]
    out = out.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)

    # the extended-BBW merge (lowercased): the PUBLISHED pre-2002-08 series, fetched by
    # lib/extended_factors.py -- see its module docstring for the provenance
    ext = extended_factors.load_extended_bbw()
    ext.columns = [c if c == "date" else c.lower() for c in ext.columns]
    out = out.merge(ext, on="date", how="outer")
    out = out.sort_values("date").reset_index(drop=True)
    out = out.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
    return out


def _pinned_factors(force_fetch: bool = False) -> Path:
    """The published factor panel for a vintage, as a local parquet.

    Precedence: an explicit FACTORS_PINNED_FILE, then a local copy already downloaded,
    then the published zip for this vintage. Publishing these is what lets a released
    number be reproduced -- the public sources revise, so a fresh build will not match an
    older release.
    """
    explicit = cfg.GOLDEN_OUTPUTS.get("factors") or cfg.FACTORS_PINNED_FILE
    if explicit:
        return Path(explicit)

    vintage = cfg.release_vintage()
    cache = cfg.STAGE2_DATA / cfg.FACTORS_PINNED_ZIPKEY.format(vintage=vintage)
    if cache.exists() and not force_fetch:
        return cache

    url = cfg.FACTORS_PINNED_URL.get(vintage)
    if not url:
        raise FileNotFoundError(
            f"No published factor panel is registered for vintage {vintage}.\n"
            f"    Known vintages: {sorted(cfg.FACTORS_PINNED_URL) or 'none'}.\n"
            f"    Either use FACTOR_SOURCE = 'public' (the default, builds from live "
            f"sources), or set FACTORS_PINNED_FILE to a factor parquet you already have.")

    import io
    import zipfile
    import requests
    print(f"[factors] fetching the published {vintage} factor panel: {url}")
    blob = requests.get(url, timeout=300)
    blob.raise_for_status()
    member = cfg.FACTORS_PINNED_ZIPKEY.format(vintage=vintage)
    with zipfile.ZipFile(io.BytesIO(blob.content)) as z:
        names = z.namelist()
        if member not in names:
            raise FileNotFoundError(
                f"{member} not found in {url}; the zip holds {names}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(z.read(member))
    print(f"[factors] cached {cache}")
    return cache


def ensure(mode: str | None = None, source: str | None = None,
           force_fetch: bool = False) -> Path:
    """Materialize blocks/<mode>/factors.parquet from `source` ('pinned' | 'public'); returns its path."""
    mode = mode or cfg.INPUT_MODE
    source = source or cfg.FACTOR_SOURCE
    dst = cfg.BLOCKS_DIR / mode / "factors.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)

    if source == "pinned":
        from lib import manifest as mf
        src = _pinned_factors(force_fetch)
        if not dst.exists() or mf.sha256_file(dst) != mf.sha256_file(src):
            shutil.copy2(src, dst)
    elif source == "public":
        build_public(force_fetch).to_parquet(dst, index=False)
    else:
        raise ValueError(f"unknown factor source {source!r} (want 'pinned' | 'public')")
    return dst


def divergence_report(public: pd.DataFrame) -> dict:
    """Per-column diff of the public panel vs the pinned golden vintage, over the overlap dates."""
    gold = pd.read_parquet(cfg.GOLDEN_OUTPUTS["factors"])
    gold["date"] = pd.to_datetime(gold["date"])
    m = gold.merge(public, on="date", how="inner", suffixes=("_g", "_p"))
    cols = [c for c in gold.columns if c != "date"]
    report = {"overlap_months": len(m),
              "golden_span": [str(gold['date'].min().date()), str(gold['date'].max().date())],
              "public_span": [str(public['date'].min().date()), str(public['date'].max().date())],
              "columns": {}}
    for c in cols:
        g = m[f"{c}_g"].to_numpy(dtype=float, na_value=np.nan)
        p = m[f"{c}_p"].to_numpy(dtype=float, na_value=np.nan)
        both = ~np.isnan(g) & ~np.isnan(p)
        d = np.abs(g - p)[both]
        corr = float(np.corrcoef(g[both], p[both])[0, 1]) if both.sum() > 2 else np.nan
        report["columns"][c] = {
            "n_both": int(both.sum()),
            "nan_mismatch": int((np.isnan(g) != np.isnan(p)).sum()),
            "max_abs_d": float(d.max()) if len(d) else np.nan,
            "n_exact": int((d == 0).sum()),
            "corr": corr}
    return report


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="public factor panel builder + divergence reporter")
    ap.add_argument("--refresh", action="store_true", help="force re-fetch of every public source")
    ap.add_argument("--report", action="store_true",
                    help="diff the public panel vs the pinned golden vintage")
    args = ap.parse_args()

    cfg.ensure_dirs()
    panel = build_public(force_fetch=args.refresh)
    print(f"public factor panel: {panel.shape[0]} months x {panel.shape[1]} cols, "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}")

    if args.report:
        rep = divergence_report(panel)
        out = cfg.OUTPUT_DIR / "factors_public_divergence.json"
        out.write_text(json.dumps(rep, indent=1))
        print(f"\n{'column':10s} {'n':>4s} {'exact':>6s} {'max|d|':>10s} {'corr':>8s}")
        for c, r in rep["columns"].items():
            print(f"{c:10s} {r['n_both']:4d} {r['n_exact']:6d} "
                  f"{r['max_abs_d']:10.3e} {r['corr']:8.4f}")
        print(f"\nwrote {out}")
